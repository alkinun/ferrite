# A Llama-family model on ferrite (SmolLM-135M: RMSNorm, RoPE, SwiGLU, grouped-query attention, a KV cache): the real
# weights, greedy generation, each step checked against the same model in numpy (the same bf16 weights, float32 math).
# The model is written once, for ferrite Tensors and numpy arrays alike, as test/gpt2.py is. Its KV cache has a fixed
# size and a token's place is data, not a shape, so on ferrite every decode step launches the same kernels: after the
# first two, each is a replay (tensor.jit). Several sequences can decode together, their caches side by side (forward's
# starts), a token each a step.
#   TTSIM=/path/to/libttsim.so python3 test/llama.py path/to/SmolLM-135M "prompt" [tokens] [bfp8|bfp4|tt]
#   ... python3 test/llama.py path/to/model path/to/<model>.refpt [steps] [mode] [numpy]   # token accuracy, Tenstorrent's way
# (numpy: the weight values the chip holds in mode's formats, with exact products, in numpy alone: no chip)
# The folder holds Hugging Face's HuggingFaceTB/SmolLM-135M config.json, tokenizer.json and model.safetensors (or another
# Llama's: SmolLM2-360M's, or Llama-3.1-8B-Instruct's, whose weights come in several model-*.safetensors with an index).
# bfp8: the weight matrices in bfp8 on the chip, bfp4: the MLP's gate and up in bfp4 and the rest in bfp8 (numpy then has
# them as the chip does), tt: TT's performance precision, formats and fidelity phases both (precision).
import functools, json, math, os, re, sys, time, unicodedata
from types import SimpleNamespace
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ferrite.tensor import Tensor, jit
import gpt2

def fns(x):  # what numpy and ferrite spell differently: gpt2's, and what a Llama adds
  if isinstance(x, np.ndarray):
    topk = lambda a, k: (np.take_along_axis(a, i := np.argsort(-a, -1, kind="stable")[..., :k], -1), i)
    sink_softmax = lambda a, sink: (e := np.exp(a - (m := np.maximum(a.max(-1, keepdims=True), sink)))) / (e.sum(-1, keepdims=True) + np.exp(sink - m))
    return SimpleNamespace(**vars(gpt2.fns(x)), mean=lambda a: a.mean(-1, keepdims=True), rsqrt=lambda a: 1 / np.sqrt(a), topk=topk,
                           silu=lambda a: a / (1 + np.exp(-a)), sigmoid=lambda a: 1 / (1 + np.exp(-a)), repeat=lambda a, n: np.repeat(a, n, 0),
                           mm=lambda a, b, phases: a @ b, experts=numpy_experts, where=np.where, minimum=np.minimum, maximum=np.maximum,
                           sink_softmax=sink_softmax)
  sink_softmax = lambda a, sink: (e := (a - (m := a.max(2).maximum(sink))).exp()) / (e.sum(2) + (sink - m).exp())  # one more logit, left out
  return SimpleNamespace(**vars(gpt2.fns(x)), mean=lambda a: a.mean(2), rsqrt=Tensor.rsqrt, silu=Tensor.silu, sigmoid=Tensor.sigmoid,
                         repeat=Tensor.repeat_interleave, mm=lambda a, b, phases: a.matmul(b, fidelity=phases), topk=Tensor.topk,  # numpy's
                         experts=chip_experts, where=lambda cond, a, b: cond.where(a, b), minimum=Tensor.minimum, maximum=Tensor.maximum,  # products
                         sink_softmax=sink_softmax)  # are exact at any phases

def precision(c, mode="bf16"):
  """Each matmul's weights' format and fidelity phases, {(op, block): (format, phases)}: op qkv, o, gate_up or down by
  block, head (block None), and attention, whose B is the cache (block "decode" or "prefill", format None). bf16, bfp8, or
  bfp4 (the MLP's gate and up in bfp4, the rest in bfp8): four phases, exact products. tt: tt_transformers' performance
  mode at tt-metal 9f98dee9 (COMPARE, "TT's performance precision"): the MLP's gate and up in bfp4 at one phase, the other
  matrices and the output layer in bfp8 at two, attention at two phases in decode and four in prefill; and for
  Llama-3.1-8B (32 blocks 4096 wide) its last block's gate and up in bfp8 at two, as its
  model_params/Llama-3.1-8B-Instruct/performance_decoder_config.json has them; the KV cache (kv, None) in bfp8."""
  tt, fmt = mode == "tt", lambda op: "bfp4" if op == "gate_up" else "bfp8"
  p = {(op, l): (fmt(op), 1 if op == "gate_up" else 2) if tt else (fmt(op) if mode == "bfp4" else mode, 4)
       for l in range(c.layers) for op in ("qkv", "o", "gate_up", "down")}
  if tt and (c.layers, c.dim) == (32, 4096): p["gate_up", 31] = ("bfp8", 2)
  p["head", None] = ("bfp8", 2) if tt else ("bfp8" if mode == "bfp4" else mode, 4)
  return p | {("attention", "decode"): (None, 2 if tt else 4), ("attention", "prefill"): (None, 4), ("kv", None): ("bfp8" if tt else "bf16", None)}

def rms(x, w, eps): f = fns(x); return x * f.rsqrt(f.mean(x * x) + eps) * w

def head(c): return getattr(c, "head_dim", None) or c.dim // c.heads  # a head's width: Qwen3's says it, Llama's is dim / heads

def rope(x, cos, sin, turn):  # Hugging Face's form: x cos + (-x2, x1) sin, x1 and x2 a head's halves, (-x2, x1) as x @ turn
  return x * cos + (x @ turn) * sin

def cache_format(c): return (getattr(c, "precision", None) or {}).get(("kv", None), ("bf16", None))[0]  # the KV cache's format

def layer(x, p, c, rotary, mask, cache, rows, l=0, prefill=False):
  """One block, block l: attention then a SwiGLU MLP (or a mixture of experts: moe), each after an RMSNorm and added
  back. p: the block's weights (as x @ w, so the Hugging Face ones transposed, q, k and v's side by side, and gate and
  up's; and what other families add: Qwen2's qkv_bias, Qwen3's qk_norm and qk_on, gpt-oss's sinks and o_proj.bias), c:
  the config (its precision, if it has one, says each matmul's fidelity phases), rotary: (cos, sin, turn), the tables'
  rows for this call's places; cache: this block's keys' heads then its values', each with a row for every place, this
  call's written at its rows; mask: -1e9 where a token must not look (gpt2.mask); prefill: a prompt's tokens at once,
  not decode's."""
  f, n = fns(x), c.heads + 2 * c.kv
  phases = lambda op, block=l: (getattr(c, "precision", None) or {}).get((op, block), (None, None))[1]  # None: FIDELITY
  h = rms(x, p["input_layernorm"], c.eps)
  qkv = f.mm(h, p["qkv_proj"], phases("qkv"))
  qkv = f.split(qkv + p["qkv_bias"] if "qkv_bias" in p else qkv, n)  # (Qwen2's bias)
  if "qk_norm" in p:  # Qwen3's: each head of q and of k RMS-normed before it turns, in one kernel with v's, which qk_on keeps
    qkv = qkv * f.where(p["qk_on"], f.rsqrt(f.mean(qkv * qkv) + c.eps) * p["qk_norm"], 1)
  qkv = rope(qkv, f.split(rotary[0], n), f.split(rotary[1], n), rotary[2])
  kv = qkv[c.heads:, :, :]  # this call's keys and values, at once (v's heads do not turn); numpy rounds them as a bfp8 cache does
  f.put(cache, rows, gpt2.bfp(kv) if isinstance(kv, np.ndarray) and cache_format(c) == "bfp8" else kv)
  k, v = (f.repeat(t, c.heads // c.kv) for t in (cache[:c.kv, :, :], cache[c.kv:, :, :]))  # a head for each query's, in place
  attention = phases("attention", "prefill" if prefill else "decode")
  s = f.mm(qkv[:c.heads, :, :], f.t(k), attention) * (1 / math.sqrt(head(c)))
  probs = f.softmax(s + mask) if "sinks" not in p else f.sink_softmax(s + mask, p["sinks"])  # (gpt-oss's: a logit of each head's own, left out)
  o = f.mm(f.merge(f.mm(probs, v, attention)), p["o_proj"], phases("o"))
  x = x + (o + p["o_proj.bias"] if "o_proj.bias" in p else o)
  h = rms(x, p["post_attention_layernorm"], c.eps)
  return x + (moe(h, p, c, phases) if "router" in p else mlp(h, p, phases))

def mlp(h, p, phases, pre=""):  # a SwiGLU: silu(h @ gate) * (h @ up), then down; p[pre + "gate_up_proj"] holds gate and up side by side
  f, m = fns(h), p[pre + "down_proj"].shape[-2]
  gate_up = f.mm(h, p[pre + "gate_up_proj"], phases("gate_up"))
  return f.mm(glu(gate_up[:, :m], gate_up[:, m:]), p[pre + "down_proj"], phases("down"))

def glu(gate, up, limit=None):  # SwiGLU's silu(gate) * up; with a limit, gpt-oss's: both clamped there, gate's from above
  f = fns(gate)
  if limit is None: return f.silu(gate) * up
  gate = f.minimum(gate, limit)
  return gate * f.sigmoid(gate * 1.702) * (f.minimum(f.maximum(up, -limit), limit) + 1)

def moe(h, p, c, phases=lambda op: None):
  """A mixture of experts in the MLP's place (Qwen3's, Mixtral's): each token's router logits, its c.k best experts
  (topk, on the chip), and their SwiGLUs weighted by the softmax of their logits (which Qwen3's top-k probabilities,
  renormalized, are too) and summed; and a shared expert's SwiGLU, if the block has one, gated by a sigmoid of its own.
  p: router (dim x experts), experts.gate_up and experts.down (a batch of the experts' matrices), and gpt-oss's biases
  (router.bias, experts.gate_up_bias and experts.down_bias, a row each expert); phases(op): the fidelity phases of the
  experts' gate_up and down. c.limit: gpt-oss's GLU (glu)."""
  f, norm = fns(h), getattr(c, "norm_topk", True)
  logits = f.mm(h, p["router"], None)
  logits = logits + p["router.bias"] if "router.bias" in p else logits  # (gpt-oss's bias)
  values, ids = f.topk(logits if norm else f.softmax(logits), c.k)  # (without norm_topk_prob, the chosen ones' probabilities among all)
  y = f.experts(h, p, ids, f.softmax(values) if norm else values, (phases("gate_up"), phases("down")), getattr(c, "limit", None))
  if "shared_expert_gate" not in p: return y
  return y + f.sigmoid(f.mm(h, p["shared_expert_gate"], None)) * mlp(h, p, phases, "shared_expert.")  # Qwen2-MoE's, every token's

def numpy_experts(h, p, ids, w, phases, limit=None):  # each token's chosen experts' GLUs, weighted by w and summed
  gate_up, down, m = p["experts.gate_up"], p["experts.down"], p["experts.down"].shape[-2]
  bias = lambda name, e: p[name][e] if name in p else 0
  def expert(x, e):
    g = x @ gate_up[e] + bias("experts.gate_up_bias", e)
    return glu(g[:m], g[m:], limit) @ down[e] + bias("experts.down_bias", e)
  return np.stack([sum(w[t, j] * expert(h[t], e) for j, e in enumerate(ids[t])) for t in range(len(h))])

def chip_experts(h, p, ids, w, phases, limit=None):
  """numpy_experts on the chip, by matmuls that pick each expert's weights where they lie (index=). For a few tokens (a
  decode step) a batch entry for each token and expert, the token's row by its expert's matrices; for more choices
  than twice the experts (a prompt), each expert's tokens grouped into whole tiles (Tensor.group), a batch entry for
  each tile, so its weights are read once for every 32 of its tokens rather than once for each. Either replays. An
  expert's biases are gathered by the same picks."""
  gate_up, down = p["experts.gate_up"], p["experts.down"]
  (n, k), m, dim = ids.shape[1:], down.rows, down.cols
  if n * k > 2 * gate_up.batch:  # the entries: each expert's tiles of tokens, or each token's row k times
    order, picks, places = ids.group(gate_up.batch)
    x = h.gather(order).split(picks.rows, 0)
  else: x, picks = h.reshape(n, 1, dim) if n == 1 else h.reshape(n, 1, dim).repeat_interleave(k), ids
  entries = picks.rows * picks.cols
  row = picks.reshape(1, entries) if {"experts.gate_up_bias", "experts.down_bias"} & set(p) else None  # the picks as a row, to gather biases by
  plus = lambda y, name: y + p[name].gather(row).reshape(entries, 1, p[name].cols) if name in p else y  # each entry's expert's bias
  x = plus(x.matmul(gate_up, fidelity=phases[0], index=picks), "experts.gate_up_bias")
  y = plus(glu(x[:, :, :m], x[:, :, m:], limit).matmul(down, fidelity=phases[1], index=picks), "experts.down_bias")
  if picks is ids: return (y * w.reshape(n * k, 1, 1)).reshape(n, k, dim).sum(1).reshape(n, dim)
  y, run = y.reshape(order.cols, dim), places.cols // k  # choice j's places
  terms = [y.gather(places[:, :, j * run:j * run + n]) * w[:, :, j:j + 1] for j in range(k)]
  return sum(terms[1:], terms[0])

def frequencies(c):  # RoPE's, a head's half of them, in fp32 as Hugging Face computes them; with Llama 3's scaling
  hd = head(c)  # (rope_type llama3), long wavelengths slowed down by its factor, short ones kept, and those
  inv = 1 / c.theta ** (np.arange(0, hd, 2, dtype=np.float32) / hd)  # between blended
  if (r := getattr(c, "rope", None)) and r.get("rope_type") == "llama3":
    old, wavelen = r["original_max_position_embeddings"], 2 * math.pi / inv
    scaled = np.where(wavelen > old / r["low_freq_factor"], inv / r["factor"], inv)
    smooth = (old / wavelen - r["low_freq_factor"]) / (r["high_freq_factor"] - r["low_freq_factor"])
    between = (wavelen >= old / r["high_freq_factor"]) & (wavelen <= old / r["low_freq_factor"])
    inv = np.where(between, (1 - smooth) * scaled / r["factor"] + smooth * scaled, scaled).astype(np.float32)
  if r and r.get("rope_type") == "yarn":  # YaRN's (gpt-oss's): short wavelengths kept, long ones slowed by its factor, a ramp between
    turns = lambda n: hd * math.log(r["original_max_position_embeddings"] / (n * 2 * math.pi)) / (2 * math.log(c.theta))  # the dimension
    low, high = turns(r.get("beta_fast") or 32), turns(r.get("beta_slow") or 1)                                            # turning n times
    if r.get("truncate", True): low, high = math.floor(low), math.ceil(high)
    low, high = max(low, 0), min(high, hd - 1)
    ramp = np.clip((np.arange(hd // 2, dtype=np.float32) - low) / (high - low if high != low else 0.001), 0, 1)
    inv = (inv / r["factor"] * ramp + inv * (1 - ramp)).astype(np.float32)
  return inv

def rotary(c, n):  # cos and sin for places 0 to n - 1, as Hugging Face's (angles in fp32, the tables in bf16), a head's
  # for each of q's, k's and v's heads (v's 1 and 0: they do not turn, so one kernel turns them all), and turn:
  hd, inv = head(c), frequencies(c)  # x @ turn is (-x2, x1), each of its entries one of x's, so exact
  angles = np.arange(n, dtype=np.float32)[:, None] * inv[None]
  bf16 = lambda a: gpt2.bf16_floats(gpt2.bf16_bits(a))
  turn = np.zeros((hd, hd), np.float32)
  turn[np.arange(hd // 2) + hd // 2, np.arange(hd // 2)], turn[np.arange(hd // 2), np.arange(hd // 2) + hd // 2] = -1, 1
  heads = lambda t, v: np.concatenate([t] * (c.heads + c.kv) + [np.full_like(t, v)] * c.kv, 1)
  r = getattr(c, "rope", None) or {}  # YaRN's tables scaled up by its attention factor
  scale = r.get("attention_factor") or (0.1 * math.log(r["factor"]) + 1 if r.get("rope_type") == "yarn" and r["factor"] > 1 else 1)
  return [bf16(heads(np.concatenate([g(angles) * np.float32(scale)] * 2, 1), v)) for g, v in ((np.cos, 1), (np.sin, 0))] + [turn]

class Weight:
  """A weight read from the checkpoint only when it is wanted, so the host never holds more of a big model at once than
  a matrix or two: bits() gives its bf16 bits as x @ w takes them, values() the same as float32."""
  def __init__(self, bits): self.bits = bits
  def values(self): return gpt2.bf16_floats(self.bits())

def fuse(w, c):  # each block's q, k and v projections as one matrix, and its gate and up: fewer and wider matmuls; a
  # mixture of experts' router (gate, Qwen3's) transposed for x @ w, and its experts' gates and ups, and downs, each a
  # batch; Qwen3's norms of q's and k's heads as a batch of every head's (v's 1s), and which heads they norm (qk_on)
  cat = lambda xs: np.ascontiguousarray(np.concatenate(xs, 1))
  join = lambda *xs: Weight(lambda: cat([x.bits() for x in xs])) if isinstance(xs[0], Weight) else cat(xs)
  stack = lambda xs: Weight(lambda: np.stack([x.bits() for x in xs])) if isinstance(xs[0], Weight) else np.stack(xs)
  const = lambda shape, v: Weight(lambda: np.full(shape, gpt2.bf16_bits(np.float32(v)), np.uint16))
  for pre in {k[:k.index(".", 7) + 1] for k in w if k.startswith("layers.")}:
    w[pre + "qkv_proj"] = join(*(w.pop(pre + n) for n in ("q_proj", "k_proj", "v_proj")))
    if pre + "q_proj.bias" in w: w[pre + "qkv_bias"] = join(*(w.pop(pre + n) for n in ("q_proj.bias", "k_proj.bias", "v_proj.bias")))
    if pre + "shared_expert_gate" in w:  # Qwen2-MoE's shared expert, and its gate transposed for x @ w
      w[pre + "shared_expert.gate_up_proj"] = join(w.pop(pre + "shared_expert.gate_proj"), w.pop(pre + "shared_expert.up_proj"))
      g = w.pop(pre + "shared_expert_gate")
      w[pre + "shared_expert_gate"] = Weight(lambda g=g: np.ascontiguousarray(g.bits().T)) if isinstance(g, Weight) else np.ascontiguousarray(g.T)
    if pre + "q_norm" in w:
      q, k = w.pop(pre + "q_norm"), w.pop(pre + "k_norm")
      w[pre + "qk_norm"] = stack([q] * c.heads + [k] * c.kv + [const((1, head(c)), 1)] * c.kv)
      w[pre + "qk_on"] = stack([const((1, 1), 1)] * (c.heads + c.kv) + [const((1, 1), 0)] * c.kv)
    if pre + "sinks" in w: w[pre + "sinks"] = Weight(lambda s=w[pre + "sinks"]: s.bits().reshape(-1, 1, 1))  # gpt-oss's, a head's each
    if pre + "experts.gate_up_proj" in w:  # gpt-oss's experts, a batch as x @ w takes them already, gate and up interleaved
      apart = lambda x: Weight(lambda: np.ascontiguousarray(np.concatenate([x.bits()[..., ::2], x.bits()[..., 1::2]], -1)))
      w[pre + "experts.gate_up"], w[pre + "experts.down"] = apart(w.pop(pre + "experts.gate_up_proj")), w.pop(pre + "experts.down_proj")
      w[pre + "experts.gate_up_bias"] = apart(w.pop(pre + "experts.gate_up_proj_bias"))
      w[pre + "experts.down_bias"] = w.pop(pre + "experts.down_proj_bias")
      r = w.pop(pre + "router")
      w[pre + "router"] = Weight(lambda r=r: np.ascontiguousarray(r.bits().T))
      continue
    if pre + "gate" not in w: w[pre + "gate_up_proj"] = join(w.pop(pre + "gate_proj"), w.pop(pre + "up_proj")); continue
    r = w.pop(pre + "gate")
    w[pre + "router"] = Weight(lambda r=r: np.ascontiguousarray(r.bits().T)) if isinstance(r, Weight) else np.ascontiguousarray(r.T)
    n = sum(k.startswith(pre + "experts.") and k.endswith(".down_proj") for k in w)
    expert = lambda name: [w.pop(f"{pre}experts.{e}.{name}") for e in range(n)]
    gates, ups = expert("gate_proj"), expert("up_proj")
    w[pre + "experts.gate_up"], w[pre + "experts.down"] = stack([join(g, u) for g, u in zip(gates, ups)]), stack(expert("down_proj"))
  return w

def forward(ids, places, w, c, caches, starts=None, every=False):
  """The logits for the token after ids, the tokens at places in the sequence: their keys and values go into caches (a
  block's keys' heads then its values', each with a row for every place), and each sees the places up to its own. ids and places are
  rows: lists for numpy, int32 tensors for ferrite. starts: several sequences decoding together (a batch of users), their
  caches side by side, each token's sequence starting at its start's row and each token seeing only its own sequence;
  then the logits are every token's, the next token of each (on the chip only). every: every token's logits, not only
  the last's. w is looked up a block at a time (arrays makes numpy's weights as they are looked up)."""
  f = fns(w["norm"])
  x, rows = f.rows(w["embed"], ids), places if starts is None else places + starts
  rotary, mask = (f.rows(w["cos"], places), f.rows(w["sin"], places), w["turn"]), gpt2.mask(rows, caches[0].shape[-2], starts)
  window = gpt2.mask(rows, caches[0].shape[-2], starts, c.sliding) if getattr(c, "sliding", None) else None  # (a sliding window's blocks)
  sliding = lambda l: window is not None and (not getattr(c, "layer_types", None) or c.layer_types[l] == "sliding_attention")
  prefill = starts is None and x.shape[-2] > 1
  for l in range(c.layers):
    p = {k[len(f"layers.{l}."):]: w[k] for k in w if k.startswith(f"layers.{l}.")}
    x = layer(x, p, c, rotary, window if sliding(l) else mask, caches[l], rows, l, prefill)
  h = rms(x, w["norm"], c.eps)
  phases = (getattr(c, "precision", None) or {}).get(("head", None), (None, None))[1]  # the output layer's
  return f.mm(h if starts is not None or every else f.rows(h, [x.shape[-2] - 1]), w["embed.T"], phases)

def prefill(ids, w, c, caches, chunk, on=lambda row: row):
  """forward over a prompt from place 0, chunk tokens at a time, as Tenstorrent's prefill takes a long one: each chunk's
  keys and values go into caches before the next chunk reads them, so attention's scores are a chunk's rows by the
  cache's places rather than the whole prompt squared. The logits for the token after the prompt. on: a row of ids or
  places as forward takes it (on ferrite, an int32 tensor)."""
  for i in range(0, len(ids), chunk): out = forward(on(ids[i:i + chunk]), on(list(range(i, min(i + chunk, len(ids))))), w, c, caches)
  return out

def kv_cache(c, size, zeros):  # each block's keys' heads then its values', a row for each of size places, made by zeros(shape)
  return [zeros((2 * c.kv, size, head(c))) for _ in range(c.layers)]

def read_json(path):
  with open(path, encoding="utf-8") as f: return json.load(f)

def load(folder, places=4096):
  """The config, and {name: Weight} from model.safetensors, or from the files model.safetensors.index.json names (each 8
  bytes of header size, a JSON header naming each tensor's dtype, shape and byte range, then the data), named without
  "model.", linear weights transposed (so x @ w), vectors as one-row matrices; the output layer is the embedding unless
  the model has its own (lm_head); a mixture of experts' as fuse stacks them. RoPE's tables have a row for each of the first
  `places` places (at most the model's)."""
  j = read_json(os.path.join(folder, "config.json"))
  rope = j.get("rope_parameters") or (j.get("rope_scaling") or {}) | {"rope_theta": j["rope_theta"]}  # transformers 5's, else the two before it
  theta, kind = rope.pop("rope_theta"), rope.pop("type", None)  # (older configs say type, not rope_type)
  rope.setdefault("rope_type", kind or "default")
  assert rope["rope_type"] in ("default", "llama3", "yarn"), f"RoPE scaling {rope}: not yet"
  c = SimpleNamespace(dim=j["hidden_size"], heads=j["num_attention_heads"], kv=j["num_key_value_heads"], layers=j["num_hidden_layers"],
                      eps=j["rms_norm_eps"], theta=theta, places=min(j["max_position_embeddings"], places), vocab=j["vocab_size"],
                      rope=rope if rope["rope_type"] != "default" else None, ffn=j["intermediate_size"], head_dim=j.get("head_dim"),
                      k=j.get("num_experts_per_tok"), norm_topk=j.get("norm_topk_prob", True), layer_types=j.get("layer_types"),
                      limit=j.get("swiglu_limit", 7.0 if j.get("model_type") == "gpt_oss" else None),  # gpt-oss's GLU
                      sliding=j.get("sliding_window") if j.get("use_sliding_window", True) else None)
  index = os.path.join(folder, "model.safetensors.index.json")
  files = sorted(set(read_json(index)["weight_map"].values())) if os.path.exists(index) else ["model.safetensors"]
  out, tensors = {}, {}
  for file in files:
    header, data = gpt2.safetensors(os.path.join(folder, file))
    tensors |= {name: (data, t) for name, t in header.items() if name != "__metadata__"}
  for full, (data, t) in tensors.items():
    if full.endswith("_scales"): continue  # (an MXFP4 tensor's, read with its blocks)
    name = full.removeprefix("model.").replace("self_attn.", "").replace("mlp.", "").replace("block_sparse_moe.", "").removesuffix(".weight")
    name = re.sub(r"(experts\.\d+)\.w([123])$", lambda m: m[1] + (".gate_proj", ".down_proj", ".up_proj")[int(m[2]) - 1], name)  # Mixtral's
    if name.startswith("layers.") and int(name.split(".")[1]) >= c.layers: continue  # (a config cut to its first blocks)
    if name.endswith("_blocks"):
      scales = tensors[full.removesuffix("_blocks") + "_scales"]
      out[name.removesuffix("_blocks")] = Weight(lambda b=(data, t), s=scales: mxfp4(gpt2.u8(*b), gpt2.u8(*s)))
    else: out[name] = Weight(functools.partial(gpt2.tensor_bits, data, t, (name.endswith("_proj") or name == "lm_head") and len(t["shape"]) == 2))
  out["embed"] = out.pop("embed_tokens")
  out["embed.T"] = out.pop("lm_head", None) or Weight(lambda e=out["embed"]: np.ascontiguousarray(e.bits().T))  # tied, or its own
  for name, a in zip(("cos", "sin", "turn"), rotary(c, c.places)): out[name] = Weight(lambda a=a: gpt2.bf16_bits(a))
  return c, fuse(out, c)

def mxfp4(blocks, scales):
  """gpt-oss's experts in MXFP4: 4-bit values (sign, 2 bits of exponent, 1 of mantissa), two to a byte, the low one
  first, each 32 sharing a power of 2 (a byte, 127 for 1), as a batch of (out x in) matrices; as bf16 bits, exactly (a
  value has 2 significant bits at most), each matrix transposed for x @ w."""
  values = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], np.float32)
  v = np.stack([values[blocks & 15], values[blocks >> 4]], -1).reshape(*scales.shape, 32)
  v = np.ldexp(v, scales.astype(np.int32)[..., None] - 127).reshape(*scales.shape[:-1], -1)
  return gpt2.bf16_bits(np.ascontiguousarray(np.swapaxes(v, -1, -2)))

def weight_format(name, prec):  # a weight's format on the chip: the matrices a matmul reads as its B (the blocks', their
  # experts' as their MLP's, and the output layer) in the one prec (precision's) gives them, the rest, and all without prec, bf16
  op = name.split(".")[-1].removesuffix("_proj")
  matrix = name.endswith(("_proj", "experts.gate_up", "experts.down"))
  key = ("head", None) if name == "embed.T" else (op, int(name.split(".")[1])) if matrix else None
  return prec[key][0] if prec and key else "bf16"

def upload(rt, weights, prec=None):  # the weights as ferrite Tensors, one at a time, each cast on the chip to its format
  t = {}                                # and waited for, so the next weight's bf16 copy goes where this one's was: a model
  for name, w in weights.items():       # takes the DRAM its formats need and one matrix more, not all its bf16 besides
    bits = w.bits()
    x = t[name] = Tensor.frombytes(rt, *bits.shape[-2:], bits.tobytes(), bits.shape[0] if bits.ndim == 3 else 1)
    if (f := weight_format(name, prec)) != "bf16":
      t[name] = x.cast(f).realize()
      del x
      rt.wait(3600)  # a big matrix's cast is one launch, minutes long on ttsim
  return t

class Made(dict):  # name -> what makes its value: made at each lookup and not kept
  def __getitem__(self, k): return super().__getitem__(k)()

def arrays(weights, prec=None):
  """The same as numpy float32 arrays, each as the chip holds it in its format (weight_format), a block float rounded as
  the chip rounds it (gpt2.bfp, bit for bit: test_bfp), a thousand rows at a time. Each is read and made only when looked
  up and not kept, so numpy holds no more of a model than the block it is running, whatever the model's size."""
  def made(name, w):
    v, f = w.values(), weight_format(name, prec)
    if f == "bf16": return v
    rows = v.reshape(-1, v.shape[-1])  # (a batch of experts' too)
    return np.concatenate([gpt2.bfp(rows[i:i + 1024], f).astype(np.float32) for i in range(0, len(rows), 1024)]).reshape(v.shape)
  return Made({name: functools.partial(made, name, w) for name, w in weights.items()})

def logits(f_out): return gpt2.values(f_out)[0]  # a ferrite row of logits, as a numpy vector

def zeros(rt, dtype="bf16"):  # zeros of a shape on the chip, in dtype (bfp8: a bfp8 KV cache), for kv_cache
  return lambda shape: Tensor.frombytes(rt, shape[1], shape[2], bytes(2 * math.prod(shape)), shape[0]).cast(dtype).realize()

def reference(path):
  """Tenstorrent's reference file for a model (tt_transformers' tests/reference_outputs/<model>.refpt, made by its
  tests/generate_reference_hf.py): the first 1,024 tokens of A Tale of Two Cities, and at each place the Hugging Face
  model's five most likely next tokens. The prompt (the first half of the tokens), the tokens after it, and the five at
  each place from the prompt's last on (so the i-th are for the token after the i-th step's input)."""
  import torch
  data = torch.load(path)
  tokens = data["reference_tokens"][0].tolist()
  half = len(tokens) // 2
  return tokens[:half], tokens[half:], data["top5_tokens"][half - 1:].tolist()

def accuracy(rt, t, c, prompt, after, top5, steps=500):
  """Tenstorrent's token accuracy (simple_text_demo.py's TokenAccuracy) of the model on the chip: the prompt prefilled,
  then each step fed the reference's next token whatever the model predicted (teacher forcing), the steps replayed
  (jit). (top-1, top-5, the model's tokens): the shares of steps whose most likely token is the reference model's most
  likely, or among its five. rt None: the model in numpy, t its weights as numpy arrays (arrays: as the chip holds
  them), with exact products, which tells what the formats cost from what the chip's arithmetic does. Teacher forcing
  knows every step's input beforehand, so numpy takes the prompt and the steps as one prefill, every place's logits at
  once: what decode a step at a time computes, each block's weights read once rather than once a step."""
  n, size = len(prompt), -(-(len(prompt) + steps) // 32) * 32
  if rt is None:
    ids = prompt + after[:steps - 1]
    picks = [int(row.argmax()) for row in forward(ids, list(range(len(ids))), t, c, kv_cache(c, size, np.zeros), every=True)[n - 1:]]
  else:
    on, kv = lambda row: Tensor.of(rt, [row], "int32"), kv_cache(c, size, zeros(rt, cache_format(c)))
    replay = jit(lambda ids, places: forward(ids, places, t, c, kv))
    picks = [int(logits(forward(on(prompt), on(list(range(n))), t, c, kv)).argmax())]
    for i in range(steps - 1): picks.append(int(logits(replay(on([after[i]]), on([n + i]))).argmax()))
  return sum(p == f[0] for p, f in zip(picks, top5)) / steps, sum(p in f for p, f in zip(picks, top5)) / steps, picks

WHITE = "\t-\r \x85\xa0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000"  # Unicode's White_Space: what tiktoken's \s matches
SPLIT = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""  # SmolLM's (GPT-2's, digits one at a time)

@functools.cache
def categories(names):  # Unicode's general categories (L, or Lu, ...), which Python's re lacks, as a character class's ranges
  runs = {n: [] for n in names}
  for cp in range(0x110000):
    cat = unicodedata.category(chr(cp))
    for r in (runs.get(cat[0]), runs.get(cat)):
      if r is None: continue
      if r and r[-1][1] == cp - 1: r[-1][1] = cp
      else: r.append([cp, cp])
  return {k: "".join(re.escape(chr(a)) + (f"-{re.escape(chr(b))}" if b > a else "") for a, b in r) for k, r in runs.items()}

def python_regex(pat):  # tokenizer.json's split for Python's re: \p{L}, \p{Lu}, ... and \s spelled out (Python's \s also
  ranges = categories(tuple(sorted(set(re.findall(r"\\p\{(\w+)\}", pat))))) | {"s": WHITE}  # takes in four controls), bare inside a class
  out, inside, i = [], False, 0
  while i < len(pat):
    if pat.startswith(("\\p{", "\\s", "\\S"), i):
      key, i = (pat[i + 3:pat.index("}", i)], pat.index("}", i) + 1) if pat[i + 1] == "p" else (pat[i + 1], i + 2)
      assert not (inside and key == "S"), "\\S in a class"
      out.append(f"[^{WHITE}]" if key == "S" else ranges[key] if inside else f"[{ranges[key]}]")
    else:
      n = 2 if pat[i] == "\\" else 1
      inside = pat[i] == "[" if not inside else pat[i] != "]"
      out.append(pat[i:i + n])
      i += n
  return re.compile("".join(out))

class Tokenizer:
  """tokenizer.json's byte-level BPE in plain Python, as tiktoken does it: the text split by its regex, and each piece
  that is not a token itself merged a pair of parts at a time, the lowest-ranked first (ids are in merge order, so they
  are the ranks). Special tokens in the text are taken as text, unless encode is told to take them as tokens."""
  def __init__(self, folder):
    t = read_json(os.path.join(folder, "tokenizer.json"))
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))  # GPT-2's bytes as printable characters
    byte = dict(zip(map(chr, bs + [256 + i for i in range(256 - len(bs))]), bs + [b for b in range(256) if b not in bs]))
    special = {a["content"]: a["id"] for a in t["added_tokens"]}
    self.ranks = {bytes(byte[ch] for ch in tok): i for tok, i in t["model"]["vocab"].items() if tok not in special}
    self.bytes = {i: b for b, i in self.ranks.items()} | {i: s.encode() for s, i in special.items()}
    self.special, self.specials = special, re.compile("(" + "|".join(map(re.escape, sorted(special, key=len, reverse=True))) + ")")
    pre = t.get("pre_tokenizer") or {}  # the split before BPE: tokenizer.json's own (Llama 3's), else SmolLM's
    split = [p["pattern"]["Regex"] for p in pre.get("pretokenizers", [pre]) if p.get("type") == "Split" and "Regex" in p.get("pattern", {})]
    self.split, self.pieces = python_regex(split[0] if split else SPLIT), {}
    self.nfc = (t.get("normalizer") or {}).get("type") == "NFC"  # Qwen's: text composed first (an e and its accent as one letter)

  def bpe(self, piece):  # a piece's bytes as ids
    if piece not in self.pieces:
      parts = [piece] if piece in self.ranks else [piece[i:i + 1] for i in range(len(piece))]
      while len(parts) > 1:
        rank, i = min((self.ranks.get(a + b, math.inf), i) for i, (a, b) in enumerate(zip(parts, parts[1:])))
        if rank == math.inf: break
        parts[i:i + 2] = [parts[i] + parts[i + 1]]
      self.pieces[piece] = [self.ranks[p] for p in parts]
    return self.pieces[piece]
  def encode(self, text, special=False):  # special: a special token's text in it is that token (a chat template's)
    if special: return [i for part in self.specials.split(text) if part for i in ([self.special[part]] if part in self.special else self.encode(part))]
    return [i for piece in self.split.findall(unicodedata.normalize("NFC", text) if self.nfc else text) for i in self.bpe(piece.encode())]
  def decode(self, ids): return b"".join(self.bytes[i] for i in ids).decode("utf-8", "replace")

def main():
  from ferrite import pci
  from ferrite.chip import Chip
  from ferrite.runtime import Runtime
  folder, prompt, steps = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "The meaning of life is", int(sys.argv[3]) if len(sys.argv) > 3 else 8
  mode = sys.argv[4] if len(sys.argv) > 4 else "bf16"
  if prompt.endswith(".refpt"):  # token accuracy against Tenstorrent's reference file, `steps` steps (500 as it runs it)
    first, after, top5 = reference(prompt)
    c, weights = load(folder, -(-(len(first) + steps) // 32) * 32)
    c.precision, on_host, start = precision(c, mode), "numpy" in sys.argv[5:], time.time()  # numpy: the same weight values, exact products
    if on_host: top1, top5_share, _ = accuracy(None, arrays(weights, c.precision), c, first, after, top5, steps)
    else:
      chip = Chip(pci.open_device())
      rt = Runtime(chip, chip.tensix[0], chip.tensix[-1])
      t = upload(rt, weights, c.precision)
      print(f"weights on the chip in {time.time() - start:.0f}s, {mode}")
      top1, top5_share, _ = accuracy(rt, t, c, first, after, top5, steps)
    return print(f"token accuracy over {steps} steps after a {len(first)}-token prompt{' in numpy' if on_host else ''}: top-1 {100 * top1:.1f}%, top-5 {100 * top5_share:.1f}%"
                 f" ({time.time() - start:.0f}s)")
  enc, chip = Tokenizer(folder), Chip(pci.open_device())
  rt = Runtime(chip, chip.tensix[0], chip.tensix[-1])
  start = time.time()
  ids, agree = enc.encode(prompt), 0
  size = -(-(len(ids) + steps) // 32) * 32  # a row for each place the generation reaches, in whole tiles
  c, weights = load(folder, size)
  c.precision = precision(c, mode)
  t, w = upload(rt, weights, c.precision), arrays(weights, c.precision)
  count = sum(x.rows * x.cols for k, x in t.items() if k not in ("cos", "sin", "turn"))  # (a tied output layer twice)
  print(f"{count / 1e6:.0f}M weights on the chip in {time.time() - start:.0f}s")
  kv, np_kv = kv_cache(c, size, zeros(rt, cache_format(c))), kv_cache(c, size, np.zeros)
  on = lambda row: Tensor.of(rt, [row], "int32")  # a row of ids or places, on the chip
  step, new = jit(lambda ids, places: forward(ids, places, t, c, kv)), ids  # a token at a time, replayed from the third
  print(prompt, end="", flush=True)
  for i in range(steps):  # the prompt at once, then a token at a time, each seeing the keys and values cached so far
    start, places = time.time(), list(range(len(ids) - len(new), len(ids)))
    got = logits(forward(on(new), on(places), t, c, kv) if i == 0 else step(on(new), on(places)))
    want = forward(new, places, w, c, np_kv)[0]
    agree += int(got.argmax()) == int(want.argmax())
    new = [int(want.argmax())]  # numpy's pick, so that one near tie does not send the two down different paths
    ids += new
    print(enc.decode(new) + ("" if got.argmax() == want.argmax() else f"[ferrite: {enc.decode([int(got.argmax())])!r}]"), end="", flush=True)
    print(f" ({time.time() - start:.0f}s, logits within {np.abs(got - want).max():.2f})", end="", file=sys.stderr, flush=True)
  print(f"\nferrite picked the same token as numpy {agree} times out of {steps}")

if __name__ == "__main__": main()
