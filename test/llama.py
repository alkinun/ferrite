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
    return SimpleNamespace(**vars(gpt2.fns(x)), mean=lambda a: a.mean(-1, keepdims=True), rsqrt=lambda a: 1 / np.sqrt(a),
                           silu=lambda a: a / (1 + np.exp(-a)), repeat=lambda a, n: np.repeat(a, n, 0), mm=lambda a, b, phases: a @ b)
  return SimpleNamespace(**vars(gpt2.fns(x)), mean=lambda a: a.mean(2), rsqrt=Tensor.rsqrt, silu=Tensor.silu, repeat=Tensor.repeat_interleave,
                         mm=lambda a, b, phases: a.matmul(b, fidelity=phases))  # numpy's products are exact at any phases

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

def rope(x, cos, sin, turn):  # Hugging Face's form: x cos + (-x2, x1) sin, x1 and x2 a head's halves, (-x2, x1) as x @ turn
  return x * cos + (x @ turn) * sin

def cache_format(c): return (getattr(c, "precision", None) or {}).get(("kv", None), ("bf16", None))[0]  # the KV cache's format

def layer(x, p, c, rotary, mask, cache, rows, l=0, prefill=False):
  """One block, block l: attention then a SwiGLU MLP, each after an RMSNorm and added back. p: the block's weights (as
  x @ w, so the Hugging Face ones transposed, q, k and v's side by side, and gate and up's), c: the config (its precision,
  if it has one, says each matmul's fidelity phases), rotary: (cos, sin, turn), the tables' rows for this call's places;
  cache: this block's keys' heads then its values', each with a row for every place, this call's written at its rows;
  mask: -1e9 where a token must not look (gpt2.mask); prefill: a prompt's tokens at once, not decode's."""
  f, n = fns(x), c.heads + 2 * c.kv
  phases = lambda op, block=l: (getattr(c, "precision", None) or {}).get((op, block), (None, None))[1]  # None: FIDELITY
  h = rms(x, p["input_layernorm"], c.eps)
  qkv = rope(f.split(f.mm(h, p["qkv_proj"], phases("qkv")), n), f.split(rotary[0], n), f.split(rotary[1], n), rotary[2])
  kv = qkv[c.heads:, :, :]  # this call's keys and values, at once (v's heads do not turn); numpy rounds them as a bfp8 cache does
  f.put(cache, rows, gpt2.bfp(kv) if isinstance(kv, np.ndarray) and cache_format(c) == "bfp8" else kv)
  k, v = (f.repeat(t, c.heads // c.kv) for t in (cache[:c.kv, :, :], cache[c.kv:, :, :]))  # a head for each query's, in place
  attention = phases("attention", "prefill" if prefill else "decode")
  s = f.mm(qkv[:c.heads, :, :], f.t(k), attention) * (1 / math.sqrt(c.dim // c.heads))
  x = x + f.mm(f.merge(f.mm(f.softmax(s + mask), v, attention)), p["o_proj"], phases("o"))
  h = rms(x, p["post_attention_layernorm"], c.eps)
  gate_up, m = f.mm(h, p["gate_up_proj"], phases("gate_up")), p["down_proj"].shape[-2]
  return x + f.mm(f.silu(gate_up[:, :m]) * gate_up[:, m:], p["down_proj"], phases("down"))

def frequencies(c):  # RoPE's, a head's half of them, in fp32 as Hugging Face computes them; with Llama 3's scaling
  hd = c.dim // c.heads  # (rope_type llama3), long wavelengths slowed down by its factor, short ones kept, and those
  inv = 1 / c.theta ** (np.arange(0, hd, 2, dtype=np.float32) / hd)  # between blended
  if (r := getattr(c, "rope", None)) and r.get("rope_type") == "llama3":
    old, wavelen = r["original_max_position_embeddings"], 2 * math.pi / inv
    scaled = np.where(wavelen > old / r["low_freq_factor"], inv / r["factor"], inv)
    smooth = (old / wavelen - r["low_freq_factor"]) / (r["high_freq_factor"] - r["low_freq_factor"])
    between = (wavelen >= old / r["high_freq_factor"]) & (wavelen <= old / r["low_freq_factor"])
    inv = np.where(between, (1 - smooth) * scaled / r["factor"] + smooth * scaled, scaled).astype(np.float32)
  return inv

def rotary(c, n):  # cos and sin for places 0 to n - 1, as Hugging Face's (angles in fp32, the tables in bf16), a head's
  # for each of q's, k's and v's heads (v's 1 and 0: they do not turn, so one kernel turns them all), and turn:
  hd, inv = c.dim // c.heads, frequencies(c)  # x @ turn is (-x2, x1), each of its entries one of x's, so exact
  angles = np.arange(n, dtype=np.float32)[:, None] * inv[None]
  bf16 = lambda a: gpt2.bf16_floats(gpt2.bf16_bits(a))
  turn = np.zeros((hd, hd), np.float32)
  turn[np.arange(hd // 2) + hd // 2, np.arange(hd // 2)], turn[np.arange(hd // 2), np.arange(hd // 2) + hd // 2] = -1, 1
  heads = lambda t, v: np.concatenate([t] * (c.heads + c.kv) + [np.full_like(t, v)] * c.kv, 1)
  return [bf16(heads(np.concatenate([g(angles)] * 2, 1), v)) for g, v in ((np.cos, 1), (np.sin, 0))] + [turn]

class Weight:
  """A weight read from the checkpoint only when it is wanted, so the host never holds more of a big model at once than
  a matrix or two: bits() gives its bf16 bits as x @ w takes them, values() the same as float32."""
  def __init__(self, bits): self.bits = bits
  def values(self): return gpt2.bf16_floats(self.bits())

def fuse(w):  # each block's q, k and v projections as one matrix, and its gate and up: fewer and wider matmuls
  cat = lambda xs: np.ascontiguousarray(np.concatenate(xs, 1))
  join = lambda *xs: Weight(lambda: cat([x.bits() for x in xs])) if isinstance(xs[0], Weight) else cat(xs)
  for pre in {k[:k.index(".", 7) + 1] for k in w if k.startswith("layers.")}:
    w[pre + "qkv_proj"] = join(*(w.pop(pre + n) for n in ("q_proj", "k_proj", "v_proj")))
    w[pre + "gate_up_proj"] = join(w.pop(pre + "gate_proj"), w.pop(pre + "up_proj"))
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
  prefill = starts is None and x.shape[-2] > 1
  for l in range(c.layers):
    x = layer(x, {k[len(f"layers.{l}."):]: w[k] for k in w if k.startswith(f"layers.{l}.")}, c, rotary, mask, caches[l], rows, l, prefill)
  h = rms(x, w["norm"], c.eps)
  head = (getattr(c, "precision", None) or {}).get(("head", None), (None, None))[1]
  return f.mm(h if starts is not None or every else f.rows(h, [x.shape[-2] - 1]), w["embed.T"], head)

def prefill(ids, w, c, caches, chunk, on=lambda row: row):
  """forward over a prompt from place 0, chunk tokens at a time, as Tenstorrent's prefill takes a long one: each chunk's
  keys and values go into caches before the next chunk reads them, so attention's scores are a chunk's rows by the
  cache's places rather than the whole prompt squared. The logits for the token after the prompt. on: a row of ids or
  places as forward takes it (on ferrite, an int32 tensor)."""
  for i in range(0, len(ids), chunk): out = forward(on(ids[i:i + chunk]), on(list(range(i, min(i + chunk, len(ids))))), w, c, caches)
  return out

def kv_cache(c, size, zeros):  # each block's keys' heads then its values', a row for each of size places, made by zeros(shape)
  return [zeros((2 * c.kv, size, c.dim // c.heads)) for _ in range(c.layers)]

def load(folder, places=4096):
  """The config, and {name: Weight} from model.safetensors, or from the files model.safetensors.index.json names (each 8
  bytes of header size, a JSON header naming each tensor's dtype, shape and byte range, then the data), named without
  "model.", linear weights transposed (so x @ w), vectors as one-row matrices; the output layer is the embedding unless
  the model has its own (lm_head). RoPE's tables have a row for each of the first `places` places (at most the model's)."""
  j = json.load(open(os.path.join(folder, "config.json")))
  c = SimpleNamespace(dim=j["hidden_size"], heads=j["num_attention_heads"], kv=j["num_key_value_heads"], layers=j["num_hidden_layers"],
                      eps=j["rms_norm_eps"], theta=j["rope_theta"], places=min(j["max_position_embeddings"], places),
                      rope=j.get("rope_scaling"), vocab=j["vocab_size"], ffn=j["intermediate_size"])
  index = os.path.join(folder, "model.safetensors.index.json")
  files = sorted(set(json.load(open(index))["weight_map"].values())) if os.path.exists(index) else ["model.safetensors"]
  out = {}
  for file in files:
    header, data = gpt2.safetensors(os.path.join(folder, file))
    for name, t in header.items():
      if name == "__metadata__": continue
      name = name.removeprefix("model.").replace("self_attn.", "").replace("mlp.", "").removesuffix(".weight")
      out[name] = Weight(functools.partial(gpt2.tensor_bits, data, t, name.endswith("_proj") or name == "lm_head"))
  out["embed"] = out.pop("embed_tokens")
  out["embed.T"] = out.pop("lm_head", None) or Weight(lambda e=out["embed"]: np.ascontiguousarray(e.bits().T))  # tied, or its own
  for name, a in zip(("cos", "sin", "turn"), rotary(c, c.places)): out[name] = Weight(lambda a=a: gpt2.bf16_bits(a))
  return c, fuse(out)

def weight_format(name, prec):  # a weight's format on the chip: the matrices a matmul reads as its B (the blocks' and the
  # output layer) in the one prec (precision's) gives them, the rest, and all without prec, bf16
  key = ("head", None) if name == "embed.T" else (name.split(".")[2].removesuffix("_proj"), int(name.split(".")[1])) if name.endswith("_proj") else None
  return prec[key][0] if prec and key else "bf16"

def upload(rt, weights, prec=None):  # the weights as ferrite Tensors, one at a time, each cast on the chip to its format
  t = {}                                # and waited for, so the next weight's bf16 copy goes where this one's was: a model
  for name, w in weights.items():       # takes the DRAM its formats need and one matrix more, not all its bf16 besides
    bits = w.bits()
    x = t[name] = Tensor.frombytes(rt, *bits.shape, bits.tobytes())
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
    return v if f == "bf16" else np.concatenate([gpt2.bfp(v[i:i + 1024], f).astype(np.float32) for i in range(0, len(v), 1024)])
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
def letters_numbers():  # \p{L} and \p{N}, which Python's re lacks, as a character class's ranges from unicodedata
  runs = {"L": [], "N": []}
  for cp in range(0x110000):
    if (r := runs.get(unicodedata.category(chr(cp))[0])) is not None:
      if r and r[-1][1] == cp - 1: r[-1][1] = cp
      else: r.append([cp, cp])
  return {k: "".join(re.escape(chr(a)) + (f"-{re.escape(chr(b))}" if b > a else "") for a, b in r) for k, r in runs.items()}

def python_regex(pat):  # tokenizer.json's split for Python's re: \p{L}, \p{N} and \s spelled out (Python's \s also takes
  ranges, out, inside, i = letters_numbers() | {"s": WHITE}, [], False, 0  # in four controls), bare inside a class
  while i < len(pat):
    if pat.startswith(("\\p{", "\\s", "\\S"), i):
      key, i = (pat[i + 3], i + 5) if pat[i + 1] == "p" else (pat[i + 1], i + 2)
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
  are the ranks). Special tokens in the text are taken as text."""
  def __init__(self, folder):
    t = json.loads(open(os.path.join(folder, "tokenizer.json"), encoding="utf-8").read())
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))  # GPT-2's bytes as printable characters
    byte = dict(zip(map(chr, bs + [256 + i for i in range(256 - len(bs))]), bs + [b for b in range(256) if b not in bs]))
    special = {a["content"]: a["id"] for a in t["added_tokens"]}
    self.ranks = {bytes(byte[ch] for ch in tok): i for tok, i in t["model"]["vocab"].items() if tok not in special}
    self.bytes = {i: b for b, i in self.ranks.items()} | {i: s.encode() for s, i in special.items()}
    pre = t.get("pre_tokenizer") or {}  # the split before BPE: tokenizer.json's own (Llama 3's), else SmolLM's
    split = [p["pattern"]["Regex"] for p in pre.get("pretokenizers", [pre]) if p.get("type") == "Split" and "Regex" in p.get("pattern", {})]
    self.split, self.pieces = python_regex(split[0] if split else SPLIT), {}

  def bpe(self, piece):  # a piece's bytes as ids
    if piece not in self.pieces:
      parts = [piece] if piece in self.ranks else [piece[i:i + 1] for i in range(len(piece))]
      while len(parts) > 1 and (best := min((self.ranks.get(a + b, math.inf), i) for i, (a, b) in enumerate(zip(parts, parts[1:]))))[0] < math.inf:
        parts[best[1]:best[1] + 2] = [parts[best[1]] + parts[best[1] + 1]]
      self.pieces[piece] = [self.ranks[p] for p in parts]
    return self.pieces[piece]
  def encode(self, text): return [i for piece in self.split.findall(text) for i in self.bpe(piece.encode())]
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
