# A Llama-family model on ferrite (SmolLM-135M: RMSNorm, RoPE, SwiGLU, grouped-query attention, a KV cache): the real
# weights, greedy generation, each step checked against the same model in numpy (the same bf16 weights, float32 math).
# The model is written once, for ferrite Tensors and numpy arrays alike, as test/gpt2.py is. Its KV cache has a fixed
# size and a token's place is data, not a shape, so on ferrite every decode step launches the same kernels: after the
# first two, each is a replay (tensor.jit). Several sequences can decode together, their caches side by side (forward's
# starts), a token each a step.
#   TTSIM=/path/to/libttsim.so python3 test/llama.py path/to/SmolLM-135M "prompt" [tokens] [bfp8|bfp4]
# The folder holds Hugging Face's HuggingFaceTB/SmolLM-135M config.json, tokenizer.json and model.safetensors (or another
# Llama's, SmolLM2-360M's say). bfp8: the weight matrices in bfp8 on the chip, bfp4: the MLP's gate and up in bfp4 and
# the rest in bfp8 (numpy then has them as the chip does).
import json, math, os, struct, sys, time
from types import SimpleNamespace
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ferrite.tensor import Tensor, jit
import gpt2

def fns(x):  # what numpy and ferrite spell differently
  if isinstance(x, np.ndarray):
    return SimpleNamespace(rows=lambda a, ids: a[list(ids)], split=lambda a, n: a.reshape(a.shape[-2], n, -1).transpose(1, 0, 2),
                           merge=lambda a: a.transpose(1, 0, 2).reshape(a.shape[1], -1), t=lambda a: a.swapaxes(-1, -2),
                           softmax=lambda a: (e := np.exp(a - a.max(-1, keepdims=True))) / e.sum(-1, keepdims=True),
                           mean=lambda a: a.mean(-1, keepdims=True), rsqrt=lambda a: 1 / np.sqrt(a), silu=lambda a: a / (1 + np.exp(-a)),
                           put=lambda a, places, b: a.__setitem__((slice(None), list(places)), b), repeat=lambda a, n: np.repeat(a, n, 0))
  return SimpleNamespace(rows=Tensor.gather, split=Tensor.split, merge=Tensor.merge, t=lambda a: a.T, softmax=Tensor.softmax,
                         mean=lambda a: a.mean(2),
                         rsqrt=Tensor.rsqrt, silu=Tensor.silu, put=Tensor.put, repeat=Tensor.repeat_interleave)

def rms(x, w, eps): f = fns(x); return x * f.rsqrt(f.mean(x * x) + eps) * w

def rope(x, cos, sin, turn):  # Hugging Face's form: x cos + (-x2, x1) sin, x1 and x2 a head's halves, (-x2, x1) as x @ turn
  return x * cos + (x @ turn) * sin

def layer(x, p, c, rotary, mask, cache, rows):
  """One block: attention then a SwiGLU MLP, each after an RMSNorm and added back. p: the block's weights (as x @ w, so
  the Hugging Face ones transposed, q, k and v's side by side, and gate and up's), c: the config, rotary: (cos, sin,
  turn), the tables' rows for this call's places; cache: this block's keys' heads then its values', each with a row for
  every place, this call's written at its rows; mask: -1e9 where a token must not look (gpt2.mask)."""
  f, n = fns(x), c.heads + 2 * c.kv
  h = rms(x, p["input_layernorm"], c.eps)
  qkv = rope(f.split(h @ p["qkv_proj"], n), f.split(rotary[0], n), f.split(rotary[1], n), rotary[2])  # v's heads do not turn
  f.put(cache, rows, qkv[c.heads:, :, :])  # this call's keys and values, at once
  k, v = (f.repeat(t, c.heads // c.kv) for t in (cache[:c.kv, :, :], cache[c.kv:, :, :]))  # a head for each query's, in place
  s = qkv[:c.heads, :, :] @ f.t(k) * (1 / math.sqrt(c.dim // c.heads))
  x = x + f.merge(f.softmax(s + mask) @ v) @ p["o_proj"]
  h = rms(x, p["post_attention_layernorm"], c.eps)
  gate_up, m = h @ p["gate_up_proj"], p["down_proj"].shape[-2]
  return x + (f.silu(gate_up[:, :m]) * gate_up[:, m:]) @ p["down_proj"]

def rotary(c, n):  # cos and sin for places 0 to n - 1, as Hugging Face's (angles in fp32, the tables in bf16), a head's
  # for each of q's, k's and v's heads (v's 1 and 0: they do not turn, so one kernel turns them all), and turn:
  hd = c.dim // c.heads  # x @ turn is (-x2, x1), each of its entries one of x's, so exact
  inv = 1 / c.theta ** (np.arange(0, hd, 2, dtype=np.float32) / hd)
  angles = np.arange(n, dtype=np.float32)[:, None] * inv[None]
  bf16 = lambda a: ((a.astype(np.float32).view(np.uint32) + 0x7FFF + (a.astype(np.float32).view(np.uint32) >> 16 & 1)) & 0xFFFF0000).view(np.float32)
  turn = np.zeros((hd, hd), np.float32)
  turn[np.arange(hd // 2) + hd // 2, np.arange(hd // 2)], turn[np.arange(hd // 2), np.arange(hd // 2) + hd // 2] = -1, 1
  heads = lambda t, v: np.concatenate([t] * (c.heads + c.kv) + [np.full_like(t, v)] * c.kv, 1)
  return [bf16(heads(np.concatenate([g(angles)] * 2, 1), v)) for g, v in ((np.cos, 1), (np.sin, 0))] + [turn]

def fuse(w):  # each block's q, k and v projections as one matrix, and its gate and up: fewer and wider matmuls
  join = lambda *xs: tuple(map(join, *xs)) if isinstance(xs[0], tuple) else np.ascontiguousarray(np.concatenate(xs, 1))
  for pre in {k[:k.index(".", 7) + 1] for k in w if k.startswith("layers.")}:
    w[pre + "qkv_proj"] = join(*(w.pop(pre + n) for n in ("q_proj", "k_proj", "v_proj")))
    w[pre + "gate_up_proj"] = join(w.pop(pre + "gate_proj"), w.pop(pre + "up_proj"))
  return w

def forward(ids, places, w, c, caches, starts=None):
  """The logits for the token after ids, the tokens at places in the sequence: their keys and values go into caches (a
  block's keys' heads then its values', each with a row for every place), and each sees the places up to its own. ids and places are
  rows: lists for numpy, int32 tensors for ferrite. starts: several sequences decoding together (a batch of users), their
  caches side by side, each token's sequence starting at its start's row and each token seeing only its own sequence;
  then the logits are every token's, the next token of each (rows: numpy arrays for numpy)."""
  f = fns(w["embed"])
  x, rows = f.rows(w["embed"], ids), places if starts is None else places + starts
  rotary, mask = (f.rows(w["cos"], places), f.rows(w["sin"], places), w["turn"]), gpt2.mask(rows, caches[0].shape[-2], starts)
  for l in range(c.layers):
    x = layer(x, {k[len(f"layers.{l}."):]: v for k, v in w.items() if k.startswith(f"layers.{l}.")}, c, rotary, mask, caches[l], rows)
  h = rms(x, w["norm"], c.eps)
  return (h if starts is not None else f.rows(h, [x.shape[-2] - 1])) @ w["embed.T"]

def kv_cache(c, size, zeros):  # each block's keys' heads then its values', a row for each of size places, made by zeros(shape)
  return [zeros((2 * c.kv, size, c.dim // c.heads)) for _ in range(c.layers)]

def load(folder):
  """The config, and {name: (bf16 bits, the same as float32)} from model.safetensors (8 bytes of header size, a JSON header
  naming each tensor's dtype, shape and byte range, then the data; fp32 rounded to bf16), named without "model.", linear
  weights transposed (so x @ w), vectors as one-row matrices."""
  j = json.load(open(os.path.join(folder, "config.json")))
  c = SimpleNamespace(dim=j["hidden_size"], heads=j["num_attention_heads"], kv=j["num_key_value_heads"], layers=j["num_hidden_layers"],
                      eps=j["rms_norm_eps"], theta=j["rope_theta"], places=j["max_position_embeddings"])
  path = os.path.join(folder, "model.safetensors")
  with open(path, "rb") as fh:
    size = struct.unpack("<Q", fh.read(8))[0]
    header = json.loads(fh.read(size))
  data, out = np.memmap(path, np.uint8, "r", offset=8 + size), {}
  for name, t in header.items():
    if name == "__metadata__": continue
    raw = data[t["data_offsets"][0]:t["data_offsets"][1]]
    if t["dtype"] == "F32": bits = raw.view(np.uint32); bits = ((bits + 0x7FFF + (bits >> 16 & 1)) >> 16).astype(np.uint16)  # to bf16, nearest even
    else: assert t["dtype"] == "BF16"; bits = raw.view(np.uint16)
    bits = bits.reshape(t["shape"] if len(t["shape"]) == 2 else (1, -1))
    name = name.removeprefix("model.").replace("self_attn.", "").replace("mlp.", "").removesuffix(".weight")
    bits = np.ascontiguousarray(bits.T if name.endswith("_proj") else bits)
    out[name] = (bits, (bits.astype(np.uint32) << 16).view(np.float32))
  out["embed"] = out.pop("embed_tokens")
  out["embed.T"] = tuple(np.ascontiguousarray(a.T) for a in out["embed"])  # tied: the output layer is the embedding
  for name, a in zip(("cos", "sin", "turn"), rotary(c, c.places)): out[name] = ((a.view(np.uint32) >> 16).astype(np.uint16), a)
  return c, fuse(out)

def upload(rt, weights, dtype="bf16"):  # the weights as ferrite Tensors, the matrices a matmul reads as its B (the
  # blocks' and the output layer) cast to dtype on the chip; bfp4: the MLP's gate and up in bfp4 and the rest in bfp8, as
  # TT's own LLM code keeps them in its performance mode
  t = {name: Tensor.frombytes(rt, *bits.shape, bits.tobytes()) for name, (bits, _) in weights.items()}
  cast = lambda k: dtype if dtype != "bfp4" else "bfp4" if k.endswith("gate_up_proj") else "bfp8"
  return t | {k: x.cast(cast(k)).realize() for k, x in t.items() if k == "embed.T" or k.startswith("layers.") and x.rows > 1}

def arrays(weights, t=None):  # the same as numpy arrays; given them on the chip (upload), those not in bf16 as the chip has them
  return {name: values for name, (_, values) in weights.items()} | {k: gpt2.values(x) for k, x in (t or {}).items() if x.dtype != "bf16"}

def logits(f_out): return gpt2.values(f_out)[0]  # a ferrite row of logits, as a numpy vector

def tokenizer(folder):
  """tiktoken over tokenizer.json's byte-level BPE (ids in merge order, so they are the ranks), digits one at a time."""
  import tiktoken
  t = json.load(open(os.path.join(folder, "tokenizer.json")))
  bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))  # GPT-2's bytes as printable characters
  cs = bs + [256 + i for i in range(256 - len(bs))]
  byte = dict(zip(map(chr, cs), bs + [b for b in range(256) if b not in bs]))
  special = {a["content"]: a["id"] for a in t["added_tokens"]}
  ranks = {bytes(byte[ch] for ch in tok): i for tok, i in t["model"]["vocab"].items() if tok not in special}
  pat = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
  return tiktoken.Encoding("smollm", pat_str=pat, mergeable_ranks=ranks, special_tokens=special)

def main():
  from ferrite import pci
  from ferrite.chip import Chip
  from ferrite.runtime import Runtime
  folder, prompt, steps = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "The meaning of life is", int(sys.argv[3]) if len(sys.argv) > 3 else 8
  dtype = sys.argv[4] if len(sys.argv) > 4 else "bf16"
  enc, chip = tokenizer(folder), Chip(pci.open_device())
  rt = Runtime(chip, chip.tensix[0], chip.tensix[-1])
  start = time.time()
  c, weights = load(folder)
  t = upload(rt, weights, dtype)
  w = arrays(weights, t)
  print(f"{sum(bits.size for k, (bits, _) in weights.items() if k != 'embed.T') / 1e6:.0f}M weights on the chip in {time.time() - start:.0f}s")
  ids, agree = enc.encode(prompt), 0
  size = -(-(len(ids) + steps) // 32) * 32  # a row for each place the generation reaches, in whole tiles
  kv, np_kv = kv_cache(c, size, lambda shape: Tensor.of(rt, np.zeros(shape).tolist())), kv_cache(c, size, np.zeros)
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
