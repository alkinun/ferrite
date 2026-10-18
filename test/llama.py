# A Llama-family model on ferrite (SmolLM-135M: RMSNorm, RoPE, SwiGLU, grouped-query attention, a KV cache): the real
# weights, greedy generation, each step checked against the same model in numpy (the same bf16 weights, float32 math).
# The model is written once, for ferrite Tensors and numpy arrays alike, as test/gpt2.py is.
#   TTSIM=/path/to/libttsim.so python3 test/llama.py path/to/SmolLM-135M "prompt" [tokens] [bfp8]
# The folder holds Hugging Face's HuggingFaceTB/SmolLM-135M config.json, tokenizer.json and model.safetensors. bfp8:
# the weight matrices in bfp8 on the chip (numpy then has them as the chip does).
import json, math, os, struct, sys, time
from types import SimpleNamespace
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ferrite.tensor import Tensor
import gpt2

def fns(x):  # what numpy and ferrite spell differently
  if isinstance(x, np.ndarray):
    return SimpleNamespace(rows=lambda a, ids: a[list(ids)], split=lambda a, n: a.reshape(a.shape[-2], n, -1).transpose(1, 0, 2),
                           merge=lambda a: a.transpose(1, 0, 2).reshape(a.shape[1], -1), t=lambda a: a.swapaxes(-1, -2),
                           softmax=lambda a: (e := np.exp(a - a.max(-1, keepdims=True))) / e.sum(-1, keepdims=True),
                           cat=lambda a, b, axis: np.concatenate([a, b], axis), reshape=lambda a, *s: a.reshape(s),
                           mean=lambda a: a.mean(-1, keepdims=True), rsqrt=lambda a: 1 / np.sqrt(a), silu=lambda a: a / (1 + np.exp(-a)))
  return SimpleNamespace(rows=Tensor.gather, split=Tensor.split, merge=Tensor.merge, t=lambda a: a.T, softmax=Tensor.softmax,
                         cat=lambda a, b, axis: a.cat(b, axis=axis), reshape=Tensor.reshape, mean=lambda a: a.mean(2),
                         rsqrt=Tensor.rsqrt, silu=Tensor.silu)

def rms(x, w, eps): f = fns(x); return x * f.rsqrt(f.mean(x * x) + eps) * w

def rope(x, cos, sin):  # Hugging Face's form: x cos + (-x2, x1) sin, x1 and x2 a head's halves
  f, h = fns(x), x.shape[-1] // 2
  return x * cos + f.cat(-x[:, :, h:], x[:, :, :h], axis=2) * sin

def layer(x, p, c, cos, sin, mask, cache):
  """One block: attention then a SwiGLU MLP, each after an RMSNorm and added back. p: the block's weights (as x @ w, so
  the Hugging Face ones transposed), c: the config, cache: this block's [keys, values], grown by this call's."""
  f, n, hd = fns(x), x.shape[-2], c.dim // c.heads
  h = rms(x, p["input_layernorm"], c.eps)
  q, k, v = (f.split(h @ p[w], heads) for w, heads in (("q_proj", c.heads), ("k_proj", c.kv), ("v_proj", c.kv)))
  q, k = rope(q, cos, sin), rope(k, cos, sin)
  if cache[0] is not None: k, v = f.cat(cache[0], k, 1), f.cat(cache[1], v, 1)
  cache[:] = k, v
  group = c.heads // c.kv  # each key and value head serves this many query heads: their rows, one after another
  s = f.reshape(q, c.kv, group * n, hd) @ f.t(k) * (1 / math.sqrt(hd))
  y = f.reshape(f.softmax(s if mask is None else s + mask) @ v, c.heads, n, hd)
  x = x + f.merge(y) @ p["o_proj"]
  h = rms(x, p["post_attention_layernorm"], c.eps)
  return x + (f.silu(h @ p["gate_proj"]) * (h @ p["up_proj"])) @ p["down_proj"]

def rotary(c, start, n):  # cos and sin for positions start.., as Hugging Face's: angles in fp32, the tables in bf16
  inv = 1 / c.theta ** (np.arange(0, c.dim // c.heads, 2, dtype=np.float32) / (c.dim // c.heads))
  angles = np.arange(start, start + n, dtype=np.float32)[:, None] * inv[None]
  bf16 = lambda a: ((a.astype(np.float32).view(np.uint32) + 0x7FFF + (a.astype(np.float32).view(np.uint32) >> 16 & 1)) & 0xFFFF0000).view(np.float32)
  return [bf16(np.concatenate([g(angles)] * 2, 1)) for g in (np.cos, np.sin)]

def forward(ids, w, c, caches, rt=None):
  """The logits for the token after ids, which follow len(caches' keys) tokens already seen (their keys and values in
  caches, one [keys, values] a block, grown here)."""
  f, start = fns(w["embed"]), 0 if caches[0][0] is None else caches[0][0].shape[-2]
  n, group = len(ids), c.heads // c.kv
  causal = np.triu(np.full((n, start + n), -1e9, np.float32), start + 1)
  cos, sin = rotary(c, start, n)
  mask = np.tile(causal, (group, 1)) if n > 1 else None
  if rt: cos, sin, mask = (Tensor.of(rt, a.tolist()) if a is not None else None for a in (cos, sin, mask))
  x = f.rows(w["embed"], ids)
  for l in range(c.layers):
    x = layer(x, {k[len(f"layers.{l}."):]: v for k, v in w.items() if k.startswith(f"layers.{l}.")}, c, cos, sin, mask, caches[l])
  return f.rows(rms(x, w["norm"], c.eps), [n - 1]) @ w["embed.T"]

def load(folder):
  """The config, and {name: (bf16 bits, the same as float32)} from model.safetensors (8 bytes of header size, a JSON header
  naming each tensor's dtype, shape and byte range, then the data; fp32 rounded to bf16), named without "model.", linear
  weights transposed (so x @ w), vectors as one-row matrices."""
  j = json.load(open(os.path.join(folder, "config.json")))
  c = SimpleNamespace(dim=j["hidden_size"], heads=j["num_attention_heads"], kv=j["num_key_value_heads"], layers=j["num_hidden_layers"],
                      eps=j["rms_norm_eps"], theta=j["rope_theta"])
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
  return c, out

def upload(rt, weights, dtype="bf16"):  # the weights as ferrite Tensors, the matrices a matmul reads as its B (the
  # blocks' and the output layer) cast to dtype on the chip
  t = {name: Tensor.frombytes(rt, *bits.shape, bits.tobytes()) for name, (bits, _) in weights.items()}
  return t | {k: x.cast(dtype).realize() for k, x in t.items() if k == "embed.T" or k.startswith("layers.") and x.rows > 1}

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
  caches, np_caches, new = [[None, None] for _ in range(c.layers)], [[None, None] for _ in range(c.layers)], ids
  print(prompt, end="", flush=True)
  for _ in range(steps):  # the prompt once, then a token at a time, each reading the keys and values cached so far
    start = time.time()
    got, want = logits(forward(new, t, c, caches, rt)), forward(new, w, c, np_caches)[0]
    agree += int(got.argmax()) == int(want.argmax())
    new = [int(want.argmax())]  # numpy's pick, so that one near tie does not send the two down different paths
    ids += new
    print(enc.decode(new) + ("" if got.argmax() == want.argmax() else f"[ferrite: {enc.decode([int(got.argmax())])!r}]"), end="", flush=True)
    print(f" ({time.time() - start:.0f}s, logits within {np.abs(got - want).max():.2f})", end="", file=sys.stderr, flush=True)
  print(f"\nferrite picked the same token as numpy {agree} times out of {steps}")

if __name__ == "__main__": main()
