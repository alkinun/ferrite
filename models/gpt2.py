# GPT-2 small on ferrite: the real weights, greedy generation, each step checked against the same model in numpy (the
# same bf16 weights, float32 math). The model is written once, for ferrite Tensors and numpy arrays alike. Its KV cache
# has a fixed size and a token's place is data, not a shape, so every decode step launches the same kernels: after the
# first two, each is a replay (tensor.jit).
#   TTSIM=/path/to/libttsim.so python3 models/gpt2.py path/to/model.safetensors "prompt" [tokens] [bfp8]
# The weights are Hugging Face's openai-community/gpt2 model.safetensors; the tokenizer is tiktoken's "gpt2". bfp8: the
# weight matrices in bfp8 on the chip (numpy then has them as the chip does).
import functools, json, math, os, struct, sys, time
from types import SimpleNamespace
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ferrite.tensor import Tensor, jit

HEADS = 12

def bf16_bits(a, packer=False):  # numpy values as bf16 bits: rounded to nearest even as uploads round, or half up as the packer does
  b = np.asarray(a, np.float32).view(np.uint32)
  return ((b + (0x8000 if packer else 0x7FFF + (b >> 16 & 1))) >> 16).astype(np.uint16)
def bf16_floats(bits): return (np.asarray(bits).astype(np.uint32) << 16).view(np.float32)  # bf16 bits as float32 values

def bfp(a, dtype="bfp8"):
  """numpy's model of a block float as the chip makes it (cast, or put into a bfp8 KV cache), as values: rounded to
  bf16, then each run of 16 along a row to its largest exponent, 7 bits of magnitude below it rounded half up (bfp4: 3
  bits, to the nearest)."""
  bits = bf16_bits(a).astype(np.uint32)
  s, e, m = bits >> 15, bits >> 7 & 255, (bits & 127) | np.where(bits >> 7 & 255, 128, 0)
  runs = lambda x: x.reshape(*x.shape[:-1], -1, 16)
  top = runs(e).max(-1, keepdims=True)
  shift = np.minimum(top - runs(e), 8)  # 8 and more leave 0
  q = np.minimum((runs(m) + (1 << shift)) >> (shift + 1), 127)
  if dtype == "bfp4": q = np.where(shift < 4, np.minimum((runs(m) + (16 << shift)) >> (shift + 5), 7), 0) << 4  # nearest 3 bits
  return (np.where(runs(s), -1.0, 1.0) * q * 2.0 ** (top.astype(np.int64) - 133)).reshape(np.shape(a))

def safetensors(path):  # a safetensors file: its JSON header (each tensor's dtype, shape and byte range) and its data, mapped
  with open(path, "rb") as f:
    size = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(size))
  return header, np.memmap(path, np.uint8, "r", offset=8 + size)

def u8(data, t): assert t["dtype"] == "U8"; return data[t["data_offsets"][0]:t["data_offsets"][1]].reshape(t["shape"])  # bytes (MXFP4's) as they are

def tensor_bits(data, t, transpose=False):  # one tensor of a safetensors file as bf16 bits (fp32 rounded to nearest even), a vector as one row
  raw = data[t["data_offsets"][0]:t["data_offsets"][1]]
  if t["dtype"] == "F32": bits = bf16_bits(raw.view(np.float32))
  else: assert t["dtype"] == "BF16", t["dtype"]; bits = raw.view(np.uint16)
  bits = bits.reshape(t["shape"] if len(t["shape"]) >= 2 else (1, -1))  # (a batch of matrices: gpt-oss's experts)
  return np.ascontiguousarray(bits.T) if transpose else bits

def fns(x):  # what numpy and ferrite spell differently
  if isinstance(x, np.ndarray):
    split = lambda a, n: a.reshape(a.shape[0], n, -1).transpose(1, 0, 2)
    softmax = lambda a: (e := np.exp(a - a.max(-1, keepdims=True))) / e.sum(-1, keepdims=True)
    layernorm = lambda a: (a - a.mean(-1, keepdims=True)) / np.sqrt(a.var(-1, keepdims=True) + 1e-5)
    return SimpleNamespace(rows=lambda a, ids: a[list(ids)], split=split, softmax=softmax, layernorm=layernorm,
                           merge=lambda a: a.transpose(1, 0, 2).reshape(a.shape[1], -1), t=lambda a: a.swapaxes(-1, -2),
                           gelu=lambda a: 0.5 * a * (1 + np.tanh(math.sqrt(2 / math.pi) * (a + 0.044715 * a**3))),
                           put=lambda a, places, b: a.__setitem__((slice(None), list(places)), b))
  if type(x).__module__ == "torch":  # test/train.py's reference
    import torch
    F = torch.nn.functional
    return SimpleNamespace(rows=lambda a, ids: a[list(ids)], softmax=lambda a: a.softmax(-1),
                           split=lambda a, n: a.reshape(a.shape[0], n, -1).transpose(0, 1), merge=lambda a: a.transpose(0, 1).reshape(a.shape[1], -1),
                           layernorm=lambda a: F.layer_norm(a, a.shape[-1:]), t=lambda a: a.transpose(-1, -2),
                           gelu=lambda a: F.gelu(a, approximate="tanh"))
  return SimpleNamespace(rows=Tensor.gather, split=Tensor.split, softmax=Tensor.softmax,
                         layernorm=Tensor.layernorm, merge=Tensor.merge, t=lambda a: a.T, gelu=Tensor.gelu, put=Tensor.put)

def layernorm(x, g, b): return fns(x).layernorm(x) * g + b

def block(w, pre): return {k[len(pre):]: w[k] for k in w if k.startswith(pre)}  # the weights named pre..., by the rest of their names

def layer(x, p, mask, heads=HEADS, cache=None, places=None):
  """One transformer block: attention then MLP, each after a layer norm and added back. p: the block's weights by their
  names in the file, less "h.<layer>."; mask: 0 where a token may look, -1e9 where it may not (see mask); cache: this
  block's keys and values (kv_cache: the keys' heads, then the values'), a row for every place: this call's are put at
  places, both in one put, and attention reads all of them."""
  f, d = fns(x), x.shape[-1]
  hd = d // heads
  qkv = layernorm(x, p["ln_1.weight"], p["ln_1.bias"]) @ p["attn.c_attn.weight"] + p["attn.c_attn.bias"]
  q, k, v = (f.split(qkv[:, i * d:(i + 1) * d], heads) for i in range(3))  # the heads as a batch
  if cache is not None:
    f.put(cache, places, f.split(qkv[:, d:], 2 * heads))
    k, v = cache[:heads, :, :], cache[heads:, :, :]
  s = q @ f.t(k) * (1 / math.sqrt(hd)) + mask
  y = f.merge(f.softmax(s) @ v) @ p["attn.c_proj.weight"]  # the heads side by side, then c_proj
  x = x + (y + p["attn.c_proj.bias"])  # x last: it is the large one, so it is rounded once
  h = layernorm(x, p["ln_2.weight"], p["ln_2.bias"]) @ p["mlp.c_fc.weight"] + p["mlp.c_fc.bias"]
  return x + (f.gelu(h) @ p["mlp.c_proj.weight"] + p["mlp.c_proj.bias"])

def forward(ids, w, mask, places=None, caches=None):
  """The logits for the token after ids: token and position embeddings, the blocks, a last layer norm, and the output
  layer, which is the token embedding again (w["wte.T"], its transpose). places: where ids sit in the sequence (0 on
  unless given), and caches each block's keys and values (kv_cache), for a token at a time: rows, lists for numpy and
  int32 tensors for ferrite, so a step's places are data and every step launches the same kernels (jit)."""
  f, layers = fns(w["wte.weight"]), 1 + max(int(k.split(".")[1]) for k in w if k.startswith("h."))
  x = f.rows(w["wte.weight"], ids) + f.rows(w["wpe.weight"], range(len(ids)) if places is None else places)
  for l in range(layers):
    x = layer(x, block(w, f"h.{l}."), mask, HEADS, caches and caches[l], places)
  return f.rows(layernorm(x, w["ln_f.weight"], w["ln_f.bias"]), [x.shape[-2] - 1]) @ w["wte.T"]

def mask(places, size, window=None):
  """-1e9 where a token at places, among size places, would look past its own (or, given a window, as far back as
  window places or more), else 0: in numpy, or on the chip from an int32 row of places, so a decode step's mask is
  data, not a shape."""
  return masked(np.asarray(places)[:, None] if not isinstance(places, Tensor) else places.reshape(1, places.cols, 1), size, window)

def masked(column, size, window=None):  # mask's, from a column of places: a batch of them, laid out as attention's scores' rows
  if not isinstance(column, Tensor):
    k = np.arange(size)
    return np.where((k > column) | ((k <= column - window) if window else False), np.float32(-1e9), np.float32(0))
  cols = columns(column.rt, size)
  out = cols > column
  if window: out = out + (cols < column - (window - 1))
  return (out * -1e9).cast("bf16")

@functools.cache
def columns(rt, size): return Tensor.frombytes(rt, 1, size, np.arange(size, dtype=np.int32).tobytes(), 1, "int32")  # a mask's keys, 0 on: once

def kv_cache(layers, size, zeros): return [zeros((2 * HEADS, size, 64)) for _ in range(layers)]  # each block's, by zeros(shape)

def load(path):
  """{name: (bf16 bits, the same as float32)} from a safetensors file (8 bytes of header size, a JSON header naming each
  tensor's dtype, shape and byte range, then the data), vectors as one-row matrices."""
  header, data = safetensors(path)
  bits = {name: tensor_bits(data, t) for name, t in header.items()  # but the attention's causal masks
          if name != "__metadata__" and not name.endswith((".attn.bias", ".attn.masked_bias"))}
  return {name: (b, bf16_floats(b)) for name, b in bits.items()}

def upload(rt, weights, dtype="bf16"):  # the weights as ferrite Tensors, the output layer's transposed on the chip, and
  # the matrices a matmul reads as its B (the blocks' and the output layer) cast to dtype there
  t = {name: Tensor.frombytes(rt, *bits.shape, bits.tobytes()) for name, (bits, _) in weights.items()}
  t["wte.T"] = t["wte.weight"].T.realize()  # once, rather than read transposed every token
  return t | {k: x.cast(dtype).realize() for k, x in t.items() if k == "wte.T" or k.startswith("h.") and x.rows > 1}

def arrays(weights, t=None):  # the same as numpy arrays; given them on the chip (upload), those not in bf16 as the chip has them
  w = {name: values for name, (_, values) in weights.items()} | {"wte.T": weights["wte.weight"][1].T}
  return w | {k: values(x) for k, x in (t or {}).items() if x.dtype != "bf16"}

def values(x):  # a ferrite matrix as numpy: fp32 and int32 as they are, bf16 (and a block float, as bf16) as float32
  raw, kind = x.tobytes(), {"fp32": np.float32, "int32": np.int32}.get(x.dtype)
  return (np.frombuffer(raw, kind) if kind else bf16_floats(np.frombuffer(raw, np.uint16))).reshape(x.rows, x.cols)

def logits(ids, w):  # the logits for the token after ids, in numpy, as a vector
  return forward(ids, w, mask(range(len(ids)), len(ids)))[0]

def main():
  import tiktoken
  from ferrite import pci
  from ferrite.chip import Chip
  from ferrite.runtime import Runtime
  path, prompt, steps = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "The meaning of life is", int(sys.argv[3]) if len(sys.argv) > 3 else 8
  dtype = sys.argv[4] if len(sys.argv) > 4 else "bf16"
  enc, chip = tiktoken.get_encoding("gpt2"), Chip(pci.open_device())
  rt = Runtime(chip)
  start = time.time()
  weights = load(path)
  t = upload(rt, weights, dtype)
  w = arrays(weights, t)
  print(f"{sum(bits.size for bits, _ in weights.values()) / 1e6:.0f}M weights on the chip in {time.time() - start:.0f}s")
  ids, agree, layers = enc.encode(prompt), 0, 1 + max(int(k.split(".")[1]) for k in w if k.startswith("h."))
  size = -(-(len(ids) + steps) // 32) * 32  # a row for each place the generation reaches, in whole tiles
  kv, np_kv = kv_cache(layers, size, lambda shape: Tensor.of(rt, np.zeros(shape).tolist())), kv_cache(layers, size, np.zeros)
  on = lambda row: Tensor.of(rt, [row], "int32")  # a row of ids or places, on the chip
  run = lambda ids, places, w, kv: forward(ids, w, mask(places, size), places, kv)
  step, new = jit(lambda ids, places: run(ids, places, t, kv)), ids  # a token at a time, replayed from the third
  print(prompt, end="", flush=True)
  for i in range(steps):  # the prompt at once, then a token at a time, each seeing the keys and values cached so far
    start, since, places = time.time(), rt.launches, list(range(len(ids) - len(new), len(ids)))
    got = values(run(on(new), on(places), t, kv) if i == 0 else step(on(new), on(places)))[0]
    want = run(new, places, w, np_kv)[0]
    agree += int(got.argmax()) == int(want.argmax())
    new = [int(want.argmax())]  # numpy's pick, so that one near tie does not send the two down different paths
    ids += new
    print(enc.decode(new) + ("" if got.argmax() == want.argmax() else f"[ferrite: {enc.decode([int(got.argmax())])!r}]"), end="", flush=True)
    print(f" ({time.time() - start:.0f}s, logits within {np.abs(got - want).max():.2f})", end="", file=sys.stderr, flush=True)
  print(f"\nferrite picked the same token as numpy {agree} times out of {steps}")
  print("the last token's kernels, by cycles on the chip's clock (Runtime.profile):")
  for name, n, cycles, _ in rt.profile(since)[:8]: print(f"{cycles:9d} {n:4d}  {name[:90]}")

if __name__ == "__main__": main()
