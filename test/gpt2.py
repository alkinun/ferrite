# GPT-2 small on ferrite: the real weights, greedy generation, each step checked against the same model in numpy (the
# same bf16 weights, float32 math). The model is written once, for ferrite Tensors and numpy arrays alike.
#   TTSIM=/path/to/libttsim.so python3 test/gpt2.py path/to/model.safetensors "prompt" [tokens]
# The weights are Hugging Face's openai-community/gpt2 model.safetensors; the tokenizer is tiktoken's "gpt2".
import json, math, os, struct, sys, time
from types import SimpleNamespace
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ferrite.tensor import Tensor

HEADS = 12

def fns(x):  # what numpy and ferrite spell differently
  if isinstance(x, np.ndarray):
    split = lambda a, n, axis=1: a.reshape(a.shape[0], n, -1).transpose(1, 0, 2) if axis else a.reshape(n, -1, a.shape[1])
    softmax = lambda a: (e := np.exp(a - a.max(-1, keepdims=True))) / e.sum(-1, keepdims=True)
    layernorm = lambda a: (a - a.mean(-1, keepdims=True)) / np.sqrt(a.var(-1, keepdims=True) + 1e-5)
    return SimpleNamespace(rows=lambda a, ids: a[list(ids)], width=lambda a: a.shape[-1], split=split, softmax=softmax, layernorm=layernorm,
                           merge=lambda a: a.transpose(1, 0, 2).reshape(a.shape[1], -1), t=lambda a: a.swapaxes(-1, -2),
                           gelu=lambda a: 0.5 * a * (1 + np.tanh(math.sqrt(2 / math.pi) * (a + 0.044715 * a**3))))
  if type(x).__module__ == "torch":  # test/train.py's reference
    import torch
    F = torch.nn.functional
    return SimpleNamespace(rows=lambda a, ids: a[list(ids)], width=lambda a: a.shape[-1], softmax=lambda a: a.softmax(-1),
                           split=lambda a, n: a.reshape(a.shape[0], n, -1).transpose(0, 1), merge=lambda a: a.transpose(0, 1).reshape(a.shape[1], -1),
                           layernorm=lambda a: F.layer_norm(a, a.shape[-1:]), t=lambda a: a.transpose(-1, -2),
                           gelu=lambda a: F.gelu(a, approximate="tanh"))
  return SimpleNamespace(rows=Tensor.gather, width=lambda a: a.cols, split=Tensor.split, softmax=Tensor.softmax,
                         layernorm=Tensor.layernorm, merge=Tensor.merge, t=lambda a: a.T, gelu=Tensor.gelu)

def layernorm(x, g, b): return fns(x).layernorm(x) * g + b

def layer(x, p, mask, heads=HEADS):
  """One transformer block: attention then MLP, each after a layer norm and added back. p: the block's weights by their
  names in the file, less "h.<layer>."; mask: 0 on and below the diagonal, -1e9 above."""
  f = fns(x)
  d = f.width(x)
  hd = d // heads
  qkv = layernorm(x, p["ln_1.weight"], p["ln_1.bias"]) @ p["attn.c_attn.weight"] + p["attn.c_attn.bias"]
  q, k, v = (f.split(qkv[:, i * d:(i + 1) * d], heads) for i in range(3))  # the heads as a batch
  s = q @ f.t(k) * (1 / math.sqrt(hd)) + mask
  y = f.merge(f.softmax(s) @ v) @ p["attn.c_proj.weight"]  # the heads side by side, then c_proj
  x = x + (y + p["attn.c_proj.bias"])  # x last: it is the large one, so it is rounded once
  h = layernorm(x, p["ln_2.weight"], p["ln_2.bias"]) @ p["mlp.c_fc.weight"] + p["mlp.c_fc.bias"]
  return x + (f.gelu(h) @ p["mlp.c_proj.weight"] + p["mlp.c_proj.bias"])

def forward(ids, w, mask):
  """The logits for the token after ids: token and position embeddings, the blocks, a last layer norm, and the output
  layer, which is the token embedding again (w["wte.T"], its transpose)."""
  f, layers = fns(w["wte.weight"]), 1 + max(int(k.split(".")[1]) for k in w if k.startswith("h."))
  x = f.rows(w["wte.weight"], ids) + f.rows(w["wpe.weight"], range(len(ids)))
  for l in range(layers): x = layer(x, {k[len(f"h.{l}."):]: v for k, v in w.items() if k.startswith(f"h.{l}.")}, mask)
  return f.rows(layernorm(x, w["ln_f.weight"], w["ln_f.bias"]), [len(ids) - 1]) @ w["wte.T"]

def load(path):
  """{name: (bf16 bits, the same as float32)} from a safetensors file: 8 bytes of header size, a JSON header naming each
  tensor's dtype, shape and byte range, then the data. Vectors become one-row matrices."""
  with open(path, "rb") as f:
    size = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(size))
  data, out = np.memmap(path, np.uint8, "r", offset=8 + size), {}
  for name, t in header.items():
    if name == "__metadata__" or name.endswith(".attn.bias") or name.endswith(".attn.masked_bias"): continue  # causal masks
    assert t["dtype"] == "F32"
    bits = data[t["data_offsets"][0]:t["data_offsets"][1]].view(np.uint32).reshape(t["shape"] if len(t["shape"]) == 2 else (1, -1))
    bits = ((bits + 0x7FFF + (bits >> 16 & 1)) >> 16).astype(np.uint16)  # to bf16, nearest even
    out[name] = (bits, (bits.astype(np.uint32) << 16).view(np.float32))
  return out

def upload(rt, weights):  # the weights as ferrite Tensors, the output layer's transposed on the chip
  t = {name: Tensor.frombytes(rt, *bits.shape, bits.tobytes()) for name, (bits, _) in weights.items()}
  t["wte.T"] = t["wte.weight"].T.realize()  # once, rather than read transposed every token
  return t

def arrays(weights):  # the same as numpy arrays
  return {name: values for name, (_, values) in weights.items()} | {"wte.T": weights["wte.weight"][1].T}

def mask(n): return np.triu(np.full((n, n), -1e9, np.float32), 1)

def logits(ids, w, rt=None):
  """The logits for the token after ids, as a numpy vector: on the chip if rt is given (w: from upload), else in numpy."""
  if rt is None: return forward(ids, w, mask(len(ids)))[0]
  bits = np.frombuffer(forward(ids, w, Tensor.of(rt, mask(len(ids)).tolist())).tobytes(), np.uint16)
  return (bits.astype(np.uint32) << 16).view(np.float32)

def main():
  import tiktoken
  from ferrite import pci
  from ferrite.chip import Chip
  from ferrite.runtime import Runtime
  path, prompt, steps = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "The meaning of life is", int(sys.argv[3]) if len(sys.argv) > 3 else 8
  enc, chip = tiktoken.get_encoding("gpt2"), Chip(pci.open_device())
  rt = Runtime(chip, chip.tensix[0], chip.tensix[-1])
  start = time.time()
  weights = load(path)
  w, t = arrays(weights), upload(rt, weights)
  print(f"{sum(bits.size for bits, _ in weights.values()) / 1e6:.0f}M weights on the chip in {time.time() - start:.0f}s")
  ids, agree = enc.encode(prompt), 0
  print(prompt, end="", flush=True)
  for _ in range(steps):
    start = time.time()
    got, want = logits(ids, t, rt), logits(ids, w)
    agree += int(got.argmax()) == int(want.argmax())
    ids.append(int(want.argmax()))  # numpy's pick, so that one near tie does not send the two down different paths
    print(enc.decode([ids[-1]]) + ("" if got.argmax() == want.argmax() else f"[ferrite: {enc.decode([int(got.argmax())])!r}]"), end="", flush=True)
    print(f" ({time.time() - start:.0f}s, logits within {np.abs(got - want).max():.2f})", end="", file=sys.stderr, flush=True)
  print(f"\nferrite picked the same token as numpy {agree} times out of {steps}")

if __name__ == "__main__": main()
