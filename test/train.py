# A small GPT trained on ferrite: GPT-2's blocks (test/gpt2.py), tiny, in fp32, learning ferrite's own README and
# LESSONS a character at a time with Adam, checked against PyTorch two ways: every step's loss and gradients against
# PyTorch's on ferrite's own weights (each step is right), and the loss curve against PyTorch training the same model on
# the same text from the same weights by itself (the two runs drift apart, as any two fp32 runs do: see train()).
#   TTSIM=/path/to/libttsim.so python3 test/train.py [steps]
import math, os, sys, time
from types import SimpleNamespace
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ferrite.tensor import Tensor
import gpt2

def text(): return "".join(open(os.path.join(os.path.dirname(__file__), "..", name)).read() for name in ("README.md", "LESSONS.md"))

def init(c, seed=0):
  """GPT-2's weights by its names, as float32 numpy: N(0, 0.02), the projections back into the residual scaled down by
  the depth, layer norms 1 and biases 0."""
  rng, d, w = np.random.default_rng(seed), c.dim, {}
  normal = lambda *shape, std=0.02: rng.normal(0, std, shape).astype(np.float32)
  w["wte.weight"], w["wpe.weight"] = normal(c.vocab, d), normal(c.ctx, d, std=0.01)
  for l in range(c.layers):
    for name, (rows, cols) in {"attn.c_attn": (d, 3 * d), "attn.c_proj": (d, d), "mlp.c_fc": (d, 4 * d), "mlp.c_proj": (4 * d, d)}.items():
      w[f"h.{l}.{name}.weight"] = normal(rows, cols, std=0.02 / math.sqrt(2 * c.layers) if name.endswith("c_proj") else 0.02)
      w[f"h.{l}.{name}.bias"] = np.zeros((1, cols), np.float32)
    for ln in ("ln_1", "ln_2"): w[f"h.{l}.{ln}.weight"], w[f"h.{l}.{ln}.bias"] = np.ones((1, d), np.float32), np.zeros((1, d), np.float32)
  w["ln_f.weight"], w["ln_f.bias"] = np.ones((1, d), np.float32), np.zeros((1, d), np.float32)
  return w

def forward(ids, w, mask, c):  # the logits at every position: gpt2.forward's, the output layer tied to wte
  f = gpt2.fns(w["wte.weight"])
  x = f.rows(w["wte.weight"], ids) + f.rows(w["wpe.weight"], range(len(ids)))
  for l in range(c.layers): x = gpt2.layer(x, {k[len(f"h.{l}."):]: v for k, v in w.items() if k.startswith(f"h.{l}.")}, mask, c.heads)
  return gpt2.layernorm(x, w["ln_f.weight"], w["ln_f.bias"]) @ f.t(w["wte.weight"])

def cross_entropy(logits, targets):  # the mean over positions of -log p(target), the targets picked out by a one-hot mask
  onehot = Tensor.of(logits.rt, np.eye(logits.cols, dtype=np.float32)[targets].tolist(), "fp32")
  return (logits.log_softmax() * onehot).sum() * (-1 / len(targets))

def adam(params, state, step, lr, b1=0.9, b2=0.999, eps=1e-8):  # PyTorch's Adam, written the same way
  for p in params:
    g, (m, v) = p.grad.realize(), state[id(p)]
    m.assign(m * b1 + g * (1 - b1))
    v.assign(v * b2 + g * g * (1 - b2))
    p.assign(p - m * (lr / (1 - b1**step)) / (v.sqrt() * (1 / math.sqrt(1 - b2**step)) + eps))
    p.grad = None

def train(rt, steps, c=None, lr=3e-3, log=None):
  """steps of training on ferrite, and PyTorch beside it. Each step, PyTorch takes ferrite's weights as they are and
  computes the same loss and gradients; and PyTorch also trains by itself, from the same first weights on the same text.
  The two runs part, though each step is right: Adam's first step moves every weight by lr times the sign of its
  gradient, so a weight whose gradient is below the matmul's 2^-11 moves either way, and the runs go on from there.
  Returns, each step: ferrite's loss, PyTorch's on the same weights, the gradients' worst error (relative to each
  weight's largest), and PyTorch's own run's loss."""
  import torch
  data = text()
  chars = sorted(set(data))
  c = c or SimpleNamespace(ctx=64, dim=64, heads=2, layers=2)
  c.vocab, ids, rng = len(chars), np.array([chars.index(ch) for ch in data]), np.random.default_rng(1)
  w = init(c)
  mask = np.triu(np.full((c.ctx, c.ctx), -1e9, np.float32), 1)
  tw = {k: Tensor.of(rt, a.tolist(), "fp32").requires_grad_() for k, a in w.items()}
  state = {id(t): [Tensor.full(rt, t.rows, t.cols, 0.0, "fp32") for _ in range(2)] for t in tw.values()}
  pw = {k: torch.tensor(a, requires_grad=True) for k, a in w.items()}
  opt = torch.optim.Adam(pw.values(), lr=lr)
  tmask, pmask, losses = Tensor.of(rt, mask.tolist(), "fp32"), torch.tensor(mask), []
  for step in range(1, steps + 1):
    start = time.time()
    at = int(rng.integers(0, len(ids) - c.ctx - 1))
    x, y = ids[at:at + c.ctx].tolist(), ids[at + 1:at + c.ctx + 1].tolist()
    loss = cross_entropy(forward(x, tw, tmask, c), y)
    loss.backward()
    same = {k: torch.tensor(array(t), requires_grad=True) for k, t in tw.items()}  # PyTorch on ferrite's weights
    want = torch.nn.functional.cross_entropy(forward(x, same, pmask, c), torch.tensor(y))
    want.backward()
    grads = max(float(np.abs(array(t.grad) - same[k].grad.numpy()).max() / np.abs(same[k].grad.numpy()).max()) for k, t in tw.items())
    adam(tw.values(), state, step, lr)
    opt.zero_grad()
    own = torch.nn.functional.cross_entropy(forward(x, pw, pmask, c), torch.tensor(y))
    own.backward()
    opt.step()
    losses.append((loss.tolist()[0][0], want.item(), grads, own.item()))
    if log: log(f"step {step}: loss {losses[-1][0]:.4f}, PyTorch's {want.item():.4f} on the same weights (gradients within {grads:.1e}), "
                f"{own.item():.4f} on its own ({time.time() - start:.1f}s)")
  return losses

def array(t): return np.frombuffer(t.tobytes(), np.float32).reshape(t.rows, t.cols)  # an fp32 matrix, as numpy

def main():
  from ferrite import pci
  from ferrite.chip import Chip
  from ferrite.runtime import Runtime
  chip = Chip(pci.open_device())
  losses = train(Runtime(chip, chip.tensix[0], chip.tensix[-1]), int(sys.argv[1]) if len(sys.argv) > 1 else 300, log=print)
  mean = lambda i, part: sum(step[i] for step in part) / len(part)
  print(f"every step: loss within {max(abs(f - p) for f, p, _, _ in losses):.0e} of PyTorch's on the same weights, gradients within "
        f"{max(g for _, _, g, _ in losses):.0e}")
  print(f"mean loss over 50 steps: {mean(0, losses[:50]):.2f} -> {mean(0, losses[-50:]):.2f}, PyTorch on its own "
        f"{mean(3, losses[:50]):.2f} -> {mean(3, losses[-50:]):.2f}")

if __name__ == "__main__": main()
