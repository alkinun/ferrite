# ferrite tensors of any rank, for the framework adapters: a view (shape, strides, offset, and a mask for padding: a
# small ShapeTracker) of a ferrite tensor's elements in row-major order, folded into ferrite's three axes when an op needs
# the data: the tensor as it is, a ferrite transpose (which a matmul reads in place), one move, or moves that swap blocks
# of axes. Frameworks build their graphs, or run eagerly, on these; ferrite fuses and launches the ops underneath.
import math, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ferrite import pci, tensor
from ferrite.chip import Chip
from ferrite.runtime import Runtime

NP = {"fp32": np.float32, "bf16": np.uint16, "int32": np.int32}  # ferrite's dtypes in numpy (bf16 as its bits)

RT = None  # the Runtime the adapters run on: every worker of the chip, unless set first (one process opens the simulator once)
def runtime():
  global RT
  if RT is None:
    chip = Chip(pci.open_device())
    RT = Runtime(chip, chip.tensix[0], chip.tensix[-1])
  return RT

def fold(shape):  # any rank as ferrite's (batch, rows, columns): all but the last axis are rows, so no tile pads a short axis
  return (1, math.prod(shape[:-1]), shape[-1]) if shape else (1, 1, 1)

class View:
  """shape over a tensor's elements in row-major order: element (i0, i1, ...) is element offset + sum(i_k strides_k),
  or 0 where an index is outside mask's (lo, hi) on its axis (padding)."""
  def __init__(self, shape, strides=None, offset=0, mask=None):
    self.shape, self.offset, self.mask = tuple(shape), offset, mask
    self.strides = tuple(strides) if strides is not None else tuple(math.prod(shape[k + 1:]) for k in range(len(shape)))
  def contiguous(self):  # the elements as they lie (an axis of 1 has any stride)
    return self.offset == 0 and self.mask is None and all(n == 1 or s == c for n, s, c in zip(self.shape, self.strides, View(self.shape).strides))
  def permute(self, order):
    return View([self.shape[k] for k in order], [self.strides[k] for k in order], self.offset, self.mask and [self.mask[k] for k in order])
  def expand(self, shape):
    return View(shape, [0 if n != m else s for s, n, m in zip(self.strides, self.shape, shape)], self.offset,
                self.mask and [(0, m if w[0] < w[1] else 0) if n != m else w for w, n, m in zip(self.mask, self.shape, shape)])  # padding stays padding
  def shrink(self, ranges):
    mask = self.mask and [(max(lo - a, 0), min(hi - a, b - a)) for (lo, hi), (a, b) in zip(self.mask, ranges)]
    return View([b - a for a, b in ranges], self.strides, self.offset + sum(a * s for (a, _), s in zip(ranges, self.strides)), mask)
  def pad(self, widths):
    mask = [(lo + a, hi + a) for (lo, hi), (a, _) in zip(self.mask or [(0, n) for n in self.shape], widths)]
    offset = self.offset - sum(a * s for (a, _), s in zip(widths, self.strides))
    return View([n + a + b for n, (a, b) in zip(self.shape, widths)], self.strides, offset, mask)
  def reshape(self, shape):
    """The same elements in shape, without moving them, or None. Axes of 1 come and go; each new axis is a run of old
    ones merged, or an old one split, and runs only merge where their strides line up."""
    if math.prod(shape) != math.prod(self.shape): return None
    if self.contiguous(): return View(shape)
    if self.mask is not None and any(w != (0, n) for w, n in zip(self.mask, self.shape)): return None  # (an axis of 1 may be all padding)
    old = [(n, s) for n, s in zip(self.shape, self.strides) if n > 1]
    strides, i = [], 0
    for n in shape:
      if n == 1: strides.append(0); continue
      size, s = old[i]
      while size < n and i + 1 < len(old) and old[i][1] == old[i + 1][0] * old[i + 1][1]: i += 1; size *= old[i][0]; s = old[i][1]
      if size == n: strides.append(s); i += 1
      elif size % n == 0: strides.append(s * (size // n)); old[i] = (size // n, s)  # split: the rest stays for the next axes
      else: return None
    return View(shape, strides, self.offset)

class Value:
  """A value on the chip: a view of a ferrite tensor's elements, which tinygrad sees as dtype."""
  def __init__(self, t, view, dtype): self.t, self.view, self.dtype = t, view, dtype
  @property
  def shape(self): return self.view.shape
  def whole(self): return self.view.contiguous() and math.prod(self.shape) == math.prod(self.t.shape)  # its tensor, all of it, as it lies

def merge(view, groups):
  """(size, stride, lo, hi) for each group of view's axes taken as one, or None if a group's strides do not line up
  (or it masks more than one axis)."""
  out = []
  for g in groups:
    axes, size = [k for k in g if view.shape[k] > 1], math.prod(view.shape[k] for k in g)
    masked = [k for k in g if view.mask and view.mask[k] != (0, view.shape[k])]  # an axis of 1 too: it may be padding
    if any(view.strides[a] != view.strides[b] * view.shape[b] for a, b in zip(axes, axes[1:])): return None
    if any(view.mask[k][0] >= view.mask[k][1] for k in masked): lo, hi = 0, 0  # an axis of nothing but padding: so is the group
    elif len(masked) > 1 or masked and len(axes) > 1: return None
    else: lo, hi = view.mask[masked[0]] if masked else (0, size)
    out.append((size, view.strides[axes[-1]] if axes else 0, lo, hi))
  return out

def groups3(shape):  # fold's groups of axes
  return [[], list(range(len(shape) - 1)), [len(shape) - 1] if shape else []]

def grouping(shape, shape3):  # shape's axes in three consecutive groups whose sizes are shape3, or None
  for i in range(len(shape) + 1):
    for j in range(i, len(shape) + 1):
      if (math.prod(shape[:i]), math.prod(shape[i:j]), math.prod(shape[j:])) == tuple(shape3):
        return [list(range(i)), list(range(i, j)), list(range(j, len(shape)))]
  return None

def materialize(v, groups=None):
  """v as a ferrite tensor of three axes, each a group of v's axes (fold's, by default): v's own tensor (reshaped if need
  be) if the view is it as it lies, a transpose for a swapped one (which a matmul reads in place), one move, or moves that
  swap blocks of axes (transposes); any other view goes through the host."""
  groups = groups or groups3(v.shape)
  shape3, t = tuple(math.prod(v.shape[k] for k in g) for g in groups), v.t
  if v.whole() and t.shape == shape3: return t
  if v.whole(): return t.reshape(*shape3)
  m = merge(v.view, groups)
  swapped = [shape3[1] * shape3[2], 1, shape3[1]]  # the strides of t's last two axes swapped (a batch of 1 has any)
  if m is not None and all(n == 1 or s == w for n, (_, s, _, _), w in zip(shape3, m, swapped)) and v.view.offset == 0 and not v.view.mask \
     and t.shape == (shape3[0], *shape3[:0:-1]): return t.T
  if m is not None:
    return t._move(shape3, v.view.offset, [s for _, s, _, _ in m], [lo for _, _, lo, _ in m], [hi for _, _, _, hi in m])
  if (p := transposes(v)) is not None: return p if p.shape == shape3 else p.reshape(*shape3)
  return upload(download(v).reshape(shape3), v.t.dtype)

def transposes(v):
  """A view that only reorders its tensor's axes (attention's heads: (B, T, H, D) as (B, H, T, D)) in moves that each
  swap two blocks: the next axis comes to its place, with all after it, past the axes in between. A move reads its
  source in row-major order whatever its three axes, so a swap is one (two for heads apart or together). None for any
  other view."""
  axes = [k for k, n in enumerate(v.shape) if n > 1]
  order = sorted(axes, key=lambda k: -v.view.strides[k])  # the axes as the tensor holds them
  if v.view.offset or v.view.mask or [v.view.strides[k] for k in order] != list(View([v.shape[k] for k in order]).strides): return None
  t, size = v.t, lambda ks: math.prod(v.shape[q] for q in ks)
  for i, k in enumerate(axes):
    j = order.index(k)
    if j == i: continue
    p, y, z = size(order[:i]), size(order[i:j]), size(order[j:])
    t = t._move((p, z, y), 0, (y * z, 1, z))  # blocks y and z swapped
    order = order[:i] + order[j:] + order[i:j]
  return t

def indices(view):  # each element's place among the tensor's, in the view's shape (masked ones too)
  if not view.shape: return np.array(view.offset)
  return view.offset + sum(np.arange(n).reshape([-1 if j == k else 1 for j in range(len(view.shape))]) * s
                           for k, (n, s) in enumerate(zip(view.shape, view.strides)))

def download(v):  # a chip value as numpy, in its shape (bf16 as its bits)
  a = np.frombuffer(v.t.tobytes(), NP[v.t.dtype])
  view = v.view
  out = np.asarray(a[np.clip(indices(view), 0, a.size - 1)])  # (a 0-d array for no shape, not a scalar)
  if view.mask is not None:
    for k, (lo, hi) in enumerate(view.mask):
      keep = (np.arange(view.shape[k]) >= lo) & (np.arange(view.shape[k]) < hi)
      out = out * keep.reshape([-1 if j == k else 1 for j in range(len(view.shape))])
  return out

def upload(a, dtype=None, shape3=None):
  """numpy a (bf16 as its bits) onto the chip as ferrite dtype (by default fp32, or int32 for integers), shape3 or folded."""
  dtype = dtype or ("int32" if a.dtype.kind in "iub" else "fp32")
  shape3 = shape3 or fold(a.shape)
  return tensor.Tensor.frombytes(runtime(), *shape3[1:], np.ascontiguousarray(a, NP[dtype]).tobytes(), shape3[0], dtype)

def broadcast(v, shape):  # numpy's broadcasting of a value: axes aligned at the end, a dimension of 1 spread (as a view)
  if v.shape == tuple(shape): return v
  pad = len(shape) - len(v.shape)
  view = View((1,) * pad + v.shape, (0,) * pad + v.view.strides, v.view.offset, v.view.mask and [(0, 1)] * pad + list(v.view.mask))
  return Value(v.t, view.expand(shape), v.dtype)

def common_groups(shape, values):  # how an elementwise op folds shape: as an operand's tensor already is, else fold's way
  return next((g for v in values if isinstance(v, Value) and v.whole() and (g := grouping(shape, v.t.shape))), groups3(shape))

def operand(v, shape, groups):
  """v as ferrite takes it in an elementwise op of shape folded in groups: a dimension of 1 broadcast as it is where the
  groups keep it one, everything else folded."""
  whole = [not v.view.mask or v.view.mask[k] == (0, n) for k, n in enumerate(v.view.shape)]  # nothing padded along it
  small = [1 if s == 0 and n > 1 and whole[k] else n for k, (n, s) in enumerate(zip(v.view.shape, v.view.strides))]
  view = View(small, [0 if n == 1 else s for n, s in zip(small, v.view.strides)], v.view.offset, v.view.mask and
              [(0, 1) if n < m else w for w, n, m in zip(v.view.mask, small, v.view.shape)])  # a broadcast axis's, now 1
  sizes = lambda shp: [math.prod(shp[k] for k in g) for g in groups]
  if small != list(shape) and all(a in (b, 1) for a, b in zip(sizes(small), sizes(shape))): return materialize(Value(v.t, view, v.dtype), groups)
  return materialize(v, groups)

def spread(t, shape, groups):  # an elementwise result whose operands all broadcast, spread to shape
  shape3 = tuple(math.prod(shape[k] for k in g) for g in groups)
  return t if t.shape == shape3 else t.expand(*shape3)

def reduce(x, axes, op):
  """x summed (op "sum") or maxed over axes, as a ferrite tensor whose elements are the other axes' in row-major order:
  along an axis (or two) of x's tensor when the axes are one, else after one move that puts them last."""
  kept = [k for k in range(len(x.shape)) if k not in axes]
  w = x.view.permute(kept + list(axes))
  f = math.prod(x.shape[k] for k in axes)
  b, r, c = x.t.shape
  w_whole, front = w.contiguous() and math.prod(w.shape) == b * r * c, x.whole() and list(axes) == list(range(len(axes)))
  if w_whole and c == f: t, along = x.t, [2]  # already its last axis, or last two, or first: no move
  elif w_whole and r * c == f: t, along = x.t, [2, 1]
  elif front and b == f: t, along = x.t, [0]
  elif front and b * r == f: t, along = x.t, [0, 1]
  else: t, along = materialize(Value(x.t, w, x.dtype), [[], list(range(len(kept))), list(range(len(kept), len(w.shape)))]), [2]
  for ax in along: t = t.sum(ax) if op == "sum" else t.max(ax)
  return t
