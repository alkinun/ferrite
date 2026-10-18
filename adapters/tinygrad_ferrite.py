# ferrite as a tinygrad device: tinygrad is the frontend (Tensor, nn, autograd, optimizers), ferrite the compiler and the
# runtime. Realizing a FERRITE tensor hands tinygrad's tensor graph (its primitives on shaped tensors, before tinygrad
# would schedule them into kernels) to ferrite op by op, and ferrite fuses and launches them its own way.
# tinygrad's tensors have any rank and ferrite's three axes, so movement stays a view of a ferrite tensor (shape,
# strides, offset, and a mask for padding: a small ShapeTracker), folded into three axes when an op needs the data. A
# multiply summed over axes both operands span, broadcast against each other elsewhere, is a matmul. What ferrite does
# not do (the integer arithmetic of random numbers, other dtypes) runs on tinygrad's own CPU device and comes over.
#   import adapters.tinygrad_ferrite, then DEV=FERRITE or Tensor(..., device="FERRITE")
#   TTSIM=/path/to/libttsim.so python3 adapters/tinygrad_ferrite.py script.py [args]   # a tinygrad script, unchanged
import collections, math, os, runpy, sys, types
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import tinygrad, tinygrad.tensor
from tinygrad import Tensor, dtypes
from tinygrad.dtype import _to_np_dtype
from tinygrad.device import Allocator, BufferStorage, Compiled, Device
from tinygrad.uop.ops import Ops, UOp
from ferrite import pci, tensor
from ferrite.chip import Chip

DTYPES = {dtypes.float32: "fp32", dtypes.bfloat16: "bf16", dtypes.int32: "int32", dtypes.bool: None}  # bool: 0 or 1, any dtype
NP = {"fp32": np.float32, "bf16": np.uint16, "int32": np.int32}  # ferrite's dtypes in numpy (bf16 as its bits)
def npdtype(dt): return np.uint16 if dt is dtypes.bfloat16 else _to_np_dtype(dt)

RT = None  # the Runtime FERRITE runs on: every worker of the chip, unless set first (one process opens the simulator once)
def runtime():
  global RT
  if RT is None:
    from ferrite.runtime import Runtime
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
                self.mask and [(0, m) if n != m else w for w, n, m in zip(self.mask, self.shape, shape)])
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
    if self.mask is not None and any(w != (0, n) for w, n in zip(self.mask, self.shape) if n > 1): return None
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

def merge(view, groups):
  """(size, stride, lo, hi) for each group of view's axes taken as one, or None if a group's strides do not line up
  (or it masks more than one axis)."""
  out = []
  for g in groups:
    axes = [k for k in g if view.shape[k] > 1]
    masked = [k for k in axes if view.mask and view.mask[k] != (0, view.shape[k])]
    if len(axes) > 1 and masked: return None
    if any(view.strides[a] != view.strides[b] * view.shape[b] for a, b in zip(axes, axes[1:])): return None
    lo, hi = view.mask[masked[0]] if masked else (0, math.prod(view.shape[k] for k in g))
    out.append((math.prod(view.shape[k] for k in g), view.strides[axes[-1]] if axes else 0, lo, hi))
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
  """v as a ferrite tensor of three axes, each a group of v's axes (fold's, by default): v's own tensor if the view is
  the tensor as it is, a transpose for a swapped one (which a matmul reads in place), else one move; a view that does not
  fold into three axes goes through the host."""
  groups = groups or groups3(v.shape)
  shape3, t = tuple(math.prod(v.shape[k] for k in g) for g in groups), v.t
  if v.view.contiguous() and t.shape == shape3: return t
  if v.view.contiguous(): return t.reshape(*shape3)
  m = merge(v.view, groups)
  swapped = [shape3[1] * shape3[2], 1, shape3[1]]  # the strides of t's last two axes swapped (a batch of 1 has any)
  if m is not None and all(n == 1 or s == w for n, (_, s, _, _), w in zip(shape3, m, swapped)) and v.view.offset == 0 and not v.view.mask \
     and t.shape == (shape3[0], *shape3[:0:-1]): return t.T
  if m is not None:
    return t._move(shape3, v.view.offset, [s for _, s, _, _ in m], [lo for _, _, lo, _ in m], [hi for _, _, _, hi in m])
  if (p := transposes(v)) is not None: return p if p.shape == shape3 else p.reshape(*shape3)
  return upload(download(v).reshape(shape3), v.dtype)

def transposes(v):
  """A view that only reorders its tensor's axes (attention's heads: (B, T, H, D) as (B, H, T, D)) in moves that each
  swap two blocks: the next axis comes to its place, with all after it, past the axes in between. A move reads its
  source in row-major order whatever its three axes, so a swap is one (two for heads apart or together). None for any
  other view."""
  axes = [k for k, n in enumerate(v.shape) if n > 1]
  order = sorted(axes, key=lambda k: -v.view.strides[k])  # the axes as the tensor holds them
  if v.view.offset or v.view.mask or [v.view.strides[k] for k in order] != list(View([v.shape[k] for k in order]).strides): return None
  t = v.t
  for i, k in enumerate(axes):
    j = order.index(k)
    if j == i: continue
    size = lambda ks: math.prod(v.shape[q] for q in ks)
    p, y, z = size(order[:i]), size(order[i:j]), size(order[j:])
    t = t._move((p, z, y), 0, (y * z, 1, z))  # blocks y and z swapped
    order = order[:i] + order[j:] + order[i:j]
  return t

def download(v):  # a chip value as numpy, in its shape (bf16 as its bits)
  a = np.frombuffer(v.t.tobytes(), NP[v.t.dtype])
  view = v.view
  idx = view.offset + sum(np.arange(n).reshape([-1 if j == k else 1 for j in range(len(view.shape))]) * s
                          for k, (n, s) in enumerate(zip(view.shape, view.strides)))
  out = a[np.clip(idx, 0, a.size - 1)] if view.shape else a[view.offset:view.offset + 1].reshape(())
  if view.mask is not None:
    for k, (lo, hi) in enumerate(view.mask):
      keep = (np.arange(view.shape[k]) >= lo) & (np.arange(view.shape[k]) < hi)
      out = out * keep.reshape([-1 if j == k else 1 for j in range(len(view.shape))])
  return out

def cpu(a):  # numpy (bf16 as its bits) as a tinygrad tensor on its CPU device
  t = Tensor(np.ascontiguousarray(a), device="CPU")
  return t.bitcast(dtypes.bfloat16) if a.dtype == np.uint16 else t
def array(t):  # a host value as numpy (bf16 as its bits), computed now, once
  t = t.realize()
  return t.bitcast(dtypes.uint16).numpy() if t.dtype is dtypes.bfloat16 else t.numpy()
def constant(t):  # a host value that is one number spread over a shape: the number, else None
  return t.uop.base.arg if t.uop.base.op is Ops.CONST else None

def flat(a, n, npd):  # a host value as n values' bytes
  a = np.asarray(a)
  return np.ascontiguousarray(np.broadcast_to(a, (n,)) if a.ndim == 0 else a.reshape(-1), npd).tobytes()

def to_chip(v, dtype):  # a host value (a tinygrad CPU tensor) as a chip value, in its own shape
  return Value(upload(array(v), dtype), View(v.shape), dtype)

def upload(a, dtype, shape3=None):
  """numpy a (bf16 as its bits) onto the chip, shape3 or folded."""
  d = DTYPES[dtype] or ("int32" if a.dtype.kind in "iub" else "fp32")
  a = np.ascontiguousarray(a, NP[d])
  return tensor.Tensor.frombytes(runtime(), *(shape3 or fold(a.shape))[1:], a.tobytes(), (shape3 or fold(a.shape))[0], d)

def on_ferrite(u): return isinstance(u.device, str) and u.device.startswith("FERRITE")

# Elementwise ops: tinygrad's on ferrite's (ferrite's comparisons give 1 or 0 in their operands' dtype, which is how a bool lives here)
UNARY = {Ops.EXP2: tensor.Tensor.exp2, Ops.LOG2: tensor.Tensor.log2, Ops.SIN: tensor.Tensor.sin, Ops.SQRT: tensor.Tensor.sqrt,
         Ops.RECIPROCAL: tensor.Tensor.reciprocal, Ops.NEG: lambda x: -x}
def chip(x): return isinstance(x, tensor.Tensor)  # else a number
BINARY = {Ops.ADD: lambda a, b: a + b, Ops.SUB: lambda a, b: a - b, Ops.MUL: lambda a, b: a * b, Ops.CMPLT: lambda a, b: a < b,
          Ops.MAX: lambda a, b: a.maximum(b) if chip(a) else b.maximum(a), Ops.CMPNE: lambda a, b: a.ne(b) if chip(a) else b.ne(a),
          Ops.CMPEQ: lambda a, b: a.eq(b) if chip(a) else b.eq(a), Ops.FDIV: lambda a, b: a * b.reciprocal() if chip(b) else a / b,
          Ops.AND: lambda a, b: a * b, Ops.OR: lambda a, b: a.maximum(b) if chip(a) else b.maximum(a)}  # AND and OR of bools only
ELEMENTWISE = {*UNARY, *BINARY, Ops.WHERE, Ops.CAST}

def broadcast(v, shape):  # an operand of a binary op, numpy's way: axes aligned at the end, a dimension of 1 spread (as a view)
  if isinstance(v, Tensor): return v.reshape((1,) * (len(shape) - v.ndim) + v.shape).expand(shape)
  if not isinstance(v, Value) or v.shape == tuple(shape): return v
  pad = len(shape) - len(v.shape)
  view = View((1,) * pad + v.shape, (0,) * pad + v.view.strides, v.view.offset, v.view.mask and [(0, 1)] * pad + list(v.view.mask))
  return Value(v.t, view.expand(shape), v.dtype)

DEFER = object()  # a product only sums read: the sum makes it a matmul, or asks for it (force)

class Evaluator:
  """tinygrad's tensor graph, evaluated: each UOp to a Value on the chip, a number, or a tinygrad tensor on its CPU device."""
  def __init__(self, users): self.values, self.users = {}, users

  def summed(self, u):  # every use of u is a sum (through a permute): a matmul's product
    return all(r.op is Ops.REDUCE and r.arg[0] is Ops.ADD or r.op is Ops.PERMUTE and self.summed(r) for r in self.users[u])

  def force(self, u):  # a deferred value, computed after all
    if self.values[u] is DEFER: self.values[u] = self.eval(u, defer=False)
    return self.values[u]

  def operand(self, x, shape, groups=None):
    """x as ferrite takes it in an elementwise op of tinygrad shape: a number, or a tensor whose axes are shape's in
    groups (fold's by default), a dimension of 1 broadcast as it is where the groups keep it one."""
    if isinstance(x, (bool, int, float)): return int(x) if isinstance(x, bool) else x
    if isinstance(x, Tensor):  # from the host: a number if it is one spread over a shape
      if (c := constant(x)) is not None: return int(c) if isinstance(c, bool) else c
      shape3 = [math.prod(shape[k] for k in g) for g in groups] if groups else None
      return upload(array(x), dtypes.float32 if dtypes.is_float(x.dtype) else dtypes.int32, shape3)
    groups = groups or groups3(shape)
    small = [1 if s == 0 and n > 1 else n for n, s in zip(x.view.shape, x.view.strides)]
    view = View(small, [0 if n == 1 else s for n, s in zip(small, x.view.strides)], x.view.offset, x.view.mask and
                [(0, 1) if n == 1 else w for w, n in zip(x.view.mask, small)])
    sizes = lambda shp: [math.prod(shp[k] for k in g) for g in groups]
    if small != list(shape) and all(a in (b, 1) for a, b in zip(sizes(small), sizes(shape))): return materialize(Value(x.t, view, x.dtype), groups)
    return materialize(x, groups)

  def chip_op(self, u, srcs):
    """u on the chip, or None if ferrite does not do it."""
    if u.op in ELEMENTWISE and (u.dtype not in DTYPES or any(isinstance(s, Value) and s.dtype not in DTYPES for s in srcs)): return None
    shape = tuple(u.shape)
    groups = next((g for s in srcs if isinstance(s, Value) and s.view.contiguous() and (g := grouping(shape, s.t.shape))), groups3(shape))
    shape3 = tuple(math.prod(shape[k] for k in g) for g in groups)
    whole = lambda t: Value(t if t.shape == shape3 else t.expand(*shape3), View(shape), u.dtype)  # operands all broadcast: spread
    if u.op in UNARY: return whole(UNARY[u.op](self.operand(srcs[0], shape, groups)))
    if u.op in BINARY: return whole(BINARY[u.op](*(self.operand(s, shape, groups) for s in srcs)))
    if u.op is Ops.WHERE:
      c, x, y = (self.operand(s, shape, groups) for s in srcs)
      if not chip(c): return None
      return whole(c.where(x, y))
    if u.op is Ops.CAST:
      x = self.operand(srcs[0], shape, groups)
      if not chip(x): return None
      return whole(x.ne(0) if u.dtype is dtypes.bool else x.cast(DTYPES[u.dtype]))
    return None

  def reduce(self, u, x):
    """REDUCE(op, n) folds x's first n axes: along an axis (or two) of x's tensor when they are one, else after one move
    that puts them last."""
    op, n = u.arg
    kept, folded = list(range(n, len(x.shape))), list(range(n))
    w = x.view.permute(kept + folded)
    f = math.prod(x.shape[i] for i in folded)
    b, r, c = x.t.shape
    if w.contiguous() and c == f: t, axes = x.t, [2]  # already its last axis, or last two, or first: no move
    elif w.contiguous() and r * c == f: t, axes = x.t, [2, 1]
    elif x.view.contiguous() and b == f: t, axes = x.t, [0]
    elif x.view.contiguous() and b * r == f: t, axes = x.t, [0, 1]
    else: t, axes = materialize(Value(x.t, w, x.dtype), [[], list(range(len(kept))), list(range(len(kept), len(w.shape)))]), [2]
    for ax in axes: t = t.sum(ax) if op is Ops.ADD else t.max(ax)
    return Value(t, View(tuple(u.shape)), u.dtype)

  def matmul(self, u):
    """The REDUCE(ADD) u as a matmul, or None: its operand a MUL (behind a PERMUTE) of a and b, which do not broadcast
    along the summed axes; along the others both span the batch, a alone the rows and b alone the columns."""
    p = u.src[0]
    order = p.arg if p.op is Ops.PERMUTE else tuple(range(len(p.shape)))
    m = p.src[0] if p.op is Ops.PERMUTE else p
    if m.op is not Ops.MUL: return None
    up = lambda s, v: to_chip(v, s.dtype) if isinstance(v, Tensor) and constant(v) is None else v  # a weight still on the host goes up
    a, b = (broadcast(up(s, self.read(s, self.values.get(s))), m.shape) for s in m.src)
    if not (isinstance(a, Value) and isinstance(b, Value)): return None
    if a.view.mask or b.view.mask or a.dtype not in DTYPES or b.dtype not in DTYPES: return None
    shape, summed = m.shape, list(order[:u.arg[1]])
    full = lambda v, k: shape[k] == 1 or v.view.strides[k] != 0
    if not all(full(a, k) and full(b, k) for k in summed): return None
    free = [k for k in order[u.arg[1]:]]
    batch = [k for k in free if full(a, k) and full(b, k)]
    rows, cols = [k for k in free if full(a, k) and not full(b, k)], [k for k in free if full(b, k) and not full(a, k)]
    if len(batch) + len(rows) + len(cols) != len(free) or not (rows or cols): return None
    pick = lambda v, axes: View([shape[k] for k in axes], [v.view.strides[k] for k in axes], v.view.offset)
    nb, nr, nk = len(batch), len(rows), len(summed)
    if not batch and nr > 1 and merge(pick(a, rows), [list(range(nr))]) is None: nb, nr = nr - 1, 1  # leading rows as the batch b repeats
    runs = lambda *sizes: [list(range(sum(sizes[:i]), sum(sizes[:i + 1]))) for i in range(3)]  # consecutive axes in groups of these sizes
    A = materialize(Value(a.t, pick(a, batch + rows + summed), a.dtype), runs(nb, nr, nk))
    B = materialize(Value(b.t, pick(b, batch + summed + cols), b.dtype), runs(len(batch), nk, len(cols)))
    out = A @ B
    made = batch + rows + cols  # the result's axes as made, and as u wants them (free's order)
    return Value(out, View([shape[k] for k in made]).permute([made.index(k) for k in free]), u.dtype)

  def movement(self, u, x):
    """A view of x, to become one move or none when an op needs the data; x moves first for a reshape its view cannot follow."""
    shape = tuple(u.shape)
    if u.op is Ops.RESHAPE: view = x.view.reshape(shape)
    elif u.op is Ops.PERMUTE: view = x.view.permute(u.arg)
    elif u.op is Ops.EXPAND: view = broadcast(x, shape).view
    elif u.op is Ops.SHRINK: view = x.view.shrink(self.marg(u))
    else: view = x.view.pad(self.marg(u))
    if view is None:  # a reshape the view cannot follow: moved first, straight into the three axes the new shape folds to
      x, view = Value(materialize(x, grouping(x.shape, fold(shape))), View(x.shape), x.dtype), View(shape)
    return Value(x.t, view, x.dtype)

  def marg(self, u):  # SHRINK's (start, end) and PAD's (before, after) on each axis, from their (start, size) STACK sources
    ints = lambda src: (src.arg,) if src.op is Ops.CONST else tuple(s.arg if s.op is Ops.CONST else self.values[s] for s in src.src)
    start, size = ints(u.src[1]), ints(u.src[2])
    if u.op is Ops.SHRINK: return [(a, a + n) for a, n in zip(start, size)]
    return [(a, n - old - a) for a, n, old in zip(start, size, u.src[0].shape)]

  def on_host(self, u, srcs):
    """u by tinygrad on its CPU device, lazily, so tinygrad fuses what runs there (its random numbers sum a padded view it
    never makes whole) and computes it only when the chip needs it."""
    there = lambda s, v: (cpu(download(v)) if isinstance(v, Value) else v).cast(s.dtype).uop if isinstance(v, (Value, Tensor)) else s
    return Tensor(u.replace(src=tuple(there(s, v) for s, v in zip(u.src, srcs))))

  def eval(self, u, defer=True):
    if defer and (u.op is Ops.MUL or u.op is Ops.PERMUTE and self.values.get(u.src[0]) is DEFER) and self.summed(u): return DEFER
    srcs = [self.values.get(s) for s in u.src]
    if u.op is not Ops.REDUCE: srcs = [self.force(s) if v is DEFER else v for s, v in zip(u.src, srcs)]  # a sum may make a matmul of them
    if u.op is Ops.CONST: return u.arg
    if u.op is Ops.STACK and all(isinstance(v, (bool, int, float)) for v in srcs): return tuple(srcs)  # a shape; else values, stacked on the host
    if u.op in (Ops.BUFFER, Ops.ALLOC): return self.read(u, u)  # as it is before any STORE in this graph
    if u.op is Ops.COPY and on_ferrite(u) and isinstance(v := self.read(u.src[0], srcs[0]), Tensor) and u.dtype in DTYPES and v.ndim:
      return to_chip(v, u.dtype)
    if u.op in (Ops.DETACH, Ops.CONTIGUOUS_BACKWARD, Ops.STAGE, Ops.COPY): return self.read(u.src[0], srcs[0])
    if u.op is Ops.AFTER: return self.now(u.src[0])
    if u.op is Ops.STORE: return self.store(u.src[0], self.read(u.src[1], srcs[1]))
    if u.op is Ops.SINK: return None
    x = [self.read(s, v) for s, v in zip(u.src, srcs)]
    if u.op is Ops.RESHAPE and u.src[0].op is Ops.BUFFER and on_ferrite(u.src[0]) and (v := self.lay_out(u, x[0])) is not None: return v
    if u.op in (Ops.RESHAPE, Ops.PERMUTE, Ops.EXPAND, Ops.SHRINK, Ops.PAD):
      return self.movement(u, x[0]) if isinstance(x[0], Value) else self.on_host(u, x)
    if u.op in BINARY or u.op is Ops.WHERE: x = [broadcast(v, u.shape) for v in x]
    if u.op is Ops.REDUCE and x[0] is DEFER:  # a product only sums read: a matmul, or the product after all
      if (mm := self.matmul(u)) is not None: return mm
      x[0] = self.force(u.src[0])
    if u.op is Ops.REDUCE and isinstance(x[0], Value) and x[0].dtype in DTYPES and u.arg[0] in (Ops.ADD, Ops.MAX): return self.reduce(u, x[0])
    if any(isinstance(v, Value) for v in x) and (v := self.chip_op(u, x)) is not None: return v
    return self.on_host(u, x)

  def lay_out(self, u, x):
    """A FERRITE buffer as a use reshapes it, laid out on the chip in that shape and kept so: bytes from the host go up in
    it, and a tensor kept in a shape its uses do not fold to (tinygrad makes weights flat) moves into it once. Else None."""
    h = u.src[0].buffer._buf
    if isinstance(x, Tensor) and constant(x) is None and u.dtype in DTYPES and h.value is None:
      h.value = Value(upload(array(x).reshape(u.shape), u.dtype), View(x.shape), u.dtype)
    elif isinstance(x, Value) and x.view.contiguous() and x.t.shape != fold(u.shape):
      h.value = Value(materialize(Value(x.t, View(u.shape), x.dtype)).realize(), x.view, x.dtype)
    else: return None
    return Value(h.value.t, View(u.shape), u.dtype)

  def now(self, x):  # x (a buffer, or a view of one) as the buffers hold it now, after the STOREs so far
    if x.op in (Ops.BUFFER, Ops.ALLOC): return self.read(x, x)
    if x.op is Ops.AFTER: return self.now(x.src[0])
    v = self.now(x.src[0])
    return self.movement(x, v) if isinstance(v, Value) else self.on_host(x, [v] + [self.values.get(s) for s in x.src[1:]])

  def read(self, s, v):
    """The value of source s: a buffer's contents as they are now (a FERRITE one's chip value or bytes, another device's bytes)."""
    if not (isinstance(v, UOp) and v.op in (Ops.BUFFER, Ops.ALLOC)): return v
    buf = v.buffer
    if on_ferrite(v):
      h = buf.ensure_allocated()._buf
      if h.value is not None: return Value(h.value.t, h.value.view, h.value.dtype)
      if h.bytes is None: return Tensor.zeros(buf.nbytes // v.dtype.itemsize, dtype=v.dtype, device="CPU")  # nothing stored yet
      return cpu(np.frombuffer(h.bytes, npdtype(v.dtype)))  # on the host until a use gives it a shape
    return cpu(np.frombuffer(buf.as_memoryview(), npdtype(v.dtype)).copy().reshape(v.shape))

  def store(self, target, value):
    """value into the buffer target (a whole one, perhaps reshaped)."""
    while target.op in (Ops.RESHAPE, Ops.AFTER): target = target.src[0]
    assert target.op in (Ops.BUFFER, Ops.ALLOC), f"a store into {target.op}"
    buf = target.buffer.ensure_allocated()
    n, npd = buf.nbytes // target.dtype.itemsize, npdtype(target.dtype)
    data = lambda: download(value) if isinstance(value, Value) else array(value) if isinstance(value, Tensor) else value
    if not on_ferrite(target): buf.copyin(memoryview(flat(data(), n, npd)))
    elif isinstance(value, Value):  # kept in whatever three axes it has, if its view is it as it lies
      t = value.t if value.view.contiguous() else materialize(value)
      buf._buf.value, buf._buf.bytes = Value(t.realize(), View(value.shape), value.dtype), None
    else: buf._buf.bytes, buf._buf.value = flat(data(), n, npd), None
    return None

# tinygrad's side: the FERRITE device, its buffers, and the graphs it hands over
class Holder:  # a FERRITE buffer's contents: bytes from the host, or a chip value (its own view and dtype)
  def __init__(self): self.bytes, self.value = None, None

class FerriteAllocator(Allocator):
  def _alloc(self, size, options): return BufferStorage(Holder())
  def _copyin(self, dest, src): dest.bytes, dest.value = bytes(src), None
  def _copyout(self, dest, src):
    if src.value is not None: dest[:] = np.ascontiguousarray(download(src.value)).tobytes()[:len(dest)]
    else: dest[:] = src.bytes[:len(dest)] if src.bytes is not None else bytes(len(dest))

class FerriteDevice(Compiled):
  def __init__(self, device): super().__init__(device, FerriteAllocator(self), [], None)
  def synchronize(self, timeout=None): pass  # reading a chip value back waits for what it needs

def run(sink):
  users = collections.defaultdict(list)
  for u in (order := sink.toposort()):
    for s in u.src: users[s].append(u)
  ev = Evaluator(users)
  for u in order: ev.values[u] = ev.eval(u)

_schedule = tinygrad.tensor.create_linear_with_vars
def schedule(sink):  # a graph that touches FERRITE runs here, whole; the rest is tinygrad's
  if not any(u.op in (Ops.BUFFER, Ops.COPY) and on_ferrite(u) for u in sink.toposort()): return _schedule(sink)
  run(sink)
  return UOp(Ops.LINEAR, src=()), {}

mod = types.ModuleType("tinygrad.runtime.ops_ferrite")
mod.FerriteDevice = FerriteDevice
sys.modules["tinygrad.runtime.ops_ferrite"], tinygrad.runtime.ops_ferrite = mod, mod
if "FERRITE" not in Device._devices: Device._devices.append("FERRITE")
tinygrad.tensor.create_linear_with_vars = schedule

if __name__ == "__main__":  # a tinygrad script on ferrite, as it is (tinygrad's JIT replays schedules, which this does not make)
  from tinygrad.helpers import DEV, JIT
  DEV.value, JIT.value = "FERRITE", 0
  sys.argv = sys.argv[1:]
  runpy.run_path(sys.argv[0], run_name="__main__")
