# ferrite as a tinygrad device: tinygrad is the frontend (Tensor, nn, autograd, optimizers), ferrite the compiler and the
# runtime. Realizing a FERRITE tensor hands tinygrad's tensor graph (its primitives on shaped tensors, before tinygrad
# would schedule them into kernels) to ferrite op by op, and ferrite fuses and launches them its own way.
# tinygrad's tensors have any rank and ferrite's three axes, so movement stays a view of a ferrite tensor (View),
# folded into three axes when an op needs the data. A multiply summed over axes both operands span, broadcast against
# each other elsewhere, is a matmul. What ferrite does not do (the integer arithmetic of random numbers, other dtypes)
# runs on tinygrad's own CPU device and comes over. TinyJit replays a call's launches, as ferrite's own jit does.
#   import adapters.tinygrad_ferrite, then DEV=FERRITE or Tensor(..., device="FERRITE")
#   TTSIM=/path/to/libttsim.so python3 adapters/tinygrad_ferrite.py script.py [args]   # a tinygrad script, unchanged
# PyTorch code runs the same way, through tinygrad's own PyTorch frontend (its extra/torch_backend: the "tiny" device).
import collections, dataclasses, math, os, runpy, sys, types
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import tinygrad, tinygrad.tensor, tinygrad.engine.jit
from tinygrad.helpers import JIT
from tinygrad.engine.jit import JitError, _check_no_non_tensor_return, _prepare_jit_inputs
from tinygrad.nn.state import get_parameters
from tinygrad import Tensor, dtypes
from tinygrad.dtype import _to_np_dtype
from tinygrad.device import Allocator, BufferStorage, Compiled, Device
from tinygrad.uop.ops import GroupOp, Ops, PatternMatcher, UOp, UPat, graph_rewrite, identity_element
from tinygrad.schedule.prepare import resolve_function
from ferrite import pci, tensor
from ferrite.chip import Chip
from ferrite.runtime import Runtime

# Views: a value on the chip is a view (shape, strides, offset, and a mask for padding: a small ShapeTracker) of a ferrite
# tensor's elements in row-major order, folded into ferrite's three axes when an op needs the data: the tensor as it is,
# a ferrite transpose (which a matmul reads in place), one move, or moves that swap blocks of axes.
NP = {"fp32": np.float32, "bf16": np.uint16, "int32": np.int32}  # ferrite's dtypes in numpy (bf16 as its bits)

RT = None  # the Runtime the adapter runs on: every worker of the chip, unless set first (one process opens the simulator once)
def runtime():
  global RT
  if RT is None:
    chip = Chip(pci.open_device())
    RT = Runtime(chip)
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
    mask = self.mask and [(0, m if w[0] < w[1] else 0) if n != m else w for w, n, m in zip(self.mask, self.shape, shape)]  # padding
    return View(shape, [0 if n != m else s for s, n, m in zip(self.strides, self.shape, shape)], self.offset, mask)  # stays padding
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
  def covers(self):  # all of it, in any order
    return not self.view.mask and math.prod(n for n, s in zip(self.shape, self.view.strides) if s) == math.prod(self.t.shape)

def uncast(v):  # a bf16 tensor cast to fp32 for a matmul, as the bf16: ferrite multiplies fp32 by bf16 exactly (no lo part
  if isinstance(v, Value) and v.t.op == "cast" and v.t.dtype == "fp32" and v.t.srcs[0].dtype == "bf16":  # for it), and reads its
    return Value(v.t.srcs[0], v.view, dtypes.bfloat16)  # transpose in place: the cast never runs
  return v

def merge(view, groups):
  """(size, stride, lo, hi) for each group of view's axes taken as one, or None if a group's strides do not line up, or
  if its mask does not either (on more than one axis, or on one beside another longer than 1)."""
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
      out = np.where(keep.reshape([-1 if j == k else 1 for j in range(len(view.shape))]), out, out.dtype.type(0))  # (inf * 0 is NaN)
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

DTYPES = {dtypes.float32: "fp32", dtypes.bfloat16: "bf16", dtypes.int32: "int32", dtypes.bool: None}  # bool: 0 or 1, any dtype
def npdtype(dt): return np.uint16 if dt is dtypes.bfloat16 else _to_np_dtype(dt)

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

recording, hosted = None, 0  # while TinyJit records a call: each buffer it changes, [holder, before, stored into], and
def note_host_work():  # its work on the host, with what it found there, which a replay would not do again
  global hosted
  if recording is not None: hosted += 1

def to_chip(v, dtype):  # a host value (a tinygrad CPU tensor) as a chip value, in its own shape
  return Value(upload(array(v), DTYPES[dtype]), View(v.shape), dtype)

fixed = {}  # host operands that read no buffer (a constant's arithmetic, an arange), by UOp: the same every time
def up(x, dtype, shape3):
  """A host value as an operand on the chip, folded to shape3: a number if all its values are one, else a tensor. One
  that reads no buffer is worked out and goes up once (the last 256 kept)."""
  if (key := (x.uop, dtype, shape3)) in fixed: return fixed[key]
  a = array(x.float() if x.dtype is dtypes.bfloat16 else x)  # values, not bf16's bits
  v = (int(c) if isinstance(c := a.flat[0].item(), bool) else c) if a.size and (a == a.flat[0]).all() else upload(a, dtype, shape3)
  if not any(u.op in (Ops.BUFFER, Ops.ALLOC, Ops.PARAM) for u in x.uop.toposort()):
    if len(fixed) >= 256: del fixed[next(iter(fixed))]
    fixed[key] = v
  return v

def on_ferrite(u): return isinstance(u.device, str) and u.device.startswith("FERRITE")

# Elementwise ops: tinygrad's on ferrite's (ferrite's comparisons give 1 or 0 in their operands' dtype, which is how a bool lives here)
UNARY = {Ops.EXP2: tensor.Tensor.exp2, Ops.LOG2: tensor.Tensor.log2, Ops.SIN: tensor.Tensor.sin, Ops.SQRT: tensor.Tensor.sqrt,
         Ops.RECIPROCAL: tensor.Tensor.reciprocal, Ops.NEG: lambda x: -x}
def is_tensor(x): return isinstance(x, tensor.Tensor)  # a ferrite tensor, else a number
BINARY = {Ops.ADD: lambda a, b: a + b, Ops.SUB: lambda a, b: a - b, Ops.MUL: lambda a, b: a * b, Ops.CMPLT: lambda a, b: a < b,
          Ops.MAX: lambda a, b: a.maximum(b) if is_tensor(a) else b.maximum(a), Ops.CMPNE: lambda a, b: a.ne(b) if is_tensor(a) else b.ne(a),
          Ops.CMPEQ: lambda a, b: a.eq(b) if is_tensor(a) else b.eq(a), Ops.FDIV: lambda a, b: a * b.reciprocal() if is_tensor(b) else a / b,
          Ops.AND: lambda a, b: a * b, Ops.OR: lambda a, b: a.maximum(b) if is_tensor(a) else b.maximum(a)}  # AND and OR of bools only
ELEMENTWISE = {*UNARY, *BINARY, Ops.WHERE, Ops.CAST}
EMPTIABLE = GroupOp.Elementwise | GroupOp.Movement | {Ops.REDUCE, Ops.COPY}  # ops whose value can have nothing in it

def spread_to(v, shape):  # an operand of a binary op, numpy's way: on the chip a view, on the host tinygrad's
  if isinstance(v, Tensor): return v.reshape((1,) * (len(shape) - v.ndim) + v.shape).expand(shape)
  return broadcast(v, shape) if isinstance(v, Value) else v

DEFER = object()  # a product only sums read: the sum makes it a matmul, or asks for it (force)

class Evaluator:
  """tinygrad's tensor graph, evaluated: each UOp to a Value on the chip, a number, or a tinygrad tensor on its CPU device."""
  def __init__(self, users): self.values, self.users = {}, users

  def summed(self, u):  # every use of u is a sum (through a permute): a matmul's product
    return all(r.op is Ops.REDUCE and r.arg[0] is Ops.ADD or r.op is Ops.PERMUTE and self.summed(r) for r in self.users[u])

  def force(self, u):  # a deferred value, computed after all
    if self.values[u] is DEFER: self.values[u] = self.eval(u, defer=False)
    return self.values[u]

  def operand(self, x, shape, groups):
    """x as ferrite takes it in an elementwise op of tinygrad shape folded in groups: a number, or a tensor."""
    if isinstance(x, (bool, int, float)): return int(x) if isinstance(x, bool) else x
    if isinstance(x, Tensor):  # from the host: a number if it is one spread over a shape
      if (c := constant(x)) is not None: return int(c) if isinstance(c, bool) else c
      return up(x, "fp32" if dtypes.is_float(x.dtype) else "int32", tuple(math.prod(shape[k] for k in g) for g in groups))
    return operand(x, shape, groups)

  def chip_op(self, u, srcs):
    """u on the chip, or None if ferrite does not do it."""
    if u.op in ELEMENTWISE and (u.dtype not in DTYPES or any(isinstance(s, Value) and s.dtype not in DTYPES for s in srcs)): return None
    if u.op in (Ops.AND, Ops.OR) and u.dtype is not dtypes.bool: return None  # bitwise, on the host
    shape = tuple(u.shape)
    groups = common_groups(shape, srcs)
    as_u = lambda t: t.cast(DTYPES[u.dtype]) if DTYPES[u.dtype] else t  # ferrite gives some int32 ops in fp32 (a product)
    whole = lambda t: Value(spread(as_u(t), shape, groups), View(shape), u.dtype)
    if u.op in UNARY: return whole(UNARY[u.op](self.operand(srcs[0], shape, groups)))
    if u.op in BINARY: return whole(BINARY[u.op](*(self.operand(s, shape, groups) for s in srcs)))
    if u.op is Ops.WHERE:
      c, x, y = (self.operand(s, shape, groups) for s in srcs)
      if not is_tensor(c): return None
      return whole(c.where(x, y))
    if u.op is Ops.CAST:  # of a view of all of a tensor (a weight transposed): the tensor cast, the view kept, so a matmul
      if isinstance(v := srcs[0], Value) and v.covers() and DTYPES.get(u.dtype):  # can read the cast's source in place
        return Value(v.t.cast(DTYPES[u.dtype]), v.view, u.dtype)
      x = self.operand(srcs[0], shape, groups)
      if not is_tensor(x): return None
      return whole(x.ne(0) if u.dtype is dtypes.bool else x.cast(DTYPES[u.dtype]))
    return None

  def reduce(self, u, x):  # REDUCE(op, n) folds x's first n axes
    return Value(reduce(x, list(range(u.arg[1])), "sum" if u.arg[0] is Ops.ADD else "max"), View(tuple(u.shape)), u.dtype)

  def matmul(self, u):
    """The REDUCE(ADD) u as a matmul, or None: its operand a MUL (behind a PERMUTE) of a and b, which do not broadcast
    along the summed axes; along the others both span the batch, a alone the rows and b alone the columns."""
    p = u.src[0]
    order = p.arg if p.op is Ops.PERMUTE else tuple(range(len(p.shape)))
    m = p.src[0] if p.op is Ops.PERMUTE else p
    if m.op is not Ops.MUL or not all(dtypes.is_float(s.dtype) and s.dtype in DTYPES for s in m.src): return None
    lift = lambda s, v: to_chip(v, s.dtype) if isinstance(v, Tensor) and constant(v) is None else v  # a weight still on the host goes up
    a, b = (uncast(spread_to(lift(s, self.read(self.values.get(s))), m.shape)) for s in m.src)
    if not (isinstance(a, Value) and isinstance(b, Value)) or a.view.mask or b.view.mask: return None
    shape, summed = m.shape, list(order[:u.arg[1]])
    full = lambda v, k: shape[k] == 1 or v.view.strides[k] != 0
    if not all(full(a, k) and full(b, k) for k in summed): return None
    free = list(order[u.arg[1]:])
    batch = [k for k in free if full(a, k) and full(b, k)]
    rows, cols = [k for k in free if full(a, k) and not full(b, k)], [k for k in free if full(b, k) and not full(a, k)]
    if len(batch) + len(rows) + len(cols) != len(free) or not (rows or cols): return None
    pick = lambda v, axes: View([shape[k] for k in axes], [v.view.strides[k] for k in axes], v.view.offset)
    nb, nr, nk = len(batch), len(rows), len(summed)
    if not batch and nr > 1 and merge(pick(a, rows), [list(range(nr))]) is None: nb, nr = nr - 1, 1  # leading rows as the batch b repeats
    runs = lambda *sizes: [list(range(sum(sizes[:i]), sum(sizes[:i + 1]))) for i in range(3)]  # consecutive axes in groups of these sizes
    A = materialize(Value(a.t, pick(a, batch + rows + summed), a.dtype), runs(nb, nr, nk))
    B = materialize(Value(b.t, pick(b, batch + summed + cols), b.dtype), runs(len(batch), nk, len(cols)))
    out = A.matmul(B, DTYPES[u.dtype])  # (as u's dtype: an fp32 product of bf16s that were cast up stays fp32)
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
    def there(s, v):  # a source as tinygrad has it: a chip value brought down (work on the host), a host value as it is
      if not isinstance(v, (Value, Tensor)): return s
      return (note_host_work() or cpu(download(v)) if isinstance(v, Value) else v).cast(s.dtype).uop
    return Tensor(u.replace(src=tuple(there(s, v) for s, v in zip(u.src, srcs))))

  def eval(self, u, defer=True):
    if u.op in EMPTIABLE and 0 in u.shape: return Tensor.zeros(u.shape, dtype=u.dtype, device="CPU")  # nothing in it: on the host
    if u.op is Ops.REDUCE and 0 in u.src[0].shape:  # a sum or max of nothing: the identity, on the host
      return Tensor.full(u.shape, identity_element(u.arg[0], u.dtype), dtype=u.dtype, device="CPU")
    if defer and (u.op is Ops.MUL or u.op is Ops.PERMUTE and self.values.get(u.src[0]) is DEFER) and self.summed(u): return DEFER
    srcs = [self.values.get(s) for s in u.src]
    if u.op is not Ops.REDUCE: srcs = [self.force(s) if v is DEFER else v for s, v in zip(u.src, srcs)]  # a sum may make a matmul of them
    if u.op is Ops.CONST: return u.arg
    if u.op is Ops.STACK and all(isinstance(v, (bool, int, float)) for v in srcs): return tuple(srcs)  # a shape; else values, stacked on the host
    if u.op in (Ops.BUFFER, Ops.ALLOC): return self.read(u)  # as it is before any STORE in this graph
    if u.op is Ops.COPY and on_ferrite(u) and isinstance(v := self.read(srcs[0]), Tensor) and u.dtype in DTYPES and v.ndim:
      return to_chip(v, u.dtype)
    if u.op in (Ops.DETACH, Ops.CONTIGUOUS_BACKWARD, Ops.STAGE, Ops.COPY): return self.read(srcs[0])
    if u.op is Ops.AFTER: return self.now(u.src[0])
    if u.op is Ops.STORE: return self.store(u.src[0], self.read(srcs[1]))
    if u.op is Ops.SINK: return None
    x = [self.read(v) for v in srcs]
    if u.op is Ops.RESHAPE and u.src[0].op is Ops.BUFFER and on_ferrite(u.src[0]) and (v := self.lay_out(u, x[0])) is not None: return v
    if u.op in (Ops.RESHAPE, Ops.PERMUTE, Ops.EXPAND, Ops.SHRINK, Ops.PAD):  # (symbolic shapes on the host)
      return self.movement(u, x[0]) if isinstance(x[0], Value) and all(isinstance(d, int) for d in u.shape) else self.on_host(u, x)
    if u.op in BINARY or u.op is Ops.WHERE: x = [spread_to(v, u.shape) for v in x]
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
    if isinstance(h, Part): return None
    if isinstance(x, Tensor) and constant(x) is None and u.dtype in DTYPES and h.value is None:
      h.set(value=Value(upload(array(x).reshape(u.shape), DTYPES[u.dtype]), View(x.shape), u.dtype), stored=False)
    elif isinstance(x, Value) and x.whole() and x.t.shape != fold(u.shape):
      h.set(value=Value(materialize(Value(x.t, View(u.shape), x.dtype)).realize(), x.view, x.dtype), stored=False)
    else: return None
    return Value(h.value.t, View(u.shape), u.dtype)

  def now(self, x):  # x (a buffer, or a view of one) as the buffers hold it now, after the STOREs so far
    if x.op in (Ops.BUFFER, Ops.ALLOC): return self.read(x)
    if x.op is Ops.AFTER: return self.now(x.src[0])
    v = self.now(x.src[0])
    return self.movement(x, v) if isinstance(v, Value) else self.on_host(x, [v] + [self.values.get(s) for s in x.src[1:]])

  def read(self, v):
    """A source's value v, or if it is a buffer, its contents as they are now (a FERRITE one's chip value or bytes,
    another device's bytes)."""
    if not (isinstance(v, UOp) and v.op in (Ops.BUFFER, Ops.ALLOC)): return v
    buf = v.buffer
    if buf.nbytes == 0: return Tensor.zeros(v.shape, dtype=v.dtype, device="CPU")
    if on_ferrite(v):
      h = buf.ensure_allocated()._buf
      if h.value is not None: return Value(h.value.t, h.value.view, h.value.dtype)
      if (data := h.bytes) is None: return Tensor.zeros(buf.nbytes // v.dtype.itemsize, dtype=v.dtype, device="CPU")  # nothing stored yet
      if isinstance(h, Holder) and v.dtype in DTYPES and len(data) <= 32 * v.dtype.itemsize:  # a tile's row at most (an optimizer's
        h.set(value=Value(upload(np.frombuffer(data, npdtype(v.dtype)), DTYPES[v.dtype]), View(v.shape), v.dtype))  # numbers): up
        return Value(h.value.t, h.value.view, h.value.dtype)
      return cpu(np.frombuffer(data, npdtype(v.dtype)))  # on the host until a use gives it a shape
    return cpu(np.frombuffer(buf.as_memoryview(), npdtype(v.dtype)).copy().reshape(v.shape))

  def store(self, target, value):
    """value into the buffer target: a whole one (perhaps reshaped), or a part of one (a view: the buffer's elements are
    worked on on the host, those of the part replaced, so a value read from it before keeps what it had)."""
    chain = []
    while target.op not in (Ops.BUFFER, Ops.ALLOC):
      ok = target.op in (Ops.RESHAPE, Ops.AFTER, Ops.DETACH, Ops.SHRINK, Ops.PERMUTE) or target.op is Ops.STAGE and target.arg is None
      assert ok, f"a store into {target.op}"
      chain, target = chain + [target], target.src[0]
    if target.buffer.nbytes == 0: return None
    buf = target.buffer.ensure_allocated()
    n, npd = buf.nbytes // target.dtype.itemsize, npdtype(target.dtype)
    if any(u.op in (Ops.SHRINK, Ops.PERMUTE) for u in chain):
      view = View((n,))  # the part, as a view of the buffer's elements
      for u in reversed(chain):
        if view is None or u.op in (Ops.AFTER, Ops.DETACH, Ops.STAGE): continue  # (a bare STAGE keeps its source's buffer)
        view = view.reshape(tuple(u.shape)) if u.op is Ops.RESHAPE else view.permute(u.arg) if u.op is Ops.PERMUTE else view.shrink(self.marg(u))
      assert view is not None and on_ferrite(target) and isinstance(buf._buf, Holder), "a store into part of a buffer: FERRITE's, a view"
      a = np.frombuffer(contents(buf._buf) or bytes(buf.nbytes), npd).copy()
      a[indices(view)] = np.broadcast_to(host(value, target.dtype), view.shape)
      buf._buf.set(a.tobytes())
      return None
    if isinstance(buf._buf, Part): raise NotImplementedError("a store into part of a FERRITE buffer")
    data = lambda: host(value, target.dtype)
    if not on_ferrite(target): note_host_work(); buf.copyin(memoryview(flat(data(), n, npd)))
    elif isinstance(value, Value):  # kept in whatever three axes it has, if its view is all of it as it lies, and copied if it
      t = value.t if value.whole() else materialize(value)  # is computed already: a replay writes over its tensors in place
      buf._buf.set(value=Value((t + 0 if t.op is None else t).realize(), View(value.shape), value.dtype))
    else: buf._buf.set(flat(data(), n, npd))
    return None

# tinygrad's side: the FERRITE device, its buffers, and the graphs it hands over
class Holder:  # a FERRITE buffer's contents: bytes from the host, or a chip value (its own view and dtype)
  def __init__(self): self.bytes, self.value = None, None
  def set(self, bytes=None, value=None, stored=True):  # new contents, noted while TinyJit records a call (stored: False for
    if recording is not None: recording.setdefault(id(self), [self, self.value, False])[2] |= stored  # the same laid out anew)
    if bytes is not None: note_host_work()
    self.bytes, self.value = bytes, value

def host(v, dtype):  # a chip value, host value or number as numpy, laid out as tinygrad has dtype (bf16 as its bits, a bool a byte)
  if isinstance(v, Value):
    a = download(v)
    return a != 0 if v.dtype is dtypes.bool else a
  return array(v if isinstance(v, Tensor) else Tensor(v, dtype=dtype, device="CPU"))

def contents(h):  # a holder's bytes as tinygrad lays them out, or None
  if h.value is None: return h.bytes
  note_host_work()  # a chip value brought down
  return np.ascontiguousarray(host(h.value, h.value.dtype)).tobytes()

class Part:  # part of a FERRITE buffer, as tinygrad views one at an offset: read from its base each time, never written
  value = None
  def __init__(self, base, offset, size): self.base, self.offset, self.size = base, offset, size
  @property
  def bytes(self): return None if (c := contents(self.base)) is None else c[self.offset:self.offset + self.size]

class FerriteAllocator(Allocator):
  lru = False  # a freed buffer would come back to a new one of its size with the old one's contents
  def _alloc(self, size, options): return BufferStorage(Holder())
  def _copyin(self, dest, src):
    if isinstance(dest, Part): raise NotImplementedError("a copy into part of a FERRITE buffer")
    dest.set(bytes(src))
  def _copyout(self, dest, src): dest[:] = c[:len(dest)] if (c := contents(src)) is not None else bytes(len(dest))
  def _offset(self, buf, size, offset): return Part(buf, offset, size)

class FerriteDevice(Compiled):
  def __init__(self, device): super().__init__(device, FerriteAllocator(self), [], None)
  def synchronize(self, timeout=None): pass  # reading a chip value back waits for what it needs

def inline_call(c):  # a tinygrad function as its body, storing its outputs: a precompiled one too (tinygrad would cache its
  return resolve_function(c.replace(arg=dataclasses.replace(c.arg, precompile=False)) if c.arg.precompile else c)  # program; ferrite runs the graph)
inline = PatternMatcher([(UPat(Ops.CALL, name="c"), inline_call)])
def run(sink):
  sink = graph_rewrite(sink, inline, name="inline calls")
  if allocs := {u: UOp.new_buffer(u.device, u.max_numel(), u.dtype) for u in sink.toposort() if u.op is Ops.ALLOC}:
    sink = sink.substitute(allocs)  # an inlined function's own buffers, fresh each call, as tinygrad's schedule makes them
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

def on_chip(t):  # a FERRITE tensor's holder, if it is a whole buffer, its contents on the chip (bytes go up in its shape); or None
  if t.uop.base.op is not Ops.BUFFER or not on_ferrite(t.uop.base) or t.dtype not in DTYPES: return None
  h = t.uop.base.buffer.ensure_allocated()._buf
  if not isinstance(h, Holder) or (h.value is None and h.bytes is None) or t.uop.base.buffer.nbytes != t.nbytes(): return None
  if h.value is None:
    h.set(value=Value(upload(np.frombuffer(h.bytes, npdtype(t.dtype)).reshape(t.shape), DTYPES[t.dtype]), View((t.numel(),)), t.dtype))
  return h

_tinyjit = tinygrad.engine.jit._TinyJit.__call__
def tinyjit(self, *args, **kwargs):
  """tinygrad's TinyJit on FERRITE, tensor.jit's way: the first call runs, the second is recorded, and each later one copies
  its inputs into the second's and sends its launches again as one block, running nothing on the host. Each buffer the
  recorded call stored into has its new value copied back where the recording read it from. A call runs every time
  instead if it works on the host with what it finds there (random numbers, whose counter lives there; a value brought
  down), stores into a buffer in another layout or into an input, takes an input that is not a whole buffer on the chip,
  or binds a symbolic variable (a place in a sequence: the recording has its shapes for one value); the DRAM a recording
  that cannot be replayed took stays taken. tinygrad's checks of the inputs and the result are its own
  (_prepare_jit_inputs)."""
  global recording, hosted
  given = [*args, *(v for _, v in sorted(kwargs.items()))]  # the inputs as tinygrad finds them: tensors, and those in a container
  ins = [t for t in given if t.__class__ is Tensor]
  ins += [t for x in given if isinstance(x, (tuple, list, dict)) for t in (x.values() if isinstance(x, dict) else x)
          if t.__class__ is Tensor and not any(t is u for u in ins)]
  if not JIT or Device.DEFAULT != "FERRITE" and not any(on_ferrite(t.uop) for t in ins): return _tinyjit(self, *args, **kwargs)
  _, var_vals, names, info = _prepare_jit_inputs(args, kwargs)  # tinygrad's checks of them, and the inputs realized
  holders, self.cnt = [on_chip(t) for t in ins], self.cnt + 1
  if recording is not None: raise RuntimeError("having TinyJit inside another TinyJit is not supported")
  if self.cnt == 1 or getattr(self, "replay", None) is False or None in holders or var_vals:
    ret = self.fxn(*args, **kwargs)
    if params := get_parameters(ret): Tensor.realize(*params)
    return ret
  if self.cnt == 2:
    values, recording, hosted, self.inputs = [h.value for h in holders], {}, 0, (names, info)
    try:
      with tensor.Recording(runtime()) as rec:
        ret = self.fxn(*args, **kwargs)
        if params := get_parameters(ret): Tensor.realize(*params)
        _check_no_non_tensor_return(ret)
        back = [(h, old) for h, old, _ in recording.values() if old is not None and h not in holders]  # (an input laid out
        # anew is laid out anew each replay, from its copy)
        same = lambda h, old: isinstance(h.value, Value) and (h.value.t.shape, h.value.t.dtype) == (old.t.shape, old.t.dtype)
        into_input = any(stored and h in holders for h, _, stored in recording.values())  # a replay would lose that store
        self.replay = hosted == 0 and not into_input and all(same(h, old) for h, old in back) and (rec, values, ret)
        for h, old in back if self.replay else ():
          if h.value.t is not old.t: old.t.assign(h.value.t)
          h.value = Value(old.t, h.value.view, h.value.dtype)
    finally: recording = None
    if not rec.launches: raise JitError("didn't JIT anything!")
    return ret
  rec, values, ret = self.replay
  if (names, info) != self.inputs: raise JitError(f"args mismatch in JIT: {names, info} where it recorded {self.inputs}")
  for h, v in zip(holders, values):  # each input into the recording's (a whole buffer, as it lies: reshaped if need be)
    if h.value.t is not v.t: v.t.assign(h.value.t if h.value.t.shape == v.t.shape else h.value.t.reshape(*v.t.shape))
  rec.send()
  return ret
tinygrad.engine.jit._TinyJit.__call__ = tinyjit

if __name__ == "__main__":  # a tinygrad script on ferrite, as it is
  from tinygrad.helpers import DEV
  assert len(sys.argv) > 1, "usage: python3 adapters/tinygrad_ferrite.py script.py [its arguments]"
  DEV.value = "FERRITE"
  sys.argv = sys.argv[1:]
  runpy.run_path(sys.argv[0], run_name="__main__")
