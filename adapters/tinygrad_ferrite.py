# ferrite as a tinygrad device: tinygrad is the frontend (Tensor, nn, autograd, optimizers), ferrite the compiler and the
# runtime. Realizing a FERRITE tensor hands tinygrad's tensor graph (its primitives on shaped tensors, before tinygrad
# would schedule them into kernels) to ferrite op by op, and ferrite fuses and launches them its own way.
# tinygrad's tensors have any rank and ferrite's three axes, so movement stays a view of a ferrite tensor (views.py),
# folded into three axes when an op needs the data. A
# multiply summed over axes both operands span, broadcast against each other elsewhere, is a matmul. What ferrite does
# not do (the integer arithmetic of random numbers, other dtypes) runs on tinygrad's own CPU device and comes over.
#   import adapters.tinygrad_ferrite, then DEV=FERRITE or Tensor(..., device="FERRITE")
#   TTSIM=/path/to/libttsim.so python3 adapters/tinygrad_ferrite.py script.py [args]   # a tinygrad script, unchanged
import collections, math, os, runpy, sys, types
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import tinygrad, tinygrad.tensor
from tinygrad import Tensor, dtypes
from tinygrad.dtype import _to_np_dtype
from tinygrad.device import Allocator, BufferStorage, Compiled, Device
from tinygrad.uop.ops import Ops, PatternMatcher, UOp, UPat, graph_rewrite
from tinygrad.schedule.prepare import resolve_function
from views import View, Value, broadcast, common_groups, download, fold, grouping, materialize, merge, operand, reduce, spread, upload
from ferrite import tensor  # after views, which puts ferrite on the path

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

def to_chip(v, dtype):  # a host value (a tinygrad CPU tensor) as a chip value, in its own shape
  return Value(upload(array(v), DTYPES[dtype]), View(v.shape), dtype)

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
      return upload(array(x), "fp32" if dtypes.is_float(x.dtype) else "int32", [math.prod(shape[k] for k in g) for g in groups])
    return operand(x, shape, groups)

  def chip_op(self, u, srcs):
    """u on the chip, or None if ferrite does not do it."""
    if u.op in ELEMENTWISE and (u.dtype not in DTYPES or any(isinstance(s, Value) and s.dtype not in DTYPES for s in srcs)): return None
    shape = tuple(u.shape)
    groups = common_groups(shape, srcs)
    whole = lambda t: Value(spread(t, shape, groups), View(shape), u.dtype)
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

  def reduce(self, u, x):  # REDUCE(op, n) folds x's first n axes
    return Value(reduce(x, list(range(u.arg[1])), "sum" if u.arg[0] is Ops.ADD else "max"), View(tuple(u.shape)), u.dtype)

  def matmul(self, u):
    """The REDUCE(ADD) u as a matmul, or None: its operand a MUL (behind a PERMUTE) of a and b, which do not broadcast
    along the summed axes; along the others both span the batch, a alone the rows and b alone the columns."""
    p = u.src[0]
    order = p.arg if p.op is Ops.PERMUTE else tuple(range(len(p.shape)))
    m = p.src[0] if p.op is Ops.PERMUTE else p
    if m.op is not Ops.MUL: return None
    up = lambda s, v: to_chip(v, s.dtype) if isinstance(v, Tensor) and constant(v) is None else v  # a weight still on the host goes up
    a, b = (spread_to(up(s, self.read(s, self.values.get(s))), m.shape) for s in m.src)
    if not (isinstance(a, Value) and isinstance(b, Value)): return None
    if a.view.mask or b.view.mask or a.dtype not in DTYPES or b.dtype not in DTYPES: return None
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
    if isinstance(x, Tensor) and constant(x) is None and u.dtype in DTYPES and h.value is None:
      h.value = Value(upload(array(x).reshape(u.shape), DTYPES[u.dtype]), View(x.shape), u.dtype)
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
  lru = False  # a freed buffer would come back to a new one of its size with the old one's contents
  def _alloc(self, size, options): return BufferStorage(Holder())
  def _copyin(self, dest, src): dest.bytes, dest.value = bytes(src), None
  def _copyout(self, dest, src):
    if src.value is not None: dest[:] = np.ascontiguousarray(download(src.value)).tobytes()[:len(dest)]
    else: dest[:] = src.bytes[:len(dest)] if src.bytes is not None else bytes(len(dest))

class FerriteDevice(Compiled):
  def __init__(self, device): super().__init__(device, FerriteAllocator(self), [], None)
  def synchronize(self, timeout=None): pass  # reading a chip value back waits for what it needs

inline = PatternMatcher([(UPat(Ops.CALL, name="c"), resolve_function)])  # a tinygrad function as its body, storing its outputs
def run(sink):
  sink = graph_rewrite(sink, inline, name="inline calls")
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
  assert len(sys.argv) > 1, "usage: python3 adapters/tinygrad_ferrite.py script.py [its arguments]"
  DEV.value, JIT.value = "FERRITE", 0
  sys.argv = sys.argv[1:]
  runpy.run_path(sys.argv[0], run_name="__main__")
