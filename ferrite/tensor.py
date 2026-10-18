# Lazy tensors: bf16, fp32 or int32, of any shape, batched or not, in DRAM as tiles (tile i in bank i % 8). Their ops
# build a tree that realizing turns into kernels (compiler.py): as much of it as one kernel holds fused into one launch,
# and what one kernel cannot hold (a matmul's operands, a move, a cast, a reduction of another shape) realized first.
import array, math
from ferrite import bh
from ferrite.compiler import ACCS, COMBINE, FOLDS, INT32, ITEMSIZE, fused_kernel, gather_kernel, move_kernel, tile_bytes, transpose_kernel

class Tensor:
  """batch x rows x cols of bf16, fp32 or int32 (dtype; batch 1 unless said): in DRAM as 32x32 tiles, matrix after matrix, each in row-major
  tile order, tile i in bank i % 8; or, until realized, an op on other tensors. Realizing compiles the whole tree of
  unrealized ops under it into one kernel; a matmul's own operands are realized first, since each of its output tiles
  needs a whole row and column of them (a transposed B is read transposed rather than realized).
  Any shape: the last tile row and column are padded, with zeros, except that a dimension of 1 fills its tile with
  copies (a row vector's tile holds the row 32 times), so broadcasting reuses the tile as it is. Every kernel leaves
  its result that way too, so a matmul can sum over a padded K.
  A view (x[b], slices on tile boundaries, x.split(n)) shares its owner's DRAM: its tile (b, i, j) is tile
  b * bstride + i * stride + j + offset. Any other movement (reshape, permute, expand, pad, other slices) is a copy,
  made by move_kernel from a map over the source's elements."""
  def __init__(self, rt, rows, cols, op=None, srcs=(), batch=1, dtype="bf16"):
    self.rt, self.rows, self.cols, self.batch, self.op, self.srcs, self.dtype = rt, rows, cols, batch, op, srcs, dtype
    tr, tc = -(-rows // 32), -(-cols // 32)  # tile rows and columns
    self.tiles, self.bstride, self.stride, self.offset, self.owner = batch * tr * tc, tr * tc, tc, 0, None
    self.addr = None if op else self._alloc()
  def _alloc(self): return self.rt.dram_alloc(-(-self.tiles // len(bh.DRAM)) * tile_bytes(self.dtype))

  def __del__(self):  # its DRAM goes back once nothing refers to it (and no queued launch can still use it)
    if self.addr is not None and self.owner is None: self.rt.dram_free(self.addr)

  def _where(self, b, i, j):  # tile (i, j) of matrix b in DRAM: (x, y, address)
    t = b * self.bstride + i * self.stride + j + self.offset
    return (*bh.DRAM[t % 8], self.addr + t // 8 * tile_bytes(self.dtype))

  @classmethod
  def full(cls, rt, rows, cols, value, dtype="bf16"): return cls.of(rt, [[value] * cols for _ in range(rows)], dtype)

  @classmethod
  def of(cls, rt, data, dtype="bf16"):
    """From a list of rows of numbers, or a list of those (a batch), rounded to dtype (bf16: to nearest even)."""
    matrices = data if isinstance(data[0][0], (list, tuple)) else [data]
    values = array.array("i" if dtype == "int32" else "f", [v for m in matrices for row in m for v in row])
    if dtype == "bf16": values = array.array("H", [(b + 0x7FFF + (b >> 16 & 1)) >> 16 for b in array.array("I", values.tobytes())])
    return cls.frombytes(rt, len(matrices[0]), len(matrices[0][0]), values, len(matrices), dtype)

  @classmethod
  def frombytes(cls, rt, rows, cols, data, batch=1, dtype="bf16"):
    """From dtype's values in row-major order (little-endian), matrix after matrix. Runs of 16 values are copied into
    tile order as they are, nothing per value, which is what makes a model's weights quick to load."""
    t, n, tr, tc, data = cls(rt, rows, cols, batch=batch, dtype=dtype), ITEMSIZE[dtype], -(-rows // 32), -(-cols // 32), memoryview(data).cast("B")
    assert len(data) == n * batch * rows * cols, f"{len(data)} bytes for {batch} x {rows} x {cols} of {dtype}"
    for b, i in ((b, i) for b in range(batch) for i in range(tr * tc)):
      m, r0, c0, tile = data[n * b * rows * cols:n * (b + 1) * rows * cols], i // tc * 32, i % tc * 32, []
      for face in range(4):  # 16 rows of 16, face by face (the tile layout, at the top)
        c = c0 + face % 2 * 16
        for r in range(r0 + face // 2 * 16, r0 + face // 2 * 16 + 16):
          r = 0 if rows == 1 else r  # a dimension of 1 fills the tile; the rest pads with 0
          at = n * (r * cols + c)
          run = b"" if r >= rows else bytes(m[n * r:n * r + n]) * 16 if cols == 1 else bytes(m[at:at + n * max(0, min(16, cols - c))])
          tile.append(run + bytes(16 * n - len(run)))
      t.rt.chip.write(*t._where(b, i // tc, i % tc), b"".join(tile))
    return t

  def tobytes(self):
    """dtype's values in row-major order, matrix after matrix, as frombytes takes them. Waits for the launches queued."""
    self.realize()
    self.rt.wait()
    n, rows, cols, ct = ITEMSIZE[self.dtype], self.rows, self.cols, -(-self.cols // 32)
    out = bytearray(n * self.batch * rows * cols)
    for b, i in ((b, i) for b in range(self.batch) for i in range(self.tiles // self.batch)):
      r0, c0, base = i // ct * 32, i % ct * 32, b * rows * cols
      tile = self.rt.chip.read(*self._where(b, i // ct, i % ct), tile_bytes(self.dtype))
      for face in range(4):
        c, k = c0 + face % 2 * 16, min(16, cols - c0 - face % 2 * 16)
        for row, r in enumerate(range(r0 + face // 2 * 16, min(r0 + face // 2 * 16 + 16, rows))):
          at = (face * 256 + row * 16) * n
          if k > 0: out[n * (base + r * cols + c):n * (base + r * cols + c + k)] = tile[at:at + n * k]
    return bytes(out)

  def tolist(self):  # rows of numbers, or a list of those if this is a batch
    raw = self.tobytes()
    if self.dtype == "bf16": raw = array.array("I", [b << 16 for b in array.array("H", raw)]).tobytes()
    values = array.array("i" if self.dtype == "int32" else "f", raw).tolist()
    m = [[values[(b * self.rows + r) * self.cols:(b * self.rows + r + 1) * self.cols] for r in range(self.rows)] for b in range(self.batch)]
    return m if self.batch > 1 else m[0]

  def gather(self, ids):
    """Rows ids of this tensor, as a tensor: an embedding lookup, by a data-movement kernel (see gather_kernel). ids is
    an int32 tensor of one row, read on the chip (so it can be computed there; nothing checks its bounds), or a list."""
    if not isinstance(ids, Tensor):
      assert all(0 <= i < self.rows for i in ids)
      ids = Tensor.of(self.rt, [list(ids)], "int32")
    assert self.batch == 1 and ids.dtype == "int32" and ids.shape[:2] == (1, 1), "rows of a matrix, by a row of int32"
    out, d, ids = Tensor(self.rt, ids.cols, self.cols, dtype=self.dtype), self.dtype, ids.realize()
    return out._run(_cached(("gather", d), lambda: gather_kernel(d)), out.addr, self.realize().addr, self.stride, self.offset,
                    -(-self.cols // 32), ids.cols, ids.addr, ids.offset)

  def _view(self, rows, cols, batch, offset, bstride=0, stride=None):
    v = Tensor(self.rt, rows, cols, "view", batch=batch, dtype=self.dtype)
    v.op, v.addr, v.owner = None, self.realize().addr, self.owner or self
    v.offset, v.bstride, v.stride = self.offset + offset, bstride, self.stride if stride is None else stride
    return v

  def __getitem__(self, key):
    """x[b], one matrix of a batch; x[rows, cols] of every matrix, or x[batch, rows, cols], with slices (any step). A
    slice of whole tiles (starting on tile boundaries, ending on one or at the end, step 1) is a view, read in place;
    any other is a copy (_move)."""
    if isinstance(key, int): return self._view(self.rows, self.cols, 1, key * self.bstride)
    assert isinstance(key, tuple) and len(key) in (2, 3) and all(isinstance(k, slice) for k in key), "x[b], x[rows, cols] or x[batch, rows, cols]"
    ranges = [range(*k.indices(n)) for k, n in zip((slice(None),) * (3 - len(key)) + tuple(key), self.shape)]
    shape = tuple(map(len, ranges))
    assert all(shape), f"an empty slice of {self.shape}"
    if all(g.step == 1 and (ax == 0 or g.start % 32 == 0 and (g.stop % 32 == 0 or g.stop == n) and (len(g) > 1 or n == 1))
           for ax, (g, n) in enumerate(zip(ranges, self.shape))):  # whole tiles; a dimension of 1 must fill its tile
      b, r, c = (g.start for g in ranges)
      return self._view(shape[1], shape[2], shape[0], b * self.bstride + r // 32 * self.stride + c // 32, self.bstride)
    st = self._strides()
    return self._move(shape, sum(g.start * s for g, s in zip(ranges, st)), tuple(g.step * s for g, s in zip(ranges, st)))

  # Movement: each is one map from the result's elements to this tensor's (_move), copied by one launch of move_kernel
  def _strides(self): return (self.rows * self.cols, self.cols, 1)  # of its elements, counted in row-major order
  def _move(self, shape, offset, strides, lo=(0, 0, 0), hi=None):
    """The tensor of shape (batch, rows, cols) whose element (b, r, c) is this one's offset + b strides[0] + r
    strides[1] + c strides[2] (counted in row-major order) inside lo..hi on each axis, and 0 outside."""
    t = Tensor(self.rt, shape[1], shape[2], "move", (self,), shape[0], self.dtype)
    t.map = (offset, *strides, *lo, *(hi or shape))
    return t
  def reshape(self, *shape):  # up to three sizes, missing ones in front being 1
    shape = (1,) * (3 - len(shape)) + shape
    assert math.prod(shape) == math.prod(self.shape), f"cannot reshape {self.shape} to {shape}"
    return self if shape == self.shape else self._move(shape, 0, (shape[1] * shape[2], shape[2], 1))
  def permute(self, *order):  # the axes in a new order; swapping rows and columns is the transpose
    assert sorted(order) == [0, 1, 2]
    if order == (0, 2, 1): return self.T
    return self if order == (0, 1, 2) else self._move(tuple(self.shape[k] for k in order), 0, tuple(self._strides()[k] for k in order))
  def expand(self, *shape):  # dimensions of 1 repeated, missing ones in front being 1
    shape = (1,) * (3 - len(shape)) + shape
    assert all(n in (m, 1) for n, m in zip(self.shape, shape)), f"cannot expand {self.shape} to {shape}"
    return self if shape == self.shape else self._move(shape, 0, tuple(s if n == m else 0 for s, n, m in zip(self._strides(), self.shape, shape)))
  def pad(self, widths):  # zeros: (before, after) for each of the last len(widths) axes, as numpy's
    widths = ((0, 0),) * (3 - len(widths)) + tuple(widths)
    lo, shape = tuple(w[0] for w in widths), tuple(n + w[0] + w[1] for n, w in zip(self.shape, widths))
    return self._move(shape, -sum(w * s for w, s in zip(lo, self._strides())), self._strides(), lo, tuple(w + n for w, n in zip(lo, self.shape)))

  def split(self, n, axis=1):
    """The matrix cut into n equal parts along axis (1: columns, 0: rows) as a batch of n, in place: attention's heads.
    The parts are whole tiles."""
    assert self.batch == 1
    if axis: assert self.cols % (32 * n) == 0; return self._view(self.rows, self.cols // n, n, 0, self.cols // n // 32)
    assert self.rows % (32 * n) == 0
    return self._view(self.rows // n, self.cols, n, 0, self.rows // n // 32 * self.stride)

  def merge(self):
    """The batch side by side as one matrix, the inverse of split: x.split(n).merge() is x. The kernel computing it
    writes each matrix into its columns (realize into a view); one already computed is copied there."""
    out = Tensor(self.rt, self.rows, self.batch * self.cols, dtype=self.dtype)
    (self if self.op not in (None, "T", "move") else self + 0.0).realize(out.split(self.batch))  # nor can these write a view
    return out

  @property
  def T(self):
    """The transpose (of each matrix of a batch). A matmul reads it as its B straight from this tensor, transposing
    tiles as they come in; anything else realizes it in a kernel of its own (transpose_kernel). The unpacker transposes
    only on its way into SrcA, so fp32 moves instead."""
    if self.dtype == "fp32": return self._move((self.batch, self.cols, self.rows), 0, (self.rows * self.cols, 1, self.cols))
    return Tensor(self.rt, self.cols, self.rows, "T", (self,), self.batch)

  def realize(self, into=None):
    """Compute it now, into DRAM of its own, or into the view `into` (see merge). Returns it, now realized."""
    if self.op is None: return self
    if self.op in ("T", "move"):  # data movement, into a tensor of its own
      assert into is None, "a transpose or a move writes a tensor of its own"
      x, op, self.op, self.srcs, d = self.srcs[0].realize(), self.op, None, (), self.dtype
      self.addr = self._alloc()
      where = (self.addr, x.addr, x.bstride if x.batch > 1 else 0, x.stride, x.offset)
      if op == "T": return self._run(_cached(("transpose", x.rows, x.cols), lambda: transpose_kernel(x.rows, x.cols)), *where)
      return self._run(_cached(("move", d), lambda: move_kernel(d)), *where, x.rows, x.cols, self.rows, self.cols,
                       *(v & 0xFFFFFFFF for v in self.map))
    if any(t.op in COMBINE for t in _lazy(self, set())):  # rows are swept in passes, each recomputing what it reads:
      for t in [t for t in _lazy(self, set()) if t.op == "matmul" and t.srcs[0].cols > 256]: t.realize()  # long matmuls once
      # and an elementwise value that a reduction and something else both read: once, on as many workers as it has tiles
      reads = [id(s) for t in _lazy(self, set()) for s in t.srcs]
      for x in [t.srcs[0] for t in _lazy(self, set()) if t.op in COMBINE]:
        if reads.count(id(x)) > 1 and all(t.op not in COMBINE and t.op != "matmul" for t in _lazy(x, set())): x.realize()
    while _ops(self, set()) > 15:  # too big for one kernel (SEM_L1 counts 15): down the heaviest branch, the first part
      t = self                    # that fits goes first, on its own, so what is on top (a layer norm, say) stays whole
      while _ops(t, set()) > 15: t = max((s for s in t.srcs if s.op), key=lambda s: _ops(s, set()))
      t.realize()
    inputs = []
    e, dtype = _tree(self, inputs)
    key = (e, tuple(u.shape for u in inputs), dtype, self.dtype)
    kernel, units = _cached(key, lambda: fused_kernel(*key))
    if into: self.addr, self.owner, self.bstride, self.stride, self.offset = into.addr, into.owner or into, into.bstride, into.stride, into.offset
    else: self.addr = self._alloc()
    self.op, self.srcs = None, ()
    args = [v for t in [self] + inputs for v in (t.addr, t.bstride if t.batch > 1 else 0, t.stride, t.offset)]
    return self._run(kernel, *args, units=units)

  def _run(self, kernel, *args, units=None):
    """Queue kernel over units of this tensor (tiles unless told), split evenly over as many workers as have any. Nothing waits for
    it: launches run in order, so the next kernel reads this result, and reading it back (tobytes) waits."""
    self.rt.launch(kernel, args, units or self.tiles)
    return self

  @property
  def shape(self): return (self.batch, self.rows, self.cols)

  # Elementwise: tinygrad's primitives (exp2, log2, sin, sqrt, reciprocal; add, mul, max, cmplt, cmpne; where), and the
  # rest built from them. exp, log and cos are SFPU programs too, with log2(e), ln 2 and the quarter turn kept in fp32
  # inside: as numbers they would be bf16 tensors, and exp(10) as exp2(10 * log2(e)) would be 2% off.
  def _unary(self, op):  # int32 goes through fp32, but for relu (a max with 0)
    if self.dtype == "int32" and op != "relu": return self.float()._unary(op)
    return Tensor(self.rt, self.rows, self.cols, op, (self,), self.batch, self.dtype)
  def _binop(self, op, x, reverse=False):
    """numpy's broadcasting: shapes match, or one side has 1 where the other does not. A number is a 1 x 1 tensor of
    this one's dtype (fp32 for a float with int32). Different dtypes meet in fp32, and so does int32 for what it does
    not do itself (mul: exact below 2^24)."""
    if not isinstance(x, Tensor): x = Tensor.full(self.rt, 1, 1, x, "fp32" if self.dtype == "int32" and isinstance(x, float) else self.dtype)
    a, b = (x, self) if reverse else (self, x)
    if a.dtype != b.dtype or (a.dtype == "int32" and op not in INT32): a, b = a.float(), b.float()
    assert all(n == m or 1 in (n, m) for n, m in zip(a.shape, b.shape)), f"cannot broadcast {a.shape} with {b.shape}"
    batch, rows, cols = map(max, a.shape, b.shape)
    return Tensor(self.rt, rows, cols, op, (a, b), batch, a.dtype)
  def cast(self, dtype): return self if dtype == self.dtype else Tensor(self.rt, self.rows, self.cols, "cast", (self,), self.batch, dtype)
  def float(self): return self.cast("fp32")
  def bfloat16(self): return self.cast("bf16")
  def exp2(self): return self._unary("exp2")
  def log2(self): return self._unary("log2")
  def sin(self): return self._unary("sin")
  def sqrt(self): return self._unary("sqrt")
  def reciprocal(self): return self._unary("recip")
  def exp(self): return self._unary("exp")
  def log(self): return self._unary("log")
  def cos(self): return self._unary("cos")
  def relu(self): return self._unary("relu")
  def rsqrt(self): return self.sqrt().reciprocal()
  def maximum(self, x): return self._binop("max", x)
  def minimum(self, x): return -((-self).maximum(-x))
  def where(self, x, y):  # x where self is not 0, else y; the condition, if of another dtype, as 0 or 1 of theirs
    dtype = next((t.dtype for t in (x, y) if isinstance(t, Tensor)), self.dtype)
    c = self if self.dtype == dtype else self.ne(0).cast(dtype)
    return c._binop("keep", x) + c._binop("drop", y)
  def __add__(self, x): return self._binop("add", x)
  def __sub__(self, x): return self._binop("sub", x)
  def __mul__(self, x): return self._binop("mul", x)
  def __rsub__(self, x): return self._binop("sub", x, True)
  __radd__, __rmul__ = __add__, __mul__
  def __truediv__(self, x): return self * (1 / x if isinstance(x, (int, float)) else x.reciprocal())
  def __neg__(self): return 0 - self if self.dtype == "int32" else self * -1  # int32 subtracts exactly, multiplies through fp32
  def __lt__(self, x): return self._binop("cmplt", x)
  def __gt__(self, x): return self._binop("cmplt", x, True)
  def __ge__(self, x): return 1 - (self < x)
  def __le__(self, x): return 1 - (self > x)
  def ne(self, x): return self._binop("cmpne", x)
  def eq(self, x): return 1 - self.ne(x)
  __ne__, __eq__ = ne, eq
  def __hash__(self): return id(self)  # == is elementwise, so a tensor hashes by identity, as in tinygrad
  def gelu(self):  # GPT-2's: 0.5 x (1 + tanh(z)), z = sqrt(2/pi) (x + 0.044715 x^3), which is x / (1 + e^-2z)
    k = 2 * math.sqrt(2 / math.pi)
    return self * (1 + (self * (self * self * (-0.044715 * k) - k)).exp()).reciprocal()

  # Reductions, and what is built from them
  def sum(self, axis=None): return self._reduce("sum", axis)
  def max(self, axis=None): return self._reduce("max", axis)
  def _reduce(self, f, axis):
    """numpy's, keeping the axis as 1: over the batch (0, one matrix), down the rows (1, one row) or along each row
    (2, one column); over all three if None. Each axis is a kernel of its own, sweeping lines of tiles along it."""
    if self.dtype == "int32": return self.float()._reduce(f, axis).cast("int32")  # exact below 2^24
    if axis is None: return self._reduce(f, 0)._reduce(f, 1)._reduce(f, 2)
    shape = [1 if ax == axis % 3 else s for ax, s in enumerate(self.shape)]
    return self if shape == list(self.shape) else Tensor(self.rt, shape[1], shape[2], f"{f}{axis % 3}", (self,), shape[0], self.dtype)
  def mean(self, axis=None):
    s = self.sum(axis)
    return s * (math.prod(s.shape) / math.prod(self.shape))
  def softmax(self, axis=-1):  # max first, so exp stays in range
    e = (self - self.max(axis)).exp()
    return e / e.sum(axis)
  def layernorm(self, axis=-1, eps=1e-5):  # to mean 0 and variance 1 along axis
    c = self - self.mean(axis)
    return c * ((c * c).mean(axis) + eps).rsqrt()

  def __matmul__(self, x): return self.matmul(x)
  def matmul(self, x, dtype=None):
    """Matrix by matrix through a batch (a batch of 1 repeats), summed in fp32 in Dst and written as dtype: bf16, or fp32
    if either operand is, unless said. fp32 operands go as bf16x3: each is hi + lo, both bf16, and hi@hi + hi@lo + lo@hi
    sum in fp32 (lo@lo, 2^-16 of the rest, is left out), about 16 bits of each operand where fp32 has 24."""
    assert "int32" not in (self.dtype, x.dtype), "the FPU multiplies floats"
    dtype = dtype or ("fp32" if "fp32" in (self.dtype, x.dtype) else "bf16")
    if "fp32" in (self.dtype, x.dtype):
      (ah, al), (bh, bl) = ((t.bfloat16(), (t.float() - t.bfloat16()).bfloat16()) for t in (self, x))
      return (ah.matmul(bh, "fp32") + (ah.matmul(bl, "fp32") + al.matmul(bh, "fp32"))).cast(dtype)
    assert self.cols == x.rows and (self.batch == x.batch or 1 in (self.batch, x.batch)), f"{self.shape} @ {x.shape}"
    if self.cols == 1: return (self * x).cast(dtype)  # (R, 1) @ (1, C) is broadcasting
    kt, tiles = self.cols // 32, -(-self.rows // 32) * -(-x.cols // 32)
    s = max([s for s in (2, 4, 8) if self.batch == x.batch == 1 and self.cols % (32 * s) == 0 and kt // s >= 4 and
             s * tiles <= len(self.rt.cores)], default=1)
    if s > 1:  # too few result tiles to keep the workers busy (a token's MLP): K cut in s, a batch, and the parts summed
      parts = self.split(s).matmul(x.split(s, 0), dtype)
      return sum((parts[i] for i in range(1, s)), parts[0])
    return Tensor(self.rt, self.rows, x.cols, "matmul", (self, x), max(self.batch, x.batch), dtype)

WHOLE = ("matmul", "T", "move", "cast")  # ops whose operands are whole tensors, realized first, not tiles in the same kernel
def _lazy(t, seen):  # the unrealized ops under t, t first
  if t.op is None or id(t) in seen: return
  seen.add(id(t))
  yield t
  for s in () if t.op in WHOLE else t.srcs: yield from _lazy(s, seen)

def _ops(t, seen):  # how many ops a kernel for t would hold, at most (an SFPU op on another's result adds none)
  if t.op is None or id(t) in seen: return 0
  seen.add(id(t))
  return 1 + (0 if t.op in WHOLE else sum(_ops(s, seen) for s in t.srcs))

def _tree(t, inputs):
  """The expression a kernel for t computes (fused_kernel's), realized tensors as numbers into inputs, and the dtype it
  computes in: t's, or for a cast or a matmul its operands'. The kernel sweeps along the axis of t's reductions (the
  first found; any along another axis compute first, on their own)."""
  dtype = t.srcs[0].dtype if t.op in ("cast", "matmul") else t.dtype
  axis = next((FOLDS[u.op] for u in _lazy(t, set()) if u.op in COMBINE), 2)
  n = _ext(t, axis)  # the sweep is as long as t along the axis, unless t is one tile there: then its first reduction says
  return _expr(t, t.shape, inputs, [axis, n if n > 1 else None, set(), dtype, t]), dtype

def _ext(t, ax): return t.shape[ax] if ax == 0 else -(-t.shape[ax] // 32)  # t's extent along an axis: matrices, or tiles

def _expr(t, shape, inputs, kernel):
  """The tree of unrealized ops under t, for a kernel of shape (t's). kernel: [the axis it sweeps, the sweep's length
  (in tiles; the first seen, if t is one tile along it), the reductions it holds (two accumulators), the dtype it
  computes in, its root]. Computed first, on their own: an op of another dtype, a cast, a matmul inside an fp32 kernel, an
  op of 1 along an axis the kernel keeps (repeated there: compute it once), and a reduction along another axis, of
  another length, or past the second."""
  if t.op is None:
    assert t.dtype == kernel[3], f"a {t.dtype} input to a {kernel[3]} kernel"
    if all(t is not u for u in inputs): inputs.append(t)
    return next(i for i, u in enumerate(inputs) if u is t)
  if t is kernel[4] and t.op == "cast": return ("cast", _expr(t.srcs[0].realize(), shape, inputs, kernel))
  if t is not kernel[4] and (t.dtype != kernel[3] or t.op == "cast" or (t.op == "matmul" and kernel[3] != "bf16")):
    return _expr(t.realize(), shape, inputs, kernel)
  axis = kernel[0]
  if any(t.shape[ax] != shape[ax] for ax in range(3) if ax != axis): return _expr(t.realize(), shape, inputs, kernel)
  if t.op in COMBINE:
    if FOLDS[t.op] != axis: return _expr(t.realize(), shape, inputs, kernel)
    n = _ext(t.srcs[0], axis)
    if n > 1 and kernel[1] is None: kernel[1] = n
    if (n > 1 and n != kernel[1]) or (id(t) not in kernel[2] and len(kernel[2]) == len(ACCS)): return _expr(t.realize(), shape, inputs, kernel)
    kernel[2].add(id(t))
  if t.op in ("T", "move"): return _expr(t.realize(), shape, inputs, kernel)  # kernels of their own (a matmul's B may be read transposed, below)
  if t.op == "matmul":  # its operands are whole matrices, not tiles: realize them on their own first
    a, b = t.srcs
    b = ("T", _expr(b.srcs[0].realize(), shape, inputs, kernel)) if b.op == "T" else _expr(b.realize(), shape, inputs, kernel)
    return ("matmul", _expr(a.realize(), shape, inputs, kernel), b)
  return (t.op, *(_expr(s, shape, inputs, kernel) for s in t.srcs))

kernels = {}  # (expression, input shapes) -> (image, units a matrix), or a kernel's name and shape -> image
def _cached(key, compile):
  if key not in kernels: kernels[key] = compile()
  return kernels[key]
