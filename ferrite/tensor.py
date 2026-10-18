# Lazy tensors: bf16, fp32 or int32, of any shape, batched or not, in DRAM as tiles (tile i in bank i % 8). Their ops
# build a tree that realizing turns into kernels (compiler.py): as much of it as one kernel holds fused into one launch,
# and what one kernel cannot hold (a matmul's operands, a move, a cast, a reduction of another shape) realized first.
import array, math
from ferrite import bh
from ferrite.compiler import ACCS, COMBINE, FOLDS, INT32, ITEMSIZE, fused_kernel, gather_kernel, move_kernel, tile_bytes, transpose_kernel

FIDELITY = 4  # fidelity phases a bf16 matmul takes unless told: 4 makes exact products (compiler.matmul says what fewer leave out)

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
  made by move_kernel from a map over the source's elements; a chain of them is one copy.
  A tensor made from one that requires_grad remembers how (_ctx: op, operands, arguments), past realizing, for
  backward (see gradient)."""
  def __init__(self, rt, rows, cols, op=None, srcs=(), batch=1, dtype="bf16"):
    self.rt, self.rows, self.cols, self.batch, self.op, self.srcs, self.dtype = rt, rows, cols, batch, op, srcs, dtype
    tr, tc = -(-rows // 32), -(-cols // 32)  # tile rows and columns
    self.tiles, self.bstride, self.stride, self.offset, self.owner = batch * tr * tc, tr * tc, tc, 0, None
    self.addr = None if op else self._alloc()
    self.requires_grad, self.grad, self._ctx = False, None, None
    if op not in (None, "move", "view"): self._track(op, srcs)  # moves and views say what they are themselves
  def _track(self, op, srcs, *args):  # remember how this was made, if anything it came from needs a gradient
    if any(s.requires_grad for s in srcs): self.requires_grad, self._ctx = True, (op, srcs, args)
    return self
  def requires_grad_(self, on=True):
    self.requires_grad = on
    return self
  def _alloc(self): return self.rt.dram_alloc(-(-self.tiles // len(bh.DRAM)) * tile_bytes(self.dtype))

  def __del__(self):  # its DRAM goes back once nothing refers to it (and no queued launch can still use it)
    if self.addr is not None and self.owner is None: self.rt.dram_free(self.addr)

  def _where(self, b, i, j):  # tile (i, j) of matrix b in DRAM: (x, y, address)
    t = b * self.bstride + i * self.stride + j + self.offset
    return (*bh.DRAM[t % len(bh.DRAM)], self.addr + t // len(bh.DRAM) * tile_bytes(self.dtype))

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
                    -(-self.cols // 32), ids.cols, ids.addr, ids.offset)._track("gather", (self,), ids)

  def _view(self, rows, cols, batch, offset, bstride=0, stride=None):
    v = Tensor(self.rt, rows, cols, "view", batch=batch, dtype=self.dtype)
    v.op, v.addr, v.owner = None, self.realize().addr, self.owner or self
    v.offset, v.bstride, v.stride = self.offset + offset, bstride, self.stride if stride is None else stride
    return v

  def __getitem__(self, key):
    """x[b], one matrix of a batch; x[rows, cols] of every matrix, or x[batch, rows, cols], with slices (any step). A
    slice of whole tiles (starting on tile boundaries, ending on one or at the end, step 1) is a view, read in place;
    any other is a copy (_move)."""
    if isinstance(key, int): return self._view(self.rows, self.cols, 1, key * self.bstride)._track("index", (self,), key)
    assert isinstance(key, tuple) and len(key) in (2, 3) and all(isinstance(k, slice) for k in key), "x[b], x[rows, cols] or x[batch, rows, cols]"
    ranges = [range(*k.indices(n)) for k, n in zip((slice(None),) * (3 - len(key)) + tuple(key), self.shape)]
    shape = tuple(map(len, ranges))
    assert all(shape), f"an empty slice of {self.shape}"
    if all(g.step == 1 and (ax == 0 or g.start % 32 == 0 and (g.stop % 32 == 0 or g.stop == n) and (len(g) > 1 or n == 1))
           for ax, (g, n) in enumerate(zip(ranges, self.shape))):  # whole tiles; a dimension of 1 must fill its tile
      b, r, c = (g.start for g in ranges)
      t = self._view(shape[1], shape[2], shape[0], b * self.bstride + r // 32 * self.stride + c // 32, self.bstride)
    else:
      t = self._map(shape, (0, 1, 2), [g.start for g in ranges], [g.step for g in ranges])
    return t._track("slice", (self,), ranges)

  # Movement: each is one map from the result's elements to this tensor's (_move), copied by one launch of move_kernel
  def _move(self, shape, offset, strides, lo=(0, 0, 0), hi=None):
    """The tensor of shape (batch, rows, cols) whose element (b, r, c) is this one's offset + b strides[0] + r
    strides[1] + c strides[2] (counted in row-major order) inside lo..hi on each axis, and 0 outside."""
    t = Tensor(self.rt, shape[1], shape[2], "move", (self,), shape[0], self.dtype)
    t.map = (offset, *strides, *lo, *(hi or shape))
    return t
  def _map(self, shape, axes, starts, steps):
    """The tensor of shape (batch, rows, cols) whose element e along each axis a is this one's element starts[a] +
    steps[a] e along its axis axes[a], and 0 where that is outside it (permute, expand, pad, slices). Of a move not yet
    made, it is one move: the two maps composed, as tinygrad composes views, so a chain of these is one launch."""
    if self.op == "move": x, offset, strides, lo, hi = self.srcs[0], self.map[0], self.map[1:4], self.map[4:7], self.map[7:]
    else: x, offset, strides, lo, hi = self, 0, (self.rows * self.cols, self.cols, 1), (0, 0, 0), self.shape
    inside = [_inside(o, s, lo[k], hi[k], n) for k, o, s, n in zip(axes, starts, steps, shape)]
    return x._move(shape, offset + sum(strides[k] * o for k, o in zip(axes, starts)), [strides[k] * s for k, s in zip(axes, steps)], *zip(*inside))
  def reshape(self, *shape):  # up to three sizes, missing ones in front being 1
    shape = (1,) * (3 - len(shape)) + shape
    assert math.prod(shape) == math.prod(self.shape), f"cannot reshape {self.shape} to {shape}"
    return self if shape == self.shape else self._move(shape, 0, (shape[1] * shape[2], shape[2], 1))._track("reshape", (self,))
  def permute(self, *order):  # the axes in a new order; swapping rows and columns is the transpose
    assert sorted(order) == [0, 1, 2]
    if order == (0, 2, 1): return self.T
    if order == (0, 1, 2): return self
    return self._map(tuple(self.shape[k] for k in order), order, (0, 0, 0), (1, 1, 1))._track("permute", (self,), order)
  def expand(self, *shape):  # dimensions of 1 repeated, missing ones in front being 1
    shape = (1,) * (3 - len(shape)) + shape
    assert all(n in (m, 1) for n, m in zip(self.shape, shape)), f"cannot expand {self.shape} to {shape}"
    if shape == self.shape: return self
    return self._map(shape, (0, 1, 2), (0, 0, 0), [int(n == m) for n, m in zip(self.shape, shape)])._track("expand", (self,))
  def pad(self, widths):  # zeros: (before, after) for each of the last len(widths) axes, as numpy's
    widths = ((0, 0),) * (3 - len(widths)) + tuple(widths)
    shape = tuple(n + w[0] + w[1] for n, w in zip(self.shape, widths))
    return self._map(shape, (0, 1, 2), [-w[0] for w in widths], (1, 1, 1))._track("pad", (self,), widths)

  def cat(self, *xs, axis=0):  # along axis, the rest the same: each padded with zeros into its place, and summed (tinygrad's way)
    ts, axis = (self, *xs), axis % 3
    n, out, at = sum(t.shape[axis] for t in ts), None, 0
    for t in ts:
      widths = [(0, 0)] * 3
      widths[axis] = (at, n - at - t.shape[axis])
      out, at = t.pad(widths) if out is None else out + t.pad(widths), at + t.shape[axis]
    return out

  def split(self, n, axis=1):
    """The matrix cut into n equal parts along axis (1: columns, 0: rows) as a batch of n, in place: attention's heads.
    The parts are whole tiles."""
    assert self.batch == 1
    if axis: assert self.cols % (32 * n) == 0; return self._view(self.rows, self.cols // n, n, 0, self.cols // n // 32)._track("split", (self,), axis)
    assert self.rows % (32 * n) == 0
    return self._view(self.rows // n, self.cols, n, 0, self.rows // n // 32 * self.stride)._track("split", (self,), axis)

  def merge(self):
    """The batch side by side as one matrix, the inverse of split: x.split(n).merge() is x. The kernel computing it
    writes each matrix into its columns (realize into a view); one already computed is copied there."""
    out = Tensor(self.rt, self.rows, self.batch * self.cols, dtype=self.dtype)
    (self if self.op not in (None, "T", "move") else self + 0).realize(out.split(self.batch))  # computed already, or a move: copied in by + 0
    return out._track("merge", (self,))

  @property
  def T(self):
    """The transpose (of each matrix of a batch). A matmul reads it as its B straight from this tensor, transposing
    tiles as they come in; anything else realizes it in a kernel of its own (transpose_kernel). The unpacker transposes
    only on its way into SrcA, so fp32 and int32 move instead (an fp32 matmul transposes its bf16 parts)."""
    return Tensor(self.rt, self.cols, self.rows, "T", (self,), self.batch, self.dtype)

  def realize(self, into=None):
    """Compute it now, into DRAM of its own, or into the view `into` (see merge). Returns it, now realized."""
    if self.op is None: return self
    if self.op in ("T", "move"):  # data movement, into a tensor of its own
      assert into is None, "a transpose or a move writes a tensor of its own"
      x, op, self.op, self.srcs, d = self.srcs[0].realize(), self.op, None, (), self.dtype
      self.addr = self._alloc()
      where = (self.addr, x.addr, x.bstride if x.batch > 1 else 0, x.stride, x.offset)
      if op == "T" and d == "bf16": return self._run(_cached(("transpose", x.rows, x.cols), lambda: transpose_kernel(x.rows, x.cols)), *where)
      if op == "T": self.map = (0, x.rows * x.cols, 1, x.cols, 0, 0, 0, *self.shape)
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
    key = (e, tuple(u.shape for u in inputs), dtype, self.dtype, self.rt.grid)
    kernel, units = _cached(key, lambda: fused_kernel(*key))
    if into: self.addr, self.owner, self.bstride, self.stride, self.offset = into.addr, into.owner or into, into.bstride, into.stride, into.offset
    else: self.addr = self._alloc()
    self.op, self.srcs = None, ()
    args = [v for t in [self] + inputs for v in (t.addr, t.bstride if t.batch > 1 else 0, t.stride, t.offset)]
    return self._run(kernel, *args, units=units)
  def assign(self, x):
    """x's value in this tensor, which stays the same object, as an optimizer updates a weight: x is computed into DRAM
    of its own, which this tensor takes (its old DRAM going back), and x becomes a view of it."""
    assert self.op is None and self.owner is None and (x.shape, x.dtype) == (self.shape, self.dtype), "a whole tensor, its own shape and dtype"
    x = (x if x.op else x + 0).realize()  # a copy of a tensor already computed, which keeps its own DRAM
    self.rt.dram_free(self.addr)
    self.addr, x.owner = x.addr, self
    return self

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
  def sigmoid(self): return (1 + (-self).exp()).reciprocal()
  def silu(self): return self * self.sigmoid()
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
  def log_softmax(self, axis=-1):
    m = self - self.max(axis)
    return m - m.exp().sum(axis).log()
  def layernorm(self, axis=-1, eps=1e-5):  # to mean 0 and variance 1 along axis
    c = self - self.mean(axis)
    return c * ((c * c).mean(axis) + eps).rsqrt()

  def __matmul__(self, x): return self.matmul(x)
  def matmul(self, x, dtype=None, fidelity=None):
    """Matrix by matrix through a batch (a batch of 1 repeats), summed in fp32 in Dst and written as dtype: bf16, or fp32
    if either operand is, unless said. fp32 operands go as bf16x3: each is hi + lo, both bf16, and hi@hi + hi@lo + lo@hi
    sum in fp32 (lo@lo, 2^-16 of the rest, is left out), about 16 bits of each operand where fp32 has 24. fidelity: the
    phases a bf16 product takes (FIDELITY unless said; compiler.matmul says what each leaves out); fp32's take all four."""
    fidelity = fidelity or FIDELITY
    assert "int32" not in (self.dtype, x.dtype), "the FPU multiplies floats"
    dtype = dtype or ("fp32" if "fp32" in (self.dtype, x.dtype) else "bf16")
    if "fp32" in (self.dtype, x.dtype):  # one op to backward, which is two more fp32 matmuls, not through its bf16 parts
      (ah, al), (bh, bl) = _parts(self), _parts(x)
      out, lo = ah.matmul(bh, "fp32", 4), [a.matmul(b, "fp32", 4) for a, b in ((ah, bl), (al, bh)) if a is not None and b is not None]
      return (out + sum(lo[1:], lo[0]) if lo else out)._track("matmul", (self, x)).cast(dtype)
    assert self.cols == x.rows and (self.batch == x.batch or 1 in (self.batch, x.batch)), f"{self.shape} @ {x.shape}"
    if self.cols == 1: return (self * x).cast(dtype)  # (R, 1) @ (1, C) is broadcasting
    kt, tiles = self.cols // 32, -(-self.rows // 32) * -(-x.cols // 32)
    s = max([s for s in (2, 4, 8) if self.batch == x.batch == 1 and self.cols % (32 * s) == 0 and kt // s >= 4 and
             s * tiles <= len(self.rt.cores)], default=1)
    if s > 1:  # too few result tiles to keep the workers busy (a token's MLP): K cut in s, a batch, and the parts summed
      parts = self.split(s).matmul(x.split(s, 0), dtype, fidelity)
      return sum((parts[i] for i in range(1, s)), parts[0])
    out = Tensor(self.rt, self.rows, x.cols, "matmul", (self, x), max(self.batch, x.batch), dtype)
    out.fidelity = fidelity
    return out

  # Convolution and pooling of an image (channels, height, width), built from the ops above, so autograd has them too
  def conv2d(self, w, padding=0):
    """By w (out channels, in channels, k * k): a k x k kernel, im2col's way, as one matmul. With each channel flattened
    into a row, tap (ky, kx) of the kernel reads those rows from ky W + kx on; the k * k taps stacked are the patches,
    column y W + x the patch at (y, x) (those with x past the last wrap around, and are dropped from the result). Each
    tap is one move: the flattening, its slice and its place in the stack composed."""
    k = math.isqrt(w.cols)
    assert k * k == w.cols and w.rows == self.batch, f"{self.shape} by {w.shape}: (out, in, k * k)"
    x = self.pad(((padding, padding), (padding, padding))) if padding else self
    (c, h, wd), ho, wo = x.shape, x.rows - k + 1, x.cols - k + 1
    rows = x.reshape(1, c, h * wd).pad(((0, 0), (0, k - 1)))  # k - 1 zeros on the end, for the last tap's wrapped columns
    taps = [rows[:, :, ky * wd + kx:ky * wd + kx + ho * wd] for ky in range(k) for kx in range(k)]
    out = w.permute(0, 2, 1).reshape(1, w.batch, k * k * c) @ taps[0].cat(*taps[1:], axis=1)
    return out.reshape(w.batch, ho, wd)[:, :, :wo]
  def max_pool2d(self, k=2):  # the largest of each k x k block: of each k rows, then of each k columns
    c, h, w = self.shape
    assert h % k == 0 and w % k == 0, f"{self.shape} in {k} x {k} blocks"
    return self.reshape(c * h // k, k, w).max(1).reshape(1, c * h * w // (k * k), k).max(2).reshape(c, h // k, w // k)

  # Autograd: tinygrad's, a rule per op (_backward) applied from the result back, each made of the same lazy ops, so
  # the backward fuses into kernels as the forward does (and recomputes what the forward fused, rather than store it)
  def gradient(self, *targets, gradient=None):
    """d self / d each target (None where self does not depend on it), lazy. self is a single value, unless gradient,
    d (something) / d self, is given."""
    assert gradient is not None or self.shape == (1, 1, 1), "the gradient of a single value, or given one"
    order, seen = [], set()
    def visit(t):  # what self was made of, operands first
      if id(t) in seen or t._ctx is None: return
      seen.add(id(t))
      for s in t._ctx[1]:
        if s.requires_grad: visit(s)
      order.append(t)
    visit(self)
    grads = {id(self): gradient if gradient is not None else Tensor.full(self.rt, 1, 1, 1.0, self.dtype)}
    for t in reversed(order):
      if (g := grads.pop(id(t), None)) is None: continue
      for s, gs in zip(t._ctx[1], _backward(t, g)):
        if gs is not None and s.requires_grad: grads[id(s)] = grads[id(s)] + gs if id(s) in grads else gs
    return [_full(grads[id(t)], t.shape) if id(t) in grads else None for t in targets]
  def backward(self, gradient=None):
    """The gradient of self into .grad of every tensor it came from that requires_grad and is not made of others (the
    leaves), added to what is there, as PyTorch and tinygrad do. Returns self."""
    leaves, seen = [], set()
    def visit(t):
      if id(t) in seen: return
      seen.add(id(t))
      if t._ctx is None: leaves.append(t)
      for s in t._ctx[1] if t._ctx else ():
        if s.requires_grad: visit(s)
    visit(self)
    for t, g in zip(leaves, self.gradient(*leaves, gradient=gradient)): t.grad = g if t.grad is None else t.grad + g
    return self

def _unbroadcast(g, shape):  # g summed over the axes where shape is 1 and g is not: the gradient of what was broadcast
  for ax in range(3):
    if shape[ax] == 1 and g.shape[ax] > 1: g = g.sum(ax)
  return g
def _full(g, shape): return g if g.shape == shape else g.expand(*shape)  # for what needs every element, not broadcasting

def _backward(t, g):
  """The gradients of t's operands, given g, t's: None for one with none (a condition, a comparison)."""
  op, srcs, args = t._ctx
  x, y = (srcs + (None,))[:2]
  if op in ("add", "sub", "mul", "max", "keep", "drop", "cmplt", "cmpne"):
    if op in ("cmplt", "cmpne"): return (None, None)
    if op in ("keep", "drop"): return (None, _unbroadcast(x._binop(op, g), y.shape))  # where's halves: the condition has none
    if op == "add": gx, gy = g, g
    elif op == "sub": gx, gy = g, -g
    elif op == "mul": gx, gy = g * y, g * x
    else: m = (x > y) + x.eq(y) * 0.5; gx, gy = g * m, g * (1 - m)  # max: ties share it, as in tinygrad
    return (_unbroadcast(gx, x.shape), _unbroadcast(gy, y.shape))
  if op in FOLDS:  # a sum broadcasts back; a max goes to its maxima, shared
    if op[:3] == "sum": return (g,)
    m = x.eq(t)
    return (g * m / m.sum(FOLDS[op]),)
  unary = {"exp": lambda: g * t, "exp2": lambda: g * t * math.log(2), "log": lambda: g * x.reciprocal(),
           "log2": lambda: g * (x * math.log(2)).reciprocal(), "sin": lambda: g * x.cos(), "cos": lambda: -(g * x.sin()),
           "sqrt": lambda: g * (t * 2).reciprocal(), "recip": lambda: -(g * t * t), "relu": lambda: g * (x > 0),
           "cast": lambda: g.cast(x.dtype)}
  if op in unary: return (unary[op](),)
  g = _full(g, t.shape)  # movement and matmul need every element of it
  if op == "matmul": return (_unbroadcast(g.matmul(y.T, x.dtype), x.shape), _unbroadcast(x.T.matmul(g, y.dtype), y.shape))
  if op == "T": return (g.T,)
  if op == "reshape": return (g.reshape(*x.shape),)
  if op == "permute": return (g.permute(*sorted(range(3), key=args[0].__getitem__)),)
  if op == "expand": return (_unbroadcast(g, x.shape),)
  if op == "pad": return (g[tuple(slice(w[0], w[0] + n) for w, n in zip(args[0], x.shape))],)
  if op == "index": return (g.pad(((args[0], x.batch - args[0] - 1), (0, 0), (0, 0))),)
  if op == "split": return (g.merge() if args[0] else g.reshape(*x.shape),)
  if op == "merge": return (g.split(x.batch),)
  if op == "slice":  # back into place, zeros around it; a reversed axis reversed again
    assert all(abs(r.step) == 1 for r in args[0]), "a slice with a step has no backward yet"
    g = g[tuple(slice(None, None, r.step) for r in args[0])] if any(r.step < 0 for r in args[0]) else g
    lo = [min(r) for r in args[0]]
    return (g.pad([(a, n - a - len(r)) for a, n, r in zip(lo, x.shape, args[0])]),)
  if op == "gather":  # the rows' gradients summed into the rows they came from: one-hot(ids)^T @ g, its row i where ids are i
    return ((Tensor.of(t.rt, [[i] for i in range(x.rows)], "int32") == args[0]).bfloat16().matmul(g, x.dtype),)
  raise NotImplementedError(f"no backward for {op}")

def _parts(t):  # an fp32 matmul's operand as bf16 hi + lo (no lo for a bf16 one); a transpose split first, its parts transposed as bf16
  if t.op == "T": return tuple(p if p is None else p.T for p in _parts(t.srcs[0]))
  if t.dtype == "bf16": return t, None
  hi = t.bfloat16()
  return hi, (t - hi).bfloat16()

def _inside(o, s, lo, hi, n):  # the e in [0, n) with lo <= o + s e < hi, as (first, end)
  if s == 0: return (0, n) if lo <= o < hi else (0, 0)
  if s < 0: o, s, lo, hi = -o, -s, 1 - hi, 1 - lo
  first, end = max(-((o - lo) // s), 0), min(-((o - hi) // s), n)
  return (first, end) if first < end else (0, 0)

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
    return ("matmul", _expr(a.realize(), shape, inputs, kernel), b) + ((t.fidelity,) if t.fidelity != 4 else ())
  return (t.op, *(_expr(s, shape, inputs, kernel) for s in t.srcs))

# (expression, input shapes, dtype, result dtype, workers' grid) -> (image, units); a move or gather's name and dtype
# (a transpose's: its shape) -> image
kernels = {}
def _cached(key, compile):
  if key not in kernels: kernels[key] = compile()
  return kernels[key]
