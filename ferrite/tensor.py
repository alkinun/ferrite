# Lazy tensors: bf16, fp32 or int32 (and bfp8 or bfp4 for a matmul's B), of any shape, batched or not, in DRAM as tiles (tile i
# in bank i % 8). Their ops build a tree that realizing turns into kernels (compiler.py): as much of it as one kernel
# holds fused into one launch, and what one kernel cannot hold (a matmul's operands, a move, a cast below the root, a
# reduction of another shape) realized first.
import array, collections, math
from ferrite import bh
from ferrite.compiler import ACCS, BFP, COMBINE, FOLDS, INT32_OPS, ITEMSIZE, ext, tile_bytes, shared_k
from ferrite.compiler import bfp4_kernel, fused_kernel, gather_kernel, move_kernel, put_kernel, readback_kernel, transpose_kernel

FIDELITY = 4  # fidelity phases a bf16 matmul takes unless told: 4 makes exact products (compiler.matmul says what fewer leave out)

class Tensor:
  """batch x rows x cols of bf16, fp32, int32, bfp8 or bfp4 (dtype; batch 1 unless said): in DRAM as 32x32 tiles, matrix after
  matrix, each in row-major tile order, tile i in bank i % 8; or, until realized, an op on other tensors. Realizing
  compiles the whole tree of unrealized ops under it into one kernel; a matmul's own operands are realized first, since
  each of its output tiles needs a whole row and column of them (a transposed B is read transposed rather than
  realized).
  Any shape: the last tile row and column are padded, with zeros, except that a dimension of 1 fills its tile with
  copies (a row vector's tile holds the row 32 times), so broadcasting reuses the tile as it is. Every kernel leaves
  its result that way too, so a matmul can sum over a padded K.
  A view (x[b], slices on tile boundaries, x.split(n), x.repeat_interleave(n)) shares its owner's DRAM: its tile
  (b, i, j) is tile b // rep * bstride + i * stride + j + offset (rep 1 but for a repeat). Any other movement
  (reshape, permute, expand, pad, other slices) is a copy, made by move_kernel from a map over the source's elements; a
  chain of them is one copy.
  A tensor made from one that requires_grad remembers how (_ctx: op, operands, arguments), past realizing, for
  backward (see gradient)."""
  def __init__(self, rt, rows, cols, op=None, srcs=(), batch=1, dtype="bf16"):
    self.rt, self.rows, self.cols, self.batch, self.op, self.srcs, self.dtype = rt, rows, cols, batch, op, srcs, dtype
    tr, tc = -(-rows // 32), -(-cols // 32)  # tile rows and columns
    self.tiles, self.bstride, self.stride, self.offset, self.owner, self.rep = batch * tr * tc, tr * tc, tc, 0, None, 1
    self.addr = None if op else self._alloc()
    self.requires_grad, self.grad, self._ctx = False, None, None
    if op not in (None, "move", "view"): self._track(op, srcs)  # moves and views say what they are themselves
  def _track(self, op, srcs, *args):  # remember how this was made, if anything it came from needs a gradient
    if any(s.requires_grad for s in srcs): self.requires_grad, self._ctx = True, (op, srcs, args)
    return self
  def requires_grad_(self, on=True):
    self.requires_grad = on
    return self
  def _where(self): return self.addr, self.bstride if self.batch > 1 else 0, self.stride, self.offset  # for a kernel: tiles
  def _alloc(self): return self.rt.dram_alloc(-(-self.tiles // len(bh.DRAM)) * tile_bytes(self.dtype))

  def __del__(self):  # its DRAM goes back once nothing refers to it (and no queued launch can still use it)
    if self.addr is not None and self.owner is None: self.rt.dram_free(self.addr)

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
    """From dtype's values in row-major order (little-endian), matrix after matrix. A row's runs of 16 values go into
    tile order as 8-byte words, a strided copy for each word of a run (its place in every tile of the row's band at
    once), and each DRAM bank's tiles in one write: nothing per value, and little per tile, which is what makes a
    model's weights quick to load."""
    assert dtype not in BFP, "a block float is made on the chip, by cast"
    t, n, tr, tc, data = cls(rt, rows, cols, batch=batch, dtype=dtype), ITEMSIZE[dtype], -(-rows // 32), -(-cols // 32), memoryview(data).cast("B")
    assert len(data) == n * batch * rows * cols, f"{len(data)} bytes for {batch} x {rows} x {cols} of {dtype}"
    tb, run = tile_bytes(dtype), 2 * n  # bytes a tile; 8-byte words a run of 16 values (a face's row)
    tiles = memoryview(bytearray(t.tiles * tb)).cast("Q")  # every tile in order, 0 where it pads
    for b, r in ((b, r) for b in range(batch) for r in range(rows if rows > 1 else 32)):  # a row of 1 fills its tile
      at = n * (b * rows + (r if rows > 1 else 0)) * cols
      row = bytes(data[at:at + n]) * 32 if cols == 1 else bytes(data[at:at + n * cols]) + bytes(n * (32 * tc - cols))
      src, first = memoryview(row).cast("Q"), (b * tr + r // 32) * tc * tb // 8 + r % 16 * run
      for half, q in ((half, q) for half in (0, 1) for q in range(run)):  # word q of the runs in the left or right faces
        start = first + (r % 32 // 16 * 2 + half) * tb // 32 + q  # (the tile layout, at the top)
        tiles[start:start + tc * tb // 8:tb // 8] = src[half * run + q::2 * run]
    tiles = tiles.cast("B")
    for bank in range(len(bh.DRAM)):  # tile i is in bank i % 8, at addr + i // 8 tiles: each bank's are one run
      t.rt.chip.write(*bh.DRAM[bank], t.addr, b"".join(tiles[i * tb:(i + 1) * tb] for i in range(bank, t.tiles, len(bh.DRAM))))
    return t

  def tobytes(self):
    """dtype's values in row-major order, matrix after matrix, as frombytes takes them (a block float's as bf16, each of
    its values being one). Waits for the launches queued. The chip writes the tiles into host memory, as many at a time
    as fit (readback_kernel), and they are read there: read over the BARs, each load would wait on PCIe."""
    self.realize()
    n, rows, cols, ct, tb = 2 if self.dtype in BFP else ITEMSIZE[self.dtype], self.rows, self.cols, -(-self.cols // 32), tile_bytes(self.dtype)
    v = 32 if self.dtype in BFP else min(rows, 32)  # the rows a tile sends (readback_kernel), their bytes, and where each face starts
    sent, faces = tb if v == 32 else 32 * v * n, [16 * n * f for f in (0, min(v, 16), 2 * min(v, 16), 2 * min(v, 16) + max(v - 16, 0))]
    per, tiles, kernel = len(self.rt.back) // sent, bytearray(), _cached(("readback", self.dtype), lambda: readback_kernel(self.dtype))
    for start in range(0, self.tiles, per):
      k = min(per, self.tiles - start)
      self.rt.launch(kernel, (start, self.addr, self.bstride, self.stride, self.offset, ct, self.tiles // self.batch, self.rep,
                              self.rt.back_bus & 0xFFFFFFFF, self.rt.back_bus >> 32, v), units=k, name="readback")
      self.rt.wait()
      tiles += self.rt.back[:k * sent]
    if self.dtype in BFP: tiles, sent = b"".join(_bf16s(tiles[i * sent:(i + 1) * sent], self.dtype) for i in range(self.tiles)), 2048
    words, sw, run, per = memoryview(tiles).cast("Q"), sent // 8, 2 * n, self.tiles // self.batch  # 8-byte words: a tile's, a run's
    out, row = bytearray(n * self.batch * rows * cols), memoryview(bytearray(32 * n * ct)).cast("Q")  # (a row of a tile band)
    for b, r in ((b, r) for b in range(self.batch) for r in range(rows)):  # frombytes' copies the other way: a row at a time,
      first = (b * per + r // 32 * ct) * sw + (r % 16 * 16 * n) // 8  # word q of its runs in the left or right faces of every
      for half, q in ((half, q) for half in (0, 1) for q in range(run)):  # tile of its band at once
        start = first + faces[r % 32 // 16 * 2 + half] // 8 + q
        row[half * run + q::2 * run] = words[start:start + ct * sw:sw]
      at = n * (b * rows + r) * cols
      out[at:at + n * cols] = row.cast("B")[:n * cols]
    return bytes(out)

  def tolist(self):  # rows of numbers, or a list of those if this is a batch
    raw = self.tobytes()
    if self.dtype == "bf16" or self.dtype in BFP: raw = array.array("I", [b << 16 for b in array.array("H", raw)]).tobytes()
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
    src = _plain(self.realize())
    return out._run(_cached(("gather", d), lambda: gather_kernel(d)), out.addr, src.addr, src.stride, src.offset,
                    -(-self.cols // 32), ids.cols, ids.addr, ids.offset, name="gather")._track("gather", (self,), ids)

  def put(self, ids, x):
    """x's rows into rows ids of each matrix of this tensor, in place: gather's inverse, ids an int32 tensor of one row
    read on the chip. A KV cache of a fixed size takes a token's keys at its place this way, the place data rather than a
    shape, so a decode step launches the same kernels each token (jit). Into a bfp8 tensor, x is bf16, each of its runs
    of 16 rounded as cast("bfp8") rounds."""
    assert ids.dtype == "int32" and ids.shape[:2] == (1, 1) and ids.cols == x.rows, "rows of x, by a row of int32 places"
    assert (x.batch, x.cols) == (self.batch, self.cols) and x.dtype == ("bf16" if self.dtype == "bfp8" else self.dtype), \
      f"{x.shape} {x.dtype} into {self.shape} {self.dtype}"  # into bfp8 (a bfp8 KV cache), bf16 rows rounded on the way
    assert self.rep == 1, "into a tensor of its own, not a repeat"
    x, ids, ct = _plain(x.realize()), ids.realize(), -(-self.cols // 32)
    tr = -(-self.rows // 32)
    return self.realize()._run(_cached(("put", self.dtype), lambda: put_kernel(self.dtype)), *self._where(), *x._where(), ids.addr,
                               ids.offset, x.rows, ct, tr, units=self.batch * tr * ct, name="put")

  def _view(self, rows, cols, batch, offset, bstride=0, stride=None):
    assert self.rep == 1, "a view's offsets and strides are the stored tensor's, which a repeat's matrices are not"
    v = Tensor(self.rt, rows, cols, "view", batch=batch, dtype=self.dtype)
    v.op, v.addr, v.owner = None, self.realize().addr, self.owner or self
    v.offset, v.bstride, v.stride = self.offset + offset, bstride, self.stride if stride is None else stride
    return v

  def __getitem__(self, key):
    """x[b], one matrix of a batch; x[rows, cols] of every matrix, or x[batch, rows, cols], with slices (any step). A
    slice of whole tiles (starting on tile boundaries, ending on one or at the end, step 1) is a view, read in place;
    any other is a copy (_move)."""
    if self.rep > 1: return (self + 0)[key]  # of a repeat's copy, whose matrices are stored one after another
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

  def repeat_interleave(self, n):
    """Each matrix n times over, one after another along the batch (numpy's repeat), in place: kernels read matrix b as
    the one stored at b // n. Grouped-query attention's keys and values, each head once for every query head it serves."""
    if self.rep > 1: return (self + 0).repeat_interleave(n)  # of a repeat's copy
    v = self._view(self.rows, self.cols, self.batch * n, 0, self.bstride)
    v.rep = n
    return v._track("repeat", (self,), n)

  def split(self, n, axis=1):
    """The matrix cut into n equal parts along axis (1: columns, 0: rows) as a batch of n, in place: attention's heads.
    The parts are whole tiles. A sum not yet computed of computed tensors (a matmul's parts of K) splits term by term,
    and stays a sum."""
    if terms := _terms(self): return sum((t.split(n, axis) for t in terms[1:]), terms[0].split(n, axis))
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
    only on its way into SrcA, so fp32 and int32 move instead (an fp32 matmul transposes its bf16 parts). The transpose
    of one not yet made is what it transposes."""
    if self.op == "T": return self.srcs[0]
    return Tensor(self.rt, self.cols, self.rows, "T", (self,), self.batch, self.dtype)

  def realize(self, into=None):
    """Compute it now, into DRAM of its own, or into `into`'s: a view (merge) or a whole tensor (assign). Returns it,
    realized."""
    if self.op is None: return self
    if self.op in ("T", "move"):  # data movement, into a tensor of its own
      assert into is None, "a transpose or a move writes a tensor of its own"
      x, op, self.op, self.srcs, d = _plain(self.srcs[0].realize()), self.op, None, (), self.dtype
      self.addr = self._alloc()
      where = (self.addr, *x._where())
      if op == "T" and d == "bf16":
        return self._run(_cached(("transpose", x.rows, x.cols), lambda: transpose_kernel(x.rows, x.cols)), *where, name="transpose")
      if op == "T": self.map = (0, x.rows * x.cols, 1, x.cols, 0, 0, 0, *self.shape)
      return self._run(_cached(("move", d), lambda: move_kernel(d)), *where, x.rows, x.cols, self.rows, self.cols,
                       *(v & 0xFFFFFFFF for v in self.map), name="move")
    _fit(self)
    inputs = []
    e, dtype = _tree(self, inputs)
    key = (e, tuple(u.shape for u in inputs), dtype, self.dtype, self.rt.grid, tuple(u.rep for u in inputs))
    kernel, units, name = _cached(key, lambda: (*fused_kernel(*key), _name(e)))
    if into: self.addr, self.owner, self.bstride, self.stride, self.offset = into.addr, into.owner or into, into.bstride, into.stride, into.offset
    else: self.addr = self._alloc()
    self.op, self.srcs = None, ()
    args = [v for t in [self] + inputs for v in t._where()]
    return self._run(kernel, *args, units=units, name=name)
  def assign(self, x):
    """x's value in this tensor, in place, as an optimizer updates a weight: x is computed into its DRAM and becomes a
    view of it. A kernel reads each tile of its elementwise inputs before it writes that tile, but a matmul reads whole
    rows and columns, so one is computed first, on its own."""
    assert self.op is None and self.owner is None and (x.shape, x.dtype) == (self.shape, self.dtype), "a whole tensor, its own shape and dtype"
    if any(t.op == "matmul" for t in _lazy(x, set())): x = x.realize()
    (x if x.op not in (None, "T", "move") else x + 0).realize(self)  # computed already, or a move: copied in by + 0
    return self

  def _run(self, kernel, *args, units=None, name=""):
    """Queue kernel over units of this tensor (tiles unless told), split evenly over as many workers as have any.
    Nothing waits for it: launches run in order, so the next kernel reads this result, and reading it back (tobytes)
    waits."""
    self.rt.launch(kernel, args, units or self.tiles, name)
    return self

  @property
  def shape(self): return (self.batch, self.rows, self.cols)

  # Elementwise: tinygrad's primitives (exp2, log2, sin, sqrt, reciprocal; add, mul, max, cmplt, cmpne; where), and the
  # rest built from them. exp, log and cos are SFPU programs too, with log2(e), ln 2 and the quarter turn kept in fp32
  # inside: as numbers they would be bf16 tensors, and exp(10) as exp2(10 * log2(e)) would be 2% off.
  def _unary(self, op):  # int32 goes through fp32, but for relu (a max with 0)
    if self.dtype == "int32" and op != "relu": return self.float()._unary(op)
    return Tensor(self.rt, self.rows, self.cols, op, (self,), self.batch, self.dtype)
  def _number(self, v, dtype):  # a 1 x 1 tensor of v, uploaded once a runtime: a number in an expression costs no PCIe write
    key = (repr(v), dtype)  # by repr: -0.0 is not 0.0, and a NaN finds itself
    if key not in self.rt.numbers: self.rt.numbers[key] = Tensor.full(self.rt, 1, 1, v, dtype)
    return self.rt.numbers[key]
  def _binop(self, op, x, reverse=False):
    """numpy's broadcasting: shapes match, or one side has 1 where the other does not. A number is a 1 x 1 tensor of
    this one's dtype (fp32 for a float with int32). Different dtypes meet in fp32, and so does int32 for what it does
    not do itself (mul: exact below 2^24)."""
    if not isinstance(x, Tensor): x = self._number(x, "fp32" if self.dtype == "int32" and isinstance(x, float) else self.dtype)
    a, b = (x, self) if reverse else (self, x)
    if a.dtype != b.dtype or (a.dtype == "int32" and op not in INT32_OPS): a, b = a.float(), b.float()
    assert all(n == m or 1 in (n, m) for n, m in zip(a.shape, b.shape)), f"cannot broadcast {a.shape} with {b.shape}"
    batch, rows, cols = map(max, a.shape, b.shape)
    return Tensor(self.rt, rows, cols, op, (a, b), batch, a.dtype)
  def cast(self, dtype):  # bfp4 is made at once, from the bf16 value, by a kernel of its own (bfp4_kernel, which says why)
    if dtype == self.dtype: return self
    if dtype != "bfp4": return Tensor(self.rt, self.rows, self.cols, "cast", (self,), self.batch, dtype)
    x, out, tc = _plain(self.cast("bf16").realize()), Tensor(self.rt, self.rows, self.cols, batch=self.batch, dtype="bfp4"), -(-self.cols // 32)
    return out._run(_cached(("bfp4",), bfp4_kernel), out.addr, *x._where(), tc,
                    -(-self.rows // 32) * tc, name="bfp4")
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
    sum in fp32 (lo@lo, 2^-16 of the rest, is left out), about 16 bits of each operand where fp32 has 24; but the FPU
    adds products 8 at a time to about 2^-11 of the largest (NOTES, "Matmul precision"), which bounds any matmul's
    result. fidelity: the phases a bf16 product takes (FIDELITY unless said; compiler.matmul says what each leaves out);
    fp32's take all four. x may be a block float (a weight cast once: bfp8, read at about half a bf16's bytes, or bfp4, a
    quarter) when self is bf16."""
    fidelity = fidelity or FIDELITY
    if x.tiles <= 16 and (terms := _terms(self)):  # by a small matrix, a sum not yet computed (RoPE's turn of a split K's
      return sum((t.matmul(x, dtype, fidelity) for t in terms[1:]), terms[0].matmul(x, dtype, fidelity))  # parts): term by term
    assert "int32" not in (self.dtype, x.dtype), "the FPU multiplies floats"
    dtype = dtype or ("fp32" if "fp32" in (self.dtype, x.dtype) else "bf16")
    if "fp32" in (self.dtype, x.dtype):  # one op to backward, which keeps its bf16 parts for its own two fp32 matmuls
      a, b = _parts(self), _parts(x)
      return _bf16x3(a, b)._track("matmul", (self, x), a, b).cast(dtype)
    assert self.cols == x.rows and (self.batch == x.batch or 1 in (self.batch, x.batch)), f"{self.shape} @ {x.shape}"
    if self.cols == 1: return (self * x).cast(dtype)  # (R, 1) @ (1, C) is broadcasting
    kt, tiles = self.cols // 32, -(-self.rows // 32) * -(-x.cols // 32)
    s = max([s for s in (2, 4, 8) if self.batch == x.batch == 1 and self.cols % (32 * s) == 0 and kt // s >= 4 and
             s * tiles <= len(self.rt.cores)], default=1)
    if s > 1:  # too few result tiles to keep the workers busy (a token's MLP): K cut in s, a batch, and the parts summed
      parts = self.split(s).matmul(x.split(s, 0), dtype, fidelity)
      return sum((parts[i] for i in range(1, s)), parts[0])
    k, rt = -(-self.cols // 32), -(-self.rows // 32)
    if self.batch == x.batch == 1 and rt > 1 and k > shared_k(rt, self.rt.grid):  # K too long for the workers to share the
      most = shared_k(rt, self.rt.grid, 1)                                         # tiles: in parts that fit, each added onto
      per, out = 32 * -(-k // -(-k // most)), None                                 # the parts before as it is made
      for c in range(0, self.cols, per):
        part = self[:, c:c + per].matmul(x[c:c + per, :], dtype, fidelity)
        out = (part if out is None else part + out).realize()
      return out
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
  if op == "matmul":  # a transposed B's gradient is made transposed, (g.T @ x).T, which the transpose's own takes back
    turn = y._ctx is not None and y._ctx[0] == "T"
    if args:  # fp32: from the forward's parts, transposed, and g's, made once for both
      gp, (xp, yp) = _parts(g), args
      gx, gy = _bf16x3(gp, _pT(yp)), _bf16x3(_pT(gp), xp) if turn else _bf16x3(_pT(xp), gp)
    else: gx, gy = g.matmul(y.T, x.dtype), g.T.matmul(x, y.dtype) if turn else x.T.matmul(g, y.dtype)
    gx, gy = _unbroadcast(gx.cast(x.dtype), x.shape), gy.cast(y.dtype)
    return gx, _unbroadcast(gy, (y.batch, y.cols, y.rows)).T if turn else _unbroadcast(gy, y.shape)
  if op == "T": return (g.T,)
  if op == "reshape": return (g.reshape(*x.shape),)
  if op == "permute": return (g.permute(*sorted(range(3), key=args[0].__getitem__)),)
  if op == "expand": return (_unbroadcast(g, x.shape),)
  if op == "pad": return (g[tuple(slice(w[0], w[0] + n) for w, n in zip(args[0], x.shape))],)
  if op == "index": return (g.pad(((args[0], x.batch - args[0] - 1), (0, 0), (0, 0))),)
  if op == "split": return (g.merge() if args[0] else g.reshape(*x.shape),)
  if op == "repeat": return (g.reshape(x.batch, args[0], x.rows * x.cols).sum(1).reshape(*x.shape),)
  if op == "merge": return (g.split(x.batch),)
  if op == "slice":  # back into place, zeros around it; a reversed axis reversed again
    assert all(abs(r.step) == 1 for r in args[0]), "a slice with a step has no backward yet"
    g = g[tuple(slice(None, None, r.step) for r in args[0])] if any(r.step < 0 for r in args[0]) else g
    lo = [min(r) for r in args[0]]
    return (g.pad([(a, n - a - len(r)) for a, n, r in zip(lo, x.shape, args[0])]),)
  if op == "gather":  # the rows' gradients summed into the rows they came from: one-hot(ids)^T @ g, its row i where ids are i
    return ((Tensor.of(t.rt, [[i] for i in range(x.rows)], "int32") == args[0]).bfloat16().matmul(g, x.dtype),)
  raise NotImplementedError(f"no backward for {op}")

def _fit(t):
  """Parts of t's tree computed first. A tree that reduces sweeps its rows in passes, each recomputing what it reads,
  so a long matmul goes first, and so does an elementwise value that a reduction and something else both read (once,
  on as many workers as it has tiles). Then until the rest fits one kernel (SEM_L1 counts 15 ops): a value two of its
  ops read goes first, the biggest that fits (the residual stream an RMSNorm reads twice), else down the heaviest
  branch the first part that fits, so what is on top (a layer norm, say) stays whole."""
  if any(u.op in COMBINE for u in _lazy(t, set())):
    for u in [u for u in _lazy(t, set()) if u.op == "matmul" and u.srcs[0].cols > 256]: u.realize()
    lazy = list(_lazy(t, set()))
    reads = collections.Counter(id(s) for u in lazy for s in u.srcs)
    for x in [u.srcs[0] for u in lazy if u.op in COMBINE]:
      if reads[id(x)] > 1 and all(u.op not in COMBINE and u.op != "matmul" for u in _lazy(x, set())): x.realize()
  while _ops(t, set()) > 15:
    lazy = list(_lazy(t, set()))
    readers = collections.Counter(id(s) for u in lazy for s in {id(s): s for s in u.srcs}.values())
    shared = [u for u in lazy[1:] if readers[id(u)] > 1 and _ops(u, set()) <= 15]
    u = max(shared, key=lambda s: _ops(s, set())) if shared else t
    while _ops(u, set()) > 15: u = max((s for s in u.srcs if s.op), key=lambda s: _ops(s, set()))
    u.realize()
def _terms(t):  # t, a sum not yet computed of computed tensors its shape, as those tensors; else None
  if t.op != "add" or any(s.shape != t.shape or s.dtype != t.dtype for s in t.srcs): return None
  terms = [[s] if s.op is None else _terms(s) for s in t.srcs]
  return None if None in terms else [u for ts in terms for u in ts]
def _parts(t):  # an fp32 matmul's operand as bf16 (hi, lo), no lo for a bf16 one; a transpose split first, its parts transposed as bf16
  if t.op == "T": return _pT(_parts(t.srcs[0]))
  if t.dtype == "bf16": return t, None
  return t.bfloat16(), (t - t._unary("ftob")).bfloat16()  # lo: t less itself rounded as the cast to bf16 rounds, in one kernel
def _pT(parts): return tuple(p if p is None else p.T for p in parts)
def _plain(t): return (t + 0).realize() if t.rep > 1 else t  # a repeat copied, for a kernel that reads tiles by their place
def _bf16x3(a, b):  # hi@hi + (hi@lo + lo@hi) in fp32, from two operands' parts
  (ah, al), (bh, bl) = a, b
  out, lo = ah.matmul(bh, "fp32", 4), [p.matmul(q, "fp32", 4) for p, q in ((ah, bl), (al, bh)) if p is not None and q is not None]
  return out + sum(lo[1:], lo[0]) if lo else out

def _inside(o, s, lo, hi, n):  # the e in [0, n) with lo <= o + s e < hi, as (first, end)
  if s == 0: return (0, n) if lo <= o < hi else (0, 0)
  if s < 0: o, s, lo, hi = -o, -s, 1 - hi, 1 - lo
  first, end = max(-((o - lo) // s), 0), min(-((o - hi) // s), n)
  return (first, end) if first < end else (0, 0)

WHOLE = ("matmul", "T", "move", "cast")  # ops whose operands are whole tensors, realized first, not tiles in the same kernel
def _below(t, root):  # what a kernel for t computes along with it: not a whole op's operands, unless it is the root's cast
  return t.srcs if t.op not in WHOLE or root and t.op == "cast" else ()
def _lazy(t, seen, root=True):  # the unrealized ops under t, t first
  if t.op is None or id(t) in seen: return
  seen.add(id(t))
  yield t
  for s in _below(t, root): yield from _lazy(s, seen, False)

def _ops(t, seen, root=True):  # how many ops a kernel for t would hold, at most (an SFPU op on another's result adds none)
  if t.op is None or id(t) in seen: return 0
  seen.add(id(t))
  return 1 + sum(_ops(s, seen, False) for s in _below(t, root))

def _tree(t, inputs):
  """The expression a kernel for t computes (fused_kernel's), realized tensors as numbers into inputs, and the dtype it
  computes in: t's, or for a cast or a matmul its operands'. The kernel sweeps along the axis of t's reductions (the
  first found; any along another axis compute first, on their own)."""
  dtype = t.srcs[0].dtype if t.op in ("cast", "matmul") else t.dtype
  assert dtype not in BFP, "a block float is read only as a matmul's B"
  axis = next((FOLDS[u.op] for u in _lazy(t, set()) if u.op in COMBINE), 2)
  n = ext(t.shape, axis)  # the sweep is as long as t along the axis, unless t is one tile there: then its first reduction says
  return _expr(t, t.shape, inputs, [axis, n if n > 1 else None, set(), dtype, t]), dtype


def _bf16s(tile, dtype):  # a block-float tile's values as bf16 (the unpacker's way, ISA FormatConversion.md): normalized,
  out = array.array("H", bytes(2048))  # 7 bits a value (bfp4's 3 shifted up to bfp8's place)
  for i in range(1024):
    x, e = tile[64 + i] if dtype == "bfp8" else (tile[64 + i // 2] >> 4 * (i % 2) & 15) << 4, tile[i // 16]
    if not x & 127: out[i] = 0xFF80 if x >> 7 else 0  # a magnitude of 0 is 0, or with the sign set, -inf
    else:
      m = (x & 127) << 1
      shift = 8 - m.bit_length()
      out[i] = (x >> 7) << 15 | (e - shift) % 256 << 7 | (m << shift) & 0x7E
  return out.tobytes()

def _name(e):  # a kernel's name in profiles: its ops, operands first, each once
  ops = []
  def walk(e):
    if not isinstance(e, tuple): return
    for c in e[1:]: walk(c)
    if e[0] not in ops: ops.append(e[0])
  walk(e)
  return " ".join(ops)

def _expr(t, shape, inputs, kernel):
  """The tree of unrealized ops under t, for a kernel of shape (t's). kernel: [the axis it sweeps, the sweep's length
  (in tiles; the first seen, if t is one tile along it), the reductions it holds (two accumulators), the dtype it
  computes in, its root]. Computed first, on their own: an op of another dtype, a cast, a matmul inside an fp32 kernel, an
  op of 1 along an axis the kernel keeps (repeated there: compute it once), and a reduction along another axis, of
  another length, or past the second."""
  if t.op is None:
    assert t.dtype == kernel[3], f"a {t.dtype} input to a {kernel[3]} kernel"
    return _input(t, inputs)
  if t is kernel[4] and t.op == "cast": return ("cast", _expr(t.srcs[0], shape, inputs, kernel))
  if t is not kernel[4] and (t.dtype != kernel[3] or t.op == "cast" or (t.op == "matmul" and kernel[3] != "bf16")):
    return _expr(t.realize(), shape, inputs, kernel)
  axis = kernel[0]
  if any(t.shape[ax] != shape[ax] for ax in range(3) if ax != axis): return _expr(t.realize(), shape, inputs, kernel)
  if t.op in COMBINE:
    if FOLDS[t.op] != axis: return _expr(t.realize(), shape, inputs, kernel)
    n = ext(t.srcs[0].shape, axis)
    if n > 1 and kernel[1] is None: kernel[1] = n
    if (n > 1 and n != kernel[1]) or (id(t) not in kernel[2] and len(kernel[2]) == len(ACCS)): return _expr(t.realize(), shape, inputs, kernel)
    kernel[2].add(id(t))
  if t.op in ("T", "move"): return _expr(t.realize(), shape, inputs, kernel)  # kernels of their own (a matmul's B may be read transposed, below)
  if t.op == "matmul":  # its operands are whole matrices, not tiles: realize them on their own first
    a, b = t.srcs
    if b.op == "T":  # read transposed; a block float (a bfp8 key cache) as it is, transposed on its way in too
      src = b.srcs[0].realize()
      b = ("T" + src.dtype, _input(src, inputs)) if src.dtype in BFP else ("T", _expr(src, shape, inputs, kernel))
    else: b = (b.dtype, _input(b.realize(), inputs)) if b.dtype in BFP else _expr(b.realize(), shape, inputs, kernel)
    return ("matmul", _expr(a.realize(), shape, inputs, kernel), b) + ((t.fidelity,) if t.fidelity != 4 else ())
  return (t.op, *(_expr(s, shape, inputs, kernel) for s in t.srcs))

def _input(t, inputs):  # a realized tensor's number among a kernel's inputs, added if new
  for i, u in enumerate(inputs):
    if u is t: return i
  inputs.append(t)
  return len(inputs) - 1

class Recording:
  """Launches recorded once (rt.log, the DRAM they take kept) and sent again as one block of commands, packed into the
  ring's bytes once, and again if the kernels they name have left L1 since (Runtime.evictions): every time, if they do
  not fit at once, packing then sending parts itself. jit's, and the tinygrad adapter's TinyJit's."""
  def __init__(self, rt): self.rt, self.launches, self.block = rt, [], None
  def __enter__(self): self.rt.log = []; return self
  def __exit__(self, *_): self.launches, self.rt.log = self.rt.log, None
  def send(self):
    if self.block is None or self.block[2] != self.rt.evictions:
      evictions = self.rt.evictions
      self.block = (*self.rt.block(self.launches), evictions)
    self.rt.send(*self.block[:2])

def jit(f):
  """f, a function of tensors that returns one, run from a recording: its first call runs as any would, its second is
  recorded (each launch, and the DRAM it takes, kept from then on), and each later call copies its inputs into the second
  call's and sends the recorded launches again as one block of commands, building and compiling nothing, so the host's part
  is the inputs' copies (a launch each) and one write to the ring and one to the doorbell (a few, if its kernels do not
  fit in L1 together: a training step's).
  Its inputs keep their shapes and dtypes; nothing else it reads may change, and it may not read back. Its result is
  the same tensor every call, written again by the next, and so is every tensor it made and kept (a gradient in .grad)."""
  calls, inputs, out, rec = 0, (), None, None
  def run(*xs):
    nonlocal calls, inputs, out, rec
    calls, rt = calls + 1, xs[0].rt
    if calls == 1: return f(*xs).realize()
    if calls == 2:
      inputs = [x.realize() for x in xs]
      with Recording(rt) as rec: out = f(*inputs).realize()
      return out
    for x, to in zip(xs, inputs):
      assert (x.shape, x.dtype) == (to.shape, to.dtype), f"{x.shape} {x.dtype} where the recording had {to.shape} {to.dtype}"
      if x is not to: (x + 0).realize(to)
    rec.send()
    return out
  return run

# (expression, input shapes, dtype, result dtype, workers' grid, input repeats) -> (image, units, name); a move, gather,
# put or readback's name and dtype (a transpose's: its shape; bfp4's: its name) -> image
kernels = {}
def _cached(key, compile):
  if key not in kernels: kernels[key] = compile()
  return kernels[key]
