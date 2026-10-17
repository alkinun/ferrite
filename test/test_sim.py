# Every step, one test each, on ttsim (TTSIM=/path/to/libttsim.so python3 -m pytest test) or, without TTSIM, on the
# first Blackhole on the PCI bus (as root; NOTES.md, "Bring-up").
import math, os, random, struct, pytest
import numpy as np
from ferrite import bh, pci
from ferrite.chip import Chip
from ferrite import compiler
from ferrite.compiler import Tensor, from_bf16, to_bf16
from ferrite.hello import hello
from ferrite.runtime import DATA, KERNELS, Runtime, compile_kernel
import gpt2

def bf16(a, packer=False):  # numpy, rounded to bf16 and back: to nearest even as we upload, or half up as the packer does
  b = np.asarray(a, np.float32).view(np.uint32)
  return ((b + (0x8000 if packer else 0x7FFF + (b >> 16 & 1))) & 0xFFFF0000).view(np.float32).astype(np.float64)

def close(t, want, tol):  # a tensor's values within tol (relative, and absolute near 0) of numpy's
  got = np.array(t.tolist())
  assert got.shape == want.shape and np.allclose(got, want, rtol=tol, atol=tol), f"off by up to {np.abs(got - want).max()}"

@pytest.fixture(scope="session")
def chip(): return Chip(pci.open_device())  # one per process: libttsim can only be initialized once

def test_config_space(chip):
  assert chip.pci.cfg_read(0x00) == bh.VENDOR | bh.DEVICE << 16
  assert chip.pci.cfg_read(0x04) & 0x2, "memory decode is off"
  assert len(set(chip.pci.bars.values())) == 3

def test_arc(chip):
  chip.arc_msg(0x90)  # TEST
  t = chip.telemetry()
  assert len(chip.tensix) == 120 and t[bh.TAG_AICLK] > 0, "a p150 has 12 of 14 Tensix columns working, as ttsim models"

def test_dma(chip):
  # the PCIe tile forwards NoC addresses below 2**58 to the host as bus addresses: point the TLB there and the
  # device reads and writes host memory
  buf, bus = chip.pci.alloc(4096)
  buf[:16] = data = random.randbytes(16)
  assert chip.read(*bh.PCIE, bus, 16) == data
  chip.write(*bh.PCIE, bus + 1024, data[::-1])
  assert bytes(buf[1024:1040]) == data[::-1]

def test_l1(chip):
  # write every target before reading any back, so two targets that alias each other fail
  addr, data = bh.L1_SIZE - 65536, {xy: random.randbytes(65536) for xy in (chip.tensix[0], chip.tensix[-1])}
  for (x, y), d in data.items(): chip.write(x, y, addr, d)
  for (x, y), d in data.items(): assert chip.read(x, y, addr, len(d)) == d, f"Tensix {x},{y}"

def test_dram(chip):
  addr, data = bh.TLB_WINDOW - 2048, [random.randbytes(4096) for _ in bh.DRAM]  # straddles two 2 MiB windows
  for (x, y), d in zip(bh.DRAM, data): chip.write(x, y, addr, d)
  for bank, ((x, y), d) in enumerate(zip(bh.DRAM, data)):
    for port in range(3): assert chip.read(x, y + port, addr, len(d)) == d, f"bank {bank} port {port}"

def test_hello(chip):
  assert set(hello(chip, chip.tensix[0], chip.tensix[-1], 30).values()) == {832040}

def test_every_core(chip):
  # every core but the dispatcher; each answers with its own coordinates, read from its NIU
  rt = Runtime(chip, chip.tensix[0], chip.tensix[-1])
  rt.launch(compile_kernel("void kmain(volatile unsigned *a) { a[0] = noc_me(); }"))
  rt.wait()
  assert len(rt.cores) == len(chip.tensix) - 1 == 119  # and one dispatcher
  for x, y in rt.cores: assert chip.read32(x, y, rt.args()) == x | y << 6

def test_relaunch_without_reset(chip):
  # different kernels at the same address, back to back (each loaded after the last is evicted): each must run, not a
  # stale copy of the one before
  rt = Runtime(chip, chip.tensix[0], chip.tensix[2])
  cores = rt.cores
  for i, (op, f) in enumerate([("*", int.__mul__), ("+", int.__add__)] * 2):
    rt.evict()
    rt.launch(compile_kernel(f"void kmain(volatile unsigned *a) {{ a[1] = a[0] {op} 3; }}"), lambda x, y: (x * 100 + y + i, 0))
    rt.wait()
    for x, y in cores: assert chip.read32(x, y, rt.args() + 4) == f(x * 100 + y + i, 3), f"launch {i} on {x},{y}"
  # kernels stay loaded until the room is full; then the next load evicts them all (once the queue is done)
  big = [compile_kernel(f"void kmain(volatile unsigned *a) {{ a[1] = a[0] + {k}; }}").ljust(300_000, b"\0") for k in (1, 2)]
  rt.evict()
  for k, image in enumerate(big + big):
    rt.launch(image, lambda x, y: (x, 0))
    rt.wait()
    assert rt.loaded == {image: KERNELS} and all(chip.read32(x, y, rt.args() + 4) == x + k % 2 + 1 for x, y in cores)

def test_kernel_moves_data(chip):
  # each core pulls its own 40 KiB slice of DRAM into L1 (3 NoC packets), adds its index to every word, writes it
  # back elsewhere in DRAM, then sends its first 64 bytes to the next core's L1
  rt, n = Runtime(chip, chip.tensix[0], chip.tensix[4]), 40 * 1024  # four workers in the first row
  cores, (bx, by), src, dst = rt.cores, bh.DRAM[3], 1 << 20, 2 << 20
  data = [random.randbytes(n) for _ in cores]
  for i, d in enumerate(data): chip.write(bx, by, src + i * n, d)
  args = {xy: (i, bx, by, src + i * n, dst + i * n, n, *cores[(i + 1) % len(cores)]) for i, xy in enumerate(cores)}
  rt.launch(compile_kernel("""
void kmain(volatile unsigned *a) {  // a: [index, bank x, bank y, src, dst, n, next x, next y]
  unsigned *buf = (unsigned *)L1_DATA;
  noc_read(a[1], a[2], a[3], L1_DATA, a[5]);
  noc_wait();
  for (unsigned i = 0; i < a[5] / 4; i++) buf[i] += a[0];
  noc_write(L1_DATA, a[1], a[2], a[4], a[5]);
  noc_write(L1_DATA, a[6], a[7], L1_DATA + a[5], 64);
}"""), lambda x, y: args[x, y])
  rt.wait()
  out = [struct.pack(f"<{n // 4}I", *[(w + i) & 0xFFFFFFFF for w in struct.unpack(f"<{n // 4}I", d)]) for i, d in enumerate(data)]
  for i, ((x, y), o) in enumerate(zip(cores, out)):
    assert chip.read(bx, by, dst + i * n, n) == o, f"DRAM slice {i}"
    assert chip.read(*cores[(i + 1) % len(cores)], DATA + n, 64) == o[:64], f"L1 of the core after {i}"

def test_queue(chip):
  # 100 launches queued back to back through a ring that holds 16 of them: it fills, the host waits for space, the
  # dispatcher wraps around it six times. Each launch adds its index to a word in every worker's L1.
  rt = Runtime(chip, chip.tensix[0], chip.tensix[2], ring=2048)
  for x, y in rt.cores: chip.write32(x, y, DATA, 0)
  add = compile_kernel("void kmain(volatile unsigned *a) { *(volatile unsigned *)L1_DATA += a[0]; }")
  rt.load(add)  # once, before: then it stays
  writes, write = [], chip.pci.write
  chip.pci.write = lambda *a: (writes.append(a), write(*a))
  try:
    for i in range(100):
      before = len(writes)
      rt.launch(add, lambda x, y: (i,))
      assert len(writes) - before == 1, "a launch is one PCIe write: the doorbell"
  finally: chip.pci.write = write
  rt.wait()
  for x, y in rt.cores: assert chip.read32(x, y, DATA) == sum(range(100))

def test_hung_kernel_times_out(chip):
  rt = Runtime(chip, chip.tensix[0], chip.tensix[1])
  rt.launch(compile_kernel("void kmain(volatile unsigned *a) { for (;;); }"))
  with pytest.raises(TimeoutError, match=r"launch 1, still running on \[\(1, 2\)\]"): rt.wait(timeout=0.2)

def test_dram_is_freed(chip):
  # first fit, 64-byte aligned; a freed block merges with the holes on both sides; a tensor gives its DRAM back once
  # nothing refers to it, intermediates included, so a loop of launches leaves DRAM as it found it
  rt = Runtime(chip, chip.tensix[0], chip.tensix[1])
  a, b, c = (rt.dram_alloc(n) for n in (100, 4096, 1))
  assert (a, b, c) == (0, 128, 128 + 4096)
  rt.dram_free(b)
  assert rt.dram_alloc(1000) == 128 and rt.dram_alloc(4096) == 128 + 4096 + 64, "first fit"
  for addr in (128, a, c, 128 + 4096 + 64): rt.dram_free(addr)
  assert rt.holes == [(0, bh.DRAM_BANK_SIZE)] and not rt.used
  with pytest.raises(MemoryError): rt.dram_alloc(bh.DRAM_BANK_SIZE + 1)
  x = Tensor.of(rt, [[float(r - c) for c in range(64)] for r in range(64)])
  for _ in range(3):
    y = (x + x) * x
    assert y.tolist()[1][0] == 2.0
  assert len(rt.used) == 2, "x and the last y"
  del x, y
  assert rt.holes == [(0, bh.DRAM_BANK_SIZE)] and not rt.used
  # launches are queued, not waited for: DRAM freed while one may still read it waits for that launch to be done
  x = Tensor.of(rt, [[float(r - c) for c in range(64)] for r in range(64)])
  y, at = (x * 3.0).realize(), x.addr  # queued; on ttsim nothing runs until the host waits
  del x
  z = Tensor.of(rt, [[1.0] * 64] * 64)
  assert z.addr != at and y.tolist()[1][0] == 3.0

def test_tile_math(chip):
  # one tile of small integers through unpack -> FPU -> pack: add, sub and mul must be exact
  rt = Runtime(chip, chip.tensix[0], chip.tensix[1])
  a = [[float((r * 32 + c) % 17 - 8) for c in range(32)] for r in range(32)]
  b = [[float((r + 3 * c) % 11 - 5) for c in range(32)] for r in range(32)]
  ta, tb = Tensor.of(rt, a), Tensor.of(rt, b)
  for got, f in ((ta + tb, float.__add__), (ta - tb, float.__sub__), (ta * tb, float.__mul__)):
    assert got.tolist() == [[f(x, y) for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]

def test_tensor_math(chip):
  # random bf16 over 8 tiles on 8 workers. mul runs all four fidelity phases into fp32 Dst, so it is the exact product
  # rounded once by the packer; add and sub align mantissas at 11 bits first, so allow one bf16 step of the larger input
  rt, rows, cols = Runtime(chip, chip.tensix[0], chip.tensix[-1]), 64, 128
  a, b = ([[from_bf16(to_bf16(random.uniform(-4, 4))) for _ in range(cols)] for _ in range(rows)] for _ in range(2))
  ta, tb = Tensor.of(rt, a), Tensor.of(rt, b)
  packed = lambda v: from_bf16((struct.unpack("<I", struct.pack("<f", v))[0] + 0x8000) >> 16)  # the packer's rounding
  assert (ta * tb).tolist() == [[packed(x * y) for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]
  for got, f in (((ta + tb).tolist(), float.__add__), ((ta - tb).tolist(), float.__sub__)):
    for r in range(rows):
      for c in range(cols):
        x, y = a[r][c], b[r][c]
        assert abs(got[r][c] - f(x, y)) <= max(abs(x), abs(y)) * 2**-7, (r, c, x, y, got[r][c])

def test_matmul_exact(chip):
  # small integers, K = 2 tiles so Dst accumulates across them; then A @ I == A bit for bit, which pins the tile layout
  rt = Runtime(chip, chip.tensix[0], chip.tensix[-1])
  a = [[float((r * 7 + c * 3) % 3 - 1) for c in range(64)] for r in range(32)]
  b = [[float((r * 5 + c) % 3 - 1) for c in range(32)] for r in range(64)]
  assert (Tensor.of(rt, a) @ Tensor.of(rt, b)).tolist() == [[sum(a[i][k] * b[k][j] for k in range(64)) for j in range(32)] for i in range(32)]
  x = [[from_bf16(to_bf16(random.uniform(-8, 8))) for _ in range(64)] for _ in range(96)]
  eye = [[float(r == c) for c in range(64)] for r in range(64)]
  assert (Tensor.of(rt, x) @ Tensor.of(rt, eye)).tolist() == x
  # X @ diag(d) with full-mantissa d: every output is one product, exact in fp32 Dst only if all four fidelity phases
  # ran (each holds a different slice of the mantissas), then rounded once by the packer
  d = [from_bf16(to_bf16(random.uniform(1, 2))) for _ in range(64)]
  packed = lambda v: from_bf16((struct.unpack("<I", struct.pack("<f", v))[0] + 0x8000) >> 16)
  diag = [[d[c] if r == c else 0.0 for c in range(64)] for r in range(64)]
  assert (Tensor.of(rt, x) @ Tensor.of(rt, diag)).tolist() == [[packed(v * d[c]) for c, v in enumerate(row)] for row in x]

def test_matmul_random(chip):
  # 64x96 @ 96x128: 8 output tiles on 8 workers, 3 K tiles each. Four fidelity phases make every product exact, Dst sums
  # in fp32, the packer rounds once to bf16: the error stays under one bf16 step of sum |a*b|
  rt, (m, k, n) = Runtime(chip, chip.tensix[0], chip.tensix[-1]), (64, 96, 128)
  a = [[from_bf16(to_bf16(random.uniform(-1, 1))) for _ in range(k)] for _ in range(m)]
  b = [[from_bf16(to_bf16(random.uniform(-1, 1))) for _ in range(n)] for _ in range(k)]
  got = (Tensor.of(rt, a) @ Tensor.of(rt, b)).tolist()
  for i in range(m):
    for j in range(n):
      terms = [a[i][x] * b[x][j] for x in range(k)]
      assert abs(got[i][j] - sum(terms)) <= 2**-8 * sum(map(abs, terms)), (i, j, got[i][j], sum(terms))

def test_matmul_blocks(chip):
  # result tiles go in blocks of up to 4 x 4, with the block's rows of A and column of B kept in L1. 5 x 6 tiles cut
  # into blocks leaves part-blocks at the edges, whose missing tiles must not be computed: they would land past the
  # result in DRAM, on the tensor allocated after it (the canary, written before the queued kernel runs)
  rt = Runtime(chip, chip.tensix[0], chip.tensix[-1])
  a = [[float((r * 7 + c * 3) % 5 - 2) for c in range(96)] for r in range(160)]
  b = [[float((r + 2 * c) % 3 - 1) for c in range(192)] for r in range(96)]
  y = (Tensor.of(rt, a) @ Tensor.of(rt, b)).realize()  # queued
  canary = Tensor.of(rt, [[1.0] * 192] * 160)
  assert y.tolist() == [[sum(a[i][k] * b[k][j] for k in range(96)) for j in range(192)] for i in range(160)]
  assert canary.tolist() == [[1.0] * 192] * 160

def test_stream_on_one_worker(chip):
  # one worker gets every tile, so the pipeline really cycles: L1 slots and Dst halves alternate, the threads hand
  # tiles along by semaphore, and BRISC fetches and writes back around them. 32 elementwise tiles, then a 128x96 @ 96x128
  # matmul: 16 output tiles of 3 steps each
  rt = Runtime(chip, chip.tensix[0], chip.tensix[1])
  a, b = ([[from_bf16(to_bf16(random.uniform(-2, 2))) for _ in range(128)] for _ in range(256)] for _ in range(2))
  packed = lambda v: from_bf16((struct.unpack("<I", struct.pack("<f", v))[0] + 0x8000) >> 16)
  assert (Tensor.of(rt, a) * Tensor.of(rt, b)).tolist() == [[packed(x * y) for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]
  x = [[float((r + 2 * c) % 5 - 2) for c in range(96)] for r in range(128)]
  y = [[float((3 * r + c) % 3 - 1) for c in range(128)] for r in range(96)]
  assert (Tensor.of(rt, x) @ Tensor.of(rt, y)).tolist() == [[sum(x[i][k] * y[k][j] for k in range(96)) for j in range(128)] for i in range(128)]

def unfused(t):  # the same tree of ops, realized one launch per op
  if t.op is None: return t
  srcs = [unfused(s) for s in t.srcs]
  return (srcs[0] @ srcs[1] if t.op == "matmul" else Tensor(t.rt, t.rows, t.cols, t.op, tuple(srcs))).realize()

def test_fusion(chip):
  # a fused chain is one launch, its intermediates never get DRAM, and it is bit for bit the same as running each op on
  # its own: every intermediate is rounded to bf16 on its way to L1 just as it would be on its way to DRAM. Both trees
  # have ops that read two intermediates, and in the second one an op reads an intermediate made two ops earlier
  for workers in (chip.tensix[1], chip.tensix[-1]):  # one worker cycling through 12 tiles, then a tile per worker
    rt = Runtime(chip, chip.tensix[0], workers)
    a, b, c, d = (Tensor.of(rt, [[from_bf16(to_bf16(random.uniform(-3, 3))) for _ in range(96)] for _ in range(128)]) for _ in range(4))
    for build in (lambda a, b, c, d: (a + b) * c - d * a, lambda a, b, c, d: (a + b) * ((c - d) + a * a)):
      fused, launches = build(a, b, c, d), rt.launches
      lazy = [t for t in (fused, *fused.srcs, *fused.srcs[0].srcs, *fused.srcs[1].srcs) if t.op]
      got = fused.tolist()
      assert rt.launches == launches + 1 and all(t.addr is None for t in lazy[1:]), "one kernel, intermediates stay on chip"
      assert got == unfused(build(a, b, c, d)).tolist()

def test_threads_wait(chip):
  # each TRISC feeds its thread as fast as it can and ttsim runs every instruction the moment it can, so only the waits
  # hold a thread back. The pack TRISC, with the least to push, has each pack queued before its math is done (SEM_DST);
  # thread 0 reaches the ops that read intermediates before the pack thread has written them (SEM_L1); the math thread
  # gets three ops in a row that read only inputs, so the third waits for the pack thread to drain a Dst half (SEM_DST).
  # On one worker, so slots and halves are reused
  rt = Runtime(chip, chip.tensix[0], chip.tensix[1])
  a, b, c, d = (Tensor.of(rt, [[float((r * 3 + c * k) % 7 - 3) for c in range(64)] for r in range(128)]) for k in (1, 2, 5, 6))
  assert ((a + b) + ((c - d) + a * b)).tolist() == unfused((a + b) + ((c - d) + a * b)).tolist()

def test_matmul_fusion(chip):
  # matmuls inside elementwise trees, as one kernel: the matmul's result tile is packed into L1 and read from there.
  # Bit for bit the same as one launch per op, since it is rounded to bf16 on its way to L1 as it would be to DRAM
  rnd = lambda r, c: [[from_bf16(to_bf16(random.uniform(-1, 1))) for _ in range(c)] for _ in range(r)]
  for workers in (chip.tensix[1], chip.tensix[-1]):  # one worker cycling through 8 tiles, then a tile per worker
    rt = Runtime(chip, chip.tensix[0], workers)
    a, b, c, d, e, f = (Tensor.of(rt, rnd(*shape)) for shape in ((64, 96), (96, 128), (64, 128), (64, 128), (64, 64), (64, 128)))
    for build in (lambda: (a @ b) + c, lambda: (a @ b) * d - c, lambda: (a @ b) + (e @ f)):
      launches, got = rt.launches, build().tolist()
      assert rt.launches == launches + 1, "one kernel"
      assert got == unfused(build()).tolist()

def test_replay_keeps_pushes_small():
  # the TRISCs push one Tensix instruction a cycle (four when they come in a row), so what they push per tile is code
  # they fetch and time they take. The repeating runs (a matmul phase, a mul phase, 15 PACRs) are recorded in the
  # threads' replay buffers once per kernel and replayed by one word each
  c, slots = compiler, compiler.allocate({"mm", "mul"})
  tile = lambda op: len(c.unpack(0, 1)) + len(c.math_begin(0, op == "mul") + c.elementwise(op, slots) + c.MATH_END) + len(c.pack(0, 0, c.SEM_L1))
  assert tile("add") <= 26 and tile("mul") <= 27, "was 48 and 80 before replay"
  assert len(c.step_unpack(0) + c.matmul(slots)) <= 12, "a matmul step was 71"
  assert len(c.sfpu(["exp"], c.allocate({"exp"}))) <= 45, "32 passes of 19 words, replayed"
  for names in ({"mm", "mul"}, {"exp", "recip", "relu", "mm", "mul"}):  # what is recorded fits the 32-entry buffer
    assert sum(len(c.MATH_RUNS[n]) for n in c.allocate(names)) <= 32

def test_sfpu(chip):
  # relu exactly, exp and 1/x to a bf16 step (they are fp32 on the SFPU, rounded once by the packer), including the
  # edges. The Tensix flushes denormals to zero on the way in and out, so the reference does too: 1/denormal is inf,
  # exp(-100) is 0. On an input (through x + 0), and in place after an add, where exp sees the add's fp32 result: no
  # bf16 rounding in between, so the reference is exp of the exact sum
  rt = Runtime(chip, chip.tensix[0], chip.tensix[2])
  bf = lambda v: from_bf16(to_bf16(v))
  flush = lambda v: 0.0 if abs(v) < 2**-126 else v
  x = [[bf(random.uniform(-6, 6)) for _ in range(64)] for _ in range(64)]
  x[0][:10] = map(bf, [0.0, -0.0, 1.0, -1.0, -100.0, 3.0e-39, 1e30, -1e30, math.inf, -math.inf])
  y = [[bf(random.uniform(-1, 1)) for _ in range(64)] for _ in range(64)]
  tx, ty = Tensor.of(rt, x), Tensor.of(rt, y)
  def check(got_rows, ins, f):  # within a bf16 step; 1/0 only has to be infinite
    for got, row in zip(got_rows, ins):
      for g, v in zip(got, row):
        want = f(flush(v))
        assert (math.isinf(g) if flush(v) == 0 and f is recip else g == want or abs(g - want) <= abs(want) * 2**-7), (v, g, want)
  exp = lambda v: math.inf if v > 88.7 else 0.0 if v < -104 else flush(bf(math.exp(v)))
  recip = lambda v: math.inf if v == 0 else flush(bf(1 / v))  # 1/inf is 0
  assert tx.relu().tolist() == [[max(flush(v), 0.0) for v in row] for row in x]
  check(tx.exp().tolist(), x, exp)
  check(tx.reciprocal().tolist(), x, recip)
  check((tx + ty).exp().tolist(), [[a + b for a, b in zip(ra, rb)] for ra, rb in zip(x, y)], exp)
  # relu fused onto a matmul plus an add, one launch; rounding commutes with relu, so bit for bit as op by op
  a, b, c = (Tensor.of(rt, [[bf(random.uniform(-1, 1)) for _ in range(64)] for _ in range(64)]) for _ in range(3))
  launches, fused = rt.launches, ((a @ b) + c).relu().tolist()
  assert rt.launches == launches + 1 and fused == unfused(((a @ b) + c).relu()).tolist()
  # all three SFPU programs in one kernel: exp and recip fill the 32-entry replay buffer, so relu goes word by word
  u = [[bf(random.uniform(-3, 3)) for _ in range(64)] for _ in range(64)]
  w = [[bf(random.choice((-1, 1)) * random.uniform(0.5, 2)) for _ in range(64)] for _ in range(64)]
  got = (Tensor.of(rt, u).exp().relu() + Tensor.of(rt, w).reciprocal()).tolist()
  for g, ru, rw in zip(got, u, w):
    for gv, uv, wv in zip(g, ru, rw):
      e, r = exp(uv), recip(wv)
      assert abs(gv - (e + r)) <= 2**-6 * max(abs(e), abs(r)), (uv, wv, gv, e + r)

def test_reductions(chip):
  # row max exactly, row sum to a bf16 step of the sum of magnitudes, each a column; then broadcast back:
  # x - rowmax(x), and a column times a number, all against Python
  rt, rows, cols = Runtime(chip, chip.tensix[0], chip.tensix[-1]), 64, 96
  bf = lambda v: from_bf16(to_bf16(v))
  x = [[bf(random.uniform(-4, 4)) for _ in range(cols)] for _ in range(rows)]
  tx = Tensor.of(rt, x)
  assert tx.rowmax().tolist() == [[max(row)] for row in x]
  for (got,), row in zip(tx.rowsum().tolist(), x):
    assert abs(got - sum(row)) <= 2**-8 * sum(map(abs, row)), (got, sum(row))
  for got, row in zip((tx - tx.rowmax()).tolist(), x):
    assert all(abs(g - (v - max(row))) <= 2**-7 * max(abs(v), abs(max(row))) for g, v in zip(got, row))
  assert (tx.rowmax() * 0.5).tolist() == [[max(row) / 2] for row in x]

def test_any_shape(chip):
  # 45 x 70 is two tile rows by three tile columns, the last of each cut short; a dimension of 1 broadcasts. Results go
  # home with zero padding (a matmul of two results needs that, or the padding of K adds in), and row reductions mask
  # their last tile's padding (with every value negative, an unmasked 0 would be the max and add up in softmax)
  rt, rng = Runtime(chip, chip.tensix[0], chip.tensix[-1]), np.random.default_rng(0)
  def make(rows, cols, lo=-2.0, hi=2.0):
    a = bf16(rng.uniform(lo, hi, (rows, cols)))
    return a, Tensor.of(rt, a.tolist())
  (x, tx), (y, ty), (col, tcol), (row, trow) = make(45, 70), make(45, 70), make(45, 1), make(1, 70)
  assert (tx * ty).tolist() == bf16(x * y, True).tolist() and (tcol @ trow).tolist() == bf16(col * row, True).tolist()
  close(tx + ty, x + y, 2**-7)
  close(tx * trow + tcol - 2.5, x * row + col - 2.5, 2**-6)
  neg, tneg = make(45, 70, -6, -1)
  assert tneg.rowmax().tolist() == neg.max(1, keepdims=True).tolist()
  close(tneg.rowsum(), neg.sum(1, keepdims=True), 2**-8)
  e = np.exp(neg - neg.max(1, keepdims=True))
  close((tneg - tneg.rowmax()).exp() / (tneg - tneg.rowmax()).exp().rowsum(), e / e.sum(1, keepdims=True), 0.02)
  narrow, tnarrow = make(45, 20, -3, -1)  # one tile wide: reduced on its own, and inside a three-tile-wide kernel
  assert tnarrow.rowmax().tolist() == narrow.max(1, keepdims=True).tolist()
  close(tnarrow.rowsum() + tx, narrow.sum(1, keepdims=True) + x, 2**-6)
  (a, ta), (b, tb) = make(45, 50, -1, 1), make(50, 70, -1, 1)
  close(ta.exp() @ tb.exp(), bf16(np.exp(a)) @ bf16(np.exp(b)), 2**-6)

def test_views_transpose_gelu(chip):
  # views are read in place: tile-aligned slices as inputs to elementwise ops and to matmuls, like a GPT-2 layer's
  # 64-column attention heads. Transpose is exact, of odd shapes and of views, and turns a filled row into a filled
  # column. gelu against numpy, within the bf16 rounding of its steps
  rt, rng = Runtime(chip, chip.tensix[0], chip.tensix[-1]), np.random.default_rng(1)
  def make(rows, cols, lo=-2.0, hi=2.0):
    a = bf16(rng.uniform(lo, hi, (rows, cols)))
    return a, Tensor.of(rt, a.tolist())
  (x, tx), (y, ty), (row, trow) = make(70, 100), make(70, 192), make(1, 100)
  v = tx[32:70, 64:100]
  assert v.tolist() == x[32:70, 64:100].tolist() and (v * 2.0).tolist() == (x[32:70, 64:100] * 2).tolist()
  assert tx.T.tolist() == x.T.tolist() and v.T.tolist() == x[32:70, 64:100].T.tolist()
  close(tx.T + trow.T, x.T + row.T, 2**-7)
  q, k = ty[:, 64:128], ty[:, 128:192]  # one head's query and key
  close(q @ k.T, y[:, 64:128] @ y[:, 128:192].T, 2**-6)
  assert tx.gather([5, 0, 69, 37, 5]).tolist() == x[[5, 0, 69, 37, 5]].tolist()  # rows from both tile rows, one twice
  close(tx + tx.gather([7]), x + x[7], 2**-7)  # a single row fills its tile, so it broadcasts
  gelu = lambda a: 0.5 * a * (1 + np.tanh(math.sqrt(2 / math.pi) * (a + 0.044715 * a**3)))
  close(tx.gelu(), gelu(x), 0.02)

def test_batch(chip):
  # a batch of matrices: elementwise ops broadcast over it (a batch of 1 repeats), row reductions and softmax per
  # matrix, matmul matrix by matrix (or by one matrix for all), views that split a matrix into a batch of heads and a
  # batch back into matrices, all against numpy. Shapes are cut short on purpose (40 rows)
  rt, rng = Runtime(chip, chip.tensix[0], chip.tensix[-1]), np.random.default_rng(4)
  def make(*shape):
    a = bf16(rng.uniform(-2, 2, shape))
    return a, Tensor.of(rt, a.tolist())
  (x, tx), (y, ty), (row, trow) = make(3, 40, 70), make(1, 40, 70), make(1, 70)
  assert tx.shape == (3, 40, 70) and ty.shape == (1, 40, 70) and np.array(tx.tolist()).tolist() == x.tolist()
  assert (tx * ty).tolist() == bf16(x * y, True).tolist() and (tx[1] * 2.0).tolist() == (x[1] * 2).tolist()
  close(tx - trow + 1.0, x - row + 1, 2**-6)
  assert tx.rowmax().tolist() == x.max(2, keepdims=True).tolist()
  e = np.exp(x - x.max(2, keepdims=True))
  close((tx - tx.rowmax()).exp() / (tx - tx.rowmax()).exp().rowsum(), e / e.sum(2, keepdims=True), 0.02)
  (a, ta), (b, tb), (w, tw) = make(3, 40, 96), make(3, 96, 70), make(96, 70)
  close(ta @ tb, a @ b, 2**-6)
  close(ta @ tw + trow, a @ w + row, 2**-6)
  (qkv, tqkv), (kt, tkt) = make(40, 192), make(192, 40)  # 3 heads of 64: split columns, split rows
  heads, keys = tqkv.split(3), tkt.split(3, axis=0)
  assert heads.shape == (3, 40, 64) and np.array(heads.tolist()).tolist() == qkv.reshape(40, 3, 64).transpose(1, 0, 2).tolist()
  assert np.array(keys.tolist()).tolist() == kt.reshape(3, 64, 40).tolist()
  close(heads @ keys, qkv.reshape(40, 3, 64).transpose(1, 0, 2) @ kt.reshape(3, 64, 40), 2**-6)
  one, a = Runtime(chip, chip.tensix[0], chip.tensix[1]), a[:, :20]  # last: it takes two of rt's cores. One worker, and one
  close(Tensor.of(one, a.tolist()) @ Tensor.of(one, b.tolist()), a @ b, 2**-6)  # tile row a matrix: A's row in L1 changes with the matrix

def test_fuzz(chip, monkeypatch):
  # random expressions (elementwise ops with broadcasting, unary ops, row reductions, matmuls; odd shapes, batches), and
  # every kernel they turn into checked against numpy on the same inputs, read back from the chip: whatever bf16 rounding
  # piles up through a tree, each kernel must be right about its own inputs
  rt, checked = Runtime(chip, chip.tensix[0], chip.tensix[-1]), []
  def value(t): return np.array(t.tolist()).reshape(t.shape)
  def numpy(e, xs):
    if isinstance(e, int): return xs[e]
    v = [numpy(s, xs) for s in e[1:]]
    return {"add": np.add, "sub": np.subtract, "mul": np.multiply, "matmul": np.matmul, "relu": lambda a: np.maximum(a, 0),
            "exp": np.exp, "recip": lambda a: 1 / a, "rsqrt": lambda a: 1 / np.sqrt(a),
            "rowsum": lambda a: a.sum(-1, keepdims=True), "rowmax": lambda a: a.max(-1, keepdims=True)}[e[0]](*v)
  realize = Tensor.realize
  def checked_realize(t):
    if t.op is None: return t
    while compiler._ops(t, set()) > 15:  # too big for one kernel: its parts first, as realize does, each checked too
      u = t
      while compiler._ops(u, set()) > 15: u = max((s for s in u.srcs if s.op), key=lambda s: compiler._ops(s, set()))
      u.realize()
    inputs = []
    e = compiler._expr(t, (t.batch, t.rows), inputs)
    xs = [value(u) for u in inputs]
    realize(t)
    want, got = numpy(e, xs), value(t)
    assert np.abs(got - want).max() <= 0.03 * (np.abs(want).max() + 1), (e, np.abs(got - want).max())
    checked.append(e)
    return t
  monkeypatch.setattr(Tensor, "realize", checked_realize)
  rng = np.random.default_rng(5)
  def leaf(shape):  # some all negative, so an unmasked 0 in the padding would win a max
    lo = rng.choice([-2.0, -4.0])
    a = bf16(rng.uniform(lo, lo + 4 if lo == -2 else lo + 3, shape))
    return Tensor.of(rt, a.tolist() if shape[0] > 1 else a[0].tolist())
  def gen(shape, depth):
    b, r, c = shape
    kind = "leaf" if depth == 0 or rng.random() < 0.2 else rng.choice(["bin", "bin", "unary", "red", "mm"])
    if kind == "bin":  # each operand keeps a dimension, or has 1 there (not both)
      s1 = tuple(1 if rng.random() < 0.2 else d for d in shape)
      s2 = tuple(d if s1[i] == 1 else (1 if rng.random() < 0.2 else d) for i, d in enumerate(shape))
      return getattr(gen(s1, depth - 1), rng.choice(["__add__", "__sub__", "__mul__"]))(gen(s2, depth - 1))
    if kind == "unary":
      x = gen(shape, depth - 1)
      return rng.choice([x.relu, (x * 0.5).exp, (x * x + 1.0).reciprocal, (x * x + 1.0).rsqrt])()
    if kind == "red" and c == 1:  # values stay near 1 (means, not sums), or exp would turn rounding into large errors
      n = int(rng.choice([20, 32, 70, 96, 192]))
      x = gen((b, r, n), depth - 1)
      return x.rowsum() * (1 / n) if rng.random() < 0.5 else x.rowmax()
    if kind == "mm" and r > 1 and c > 1:
      k = int(rng.choice([32, 50, 64]))
      return (leaf((b, r, k)) @ leaf((b if rng.random() < 0.5 else 1, k, c))) * (1 / k)
    return leaf(shape)
  for _ in range(30):  # sizes past 4 tiles make part-blocks of matmul results
    shape = (int(rng.choice([1, 1, 2])), int(rng.choice([1, 20, 32, 45, 64, 160])), int(rng.choice([1, 20, 32, 70, 96, 192])))
    gen(shape, 4).realize()
  assert len(checked) >= 20 and any(e[0] == "matmul" or "rowsum" in str(e) for e in checked)

def test_mnist_mlp(chip):
  # goal 1's halfway mark: an MNIST-shaped network, 784 -> 128 -> 10 with relu and biases, on a batch of 16, against
  # numpy in float64. Two launches: matmul + bias + relu, then matmul + bias. Random weights (no dataset here); the
  # logits must match closely and so must the predicted classes, wherever the top two are not a near tie. K = 784 is
  # 25 tiles, more than the result is wide (which once made the kernel sweep the wrong tiles)
  rt, rng = Runtime(chip, chip.tensix[0], chip.tensix[-1]), np.random.default_rng(2)
  x = bf16(rng.uniform(0, 1, (16, 784)))
  w1, b1 = bf16(rng.normal(0, 784**-0.5, (784, 128))), bf16(rng.normal(0, 0.1, (1, 128)))
  w2, b2 = bf16(rng.normal(0, 128**-0.5, (128, 10))), bf16(rng.normal(0, 0.1, (1, 10)))
  tx, tw1, tb1, tw2, tb2 = (Tensor.of(rt, a.tolist()) for a in (x, w1, b1, w2, b2))
  launches = rt.launches
  logits = np.array(((tx @ tw1 + tb1).relu() @ tw2 + tb2).tolist())
  assert rt.launches == launches + 2
  want = np.maximum(x @ w1 + b1, 0) @ w2 + b2
  assert np.allclose(logits, want, rtol=0.02, atol=0.02), np.abs(logits - want).max()
  top2 = np.sort(want, 1)[:, -2:]
  clear = top2[:, 1] - top2[:, 0] > 0.05
  assert clear.sum() >= 8 and (logits.argmax(1) == want.argmax(1))[clear].all()

def test_gpt2_layer(chip):
  # goal 1's finish line: a GPT-2 small block (768 wide, 12 heads of 64, MLP 3072) on 16 tokens, against the same code
  # (test/gpt2.py) in numpy, float64, with the same bf16 weights. Weights as GPT-2 initialises them; the attention is
  # causal and not uniform (scores of a few units). Every intermediate is bf16 here, so the block's change to x (up to
  # ~2) must match to within 3% of its size
  rt, rng, n, d = Runtime(chip, chip.tensix[0], chip.tensix[-1]), np.random.default_rng(3), 16, 768
  shapes = {"ln_1.weight": (1, d, 1), "ln_1.bias": (1, d, 0), "attn.c_attn.weight": (d, 3 * d, 0), "attn.c_attn.bias": (1, 3 * d, 0),
            "attn.c_proj.weight": (d, d, 0), "attn.c_proj.bias": (1, d, 0), "ln_2.weight": (1, d, 1), "ln_2.bias": (1, d, 0),
            "mlp.c_fc.weight": (d, 4 * d, 0), "mlp.c_fc.bias": (1, 4 * d, 0), "mlp.c_proj.weight": (4 * d, d, 0), "mlp.c_proj.bias": (1, d, 0)}
  p = {k: bf16(rng.normal(mean, 0.1 if r == 1 else 0.02, (r, c))) for k, (r, c, mean) in shapes.items()}
  x = bf16(rng.normal(0, 1, (n, d)))
  got = gpt2.layer(Tensor.of(rt, x.tolist()), {k: Tensor.of(rt, v.tolist()) for k, v in p.items()}, Tensor.of(rt, gpt2.mask(n).tolist()))
  got = np.array(got.tolist())
  want = gpt2.layer(x, p, gpt2.mask(n))
  change = np.abs(want - x).max()
  assert change > 1 and np.abs(got - want).max() < 0.03 * change, (np.abs(got - want).max(), change)

@pytest.mark.skipif("GPT2" not in os.environ, reason="GPT2=/path/to/model.safetensors (Hugging Face openai-community/gpt2) runs it")
def test_gpt2_model(chip):
  # goal 1, bigger: GPT-2 small's real weights, all 12 blocks, the embeddings (gathered on the chip) and the output layer
  # (the token embedding transposed on the chip). After "The meaning of life is", the next token's 50257 logits against
  # the same model in numpy with the same bf16 weights (in fp32 it matches Hugging Face's to 1e-4), and a pick numpy
  # rates as good: " not" (407) by numpy, and " to" just 0.36 behind, closer than bf16's rounding through 12 blocks can
  # tell apart (the residual reaches 3000, where bf16 steps by 16). About 30 s on ttsim; test/gpt2.py generates text
  rt = Runtime(chip, chip.tensix[0], chip.tensix[-1])
  weights = gpt2.load(os.environ["GPT2"])
  ids = [464, 3616, 286, 1204, 318]  # "The meaning of life is" (tiktoken's gpt2 encoding)
  got, want = gpt2.logits(ids, gpt2.upload(rt, weights), rt), gpt2.logits(ids, gpt2.arrays(weights))
  err = np.abs(got - want).max()
  assert want.argmax() == 407 and err < 2 and want[got.argmax()] >= want.max() - 2 * err, (err, got.argmax())

def test_softmax_and_layernorm(chip):
  # built from the primitives, each one launch: every worker owns whole rows and sweeps each in passes, reductions
  # staying in Dst and their results in L1. softmax(x) = e / rowsum(e) with e = exp(x - rowmax(x)), three passes;
  # layernorm(x) = (x - mean) * rsqrt(var + eps) * gamma + beta, gamma and beta rows, repeated down.
  # On one worker (3 rows in a row, slots and accumulators reused) and on many
  bf = lambda v: from_bf16(to_bf16(v))
  for workers, rows, cols in ((chip.tensix[1], 96, 128), (chip.tensix[-1], 64, 160)):
    rt = Runtime(chip, chip.tensix[0], workers)
    x = [[bf(random.gauss(0, 2)) for _ in range(cols)] for _ in range(rows)]
    tx = Tensor.of(rt, x)
    e = (tx - tx.rowmax()).exp()
    launches, got = rt.launches, (e * e.rowsum().reciprocal()).tolist()
    assert rt.launches == launches + 1
    for g, row in zip(got, x):
      want = [math.exp(v - max(row)) for v in row]
      assert all(abs(gv - w / sum(want)) <= 0.02 * w / sum(want) + 1e-4 for gv, w in zip(g, want))
    gamma, beta = [bf(random.uniform(0.5, 1.5)) for _ in range(cols)], [bf(random.uniform(-1, 1)) for _ in range(cols)]
    tg, tb = Tensor.of(rt, [gamma]), Tensor.of(rt, [beta])
    d = tx - tx.rowsum() * (1 / cols)
    launches, got = rt.launches, (d * ((d * d).rowsum() * (1 / cols) + 1e-5).rsqrt() * tg + tb).tolist()
    assert rt.launches == launches + 1
    for g, row in zip(got, x):
      mean = sum(row) / cols
      sd = math.sqrt(sum((v - mean) ** 2 for v in row) / cols + 1e-5)
      assert all(abs(gv - ((v - mean) / sd * gm + bt)) <= 0.05 + 0.02 * abs(v) for gv, v, gm, bt in zip(g, row, gamma, beta))
