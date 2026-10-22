# Every step, one test each, on ttsim (TTSIM=/path/to/libttsim.so python3 -m pytest test) or, without TTSIM, on the
# first Blackhole on the PCI bus (as root; NOTES.md, "Bring-up").
import ctypes, json, math, os, pathlib, random, struct, sys, time, pytest
import numpy as np
from ferrite import bh, compiler, pci, tensor
from ferrite.chip import Chip
from ferrite.tensor import Tensor
from ferrite.hello import hello
from ferrite.runtime import DATA, KERNELS, TIMES, Runtime, compile_kernel
import benchmark, gpt2, llama, serve, train
from types import SimpleNamespace

def bf16(a, packer=False):  # numpy, rounded to bf16 and back: to nearest even as we upload, or half up as the packer does
  return gpt2.bf16_floats(gpt2.bf16_bits(a, packer)).astype(np.float64)

def kernels(rt, since):  # launches after launch number `since`, not counting the reads back (tobytes is one)
  return sum(rt.names[n % TIMES] != "readback" for n in range(since + 1, rt.launches + 1))

def close(t, want, tol):  # a tensor's values within tol (relative, and absolute near 0) of numpy's
  got = np.array(t.tolist())
  assert got.shape == want.shape and np.allclose(got, want, rtol=tol, atol=tol), f"off by up to {np.abs(got - want).max()}"

def value(t): return np.array(t.tolist()).reshape(t.shape)  # a tensor as numpy, in its shape
def bf(v): return float(bf16(v))  # a float rounded to bf16, as uploads round
def packed(v): return float(bf16(v, True))  # rounded half up, as the packer does
def gelu(a): return 0.5 * a * (1 + np.tanh(math.sqrt(2 / math.pi) * (a + 0.044715 * a**3)))  # GPT-2's, in numpy

def conv(x, w, padding=0):  # numpy's convolution of x (channels, height, width) by w (out, in, k * k), tap by tap
  k = math.isqrt(w.shape[2])
  x = np.pad(x, ((0, 0), (padding, padding), (padding, padding)))
  ho, wo = x.shape[1] - k + 1, x.shape[2] - k + 1
  return sum(np.einsum("oc,chw->ohw", w[:, :, ky * k + kx], x[:, ky:ky + ho, kx:kx + wo]) for ky in range(k) for kx in range(k))

@pytest.fixture(scope="session")
def chip(): return Chip(pci.open_device())  # one per process: libttsim can only be initialized once

def test_config_space(chip):
  assert chip.pci.cfg_read(0x00) == bh.VENDOR | bh.DEVICE << 16
  assert chip.pci.cfg_read(0x04) & 0x2, "memory decode is off"
  assert chip.pci.cfg_read(0x10) & ~0xF | chip.pci.cfg_read(0x14) << 32, "BAR0 has no address"

def test_arc(chip):
  # ARC's messages and telemetry. A message sent after a response nobody read (left by a client that died) gets its own
  # answer, not the one left: here TEST's, its slot marked as if ARC had written something else there
  chip.arc_msg(0x90)  # TEST
  (x, y), entry = bh.ARC, bh.ARC_QUEUE_ENTRY
  qcb = chip.read32(x, y, bh.ARC_QUEUES)
  base, n = chip.read32(x, y, qcb), chip.read32(x, y, qcb + 4) & 0xFF
  q = base + bh.APP_QUEUE * (entry + 2 * n * entry)
  rptr = (chip.read32(x, y, q + bh.ARC_QUEUE_PTRS[1]) - 1) % (2 * n)
  chip.write32(x, y, q + entry * (1 + n + rptr % n), 0xABCD0000)
  chip.write32(x, y, q + bh.ARC_QUEUE_PTRS[1], rptr)  # its answer, as if never read
  assert chip.arc_msg(0x90)[0] != 0xABCD0000, "TEST's own answer, not the one left in the queue"
  t = chip.telemetry()
  assert len(chip.tensix) in (120, 140) and t[bh.TAG_AICLK] > 0, "14 Tensix columns, 2 harvested from firmware 19.5"

def test_dma(chip):
  # the PCIe tile forwards NoC addresses below 2**58 to the host as bus addresses: point the TLB there and the
  # device reads and writes host memory
  buf, bus = chip.pci.alloc(4096)
  buf[:16] = data = random.randbytes(16)
  assert chip.read(*bh.PCIE, bus, 16) == data
  chip.write(*bh.PCIE, bus + 1024, data[::-1])
  assert bytes(buf[1024:1040]) == data[::-1]

def test_l1(chip):
  # write every target before reading any back, so two targets that alias each other fail; then the same through a TLB
  # of its own (map), which a second ask for that block gets again rather than another (each Runtime's doorbell)
  addr, data = bh.L1_SIZE - 65536, {xy: random.randbytes(65536) for xy in (chip.tensix[0], chip.tensix[-1])}
  for (x, y), d in data.items(): chip.write(x, y, addr, d)
  for (x, y), d in data.items(): assert chip.read(x, y, addr, len(d)) == d, f"Tensix {x},{y}"
  (x, y), d = next(iter(data.items()))
  at = chip.map(x, y, addr)
  assert chip.map(x, y, addr) == at and chip.pci.read(at, 64) == d[:64]

def test_dram(chip):
  addr, data = bh.TLB_WINDOW - 2048, [random.randbytes(4096) for _ in bh.DRAM]  # straddles two 2 MiB windows
  for (x, y), d in zip(bh.DRAM, data): chip.write(x, y, addr, d)
  for bank, ((x, y), d) in enumerate(zip(bh.DRAM, data)):
    for port in range(3): assert chip.read(x, y + port, addr, len(d)) == d, f"bank {bank} port {port}"

def test_cache_prune(tmp_path, monkeypatch):
  # compiled images unused for a week go, once a process, before its first compile; a used one is touched, so it stays,
  # and a folder (an older checkout's tests built one here) is not an image
  import ferrite.runtime as runtime
  monkeypatch.setattr(runtime, "CACHE", tmp_path)
  old, new, folder = tmp_path / "old", tmp_path / "new", tmp_path / "folder"
  old.write_bytes(b"x"), new.write_bytes(b"y"), folder.mkdir()
  for f in (old, folder): os.utime(f, (0, time.time() - 8 * 86400))
  runtime._prune.__wrapped__()
  assert not old.exists() and new.exists() and folder.exists()

def test_card_copies():
  # a card's BARs are read and written a word at a time (8 bytes, or 4 where the address and the length allow no more,
  # a byte only where neither does), since libc's memcpy may go a byte at a time on uncached memory: here on ordinary memory
  buf = (ctypes.c_char * 256)()
  for off, n in ((0, 64), (8, 16), (4, 12), (4, 4), (1, 7), (0, 256)):
    data = random.randbytes(n)
    pci._store(ctypes.addressof(buf) + off, data)
    assert pci._load(ctypes.addressof(buf) + off, n) == data == bytes(buf[off:off + n]), (off, n)
  assert [pci._width(a, n) for a, n in ((0, 64), (4, 12), (1, 7))] == [8, 4, None]

def test_card_wants_root():
  # pci.Card's and pci.reset's first check, which needs no card: without root each says so, rather than failing on a
  # sysfs write later
  if os.geteuid() == 0: pytest.skip("as root, Card would go on to open the device")
  with pytest.raises(PermissionError, match="root"): pci.Card("0000:00:00.0")
  with pytest.raises(PermissionError, match="root"): pci.reset("0000:00:00.0")  # and so does a reset

def test_hello(chip):
  assert set(hello(chip, chip.tensix[0], chip.tensix[-1], 30).values()) == {832040}

def test_every_core(chip):
  # every core but the dispatcher; each answers with its own coordinates, read from its NIU
  rt = Runtime(chip, chip.tensix[0], chip.tensix[-1])
  rt.launch(compile_kernel("void kmain(volatile unsigned *a) { a[0] = noc_me(); }"))
  rt.wait()
  assert len(rt.cores) == len(chip.tensix) - 1  # and one dispatcher
  for x, y in rt.cores: assert chip.read32(x, y, rt.args()) == x | y << 6

def test_relaunch_without_reset(chip):
  # different kernels at the same address, back to back (each loaded after the last is evicted): each must run, not a
  # stale copy of the one before
  rt = Runtime(chip, chip.tensix[0], chip.tensix[2])
  cores = rt.cores
  for i, (op, f) in enumerate([("*", int.__mul__), ("+", int.__add__)] * 2):
    rt.evict()
    rt.launch(compile_kernel(f"void kmain(volatile unsigned *a) {{ a[3] = (noc_me() + a[2]) {op} 3; }}"), (i, 0))
    rt.wait()
    for x, y in cores: assert chip.read32(x, y, rt.args() + 12) == f((x | y << 6) + i, 3), f"launch {i} on {x},{y}"
  # kernels stay loaded until the room is full; then the next load evicts them all (once the queue is done)
  big = [compile_kernel(f"void kmain(volatile unsigned *a) {{ a[1] = a[0] + {k}; }}").ljust(300_000, b"\0") for k in (1, 2)]
  rt.evict()
  for k, image in enumerate(big + big):
    rt.launch(image)
    rt.wait()
    assert rt.loaded == {image: KERNELS} and all(chip.read32(x, y, rt.args() + 4) == i + k % 2 + 1 for i, (x, y) in enumerate(cores))

def test_kernel_moves_data(chip):
  # each core pulls its own 40 KiB slice of DRAM into L1 (3 NoC packets), adds its index to every word, writes it
  # back elsewhere in DRAM, then sends its first 64 bytes to the next core's L1
  rt, n = Runtime(chip, chip.tensix[0], chip.tensix[4]), 40 * 1024  # four workers in the first row
  cores, (bx, by), src, dst = rt.cores, bh.DRAM[3], 1 << 20, 2 << 20
  data = [random.randbytes(n) for _ in cores]
  for i, d in enumerate(data): chip.write(bx, by, src + i * n, d)
  rt.launch(compile_kernel("""
void kmain(volatile unsigned *a) {  // a: [index, 1, bank x, bank y, src, dst, n, workers, each worker's x | y << 6]
  unsigned i = a[0], *buf = (unsigned *)L1_DATA, next = a[8 + (i + 1) % a[7]];
  noc_read(a[2], a[3], a[4] + i * a[6], L1_DATA, a[6]);
  noc_wait();
  for (unsigned k = 0; k < a[6] / 4; k++) buf[k] += i;
  noc_write(L1_DATA, a[2], a[3], a[5] + i * a[6], a[6]);
  noc_write(L1_DATA, next & 63, next >> 6, L1_DATA + a[6], 64);
}"""), (bx, by, src, dst, n, len(cores), *[x | y << 6 for x, y in cores]))
  rt.wait()
  out = [struct.pack(f"<{n // 4}I", *[(w + i) & 0xFFFFFFFF for w in struct.unpack(f"<{n // 4}I", d)]) for i, d in enumerate(data)]
  for i, o in enumerate(out):
    assert chip.read(bx, by, dst + i * n, n) == o, f"DRAM slice {i}"
    assert chip.read(*cores[(i + 1) % len(cores)], DATA + n, 64) == o[:64], f"L1 of the core after {i}"

def test_queue(chip):
  # 100 launches queued back to back through a ring that holds 32 of them (64 bytes each): it fills, the host waits for
  # space, the dispatcher wraps around it three times. Each launch adds its index to a word in every worker's L1.
  rt = Runtime(chip, chip.tensix[0], chip.tensix[2], ring=2048)
  for x, y in rt.cores: chip.write32(x, y, DATA, 0)
  add = compile_kernel("void kmain(volatile unsigned *a) { *(volatile unsigned *)L1_DATA += a[2]; }")
  rt.load(add)  # once, before: then it stays
  writes, write = [], chip.pci.write
  chip.pci.write = lambda *a: (writes.append(a), write(*a))
  try:
    for i in range(100):
      before = len(writes)
      rt.launch(add, (i,))
      assert len(writes) - before == 1, "a launch is one PCIe write: the doorbell"
  finally: chip.pci.write = write
  rt.wait()
  for x, y in rt.cores: assert chip.read32(x, y, DATA) == sum(range(100))

def test_probe(chip):
  # ferrite.probe's kernels all run and report, at small sizes (on ttsim the numbers say nothing about silicon), and the
  # workers' NIUs count at least the reads the DRAM probes say they made (the other probes read too)
  from ferrite import probe
  reads = lambda: sum(chip.read32(x, y, bh.NIU_READS) for x, y in chip.tensix)
  before = reads()
  rows = probe.probe(chip, n=4, rounds=2, upload=1 << 16)
  names = [what for what, *_ in rows]
  assert len(names) == len(set(names)) == 24 and all(value > 0 for _, value, *_ in rows), rows
  workers = len(chip.tensix) - 1
  assert reads() - before >= sum(2 * d * (1 + workers) for d in (1, 2, 4, 8)), "the DRAM probes' reads"

def test_profile(chip):
  # each launch's [go, last report] on the dispatcher's wall clock lands in host memory, and Runtime.profile adds them
  # up by name: three kernels, queued back to back four times, take about the time the host saw (the rest is between
  # launches)
  rt = Runtime(chip, chip.tensix[0], chip.tensix[-1])
  x, y = (Tensor.of(rt, [[float((r + c * k) % 7) for c in range(256)] for r in range(256)]) for k in (1, 2))
  run = lambda: ((x + y).realize(), (x @ y).realize(), x.softmax().realize())
  run(), rt.wait()  # compiled
  since, start = rt.launches, chip.pci.now()
  for _ in range(4): run()
  rows, elapsed = rt.profile(since), (chip.pci.now() - start) * bh.AICLK
  assert sorted((name, n) for name, n, _ in rows) == [("add", 4), ("matmul", 4), ("max2 sub exp sum2 recip mul", 4)], rows
  assert all(cycles > 0 for *_, cycles in rows) and 0.5 * elapsed < sum(cycles for *_, cycles in rows) <= elapsed

def test_jit(chip, monkeypatch):
  # tensor.jit records a function's second call and replays its launches after: each later call copies its inputs in
  # and sends the same launches, building nothing (one realize, the copy, a call), and gets what the function would.
  # A block of matmuls, a softmax and a layer norm, and its DRAM: what the recording took stays where the replays write
  rt = Runtime(chip, chip.tensix[0], chip.tensix[-1])
  rng = np.random.default_rng(7)
  w, v = (Tensor.of(rt, bf16(rng.normal(0, 0.1, (96, 96))).tolist()) for _ in range(2))
  f = lambda x: ((x @ w).relu() @ v).softmax().layernorm() * 2.0
  step, realize, realized = tensor.jit(f), Tensor.realize, []
  for call in range(4):
    x = Tensor.of(rt, bf16(rng.normal(0, 1, (64, 96))).tolist())
    if call == 3: monkeypatch.setattr(Tensor, "realize", lambda t, into=None: (realized.append(t), realize(t, into))[1])
    y = step(x)
    monkeypatch.setattr(Tensor, "realize", realize)
    assert y.tobytes() == f(x).tobytes(), call
  assert len(realized) == 1 and rt.pinned, realized

def test_hung_kernel_times_out(chip):
  # a launch that never finishes is named, with each stuck worker's report: which of its cores are still in it (here
  # BRISC, then TRISC1 and BRISC, which waits for it), its kernel's counters, and its NoC requests in flight
  rt = Runtime(chip, chip.tensix[0], chip.tensix[1])
  rt.launch(compile_kernel("void kmain(volatile unsigned *a) { for (;;); }"))
  with pytest.raises(TimeoutError, match=r"launch 1, still running on \[\(1, 2\)\]\n  \(1, 2\): launch 1 not finished by "
                     r"BRISC\n    SYNC fetched 0, .*; FLAGS set: none\n    NoC requests in flight by id: none"): rt.wait(timeout=0.2)
  rt = Runtime(chip, chip.tensix[0], chip.tensix[1])
  rt.launch(compile_kernel(None, None, "void kmain(volatile unsigned *a) { for (;;); }"))
  with pytest.raises(TimeoutError, match=r"\(1, 2\): launch 1 not finished by BRISC, TRISC1\n"): rt.wait(timeout=0.2)

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
  a, b = ([[bf(random.uniform(-4, 4)) for _ in range(cols)] for _ in range(rows)] for _ in range(2))
  ta, tb = Tensor.of(rt, a), Tensor.of(rt, b)
  assert (ta * tb).tolist() == [[packed(x * y) for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]
  for got, f in (((ta + tb).tolist(), float.__add__), ((ta - tb).tolist(), float.__sub__)):
    for r in range(rows):
      for c in range(cols):
        x, y = a[r][c], b[r][c]
        assert abs(got[r][c] - f(x, y)) <= max(abs(x), abs(y)) * 2**-7, (r, c, x, y, got[r][c])

def test_numbers_once(chip):
  # a number in an elementwise kernel is a tile, read once by each worker into a slot of its own, not once a tile:
  # x * 0.5 + 1.5 over 768 tiles reads x and two tiles a worker
  rt = Runtime(chip, chip.tensix[0], chip.tensix[-1])
  reads = lambda: sum(chip.read32(x, y, bh.NIU_READS) for x, y in rt.cores)
  a = bf16(np.random.default_rng(6).normal(0, 1, (1024, 768)))
  x = Tensor.of(rt, a.tolist())
  y, before = x * 0.5 + 1.5, reads()
  y.realize(), rt.wait()
  assert reads() - before == x.tiles + 2 * len(rt.cores), reads() - before
  close(y, a * 0.5 + 1.5, 1e-2)

def test_matmul_exact(chip):
  # small integers, K = 2 tiles so Dst accumulates across them; then A @ I == A bit for bit, which pins the tile layout
  rt = Runtime(chip, chip.tensix[0], chip.tensix[-1])
  a = [[float((r * 7 + c * 3) % 3 - 1) for c in range(64)] for r in range(32)]
  b = [[float((r * 5 + c) % 3 - 1) for c in range(32)] for r in range(64)]
  assert (Tensor.of(rt, a) @ Tensor.of(rt, b)).tolist() == [[sum(a[i][k] * b[k][j] for k in range(64)) for j in range(32)] for i in range(32)]
  x = [[bf(random.uniform(-8, 8)) for _ in range(64)] for _ in range(96)]
  eye = [[float(r == c) for c in range(64)] for r in range(64)]
  assert (Tensor.of(rt, x) @ Tensor.of(rt, eye)).tolist() == x
  # X @ diag(d) with full-mantissa d: every output is one product, exact in fp32 Dst only if all four fidelity phases
  # ran (each holds a different slice of the mantissas), then rounded once by the packer
  d = [bf(random.uniform(1, 2)) for _ in range(64)]
  diag = [[d[c] if r == c else 0.0 for c in range(64)] for r in range(64)]
  assert (Tensor.of(rt, x) @ Tensor.of(rt, diag)).tolist() == [[packed(v * d[c]) for c, v in enumerate(row)] for row in x]

def test_matmul_random(chip):
  # 64x96 @ 96x128: 8 output tiles on 8 workers, 3 K tiles each. Four fidelity phases make every product exact, Dst sums
  # in fp32, the packer rounds once to bf16: the error stays under one bf16 step of sum |a*b|
  rt, (m, k, n) = Runtime(chip, chip.tensix[0], chip.tensix[-1]), (64, 96, 128)
  a = [[bf(random.uniform(-1, 1)) for _ in range(k)] for _ in range(m)]
  b = [[bf(random.uniform(-1, 1)) for _ in range(n)] for _ in range(k)]
  got = (Tensor.of(rt, a) @ Tensor.of(rt, b)).tolist()
  for i in range(m):
    for j in range(n):
      terms = [a[i][x] * b[x][j] for x in range(k)]
      assert abs(got[i][j] - sum(terms)) <= 2**-8 * sum(map(abs, terms)), (i, j, got[i][j], sum(terms))

def test_matmul_blocks(chip):
  # on one worker, result tiles go in blocks of up to 4 x 4, with the block's rows of A and column of B kept in L1; on
  # every worker, a block each of the grid. 5 x 6 tiles leaves part-blocks at the edges either way, whose missing tiles
  # must not be computed: they would land past the result in DRAM, on the tensor allocated after it (the canary,
  # written before the queued kernel runs)
  a = [[float((r * 7 + c * 3) % 5 - 2) for c in range(96)] for r in range(160)]
  b = [[float((r + 2 * c) % 3 - 1) for c in range(192)] for r in range(96)]
  for workers in (chip.tensix[1], chip.tensix[-1]):
    rt = Runtime(chip, chip.tensix[0], workers)
    y = (Tensor.of(rt, a) @ Tensor.of(rt, b)).realize()  # queued
    canary = Tensor.of(rt, [[1.0] * 192] * 160)
    assert y.tolist() == [[sum(a[i][k] * b[k][j] for k in range(96)) for j in range(192)] for i in range(160)]
    assert canary.tolist() == [[1.0] * 192] * 160
  # one row of result tiles and a long K, as a token's MLP: K is cut into 8 parts on 8 times the workers, and summed
  # (sums under 256, so every partial and total is exact in bf16)
  a, b = [[float((r + c) % 3 - 1) for c in range(1024)] for r in range(5)], [[float(r % 16 == c % 16) for c in range(64)] for r in range(1024)]
  y = Tensor.of(rt, a) @ Tensor.of(rt, b)
  assert y.op == "add" and y.tolist() == [[sum(a[i][k] * b[k][j] for k in range(1024)) for j in range(64)] for i in range(5)]

def test_matmul_fidelity(chip):
  # fewer fidelity phases leave out just the bits compiler.matmul says: A's last significand bit below three phases, B's
  # last three below two. 1 + 2^-7 is 1 and bf16's last bit, so X @ I keeps it from three phases up, I @ X from two;
  # and on random values, three phases stay within the bf16 rounding of four, two within that and 2^-7
  rt = Runtime(chip, chip.tensix[0], chip.tensix[-1])
  x, eye = Tensor.full(rt, 64, 64, 1 + 2**-7), Tensor.of(rt, np.eye(64).tolist())
  for phases in (4, 3, 2, 1):
    assert x.matmul(eye, fidelity=phases).tolist()[0][0] == (1 + 2**-7 if phases >= 3 else 1.0), phases
    assert eye.matmul(x, fidelity=phases).tolist()[0][0] == (1 + 2**-7 if phases >= 2 else 1.0), phases
  rng = np.random.default_rng(4)
  a, b = bf16(rng.normal(0, 1, (96, 256))), bf16(rng.normal(0, 1, (256, 64)))
  ta, tb = Tensor.of(rt, a.tolist()), Tensor.of(rt, b.tolist())
  for phases, bound in ((3, 2**-8), (2, 2**-8 + 2**-7)):
    assert np.all(np.abs(value(ta.matmul(tb, fidelity=phases))[0] - a @ b) <= bound * (np.abs(a) @ np.abs(b))), phases

@pytest.mark.parametrize("dtype", ["bfp8", "bfp4"])
def test_bfp(chip, dtype):
  # a cast to a block float is bit for bit numpy's model of it (bfp above), exponents spread wide within each run; and a
  # matmul by a block-float B (read at 1088 bytes a tile for bfp8, 576 for bfp4, not 2048) is bit for bit the bf16 matmul
  # by its values, whichever way B comes in: streamed beside A (K past 256 tiles), with A's row kept (and shared, a
  # token's), shared by multicast in three buffers, a block's column kept (a batch), and K split; and fused with what
  # follows it and with a bf16 matmul, unpacker 0 switched to the block float for its steps and back
  rt = Runtime(chip, chip.tensix[0], chip.tensix[-1])
  rng = np.random.default_rng(5)
  a = bf16(rng.normal(0, 1, (64, 96)) * np.exp(rng.normal(0, 2, (64, 96))))
  assert np.array_equal(value(Tensor.of(rt, a.tolist()).cast(dtype))[0], gpt2.bfp(a, dtype))
  on = lambda *shape: Tensor.of(rt, rng.normal(0, 1, shape).tolist())
  for b, m, k, n in ((1, 64, 9000, 40), (1, 1, 768, 2304), (1, 100, 256, 200), (2, 100, 256, 96), (1, 32, 2048, 64)):
    x, w = on(b, m, k), on(b, k, n).cast(dtype)
    assert (x @ w).tolist() == (x @ Tensor.of(rt, w.tolist())).tolist(), (b, m, k, n)
  x, w, v, bias = on(64, 192), on(192, 96).cast(dtype).realize(), on(192, 96), on(1, 96)  # K too short to split: one kernel
  launches, y = rt.launches, (x @ w + x @ v + bias).relu().realize()
  assert rt.launches == launches + 1 and y.tolist() == (x @ Tensor.of(rt, w.tolist()) + x @ v + bias).relu().tolist()

def test_matmul_multicast(chip):
  # with a grid of workers (Runtime.grid), a matmul reads each input tile from DRAM once. Of more than one row of tiles:
  # worker (gy, gx) computes one block, each worker of a grid row reads its share of the block row's rows of A and
  # multicasts it along the row, each of a grid column its share of each column of B down it. Of one row (a token's):
  # each worker a block of columns, A's row read by the first grid row and multicast to all, from 2 workers to 96. Odd
  # edges; a K so long that B's three buffers nearly fill the slots below 256; few result tiles of more than one row,
  # whose K stays whole (split K's batch would read its own tiles); B read transposed (a tile a read: the unpacker
  # transposes each face, the math walks the faces transposed); and an epilogue fused after, bit for bit the same as
  # unfused
  rt = Runtime(chip, chip.tensix[0], chip.tensix[-1])
  reads = lambda: sum(chip.read32(x, y, bh.NIU_READS) for x, y in rt.cores)
  rng = np.random.default_rng(3)
  for m, k, n, bt in ((320, 96, 416, False), (288, 1920, 224, False), (64, 512, 256, False), (352, 64, 200, True),
                      (1, 768, 3072, False), (5, 96, 40, True)):
    a, b = bf16(rng.uniform(-1, 1, (m, k))), bf16(rng.uniform(-1, 1, (k, n)))
    tb = Tensor.of(rt, b.T.tolist()).T if bt else Tensor.of(rt, b.tolist())
    y, before = Tensor.of(rt, a.tolist()) @ tb, reads()
    y.realize(), rt.wait()
    tiles = -(-m // 32) * (k // 32) + (k // 32) * -(-n // 32)
    assert reads() - before == tiles, (m, k, n, reads() - before, tiles)
    assert np.all(np.abs(value(y)[0] - a @ b) <= 2**-8 * (np.abs(a) @ np.abs(b))), (m, k, n)
  x, w, d, bias = (Tensor.of(rt, bf16(rng.uniform(-1, 1, shape)).tolist()) for shape in ((96, 64), (64, 200), (96, 200), (1, 200)))
  assert (x @ w * d + bias).tolist() == unfused(x @ w * d + bias).tolist()

def test_matmul_indexed(chip):
  # x.matmul(w, index=ids), a mixture of experts' matmul: batch entry b of the result is row b // k of x (a batch of 1,
  # or of rows times k by repeat_interleave) by the matrix of w that ids picks, ids[b // k, b % k], which BRISC reads on
  # the chip, so a replay reads whatever experts its tokens chose. One token's 8 experts (a grid row each, A's row
  # shared), and 4 tokens' 3 each (12 entries, more than the grid's rows), in bf16 and bfp4: each entry bit for bit the
  # plain matmul by its expert
  rt, rng = Runtime(chip, chip.tensix[0], chip.tensix[-1]), np.random.default_rng(29)
  experts = Tensor.of(rt, rng.normal(0, 0.1, (16, 96, 160)).tolist())
  for wt in ("bf16", "bfp4"):
    w = experts if wt == "bf16" else experts.cast(wt).realize()
    for n, k in ((1, 8), (4, 3)):
      ids, x = rng.integers(0, 16, (n, k)), Tensor.of(rt, rng.normal(0, 1, (n, 1, 96)).tolist())
      got = value((x.repeat_interleave(k) if n > 1 else x).matmul(w, index=Tensor.of(rt, ids.tolist(), "int32")))
      assert all(np.array_equal(got[b], value(x[b // k] @ w[int(ids[b // k, b % k])])[0]) for b in range(n * k)), (wt, n, k)

def test_stream_on_one_worker(chip):
  # one worker gets every tile, so the pipeline really cycles: L1 slots and Dst halves alternate, the threads hand
  # tiles along by semaphore, and BRISC fetches and writes back around them. 32 elementwise tiles, then a 128x96 @ 96x128
  # matmul: 16 output tiles of 3 steps each
  rt = Runtime(chip, chip.tensix[0], chip.tensix[1])
  a, b = ([[bf(random.uniform(-2, 2)) for _ in range(128)] for _ in range(256)] for _ in range(2))
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
  # have ops that read two intermediates, and in the second one an op reads an intermediate made two ops earlier. On one
  # worker each wait holds a thread back, since only the waits do (each TRISC feeds its thread as fast as it can, and
  # ttsim runs every instruction the moment it can): the pack TRISC, with the least to push, has each pack queued before
  # its math is done (SEM_DST); thread 0 reaches the ops that read intermediates before the pack thread has written
  # them (SEM_L1); and the second tree's math has three ops in a row that read only inputs, so the third waits for the
  # pack thread to drain a Dst half (SEM_DST)
  for workers in (chip.tensix[1], chip.tensix[-1]):  # one worker cycling through 12 tiles, then a tile per worker
    rt = Runtime(chip, chip.tensix[0], workers)
    a, b, c, d = (Tensor.of(rt, [[bf(random.uniform(-3, 3)) for _ in range(96)] for _ in range(128)]) for _ in range(4))
    for build in (lambda a, b, c, d: (a + b) * c - d * a, lambda a, b, c, d: (a + b) * ((c - d) + a * a)):
      fused, launches = build(a, b, c, d), rt.launches
      lazy = [t for t in (fused, *fused.srcs, *fused.srcs[0].srcs, *fused.srcs[1].srcs) if t.op]
      got = fused.tolist()
      assert kernels(rt, launches) == 1 and all(t.addr is None for t in lazy[1:]), "one kernel, intermediates stay on chip"
      assert got == unfused(build(a, b, c, d)).tolist()

def test_fused_cast(chip):
  # a cast at the root of a tree is in its kernel: the packer is switched to the result's format for the root's pack and
  # back. Bit for bit what casting the computed value gives (a bf16 kernel's result is rounded first, as its pack would)
  rt, rng = Runtime(chip, chip.tensix[0], chip.tensix[-1]), np.random.default_rng(16)
  x, y = (Tensor.of(rt, rng.normal(0, 2, (70, 100)).astype(np.float32).tolist(), "fp32") for _ in range(2))
  a, b, c = (Tensor.of(rt, rng.normal(0, 2, shape).tolist()) for shape in ((70, 100), (70, 100), (100, 40)))
  for build in (lambda: (x * 3 + y).bfloat16(), lambda: (x * 50 + y).cast("int32"), lambda: (x * y).sum(1).bfloat16(),
                lambda: (x * y - x).cast("bfp8"), lambda: (a * b + a).float(), lambda: a.softmax(-1).float(), lambda: (a @ c).float(),
                lambda: (a * b + a).cast("bfp8")):
    launches, fused = rt.launches, build().realize()
    assert kernels(rt, launches) == 1
    t = build()
    t.srcs[0].realize()
    assert fused.tobytes() == t.realize().tobytes()

def test_assign(chip):
  # assign writes a tensor's new value in place, into its own DRAM, so views of it see the value and a recorded call
  # (jit) can update weights: tile by tile when what computes it reads the tensor elementwise, and through a copy of
  # its own when a matmul reads it, since a matmul reads whole rows and columns of what it may overwrite
  rt, rng = Runtime(chip, chip.tensix[0], chip.tensix[-1]), np.random.default_rng(17)
  w0, m = (bf16(rng.uniform(-1, 1, shape)) for shape in ((64, 96), (96, 96)))
  w, tm = Tensor.of(rt, w0.tolist()), Tensor.of(rt, m.tolist())
  addr, view = w.addr, w[:, 32:64]
  w.assign(w * 2 + 1)
  assert w.addr == addr and np.allclose(value(view)[0], (w0 * 2 + 1)[:, 32:64], rtol=1e-2), "in place: a view sees it"
  fresh = value(Tensor.of(rt, value(w)[0].tolist()) @ tm)
  w.assign(w @ tm)
  assert w.addr == addr and np.array_equal(value(w), fresh), "a matmul of itself, as a fresh one computes it"
  step = tensor.jit(lambda x: w.assign(w + x))
  for _ in range(4): step(Tensor.of(rt, np.ones((64, 96)).tolist()))
  assert np.allclose(value(w), fresh + 4, rtol=1e-2, atol=0.05), "four updates, two of them replayed"

def test_matmul_fusion(chip):
  # matmuls inside elementwise trees, as one kernel: the matmul's result tile is packed into L1 and read from there.
  # Bit for bit the same as one launch per op, since it is rounded to bf16 on its way to L1 as it would be to DRAM
  rnd = lambda r, c: [[bf(random.uniform(-1, 1)) for _ in range(c)] for _ in range(r)]
  for workers in (chip.tensix[1], chip.tensix[-1]):  # one worker cycling through 8 tiles, then a tile per worker
    rt = Runtime(chip, chip.tensix[0], workers)
    a, b, c, d, e, f = (Tensor.of(rt, rnd(*shape)) for shape in ((64, 96), (96, 128), (64, 128), (64, 128), (64, 64), (64, 128)))
    for build in (lambda: (a @ b) + c, lambda: (a @ b) * d - c, lambda: (a @ b) + (e @ f)):
      launches, got = rt.launches, build().tolist()
      assert kernels(rt, launches) == 1, "one kernel"
      assert got == unfused(build()).tolist()

def test_replay_keeps_pushes_small():
  # the TRISCs push one Tensix instruction a cycle (four when they come in a row), so what they push per tile is code
  # they fetch and time they take. The repeating runs (a matmul phase, a mul phase, 15 PACRs) are recorded in the
  # threads' replay buffers once per kernel and replayed by one word each, and an SFPU body's 32 runs a tile are a loop
  # of its one REPLAY word
  c, slots = compiler, compiler.allocate({"mm", "mul"})
  tile = lambda op: len(c.unpack(0, 1)) + len(c.math_begin(op == "mul") + c.elementwise(op, slots) + c.MATH_END) + len(c.pack(0, 0, c.SEM_L1))
  assert tile("add") <= 26 and tile("mul") <= 27, "was 48 and 80 before replay"
  assert len(c.step_unpack(0) + c.matmul(slots)) <= 12, "a matmul step was 71"
  assert len(c.sfpu(["recip"], c.allocate({"recip"}))) <= 3, "32 passes of 12 words: one REPLAY word, in a loop"
  for names in ({"mm", "mul"}, {"exp", "recip", "relu", "mm", "mul"}):  # what is recorded fits the 32-entry buffer
    assert sum(len(c.MATH_RUNS[n]) for n in c.allocate(names)) <= 32

def test_sfpu(chip):
  # relu exactly, exp and 1/x to a bf16 step (they are fp32 on the SFPU, rounded once by the packer), including the
  # edges. The Tensix flushes denormals to zero on the way in and out, so the reference does too: 1/denormal is inf,
  # exp(-100) is 0. On an input (through x + 0), and in place after an add, where exp sees the add's fp32 result: no
  # bf16 rounding in between, so the reference is exp of the exact sum
  rt = Runtime(chip, chip.tensix[0], chip.tensix[2])
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
  assert kernels(rt, launches) == 1 and fused == unfused(((a @ b) + c).relu()).tolist()
  # all three SFPU programs in one kernel: exp's 38 words never fit the 32-entry replay buffer, so its runs go word by
  # word, while recip's and relu's are replayed
  u = [[bf(random.uniform(-3, 3)) for _ in range(64)] for _ in range(64)]
  w = [[bf(random.choice((-1, 1)) * random.uniform(0.5, 2)) for _ in range(64)] for _ in range(64)]
  tu, tw = Tensor.of(rt, u), Tensor.of(rt, w)
  launches, got = rt.launches, (tu.exp().relu() + tw.reciprocal()).tolist()
  assert kernels(rt, launches) == 1, "one kernel"
  for g, ru, rw in zip(got, u, w):
    for gv, uv, wv in zip(g, ru, rw):
      e, r = exp(uv), recip(wv)
      assert abs(gv - (e + r)) <= 2**-6 * max(abs(e), abs(r)), (uv, wv, gv, e + r)

def test_elementwise(chip):
  # tinygrad's primitives on the SFPU (exp2, log2, sin, sqrt; max, cmplt, cmpne; where, two exact picks and an add) and
  # what is built from them, against numpy: to a bf16 step where they compute, exactly where they compare or pick. The
  # edges too: 0, -0, a denormal (the Tensix flushes it to 0), inf, -inf, and negatives where only positives make sense
  # (NaN, which the packer turns into inf)
  rt, rng = Runtime(chip, chip.tensix[0], chip.tensix[-1]), np.random.default_rng(8)
  edges = [0.0, -0.0, 1.0, -1.0, 3e-39, np.inf, -np.inf, 0.5, 2.0, 100.0, -100.0]
  def make(lo, hi, extra=()):
    a = bf16(np.concatenate([edges, extra, rng.uniform(lo, hi, 40 * 64 - len(edges) - len(extra))]).reshape(40, 64))
    return a, np.where(np.abs(a) < 2.0**-126, 0.0, a), Tensor.of(rt, a.tolist())
  def check(t, want, rel, atol=0.0):  # want rounded as the packer rounds (past bf16's range is inf), NaN as inf
    want, got = np.nan_to_num(bf16(want, True), nan=np.inf, posinf=np.inf, neginf=-np.inf), np.array(t.tolist())
    assert ((got == want) | (np.abs(got - want) <= rel * np.abs(want) + atol)).all(), np.argwhere(got != want)[:5]
  with np.errstate(all="ignore"):
    _, fx, tx = make(-8, 8, [1e30, -1e30])
    for f, ref in [(tx.exp2, np.exp2), (tx.exp, np.exp), (tx.log2, np.log2), (tx.log, np.log), (tx.sqrt, np.sqrt)]: check(f(), ref(fx), 2**-7)
    _, fa, ta = make(-20, 20, [np.pi / 2, np.pi])  # within 2^-7 of 1: near its zeros a sine is small, so absolutely
    check(ta.sin(), np.sin(fa), 0, 2**-7)
    check(ta.cos(), np.cos(fa), 0, 2**-7)
  pos = bf16(rng.uniform(0.1, 50, (40, 64)))
  check(Tensor.of(rt, pos.tolist()).rsqrt(), 1 / np.sqrt(pos), 2**-7)
  # comparisons, max and min, where: small integers so that ties happen, signed zeros and infinities
  ints = lambda: bf16(np.concatenate([[0.0, -0.0, np.inf, -np.inf, 3.0], rng.integers(-3, 4, 40 * 64 - 5)]).reshape(40, 64))
  a, b = ints(), ints()
  ta, tb = Tensor.of(rt, a.tolist()), Tensor.of(rt, b.tolist())
  for got, want in [(ta < tb, a < b), (ta > tb, a > b), (ta == tb, a == b), (ta != tb, a != b), (ta <= tb, a <= b), (ta >= tb, a >= b),
                    (ta.maximum(tb), np.maximum(a, b)), (ta.minimum(tb), np.minimum(a, b)), (ta < 1.0, a < 1), (2.0 > ta, 2 > a),
                    ((ta < tb).where(tx, -np.inf), np.where(a < b, fx, -np.inf)), (ta.where(tb, 7.0), np.where(a != 0, b, 7))]:
    assert np.array(got.tolist()).tolist() == np.asarray(want, float).tolist()
  # fused: a comparison, where and two unary ops are one kernel; a masked softmax, where the mask is -inf, not -1e9
  launches, got = rt.launches, (tx > 0).where(tx.log2(), tx.exp2())
  with np.errstate(all="ignore"): check(got, np.where(fx > 0, np.log2(fx), np.exp2(fx)), 2**-7)
  assert kernels(rt, launches) == 1
  s, m = bf16(rng.uniform(-4, 4, (40, 64))), np.tril(np.ones((40, 64)))
  masked = np.where(m > 0, s, -np.inf)
  e = np.exp(masked - masked.max(1, keepdims=True))
  close((Tensor.of(rt, m.tolist()) > 0).where(Tensor.of(rt, s.tolist()), -math.inf).softmax(), e / e.sum(1, keepdims=True), 0.02)

def test_reductions(chip):
  # row max exactly, row sum to a bf16 step of the sum of magnitudes, each a column; then broadcast back:
  # x - max(x) along rows, and a column times a number, all against Python
  rt, rows, cols = Runtime(chip, chip.tensix[0], chip.tensix[-1]), 64, 96
  x = [[bf(random.uniform(-4, 4)) for _ in range(cols)] for _ in range(rows)]
  tx = Tensor.of(rt, x)
  assert tx.max(2).tolist() == [[max(row)] for row in x]
  for (got,), row in zip(tx.sum(2).tolist(), x):
    assert abs(got - sum(row)) <= 2**-8 * sum(map(abs, row)), (got, sum(row))
  for got, row in zip((tx - tx.max(2)).tolist(), x):
    assert all(abs(g - (v - max(row))) <= 2**-7 * max(abs(v), abs(max(row))) for g, v in zip(got, row))
  assert (tx.max(2) * 0.5).tolist() == [[max(row) / 2] for row in x]
  # a kernel sweeps its rows at one width and has two accumulators: reductions of a wider tensor, and a third, run first
  # on their own (found by test_fuzz's longer runs)
  y = [[bf(random.uniform(-4, 4)) for _ in range(200)] for _ in range(rows)]
  ty = Tensor.of(rt, y)
  for (got,), a, b in zip((tx.max(2) + ty.max(2)).tolist(), x, y): assert abs(got - (max(a) + max(b))) <= 2**-6 * 8, (got, a, b)
  for (got,), a in zip((tx.max(2) + tx.sum(2) * 0.0 + (tx * 2.0).max(2)).tolist(), x): assert abs(got - 3 * max(a)) <= 2**-6 * 12

def test_softmax_and_layernorm(chip):
  # built from the primitives, each one launch: every worker owns whole rows and sweeps each in passes, reductions
  # staying in Dst and their results in L1. softmax(x) = e / sum(e) with e = exp(x - max(x)), along each row, three passes;
  # layernorm(x) = (x - mean) * rsqrt(var + eps) * gamma + beta, gamma and beta rows, repeated down.
  # On one worker (3 rows in a row, slots and accumulators reused) and on many
  for workers, rows, cols in ((chip.tensix[1], 96, 128), (chip.tensix[-1], 64, 160)):
    rt = Runtime(chip, chip.tensix[0], workers)
    x = [[bf(random.gauss(0, 2)) for _ in range(cols)] for _ in range(rows)]
    tx = Tensor.of(rt, x)
    launches, got = rt.launches, tx.softmax().tolist()
    assert kernels(rt, launches) == 1
    for g, row in zip(got, x):
      want = [math.exp(v - max(row)) for v in row]
      assert all(abs(gv - w / sum(want)) <= 0.02 * w / sum(want) + 1e-4 for gv, w in zip(g, want))
    gamma, beta = [bf(random.uniform(0.5, 1.5)) for _ in range(cols)], [bf(random.uniform(-1, 1)) for _ in range(cols)]
    tg, tb = Tensor.of(rt, [gamma]), Tensor.of(rt, [beta])
    launches, got = rt.launches, (tx.layernorm() * tg + tb).tolist()
    assert kernels(rt, launches) == 1
    for g, row in zip(got, x):
      mean = sum(row) / cols
      sd = math.sqrt(sum((v - mean) ** 2 for v in row) / cols + 1e-5)
      assert all(abs(gv - ((v - mean) / sd * gm + bt)) <= 0.05 + 0.02 * abs(v) for gv, v, gm, bt in zip(g, row, gamma, beta))

def test_reduce_axes(chip):
  # sum and max down the rows (one row) and through the batch (one matrix), as along each row: one launch each, the
  # padding past the last tile row masked (every value negative, so an unmasked 0 would win a max, and padding rows
  # would add in), the results broadcast back (a column softmax), a mean over everything, and a sum through the batch
  # fused with what reads it. On one worker (a column of tiles cycling through the slots) and on many
  rng = np.random.default_rng(6)
  for workers in (chip.tensix[1], chip.tensix[-1]):
    rt = Runtime(chip, chip.tensix[0], workers)
    x = bf16(rng.uniform(-6, -1, (3, 45, 70)))
    tx = Tensor.of(rt, x.tolist())
    for axis in (0, 1, 2):
      launches, top, total = rt.launches, value(tx.max(axis)), value(tx.sum(axis))
      assert kernels(rt, launches) == 2 and top.tolist() == x.max(axis, keepdims=True).tolist()
      assert np.allclose(total, x.sum(axis, keepdims=True), rtol=2**-8, atol=2**-8), np.abs(total - x.sum(axis, keepdims=True)).max()
    e = np.exp(x - x.max(1, keepdims=True))
    close((tx - tx.max(1)).exp() / (tx - tx.max(1)).exp().sum(1), e / e.sum(1, keepdims=True), 0.02)
    launches, got = rt.launches, tx.sum() * (1 / x.size)
    assert abs(got.tolist()[0][0] - x.mean()) <= 2**-7 * abs(x.mean()) and kernels(rt, launches) == 3, "an axis a launch"
    launches = rt.launches
    close(tx.sum(0) * tx[1] - 1.0, x.sum(0) * x[1] - 1, 2**-6)  # the batch sum, a view and a number in one kernel
    assert kernels(rt, launches) == 1, "fused" 
    y = bf16(rng.uniform(-2, 2, (2, 45, 200)))  # reductions of another length than the result along the axis: first, alone
    ty = Tensor.of(rt, y.tolist())
    close(tx[0:1, :, :].max(1) * ty[:, :, :70] + 1.0, x[0:1].max(1, keepdims=True) * y[:, :, :70] + 1, 2**-6)
    close(tx.max(0) * ty[:, :, :70], x.max(0, keepdims=True) * y[:, :, :70], 2**-7)
    close(tx[0].sum(2) + ty[1], x[0].sum(1, keepdims=True) + y[1], 2**-6)
    narrow = bf16(rng.uniform(-3, -1, (20, 70)))  # one tile tall: reduced on its own, and inside a taller kernel
    tnarrow = Tensor.of(rt, narrow.tolist())
    assert tnarrow.max(1).tolist() == narrow.max(0, keepdims=True).tolist()
    close(tnarrow.sum(1) + tx[2], narrow.sum(0, keepdims=True) + x[2], 2**-6)

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
  assert tneg.max(2).tolist() == neg.max(1, keepdims=True).tolist()
  close(tneg.sum(2), neg.sum(1, keepdims=True), 2**-8)
  e = np.exp(neg - neg.max(1, keepdims=True))
  close((tneg - tneg.max(2)).exp() / (tneg - tneg.max(2)).exp().sum(2), e / e.sum(1, keepdims=True), 0.02)
  narrow, tnarrow = make(45, 20, -3, -1)  # one tile wide: reduced on its own, and inside a three-tile-wide kernel
  assert tnarrow.max(2).tolist() == narrow.max(1, keepdims=True).tolist()
  close(tnarrow.sum(2) + tx, narrow.sum(1, keepdims=True) + x, 2**-6)
  (a, ta), (b, tb) = make(45, 50, -1, 1), make(50, 70, -1, 1)
  close(ta.exp() @ tb.exp(), bf16(np.exp(a)) @ bf16(np.exp(b)), 2**-6)

def test_views_transpose_gelu(chip):
  # views are read in place: tile-aligned slices as inputs to elementwise ops and to matmuls, like a GPT-2 layer's
  # 64-column attention heads. Transpose is exact, of odd shapes and of views, and turns a filled row into a filled
  # column. A repeat of heads (grouped-query attention's) indexed, sliced and repeated again reads the right ones. gelu
  # against numpy, within the bf16 rounding of its steps
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
  heads = ty[:, 0:128].split(2).repeat_interleave(2)  # heads 0, 0, 1, 1
  assert heads[1].tolist() == y[:, 0:64].tolist() and heads[2].tolist() == y[:, 64:128].tolist()
  assert heads[1:3, :, :].tolist() == [y[:, 0:64].tolist(), y[:, 64:128].tolist()]
  assert heads.repeat_interleave(2)[5].tolist() == y[:, 64:128].tolist()  # heads 0, 0, 0, 0, 1, 1, 1, 1
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
  assert tx.max(2).tolist() == x.max(2, keepdims=True).tolist()
  e = np.exp(x - x.max(2, keepdims=True))
  close((tx - tx.max(2)).exp() / (tx - tx.max(2)).exp().sum(2), e / e.sum(2, keepdims=True), 0.02)
  (a, ta), (b, tb), (w, tw) = make(3, 40, 96), make(3, 96, 70), make(96, 70)
  close(ta @ tb, a @ b, 2**-6)
  close(ta @ tw + trow, a @ w + row, 2**-6)
  (qkv, tqkv), (kt, tkt) = make(40, 192), make(192, 40)  # 3 heads of 64: split columns, split rows
  heads, keys = tqkv.split(3), tkt.split(3, axis=0)
  assert heads.shape == (3, 40, 64) and np.array(heads.tolist()).tolist() == qkv.reshape(40, 3, 64).transpose(1, 0, 2).tolist()
  assert np.array(keys.tolist()).tolist() == kt.reshape(3, 64, 40).tolist()
  close(heads @ keys, qkv.reshape(40, 3, 64).transpose(1, 0, 2) @ kt.reshape(3, 64, 40), 2**-6)
  assert np.array(heads.T.tolist()).tolist() == qkv.reshape(40, 3, 64).transpose(1, 2, 0).tolist()  # each matrix, transposed
  h = qkv.reshape(40, 3, 64).transpose(1, 0, 2)
  close(heads @ heads.T, h @ h.transpose(0, 2, 1), 2**-6)  # B transposed as the matmul reads it, no transpose kernel
  (c, tc), (d, td) = make(96, 64), make(160, 64)
  close(tc @ td.T, c @ d.T, 2**-6)  # and in blocks, with B's column kept in L1
  assert heads.merge().tolist() == qkv.tolist()  # side by side again: a copy, and a kernel's result written into its columns
  (m, tm) = make(64, 64)
  close((heads @ tm).merge(), (qkv.reshape(40, 3, 64) @ m).reshape(40, 192), 2**-6)
  one, a = Runtime(chip, chip.tensix[0], chip.tensix[1]), a[:, :20]  # last: it takes two of rt's cores. One worker, and one
  close(Tensor.of(one, a.tolist()) @ Tensor.of(one, b.tolist()), a @ b, 2**-6)  # tile row a matrix: A's row in L1 changes with the matrix

def test_movement(chip):
  # reshape, permute, expand, pad and slices at any offset and step, against numpy bit for bit (a move only copies).
  # Slices of whole tiles stay views; the rest is one launch each, and so is a chain of them. A moved tensor goes home
  # with tidy padding, so a matmul and a reduction can read it. Then random chains of moves, on one worker, whose cache
  # of tiles cycles
  rng = np.random.default_rng(7)
  rt = Runtime(chip, chip.tensix[0], chip.tensix[-1])
  x = bf16(rng.uniform(-2, 2, (3, 45, 70)))
  tx = Tensor.of(rt, x.tolist())
  cases = [(tx.reshape(5, 27, 70), x.reshape(5, 27, 70)), (tx.reshape(63, 150), x.reshape(63, 150)),
           *[(tx.permute(*o), x.transpose(o)) for o in [(1, 0, 2), (2, 1, 0), (1, 2, 0), (2, 0, 1), (0, 2, 1)]],
           (tx[1:3, 5:40:3, ::-1], x[1:3, 5:40:3, ::-1]), (tx[:, 3:4], x[:, :, 3:4]), (tx[2:3, 44:45, :], x[2:3, 44:45, :]),
           (tx[0:1, 7:8, :].expand(3, 45, 70), np.broadcast_to(x[0:1, 7:8, :], (3, 45, 70))),
           (tx.pad(((3, 1), (40, 2))), np.pad(x, ((0, 0), (3, 1), (40, 2))))]
  for t, want in cases: assert value(t).tolist() == want.reshape(t.shape).tolist(), (t.shape, want.shape)
  launches, t = rt.launches, tx.reshape(135, 70)[:, 7:100:2, ::-1].pad(((1, 0), (3, 1), (2, 5))).permute(2, 1, 0)[:, 1:4, 1:2].expand(77, 3, 40)
  want = np.broadcast_to(np.pad(x.reshape(1, 135, 70)[:, 7:100:2, ::-1], ((1, 0), (3, 1), (2, 5))).transpose(2, 1, 0)[:, 1:4, 1:2], (77, 3, 40))
  assert value(t).tolist() == want.tolist() and kernels(rt, launches) == 1, "a chain of moves is one move"
  # the move kernel's ways, in fp32 (a word a datum): whole tiles as they are (a pad on tile boundaries) or transposed, runs
  # along x's rows (a reshape, an unaligned slice) and down its columns (attention's heads apart), datum by datum (a permute)
  y = rng.uniform(-2, 2, (2, 64, 96)).astype(np.float32)
  ty = Tensor.of(rt, y.tolist(), "fp32")
  for t, want in [(ty.T, y.transpose(0, 2, 1)), (ty.pad(((0, 0), (32, 0), (0, 64))), np.pad(y, ((0, 0), (32, 0), (0, 64)))),
                  (ty.reshape(4, 48, 64), y.reshape(4, 48, 64)), (ty[:, 3:50, 5:90], y[:, 3:50, 5:90]),
                  (ty[:, :45, :70].T, y[:, :45, :70].transpose(0, 2, 1)),
                  (ty.reshape(1, 128, 96)._move((16, 96, 8), 0, (768, 1, 96)), y.reshape(16, 8, 96).transpose(0, 2, 1)),
                  (ty.permute(2, 0, 1), y.transpose(2, 0, 1))]:
    assert value(t).tolist() == want.tolist(), t.shape
  launches, v = rt.launches, tx[1:3, 32:45, 32:64]
  assert rt.launches == launches and v.addr == tx.addr and value(v).tolist() == x[1:3, 32:45, 32:64].tolist(), "a view, no copy"
  w = bf16(rng.uniform(-1, 1, (50, 20)))
  close(tx[0][5:40, 10:60] @ Tensor.of(rt, w.tolist()), x[0, 5:40, 10:60] @ w, 2**-6)
  close(tx[:, 3:50, 5:69].sum(1) * (1 / 47), x[:, 3:50, 5:69].sum(1, keepdims=True) / 47, 2**-7)
  one = Runtime(chip, chip.tensix[0], chip.tensix[1])
  def move(t, a):  # a random move of t and the same of numpy's a; batches stay small
    b, r, c = a.shape
    kind = rng.choice(["reshape", "permute", "slice", "pad", "expand" if 1 in a.shape else "slice"])
    if kind == "reshape":
      n = int(rng.choice([d for d in range(1, min(b * r * c, 200) + 1) if b * r * c % d == 0]))
      nb = int(rng.choice([d for d in range(1, 4) if b * r * c // n % d == 0]))
      shape = (nb, b * r * c // n // nb, n)
      return t.reshape(*shape), a.reshape(shape)
    if kind == "permute":
      o = tuple(int(k) for k in rng.permutation(3))
      while a.shape[o[0]] > 4: o = tuple(int(k) for k in rng.permutation(3))
      return t.permute(*o), a.transpose(o)
    if kind == "pad":
      p = [(int(rng.integers(0, 2)), int(rng.integers(0, 2)))] + [(int(rng.integers(0, 20)), int(rng.integers(0, 20))) for _ in range(2)]
      return t.pad(p), np.pad(a, p)
    if kind == "expand":
      shape = tuple(n if n > 1 else int(rng.integers(2, 4)) if ax == 0 else int(rng.choice([2, 33])) for ax, n in enumerate(a.shape))
      return t.expand(*shape), np.broadcast_to(a, shape)
    key = []
    for n in a.shape:
      lo = int(rng.integers(0, n))
      hi = int(rng.integers(lo + 1, n + 1))
      step = int(rng.choice([1, 1, 2, 3, -1]))
      key.append(slice(hi - 1, lo - 1 if lo else None, step) if step < 0 else slice(lo, hi, step))
    return t[tuple(key)], a[tuple(key)]
  for _ in range(8):
    shape = (int(rng.choice([1, 2, 3])), int(rng.choice([1, 20, 45])), int(rng.choice([1, 20, 70])))
    a = bf16(rng.uniform(-2, 2, shape))
    t = Tensor.of(one, a.tolist() if shape[0] > 1 else a[0].tolist())
    for _ in range(3):
      t, a = move(t, a)
      assert value(t).tolist() == a.reshape(t.shape).tolist(), a.shape

def test_fp32(chip):
  # fp32 tensors, 4 KiB tiles: the math thread unpacks them straight into Dst and the SFPU does the math, so add, sub, mul,
  # comparisons, where, movement and maxes are numpy's float32 exactly and sums are within its rounding; exp, log, sin
  # and the like are within a step or two of fp32. Casts both ways (to bf16 rounds as the packer does), bf16 with fp32 is
  # fp32, and a number takes its tensor's dtype, so a mean of 768 values is exact. A matmul sums in fp32, but the FPU
  # adds each group of 8 products at 12 bits (ttsim's model of it): fp32 results are good to about 2^-11, and fp32
  # operands (bf16x3) reach that too rather than bf16's 2^-8. On one worker (the pipeline cycles) and on many
  rng = np.random.default_rng(9)
  f32 = lambda *shape: rng.uniform(-2, 2, shape).astype(np.float32)
  for workers in (chip.tensix[1], chip.tensix[-1]):
    rt = Runtime(chip, chip.tensix[0], workers)
    x, y = f32(2, 45, 70), f32(2, 45, 70)
    tx, ty = Tensor.of(rt, x.tolist(), "fp32"), Tensor.of(rt, y.tolist(), "fp32")
    launches, got = rt.launches, value((tx + ty) * tx - ty * 1.5)
    assert kernels(rt, launches) == 1 and got.tolist() == ((x + y) * x - y * np.float32(1.5)).tolist(), "one kernel, fp32's rounding"
    for got, want in [(tx, x), ((tx < ty).where(tx, ty * 0.5), np.where(x < y, x, y * np.float32(0.5))), (tx.max(1), x.max(1, keepdims=True)),
                      (tx.T, x.transpose(0, 2, 1)), (tx[1][3:40:2, 5:66], x[1, 3:40:2, 5:66]), (tx[0].gather([44, 0, 7]), x[0, [44, 0, 7]])]:
      assert value(got).tolist() == np.asarray(want, np.float32).reshape(got.shape).tolist()
    assert np.allclose(value(tx.sum(2)), x.sum(2, keepdims=True), rtol=1e-6, atol=1e-5)
    w, v = np.float64(x * np.float32(4)), np.float64(x * x + np.float32(0.5))  # the transcendentals to a step or two of fp32 (2^-23 is
    for got, want, scale in [((tx * 4.0).exp(), np.exp(w), np.exp(w)), ((tx * 4.0).exp2(), np.exp2(w), np.exp2(w)),  # 1.2e-7): relative,
                             ((tx * tx + 0.5).log(), np.log(v), np.abs(np.log(v))), ((tx * tx + 0.5).log2(), np.log2(v), np.abs(np.log2(v))),
                             ((tx * tx + 0.5).sqrt(), np.sqrt(v), np.sqrt(v)), ((tx * tx + 0.5).rsqrt(), 1 / np.sqrt(v), 1 / np.sqrt(v)),
                             ((tx * 32.0).sin(), np.sin(w * 8), 1), ((tx * 32.0).cos(), np.cos(w * 8), 1)]:  # sin and cos absolute
      assert (np.abs(value(got) - want) / scale).max() <= 3e-7, (np.abs(value(got) - want) / scale).max()
    s = np.array([[np.nan, np.inf, -np.inf, 0, -2, 2]], np.float32)  # NaN goes through, and is equal to nothing, itself too
    ts = Tensor.of(rt, s.tolist(), "fp32")
    with np.errstate(all="ignore"):
      for got, want in [(ts.exp(), np.exp(s)), (ts.exp2(), np.exp2(s)), (ts.log(), np.log(s)), (ts.sqrt(), np.sqrt(s)),
                        (ts.reciprocal(), 1 / s), (ts.sin(), np.sin(s)), (ts.ne(ts), s != s), (ts < ts * 2, s < s * 2)]:
        assert np.allclose(value(got), np.asarray(want, np.float32).reshape(got.shape), rtol=1e-6, atol=0, equal_nan=True), (value(got), want)
    b = bf16(x)
    assert value(tx.bfloat16()).tolist() == bf16(x, True).tolist() and value(Tensor.of(rt, b.tolist()).float()).tolist() == b.tolist()
    assert value(Tensor.of(rt, b.tolist()) + tx).tolist() == (b.astype(np.float32) + x).tolist(), "bf16 with fp32 is fp32"
    assert value(Tensor.of(rt, np.full((1, 768), 3.0).tolist(), "fp32").mean()).tolist() == [[[3.0]]]
    a, w = f32(64, 96), f32(96, 70)
    ta, tw = Tensor.of(rt, a.tolist(), "fp32"), Tensor.of(rt, w.tolist(), "fp32")
    want = bf16(a) @ bf16(w)
    assert np.abs(value(Tensor.of(rt, bf16(a).tolist()).matmul(Tensor.of(rt, bf16(w).tolist()), "fp32")) - want).max() <= 2**-11 * np.abs(want).max()
    want = a.astype(np.float64) @ w
    assert np.abs(value(ta @ tw) - want).max() <= 2**-11 * np.abs(want).max()

def test_int32(chip):
  # int32 tensors, two's complement through L1, Dst and the SFPU (which move it raw: stored as INT32, since the FP32
  # store flushes denormals, the small positive integers). add and sub wrap as int32 does; max, min, comparisons (1 or 0)
  # and where are exact over all of int32; multiplies and sums go through fp32, exact below 2^24. Casts both ways (to
  # an integer toward zero), a float number makes fp32, a float condition picks ints, and gather reads its indices
  # from an int32 tensor on the chip, so they can be computed there. On one worker and on many
  rng = np.random.default_rng(10)
  wrap = lambda a: (a + 2**31) % 2**32 - 2**31
  for workers in (chip.tensix[1], chip.tensix[-1]):
    rt = Runtime(chip, chip.tensix[0], workers)
    x, y, s = rng.integers(-2**31 + 1, 2**31, (45, 70)), rng.integers(-2**31 + 1, 2**31, (45, 70)), rng.integers(-99, 100, (45, 70))
    x[0, :5], s[0, :3] = [0, -1, 1, 2**31 - 1, -2**31 + 1], [3, 0, -3]
    tx, ty, ts = (Tensor.of(rt, a.tolist(), "int32") for a in (x, y, s))
    f = rng.uniform(-2**30, 2**30, (45, 70)).astype(np.float32)
    f[0, :6] = [0.0, -0.0, 0.7, -0.7, 2.5, -1e9]
    tf = Tensor.of(rt, f.tolist(), "fp32")
    for got, want in [(tx, x), (tx + ty, wrap(x + y)), (tx - ty, wrap(x - y)), (tx.maximum(ty), np.maximum(x, y)), (tx.minimum(ty), np.minimum(x, y)),
                      (tx < ty, x < y), (tx != ts, x != s), (ts == 3, s == 3), (ts >= 0, s >= 0), ((ts > 0).where(tx, ty), np.where(s > 0, x, y)),
                      (ts * ts - 1, s * s - 1), (ts.sum(1), s.sum(0, keepdims=True)), (ts.max(), s.max(keepdims=True)), (ts.relu(), np.maximum(s, 0)),
                      (ts.float(), s), (ts.bfloat16(), s), (tf.cast("int32"), np.trunc(f)), ((tf > 0).where(ts, 0), np.where(f > 0, s, 0)),
                      (tf.gather(Tensor.of(rt, [[44, 0, 3, 44]], "int32")), f[[44, 0, 3, 44]])]:
      assert value(got).tolist() == np.asarray(want, np.float64).reshape(got.shape).tolist(), (got.shape, got.dtype)
    assert (ts * 0.5).dtype == "fp32" and (ts > 0).dtype == "int32" and value(ts * 0.5).tolist() == (s * 0.5).reshape(1, 45, 70).tolist()

def test_autograd(chip):
  # every op's backward against numpy: the loss sum(f(inputs) * w), w random so every element of the gradient differs,
  # and its gradient by central differences in float64. fp32 tensors, odd shapes, broadcasting; matmuls to the FPU's
  # 2^-11, the rest to fp32's rounding. Then a bf16 leaf (its gradient comes back through the cast), a tensor read twice
  # (gradients add), and .grad adding up over two backward passes
  rt, rng = Runtime(chip, chip.tensix[0], chip.tensix[-1]), np.random.default_rng(11)
  softmax = lambda a, ax: np.exp(a - a.max(ax, keepdims=True)) / np.exp(a - a.max(ax, keepdims=True)).sum(ax, keepdims=True)
  norm = lambda a: (a - a.mean(2, keepdims=True)) / np.sqrt(a.var(2, keepdims=True) + 1e-5)
  ew, row, col, pos = ((2, 20, 40), (-1, 1)), ((1, 1, 40), (-1, 1)), ((1, 20, 1), (-1, 1)), ((2, 20, 40), (0.5, 2))
  cases = [  # (ferrite, numpy, each input's shape and range[, tolerance: 2e-4 unless said])
    (lambda a, b: a + b, lambda a, b: a + b, [ew, col]), (lambda a, b: a - b, lambda a, b: a - b, [row, ew]),
    (lambda a, b: a * b, lambda a, b: a * b, [ew, row]), (lambda a, b: a / b, lambda a, b: a / b, [ew, pos]),
    (lambda a, b: a.maximum(b), np.maximum, [ew, ew]), (lambda a, b: a.minimum(b * 0.5), lambda a, b: np.minimum(a, b * 0.5), [ew, col]),
    (lambda a, b: (a > b).where(a * 2.0, b), lambda a, b: np.where(a > b, a * 2, b), [ew, row]),
    *[(getattr(Tensor, f), g, [r]) for f, g, r in [
      ("exp", np.exp, ew), ("exp2", np.exp2, ew), ("log", np.log, pos), ("log2", np.log2, pos), ("sin", np.sin, ew), ("cos", np.cos, ew),
      ("sqrt", np.sqrt, pos), ("rsqrt", lambda a: 1 / np.sqrt(a), pos), ("reciprocal", lambda a: 1 / a, pos),
      ("relu", lambda a: np.maximum(a, 0), ew), ("gelu", gelu, ew)]],
    *[(lambda a, ax=ax: a.sum(ax), lambda a, ax=ax: a.sum(ax, keepdims=True), [ew]) for ax in (0, 1, 2)],
    *[(lambda a, ax=ax: a.max(ax), lambda a, ax=ax: a.max(ax, keepdims=True), [ew]) for ax in (0, 1, 2)],
    (lambda a: a.mean(), lambda a: a.mean(keepdims=True), [ew]), (lambda a: a.softmax(), lambda a: softmax(a, 2), [ew]),
    (lambda a: a.softmax(1), lambda a: softmax(a, 1), [ew]), (lambda a: a.layernorm(), norm, [ew]),
    (lambda a, b: a @ b, lambda a, b: a @ b, [((2, 20, 40), (-1, 1)), ((1, 40, 30), (-1, 1))], 2**-11),
    (lambda a, b: a @ b.T, lambda a, b: a @ b.swapaxes(1, 2), [((2, 20, 40), (-1, 1)), ((2, 30, 40), (-1, 1))], 2**-11),
    (lambda a: a.T, lambda a: a.swapaxes(1, 2), [ew]), (lambda a: a.reshape(4, 25, 16), lambda a: a.reshape(4, 25, 16), [ew]),
    (lambda a: a.permute(2, 0, 1), lambda a: a.transpose(2, 0, 1), [ew]),
    (lambda a: a.expand(3, 20, 40), lambda a: np.broadcast_to(a, (3, 20, 40)), [col]),
    (lambda a: a.pad(((1, 2), (3, 0), (0, 5))), lambda a: np.pad(a, ((1, 2), (3, 0), (0, 5))), [ew]),
    (lambda a: a[:, 3:17, ::-1], lambda a: a[:, 3:17, ::-1], [ew]), (lambda a: a[1], lambda a: a[1:2], [ew]),
    (lambda a: a.split(2), lambda a: a.reshape(1, 20, 2, 32).transpose(2, 1, 3, 0).reshape(2, 20, 32), [((1, 20, 64), (-1, 1))]),
    (lambda a: a.merge(), lambda a: a.transpose(1, 0, 2).reshape(1, 20, 128), [((2, 20, 64), (-1, 1))]),
    (lambda a: a.gather([3, 0, 7, 3]), lambda a: a[:, [3, 0, 7, 3]], [((1, 9, 40), (-1, 1))]),
    (lambda a, b: a.cat(b, axis=2), lambda a, b: np.concatenate([a, b], 2), [ew, ((2, 20, 13), (-1, 1))]),
    (lambda a, b: b.cat(a, a, axis=0), lambda a, b: np.concatenate([b, a, a], 0), [ew, ((1, 20, 40), (-1, 1))]),
    (Tensor.silu, lambda a: a / (1 + np.exp(-a)), [ew]),
    (lambda a, b: a.conv2d(b, 1), lambda a, b: conv(a, b, 1), [((3, 9, 10), (-1, 1)), ((4, 3, 9), (-1, 1))], 2**-11),
    (Tensor.max_pool2d, lambda a: a.reshape(2, 10, 2, 20, 2).max((2, 4)), [ew])]
  for n, (f, g, specs, *tol) in enumerate(cases):
    xs = [rng.uniform(lo, hi, shape) for shape, (lo, hi) in specs]
    ts = [Tensor.of(rt, x.astype(np.float32).tolist() if x.shape[0] > 1 else x[0].astype(np.float32).tolist(), "fp32").requires_grad_() for x in xs]
    xs = [np.float64(np.float32(x)) for x in xs]
    out = g(*xs)
    w = rng.uniform(-1, 1, out.shape)
    loss = (f(*ts) * Tensor.of(rt, w.tolist() if w.shape[0] > 1 else w[0].tolist(), "fp32")).sum()
    loss.backward()
    for i, (x, t) in enumerate(zip(xs, ts)):
      want = np.zeros_like(x)
      for at in np.ndindex(x.shape):  # d loss / d x[at], centrally
        hi_, lo_ = [v.copy() for v in xs], [v.copy() for v in xs]
        hi_[i][at] += 1e-6; lo_[i][at] -= 1e-6
        want[at] = ((g(*hi_) * w).sum() - (g(*lo_) * w).sum()) / 2e-6
      got = value(t.grad)
      assert np.abs(got - want).max() <= (tol or [2e-4])[0] * (np.abs(want).max() + 1), (n, i, np.abs(got - want).max())
  a, b, w = rng.uniform(-1, 1, (20, 40)), bf16(rng.uniform(-1, 1, (20, 40))), rng.uniform(-1, 1, (20, 40))
  ta, tb, tw = Tensor.of(rt, a.tolist(), "fp32").requires_grad_(), Tensor.of(rt, b.tolist()).requires_grad_(), Tensor.of(rt, w.tolist(), "fp32")
  ((ta * tb + ta * ta) * tw).sum().backward()
  assert tb.grad.dtype == "bf16" and np.allclose(value(tb.grad)[0], a * w, rtol=2**-8, atol=2**-8), "a bf16 leaf, through the cast"
  assert np.allclose(value(ta.grad)[0], (b + 2 * a) * w, rtol=1e-6, atol=1e-6), "ta read twice: the gradients add"
  (ta * tw).sum().backward()
  assert np.allclose(value(ta.grad)[0], (b + 2 * a) * w + w, rtol=1e-6, atol=1e-6), ".grad adds up"
  assert np.allclose(value((ta * ta).sum().gradient(ta)[0])[0], 2 * a, rtol=1e-6), "gradient() gives it without touching .grad"

def test_fuzz(chip, monkeypatch):
  # random expressions (elementwise ops with broadcasting, unary ops, reductions along any axis, matmuls; odd shapes,
  # batches), and every kernel they turn into checked against numpy on the same inputs, read back from the chip:
  # whatever bf16 rounding piles up through a tree, each kernel must be right about its own inputs
  rt, checked = Runtime(chip, chip.tensix[0], chip.tensix[-1]), []
  def numpy(e, xs):
    if isinstance(e, int): return xs[e]
    v = [numpy(s, xs) for s in e[1:]]
    f = {"add": np.add, "sub": np.subtract, "mul": np.multiply, "matmul": np.matmul, "T": lambda a: a.swapaxes(-1, -2),
         "relu": lambda a: np.maximum(a, 0), "exp": np.exp, "exp2": np.exp2, "log": np.log, "log2": np.log2, "sin": np.sin,
         "cos": np.cos, "sqrt": np.sqrt, "recip": lambda a: 1 / a, "max": np.maximum, "cmplt": lambda a, b: (a < b) * 1.0,
         "cmpne": lambda a, b: (a != b) * 1.0, "keep": lambda c, x: np.where(c != 0, x, 0), "drop": lambda c, y: np.where(c != 0, 0, y),
         "cast": lambda a: a, "ftob": lambda a: ((np.float32(a).view(np.uint32) + 0x8000) & 0xFFFF0000).view(np.float32)}
    f |= {k: lambda a, k=k: getattr(a, k[:3])(int(k[3]), keepdims=True) for k in compiler.FOLDS}  # sum0, max2, ...
    return f[e[0]](*v)
  realize = Tensor.realize
  def checked_realize(t):
    if t.op in (None, "T", "move"): return realize(t)  # data movement is test_movement's
    tensor._fit(t)  # the parts realize computes first (a long matmul, a shared value, a tree too big), each checked too
    inputs = []
    e, _ = tensor._tree(t, inputs)
    xs = [value(u) for u in inputs]
    realize(t)
    want, got = numpy(e, xs), value(t)
    assert np.abs(got - want).max() <= 0.03 * (np.abs(want).max() + 1), (e, np.abs(got - want).max())
    checked.append(e)
    return t
  monkeypatch.setattr(Tensor, "realize", checked_realize)
  rng = np.random.default_rng(5)
  dtype = "bf16"
  def leaf(shape):  # some all negative, so an unmasked 0 in the padding would win a max
    lo = rng.choice([-2.0, -4.0])
    a = rng.uniform(lo, lo + 4 if lo == -2 else lo + 3, shape)
    a = bf16(a) if dtype == "bf16" else a.astype(np.float32)
    return Tensor.of(rt, a.tolist() if shape[0] > 1 else a[0].tolist(), dtype)
  def gen(shape, depth):
    b, r, c = shape
    kind = "leaf" if depth == 0 or rng.random() < 0.2 else rng.choice(["bin", "bin", "unary", "red", "mm", "where"])
    if kind in ("bin", "where"):  # each operand keeps a dimension, or has 1 there (not both)
      s1 = tuple(1 if rng.random() < 0.2 else d for d in shape)
      s2 = tuple(d if s1[i] == 1 else (1 if rng.random() < 0.2 else d) for i, d in enumerate(shape))
      # comparisons only of inputs: between computed values a rounding step flips them, so numpy cannot check them
      if kind == "where": return (leaf(shape) > 0).where(gen(s1, depth - 1), gen(s2, depth - 1))
      return getattr(gen(s1, depth - 1), rng.choice(["__add__", "__sub__", "__mul__", "maximum", "minimum"]))(gen(s2, depth - 1))
    if kind == "unary":  # sin and cos of a value held to +-8: of exp's 4e9 they are noise in fp32, and in numpy's float64
      x = gen(shape, depth - 1)
      return rng.choice([x.relu, (x * 0.5).exp, (x * 0.5).exp2, x.maximum(-8.0).minimum(8.0).sin, x.maximum(-8.0).minimum(8.0).cos,
                         (x * x + 1.0).reciprocal, (x * x + 1.0).rsqrt, (x * x + 1.0).log2, (x * x + 1.0).sqrt])()
    if kind == "red" and 1 in shape:  # along an axis that is 1 here (the batch, rows or columns). Values stay near 1
      axis = int(rng.choice([ax for ax, d in enumerate(shape) if d == 1]))  # (means, not sums), or exp would turn
      n = int(rng.choice([2, 3] if axis == 0 else [20, 32, 70, 96, 192]))   # rounding into large errors
      x = gen(tuple(n if ax == axis else d for ax, d in enumerate(shape)), depth - 1)
      return x.sum(axis) * (1 / n) if rng.random() < 0.5 else x.max(axis)
    if kind == "mm" and r > 1 and c > 1:
      k = int(rng.choice([32, 50, 64]))
      bb = b if rng.random() < 0.5 else 1
      return (leaf((b, r, k)) @ (leaf((bb, c, k)).T if rng.random() < 0.3 else leaf((bb, k, c)))) * (1 / k)  # B read transposed, or not
    return leaf(shape)
  for n in range(int(os.environ.get("FUZZ", 30))):  # every other tree fp32; sizes past 4 tiles make part-blocks of batched
    # matmuls (unbatched ones share by multicast)
    shape = (int(rng.choice([1, 1, 2])), int(rng.choice([1, 20, 32, 45, 64, 160])), int(rng.choice([1, 20, 32, 70, 96, 192])))
    dtype = ("bf16", "fp32")[n % 2]
    gen(shape, 4).realize()
  assert len(checked) >= 20 and any(e[0] == "matmul" or "sum" in str(e) for e in checked)

def test_mnist_mlp(chip):
  # an MNIST-shaped network, 784 -> 128 -> 10 with relu and biases, on a batch of 16, against
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
  assert kernels(rt, launches) == 2
  want = np.maximum(x @ w1 + b1, 0) @ w2 + b2
  assert np.allclose(logits, want, rtol=0.02, atol=0.02), np.abs(logits - want).max()
  top2 = np.sort(want, 1)[:, -2:]
  clear = top2[:, 1] - top2[:, 0] > 0.05
  assert clear.sum() >= 8 and (logits.argmax(1) == want.argmax(1))[clear].all()

def test_mnist_cnn(chip):
  # a small CNN: MNIST-shaped images through a 3x3 convolution (1 -> 8 channels, padded) and relu, a 2x2 max pool, a
  # 3x3 convolution (8 -> 16) and relu, a pool, and 784 -> 10, against numpy in float64; random weights, as above. Each
  # convolution is one matmul by the stacked patches, each of whose 9 taps is one move
  rt, rng = Runtime(chip, chip.tensix[0], chip.tensix[-1]), np.random.default_rng(3)
  w1, b1 = bf16(rng.normal(0, 1 / 3, (8, 1, 9))), bf16(rng.normal(0, 0.1, (8, 1, 1)))
  w2, b2 = bf16(rng.normal(0, 72**-0.5, (16, 8, 9))), bf16(rng.normal(0, 0.1, (16, 1, 1)))
  w3, b3 = bf16(rng.normal(0, 784**-0.5, (784, 10))), bf16(rng.normal(0, 0.1, (1, 10)))
  tw1, tb1, tw2, tb2, tw3, tb3 = (Tensor.of(rt, a.tolist()) for a in (w1, b1, w2, b2, w3, b3))
  pool = lambda a: a.reshape(a.shape[0], a.shape[1] // 2, 2, a.shape[2] // 2, 2).max((2, 4))
  for _ in range(2):
    x = bf16(rng.uniform(0, 1, (1, 28, 28)))
    launches = rt.launches
    h = ((Tensor.of(rt, x[0].tolist()).conv2d(tw1, 1) + tb1).relu().max_pool2d().conv2d(tw2, 1) + tb2).relu().max_pool2d()
    logits = np.array((h.reshape(1, 784) @ tw3 + tb3).tolist())
    assert rt.launches - launches <= 50, rt.launches - launches
    h = pool(np.maximum(conv(pool(np.maximum(conv(x, w1, 1) + b1, 0)), w2, 1) + b2, 0))
    want = h.reshape(1, 784) @ w3 + b3
    assert np.allclose(logits, want, rtol=0.02, atol=0.02), np.abs(logits - want).max()

def test_train(chip):
  # training: a small GPT (GPT-2's blocks, 2 of them, 64 wide, 64 characters of context) learning to count
  # in fp32 with Adam (test/train.py learns README and LESSONS, which change with every edit: once, a new text put one
  # gradient past the bound). Every step's loss and gradients against PyTorch's on the same weights (to 1e-3, and 2e-3
  # of the largest: the matmul's 2^-11, compounded), the step recorded on its second call and replayed from the third (tensor.jit; its kernels do not fit in L1
  # together, so a replay sends them in parts); the loss falling; and the first step the same as PyTorch's own
  pytest.importorskip("torch")
  losses = train.train(Runtime(chip, chip.tensix[0], chip.tensix[-1]), 5, data=" ".join(map(str, range(3000))))
  assert all(abs(f - p) < 1e-3 and g < 2e-3 for f, p, g, _ in losses), losses
  assert losses[-1][0] < losses[0][0] - 0.25 and abs(losses[0][0] - losses[0][3]) < 1e-3, losses

def test_tinygrad(chip):
  # tinygrad's frontend on ferrite. tinygrad builds the graph (its Tensor, nn, autograd, Adam) and
  # ferrite runs it: elementwise ops, reductions along any axis, movement of 3-D views, a matmul written as tinygrad
  # writes one (a broadcast multiply, summed), softmax, a tinygrad function (inlined); integers and bools, a view of part
  # of a buffer, stores into parts of one, and tensors with nothing in them; then an MLP trained with Adam for three
  # steps, and a layer for five under TinyJit, replayed. All against tinygrad on its own CPU device, and on the chip
  # (launches), not the host
  if "TINYGRAD" in os.environ: sys.path.insert(0, os.environ["TINYGRAD"])  # (TINYGRAD=/path/to/tinygrad, or installed)
  pytest.importorskip("tinygrad")
  from adapters import tinygrad_ferrite
  from tinygrad import Tensor, TinyJit, dtypes, function, nn
  from tinygrad.helpers import Context
  rt = tinygrad_ferrite.RT = tinygrad_ferrite.RT or Runtime(chip, chip.tensix[0], chip.tensix[-1])  # FERRITE on this process's chip
  rng = np.random.default_rng(14)
  a, b, w = (rng.standard_normal(shape).astype(np.float32) for shape in ((2, 3, 40), (2, 3, 40), (40, 5)))
  cases = [lambda x, y, w: x * 2 + y, lambda x, y, w: x.sum(1), lambda x, y, w: x.max(-1), lambda x, y, w: x.exp().log(),
           lambda x, y, w: (x > 0).where(x, y), lambda x, y, w: x @ w, lambda x, y, w: x.softmax(-1), lambda x, y, w: x.permute(2, 0, 1) - y.mean(),
           lambda x, y, w: x.pad(((0, 0), (1, 2), (0, 0))) * 1.5, lambda x, y, w: x[:, 1:3, ::2].relu(),
           lambda x, y, w: function(lambda p, q: p * 2 + q)(x, y)]
  for n, f in enumerate(cases):
    want = f(*(Tensor(v, device="CPU") for v in (a, b, w))).numpy()
    launches, got = rt.launches, f(*(Tensor(v, device="FERRITE") for v in (a, b, w))).numpy()
    assert rt.launches > launches and got.shape == want.shape, n
    assert np.abs(got - want).max() <= 1e-3 * (np.abs(want).max() + 1), (n, np.abs(got - want).max())  # a matmul to the FPU's 2^-11
  Tensor.ones(2, device="FERRITE").contiguous().realize().tolist()  # 8 bytes, freed: the next buffer of 8 bytes (this
  assert Tensor.ones((), dtype=dtypes.int64, device="FERRITE").contiguous().item() == 1  # int64) has nothing of it
  ints = Tensor([[1, -8, 1], [32, 1, 6]], dtype=dtypes.int32, device="FERRITE")  # products come back int32, bitwise and on the host
  assert (ints * ints).tolist() == [[1, 64, 1], [1024, 1, 36]] and (ints & 12).tolist() == [[0, 8, 0], [0, 0, 4]]
  assert ints.argmax(1).tolist() == [0, 0]
  made = Tensor([3, 3], dtype=dtypes.int64, device="FERRITE").cast(dtypes.bfloat16)  # bf16 made on the host: up as values
  assert (made * Tensor([1.0, 2.0], dtype=dtypes.bfloat16, device="FERRITE")).tolist() == [3.0, 6.0]
  assert (Tensor(a, device="FERRITE") > 0).any(1).tolist() == (a > 0).any(1).tolist()  # bools read back a byte each
  assert (Tensor(np.arange(8, dtype=np.float32), device="FERRITE")[:4] * 2).tolist() == [0, 2, 4, 6]  # a view of part of a tensor
  assert (ints.T @ ints).tolist() == (np.array(ints.tolist()).T @ np.array(ints.tolist())).tolist()  # summed in fp32, no FPU
  t = Tensor(a, device="FERRITE").realize()
  assert t[1, 1:3].numpy().tolist() == a[1, 1:3].tolist()  # a buffer read through tinygrad's view of part of it (_offset)
  assert t[:, :0].sum(1).numpy().tolist() == [[0] * 40] * 2 and t[:, :0].numpy().shape == (2, 0, 40)  # nothing to compute
  assert (t[0, 0, :1].pad(((1, 0),))[0:1] * t[0, 0]).numpy().tolist() == [0] * 40  # padding stays padding, broadcast too
  t, want = Tensor.zeros(6, 6, device="FERRITE").contiguous().realize(), np.zeros((6, 6))  # stores into parts of a buffer
  t[2:4, 3:5] = Tensor.ones(2, 2, device="FERRITE"); t[0] = 7.0; t[1:6:2, 0] = Tensor([1.0, 2.0, 3.0], device="FERRITE")
  want[2:4, 3:5], want[0], want[1:6:2, 0] = 1, 7, [1, 2, 3]
  assert t.numpy().tolist() == want.tolist()
  def train(device):
    rng = np.random.default_rng(15)
    l1, l2 = nn.Linear(16, 32), nn.Linear(32, 10)
    for l in (l1, l2):
      l.weight = Tensor(rng.standard_normal(l.weight.shape).astype(np.float32) * 0.2, device=device)
      l.bias = Tensor.zeros(*l.bias.shape, device=device)
    opt, losses = nn.optim.Adam([l1.weight, l1.bias, l2.weight, l2.bias], lr=1e-2), []
    x, y = (Tensor(v, device=device) for v in (rng.standard_normal((8, 16)).astype(np.float32), rng.integers(0, 10, 8).astype(np.int32)))
    with Context(TRAINING=1):
      for _ in range(3):
        opt.zero_grad()
        losses.append(l2(l1(x).relu()).sparse_categorical_crossentropy(y).backward().item())
        opt.step()
    return losses
  want, got = train("CPU"), train("FERRITE")
  assert np.allclose(got, want, atol=1e-3) and got[-1] < got[0] - 0.3, (got, want)
  def jitted(device):  # the same under TinyJit: on FERRITE its third call on replays the second's launches, the function not run
    rng, calls = np.random.default_rng(15), []
    l1 = nn.Linear(16, 10)
    l1.weight, l1.bias = Tensor(rng.standard_normal((10, 16)).astype(np.float32) * 0.2, device=device), Tensor.zeros(10, device=device).contiguous()
    opt = nn.optim.Adam([l1.weight, l1.bias], lr=1e-2)
    @TinyJit
    def step(x, y):
      calls.append(1)
      with Context(TRAINING=1):
        opt.zero_grad()
        loss = l1(x).sparse_categorical_crossentropy(y).backward()
        opt.step()
        return loss.realize()
    data = [(rng.standard_normal((8, 16)).astype(np.float32), rng.integers(0, 10, 8).astype(np.int32)) for _ in range(5)]
    return [step(Tensor(x, device=device).realize(), Tensor(y, device=device).realize()).item() for x, y in data], len(calls)
  (want, _), (got, calls) = jitted("CPU"), jitted("FERRITE")
  assert np.allclose(got, want, atol=1e-3) and calls == 2, (got, want, calls)

def test_torch_frontend(chip):
  # PyTorch through tinygrad's own frontend (extra/torch_backend, PyTorch's "tiny" device) with FERRITE as tinygrad's
  # device: ferrite has no PyTorch adapter of its own. Elementwise ops, a sum, a matmul, softmax, layer norm and a conv2d
  # against PyTorch on the CPU (the matmul and the conv to the FPU's 2^-11), each on the chip (launches), then an MLP's
  # three Adam steps against the CPU's losses
  if "TINYGRAD" in os.environ: sys.path.insert(0, os.environ["TINYGRAD"])
  pytest.importorskip("tinygrad")
  pytest.importorskip("extra.torch_backend", reason="wants a tinygrad checkout (TINYGRAD=): an installed tinygrad has no extra/")
  torch = pytest.importorskip("torch")
  from adapters import tinygrad_ferrite
  from tinygrad.helpers import Context
  rt = tinygrad_ferrite.RT = tinygrad_ferrite.RT or Runtime(chip, chip.tensix[0], chip.tensix[-1])
  with Context(DEV="FERRITE"):  # tinygrad's device for the "tiny" one, read at each op
    import extra.torch_backend.backend  # registers "tiny" with PyTorch (its C++ part built once, then cached)
    g = torch.Generator().manual_seed(0)
    a, b, w = torch.randn(4, 40, generator=g), torch.randn(4, 40, generator=g), torch.randn(40, 24, generator=g)
    image, kernel, norm = torch.randn(1, 2, 10, 10, generator=g), torch.randn(3, 2, 3, 3, generator=g), torch.nn.LayerNorm(40)
    cases = [lambda x, y, w: (x * 2 + y).relu() - y.exp() / 4, lambda x, y, w: x.sum(1), lambda x, y, w: x @ w,
             lambda x, y, w: x.softmax(-1), lambda x, y, w: norm.to(x.device)(x),
             lambda x, y, w: torch.nn.functional.conv2d(image.to(x.device), kernel.to(x.device))]
    for n, f in enumerate(cases):
      want = f(a, b, w).detach()
      launches, got = rt.launches, f(*(t.to("tiny") for t in (a, b, w))).detach().cpu()
      assert rt.launches > launches and got.shape == want.shape, n
      assert (got - want).abs().max() <= 1e-3 * (want.abs().max() + 1), (n, (got - want).abs().max())
    def train(device):
      torch.manual_seed(1)
      model = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.ReLU(), torch.nn.Linear(32, 10)).to(device)
      opt, g = torch.optim.Adam(model.parameters(), lr=1e-2), torch.Generator().manual_seed(2)
      x, y = torch.randn(8, 16, generator=g).to(device), torch.randint(0, 10, (8,), generator=g).to(device)
      losses = []
      for _ in range(3):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
      return losses
    want, got = train("cpu"), train("tiny")
    assert np.allclose(got, want, atol=1e-4) and got[-1] < got[0] - 0.2, (got, want)

def test_tinygrad_llm(chip):
  # tinygrad's own LLM code (tinygrad/llm's Transformer, which runs Llama 3 and Qwen 3) on the FERRITE device: two blocks
  # in SmolLM's shapes from the same bf16 weights as ferrite's numpy Llama (test/llama.py), against it, a prompt's logits
  # and then a decode step's from tinygrad's own KV cache (fp16, so on the host: the adapter carries no fp16). Its blocks
  # are tinygrad functions, precompiled: the adapter inlines them, their own buffers made fresh each call
  if "TINYGRAD" in os.environ: sys.path.insert(0, os.environ["TINYGRAD"])
  pytest.importorskip("tinygrad.llm.model")
  from adapters import tinygrad_ferrite
  from tinygrad import Tensor as TT, dtypes, nn
  from tinygrad.helpers import Context
  from tinygrad.llm.model import Transformer, TransformerConfig
  rt = tinygrad_ferrite.RT = tinygrad_ferrite.RT or Runtime(chip, chip.tensix[0], chip.tensix[-1])
  c, w = smol(np.random.default_rng(37), 2)
  sd = {"token_embd.weight": w["embed"], "output.weight": w["embed"], "output_norm.weight": w["norm"][0]}
  for l in range(2):  # tinygrad's Linear has (out, in), as Hugging Face's; q, k and v, and gate and up, apart
    qkv, gate_up, p = w[f"layers.{l}.qkv_proj"].T, w[f"layers.{l}.gate_up_proj"].T, f"layers.{l}."
    sd |= {f"blk.{l}.{k}": v for k, v in (("attn_q.weight", qkv[:576]), ("attn_k.weight", qkv[576:768]), ("attn_v.weight", qkv[768:]),
           ("attn_output.weight", w[p + "o_proj"].T), ("ffn_gate.weight", gate_up[:1536]), ("ffn_up.weight", gate_up[1536:]),
           ("ffn_down.weight", w[p + "down_proj"].T), ("attn_norm.weight", w[p + "input_layernorm"][0]), ("ffn_norm.weight", w[p + "post_attention_layernorm"][0]))}
  with Context(DEV="FERRITE"):
    model = Transformer(TransformerConfig(num_blocks=2, dim=576, hidden_dim=1536, n_heads=9, n_kv_heads=3, norm_eps=c.eps, vocab_size=256,
                                          head_dim=64, rope_theta=c.theta, rope_dim=64, v_head_dim=64, max_context=64))
    nn.state.load_state_dict(model, {k: TT(np.ascontiguousarray(v, np.float32)).cast(dtypes.bfloat16) for k, v in sd.items()}, verbose=False)
    def logits(ids, start):  # Transformer.forward without its sampling
      x = model.token_embd(TT([ids], dtype=dtypes.int32)).float()
      for b in model.blk: x = b(x, start)
      return model.output(model.output_norm(x[:, -1:]))[0, -1].numpy()
    np_kv = llama.kv_cache(c, 64, np.zeros)
    for ids, places in (([7, 200, 31, 64, 5], range(5)), ([42], [5])):
      launches, got, want = rt.launches, logits(ids, places[0]), llama.forward(ids, places, w, c, np_kv)[0]
      assert rt.launches - launches > 50 and np.abs(got - want).max() < 0.02 * np.ptp(want), (np.abs(got - want).max(), np.ptp(want))

@pytest.mark.parametrize("n, dtype, size", [(16, "bf16", 0), (40, "bf16", 0), (40, "bfp8", 0), (16, "bf16", 64)])
def test_gpt2_layer(chip, n, dtype, size):
  # a GPT-2 small block (768 wide, 12 heads of 64, MLP 3072) on n tokens (40: two tile rows, so attention spans tiles),
  # against the same code (test/gpt2.py) in numpy, float64, with the same weights: bf16, or the matrices cast to bfp8 on
  # the chip (numpy then has them as the chip does); with size, its keys and values put into a KV cache of that many
  # places, attention reading all of them, the empty ones masked. Matrices as GPT-2 initialises them (normal, 0.02 wide),
  # biases and norms about 0 and 1 but 0.1 wide, so that they count; the attention is causal and not uniform (scores of
  # a few units). Every intermediate is bf16 here, so the block's change to x (up to ~2) must match to within 3% of its
  # size
  rt, rng, d = Runtime(chip, chip.tensix[0], chip.tensix[-1]), np.random.default_rng(3), 768
  shapes = {"ln_1.weight": (1, d, 1), "ln_1.bias": (1, d, 0), "attn.c_attn.weight": (d, 3 * d, 0), "attn.c_attn.bias": (1, 3 * d, 0),
            "attn.c_proj.weight": (d, d, 0), "attn.c_proj.bias": (1, d, 0), "ln_2.weight": (1, d, 1), "ln_2.bias": (1, d, 0),
            "mlp.c_fc.weight": (d, 4 * d, 0), "mlp.c_fc.bias": (1, 4 * d, 0), "mlp.c_proj.weight": (4 * d, d, 0), "mlp.c_proj.bias": (1, d, 0)}
  p = {k: bf16(rng.normal(mean, 0.1 if r == 1 else 0.02, (r, c))) for k, (r, c, mean) in shapes.items()}
  x = bf16(rng.normal(0, 1, (n, d)))
  t = {k: Tensor.of(rt, v.tolist()) for k, v in p.items()}
  t |= {k: v.cast(dtype).realize() for k, v in t.items() if v.rows > 1}
  p |= {k: gpt2.values(v) for k, v in t.items() if v.dtype != "bf16"}
  cache = (lambda zeros: zeros((24, size, 64)) if size else None)  # a block's keys and values, or none
  places = Tensor.of(rt, [list(range(n))], "int32") if size else None
  got = gpt2.layer(Tensor.of(rt, x.tolist()), t, Tensor.of(rt, gpt2.mask(range(n), size or n).tolist()), 12,
                   cache(lambda shape: Tensor.of(rt, np.zeros(shape).tolist())), places)
  got = np.array(got.tolist())
  want = gpt2.layer(x, p, gpt2.mask(range(n), size or n), 12, cache(np.zeros), range(n))
  change = np.abs(want - x).max()
  assert change > 1 and np.abs(got - want).max() < 0.03 * change, (np.abs(got - want).max(), change)

@pytest.mark.skipif("GPT2" not in os.environ, reason="GPT2=/path/to/model.safetensors (Hugging Face openai-community/gpt2) runs it")
def test_gpt2_model(chip):
  # the whole model: GPT-2 small's real weights, all 12 blocks, the embeddings (gathered on the chip) and the output layer
  # (the token embedding transposed on the chip). After "The meaning of life is", then after each of three more tokens
  # (a token at a time from the KV cache, the last a replay), the next token's 50257 logits against the same model in
  # numpy with the same bf16 weights (in fp32 it matches Hugging Face's to 1e-4), and a pick numpy rates as good: first
  # " not" (407) by numpy, and " to" just 0.36 behind, closer than bf16's rounding through 12 blocks can tell apart (the
  # residual reaches 3000, where bf16 steps by 16). About 90 s on ttsim; test/gpt2.py generates text
  rt = Runtime(chip, chip.tensix[0], chip.tensix[-1])
  weights = gpt2.load(os.environ["GPT2"])
  t, w, ids = gpt2.upload(rt, weights), gpt2.arrays(weights), [464, 3616, 286, 1204, 318]  # (tiktoken's gpt2 encoding)
  kv = gpt2.kv_cache(12, 32, lambda shape: Tensor.of(rt, np.zeros(shape).tolist()))
  run, on = lambda ids, places: gpt2.forward(ids, t, gpt2.mask(places, 32), places, kv), lambda row: Tensor.of(rt, [row], "int32")
  step = tensor.jit(run)
  for i in range(4):
    new = ids if i == 0 else ids[-1:]
    got = gpt2.values((run if i == 0 else step)(on(new), on(list(range(len(ids) - len(new), len(ids))))))[0]
    want = gpt2.logits(ids, w)
    err = np.abs(got - want).max()
    assert (i or want.argmax() == 407) and err < 2 and want[got.argmax()] >= want.max() - 2 * err, (i, err, got.argmax())
    ids.append(int(want.argmax()))

def smol(rng, layers, vocab=256, places=64):  # random weights in SmolLM-135M's shapes (as test/llama.py names them), its config
  c = SimpleNamespace(dim=576, heads=9, kv=3, layers=layers, eps=1e-5, theta=10000.0)
  shapes = {"input_layernorm": (1, 576, 1), "post_attention_layernorm": (1, 576, 1), "q_proj": (576, 576, 0), "k_proj": (576, 192, 0),
            "v_proj": (576, 192, 0), "o_proj": (576, 576, 0), "gate_proj": (576, 1536, 0), "up_proj": (576, 1536, 0), "down_proj": (1536, 576, 0)}
  w = {f"layers.{l}.{k}": bf16(rng.normal(mean, 0.1 if r == 1 else 0.02, (r, n))) for l in range(layers) for k, (r, n, mean) in shapes.items()}
  w |= {"norm": bf16(rng.normal(1, 0.1, (1, 576))), "embed": bf16(rng.normal(0, 1, (vocab, 576)))}
  return c, llama.fuse(w | {"embed.T": w["embed"].T.copy()} | dict(zip(("cos", "sin", "turn"), llama.rotary(c, places))))

def test_llama_frequencies():
  # Llama 3.1's RoPE scaling (rope_type llama3: long wavelengths slowed by 8, short ones kept, those between blended):
  # test/llama.py's frequencies are Hugging Face's, in float32 to the bit
  m, transformers = pytest.importorskip("transformers.modeling_rope_utils"), pytest.importorskip("transformers")
  rope = {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0, "original_max_position_embeddings": 8192}
  hf = transformers.LlamaConfig(hidden_size=4096, num_attention_heads=32, rope_theta=500000.0, rope_scaling=rope, max_position_embeddings=131072)
  want = m.ROPE_INIT_FUNCTIONS["llama3"](hf, "cpu")[0].numpy()
  got = llama.frequencies(SimpleNamespace(dim=4096, heads=32, theta=500000.0, rope=rope))
  assert got.dtype == np.float32 and np.array_equal(got, want), np.abs(got - want).max()
  plain = llama.frequencies(SimpleNamespace(dim=4096, heads=32, theta=500000.0))
  assert plain[0] == got[0] and plain[-1] == 8 * got[-1], "short wavelengths kept, long ones slowed by the factor"

def safetensors(path, tensors):  # {name: (dtype "BF16" or "F32", array of its bits or values)} as a safetensors file
  header, data = {}, b""
  for name, (dtype, a) in tensors.items():
    header[name] = {"dtype": dtype, "shape": list(a.shape), "data_offsets": [len(data), len(data) + a.nbytes]}
    data += a.tobytes()
  head = json.dumps(header).encode()
  pathlib.Path(path).write_bytes(struct.pack("<Q", len(head)) + head + data)

def test_llama_load(tmp_path):
  # a checkpoint in two files and an index, as Llama-3.1-8B's comes, with an output layer of its own (lm_head) and a
  # norm in float32: llama.load gives each weight as a Weight, read only when asked, transposed for x @ w, q, k and v (and
  # gate and up) side by side, float32 rounded to bf16 to nearest even, and RoPE's tables for the places asked for, not
  # the model's 131072
  rng, dim, ffn, vocab = np.random.default_rng(31), 64, 96, 40
  w = lambda r, c: (rng.standard_normal((r, c)).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
  parts = {"model.embed_tokens.weight": w(vocab, dim), "lm_head.weight": w(vocab, dim), "model.norm.weight": w(1, dim)[0],
           **{f"model.layers.0.self_attn.{n}_proj.weight": w(r, dim) for n, r in (("q", dim), ("k", 32), ("v", 32), ("o", dim))},
           **{f"model.layers.0.mlp.{n}_proj.weight": w(r, cl) for n, r, cl in (("gate", ffn, dim), ("up", ffn, dim), ("down", dim, ffn))}}
  norms = {f"model.layers.0.{n}.weight": rng.standard_normal(dim).astype(np.float32) for n in ("input_layernorm", "post_attention_layernorm")}
  files = [{k: ("BF16", v) for k, v in list(parts.items())[:5]} | {k: ("F32", v) for k, v in norms.items()}, {k: ("BF16", v) for k, v in list(parts.items())[5:]}]
  for i, file in enumerate(files): safetensors(tmp_path / f"model-0000{i + 1}-of-00002.safetensors", file)
  (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: f"model-0000{i + 1}-of-00002.safetensors" for i, file in enumerate(files) for k in file}}))
  rope = {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0, "original_max_position_embeddings": 8192}
  (tmp_path / "config.json").write_text(json.dumps({"hidden_size": dim, "num_attention_heads": 2, "num_key_value_heads": 1, "num_hidden_layers": 1,
                                                    "rms_norm_eps": 1e-5, "rope_theta": 500000.0, "max_position_embeddings": 131072,
                                                    "rope_scaling": rope, "vocab_size": vocab, "intermediate_size": ffn}))
  c, weights = llama.load(str(tmp_path), 64)
  assert all(isinstance(x, llama.Weight) for x in weights.values())
  assert c.places == 64 and c.rope == rope and c.ffn == ffn and weights["cos"].bits().shape == (64, 4 * 32)
  q, k, v, gate, up = (parts[f"model.layers.0.{n}_proj.weight"] for n in ("self_attn.q", "self_attn.k", "self_attn.v", "mlp.gate", "mlp.up"))
  assert np.array_equal(weights["layers.0.qkv_proj"].bits(), np.concatenate([q.T, k.T, v.T], 1))
  assert np.array_equal(weights["layers.0.gate_up_proj"].bits(), np.concatenate([gate.T, up.T], 1))
  assert np.array_equal(weights["embed.T"].bits(), parts["lm_head.weight"].T), "its own output layer, not the embedding"
  assert np.array_equal(weights["layers.0.input_layernorm"].bits()[0], gpt2.bf16_bits(norms["model.layers.0.input_layernorm.weight"]))
  out = llama.forward([3, 7], [0, 1], llama.arrays(weights), c, llama.kv_cache(c, 32, np.zeros))
  assert out.shape == (1, vocab) and np.isfinite(out).all()

def test_tokenizer(tmp_path):
  # llama.Tokenizer, tokenizer.json's byte-level BPE in plain Python, encodes as tiktoken does over the same file: with
  # Llama 3's split (its \p{L}, \p{N} and \s spelled out for Python's re, whose \s also takes in four controls) and with
  # SmolLM's, on digits, contractions in capitals, CRLFs, odd spaces, controls (a token only Unicode's \s keeps whole),
  # accents and emoji (a whole piece that is a token is taken as it is: the lone emoji, whose bytes no merge joins); and
  # decodes it back
  tiktoken = pytest.importorskip("tiktoken")
  bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))  # GPT-2's bytes as printable characters
  char = dict(zip(bs + [b for b in range(256) if b not in bs], map(chr, bs + [256 + i for i in range(256 - len(bs))])))
  tokens = [bytes([b]) for b in range(256)] + [b"th", b"he", b"the", b" the", b"in", b"ing", b"12", b"123", b"\r\n", b"\x1cthe", "é".encode(), "😀".encode()]
  vocab = {"".join(char[b] for b in t): i for i, t in enumerate(tokens)}  # ids in merge order
  text = "The thing's THE THING'LL 1234567 x²³ ½ ٣٤٥ in\u00a0the\u2003end \t\tthing\r\n\r\nthe \x1c\x1dctrl \x1cthe café 😀😀 x😀y e\u0301 "
  llama3 = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
  for split in (llama3, None):
    pre = {"type": "Split", "pattern": {"Regex": split}} if split else {"type": "ByteLevel"}
    (tmp_path / "tokenizer.json").write_text(json.dumps({"model": {"vocab": vocab}, "added_tokens": [{"content": "<|end|>", "id": len(vocab)}], "pre_tokenizer": pre}))
    ours = llama.Tokenizer(str(tmp_path))
    theirs = tiktoken.Encoding("t", pat_str=split or llama.SPLIT, mergeable_ranks=dict(zip(tokens, range(len(tokens)))), special_tokens={"<|end|>": len(vocab)})
    assert ours.encode(text) == theirs.encode(text), split
    assert ours.decode(ours.encode(text) + [len(vocab)]) == text + "<|end|>"

@pytest.mark.parametrize("n", [16, 40])
def test_llama_layer(chip, n):
  # a Llama block (RMSNorm, RoPE, grouped-query attention, a SwiGLU MLP) in SmolLM-135M's shapes (576 wide, 9 query heads
  # sharing 3 key and value heads, MLP 1536) on n tokens, its keys and values put into a cache of n places, against the
  # same code (test/llama.py) in numpy, float64, with the same bf16 weights and rotary tables. Every intermediate is bf16
  # here, so the block's change to x must match to within 3% of its size
  rt, rng = Runtime(chip, chip.tensix[0], chip.tensix[-1]), np.random.default_rng(12)
  c, w = smol(rng, 1)
  p = {k.removeprefix("layers.0."): v for k, v in w.items() if k.startswith("layers.0.")}
  x, (cos, sin, turn) = bf16(rng.normal(0, 1, (n, 576))), llama.rotary(c, n)
  mask = gpt2.mask(range(n), n)
  on = lambda a: Tensor.of(rt, a.tolist())
  cache, np_cache = on(np.zeros((6, n, 64))), np.zeros((6, n, 64))
  places = Tensor.of(rt, [list(range(n))], "int32")
  got = llama.layer(on(x), {k: on(v) for k, v in p.items()}, c, (on(cos), on(sin), on(turn)), on(mask), cache, places)
  want = llama.layer(x, p, c, (np.float64(cos), np.float64(sin), turn), np.float64(mask), np_cache, range(n))
  change = np.abs(want - x).max()
  assert change > 0.5 and np.abs(np.array(got.tolist()) - want).max() < 0.03 * change, (np.abs(np.array(got.tolist()) - want).max(), change)

def test_kv_cache(chip):
  # a two-block Llama decoding with a KV cache of 32 places: 5 prompt tokens, then 5 more one at a time, each putting its
  # keys and values at its place (data, not a shape) and seeing those before, against numpy doing the same. Every step
  # launches the same kernels, so from the third it is a replay (tensor.jit); and numpy's cached logits equal its full
  # recompute
  rt, rng = Runtime(chip, chip.tensix[0], chip.tensix[-1]), np.random.default_rng(13)
  c, w = smol(rng, 2)
  t = {k: Tensor.of(rt, v.tolist()) for k, v in w.items()}
  kv, np_kv = llama.kv_cache(c, 32, lambda shape: Tensor.of(rt, np.zeros(shape).tolist())), llama.kv_cache(c, 32, np.zeros)
  on = lambda row: Tensor.of(rt, [row], "int32")
  step, ids = tensor.jit(lambda ids, places: llama.forward(ids, places, t, c, kv)), [7, 200, 31, 64, 5]
  for i in range(6):
    new = ids if i == 0 else ids[-1:]
    places = list(range(len(ids) - len(new), len(ids)))
    got = llama.logits(llama.forward(on(new), on(places), t, c, kv) if i == 0 else step(on(new), on(places)))
    want = llama.forward(new, places, w, c, np_kv)[0]
    assert np.abs(got - want).max() < 0.03 * np.abs(want).max(), (i, np.abs(got - want).max(), np.abs(want).max())
    ids.append(int(want.argmax()))
  assert np.allclose(want, llama.forward(ids[:-1], range(len(ids) - 1), w, c, llama.kv_cache(c, 32, np.zeros))[0], atol=1e-3)

def test_moe(chip):
  # a mixture of experts in the MLP's place (llama.moe): Tensor.topk picks each row's largest the way numpy's stable
  # argsort does (first of ties); a token's router logits choose its k experts on the chip, matmuls that pick each
  # one's weights there (index=) run their SwiGLUs, weighted by the softmax of the chosen logits and summed, as numpy
  # does for one token and for three; and a two-block model of them decodes from a replay (tensor.jit) bit for bit as
  # it does op by op, so a replayed step reads whatever experts its token chose, and near numpy
  rt, rng = Runtime(chip, chip.tensix[0], chip.tensix[-1]), np.random.default_rng(31)
  x = bf16(rng.normal(0, 2, (3, 16)))
  x[1, 9] = x[1, 3] = x[1].max()  # a tie: the first column first
  values, indices = Tensor.of(rt, x.tolist()).topk(4)
  assert np.array_equal(value(indices)[0], np.argsort(-x, -1, kind="stable")[:, :4]) and np.array_equal(value(values)[0], -np.sort(-x, -1)[:, :4])
  experts = lambda dim, n, m: {"router": bf16(rng.normal(0, 0.5, (dim, n))), "experts.gate_up": bf16(rng.normal(0, 0.1, (n, dim, 2 * m))),
                               "experts.down": bf16(rng.normal(0, 0.1, (n, m, dim)))}
  c, w = SimpleNamespace(dim=64, k=4), experts(64, 16, 64)
  t = {k: Tensor.of(rt, v.tolist()) for k, v in w.items()}
  for n in (1, 3):
    h = bf16(rng.normal(0, 1, (n, 64)))
    got, want = value(llama.moe(Tensor.of(rt, h.tolist()), t, c))[0], llama.moe(h, w, c)
    assert np.abs(got - want).max() < 0.02 * np.abs(want).max(), (n, np.abs(got - want).max())
  c, w = smol(rng, 2)
  c.k = 2
  for l in range(2): del w[f"layers.{l}.gate_up_proj"], w[f"layers.{l}.down_proj"]; w |= {f"layers.{l}.{k}": v for k, v in experts(576, 8, 64).items()}
  t, on = {k: Tensor.of(rt, v.tolist()) for k, v in w.items()}, lambda row: Tensor.of(rt, [row], "int32")
  def decode(replay):  # each step's logits, for a prompt and four tokens after it (numpy's picks), and numpy's
    kv, np_kv = llama.kv_cache(c, 32, lambda shape: Tensor.of(rt, np.zeros(shape).tolist())), llama.kv_cache(c, 32, np.zeros)
    step, ids, out = tensor.jit(lambda ids, places: llama.forward(ids, places, t, c, kv)), [1, 2, 3, 4, 5], []
    for i in range(5):
      new = ids if i == 0 else ids[-1:]
      places = list(range(len(ids) - len(new), len(ids)))
      out.append((llama.logits(llama.forward(on(new), on(places), t, c, kv) if i == 0 or not replay else step(on(new), on(places))),
                  want := llama.forward(new, places, w, c, np_kv)[0]))
      ids.append(int(want.argmax()))
    return out
  # (a prompt whose tokens' router logits keep their k-th and next apart by 6% of their span, so bf16's rounding leaves
  # the chip's choices numpy's; its experts' outputs, large and of either sign, cost the sums more of their bits than a
  # plain MLP's, hence 6%)
  replayed, alone = decode(True), decode(False)
  assert all(np.array_equal(a[0], b[0]) for a, b in zip(replayed, alone)), "a replayed step is the step"
  assert all(np.abs(got - want).max() < 0.06 * np.abs(want).max() for got, want in alone), [np.abs(g - w).max() / np.abs(w).max() for g, w in alone]

@pytest.mark.skipif("LLAMA8B" not in os.environ, reason="LLAMA8B=1 runs it: about four minutes on ttsim")
def test_llama_8b_shapes(chip):
  # one block in Llama-3.1-8B's shapes (4096 wide, 32 query heads sharing 8 KV heads, an MLP of 14336, Llama 3's RoPE
  # scaling), random weights, at TT's performance precision: B2's matmuls, casts and attention at their real sizes (the
  # logits over a vocabulary of 2048), a 64-token prompt and two decode steps against numpy. On ttsim the upload's casts
  # take minutes (the MLP's 4096 x 28672 to bfp4 is 114,688 tiles), which llama.upload waits for
  rt, rng = Runtime(chip, chip.tensix[0], chip.tensix[-1]), np.random.default_rng(8)
  rope = {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0, "original_max_position_embeddings": 8192}
  c = SimpleNamespace(dim=4096, heads=32, kv=8, layers=1, eps=1e-5, theta=500000.0, rope=rope, vocab=2048, ffn=14336, places=96)
  bits = lambda shape, scale, mean=0.0: gpt2.bf16_bits(rng.normal(mean, scale, shape))
  w = {"layers.0.input_layernorm": bits((1, 4096), 0.1, 1), "layers.0.post_attention_layernorm": bits((1, 4096), 0.1, 1),
       "layers.0.qkv_proj": bits((4096, 6144), 0.02), "layers.0.o_proj": bits((4096, 4096), 0.02), "layers.0.gate_up_proj": bits((4096, 28672), 0.02),
       "layers.0.down_proj": bits((14336, 4096), 0.02), "norm": bits((1, 4096), 0.1, 1), "embed": bits((2048, 4096), 1.0)}
  w |= {"embed.T": np.ascontiguousarray(w["embed"].T)} | {k: gpt2.bf16_bits(a) for k, a in zip(("cos", "sin", "turn"), llama.rotary(c, c.places))}
  weights, c.precision = {k: llama.Weight(lambda v=v: v) for k, v in w.items()}, llama.precision(c, "tt")
  t = llama.upload(rt, weights, c.precision)
  arrays, on = llama.arrays(weights, c.precision), lambda row: Tensor.of(rt, [row], "int32")
  kv, np_kv = llama.kv_cache(c, 96, llama.zeros(rt, "bfp8")), llama.kv_cache(c, 96, np.zeros)
  prompt = [int(i) for i in rng.integers(0, 2048, 64)]
  for new, places in ((prompt, list(range(64))), ([5], [64]), ([9], [65])):
    out = llama.forward(on(new), on(places), t, c, kv).realize()
    rt.wait(600)
    got, want = llama.logits(out), llama.forward(new, places, arrays, c, np_kv)[0]
    assert got.argmax() == want.argmax() and np.abs(got - want).max() < 0.02 * np.ptp(want), (np.abs(got - want).max(), np.ptp(want))

def test_benchmark_table():
  # the two stacks' results side by side (benchmark.table), with B1's and B2's gate: TT-NN's speed over ferrite's, at most 1.5
  ours = {"versions": {"lines of code": {"ferrite": 1}}, "clock": "the host's", "B1 matmul 4096^3 bf16, 2 phases": {"ms": 3.0},
          "B2 decode, batch 1": {"tokens/s": 40.0}, "B3 prefill, 32768 tokens": {"does not run yet": "MemoryError"}}
  theirs = {"versions": {"lines of code": {"tt-metal": 2}}, "clock": "the host's", "B1 matmul 4096^3 bf16, 2 phases": {"ms": 1.5},
            "B2 decode, batch 1": {"tokens/s": 50.0}, "B3 prefill, 32768 tokens": {"seconds to first token": 9.0}}
  rows = benchmark.table(ours, theirs).splitlines()
  assert rows[2].endswith("| 2.00 (past 1.5) |") and rows[3].endswith("| 1.25 (within 1.5) |") and rows[4].endswith("|  |"), rows
  assert all(r.count("|") == 5 for r in rows) and "MemoryError" in rows[4], rows

def test_chunked_prefill(chip):
  # a long prompt a chunk at a time (llama.prefill, as B3's 32,768 tokens must go: attention's scores a chunk by the cache,
  # not the prompt squared): 70 tokens in chunks of 32, 32 and 6, each chunk's keys and values cached before the next
  # reads them. In numpy, the logits of the prompt at once; on the chip, numpy's, and then a token decoded from the
  # cache the chunks filled
  rt, rng = Runtime(chip, chip.tensix[0], chip.tensix[-1]), np.random.default_rng(29)
  c, w = smol(rng, 2, places=96)
  t, prompt, on = {k: Tensor.of(rt, v.tolist()) for k, v in w.items()}, [int(i) for i in rng.integers(0, 256, 70)], lambda row: Tensor.of(rt, [row], "int32")
  kv, np_kv = llama.kv_cache(c, 96, lambda shape: Tensor.of(rt, np.zeros(shape).tolist())), llama.kv_cache(c, 96, np.zeros)
  want = llama.prefill(prompt, w, c, np_kv, 32)[0]
  assert np.allclose(want, llama.forward(prompt, range(70), w, c, llama.kv_cache(c, 96, np.zeros))[0], atol=1e-3)
  got = llama.logits(llama.prefill(prompt, t, c, kv, 32, on))
  assert np.abs(got - want).max() < 0.03 * np.abs(want).max(), np.abs(got - want).max()
  nxt = int(want.argmax())
  got, want = llama.logits(llama.forward(on([nxt]), on([70]), t, c, kv)), llama.forward([nxt], [70], w, c, np_kv)[0]
  assert np.abs(got - want).max() < 0.03 * np.abs(want).max(), np.abs(got - want).max()

def test_llama_precision(chip):
  # TT's performance precision (llama.precision "tt"): each matmul's weights in its format (the MLP's gate and up in bfp4,
  # the rest and the output layer in bfp8) and at its fidelity phases (one for gate and up, two for the rest, attention
  # two in decode and four in prefill), Llama-3.1-8B's last block's gate and up in bfp8 at two, the KV cache in bfp8;
  # numpy, given the same block-float values (llama.arrays rounds the weights, and layer the cache, as the chip does:
  # gpt2.bfp) but exact products, picks the same tokens, its logits within 1% of their span. upload leaves no more DRAM
  # behind than one bf16 matrix: each cast is waited for, so the next weight's bf16 copy goes where this one's was
  big = SimpleNamespace(layers=32, dim=4096)
  assert llama.precision(big, "tt")["gate_up", 31] == ("bfp8", 2) and llama.precision(big, "tt")["gate_up", 30] == ("bfp4", 1)
  assert llama.precision(big, "tt")["head", None] == ("bfp8", 2) and llama.precision(big, "bf16")["qkv", 0] == ("bf16", 4)
  assert [llama.precision(big, "tt")[k] for k in (("attention", "decode"), ("attention", "prefill"), ("kv", None))] == [(None, 2), (None, 4), ("bfp8", None)]
  rt, rng = Runtime(chip, chip.tensix[0], chip.tensix[-1]), np.random.default_rng(5)
  c, w = smol(rng, 2)
  c.precision = llama.precision(c, "tt")
  weights = {k: llama.Weight(lambda v=v: (np.float32(v).view(np.uint32) >> 16).astype(np.uint16)) for k, v in w.items()}
  t = llama.upload(rt, weights, c.precision)
  rt.wait()  # what upload freed goes back
  reached, held = max(a + n for a, n in rt.used.items()), sum(rt.used.values())  # DRAM a bank
  assert reached - held <= -(-max(x.tiles for x in t.values()) // 8) * 2048, "a cast weight's bf16 DRAM is used again"
  assert {k: x.dtype for k, x in t.items() if "proj" in k or k == "embed.T"} == {k: "bfp4" if "gate_up" in k else "bfp8" for k in t if "proj" in k or k == "embed.T"}
  sizes = SimpleNamespace(**vars(c), vocab=256, places=64, ffn=1536)  # benchmark.weights, from the sizes alone, has them as upload makes them
  assert {k: (x.rows, x.cols, x.dtype) for k, x in t.items()} == {k: (x.rows, x.cols, x.dtype) for k, x in benchmark.weights(sizes).items()}
  arrays = llama.arrays(weights, c.precision)  # numpy's rounding to the formats, which is the chip's casts' (test_bfp)
  kv, np_kv = llama.kv_cache(c, 32, llama.zeros(rt, "bfp8")), llama.kv_cache(c, 32, np.zeros)  # numpy's rounded as a bfp8 cache's
  assert kv[0].dtype == "bfp8"
  on = lambda row: Tensor.of(rt, [row], "int32")
  for ids, places in (([7, 200, 31], [0, 1, 2]), ([5], [3])):  # the prompt, then a token decoded
    got, want = llama.logits(llama.forward(on(ids), on(places), t, c, kv)), llama.forward(ids, places, arrays, c, np_kv)[0]
    assert got.argmax() == want.argmax() and np.abs(got - want).max() < 0.01 * np.ptp(want), (np.abs(got - want).max(), np.ptp(want))

def test_token_accuracy(chip):
  # llama.accuracy, Tenstorrent's token accuracy (COMPARE, Measurement): a prompt prefilled, then each step fed the
  # reference's token whatever the model picked. Against a reference made the same way from numpy (its five most likely
  # at each place) the chip's picks are numpy's top-1 throughout, and a reference that is wrong at one place costs top-1
  # exactly that step; in numpy (rt None, the weights as numpy arrays) the picks and scores are the same
  rt, rng = Runtime(chip, chip.tensix[0], chip.tensix[-1]), np.random.default_rng(6)
  c, w = smol(rng, 2)
  t = {k: Tensor.of(rt, v.tolist()) for k, v in w.items()}
  tokens = [int(i) for i in rng.integers(0, 256, 44)]
  prompt, after, top5, kv = tokens[:36], tokens[36:], [], llama.kv_cache(c, 64, np.zeros)
  for i in range(8):
    ids, places = (prompt, range(36)) if i == 0 else ([after[i - 1]], [35 + i])
    top5.append([int(k) for k in np.argsort(-llama.forward(ids, places, w, c, kv)[0])[:5]])
  assert llama.accuracy(rt, t, c, prompt, after, top5, 8)[:2] == (1, 1)
  top5[3] = top5[3][1:] + top5[3][:1]  # its most likely last: still among the five
  top1, five, picks = llama.accuracy(rt, t, c, prompt, after, top5, 8)
  assert (top1, five) == (7 / 8, 1) and picks[3] == top5[3][-1], (top1, five, picks)
  assert llama.accuracy(None, w, c, prompt, after, top5, 8) == (top1, five, picks), "in numpy (rt None), the same"

def test_benchmark_bytes():
  # what B2's effective DRAM bandwidth is computed from (benchmark.bytes_per_token; COMPARE, Measurement): the bytes a
  # decode token reads, each weight in the format TT's performance precision gives it (llama.weight_format, as upload
  # casts it; benchmark.weights has them from the model's sizes, as test_llama_precision checks). Llama-3.1-8B's sizes
  # give COMPARE's 6.20 GB a token, so 41.3 tokens/s is half the card's 512 GB/s
  c = SimpleNamespace(dim=4096, heads=32, kv=8, layers=32, vocab=128256, ffn=14336, places=736)
  c.precision = llama.precision(c, "tt")
  t = benchmark.weights(c)
  assert [t[k].dtype for k in ("layers.0.gate_up_proj", "layers.31.gate_up_proj", "layers.0.down_proj", "embed.T", "embed")] == ["bfp4", "bfp8", "bfp8", "bfp8", "bf16"]
  read = benchmark.bytes_per_token(t, c, 612.5)
  assert round(read / 1e9, 2) == 6.20 and round(41.3 * read / 512e9, 3) == 0.5, read

def test_put(chip):
  # rows into a tensor at places read on the chip (gather's inverse), against numpy: 70 rows (three tiles of places) of a
  # batch of 3, into 200 places (seven tile rows), scattered and out of order, one place twice (the later row stays),
  # bf16 and fp32; and a prompt's rows into their places in order, as a KV cache takes them
  rt, rng = Runtime(chip, chip.tensix[0], chip.tensix[-1]), np.random.default_rng(23)
  for dtype in ("bf16", "fp32"):
    cache, x = rng.normal(0, 1, (3, 200, 96)).astype(np.float32), rng.normal(0, 1, (3, 70, 96)).astype(np.float32)
    if dtype == "bf16": cache, x = bf16(cache), bf16(x)
    places = [int(p) for p in rng.permutation(200)[:70]]
    places[40] = places[7]
    t = Tensor.of(rt, cache.tolist(), dtype)
    t.put(Tensor.of(rt, [places], "int32"), Tensor.of(rt, x.tolist(), dtype))
    for i, p in enumerate(places): cache[:, p] = x[:, i]
    assert np.array_equal(value(t), cache), dtype
  kv, rows = Tensor.of(rt, np.zeros((3, 64, 64)).tolist()), bf16(rng.normal(0, 1, (3, 40, 64)))
  kv.put(Tensor.of(rt, [list(range(5, 45))], "int32"), Tensor.of(rt, rows.tolist()))
  assert np.array_equal(value(kv)[:, 5:45], rows) and not value(kv)[:, :5].any() and not value(kv)[:, 45:].any()

def test_bfp8_cache(chip):
  # a KV cache in bfp8, as Tenstorrent's models keep theirs: put writes bf16 rows into a bfp8 tensor, each run of 16
  # rounded as cast("bfp8") rounds (numpy's bfp, bit for bit: one row, rows in two tiles, a prompt's whole tile, the
  # edges), and attention reads it in place, its keys transposed (unpacker 0 transposing faces of block floats) and its
  # values as they are, each key and value head serving two query heads: both matmuls bit for bit the bf16 ones by the
  # same values
  rt, rng = Runtime(chip, chip.tensix[0], chip.tensix[-1]), np.random.default_rng(16)
  cache = Tensor.of(rt, np.zeros((4, 96, 64)).tolist()).cast("bfp8").realize()  # 2 key heads, then 2 value heads
  want = np.zeros((4, 96, 64))
  for places in ([5], [33, 34, 70], list(range(32, 64)), [0, 95]):
    x = bf16(rng.normal(0, 2, (4, len(places), 64)) * 10.0 ** rng.integers(-3, 3, (4, len(places), 1)))
    cache.put(Tensor.of(rt, [places], "int32"), Tensor.of(rt, x.tolist()))
    want[:, places] = gpt2.bfp(x)
    assert np.array_equal(value(cache), want), places
  same = Tensor.of(rt, want.tolist())  # the same values in bf16
  q, s = (Tensor.of(rt, bf16(rng.normal(0, 1, shape)).tolist()) for shape in ((4, 5, 64), (4, 5, 96)))
  for c in (cache, same): c.keys, c.values = c[0:2, :, :].repeat_interleave(2), c[2:4, :, :].repeat_interleave(2)
  assert value(q @ cache.keys.T).tolist() == value(q @ same.keys.T).tolist()
  assert value(s @ cache.values).tolist() == value(s @ same.values).tolist()

def test_batched_decode(chip):
  # four sequences decoding together, a token each a step, from prompts of different lengths: their KV caches side by
  # side in one (sequence u's places from row 32 u), each token seeing only its own sequence's places up to its own.
  # Each prompt is prefilled into its rows, then every step is one call for all four, replayed from the third
  # (tensor.jit), against each sequence decoded alone in numpy
  rt, rng = Runtime(chip, chip.tensix[0], chip.tensix[-1]), np.random.default_rng(19)
  c, w = smol(rng, 2)
  t, size, prompts = {k: Tensor.of(rt, v.tolist()) for k, v in w.items()}, 32, [[7, 200, 31], [5, 64], [9, 9, 9, 9, 9], [100]]
  kv = llama.kv_cache(c, size * len(prompts), lambda shape: Tensor.of(rt, np.zeros(shape).tolist()))
  np_kv = [llama.kv_cache(c, size, np.zeros) for _ in prompts]
  on = lambda row: Tensor.of(rt, [row], "int32")
  step, ids = tensor.jit(lambda ids, places, starts: llama.forward(ids, places, t, c, kv, starts)), [list(p) for p in prompts]
  for u, p in enumerate(prompts):  # each prompt into its own rows; its last token's logits give its next token
    got = gpt2.values(llama.forward(on(p), on(list(range(len(p)))), t, c, kv, on([size * u] * len(p))))[len(p) - 1]
    want = llama.forward(p, range(len(p)), w, c, np_kv[u])[0]
    assert np.abs(got - want).max() < 0.03 * np.abs(want).max(), (u, np.abs(got - want).max())
    ids[u].append(int(want.argmax()))
  for i in range(4):
    places = [len(s) - 1 for s in ids]
    got = gpt2.values(step(on([s[-1] for s in ids]), on(places), on([size * u for u in range(len(ids))])))
    for u, s in enumerate(ids):
      want = llama.forward(s[-1:], [places[u]], w, c, np_kv[u])[0]
      assert np.abs(got[u] - want).max() < 0.03 * np.abs(want).max(), (i, u, np.abs(got[u] - want).max())
      s.append(int(want.argmax()))

def test_serve(chip, tmp_path):
  # test/serve.py's engine: three greedy requests on two slots (the third waits for one, then takes the slot used
  # longest ago) decode together, then the last two again alone, the last first (the slot used longer ago holds the
  # other), each in the slot that holds it already, so a token of each is prefilled and the rest come from the cache:
  # the same tokens as together, every step one replayed call for both slots, a free one idling on a row no sequence
  # reaches (on any other it would spoil what its slot holds); a long prompt is prefilled a chunk between steps, so
  # the others' tokens keep coming. A request ends at a stop string (without its text), at an end-of-sequence token,
  # or when its client has gone, and says no text in which a stop string might yet begin. Over HTTP, /v1/completions
  # whole and streamed give the engine's text, and /v1/models the model; chat messages go into ChatML's or Llama 3's
  # template
  import threading, urllib.request
  rt, rng = Runtime(chip, chip.tensix[0], chip.tensix[-1]), np.random.default_rng(23)
  c, w = smol(rng, 2)
  w = {k: bf16(v * 10) if k.endswith("_proj") else v for k, v in w.items()}  # sharp attention, so every cached row counts
  t = {k: Tensor.of(rt, v.tolist()) for k, v in w.items()}
  bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))  # a byte-level vocabulary of the 256 bytes
  chars = map(chr, bs + [256 + i for i in range(256 - len(bs))])
  (tmp_path / "tokenizer.json").write_text(json.dumps({"model": {"vocab": {ch: b for ch, b in zip(chars, bs + [b for b in range(256) if b not in bs])}}, "added_tokens": []}))
  tok = llama.Tokenizer(str(tmp_path))
  engine = serve.Engine(rt, t, c, tok, slots=2, places=64, chunk=16)
  threading.Thread(target=engine.run, daemon=True).start()
  def done(r):
    while r.out.get(timeout=600) is not None: pass
    return r.made
  prompts = [[7, 200, 31, 64, 5], [5, 64], list(range(40, 60))]  # the last longer than a chunk
  together = [serve.Request(tok, p, 6, 0) for p in prompts]
  for r in together: engine.waiting.put(r)
  together = [done(r) for r in together]
  before = engine.prefilled
  assert [done(r) for r in [serve.Request(tok, p, 6, 0) for p in prompts[:0:-1]] if not engine.waiting.put(r)] == together[:0:-1]
  assert engine.prefilled - before == 2, "the prompts' last tokens, the rest in their slots already"
  class Watched(serve.Request):  # what the engine had prefilled at each of its tokens, seen from the engine's thread
    def take(self, logits, room): self.seen = getattr(self, "seen", []) + [engine.prefilled]; return super().take(logits, room)
  r, deadline = Watched(tok, [3, 1, 4], 12, 0), time.time() + 600
  engine.waiting.put(r)
  while not getattr(r, "seen", None): assert time.time() < deadline; time.sleep(0.01)
  engine.waiting.put(long := serve.Request(tok, list(range(100, 150)), 2, 0))  # 50 tokens: four chunks
  done(r), done(long)
  steps = np.diff(r.seen)
  assert 16 in steps and steps.max() <= 16, ("a chunk of the long prompt between two of its tokens, no more", r.seen)
  r = serve.Request(tok, [], 9, 0, stop=["bc"])
  said = lambda: [r.out.get_nowait() for _ in range(r.out.qsize())]
  assert not r.take(np.eye(256)[97], True) and said() == ["a"]
  assert not r.take(np.eye(256)[98], True) and said() == [], "b might begin the stop string"
  assert r.take(np.eye(256)[99], True) and said() == [None] and (r.said, r.finish) == ("a", "stop")
  r = serve.Request(tok, [], 9, 0, eos=[0])
  assert not r.take(np.eye(256)[97], True) and r.take(np.eye(256)[0], True) and (r.made, r.said, r.finish) == ([97], "a", "stop")
  r = serve.Request(tok, [], 9, 0)
  r.gone = True  # its client went
  assert r.take(np.eye(256)[97], True) and (r.made, r.finish, r.out.get_nowait()) == ([], "gone", None)
  http = serve.server(engine, "smol", port=0)
  threading.Thread(target=http.serve_forever, daemon=True).start()
  url = f"http://127.0.0.1:{http.server_address[1]}/v1"
  post = lambda body: urllib.request.urlopen(urllib.request.Request(url + "/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600)
  r = serve.Request(tok, tok.encode("Hi there"), 6, 0)
  engine.waiting.put(r)
  text = tok.decode(done(r))
  reply = json.load(post({"prompt": "Hi there", "max_tokens": 6, "temperature": 0}))
  assert reply["choices"][0]["text"] == text and reply["choices"][0]["finish_reason"] == "length" and reply["usage"]["completion_tokens"] == 6
  events = [json.loads(line[6:]) for line in post({"prompt": "Hi there", "max_tokens": 6, "temperature": 0, "stream": True}).read().decode().split("\n\n")
            if line.startswith("data: {")]
  assert "".join(e["choices"][0]["text"] for e in events) == text and events[-1]["choices"][0]["finish_reason"] == "length"
  assert json.load(urllib.request.urlopen(url + "/models", timeout=60))["data"][0]["id"] == "smol"
  http.shutdown()
  hi = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}]
  assert serve.chat("... <|im_start|> ...", hi) == "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
  assert serve.chat("... <|start_header_id|> ...", hi[1:]) == ("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>"
                                                               "<|start_header_id|>assistant<|end_header_id|>\n\n")

def test_fetch(tmp_path, monkeypatch):
  # serve.fetch: a model id's files from Hugging Face's resolve URLs (a local server in its place, by HF_ENDPOINT), the
  # checkpoint's shards by its index, the configs it has and not those it has not; a finished download is not asked
  # for again, and a folder is taken as it is
  import functools, http.server, threading
  hub = tmp_path / "hub" / "org" / "model" / "resolve" / "main"
  hub.mkdir(parents=True)
  files = {"config.json": "{}", "tokenizer.json": "{}", "generation_config.json": "{}", "model-1.safetensors": "1", "model-2.safetensors": "2",
           "model.safetensors.index.json": json.dumps({"weight_map": {"a": "model-1.safetensors", "b": "model-2.safetensors", "c": "model-1.safetensors"}})}
  for name, text in files.items(): (hub / name).write_text(text)
  quiet = type("Quiet", (http.server.SimpleHTTPRequestHandler,), {"log_message": lambda *a: None})
  hf = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(quiet, directory=str(tmp_path / "hub")))
  threading.Thread(target=hf.serve_forever, daemon=True).start()
  monkeypatch.setenv("HF_ENDPOINT", f"http://127.0.0.1:{hf.server_address[1]}")
  folder = serve.fetch("org/model", str(tmp_path / "cache"))
  assert sorted(os.listdir(folder)) == sorted([*files, ".done"]) and all((pathlib.Path(folder) / n).read_text() == t for n, t in files.items())
  hf.shutdown()
  assert serve.fetch("org/model", str(tmp_path / "cache")) == folder and serve.fetch(folder) == folder, "once, and a folder as is"

@pytest.mark.skipif("LLAMA" not in os.environ, reason="LLAMA=/path/to/SmolLM-135M (Hugging Face HuggingFaceTB/SmolLM-135M) runs it")
def test_llama_model(chip):
  # a Llama-family model's real weights (SmolLM-135M: 30 blocks, grouped-query attention, tied embeddings): after "The
  # meaning of life is", the next token's 49152 logits against the same model in numpy (whose fp32 version matches Hugging
  # Face's to 2e-5), then a token decoded with the KV cache. About a minute on ttsim; test/llama.py generates text
  rt = Runtime(chip, chip.tensix[0], chip.tensix[-1])
  c, weights = llama.load(os.environ["LLAMA"])
  w, t = llama.arrays(weights), llama.upload(rt, weights)
  kv, np_kv = llama.kv_cache(c, 32, lambda shape: Tensor.of(rt, np.zeros(shape).tolist())), llama.kv_cache(c, 32, np.zeros)
  new, places = [504, 2455, 282, 1029, 314], list(range(5))
  for _ in range(2):
    got = llama.logits(llama.forward(Tensor.of(rt, [new], "int32"), Tensor.of(rt, [places], "int32"), t, c, kv))
    want = llama.forward(new, places, w, c, np_kv)[0]
    assert np.abs(got - want).max() < 1 and got.argmax() == want.argmax(), (np.abs(got - want).max(), got.argmax(), want.argmax())
    new, places = [int(want.argmax())], [places[-1] + 1]
  assert new == [1962], "a [question]"
