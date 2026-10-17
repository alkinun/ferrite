# Milestone 1 on ttsim, one test per step. TTSIM=/path/to/libttsim.so python3 -m pytest test
import os, random, struct, pytest
from trickle import bh, pci
from trickle.chip import Chip
from trickle.compiler import Tensor, from_bf16, to_bf16
from trickle.hello import hello
from trickle.runtime import ARGS, DATA, KERNEL, Runtime, compile_rv32

@pytest.fixture(scope="session")
def chip():
  assert "TTSIM" in os.environ, "set TTSIM=/path/to/libttsim.so (ttsim built for Blackhole)"
  return Chip(pci.Sim(os.environ["TTSIM"]))  # libttsim can only be initialized once per process

def test_config_space(chip):
  assert chip.pci.cfg_read(0x00) == bh.VENDOR | bh.DEVICE << 16
  assert chip.pci.cfg_read(0x04) & 0x2, "memory decode is off"
  assert len(set(chip.pci.bars.values())) == 3

def test_arc(chip):
  chip.arc_msg(0x90)  # TEST
  t = chip.telemetry()
  assert len(chip.tensix) == 120 and t[bh.TAG_AICLK] > 0, "ttsim models a p150: 12 of 14 Tensix columns"

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
  rt.launch(compile_rv32("void kmain(volatile unsigned *a) { a[0] = noc_me(); }", KERNEL))
  rt.wait()
  assert len(rt.cores) == len(chip.tensix) - 1 == 119  # and one dispatcher
  for x, y in rt.cores: assert chip.read32(x, y, ARGS) == x | y << 6

def test_relaunch_without_reset(chip):
  # different kernels at the same address, back to back: each must run, not a stale copy of the one before
  rt = Runtime(chip, chip.tensix[0], chip.tensix[2])
  cores = rt.cores
  for i, (op, f) in enumerate([("*", int.__mul__), ("+", int.__add__)] * 2):
    rt.launch(compile_rv32(f"void kmain(volatile unsigned *a) {{ a[1] = a[0] {op} 3; }}", KERNEL), lambda x, y: (x * 100 + y + i, 0))
    rt.wait()
    for x, y in cores: assert chip.read32(x, y, ARGS + 4) == f(x * 100 + y + i, 3), f"launch {i} on {x},{y}"

def test_kernel_moves_data(chip):
  # each core pulls its own 40 KiB slice of DRAM into L1 (3 NoC packets), adds its index to every word, writes it
  # back elsewhere in DRAM, then sends its first 64 bytes to the next core's L1
  rt, n = Runtime(chip, chip.tensix[0], chip.tensix[4]), 40 * 1024  # four workers in the first row
  cores, (bx, by), src, dst = rt.cores, bh.DRAM[3], 1 << 20, 2 << 20
  data = [random.randbytes(n) for _ in cores]
  for i, d in enumerate(data): chip.write(bx, by, src + i * n, d)
  args = {xy: (i, bx, by, src + i * n, dst + i * n, n, *cores[(i + 1) % len(cores)]) for i, xy in enumerate(cores)}
  rt.launch(compile_rv32("""
void kmain(volatile unsigned *a) {  // a: [index, bank x, bank y, src, dst, n, next x, next y]
  unsigned *buf = (unsigned *)L1_DATA;
  noc_read(a[1], a[2], a[3], L1_DATA, a[5]);
  noc_wait();
  for (unsigned i = 0; i < a[5] / 4; i++) buf[i] += a[0];
  noc_write(L1_DATA, a[1], a[2], a[4], a[5]);
  noc_write(L1_DATA, a[6], a[7], L1_DATA + a[5], 64);
}""", KERNEL), lambda x, y: args[x, y])
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
  add = compile_rv32("void kmain(volatile unsigned *a) { *(volatile unsigned *)L1_DATA += a[0]; }", KERNEL)
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
  rt.launch(compile_rv32("void kmain(volatile unsigned *a) { for (;;); }", KERNEL))
  with pytest.raises(TimeoutError, match=r"launch 1, still running on \[\(1, 2\)\]"): rt.wait(timeout=0.2)

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
