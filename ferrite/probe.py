# The numbers ttsim cannot give, measured on a card (ROADMAP, Stage 1; NOTES.md, "Bring-up"):
# what a launch costs and how it grows with the rectangle, DRAM bandwidth by reads in flight, NoC latency, the rates of
# the Tensix units a kernel feeds (MVMUL alone, a matmul step with its unpacks at 1, 2 and 4 fidelity phases, the packer,
# the SFPU), BRISC's scalar work for a mixture of experts (topk, group), and PCIe both ways. Each decides something:
# DEPTH, the fidelity a matmul can use before its unpacks bound it, how many launches a token can afford, whether a
# router's choices are cheap enough on a RISC-V core. Small kernels built from the compiler's own pieces, timed by the tile's
# wall clock. On ttsim the numbers mean nothing (Tensix instructions take no time there, the NoC and DRAM answer at once):
# it only checks that each probe runs.
#   TTSIM=/path/to/libttsim.so python3 -m ferrite.probe [MiB]   # MiB: how much to upload and read back (8)
import functools, sys, time
from ferrite import bh, pci
from ferrite.chip import Chip
from ferrite.compiler import (BANKS, DST_HALF, FRAMES, MATH_END, _math, _packer, _unpacker, allocate, kernel_config, math_begin,
                              matmul, pack, run, sfpu, tt, unpack, ins)
from ferrite.runtime import DATA, Runtime, compile_kernel
from ferrite.tensor import Tensor

CLOCK = f"#define CLOCK (*(volatile unsigned *){bh.WALL_CLOCK:#x})  // the tile's wall clock, a count a cycle\n"
EMPTY = "void kmain(volatile unsigned *a) {}"

# BRISC alone: a[3] reads of a 2 KiB tile in flight at once, a[4] rounds of them, worker a[0]'s tiles from DRAM at a[2]
DRAM_READS = CLOCK + BANKS + f"""
void kmain(volatile unsigned *a) {{
  unsigned d = a[3], t0 = CLOCK;
  for (unsigned r = 0; r < a[4]; r++) {{
    for (unsigned i = 0; i < d; i++) {{ unsigned t = a[0] * d + i; noc_get(BANK(t), AT(a[2], t, 2048), {DATA:#x} + i * 2048, 2048); }}
    noc_wait();
  }}
  a[5] = CLOCK - t0;
}}"""

# BRISC alone: workers a[2] and a[3] (at a[4] and a[5]) bounce a word a[6] times; the first times it
PING = CLOCK + f"""
#define FLAG ((volatile unsigned *){DATA:#x})
#define WORD ((volatile unsigned *){DATA + 64:#x})
void kmain(volatile unsigned *a) {{
  unsigned me = a[0], first = me == a[2], other = first ? a[5] : a[4], t0 = CLOCK;
  if (!first && me != a[3]) return;
  for (unsigned r = 0, v = a[7]; r < a[6]; r++, v += 2) {{  // a[7]: this launch's first value, so none is left from another
    if (!first) while (*FLAG != v + 1) asm volatile("fence" ::: "memory");
    *WORD = v + 1 + !first;
    noc_write({DATA + 64:#x}, other & 63, other >> 6, {DATA:#x}, 4);
    if (first) while (*FLAG != v + 2) asm volatile("fence" ::: "memory");
  }}
  if (first) a[8] = CLOCK - t0;
}}"""

def tensix(unpacks="", maths="", packs="", slots=None):
  """A kernel of the three TRISCs' programs, each a C body run a[2] times (n) over a Dst half h in turn; the last of
  them that has work times its loop, to its thread's last instruction, into a[3]."""
  timed = [i for i, body in enumerate((unpacks, maths, packs)) if body][-1]
  loop = lambda i, body: (f"unsigned t0 = CLOCK; for (unsigned i = 0; i < a[2]; i++) {{ {body} }}" +
                          (" tensix_idle(); args[3] = CLOCK - t0;" if i == timed else "")) if body else ""
  return compile_kernel(None, CLOCK + _unpacker(loop(0, unpacks), kernel_config()), CLOCK + _math(slots or {}, loop(1, maths)),
                        CLOCK + _packer(loop(2, packs)))

@functools.cache
def kernels():
  mm = allocate({"mm"})
  steps = {p: tensix(tt(unpack(1, 2)), tt(matmul(mm, p)), slots=mm) for p in (1, 2, 4)}  # a matmul step: two tiles in, p phases
  # MVMULs alone: Src banks filled once and kept (the replayed phases do not give them back), released at the end
  mvmul = compile_kernel(None, CLOCK + _unpacker(tt(unpack(1, 2)), kernel_config()),
                         CLOCK + _math(mm, "unsigned t0 = CLOCK; for (unsigned i = 0; i < a[2]; i++) { " +
                                       tt([ins("SETRWC", bit_mask=0b1111)] + run(mm, "mm") * 4) +
                                       " } " + tt(matmul(mm, 1)) + " tensix_idle(); args[3] = CLOCK - t0;"), CLOCK + _packer(""))
  packs = tensix(maths=tt(math_begin(False) + MATH_END) + f" h ^= {DST_HALF};", packs=tt(pack("h", FRAMES)) + f" h ^= {DST_HALF};")
  ex = allocate({"exp"})
  return dict(steps=steps, mvmul=mvmul, pack=packs, exp=tensix(maths=tt(sfpu(["exp"], ex)), slots=ex),
              empty=compile_kernel(EMPTY), dram=compile_kernel(DRAM_READS), ping=compile_kernel(PING))

def probe(chip, n=64, rounds=32, upload=8 << 20):
  """[(what, measured, against, unit)]: the probes, n iterations of each Tensix loop, rounds of each NoC one."""
  k, rows, hz = kernels(), [], chip.telemetry().get(bh.TAG_AICLK, bh.AICLK / 1e6) * 1e6
  rt = Runtime(chip, chip.tensix[0], chip.tensix[-1])
  first, workers = rt.cores[0], len(rt.cores)
  result = lambda core, i: chip.read32(*core, rt.args() + 4 * i)
  def timed(kernel, args, units, core=first, at=3):  # one launch; the cycles one core wrote, or the launch's if at is None
    since = rt.launches
    rt.launch(kernel, args, units)
    rt.wait()
    return rt.profile(since)[0][2] if at is None else result(core, at)
  # launches: alone (host round trip, and the chip's go to last report), queued back to back, and by rectangle
  start = time.perf_counter()
  cycles = timed(k["empty"], (), workers, at=None)
  rows += [("launch, alone: host round trip", (time.perf_counter() - start) * 1e6, None, "us"), ("launch, alone: on the chip", cycles, None, "cycles")]
  since, start = rt.launches, time.perf_counter()
  rt.send(*rt.block([(k["empty"], (), workers, "")] * 100))
  rt.wait()
  rows += [("launch, 100 queued: host each", (time.perf_counter() - start) * 1e4, None, "us"),
           ("launch, 100 queued: chip each", sum(c for _, _, c in rt.profile(since)) / 100, None, "cycles")]
  # Tensix: a matmul step with its unpacks, MVMULs alone, packs, an SFPU program (cycles a tile or step)
  for p, kernel in k["steps"].items(): rows.append((f"matmul step, {p} phase{'s' * (p > 1)}, unpacks included", timed(kernel, (n,), 1) / n, 16 * p, "cycles"))
  rows.append(("MVMUL alone", 64 * n / max(timed(k["mvmul"], (n,), 1), 1), 1, "per cycle"))
  rows.append(("pack a bf16 tile", timed(k["pack"], (n,), 1) / n, None, "cycles"))
  rows.append(("SFPU exp of a tile", timed(k["exp"], (n,), 1) / n, None, "cycles"))
  # BRISC's scalar work, a mixture of experts': a token's best 8 of 128 experts (topk), a 256-token prompt's 2048 choices grouped by expert
  since = rt.launches
  Tensor.of(rt, [[float(i * 37 % 128) for i in range(128)]]).topk(8)
  rt.wait()
  rows.append(("topk, a token's 8 of 128 experts", rt.profile(since)[0][2], None, "cycles"))
  since = rt.launches
  Tensor.of(rt, [[(r * 8 + j) * 37 % 128 for j in range(8)] for r in range(256)], "int32").group(128)
  rt.wait()
  rows.append(("group, 256 tokens' 8 of 128 experts", rt.profile(since)[0][2], None, "cycles"))
  # DRAM: one worker, then all, by reads in flight (bandwidth from the slowest worker's cycles)
  for depth in (1, 2, 4, 8):
    at = rt.dram_alloc(-(-workers * depth // len(bh.DRAM)) * 2048)
    one = 2048 * depth * rounds / max(timed(k["dram"], (at, depth, rounds), 1, at=5), 1) * hz / 1e9
    timed(k["dram"], (at, depth, rounds), workers, at=5)
    every = 2048 * depth * rounds * workers / max(max(result(c, 5) for c in rt.cores), 1) * hz / 1e9
    rows += [(f"DRAM reads, {depth} in flight: one worker", one, None, "GB/s"), (f"DRAM reads, {depth} in flight: all workers", every, bh.DRAM_GBPS, "GB/s")]
    rt.dram_free(at)
  # NoC: a word there and back, between neighbours and across the grid
  for name, j in (("neighbours", 1), ("far apart", workers - 1)):
    cycles = timed(k["ping"], (0, j, bh.noc_xy(*first), bh.noc_xy(*rt.cores[j]), rounds, rt.launches << 16), workers, at=8)
    rows.append((f"NoC round trip, {name}", cycles / rounds, None, "cycles"))
  # PCIe: upload into DRAM through the BARs, and back by the chip's writes into host memory
  start = time.perf_counter()
  t = Tensor.frombytes(rt, upload // 4096, 2048, bytes(upload))
  rows.append(("upload over PCIe", upload / (time.perf_counter() - start) / 1e9, None, "GB/s"))
  start = time.perf_counter()
  t.tobytes()
  rows.append(("read back over PCIe", upload / (time.perf_counter() - start) / 1e9, None, "GB/s"))
  # launch cost by the size of the rectangle (all workers above): one, a row's (one multicast and one count, whatever the size)
  for end in (chip.tensix[1], chip.tensix[len(chip.tensix) // len(bh.TENSIX_Y) - 1]):
    small = Runtime(chip, chip.tensix[0], end)  # takes these cores over from rt, which is done
    since = small.launches
    small.launch(k["empty"])
    small.wait()
    rows.append((f"launch on {len(small.cores)} worker{'s' * (len(small.cores) > 1)}: on the chip", small.profile(since)[0][2], None, "cycles"))
  return rows

if __name__ == "__main__":
  chip = Chip(pci.open_device())
  sim = isinstance(chip.pci, pci.Sim)
  print("ttsim: these numbers check that each probe runs, they are not silicon's" if sim else f"{chip.pci.sys}: the card")
  for what, value, against, unit in probe(chip, upload=(int(sys.argv[1]) if len(sys.argv) > 1 else 8) << 20):
    print(f"  {what:48s} {value:12.2f} {unit:9s}" + (f" (bound: {against})" if against is not None else ""))
