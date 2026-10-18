# How fast a bf16 matmul runs on every worker (ROADMAP, stage 3). On a card the time is wall time; on ttsim it is
# simulated cycles, the RISC-V work on the critical path (queued Tensix instructions cost ttsim nothing, and its DRAM
# answers at once), not silicon's speed. So it also counts what the workers read from DRAM, against reading each input
# once: on silicon a big matmul waits on DRAM more than on its MVMULs.
#   TTSIM=/path/to/libttsim.so python3 -m ferrite.bench [M K N [fidelity phases]]
import sys
from ferrite import bh, pci, tensor
from ferrite.chip import Chip
from ferrite.tensor import Tensor
from ferrite.runtime import Runtime

def bench(chip, m, k, n, repeat=3):
  """Seconds per (m x k) @ (k x n), the TFLOPS that is, and how many times over the inputs were read from DRAM; inputs
  of 1.0, so the result is checkable. Reads are counted as responses, each a 2 KiB tile here."""
  rt = Runtime(chip, chip.tensix[0], chip.tensix[-1])
  a, b = (Tensor.frombytes(rt, r, c, b"\x80\x3f" * (r * c)) for r, c in ((m, k), (k, n)))  # bf16 1.0
  assert (a @ b).tolist()[0][0] == k  # compiled, and right
  reads = lambda: sum(chip.read32(x, y, bh.NIU_READS) for x, y in rt.cores)
  start, read = chip.pci.now(), reads()
  for _ in range(repeat): (a @ b).realize()  # queued back to back
  rt.wait()
  seconds, read = (chip.pci.now() - start) / repeat, (reads() - read) / repeat * 2048
  return seconds, 2 * m * k * n / seconds / 1e12, read / (2 * (m * k + k * n))

if __name__ == "__main__":
  chip = Chip(pci.open_device())
  m, k, n = map(int, sys.argv[1:4]) if len(sys.argv) > 3 else (512, 768, 3072)
  if len(sys.argv) > 4: tensor.FIDELITY = int(sys.argv[4])
  seconds, tflops, reread = bench(chip, m, k, n)
  peak = len(chip.tensix) * bh.MVMUL_FLOPS * bh.AICLK / tensor.FIDELITY / 1e12  # a product takes FIDELITY phases
  speed = (f"{seconds * bh.AICLK / 1e3:.1f}k ttsim cycles (the RISC-V work on its critical path, not silicon time)"
           if isinstance(chip.pci, pci.Sim) else f"{seconds * 1e3:.3f} ms, {tflops:.2f} TFLOPS of {peak:.0f} peak at {tensor.FIDELITY} fidelity phases")
  print(f"{m}x{k} @ {k}x{n}: {speed}; DRAM reads {reread:.1f}x the inputs")
