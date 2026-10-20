# How fast a bf16 matmul runs on every worker (ROADMAP, stage 3), its B (the weights) bf16 or a block float. On a card
# the time is wall time; on ttsim it is simulated cycles, the RISC-V work on the critical path (queued Tensix
# instructions cost ttsim nothing, and its DRAM answers at once), not silicon's speed. So it also counts what the
# workers read from DRAM, against reading each input once: on silicon a big matmul waits on DRAM more than on its
# MVMULs, and decode (M of 1) on nothing else, so on a card it gives the GB/s of reading each input once too.
#   TTSIM=/path/to/libttsim.so python3 -m ferrite.bench [M K N [fidelity phases [bf16|bfp8|bfp4]]]
import sys
from ferrite import bh, pci, tensor
from ferrite.chip import Chip
from ferrite.compiler import BFP, tile_bytes
from ferrite.tensor import Tensor
from ferrite.runtime import Runtime

def bench(chip, m, k, n, weights="bf16", repeat=3):
  """Seconds per (m x k) @ (k x n) with B in weights' dtype, the TFLOPS that is, the GB/s of reading each input once,
  and how many times over they were read from DRAM; inputs of 1.0, so the result is checkable. Reads are counted as
  responses, one a bf16 tile and two a block float's (its exponents, its values)."""
  rt = Runtime(chip, chip.tensix[0], chip.tensix[-1])
  a, b = (Tensor.frombytes(rt, r, c, b"\x80\x3f" * (r * c)) for r, c in ((m, k), (k, n)))  # bf16 1.0
  b = b.cast(weights).realize()
  assert (a @ b).tolist()[0][0] == k  # compiled, and right
  reads = lambda: sum(chip.read32(x, y, bh.NIU_READS) for x, y in rt.cores)
  start, read = chip.pci.now(), reads()
  for _ in range(repeat): (a @ b).realize()  # queued back to back
  rt.wait()
  seconds, read = (chip.pci.now() - start) / repeat, (reads() - read) / repeat
  ta, tb = -(-m // 32) * -(-k // 32), -(-k // 32) * -(-n // 32)  # each input's tiles
  once = ta * 2048 + tb * tile_bytes(weights)
  return seconds, 2 * m * k * n / seconds / 1e12, once / seconds / 1e9, read / (ta + tb * (2 if weights in BFP else 1))

if __name__ == "__main__":
  chip = Chip(pci.open_device())
  m, k, n = map(int, sys.argv[1:4]) if len(sys.argv) > 3 else (512, 768, 3072)
  if len(sys.argv) > 4: tensor.FIDELITY = int(sys.argv[4])
  weights = sys.argv[5] if len(sys.argv) > 5 else "bf16"
  seconds, tflops, gbs, reread = bench(chip, m, k, n, weights)
  peak = len(chip.tensix) * bh.MVMUL_FLOPS * bh.AICLK / tensor.FIDELITY / 1e12  # a product takes FIDELITY phases
  speed = (f"{seconds * bh.AICLK / 1e3:.1f}k ttsim cycles (the RISC-V work on its critical path, not silicon time)"
           if isinstance(chip.pci, pci.Sim) else
           f"{seconds * 1e3:.3f} ms, {tflops:.2f} TFLOPS of {peak:.0f} peak at {tensor.FIDELITY} fidelity phases, {gbs:.0f} GB/s")
  print(f"{m}x{k} @ {k}x{n} ({weights} B): {speed}; DRAM reads {reread:.1f}x the inputs")
