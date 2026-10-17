# Goal 3's number: how fast a bf16 matmul runs on every worker. On a card the time is wall time; on ttsim it is
# simulated cycles at the chip's clock, which tracks how much work the kernels do but not how fast silicon does it
# (ttsim runs each instruction at once).   TTSIM=/path/to/libttsim.so python3 -m trickle.bench [M K N]
import sys
from trickle import bh, pci
from trickle.chip import Chip
from trickle.compiler import Tensor
from trickle.runtime import Runtime

def bench(chip, m, k, n, repeat=3):
  """Seconds per (m x k) @ (k x n), and the TFLOPS that is; inputs of 1.0, so the result is checkable."""
  rt = Runtime(chip, chip.tensix[0], chip.tensix[-1])
  a, b = (Tensor.frombytes(rt, r, c, b"\x80\x3f" * (r * c)) for r, c in ((m, k), (k, n)))  # bf16 1.0
  assert (a @ b).tolist()[0][0] == k  # compiled, and right
  start = chip.pci.now()
  for _ in range(repeat): (a @ b).realize()
  seconds = (chip.pci.now() - start) / repeat
  return seconds, 2 * m * k * n / seconds / 1e12

if __name__ == "__main__":
  chip = Chip(pci.open_device())
  m, k, n = map(int, sys.argv[1:4]) if len(sys.argv) > 3 else (512, 768, 3072)
  seconds, tflops = bench(chip, m, k, n)
  peak = len(chip.tensix) * bh.MVMUL_FLOPS * bh.AICLK / 4 / 1e12  # four fidelity phases, as our matmuls do
  print(f"{m}x{k} @ {k}x{n}: {seconds * 1e3:.3f} ms, {tflops:.2f} TFLOPS of {peak:.0f} peak at full precision"
        f"{' (ttsim cycles, not silicon time)' if isinstance(chip.pci, pci.Sim) else ''}")
