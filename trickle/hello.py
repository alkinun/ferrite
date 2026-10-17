# Milestone 1 end to end, now on the runtime: boot firmware on one Tensix BRISC, launch a kernel compiled with plain
# clang and lld, and read back what it wrote.   TTSIM=/path/to/libttsim.so python3 -m trickle.hello
from trickle import pci
from trickle.chip import Chip
from trickle.runtime import Runtime, compile_kernel

FIB = """
void kmain(volatile unsigned *args) {  // args: [n, fib(n)]
  unsigned a = 0, b = 1;
  for (unsigned i = 0; i < args[0]; i++) { unsigned t = a + b; a = b; b = t; }
  args[1] = a;
}
"""

def hello(chip, start, end, n):
  """fib(n) on every worker in the rectangle from start to end: {core: answer}."""
  rt = Runtime(chip, start, end)
  rt.launch(compile_kernel(FIB), lambda x, y: (n, 0))
  rt.wait()
  return {(x, y): chip.read32(x, y, rt.args() + 4) for x, y in rt.cores}

if __name__ == "__main__":
  chip = Chip(pci.open_device())
  answers = hello(chip, chip.tensix[0], chip.tensix[-1], 30)
  print(f"{len(answers)} Tensix cores say fib(30) = {set(answers.values())}, and one more dispatched their work")
