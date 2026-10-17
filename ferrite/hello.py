# Milestone 1 end to end, now on the runtime: boot firmware on every Tensix in a rectangle, launch a kernel compiled
# with plain clang and lld on all its workers, and read back what each wrote.
#   TTSIM=/path/to/libttsim.so python3 -m ferrite.hello
from ferrite import pci
from ferrite.chip import Chip
from ferrite.runtime import Runtime, compile_kernel

FIB = """
void kmain(volatile unsigned *args) {  // args: [first unit, units, n, then fib(n)]
  unsigned a = 0, b = 1;
  for (unsigned i = 0; i < args[2]; i++) { unsigned t = a + b; a = b; b = t; }
  args[3] = a;
}
"""

def hello(chip, start, end, n):
  """fib(n) on every worker in the rectangle from start to end: {core: answer}."""
  rt = Runtime(chip, start, end)
  rt.launch(compile_kernel(FIB), (n, 0))
  rt.wait()
  return {(x, y): chip.read32(x, y, rt.args() + 12) for x, y in rt.cores}

if __name__ == "__main__":
  chip = Chip(pci.open_device())
  answers = hello(chip, chip.tensix[0], chip.tensix[-1], 30)
  print(f"{len(answers)} Tensix cores say fib(30) = {set(answers.values())}, and one more dispatched their work")
