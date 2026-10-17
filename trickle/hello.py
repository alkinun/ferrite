# Milestone 1 end to end, now on the runtime: boot firmware on one Tensix BRISC, launch a kernel compiled with plain
# clang and lld, and read back what it wrote.   TTSIM=/path/to/libttsim.so python3 -m trickle.hello
from trickle import pci
from trickle.chip import Chip
from trickle.runtime import ARGS, KERNEL, Runtime, compile_rv32

FIB = """
void kmain(volatile unsigned *args) {  // args: [n, fib(n)]
  unsigned a = 0, b = 1;
  for (unsigned i = 0; i < args[0]; i++) { unsigned t = a + b; a = b; b = t; }
  args[1] = a;
}
"""

def hello(chip, cores, n):
  rt = Runtime(chip, cores)
  rt.launch(compile_rv32(FIB, KERNEL), lambda x, y: (n, 0))
  rt.wait()
  return [chip.read32(x, y, ARGS + 4) for x, y in cores]

if __name__ == "__main__":
  chip = Chip(pci.open_device())
  print(f"BRISC on Tensix {chip.tensix[0]} says fib(30) = {hello(chip, chip.tensix[:1], 30)[0]}")
