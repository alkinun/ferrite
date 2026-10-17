# Milestone 1 end to end: compile a bare-metal rv32 program with clang and lld, run it on one Tensix tile's BRISC,
# and read back what it wrote.   TTSIM=/path/to/libttsim.so python3 -m trickle.hello
import pathlib, subprocess, tempfile
from trickle import bh, pci
from trickle.chip import Chip

IO, DONE = 0x10000, 0x600DF00D  # L1 words: [done flag, n, fib(n)]; the program writes the flag last

SRC = f"""
__attribute__((naked, section(".start"))) void _start(void) {{
  asm("li sp, {bh.BRISC_STACK_TOP:#x}\\n call main\\n 1: j 1b");  // no crt0: a stack, main, then park
}}
int main(void) {{
  volatile unsigned *io = (volatile unsigned *){IO:#x};
  unsigned a = 0, b = 1;
  for (unsigned i = 0; i < io[1]; i++) {{ unsigned t = a + b; a = b; b = t; }}
  io[2] = a;
  io[0] = {DONE:#x};
  return 0;
}}
"""

# one output section, so the flat binary is the whole image and .bss is zeros in it
LINK = "SECTIONS { .text 0 : { *(.start) *(.text*) *(.rodata*) *(.srodata*) *(.data*) *(.sdata*) *(.sbss*) *(.bss*) } }"

def compile_rv32(src):
  """C source to a flat rv32 binary that runs from address 0."""
  with tempfile.TemporaryDirectory() as d:
    (ld := pathlib.Path(d, "link.ld")).write_text(LINK)
    out = pathlib.Path(d, "prog.bin")
    subprocess.run(["clang", "--target=riscv32-unknown-elf", "-march=rv32im", "-mabi=ilp32", "-O2", "-ffreestanding", "-nostdlib",
                    "-fuse-ld=lld", f"-Wl,-T,{ld},--oformat=binary", "-x", "c", "-", "-o", str(out)], input=src.encode(), check=True)
    return out.read_bytes()

def hello(chip, x, y, n):
  """Run SRC on BRISC of Tensix (x, y): it reads n from L1 and writes back fib(n)."""
  chip.write(x, y, IO, (0).to_bytes(4, "little") + n.to_bytes(4, "little"))
  chip.run(x, y, compile_rv32(SRC))
  try: chip.wait(lambda: chip.read32(x, y, IO) == DONE, "finish the program on BRISC")
  finally: chip.halt(x, y)
  return chip.read32(x, y, IO + 8)

if __name__ == "__main__":
  chip = Chip(pci.open_device())
  x, y = chip.tensix[0]
  print(f"BRISC on Tensix ({x},{y}) says fib(30) = {hello(chip, x, y, 30)}")
