# The runtime: resident firmware on each Tensix BRISC, and kernels launched on it without resetting anything.
import pathlib, struct, subprocess, tempfile
from trickle import bh

# L1 layout, ours to choose: firmware at 0 where BRISC starts, a mailbox, kernel arguments, one kernel, then free space
MAILBOX, ARGS, KERNEL, DATA = 0x1000, 0x1100, 0x2000, 0x10000  # mailbox words: [go, done]

def compile_rv32(src, base=0):
  """C source to a flat rv32 image that runs from base: `.start` first, else `kmain`, then everything else."""
  link = f"SECTIONS {{ .text {base:#x} : {{ *(.start) *(.text.kmain) *(.text*) *(.rodata*) *(.srodata*) *(.data*) *(.sdata*) *(.sbss*) *(.bss*) }} }}"
  with tempfile.TemporaryDirectory() as d:  # one output section, so .bss is zeros inside the image
    (ld := pathlib.Path(d, "link.ld")).write_text(link)
    out = pathlib.Path(d, "prog.bin")
    subprocess.run(["clang", "--target=riscv32-unknown-elf", "-march=rv32im", "-mabi=ilp32", "-O2", "-ffreestanding", "-nostdlib",
                    "-ffunction-sections", "-fuse-ld=lld", f"-Wl,-T,{ld},--oformat=binary", "-x", "c", "-", "-o", str(out)],
                   input=src.encode(), check=True)
    return out.read_bytes()

FIRMWARE = f"""
#define MAILBOX ((volatile unsigned *){MAILBOX:#x})
#define CFG ((volatile unsigned *){bh.TENSIX_CFG:#x})
__attribute__((naked, section(".start"))) void _start(void) {{ asm("li sp, {bh.BRISC_STACK_TOP:#x}\\n j loop"); }}
void loop(void) {{
  for (unsigned seen = 0;;) {{
    while (MAILBOX[0] == seen) asm volatile("fence");  // the L0 data cache never sees the host's write; fence drops it
    seen = MAILBOX[0];
    CFG[{bh.CFG_IC_INVALIDATE}] = 1;  // forget cached instructions: the new kernel may sit where the last one was
    asm volatile("fence");  // the write is out, a read from the same block is processed after it,
    (void)CFG[0];
    asm volatile("fence");  // and we have that read's answer before jumping
    ((void (*)(volatile unsigned *)){KERNEL:#x})((volatile unsigned *){ARGS:#x});
    MAILBOX[1] = seen;  // L1 stores land in order, so the kernel's results are in place before this
  }}
}}
"""

class Runtime:
  """Firmware on the BRISC of each of cores, then any number of launches of `void kmain(volatile unsigned *args)`."""
  def __init__(self, chip, cores):
    self.chip, self.cores, self.launches = chip, cores, 0
    fw = compile_rv32(FIRMWARE)
    assert len(fw) <= MAILBOX, "firmware overlaps the mailbox"
    for x, y in cores:
      chip.halt(x, y)
      chip.write(x, y, 0, fw)
      chip.write(x, y, MAILBOX, bytes(8))
      chip.write32(x, y, bh.SOFT_RESET, bh.ALL_CORES & ~bh.BRISC)

  def launch(self, kernel, args=lambda x, y: ()):
    """Start kernel, an image from compile_rv32(src, KERNEL), on every core with args(x, y) as its arguments."""
    assert len(kernel) <= DATA - KERNEL, f"kernel is {len(kernel)} bytes, room for {DATA - KERNEL}"
    self.launches += 1
    for x, y in self.cores:
      self.chip.write(x, y, KERNEL, kernel)
      a = args(x, y)
      self.chip.write(x, y, ARGS, struct.pack(f"<{len(a)}I", *a))
      self.chip.write32(x, y, MAILBOX, self.launches)

  def wait(self, timeout=2.0):
    for x, y in self.cores:
      self.chip.wait(lambda: self.chip.read32(x, y, MAILBOX + 4) == self.launches,
                     f"finish launch {self.launches} on Tensix ({x},{y})", timeout)
