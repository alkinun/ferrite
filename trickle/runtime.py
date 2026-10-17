# The runtime: resident firmware on each Tensix BRISC, and kernels launched on it without resetting anything.
import pathlib, struct, subprocess, tempfile
from trickle import bh

# L1 layout, ours to choose: firmware at 0 where BRISC starts, a mailbox, kernel arguments, one kernel, then free space.
# Mailbox words: [go, report x | y << 6, report address lo, hi, done]; done is 16-byte aligned like its host slot.
MAILBOX, ARGS, KERNEL, DATA = 0x1000, 0x1100, 0x2000, 0x10000

# Prepended to every device program: NoC transfers through request initiator 0 of NIU 0 (ISA NoC/MemoryMap.md).
DEVICE = f"""
#define L1_DATA {DATA:#x}
#define NIU(reg) (((volatile unsigned *){bh.NIU:#x})[(reg) / 4])
enum {{ TARG_LO = 0x0, TARG_MID = 0x4, TARG_HI = 0x8, RET_LO = 0xC, RET_MID = 0x10, RET_HI = 0x14, PACKET_TAG = 0x18,
        CTRL = 0x1C, LEN = 0x20, CMD = 0x40, ID_LOGICAL = 0x148, OUTSTANDING = 0x240 }};
enum {{ READ = 0x10, WRITE = 0x12, MAX_PACKET = 16384 }};  // both acknowledged, so OUTSTANDING counts them
static inline unsigned noc_me(void) {{ return NIU(ID_LOGICAL) & 0xFFF; }}  // this tile's translated x | y << 6
static inline void noc_issue(unsigned long long targ, unsigned targ_xy, unsigned long long ret, unsigned ret_xy,
                             unsigned ctrl, unsigned n) {{
  while (NIU(CMD) & 1) {{}}  // the initiator still holds the previous request
  NIU(TARG_LO) = targ; NIU(TARG_MID) = targ >> 32; NIU(TARG_HI) = targ_xy;
  NIU(RET_LO) = ret; NIU(RET_MID) = ret >> 32; NIU(RET_HI) = ret_xy;
  NIU(PACKET_TAG) = 0; NIU(CTRL) = ctrl; NIU(LEN) = n;
  NIU(CMD) = 1;
}}
// until this core's earlier stores to L1 are processed: they are out, a load behind them is answered
static inline void l1_settle(unsigned addr) {{ asm volatile("fence"); (void)*(volatile unsigned *)addr; asm volatile("fence"); }}
// Copy n bytes between (x, y, addr) and this tile's L1. The low 4 bits of the two addresses must match (6 for DRAM).
static inline void noc_read(unsigned x, unsigned y, unsigned long long addr, unsigned l1, unsigned n) {{
  for (unsigned k; n; addr += k, l1 += k, n -= k) noc_issue(addr, x | y << 6, l1, noc_me(), READ, k = n < MAX_PACKET ? n : MAX_PACKET);
}}
static inline void noc_write(unsigned l1, unsigned x, unsigned y, unsigned long long addr, unsigned n) {{
  l1_settle(l1);  // the NIU reads L1 on its own; our stores must be there first
  for (unsigned k; n; addr += k, l1 += k, n -= k) noc_issue(l1, noc_me(), addr, x | y << 6, WRITE, k = n < MAX_PACKET ? n : MAX_PACKET);
}}
static inline void noc_wait(void) {{  // until every read has landed and every write is acknowledged
  (void)NIU(CMD);  // reading the initiator first keeps the counter read behind the last request
  while (NIU(OUTSTANDING)) {{}}
  asm volatile("fence");  // drop L0 data cache lines the NoC may have just overwritten
}}
"""

def compile_rv32(src, base=0):
  """C source, after DEVICE, to a flat rv32 image that runs from base: `.start` first, else `kmain`, then the rest."""
  link = f"SECTIONS {{ .text {base:#x} : {{ *(.start) *(.text.kmain) *(.text*) *(.rodata*) *(.srodata*) *(.data*) *(.sdata*) *(.sbss*) *(.bss*) }} }}"
  with tempfile.TemporaryDirectory() as d:  # one output section, so .bss is zeros inside the image
    (ld := pathlib.Path(d, "link.ld")).write_text(link)
    out = pathlib.Path(d, "prog.bin")
    subprocess.run(["clang", "--target=riscv32-unknown-elf", "-march=rv32im", "-mabi=ilp32", "-O2", "-ffreestanding", "-nostdlib",
                    "-ffunction-sections", "-fuse-ld=lld", f"-Wl,-T,{ld},--oformat=binary", "-x", "c", "-", "-o", str(out)],
                   input=(DEVICE + src).encode(), check=True)
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
    noc_wait();  // whatever the kernel sent is in place before we say it is done
    MAILBOX[4] = seen;
    noc_write({MAILBOX + 16:#x}, MAILBOX[1] & 63, MAILBOX[1] >> 6, MAILBOX[2] | (unsigned long long)MAILBOX[3] << 32, 4);
    noc_wait();
  }}
}}
"""

class Runtime:
  """Firmware on the BRISC of every Tensix in the rectangle from start to end, then any number of launches of
  `void kmain(volatile unsigned *args)`. What every core gets is multicast: one write over PCIe, whatever the count.
  Each core reports a finished launch by writing its count into its own 16-byte slot of a host buffer."""
  def __init__(self, chip, start, end):
    self.chip, self.start, self.end, self.launches = chip, start, end, 0
    self.cores = [(x, y) for x, y in chip.tensix if start[0] <= x <= end[0] and start[1] <= y <= end[1]]
    self.done, bus = chip.pci.alloc(16 * len(self.cores))
    fw = compile_rv32(FIRMWARE)
    assert len(fw) <= MAILBOX, "firmware overlaps the mailbox"
    chip.halt(*end, start)
    chip.write(*end, 0, fw, start)
    for i, (x, y) in enumerate(self.cores):
      chip.write(x, y, MAILBOX, struct.pack("<5I", 0, bh.PCIE[0] | bh.PCIE[1] << 6, (bus + 16 * i) & 0xFFFFFFFF, (bus + 16 * i) >> 32, 0))
    chip.write32(*end, bh.SOFT_RESET, bh.ALL_CORES & ~bh.BRISC, start)

  def launch(self, kernel, args=lambda x, y: ()):
    """Start kernel, an image from compile_rv32(src, KERNEL), on every core with args(x, y) as its arguments."""
    assert len(kernel) <= DATA - KERNEL, f"kernel is {len(kernel)} bytes, room for {DATA - KERNEL}"
    self.launches += 1
    self.chip.write(*self.end, KERNEL, kernel, self.start)
    for x, y in self.cores:
      a = args(x, y)
      self.chip.write(x, y, ARGS, struct.pack(f"<{len(a)}I", *a))
    self.chip.write32(*self.end, MAILBOX, self.launches, self.start)

  def wait(self, timeout=2.0):
    """Until every core has reported the last launch: polls host memory, not the chip."""
    for i, (x, y) in enumerate(self.cores):
      self.chip.wait(lambda: int.from_bytes(self.done[16 * i:16 * i + 4], "little") == self.launches,
                     f"finish launch {self.launches} on Tensix ({x},{y})", timeout)
