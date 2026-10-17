# The runtime: resident firmware on each Tensix BRISC, kernels launched on it without resetting anything, and a command
# ring in host memory that one of those cores, the dispatcher, reads by itself.
import pathlib, struct, subprocess, tempfile
from trickle import bh

# L1 layout, ours to choose. Every core: firmware at 0 where BRISC starts. A worker: a mailbox [go, report x | y << 6,
# report address lo, hi, done] (done is 16-byte aligned like the slot it is copied to), kernel arguments, one kernel, then
# free space. The dispatcher: its config [doorbell, ring lo, hi, ring size, status lo, hi, workers] and the workers'
# x | y << 6 after it, an outbox [go] [rptr, launches done], a done slot per worker, and staging for one command.
MAILBOX, ARGS, KERNEL, DATA = 0x1000, 0x1100, 0x2000, 0x10000
DISPATCH, WORKERS, OUTBOX, DONE, STAGING = 0x1000, 0x1040, 0x1400, 0x2000, DATA
STAGING_ROOM = bh.L1_SIZE - STAGING

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
static inline void l1_settle(unsigned addr) {{ asm volatile("fence" ::: "memory"); (void)*(volatile unsigned *)addr; asm volatile("fence" ::: "memory"); }}
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
  asm volatile("fence" ::: "memory");  // drop L0 data cache lines the NoC may have just overwritten
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
    while (MAILBOX[0] == seen) asm volatile("fence" ::: "memory");  // the L0 data cache never sees NoC writes; fence drops it
    seen = MAILBOX[0];
    CFG[{bh.CFG_IC_INVALIDATE}] = 1;  // forget cached instructions: the new kernel may sit where the last one was
    asm volatile("fence" ::: "memory");  // the write is out, a read from the same block is processed after it,
    (void)CFG[0];
    asm volatile("fence" ::: "memory");  // and we have that read's answer before jumping
    ((void (*)(volatile unsigned *)){KERNEL:#x})((volatile unsigned *){ARGS:#x});
    noc_wait();  // whatever the kernel sent is in place before we say it is done
    MAILBOX[4] = seen;
    noc_write({MAILBOX + 16:#x}, MAILBOX[1] & 63, MAILBOX[1] >> 6, MAILBOX[2] | (unsigned long long)MAILBOX[3] << 32, 4);
    noc_wait();
  }}
}}
"""

# A command is a 16-byte header [length, kernel bytes, argument bytes per worker, 0], the kernel, then each worker's
# arguments; everything 16-byte aligned, the whole 64-byte aligned. Length 0 means: the rest of the ring is unused.
DISPATCHER = f"""
#define CONFIG ((volatile unsigned *){DISPATCH:#x})
#define WORKER ((volatile unsigned *){WORKERS:#x})
#define OUT ((volatile unsigned *){OUTBOX:#x})
#define DONE ((volatile unsigned *){DONE:#x})
__attribute__((naked, section(".start"))) void _start(void) {{ asm("li sp, {bh.BRISC_STACK_TOP:#x}\\n j loop"); }}
void loop(void) {{
  unsigned long long ring = CONFIG[1] | (unsigned long long)CONFIG[2] << 32, status = CONFIG[4] | (unsigned long long)CONFIG[5] << 32;
  unsigned size = CONFIG[3], n = CONFIG[6], rptr = 0, launches = 0;
  volatile unsigned *cmd = (volatile unsigned *){STAGING:#x};
  for (;;) {{
    while (CONFIG[0] == rptr) asm volatile("fence" ::: "memory");  // the host rings the doorbell with its write pointer
    unsigned long long at = ring + rptr % size;
    noc_read({bh.PCIE[0]}, {bh.PCIE[1]}, at, {STAGING:#x}, 16);
    noc_wait();
    if (cmd[0] == 0) {{ rptr += size - rptr % size; continue; }}
    unsigned kernel = cmd[1], args = cmd[2];
    noc_read({bh.PCIE[0]}, {bh.PCIE[1]}, at + 16, {STAGING + 16:#x}, cmd[0] - 16);
    noc_wait();
    launches++;
    for (unsigned i = 0; i < n; i++) {{  // a worker is idle here: it finished the last launch before we got here
      noc_write({STAGING + 16:#x}, WORKER[i] & 63, WORKER[i] >> 6, {KERNEL:#x}, kernel);
      noc_write({STAGING + 16:#x} + kernel + i * args, WORKER[i] & 63, WORKER[i] >> 6, {ARGS:#x}, args);
    }}
    noc_wait();  // every kernel and argument has landed before any go
    OUT[0] = launches;
    for (unsigned i = 0; i < n; i++) noc_write({OUTBOX:#x}, WORKER[i] & 63, WORKER[i] >> 6, {MAILBOX:#x}, 4);
    noc_wait();
    for (unsigned i = 0; i < n; i++) while (DONE[4 * i] != launches) asm volatile("fence" ::: "memory");
    rptr += cmd[0];
    OUT[4] = rptr;
    OUT[5] = launches;
    noc_write({OUTBOX + 16:#x}, {bh.PCIE[0]}, {bh.PCIE[1]}, status, 8);
    noc_wait();
  }}
}}
"""

class Runtime:
  """A rectangle of Tensix from start to end: its last core dispatches, the rest are workers running
  `void kmain(volatile unsigned *args)`. Launches go into a ring in host memory that the dispatcher reads by itself, so
  a launch costs the host one PCIe write, the doorbell. The dispatcher runs one launch at a time and writes
  [ring read pointer, launches done] back into host memory, where wait() and the ring's flow control look."""
  def __init__(self, chip, start, end, ring=1 << 20):
    assert ring & (ring - 1) == 0 and ring % 64 == 0, "ring size must be a power of two: pointers wrap at 2**32"
    cores = [(x, y) for x, y in chip.tensix if start[0] <= x <= end[0] and start[1] <= y <= end[1]]
    assert len(cores) >= 2, "a rectangle needs a dispatcher and at least one worker"
    self.chip, self.launches, self.wptr, self.dram = chip, 0, 0, 0
    *self.cores, self.dispatcher = cores
    buf, bus = chip.pci.alloc(64 + ring)
    self.status, self.ring = buf[:64], buf[64:]
    worker, dispatcher = compile_rv32(FIRMWARE), compile_rv32(DISPATCHER)
    assert max(len(worker), len(dispatcher)) <= MAILBOX, "firmware overlaps the mailbox"
    (dx, dy), n = self.dispatcher, len(self.cores)
    chip.halt(*end, start)
    chip.write(*end, 0, worker, start)
    for i, (x, y) in enumerate(self.cores): chip.write(x, y, MAILBOX, struct.pack("<5I", 0, dx | dy << 6, DONE + 16 * i, 0, 0))
    chip.write(dx, dy, 0, dispatcher)
    chip.write(dx, dy, DISPATCH, struct.pack("<7I", 0, (bus + 64) & 0xFFFFFFFF, (bus + 64) >> 32, ring, bus & 0xFFFFFFFF, bus >> 32, n))
    chip.write(dx, dy, WORKERS, struct.pack(f"<{n}I", *[x | y << 6 for x, y in self.cores]))
    chip.write(dx, dy, DONE, bytes(16 * n))
    self.doorbell = chip.map(dx, dy, DISPATCH)  # a TLB of its own, so ringing never has to move a window first
    chip.write32(*end, bh.SOFT_RESET, bh.ALL_CORES & ~bh.BRISC, start)

  def launch(self, kernel, args=lambda x, y: ()):
    """Queue kernel, an image from compile_rv32(src, KERNEL), for every worker with args(x, y) as its arguments."""
    assert len(kernel) <= DATA - KERNEL, f"kernel is {len(kernel)} bytes, room for {DATA - KERNEL}"
    kernel = kernel.ljust(-(-len(kernel) // 16) * 16, b"\0")
    a = [struct.pack(f"<{len(v)}I", *v) for v in (args(x, y) for x, y in self.cores)]
    stride = -(-max(map(len, a)) // 16) * 16
    assert stride <= KERNEL - ARGS, f"{stride} bytes of arguments, room for {KERNEL - ARGS}"
    body = kernel + b"".join(v.ljust(stride, b"\0") for v in a)
    length = -(-(16 + len(body)) // 64) * 64
    cmd = struct.pack("<4I", length, len(kernel), stride, 0) + body.ljust(length - 16, b"\0")
    size, at = len(self.ring), self.wptr % len(self.ring)
    skip = size - at if at + length > size else 0  # commands never wrap: mark the tail unused and start over
    assert length <= STAGING_ROOM and length + skip <= size, f"a {length}-byte command does not fit"
    self.chip.wait(lambda: (self.wptr - self.read_status(0)) % 2**32 + skip + length <= size, "free space in the ring")
    if skip:
      self.ring[at:at + 4] = bytes(4)
      self.wptr, at = self.wptr + skip, 0
    self.ring[at:at + length] = cmd
    self.wptr += length
    self.launches += 1
    self.chip.pci.write(0, self.doorbell, (self.wptr % 2**32).to_bytes(4, "little"))  # the launch's one PCIe write

  def dram_alloc(self, size):
    """size bytes at the same address in every DRAM bank; a bump allocator, nothing is freed yet."""
    addr, self.dram = self.dram, self.dram + -(-size // 64) * 64
    return addr

  def read_status(self, i): return int.from_bytes(self.status[4 * i:4 * i + 4], "little")

  def wait(self, timeout=2.0):
    """Until every queued launch has finished on every worker: polls host memory, not the chip. On timeout, asks the
    dispatcher which launch it is on and which workers have not reported it."""
    try: self.chip.wait(lambda: self.read_status(1) == self.launches % 2**32, f"finish launch {self.launches}", timeout)
    except TimeoutError as e:
      current, done = self.chip.read32(*self.dispatcher, OUTBOX), self.chip.read(*self.dispatcher, DONE, 16 * len(self.cores))
      stuck = [c for i, c in enumerate(self.cores) if int.from_bytes(done[16 * i:16 * i + 4], "little") != current]
      raise TimeoutError(f"{e}: the dispatcher is on launch {current}, still running on {stuck}") from None
