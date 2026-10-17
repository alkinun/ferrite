# A Blackhole over its PCIe device interface: NoC reads and writes through one 2 MiB TLB window, and core reset.
import time
from trickle import bh

def tlb_config(**fields):
  cfg = 0
  for name, val in fields.items():
    first, width = bh.TLB_FIELDS[name]
    assert 0 <= val < 1 << width, f"TLB {name}={val:#x} does not fit in {width} bits"
    cfg |= val << first
  return cfg

class Chip:
  def __init__(self, pci, tlb=0):
    assert (ids := pci.cfg_read(0)) == bh.VENDOR | bh.DEVICE << 16, f"not a Blackhole: {ids:#x}"
    self.pci, self.tlb, self.target = pci, tlb, None
    self.wait(lambda: self.read32(*bh.ARC, bh.ARC_BOOT_STATUS) & 7 == 5, "report ARC firmware ready")

  def window(self, x, y, addr):
    """Point the TLB at the 2 MiB block holding NoC address (x, y, addr); return the BAR0 offset that reaches addr."""
    base = addr & ~(bh.TLB_WINDOW - 1)
    if self.target != (x, y, base):
      cfg = tlb_config(addr=base >> 21, x_end=x, y_end=y, ordering=bh.STRICT)
      for i in range(3): self.pci.write(0, bh.TLB_CONFIG + 12 * self.tlb + 4 * i, (cfg >> 32 * i & 0xFFFFFFFF).to_bytes(4, "little"))
      self.target = (x, y, base)
    return self.tlb * bh.TLB_WINDOW + addr - base

  def chunks(self, x, y, addr, n):  # (BAR0 offset, size) pieces of [addr, addr+n), split where the window must move
    while n:
      size = min(n, bh.TLB_WINDOW - addr % bh.TLB_WINDOW)
      yield self.window(x, y, addr), size
      addr, n = addr + size, n - size

  def read(self, x, y, addr, n): return b"".join(self.pci.read(0, off, size) for off, size in self.chunks(x, y, addr, n))
  def write(self, x, y, addr, data):
    done = 0
    for off, size in self.chunks(x, y, addr, len(data)):
      self.pci.write(0, off, data[done:done + size])
      done += size
  def read32(self, x, y, addr): return int.from_bytes(self.read(x, y, addr, 4), "little")
  def write32(self, x, y, addr, val): self.write(x, y, addr, val.to_bytes(4, "little"))

  def wait(self, cond, what, timeout=2.0):
    deadline = time.monotonic() + timeout
    while not cond():
      if time.monotonic() > deadline: raise TimeoutError(f"chip did not {what} within {timeout}s")
      self.pci.tick()

  def run(self, x, y, prog):
    """Load prog at L1 address 0 of Tensix (x, y) and release its BRISC, which starts there. The other four stay in reset."""
    self.halt(x, y)
    self.write(x, y, 0, prog)
    self.write32(x, y, bh.SOFT_RESET, bh.ALL_CORES & ~bh.BRISC)
  def halt(self, x, y): self.write32(x, y, bh.SOFT_RESET, bh.ALL_CORES)
