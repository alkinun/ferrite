# The device interface at the PCIe level: config space, BARs, and host memory the device can DMA to.
# Two backends with the same methods: Sim (ttsim's libttsim.so) and Card (a real card through sysfs).
import array, ctypes, gc, mmap, os, time
from ferrite import bh

class Sim:
  """libttsim.so: a virtual Blackhole driven by forwarding config and BAR accesses. One per process."""
  def __init__(self, path):
    self.lib = lib = ctypes.CDLL(path)
    lib.libttsim_pci_config_rd32.restype = ctypes.c_uint32
    lib.libttsim_pci_config_rd32.argtypes = [ctypes.c_uint32] * 2
    lib.libttsim_pci_config_wr32.argtypes = [ctypes.c_uint32] * 3
    lib.libttsim_pci_mem_rd_bytes.argtypes = lib.libttsim_pci_mem_wr_bytes.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint32]
    lib.libttsim_clock.argtypes = [ctypes.c_uint32]
    # the device reaches host memory through these; they must stay referenced for as long as the library lives
    cb = ctypes.CFUNCTYPE(None, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint32)
    self.cbs = (cb(lambda bus, p, n: ctypes.memmove(p, self.host(bus, n), n)),
                cb(lambda bus, p, n: ctypes.memmove(self.host(bus, n), p, n)))
    lib.libttsim_set_pci_dma_mem_callbacks(*self.cbs)
    lib.libttsim_init()
    self.bars = {i: self.cfg_read(0x10 + 4 * i) & ~0xF | self.cfg_read(0x14 + 4 * i) << 32 for i in (0, 2, 4)}
    self.dma, self.clocks = [], 0  # (bus address, size, host address, buffer); chip cycles run so far

  def cfg_read(self, off): return self.lib.libttsim_pci_config_rd32(0, off)
  def cfg_write(self, off, val): self.lib.libttsim_pci_config_wr32(0, off, val)

  def read(self, bar, off, n):
    buf = ctypes.create_string_buffer(n)
    self.lib.libttsim_pci_mem_rd_bytes(self.bars[bar] + off, buf, n)
    return buf.raw
  def write(self, bar, off, data): self.lib.libttsim_pci_mem_wr_bytes(self.bars[bar] + off, bytes(data), len(data))

  def alloc(self, size):
    """Host memory the device can reach: (memoryview, bus address). Sim bus addresses are made up: one per 4 GiB."""
    buf, bus = mmap.mmap(-1, size), (len(self.dma) + 1) << 32
    self.dma.append((bus, size, ctypes.addressof(ctypes.c_char.from_buffer(buf)), buf))
    return memoryview(buf), bus
  def host(self, bus, n):
    for base, size, addr, _ in self.dma:
      if base <= bus and bus + n <= base + size: return addr + bus - base
    raise RuntimeError(f"device DMA to {bus:#x}+{n}, which is not host memory we allocated")

  TICK = 10  # chip clocks a poll runs, so every wait (and so every cycle count) is rounded up to a multiple of it
  def tick(self):  # the simulator only runs when we let it
    self.lib.libttsim_clock(self.TICK)
    self.clocks += self.TICK
  def now(self): return self.clocks / bh.AICLK  # seconds of chip time, by ttsim's clock, which is not silicon's timing

class Card:
  """A real card from userspace: config space and BARs through sysfs, DMA memory from locked hugepages.
  Needs root, the kernel driver unbound, and the IOMMU off or in passthrough (bus address == physical address)."""
  def __init__(self, bdf):
    if os.geteuid() != 0: raise PermissionError("a card wants root: its config space, its BARs, and physical addresses for DMA")
    self.sys = f"/sys/bus/pci/devices/{bdf}"
    if os.path.exists(f"{self.sys}/iommu_group/type") and (kind := open(f"{self.sys}/iommu_group/type").read().strip()) != "identity":
      raise RuntimeError(f"the IOMMU translates {bdf}'s DMA ({kind}), and Card gives it physical addresses: boot with iommu=pt")
    if os.path.exists(f"{self.sys}/driver"): _write(f"{self.sys}/driver/unbind", bdf)
    _write(f"{self.sys}/enable", "1")
    self.cfg = os.open(f"{self.sys}/config", os.O_RDWR | os.O_SYNC)
    self.cfg_write(0x04, self.cfg_read(0x04) | 0x6)  # memory decode, and bus mastering so the device can reach host memory
    if at := self.capability(0x10):  # PCI Express: reads of host memory as requests of up to 4 KiB (Max Read Request Size),
      os.pwrite(self.cfg, (self.cfg_read(at + 8) & 0x8FFF | 5 << 12).to_bytes(2, "little"), at + 8)  # as kmd sets them up
    fd = os.open(f"{self.sys}/resource0", os.O_RDWR | os.O_SYNC)  # BAR0 alone: TLB windows and registers are all ferrite uses
    self.bars = {0: (m := mmap.mmap(fd, os.fstat(fd).st_size), ctypes.addressof(ctypes.c_char.from_buffer(m)))}  # (BAR4 may not
    os.close(fd)                                                                                                     # be all there)
    for i in range(32): self.write(0, bh.TLB_STRIDED + 4 * i, bytes(4))  # TLBs 0-31 multicast to every tile of a rectangle, as kmd
                                                                          # leaves them, whatever an earlier user set
  def cfg_read(self, off): return int.from_bytes(os.pread(self.cfg, 4, off), "little")
  def cfg_write(self, off, val): os.pwrite(self.cfg, val.to_bytes(4, "little"), off)
  def capability(self, cap):  # where a PCI capability sits in config space, or 0
    at = self.cfg_read(0x34) & 0xFC
    while at and self.cfg_read(at) & 0xFF != cap: at = self.cfg_read(at) >> 8 & 0xFC
    return at

  def read(self, bar, off, n): return _load(self.bars[bar][1] + off, n)
  def write(self, bar, off, data): _store(self.bars[bar][1] + off, data)

  def alloc(self, size):
    """One 2 MiB hugepage, physically contiguous and locked: (memoryview, bus address). The memoryview holds the page:
    it goes back to the kernel's pool once nothing refers to it (a Runtime's ring and status live as long as it does)."""
    assert size <= 1 << 21, "one hugepage at a time"
    MAP_HUGETLB, MAP_LOCKED, MAP_HUGE_2MB = 0x40000, 0x2000, 21 << 26  # 2 MiB pages whatever the host's default size
    for tries in (1, 0):  # a Runtime dropped in a reference cycle (its numbers' tensors refer to it) holds its page until collected
      try: buf = mmap.mmap(-1, 1 << 21, mmap.MAP_SHARED | mmap.MAP_ANONYMOUS | mmap.MAP_POPULATE | MAP_HUGETLB | MAP_LOCKED | MAP_HUGE_2MB); break
      except OSError:
        if not tries: raise MemoryError("no free 2 MiB hugepage for memory the card can reach: echo 4 > /proc/sys/vm/nr_hugepages") from None
        gc.collect()
    with open("/proc/self/pagemap", "rb") as f:
      f.seek(ctypes.addressof(ctypes.c_char.from_buffer(buf)) // mmap.PAGESIZE * 8)
      pfn = int.from_bytes(f.read(8), "little") & ((1 << 55) - 1)
    assert pfn, "pagemap hides physical addresses without CAP_SYS_ADMIN"
    return memoryview(buf)[:size], pfn * mmap.PAGESIZE

  def tick(self): pass  # real hardware runs on its own
  def now(self): return time.monotonic()

# A BAR is mapped uncached, where libc's memcpy may copy a byte at a time (rep movsb), each byte a PCIe transaction of its
# own: so a word at a time, the widest that both the address and the length allow (a register's 4 bytes in one access)
WORDS = {8: ("Q", ctypes.c_uint64), 4: ("I", ctypes.c_uint32)}
def _width(addr, n): return next((w for w in WORDS if addr % w == 0 and n % w == 0), None)
def _load(addr, n):
  if (w := _width(addr, n)) is None: return ctypes.string_at(addr, n)
  code, word = WORDS[w]
  return array.array(code, (word * (n // w)).from_address(addr)).tobytes()
def _store(addr, data):
  if (w := _width(addr, len(data))) is None: return ctypes.memmove(addr, bytes(data), len(data))
  code, word = WORDS[w]
  (word * (len(data) // w)).from_address(addr)[:] = memoryview(data).cast("B").cast(code)

def _write(path, s):
  with open(path, "w") as f: f.write(s)

def find_card():
  """The first Blackhole on the PCI bus, by its sysfs name (domain:bus:device.function)."""
  for bdf in sorted(os.listdir("/sys/bus/pci/devices")):
    ids = [int(open(f"/sys/bus/pci/devices/{bdf}/{f}").read(), 16) for f in ("vendor", "device")]
    if ids == [bh.VENDOR, bh.DEVICE]: return bdf
  raise RuntimeError("no Blackhole on the PCI bus; set TTSIM=/path/to/libttsim.so to use the simulator")

def open_device():
  """TTSIM=/path/to/libttsim.so selects the simulator; otherwise the first Blackhole on the PCI bus."""
  return Sim(os.environ["TTSIM"]) if "TTSIM" in os.environ else Card(find_card())

def reset(bdf, timeout=10.0):
  """Reset the whole chip, as tt-kmd's ASIC reset does, for a card whose NoC or firmware hangs (a hung kernel needs
  none: the next Runtime resets the cores it uses). Two config writes fire the PCIe controller's interface timer, and
  PCI_COMMAND's parity bit marks the reset coming: it is done when that bit is clear again, or when the card has left the
  bus and come back. Then the kernel enumerates the card again (remove, rescan), which restores its config space as at
  boot. Nothing may have the card open. Untested on silicon (NOTES.md)."""
  if os.geteuid() != 0: raise PermissionError("a reset wants root: it writes the card's config space and sysfs")
  path, deadline, gone = f"/sys/bus/pci/devices/{bdf}", time.monotonic() + timeout, False
  fd = os.open(f"{path}/config", os.O_RDWR | os.O_SYNC)
  command = int.from_bytes(os.pread(fd, 2, 0x04), "little")
  os.pwrite(fd, (command | bh.RESET_MARKER).to_bytes(2, "little"), 0x04)
  for off, val in bh.RESET_TIMER: os.pwrite(fd, val.to_bytes(4, "little"), off)
  os.close(fd)
  def back():  # whether the reset is over
    nonlocal gone
    if not os.path.exists(path): gone = True; return False
    try:
      with open(f"{path}/config", "rb") as f: f.seek(0x04); return gone or not int.from_bytes(f.read(2), "little") & bh.RESET_MARKER
    except OSError: return False  # its config space does not answer yet
  while not back():
    if time.monotonic() > deadline: raise TimeoutError(f"{bdf} did not come back from its reset within {timeout}s")
    time.sleep(0.1)
  _write(f"{path}/remove", "1")
  _write("/sys/bus/pci/rescan", "1")
  while not os.path.exists(f"{path}/config"):
    if time.monotonic() > deadline: raise TimeoutError(f"{bdf} did not come back from a rescan within {timeout}s")
    time.sleep(0.1)
