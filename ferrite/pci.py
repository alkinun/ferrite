# The device interface at the PCIe level: config space, BARs, and host memory the device can DMA to.
# Two backends with the same methods: Sim (ttsim's libttsim.so) and Card (a real card through sysfs).
import ctypes, mmap, os, time
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

  TICK = 10  # chip clocks each poll runs: the finest any measurement is (at 1000, every wait was rounded up to 1000)
  def tick(self):  # the simulator only runs when we let it
    self.lib.libttsim_clock(self.TICK)
    self.clocks += self.TICK
  def now(self): return self.clocks / bh.AICLK  # seconds of chip time, by ttsim's clock, which is not silicon's timing

class Card:
  """A real card from userspace: config space and BARs through sysfs, DMA memory from locked hugepages.
  Needs root, the kernel driver unbound, and the IOMMU off or in passthrough (bus address == physical address)."""
  def __init__(self, bdf):
    self.sys = f"/sys/bus/pci/devices/{bdf}"
    if os.path.exists(f"{self.sys}/driver"): _write(f"{self.sys}/driver/unbind", bdf)
    _write(f"{self.sys}/enable", "1")
    self.cfg = os.open(f"{self.sys}/config", os.O_RDWR | os.O_SYNC)
    self.cfg_write(0x04, self.cfg_read(0x04) | 0x6)  # memory decode, and bus mastering so the device can reach host memory
    self.bars = {}
    for i in (0, 2, 4):
      fd = os.open(f"{self.sys}/resource{i}", os.O_RDWR | os.O_SYNC)
      self.bars[i] = (m := mmap.mmap(fd, os.fstat(fd).st_size), ctypes.addressof(ctypes.c_char.from_buffer(m)))
      os.close(fd)

  def cfg_read(self, off): return int.from_bytes(os.pread(self.cfg, 4, off), "little")
  def cfg_write(self, off, val): os.pwrite(self.cfg, val.to_bytes(4, "little"), off)

  # registers want exactly one 32-bit access; bulk data can go as wide as memcpy likes
  def read(self, bar, off, n):
    addr = self.bars[bar][1] + off
    return ctypes.c_uint32.from_address(addr).value.to_bytes(4, "little") if n == 4 else ctypes.string_at(addr, n)
  def write(self, bar, off, data):
    addr = self.bars[bar][1] + off
    if len(data) == 4: ctypes.c_uint32.from_address(addr).value = int.from_bytes(data, "little")
    else: ctypes.memmove(addr, bytes(data), len(data))

  def alloc(self, size):
    """One 2 MiB hugepage, physically contiguous and locked: (memoryview, bus address). The memoryview holds the page:
    it goes back to the kernel's pool once nothing refers to it (a Runtime's ring and status live as long as it does)."""
    assert size <= 1 << 21, "one hugepage at a time"
    MAP_HUGETLB, MAP_LOCKED = 0x40000, 0x2000
    buf = mmap.mmap(-1, 1 << 21, mmap.MAP_SHARED | mmap.MAP_ANONYMOUS | mmap.MAP_POPULATE | MAP_HUGETLB | MAP_LOCKED)
    with open("/proc/self/pagemap", "rb") as f:
      f.seek(ctypes.addressof(ctypes.c_char.from_buffer(buf)) // mmap.PAGESIZE * 8)
      pfn = int.from_bytes(f.read(8), "little") & ((1 << 55) - 1)
    assert pfn, "pagemap hides physical addresses without CAP_SYS_ADMIN"
    return memoryview(buf)[:size], pfn * mmap.PAGESIZE

  def tick(self): pass  # real hardware runs on its own
  def now(self): return time.monotonic()

def _write(path, s):
  with open(path, "w") as f: f.write(s)

def open_device():
  """TTSIM=/path/to/libttsim.so selects the simulator; otherwise the first Blackhole on the PCI bus."""
  if "TTSIM" in os.environ: return Sim(os.environ["TTSIM"])
  for bdf in sorted(os.listdir("/sys/bus/pci/devices")):
    ids = [int(open(f"/sys/bus/pci/devices/{bdf}/{f}").read(), 16) for f in ("vendor", "device")]
    if ids == [bh.VENDOR, bh.DEVICE]: return Card(bdf)
  raise RuntimeError("no Blackhole on the PCI bus; set TTSIM=/path/to/libttsim.so to use the simulator")
