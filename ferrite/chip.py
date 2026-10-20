# A Blackhole over its PCIe device interface: NoC reads and writes through a shared 2 MiB TLB window (or a TLB of its
# own: map), waits with a timeout, ARC messages and telemetry, and core reset.
import struct, time
from ferrite import bh

def tlb_config(**fields):
  cfg = 0
  for name, val in fields.items():
    first, width = bh.TLB_FIELDS[name]
    assert 0 <= val < 1 << width, f"TLB {name}={val:#x} does not fit in {width} bits"
    cfg |= val << first
  return cfg

class Chip:
  def __init__(self, pci, boot=2.0):  # boot: seconds to wait for ARC's firmware (minutes after a reset: umd allows 300)
    assert (ids := pci.cfg_read(0)) == bh.VENDOR | bh.DEVICE << 16, f"not a Blackhole: {ids:#x}"
    self.pci, self.tlb, self.target = pci, 0, None  # TLB 0 is the shared window; map hands out the rest
    self.free_tlbs = iter(range(1, bh.TLB_COUNT))
    self.wait(lambda: self.read32(*bh.ARC, bh.ARC_BOOT_STATUS) & 7 == 5, "report ARC firmware ready", boot)
    self.arc_msg(bh.POWER_ON)  # tt-kmd clock-gates Tensix whenever nobody has the card open
    t = self.telemetry()
    self.tensix = [(x, y) for y in bh.TENSIX_Y for x in bh.TENSIX_X[:t[bh.TAG_TENSIX_COLS].bit_count()]]
    banks = [b for b in range(len(bh.DRAM)) if t[bh.TAG_GDDR_ENABLED] >> b & 1]
    self.wait(lambda: all(self.telemetry()[bh.TAG_GDDR_STATUS] >> 2 * b & 1 for b in banks), "finish GDDR training", 60)

  def map(self, x, y, addr, start=None, tlb=None):
    """Point a TLB (by default a fresh one, kept for good) at the 2 MiB block holding NoC address (x, y, addr), or for
    writes at that block in every Tensix of the rectangle from start to (x, y); return the BAR0 offset that reaches addr."""
    if tlb is None and (tlb := next(self.free_tlbs, None)) is None: raise RuntimeError(f"all {bh.TLB_COUNT} TLBs are taken")
    base = addr & ~(bh.TLB_WINDOW - 1)
    mcast = {} if start is None else dict(x_start=start[0], y_start=start[1], mcast=1, static_vc=1, vc_class=bh.MCAST_VC)
    cfg = tlb_config(addr=base >> 21, x_end=x, y_end=y, ordering=bh.STRICT, **mcast)
    for i in range(3): self.pci.write(0, bh.TLB_CONFIG + 12 * tlb + 4 * i, (cfg >> 32 * i & 0xFFFFFFFF).to_bytes(4, "little"))
    return tlb * bh.TLB_WINDOW + addr - base

  def window(self, x, y, addr, start=None):  # the shared TLB, moved only when an access needs another block
    base = addr & ~(bh.TLB_WINDOW - 1)
    if self.target != (x, y, base, start): self.map(x, y, base, start, self.tlb)
    self.target = (x, y, base, start)
    return self.tlb * bh.TLB_WINDOW + addr - base

  def chunks(self, x, y, addr, n, start=None):  # (BAR0 offset, size) pieces of [addr, addr+n), split where the window moves
    while n:
      size = min(n, bh.TLB_WINDOW - addr % bh.TLB_WINDOW)
      yield self.window(x, y, addr, start), size
      addr, n = addr + size, n - size

  def read(self, x, y, addr, n): return b"".join(self.pci.read(0, off, size) for off, size in self.chunks(x, y, addr, n))
  def write(self, x, y, addr, data, start=None):
    """Write data at NoC address (x, y, addr); with start, to every Tensix in the rectangle from start to (x, y)."""
    done = 0
    for off, size in self.chunks(x, y, addr, len(data), start):
      self.pci.write(0, off, data[done:done + size])
      done += size
  def read32(self, x, y, addr): return int.from_bytes(self.read(x, y, addr, 4), "little")
  def write32(self, x, y, addr, val, start=None): self.write(x, y, addr, val.to_bytes(4, "little"), start)

  def wait(self, cond, what, timeout=2.0):
    deadline = time.monotonic() + timeout
    while not cond():
      if time.monotonic() > deadline: raise TimeoutError(f"chip did not {what} within {timeout}s")
      self.pci.tick()

  def arc_msg(self, header, *payload, timeout=1.0):
    """Send ARC one message (the low byte of header is its code) and return the 8-word response."""
    (x, y), entry = bh.ARC, bh.ARC_QUEUE_ENTRY
    qcb = self.read32(x, y, bh.ARC_QUEUES)
    base, n = self.read32(x, y, qcb), self.read32(x, y, qcb + 4) & 0xFF
    q = base + bh.APP_QUEUE * (entry + 2 * n * entry)
    req_w, resp_r, req_r, resp_w = (q + at for at in bh.ARC_QUEUE_PTRS)
    wptr, rptr = self.read32(x, y, req_w), self.read32(x, y, resp_r)
    assert (wptr - self.read32(x, y, req_r)) % (2 * n) < n, "ARC message queue full"
    self.write(x, y, q + entry + wptr % n * entry, struct.pack("<8I", header, *payload, *[0] * (7 - len(payload))))
    self.write32(x, y, req_w, (wptr + 1) % (2 * n))
    self.write32(x, y, *bh.ARC_IRQ)
    self.wait(lambda: self.read32(x, y, resp_w) != rptr, f"answer ARC message {header & 0xFF:#x}", timeout)
    resp = struct.unpack("<8I", self.read(x, y, q + entry * (1 + n + rptr % n), entry))
    self.write32(x, y, resp_r, (rptr + 1) % (2 * n))
    assert resp[0] & 0xFF < 240, f"ARC message {header & 0xFF:#x} failed: {resp[0]:#x}"
    return resp

  def telemetry(self):
    """{tag: value} from the table ARC keeps up to date."""
    x, y = bh.ARC
    table, values = (self.read32(x, y, a) for a in bh.ARC_TELEMETRY)
    n = self.read32(x, y, table + 4)
    tags = struct.unpack(f"<{n}I", self.read(x, y, table + 8, 4 * n))
    vals = struct.unpack(f"<{n}I", self.read(x, y, values, 4 * n))
    return {t & 0xFFFF: vals[t >> 16] for t in tags}

  def halt(self, x, y, start=None): self.write32(x, y, bh.SOFT_RESET, bh.ALL_CORES, start)  # hold all five cores in reset

if __name__ == "__main__":  # what a card says about itself: the first thing to run on one (or on ttsim, with TTSIM set);
  import os, sys            # with `reset`, after resetting it (pci.reset), for a card whose NoC or firmware hangs
  from ferrite import pci
  if sys.argv[1:] == ["reset"]:
    assert "TTSIM" not in os.environ, "ttsim has no reset: a new process starts it afresh"
    pci.reset(bdf := pci.find_card())
    dev = pci.Card(bdf)
  else: dev = pci.open_device()
  chip = Chip(dev, 300 if sys.argv[1:] == ["reset"] else 2)
  t = chip.telemetry()
  banks = [b for b in range(len(bh.DRAM)) if t[bh.TAG_GDDR_ENABLED] >> b & 1]
  print(f"{getattr(dev, 'sys', 'ttsim')}: ARC firmware up, Tensix powered; AI clock {t.get(bh.TAG_AICLK, 0)} MHz; "
        f"{len(chip.tensix)} Tensix in {len(chip.tensix) // len(bh.TENSIX_Y)} columns; DRAM banks {banks} trained")
  print("telemetry:", " ".join(f"{tag}={value:#x}" for tag, value in sorted(t.items())))
