# Notes: talking to a Blackhole

What milestone 1 learned about the chip, where each fact comes from, and what is still unknown.
Sources: **ISA** = tt-isa-documentation/BlackholeA0, **sim** = ttsim `src/`, **kmd** = tt-kmd, **umd** = tt-umd.
Facts the code uses live in `trickle/bh.py`.

## Address map

### PCIe function

Vendor 0x1E52, device 0xB140 (sim `pcie.cpp`, kmd). Three 64-bit prefetchable BARs (ISA `PCIExpressTile/README.md`):

| BAR | Size | Contents |
|---|---|---|
| BAR0 | 512 MiB | `0x0000_0000-0x193F_FFFF`: 202 TLB windows of 2 MiB (TLBs 0-201). TLBs 0-31 can do strided multicast; TLB 201 is reserved for the kernel driver. |
| | | `0x1FC0_0000`: TLB config, `u32[3]` per TLB for TLBs 0-209, then `u32 strided[32]` at `0x1FC0_09D8` |
| | | `0x1FD0_4000`, `0x1FD1_4000`: the PCIe tile's NoC0 and NoC1 NIU registers |
| | | `0x1FE0_0000-0x1FFF_FFFF`: ARC on PCIe 1 only; reserved on PCIe 0, which is the one a p150 uses |
| BAR2 | 1 MiB | PCIe controller (DBI): eDMA read channel 0 at `0x100-0x1A8`, iATU at `0x1000` (outbound region r at `0x1000 + 0x200*r`, inbound interleaved between). Not in ISA; offsets from kmd `blackhole.c`, umd `blackhole_dma_transfer.cpp`, sim. |
| BAR4 | 32 GiB | 8 TLB windows of 4 GiB (TLBs 202-209) |

### NoC

17 x 12 grid. Raw NoC0 coordinates (ISA `Diagrams/Src/NoC.lua`, sim `tile.cpp`): DRAM at x=0 (banks 0-3) and x=9 (banks 4-7), ARC (8,0), PCIe 0 (2,0) and PCIe 1 (11,0), Ethernet on y=1, Tensix at x=1..7,10..16, y=2..11. NoC1 numbers the same grid from the other corner: (16-x, 11-y).

Firmware turns on **coordinate translation** at boot (`NIU_CFG_0` bit 14) and programs every tile's translation tables, so harvested Tensix columns move to the highest x and software never sees which physical columns are dead (ISA `NoC/Coordinates.md`). trickle uses translated coordinates everywhere:

| Translated (x, y) | What |
|---|---|
| x 1..7, 10..14, y 2..11 | Tensix, 120 on a p150 (x 15, 16 are its 2 harvested columns: firmware 19.5+ harvests 2 on every p150, umd `CHANGELOG`, `cluster_descriptor.cpp`) |
| x 17..18, y 12..23 | DRAM: bank b at (17 + b//4, 12 + 3*(b%4) + port), port 0..2, all three ports reach the same 4 GiB |
| (19, 24) | the PCIe tile connected to the host |
| x 20..31, y 25 | Ethernet |
| y 0 and 1 | not translated in x: ARC is (8, 0) |

A NoC address is (x, y, 64-bit address inside that tile).

### Inside a Tensix tile

As the tile's own cores see it; the last column says whether the NoC reaches the same address (ISA `TensixTile/BabyRISCV/README.md`):

| Address | What | Over NoC |
|---|---|---|
| `0x0000_0000-0x0017_FFFF` | L1, 1.5 MiB. Code runs only from here. | yes |
| `0xFFB0_0000` | the core's own data RAM: 8 KiB for BRISC and NCRISC, 4 KiB for each TRISC. Stack goes here. | no (yes via slow aliases at `0xFFB1_4000`...) |
| `0xFFB1_2000-0xFFB1_2FFF` | tile control and debug registers, including soft reset and reset PCs | yes |
| `0xFFB1_3138-0xFFB1_314B` | PC snapshot of each core | yes |
| `0xFFB2_0000`, `0xFFB3_0000` | NoC0 and NoC1 NIU registers (the tile issues NoC requests here) | yes |
| `0xFFE0_0000`... | Tensix coprocessor: GPRs, instruction push, PC buffers, mailboxes, config | no |

MMIO registers take exactly one 4-byte access, from either side.

### DRAM

Eight 4 GiB banks; the address inside a DRAM tile is the byte offset in its bank, starting at 0 (umd soc descriptor, kmd `test/tlbs.cpp`; sim models 4080 MiB per bank). ISA does not document the DRAM tile's address layout. On silicon, GDDR training must be done before first use: telemetry tag 22 has 2 bits per bank (trained, error); umd waits up to 300 s and retrains with ARC message `0xB6`.

### Host memory, from the device

The device reaches host memory by sending NoC requests to the PCIe tile at (19, 24). Its address space (ISA `PCIExpressTile/README.md`):

| NoC address | Goes to |
|---|---|
| below `0x0400_0000_0000_0000` (2^58) | the host, as a bus address, through the host IOMMU |
| bits 58/59 set | same, with relaxed ordering / no-snoop |
| `0x1000_0000_0000_0000` (bit 60) | through the outbound iATU (16 regions, programmed in BAR2) to the host |
| `0xF800_0000_0000_0000` | PCIe controller DBI |

So a host buffer at bus address B is NoC address (19, 24, B), with no iATU setup: firmware sets up the windows, and kmd's DMA-mask comment says the same (an IOVA below 2^58 is used directly as a NoC address, kmd `enumerate.c`). kmd and umd still route pinned memory through the bit-60 iATU window, which is only needed to make scattered memory look contiguous. test_dma checks the direct path with a TLB pointed at the PCIe tile.

## TLB setup

A TLB window turns a range of BAR0 (or BAR4) into a range of one tile's NoC address space. Each TLB has a 96-bit config at `BAR0 + 0x1FC0_0000 + 12*i`, written as three 32-bit uncached writes. 2 MiB TLBs (ISA `PCIExpressTile/HostToDeviceTLBs.md`, matches sim and umd):

| Bits | Field |
|---|---|
| 0-42 | NoC address >> 21 |
| 43-48, 49-54 | x_end, y_end (the target, or the far corner of a multicast rectangle) |
| 55-60, 61-66 | x_start, y_start (multicast only) |
| 67-68 | noc: NoC0 or NoC1 |
| 69 | multicast |
| 70-71 | ordering: 0 relaxed, 1 strict AXI, 2 posted writes, 3 counted writes |
| 72 | linked (never safe to set) |
| 73, 75, 76-77 | static VC enable, buddy, VC class (kmd names 74 stream_header and 75-77 static_vc; same bits) |

4 GiB TLBs: bits 0-31 are address >> 32, then x_end at 32, y_end 38, x_start 44, y_start 50, noc 56-57, mcast 58, ordering 59-60, linked 61, static VC 62.

Write the three dwords with 32-bit stores (odd TLBs are not 8-byte aligned), and the address must be window aligned (kmd, umd). umd turns on static VC for windows that only write or only read, and leaves it off for a window that does both: a read+write window on a static VC "crashes the host" (umd `architecture_tlbs.cpp`).

trickle uses one 2 MiB window (TLB 0), unicast, strict ordering, dynamic VC, and moves it whenever an access crosses into another 2 MiB block. That is slow on purpose: simple first. Faster later: a window per target, posted ordering for bulk writes, BAR4 windows for DRAM, multicast for loading every core.

## Reset sequence

Per Tensix tile, `RISCV_DEBUG_REG_SOFT_RESET_0` at `0xFFB1_21B0`, reachable over the NoC (ISA `TensixTile/SoftReset.md`). A set bit holds a core in reset:

| Bit | Core |
|---|---|
| 11 | BRISC |
| 12, 13, 14 | TRISC0, TRISC1, TRISC2 |
| 18 | NCRISC |

Other bits reset parts of the Tensix coprocessor. There is no atomic set/clear; it is read-modify-write. Power-on value is `0x47800`, all five cores held (sim; ISA does not say). umd releases with `0x80047000`: bit 31 is "staggered start", which spreads core start-up across the chip (power, not function; sim ignores it).

To run code on a core: write `0x47800` (hold everything), write the program into L1, write `0x47000` (release BRISC only), poll L1 for its result with a timeout, write `0x47800` again. That is `Chip.run`/`Chip.halt`.

Leaving reset: the instruction cache is invalidated, GPRs are zeroed, the PC is set (below), and the core's data RAM zeroes itself over up to 2048 cycles unless `RISCV_DEBUG_REG_DISABLE_RESET` (`0xFFB1_2224`) says not to. So: load code while the core is held, and do not preload data RAM over the NoC without setting that bit.

## What the cores run out of reset

Nothing of ours: there is no boot ROM in a Tensix tile. A released core fetches from L1 at its reset PC (ISA `SoftReset.md`):

| Core | Reset PC |
|---|---|
| BRISC | `0x0` (fixed) |
| TRISC0/1/2 | `0x6000`/`0xA000`/`0xE000`, or `RISCV_DEBUG_REG_TRISCn_RESET_PC` (`0xFFB1_2228/222C/2230`) if bit n of `0xFFB1_2234` is set |
| NCRISC | `0x12000`, or `0xFFB1_2238` if bit 0 of `0xFFB1_223C` is set |

The baby cores are rv32im with Zicsr, Zaamo (no lr/sc), Zba, Zbb, and a partial F (no fdiv/fsqrt). No C extension: its encoding space holds `.ttinsn`. trickle compiles with plain `clang --target=riscv32-unknown-elf -march=rv32im` and links a flat binary at 0 with lld. Gotchas (ISA `BabyRISCV/`):

- `mhartid` reads 0 on every core per ISA; ttsim does not implement it and kills the process. Tell cores apart by entry point.
- `fence.i` does not flush the instruction cache (ttsim rejects it). New code needs a reset release, or a write to `RISCV_IC_INVALIDATE` in Tensix config from a core.
- The L0 data cache (4 x 16 B) is not coherent with NoC or host writes: poll L1 with a `fence` or an atomic in the loop.
- Stores to the same region (e.g. L1) are processed in order, so "write the result, then write the done flag" is safe. Across regions a `fence` is not enough: store, load it back, use the value, then signal.
- Unsupported instructions are undefined behavior, not traps.

Elsewhere on the chip, firmware is already running when we arrive: ARC runs the board management firmware and set up NoC translation and the PCIe windows; Ethernet core E0 runs Tenstorrent's Ethernet firmware.

## What tt-kmd does that a userspace driver must do instead

What kmd does for a Blackhole at probe, open, reset and remove (kmd `enumerate.c`, `blackhole.c`, `chardev.c`, `memory.c`, `pcie.c`), and where trickle stands:

| kmd | trickle |
|---|---|
| `pci_enable_device`, 58-bit DMA mask, bus mastering | `Card`: unbind kmd, write sysfs `enable`, set memory + bus master in `PCI_COMMAND` |
| Max Read Request Size 4096 in PCIe Device Control | not done (throughput only) |
| Maps BAR0 TLB config + NIU, BAR2, BAR4; reserves TLB 201 for its own NoC accesses; hands other TLBs out per fd | `Card` mmaps sysfs `resource0/2/4`; with kmd gone every TLB is ours; trickle uses TLB 0 |
| Waits for ARC: `0x80030408` bit 0 (500 ms); umd waits for `& 7 == 5` (300 s after a reset) | `Chip.__init__` waits for `& 7 == 5`, 2 s (tested on sim) |
| ARC messages `0xA0` (ASIC_STATE0, then reads `0x80030410` bit 1 for a power-cable fault), `0xC1` (watchdog, 10 s), `0xC7` (firmware log buffer) | not done |
| Power policy: with no fd open, ARC message `0x21` with all flags 0 (minimum AICLK, GDDR PHY down, **Tensix clock-gated**); opening an fd sends `0x7FFE0F21` (everything on); remove sends `0xA3` | **not done: the first thing the p150 needs.** If kmd ever ran, the Tensix cores are likely clock-gated and `hello` will time out until we send `0x21` with TENSIX_ENABLE (bit 2 of the flags, header `0x21 \| validity<<8 \| flags<<16`, 14 u16 settings as payload) and probably `0x52` (AICLK go busy) |
| Pins host memory: IOVA from the IOMMU, or physically contiguous pages without one; programs an outbound iATU region so the device sees it at `0x1000_0000_0000_0000 + offset` | `Card.alloc`: a locked 2 MiB hugepage, bus address from `/proc/self/pagemap` (IOMMU off or passthrough). The device uses the bus address directly: kmd's own comment says an IOVA below 2^58 is a valid NoC address, so no iATU. With an IOMMU on, VFIO type1 `VFIO_IOMMU_MAP_DMA` would supply the IOVA; not written |
| One MSI, only for the firmware log | none needed |
| Reset: set `PCI_COMMAND` bit 6 as a marker, config writes `0x934 = 1` then `0x930 = 0x11` (or ARC `0x56` arg 3), wait >= 2 s, wait for the device to reappear and firmware to clear bit 6, restore config space and MPS (via DBI at NoC `(x,0)` `0xF800_0000_0000_0078`), wait for ARC again | not done; all of it is reachable from userspace through sysfs `config` |
| Telemetry table: `*0x80030434` (tags) and `*0x80030430` (values), each tag word is `id \| offset << 16` | not done; tag 22 GDDR training, tag 34 enabled Tensix columns (harvesting) |
| "NoC cleanup": puts cores back in reset when a process dies | `hello` halts its core in a `finally` |

ARC messages (kmd `msgqueue.c`, umd `blackhole_arc_message_queue.cpp`): `*0x8003042C` (SCRATCH_RAM[11]) points at a control block whose word 0 is the base of 4 queues and word 1 is `entries | queues << 8`. Queue i (0 = kmd, 3 = applications) is a 32-byte header (`+0` request wptr, `+4` response rptr, `+0x10` request rptr, `+0x14` response wptr; pointers wrap at 2N) followed by N request slots and N response slots of 32 bytes (`u32 header, u32 payload[7]`). Write the slot, bump the request wptr, poke the ARC (kmd: write 0 to `0x800B0000`; umd: write `0x10000` to `0x80030100`), poll the response wptr for up to 1 s. Status is the low byte of the response header, below 240 means OK. ttsim implements this queue, so it can be built and tested on the sim.

## What a minimal launch loop needs

`trickle.hello` is the degenerate launch loop: reset, load, release, poll, reset. A real one, for milestone 2:

1. **Resident firmware** on each core, loaded once with the reset sequence above: a loop that waits on a mailbox word in L1 (with a `fence` in the loop), jumps to the kernel address it names with an argument pointer, then reports done. Because `fence.i` does not work, BRISC firmware invalidates the instruction caches through Tensix config before jumping to newly loaded code, or the host reloads with a reset.
2. **Loading**: kernel binary and arguments written to L1 (multicast TLB to many tiles at once), or to DRAM and pulled by the cores over the NoC.
3. **Go**: one store to each core's mailbox.
4. **Done**: the core writes a completion word into a pinned host buffer through the PCIe tile (19, 24), so the host polls its own memory instead of doing ~1 us BAR reads. The flag must land after the data: same NoC, static VC, and the store, load, use pattern on the core side.
5. **Every wait times out**, and on timeout the host reads each core's PC snapshot (`0xFFB1_3138`...) so the error says where the chip is stuck.
6. **Queues** come after that: a command ring in host memory or DRAM that a dispatcher core reads, so the host does not touch each tile per launch.

## Still unknown or untested

- The real-card backend (`pci.Card`) has never run: the p150 has not arrived. It assumes root, the IOMMU off or in passthrough, and a free 2 MiB hugepage (`echo 1 > /proc/sys/vm/nr_hugepages`).
- The Tensix power state on a card that never saw kmd, versus one kmd left clock-gated. Expect to need the ARC power message above before `hello` works on silicon.
- Whether the host-to-PCIe-tile loopback in `test_dma` (BAR read that turns into a device read of host memory) behaves on silicon as it does on ttsim. Device-initiated writes from a core's NIU are the path that matters and are untested.
- ISA does not document BAR2 (iATU, DMA engine), the DRAM tile's internal address map, ARC's message protocol, or the power-on value of soft reset. ttsim models them; real silicon may differ.
- ttsim does not model `mhartid`, `fence.i`, or most of BAR2.
