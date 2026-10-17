# Notes: talking to a Blackhole

What the milestones so far learned about the chip, where each fact comes from, and what is still unknown.
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

trickle uses one 2 MiB window (TLB 0) with strict ordering, and moves it whenever an access crosses into another 2 MiB block. Unicast uses dynamic VC. Multicast (`Chip.write(..., start=...)`) sets x_start/y_start, mcast, and static VC class 2, as umd does. Slow on purpose: simple first. Faster later: a window per target, posted ordering for bulk writes, BAR4 windows for DRAM.

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
| ARC messages `0xA0` (ASIC_STATE0, then reads `0x80030410` bit 1 for a power-cable fault), `0xC1` (watchdog, 10 s), `0xC7` (firmware log buffer) | not done; `Chip.arc_msg` can send them |
| Power policy: with no fd open, ARC message `0x21` with all flags 0 (minimum AICLK, GDDR PHY down, **Tensix clock-gated**); opening an fd sends `0x7FFE0F21` (everything on); remove sends `0xA3` | `Chip.__init__` sends `0x21` with all 15 flags valid and set (`0x7FFF0F21`: max AI clock, GDDR PHY awake, Tensix and L2CPU on) through `Chip.arc_msg` (tested on sim). Never powers down again. `0x52` (AICLK go busy), which umd also sends, is not sent |
| Pins host memory: IOVA from the IOMMU, or physically contiguous pages without one; programs an outbound iATU region so the device sees it at `0x1000_0000_0000_0000 + offset` | `Card.alloc`: a locked 2 MiB hugepage, bus address from `/proc/self/pagemap` (IOMMU off or passthrough). The device uses the bus address directly: kmd's own comment says an IOVA below 2^58 is a valid NoC address, so no iATU. With an IOMMU on, VFIO type1 `VFIO_IOMMU_MAP_DMA` would supply the IOVA; not written |
| One MSI, only for the firmware log | none needed |
| Reset: set `PCI_COMMAND` bit 6 as a marker, config writes `0x934 = 1` then `0x930 = 0x11` (or ARC `0x56` arg 3), wait >= 2 s, wait for the device to reappear and firmware to clear bit 6, restore config space and MPS (via DBI at NoC `(x,0)` `0xF800_0000_0000_0078`), wait for ARC again | not done; all of it is reachable from userspace through sysfs `config` |
| Telemetry table: `*0x80030434` (tags) and `*0x80030430` (values), each tag word is `id \| index << 16` | `Chip.telemetry`. `Chip.__init__` takes the Tensix column count from tag 34 (enabled columns; good ones come first in translated x) and waits up to 60 s for tag 22 to show every bank in tag 36 trained |
| "NoC cleanup": puts cores back in reset when a process dies | not done: a dead process leaves its firmware polling, which is harmless; the next `Runtime` resets the cores it uses |

ARC messages (kmd `msgqueue.c`, umd `blackhole_arc_message_queue.cpp`): `*0x8003042C` (SCRATCH_RAM[11]) points at a control block whose word 0 is the base of 4 queues and word 1 is `entries | queues << 8`. Queue i (0 = kmd, 3 = applications) is a 32-byte header (`+0` request wptr, `+4` response rptr, `+0x10` request rptr, `+0x14` response wptr; pointers wrap at 2N) followed by N request slots and N response slots of 32 bytes (`u32 header, u32 payload[7]`). Write the slot, bump the request wptr, poke the ARC (kmd: write 0 to `0x800B0000`; umd: write `0x10000` to `0x80030100`), poll the response wptr for up to 1 s. Status is the low byte of the response header, below 240 means OK. trickle uses queue 3 and umd's poke (ARC_MISC_CNTL IRQ0); ttsim models both pokes and answers kmd's and umd's messages, but kills the process on an unknown code, and returns all-zero responses (so `0x57` "get harvesting" is not usable there; telemetry is).

## The launch loop and the command queue

`trickle/runtime.py`. Built and tested on ttsim.

**Cores.** `Runtime(chip, start, end)` takes a rectangle of Tensix in translated coordinates. Its last core runs the **dispatcher**; the rest are **workers**. A rectangle of good columns in translated space covers exactly those cores: harvested and non-Tensix columns in between opt out of multicast.

**L1 layout** (ours). Workers: firmware at `0x0` (BRISC's reset PC), mailbox at `0x1000` (`[go, report x|y, report address lo, hi, done]`), arguments at `0x1100`, one kernel at `0x2000`, kernel data from `0x10000` (`L1_DATA`). Dispatcher: firmware at `0x0`, config at `0x1000` (`[doorbell, ring lo, hi, ring size, status lo, hi, workers]`), worker coordinates at `0x1040`, outbox at `0x1400`, one 16-byte done slot per worker at `0x2000`, command staging from `0x10000`.

**Boot** (host, once): multicast reset and worker firmware over the rectangle, a unicast mailbox per worker pointing its done report at its slot in the dispatcher, the dispatcher firmware and config over the top of the dispatcher's tile, a TLB of its own aimed at the doorbell, then multicast release of every BRISC.

**A launch** (host): build one command (a 16-byte header `[length, kernel bytes, argument bytes per worker, 0]`, the kernel, each worker's arguments, 64-byte aligned), copy it into the ring in host memory, and write the new write pointer to the dispatcher's doorbell. That doorbell is the launch's only PCIe access (`test_queue` counts). Commands never straddle the ring's end: a header of length 0 marks the tail unused. Pointers are free-running 32-bit byte counts, so the ring size is a power of two. Before writing, the host waits (with a timeout) for the space the dispatcher has freed.

**The dispatcher** (BRISC firmware, ~1.2 KB): poll the doorbell; read the command's header and then its body from host memory with NoC reads to the PCIe tile (19, 24); write the kernel and each worker's arguments to the worker, wait for every acknowledgement; write the launch number into each worker's `go`; poll the done slots until every worker reports; write `[read pointer, launches done]` into host memory. One launch at a time: the next kernel never overwrites one that is still running.

**A worker** (BRISC firmware, 300 bytes): poll `go` with a `fence` in the loop; invalidate the instruction cache (Tensix config 185), then `fence`, read config back, `fence`, so the invalidate has landed before the jump; call `kmain(args)` at `0x2000`; `noc_wait` for the kernel's own transfers; send its done to its slot at the dispatcher.

**Waiting** (`Runtime.wait`) polls the status words in host memory, never the chip. On timeout it reads the dispatcher's current launch and done slots and names the workers that have not finished.

**Device API**, prepended to every program (`DEVICE`): `noc_me()` (own translated coordinates, `NOC_ID_LOGICAL`), `noc_read`, `noc_write`, `noc_wait`, `l1_settle`. All requests go through NIU 0, request initiator 0, transaction ID 0. They are acknowledged (`RESP_MARKED`), so `NIU_MST_REQS_OUTSTANDING_ID(0)` counts them, on a dynamic VC. Rules (ISA `NoC/MemoryMap.md`, `NoC/Counters.md`, enforced by ttsim):

- At most 16 KiB per request (`noc_read`/`noc_write` split; ttsim does not model hardware splitting).
- Source and destination agree in their low 4 bits for L1, low 6 bits when DRAM is involved.
- Reads must be acknowledged. Inline writes to L1 are broken on Blackhole; we never use inline writes.
- Wait for `NOC_CMD_CTRL` bit 0 to clear before touching an initiator; read `NOC_CMD_CTRL` back before reading counters.
- Before the NIU reads data a core just stored: `fence`, load from L1, `fence` (`l1_settle`). After data lands: `fence` before reading it (the L0 data cache). Every `fence` is also a compiler barrier (`"memory"` clobber), or the compiler can cache L1 across it.

**Not built yet:**

1. **Overlap.** The dispatcher waits for one launch to finish before loading the next. Double-buffered kernel slots would let it load launch n+1 while n runs.
2. **Multicast from the dispatcher.** It writes the kernel to each worker in turn. A NoC multicast (path reserve, static VC class 2, `BRCST_SRC_INCLUDE` off so it skips its own tile) would send it once. Acknowledged multicast leaves `REQS_OUTSTANDING_ID` off by the destination count, so it needs `NIU_MST_WR_ACK_RECEIVED` counting instead.
3. **Commands other than "launch":** host-to-DRAM copies, events, waiting on another queue.
4. **The other four cores.** TRISC0-2 drive the Tensix coprocessor, NCRISC is a second data mover. They start from their reset PCs (`0x6000`/`0xA000`/`0xE000`/`0x12000`, overridable); firmware for them, and a BRISC-to-TRISC handoff, comes with the compiler.
5. **Hang reports from silicon.** The host should also read each stuck core's PC snapshot (`0xFFB1_3138`...); ttsim does not model those registers.
6. **Ordering of the host's ring writes before the doorbell** relies on x86 ordering cached stores before an uncached MMIO write. Fine on x86, and needs a barrier anywhere else.

## The Tensix coprocessor, and the first compiler

`trickle/compiler.py`, encodings and register numbers in `trickle/bh.py`. Tested on ttsim: `test_tile_math`, `test_tensor_math`, `test_matmul_exact`, `test_matmul_random`, `test_stream_on_one_worker`, `test_pack_waits_for_math`, `test_fusion`, `test_fusion_waits`, `test_matmul_fusion`.

**Who drives it.** Three Tensix threads, each an in-order instruction FIFO; any thread can issue to any unit (unpackers, FPU, packer, ...), the threads only differ in their per-thread state (ADCs, RWCs, GPRs, ThreadConfig). BRISC can push into all three: a store of the instruction word to `0xFFE4_0000` (thread 0), `0xFFE5_0000`, `0xFFE6_0000`. Its pushes skip the MOP expander (MOP from BRISC is an error; REPLAY works), and a full FIFO stalls BRISC. TRISCs push only to their own thread. trickle: BRISC alone drives all three threads, TRISCs stay in reset (a BRISC push and a TRISC push in the same cycle drop the TRISC's). BRISC can read only thread 0's GPRs in ttsim (ISA: all three at `0xFFE0_0000 + 4*(64t + r)`), which decides who reports what.

**Instruction words.** `opcode << 24 | fields` (ISA `Diagrams/Src/Bits32.lua`; ttsim `data/bh/tensix_isa.json` agrees and names the fields). `.ttinsn` is the same word rotated left by 2, so it only covers opcodes below `0xC0`.

**Configuration.** Three kinds, all needed:

- *Backend config*: 32-bit words at `0xFFEF_0000`, two banks (thread chooses with `CFG_STATE_ID`), a RISC-V store hits bank 0. The ISA docs name the fields but defer every index, shift and width to tt-metal's `cfg_defines.h`; ours come from ttsim's `tensix_regs.json` and must be checked on silicon.
- *Thread config*: 16-bit registers per thread, written only by `SETC16` from that thread, so they are set in-band at the start of the kernel. Reset values are undocumented, so every one we depend on is written.
- *ADCs and RWCs*: per-thread address counters (`SETADC*`) for unpackers and packer, and register-window counters (`SETRWC`) for the FPU.

BRISC's config stores are not ordered with its instruction pushes (Auto TTSync covers TRISCs only), so the kernel writes config, then `fence`, reads back a config word, `fence`, and only then pushes.

**The pipeline** (`compiler._kernel`, `fused_kernel`). Each thread has one job, so unpack, math and pack work on different tiles at the same time. L1 holds an array of tile slots; the unpackers' Z counter (Z stride: one tile) and the packer's channel-1 Y counter (that Y stride: one tile) pick the slot, so moving between slots never touches config:

| Thread | Job | Per tile (or matmul step) |
|---|---|---|
| 0 | unpack, and report to BRISC | `SETADC` each unpacker's Z to its slot, `UNPACR` into SrcA and SrcB; in matmul, wait for the unpackers and BRISC's `SETGPR` 63 = steps unpacked |
| 1 | math | per result: set the address modes for its kind (elementwise or matmul; one kernel can mix them), `SEMWAIT` for a free Dst half, aim at it (`SETC16 MATH_Offset` = 0 or 64), zero it if the op accumulates (`ZEROACC` four 16-row blocks); the FPU program (one per matmul step); at the end: wait for the FPU, `SEMPOST` SEM_DST |
| 2 | pack | `SEMWAIT` SEM_DST, aim the packer at the Dst half (ADC ch0 Y) and the L1 slot (ADC ch1 Y), 16 `PACR`, wait for the packer, `SEMGET` SEM_DST, `SEMPOST` SEM_L1 (an intermediate) or SEM_PACKED (an output) |

Handoffs:

- **Unpack to math**: SrcA and SrcB have two banks each, and the hardware does the rest (`UNPACR` waits for a bank the FPU has released, the FPU for one the unpacker has filled).
- **Math to pack**: SEM_DST counts results in Dst waiting to be packed (max 2, one per half). Math waits while both halves are full, pack while none is.
- **Pack to unpack** (fused chains): an intermediate is packed into L1 and unpacked by a later op. Pack posts SEM_L1 for each; before an op that reads an intermediate, unpack takes one for every op packed since it last looked. Packs happen in op order, so that covers whichever earlier op it reads (a plain count would not: the op might read an older intermediate than the one just posted).
- **Pack to BRISC**: BRISC cannot read the semaphores, nor (on ttsim) thread 2's GPRs. So BRISC pushes a relay into thread 0 (`SEMWAIT` SEM_PACKED, `SEMGET`, `SETGPR` 62) and polls GPR 62.

**BRISC's loop**:

1. For each step, wait until the L1 slot's previous tiles (two steps back) are unpacked, then fetch the step's tiles from DRAM into that slot.
2. Push the unpack to thread 0, the math to thread 1, and after the last step of an output tile its pack to thread 2.
3. Then wait for the previous output tile to be packed and write it home, which frees its L1 slot before the next pack can reuse it.

Thread setup goes to all three threads; thread 0 initialises the semaphores, and BRISC waits for that before pushing anything to the others.

**Fusion** (`fused_kernel`, `Tensor` is lazy). `+`, `-`, `*` and `@` build a tree; `realize` (or `tolist`) compiles the whole tree into one kernel, so `(a @ b) * d - c` or `(a @ b) + (e @ f)` is one launch. A matmul's own operands are realized first: each of its output tiles needs a whole row and column of them. Equal subtrees become one op. Slots 0-3 stream matmul tiles; per output tile, a frame of slots holds the elementwise inputs, then one slot per op in post order; the last is the result. BRISC fetches the inputs into the frame and pushes, op by op: an elementwise op's unpack (from its operands' slots), math (into the next Dst half) and pack (into its own slot); or a matmul's K steps (each: wait until the stream slots from two steps back are unpacked, fetch A(i, k) and B(k, j), unpack, MVMULs) and its pack. Intermediates live in L1 only, and only the result goes to DRAM. Two frames alternate, so tile n + 1 is fetched while tile n computes, and BRISC writes tile n - 1 back. Every intermediate is rounded to bf16 on its way to L1, exactly as it would be on its way to DRAM, so a fused chain is bit for bit the same as one launch per op (`test_fusion` checks). A single op is just a one-op chain. Limits: binary ops on equal shapes, matmuls only over realized tensors (no fused prologue), at most 15 ops (SEM_L1 is 4 bits).

**Replay buffers and address modes keep BRISC's share small.** BRISC pushes every Tensix instruction, so words per tile are BRISC's time per tile. Each thread has a 32-entry replay buffer (`REPLAY` with `load` records the next N instructions; without it, plays them back; it works from BRISC because it sits after the MOP expander). Each kernel's setup records the runs that repeat: one 16-`MVMUL` fidelity phase and one 9-word `ELWMUL` phase in the math thread, 15 `PACR`s in the pack thread. The last instruction of a sequence that also has to release the Src banks is pushed by itself with `clear_dvalid`. Elementwise ops use address modes 0-1 and matmul 4-7, all set once at setup, so a kernel can mix them with no per-op rewrites. Blackhole's `addr_mode` field is 3 bits (bits 16:14 in ttsim's `tensix_isa.json`; the ISA pages are Wormhole's, with 2 bits plus an `ADDR_MOD_SET_Base` that Blackhole dropped). This needs confirming on silicon.

| words BRISC pushes | before | now |
|---|---|---|
| add or sub tile | 48 | 26 |
| mul tile | 80 | 27 |
| matmul step | 71 | 12 |
| matmul output tile, on top of its steps | 39 | 17 |

What is left is per-tile control (slots, Dst half, semaphores). `test_replay_keeps_pushes_small` holds these numbers. A matmul test multiplies by a diagonal of full-mantissa values, so every output is one product that is exact only if all four fidelity phases ran; it catches a replay one instruction short.

**Push order matters on silicon.** BRISC's path into a thread is a few entries deep and a full FIFO stalls BRISC. So BRISC must never queue work a thread will block on, unless what unblocks it has already been pushed: unpack, then math, then pack, op by op. `fused_kernel(order=...)` breaks that rule on purpose for tests, which only ttsim (512-entry FIFOs) survives: `pack_first` queues each pack before its math, `thread` queues a whole tile for thread 0, then 1, then 2.

**What ttsim can and cannot check here.** ttsim runs each Tensix instruction the moment it can, in zero time, and ignores the busy conditions of `STALLWAIT`. In the normal push order every handoff has already happened by the time its wait runs, so a missing wait would pass. Hence the test-only orders: without the pack's `SEMWAIT`, `test_pack_waits_for_math` dies on `SEMGET` of an empty semaphore; without the SEM_L1 wait or the math's Dst-half wait, `test_fusion_waits` fails. Speed-up from the overlap cannot be measured on ttsim at all; that waits for silicon.

**Tile layout** (ours; the ISA has no notion of a tile). A 32x32 tile is four 16x16 faces in row-major order (top-left, top-right, bottom-left, bottom-right), each face row-major: element (r, c) is at `(r//16*2 + c//16)*256 + r%16*16 + c%16`. An unpacker copies the 1024 datums linearly into Src rows, so face f is Src rows 16f..16f+15, which is what `MVMUL` wants; Dst and the packer keep the same order. Elementwise ops do not care. Tensors are tiles in row-major tile order, tile i in DRAM bank i % 8 at `base + i//8 * 2048`.

**Matmul, one pair of tiles** (`compiler.matmul`). `MVMUL` does `Dst[8x16] += SrcB[8x16] @ SrcA[16x16]`: SrcB rows from its row counter (a multiple of 8), SrcA rows from its counter (a multiple of 16 in ttsim), Dst rows from the instruction's `dst` field plus the Dst counter. So the left tile goes to SrcB (unpacker 1) and the right tile to SrcA (unpacker 0). Output face (i, j) = sum over k of left face (i, k) @ right face (k, j): 8 face products of 2 `MVMUL`s each, 16 per fidelity phase, 4 phases (SrcA gives 5 mantissa bits a phase, SrcB 7). The `dst` field carries the output rows; the Src counters walk in the order i, k, h (half), j, which needs only four address modes (slots 4-7): SrcA +16 rows (next j), SrcA -16 / SrcB +8 (next h), SrcA +16 / SrcB +8 (next k or i), and the same plus fidelity +1 at the end of a phase. Counters wrap at 64, so that last step lands back on row 0. The last `MVMUL` of the last phase gives both banks back to the unpackers.

**Matmul, whole tensors.** A matmul is an op in a fused kernel (alone, it is a one-op kernel). Each worker owns some output tiles; each is K steps through the pipeline: the left (i, k) and right (k, j) tiles, 64 `MVMUL`s into the same Dst half, which accumulates over k in fp32; then one pack. Error on random bf16 inputs is under 2^-8 of sum |a*b| (1.4e-3 measured on 64x96 @ 96x128).

**Precision.** Dst is kept in fp32 (`Fp32_enabled`); the packer rounds it to bf16 (`(bits + 0x8000) >> 16`, the only fp32-Dst-to-bf16 mode ttsim allows is `Read_32b_data` with In = bf16). With 4 fidelity phases a bf16 x bf16 product is exact before that rounding (bit-exact on 8192 random pairs). `ELWADD`/`ELWSUB` align the two 11-bit mantissas with round-half-up first, so about 0.4% of random sums differ from the fp32 sum by a last-place bf16 step. fp32 tensors would lose 13 mantissa bits in SrcA/SrcB (TF32); full fp32 needs unpack-to-Dst and the SFPU.

**Where the docs and ttsim disagree.** `UNPACR` bit 7 is `MultiContextMode` in the ISA and `ovrd_thread_id` in ttsim; both say it must be 1. Bit 0 is "write 0" in the ISA, `last` (must be 1) in ttsim; we follow ttsim. The Blackhole README says "4x packer", the PACR page says one packer with four Dst read ports. `SETDVALID` is unsupported on Blackhole (use `UNPACR_NOP`).

**What ttsim cannot check here.** It executes each Tensix instruction atomically and ignores most `STALLWAIT` conditions (FPU, packer, unpacker busy), so a missing wait between units passes on ttsim and fails on silicon. The `STALLWAIT`s here are what the ISA asks for.

**Next for the compiler:**

1. BRISC still pushes ~12-27 words a tile. TRISC firmware driving each thread locally (with MOP loops, which BRISC cannot use) would leave BRISC only the data movement and a per-tile signal.
2. Deeper pipelines: more L1 slots, so DRAM latency hides behind more than one step.
3. Matmul reuse: each step refetches both tiles from DRAM. Keep a row of left tiles in L1 across j, and fetch the next pair while the FPU runs (two L1 slots, two Src banks).
4. Fusion beyond this: a matmul epilogue could stay in Dst (`(a @ b) + c` by unpacking c and adding into the same half, no pack/unpack of the product), prologues (`(a + b) @ c`, per K step), unary ops, broadcasts. Keeping intermediates in fp32 (an fp32 pack, or moving Dst back into SrcA/SrcB) instead of rounding each to bf16.
5. The SFPU, for fp32, exp, recip, and the rest.

## Still unknown or untested

- The real-card backend (`pci.Card`) has never run: the p150 has not arrived. It assumes root, the IOMMU off or in passthrough, and a free 2 MiB hugepage (`echo 1 > /proc/sys/vm/nr_hugepages`).
- Whether the all-on power message is enough to wake Tensix on silicon after kmd clock-gated it, or whether `0x52` (AICLK go busy) is needed too.
- Whether the host-to-PCIe-tile loopback in `test_dma` (BAR read that turns into a device read of host memory) behaves on silicon as it does on ttsim. Device-initiated writes from a core's NIU are the path that matters and are untested.
- ISA does not document BAR2 (iATU, DMA engine), the DRAM tile's internal address map, ARC's message protocol, or the power-on value of soft reset. ttsim models them; real silicon may differ.
- ttsim does not model `mhartid`, `fence.i`, the PC snapshot registers, NoC request splitting, or most of BAR2, and it completes every NoC request instantly, so the outstanding-request counters never move there. Instruction cache invalidation is a no-op in ttsim, so `test_relaunch_without_reset` cannot catch a stale-instruction bug.
