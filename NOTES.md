# Notes: talking to a Blackhole

What the milestones so far learned about the chip, where each fact comes from, and what is still unknown.
Sources: **ISA** = tt-isa-documentation/BlackholeA0, **sim** = ttsim `src/`, **kmd** = tt-kmd, **umd** = tt-umd.
Facts the code uses live in `ferrite/bh.py`.

## Goals

Each has a finish line we can check. Ticked when a test shows it.

1. **A real model runs on the simulator.** Finish line: one GPT-2 transformer layer matches a CPU reference. **Done.**
   - [x] DRAM is freed when a tensor goes away (`test_dram_is_freed`)
   - [x] any tensor shape, not just multiples of 32 (`test_any_shape`)
   - [x] views, transpose, gelu (`test_views_transpose_gelu`)
   - [x] halfway: a small MNIST network (matmuls, bias, relu) matches (`test_mnist_mlp`)
   - [x] a GPT-2 layer (layer norm, attention, MLP) matches (`test_gpt2_layer`)
   - [x] beyond it: GPT-2 small's real weights generate text, token for token the same as numpy and Hugging Face (`test/gpt2.py`, `test_gpt2_model`)
2. **First light on silicon.** Finish line: `hello` on every core of a p150, then every test passing on the card. Before the card arrives:
   - [x] the tests run on a card when `TTSIM` is not set
   - [x] a bring-up checklist, step by step ("Bring-up on a p150")
3. **Fast.** Finish line: a matmul benchmark with a number, measured against the chip's peak and improved over time (tile reuse, deeper pipelines, the MOP expander).
   - [x] the benchmark: `python3 -m ferrite.bench [M K N]`, wall time on a card, simulated cycles on ttsim (a work count, not silicon speed), and DRAM reads against reading each input once (ttsim's DRAM is instant, silicon's is not). 512x768 @ 768x3072 on ttsim: 35k cycles, "93 of 166 TFLOPS" at four fidelity phases; 22k ("153") with A's row kept in L1, resident kernels and launches queued back to back; 17k with arguments and go multicast (past the "peak": ttsim does not time Tensix instructions, so on ttsim the benchmark now reports cycles only). DRAM reads: 14.9 times the inputs, 6.9 with blocks
   - [ ] the same on silicon

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

Firmware turns on **coordinate translation** at boot (`NIU_CFG_0` bit 14) and programs every tile's translation tables, so harvested Tensix columns move to the highest x and software never sees which physical columns are dead (ISA `NoC/Coordinates.md`). ferrite uses translated coordinates everywhere:

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

ferrite uses one 2 MiB window (TLB 0) with strict ordering, and moves it whenever an access crosses into another 2 MiB block. Unicast uses dynamic VC. Multicast (`Chip.write(..., start=...)`) sets x_start/y_start, mcast, and static VC class 2, as umd does. Slow on purpose: simple first. Faster later: a window per target, posted ordering for bulk writes, BAR4 windows for DRAM.

## Reset sequence

Per Tensix tile, `RISCV_DEBUG_REG_SOFT_RESET_0` at `0xFFB1_21B0`, reachable over the NoC (ISA `TensixTile/SoftReset.md`). A set bit holds a core in reset:

| Bit | Core |
|---|---|
| 11 | BRISC |
| 12, 13, 14 | TRISC0, TRISC1, TRISC2 |
| 18 | NCRISC |

Other bits reset parts of the Tensix coprocessor. There is no atomic set/clear; it is read-modify-write. Power-on value is `0x47800`, all five cores held (sim; ISA does not say). umd releases with `0x80047000`: bit 31 is "staggered start", which spreads core start-up across the chip (power, not function; sim ignores it).

To run code on a core: write `0x47800` (hold everything), write the program into L1, write `0x47000` (release BRISC only), poll L1 for its result with a timeout, write `0x47800` again. `Chip.halt` holds a rectangle; `Runtime` boots it and releases BRISC and the TRISCs.

Leaving reset: the instruction cache is invalidated, GPRs are zeroed, the PC is set (below), and the core's data RAM zeroes itself over up to 2048 cycles unless `RISCV_DEBUG_REG_DISABLE_RESET` (`0xFFB1_2224`) says not to. So: load code while the core is held, and do not preload data RAM over the NoC without setting that bit.

## What the cores run out of reset

Nothing of ours: there is no boot ROM in a Tensix tile. A released core fetches from L1 at its reset PC (ISA `SoftReset.md`):

| Core | Reset PC |
|---|---|
| BRISC | `0x0` (fixed) |
| TRISC0/1/2 | `0x6000`/`0xA000`/`0xE000`, or `RISCV_DEBUG_REG_TRISCn_RESET_PC` (`0xFFB1_2228/222C/2230`) if bit n of `0xFFB1_2234` is set |
| NCRISC | `0x12000`, or `0xFFB1_2238` if bit 0 of `0xFFB1_223C` is set |

The baby cores are rv32im with Zicsr, Zaamo (no lr/sc), Zba, Zbb, and a partial F (no fdiv/fsqrt). No C extension: its encoding space holds `.ttinsn`. ferrite compiles with plain `clang --target=riscv32-unknown-elf -march=rv32im` and links a flat binary at 0 with lld. Gotchas (ISA `BabyRISCV/`):

- `mhartid` reads 0 on every core per ISA; ttsim does not implement it and kills the process. Tell cores apart by entry point.
- `fence.i` does not flush the instruction cache (ttsim rejects it). New code needs a reset release, or a write to `RISCV_IC_INVALIDATE` in Tensix config from a core.
- The L0 data cache (4 x 16 B) is not coherent with NoC or host writes: poll L1 with a `fence` or an atomic in the loop.
- Stores to the same region (e.g. L1) are processed in order, so "write the result, then write the done flag" is safe. Across regions a `fence` is not enough: store, load it back, use the value, then signal.
- Unsupported instructions are undefined behavior, not traps.

Elsewhere on the chip, firmware is already running when we arrive: ARC runs the board management firmware and set up NoC translation and the PCIe windows; Ethernet core E0 runs Tenstorrent's Ethernet firmware.

## What tt-kmd does that a userspace driver must do instead

What kmd does for a Blackhole at probe, open, reset and remove (kmd `enumerate.c`, `blackhole.c`, `chardev.c`, `memory.c`, `pcie.c`), and where ferrite stands:

| kmd | ferrite |
|---|---|
| `pci_enable_device`, 58-bit DMA mask, bus mastering | `Card`: unbind kmd, write sysfs `enable`, set memory + bus master in `PCI_COMMAND` |
| Max Read Request Size 4096 in PCIe Device Control | not done (throughput only) |
| Maps BAR0 TLB config + NIU, BAR2, BAR4; reserves TLB 201 for its own NoC accesses; hands other TLBs out per fd | `Card` mmaps sysfs `resource0/2/4`; with kmd gone every TLB is ours; ferrite uses TLB 0 |
| Waits for ARC: `0x80030408` bit 0 (500 ms); umd waits for `& 7 == 5` (300 s after a reset) | `Chip.__init__` waits for `& 7 == 5`, 2 s (tested on sim) |
| ARC messages `0xA0` (ASIC_STATE0, then reads `0x80030410` bit 1 for a power-cable fault), `0xC1` (watchdog, 10 s), `0xC7` (firmware log buffer) | not done; `Chip.arc_msg` can send them |
| Power policy: with no fd open, ARC message `0x21` with all flags 0 (minimum AICLK, GDDR PHY down, **Tensix clock-gated**); opening an fd sends `0x7FFE0F21` (everything on); remove sends `0xA3` | `Chip.__init__` sends `0x21` with all 15 flags valid and set (`0x7FFF0F21`: max AI clock, GDDR PHY awake, Tensix and L2CPU on) through `Chip.arc_msg` (tested on sim). Never powers down again. `0x52` (AICLK go busy), which umd also sends, is not sent |
| Pins host memory: IOVA from the IOMMU, or physically contiguous pages without one; programs an outbound iATU region so the device sees it at `0x1000_0000_0000_0000 + offset` | `Card.alloc`: a locked 2 MiB hugepage, bus address from `/proc/self/pagemap` (IOMMU off or passthrough). The device uses the bus address directly: kmd's own comment says an IOVA below 2^58 is a valid NoC address, so no iATU. With an IOMMU on, VFIO type1 `VFIO_IOMMU_MAP_DMA` would supply the IOVA; not written |
| One MSI, only for the firmware log | none needed |
| Reset: set `PCI_COMMAND` bit 6 as a marker, config writes `0x934 = 1` then `0x930 = 0x11` (or ARC `0x56` arg 3), wait >= 2 s, wait for the device to reappear and firmware to clear bit 6, restore config space and MPS (via DBI at NoC `(x,0)` `0xF800_0000_0000_0078`), wait for ARC again | not done; all of it is reachable from userspace through sysfs `config` |
| Telemetry table: `*0x80030434` (tags) and `*0x80030430` (values), each tag word is `id \| index << 16` | `Chip.telemetry`. `Chip.__init__` takes the Tensix column count from tag 34 (enabled columns; good ones come first in translated x) and waits up to 60 s for tag 22 to show every bank in tag 36 trained |
| "NoC cleanup": puts cores back in reset when a process dies | not done: a dead process leaves its firmware polling, which is harmless; the next `Runtime` resets the cores it uses |

ARC messages (kmd `msgqueue.c`, umd `blackhole_arc_message_queue.cpp`): `*0x8003042C` (SCRATCH_RAM[11]) points at a control block whose word 0 is the base of 4 queues and word 1 is `entries | queues << 8`. Queue i (0 = kmd, 3 = applications) is a 32-byte header (`+0` request wptr, `+4` response rptr, `+0x10` request rptr, `+0x14` response wptr; pointers wrap at 2N) followed by N request slots and N response slots of 32 bytes (`u32 header, u32 payload[7]`). Write the slot, bump the request wptr, poke the ARC (kmd: write 0 to `0x800B0000`; umd: write `0x10000` to `0x80030100`), poll the response wptr for up to 1 s. Status is the low byte of the response header, below 240 means OK. ferrite uses queue 3 and umd's poke (ARC_MISC_CNTL IRQ0); ttsim models both pokes and answers kmd's and umd's messages, but kills the process on an unknown code, and returns all-zero responses (so `0x57` "get harvesting" is not usable there; telemetry is).

## The launch loop and the command queue

`ferrite/runtime.py`. Built and tested on ttsim.

**Cores.** `Runtime(chip, start, end)` takes a rectangle of Tensix in translated coordinates. Its last core runs the **dispatcher**; the rest are **workers**. A rectangle of good columns in translated space covers exactly those cores: harvested and non-Tensix columns in between opt out of multicast.

**L1 layout** (ours). Workers: firmware at `0x0`, starting with one jump per core (BRISC at `0x0`, its reset PC; TRISC i at `0x4 + 4i`, set as its reset PC), mailbox at `0x1000` (`[go, kernel, arguments, workers] [done, the dispatcher's x|y, scratch, index]`), the TRISCs' `[go, done 0, done 1, done 2]` at `0x1020`, `SYNC` at `0x1040` (64 bytes, zeroed before each launch, for the kernel's cores to talk through), two argument buffers at `0x1100` (launches alternate), the loaded kernels from `0x2000` (502 KiB; each a header of four entry offsets, then a program per core), a tile of zeros at `0x7F800` (written once at boot, read by kernels that add to zero or reduce: the compiler's slot 0), kernel data from `0x80000` (`L1_DATA`). Dispatcher: firmware at `0x0`, config at `0x1000` (`[doorbell, ring lo, hi, ring size, status lo, hi, workers' rectangle, workers]`), outbox at `0x1400`, the count of the workers' reports at `0x80000`, command staging from `0x80040` (above the kernels, since loading one writes the dispatcher's tile too).

**Boot** (host, once): multicast reset and worker firmware over the rectangle, a unicast mailbox per worker with its index and the dispatcher's coordinates, the dispatcher firmware and config over the top of the dispatcher's tile (its TRISC jumps park them), a TLB of its own aimed at the doorbell, the TRISC reset PCs and their override bits, then multicast release of every BRISC and TRISC.

**Loaded kernels** (`Runtime.load`). A kernel goes into every worker's L1 once, by one multicast write over the rectangle, and stays: launches name its address. Images run wherever they land: clang's medany model reaches code and data PC-relative, with linker relaxation and jump tables off, and the header holds entry offsets. When the 502 KiB are full, the next load waits for the queue and evicts them all (`evict`). A kernel's zeroed statics are zeroed again at each launch (`KENTRY` clears `.bss` before `kmain`), since nothing reloads it. Its TLB orders writes strictly, so a kernel has landed before the doorbell that launches it (to confirm on silicon). Before this, every launch copied the whole image (up to 15 KB) to every worker: a GPT-2 token is about 600 launches of 15 kernels.

**A launch** (host): build one command (a 16-byte header `[length, kernel address, argument bytes, workers]`, then the arguments, the same for every worker, 64-byte aligned), copy it into the ring in host memory, and write the new write pointer to the dispatcher's doorbell. That doorbell is the launch's only PCIe access (`test_queue` counts). Commands never straddle the ring's end: a header of length 0 marks the tail unused. Pointers are free-running 32-bit byte counts, so the ring size is a power of two. Before writing, the host waits (with a timeout) for the space the dispatcher has freed.

**The dispatcher** (BRISC firmware, ~1.2 KB): poll the doorbell; read the command's header and then its body from host memory with NoC reads to the PCIe tile (19, 24); multicast the arguments to every worker and wait for every acknowledgement; multicast the launch number, the addresses of the kernel and the arguments, and how many workers it is for into every mailbox; poll the count of reports until every worker has added to it; write `[read pointer, launches done]` into host memory. One launch runs at a time, but the next command is read and its arguments written (into the other buffer) while the last one runs, so it goes as soon as the last worker reports; the host hears of a finished launch after that go. Queued, an empty launch costs 190 simulated cycles, on one worker or on all 119 (5630 on 119 when arguments, go and done went worker by worker).

**Multicast from a core** (the dispatcher's `multicast`): a write with `NOC_CMD_BRCST_PACKET`, path reserve and a static VC of class 2 (ttsim insists on all three), the rectangle in `NOC_RET_ADDR_HI` as `end | start << 12`, and `NOC_CMD_BRCST_SRC_INCLUDE` off, since the dispatcher sits in the rectangle it multicasts to and its L1 holds other things at the mailbox's address. Each recipient acknowledges, so `NIU_MST_REQS_OUTSTANDING_ID` goes up once and down once per recipient (ISA `NoC/Counters.md`): the dispatcher counts `NIU_MST_WR_ACK_RECEIVED` up to the number of workers instead, then clears the outstanding counters (`0xFFFF` at `NIU + 0x60`; ttsim takes nothing else), which is safe because nothing else is in flight then.

**A worker** (firmware for all four cores, 640 bytes). BRISC: poll `go` with a `fence` in the loop; sit the launch out if its index is not below the launch's worker count; turn the arguments' first word, a number of units, into its own share `[first unit, how many]` (even, in order); zero `SYNC`; invalidate the instruction caches of BRISC and the TRISCs (Tensix config 185, bits 0-3), then `fence`, read config back, `fence`, so the invalidate has landed before anyone jumps; write the TRISCs' go; call its program; `noc_wait` for the kernel's own transfers; wait for the three TRISCs' done; note the launch as done in its own mailbox (the host reads it if a launch hangs) and add 1 to the dispatcher's count with a NoC atomic increment (`NOC_AT_LEN_BE` = `1 << 12 | 31 << 2`, a 32-bit add of `NOC_AT_DATA`; posted, so no counter moves at either end). A TRISC: poll its go, call its program (if the kernel has one for it), wait until its Tensix thread has run everything it pushed (`tensix_idle`, the Manual TTSync load), write its done.

**Launch cost.** The arguments and the go are one multicast each and the reports one counter, whatever the number of workers. A launch names how many units of work it has (`Runtime.launch(kernel, args, units)`), and goes to no more workers than that: a GPT-2 attention head's kernels have one or two tiles of work. When each worker got its own arguments and go, a launch cost 45 cycles per worker more, and a GPT-2 token was 627k simulated cycles; with multicast it is 371k, and a one-tile-row matmul (768 x 3072, a token's MLP) went from 24 to 76 "TFLOPS". Then the done slots: all 119 workers were done 240 cycles into a launch, and the dispatcher read their slots, one L1 load each, until cycle 870; with one count to read a token is 282k (after split K, 312k).

**Waiting** (`Runtime.wait`) polls the status words in host memory, never the chip. `Tensor` does not wait for its launches: they run in order, so each kernel sees the results of those before it, and only reading a result back (`tobytes`, `tolist`) waits. A timeout counts from the last launch that finished (`until`): a long queue is not a hang, a launch that never finishes is. On timeout it reads the dispatcher's current launch and each of its workers' mailboxes, and names the workers that have not finished.

**DRAM** (`Runtime.dram_alloc`, `dram_free`). A tensor takes the same address range in all 8 banks (tile i in bank i % 8). First fit over a sorted list of holes, 64-byte aligned; a freed block merges with the holes beside it. A `Tensor` frees its DRAM in `__del__`, which CPython runs as soon as the last reference goes, so intermediates and old results go back right away (a reference cycle would hold them until the garbage collector runs: `realize`'s walk is a plain function, not a closure, for that reason). A freed block goes back only once every launch queued before the free is done (`reclaim`), since those may still read or write it; an allocation that finds no room waits for the queue first. ttsim models 4080 MiB a bank.

**Device API**, prepended to every program (`DEVICE`): `noc_me()` (own translated coordinates, `NOC_ID_LOGICAL`), `noc_read`, `noc_write`, `noc_wait`, `l1_settle`. All requests go through NIU 0, request initiator 0, transaction ID 0. They are acknowledged (`RESP_MARKED`), so `NIU_MST_REQS_OUTSTANDING_ID(0)` counts them, on a dynamic VC. Rules (ISA `NoC/MemoryMap.md`, `NoC/Counters.md`, enforced by ttsim):

- At most 16 KiB per request (`noc_read`/`noc_write` split; ttsim does not model hardware splitting).
- Source and destination agree in their low 4 bits for L1, low 6 bits when DRAM is involved.
- Reads must be acknowledged. Inline writes to L1 are broken on Blackhole; we never use inline writes.
- Wait for `NOC_CMD_CTRL` bit 0 to clear before touching an initiator; read `NOC_CMD_CTRL` back before reading counters.
- Before the NIU reads data a core just stored: `fence`, load from L1, `fence` (`l1_settle`). After data lands: `fence` before reading it (the L0 data cache). Every `fence` is also a compiler barrier (`"memory"` clobber), or the compiler can cache L1 across it.

**Not built yet:**

1. **Commands other than "launch":** host-to-DRAM copies, events, waiting on another queue.
2. **NCRISC**, a second data mover, stays in reset. It could take the sends home off BRISC, or fetch for a second NoC.
3. **Hang reports from silicon.** The host should also read each stuck core's PC snapshot (`0xFFB1_3138`...); ttsim does not model those registers.
4. **Ordering of the host's ring writes before the doorbell** relies on x86 ordering cached stores before an uncached MMIO write. Fine on x86, and needs a barrier anywhere else.

## The Tensix coprocessor, and the first compiler

`ferrite/compiler.py`, encodings and register numbers in `ferrite/bh.py`. Tested on ttsim: `test_tile_math`, `test_tensor_math`, `test_matmul_exact`, `test_matmul_random`, `test_stream_on_one_worker`, `test_fusion`, `test_threads_wait`, `test_matmul_fusion`, `test_sfpu`, `test_reductions`, `test_softmax_and_layernorm`.

**Who drives it.** Three Tensix threads, each an in-order instruction FIFO; any thread can issue to any unit (unpackers, FPU, packer, ...), the threads only differ in their per-thread state (ADCs, RWCs, GPRs, ThreadConfig). TRISC t pushes to thread t, through that thread's MOP expander; BRISC can push to all three (`0xFFE4_0000 + 0x10000 t`, skipping the MOP expander), but a BRISC push and a TRISC push in the same cycle drop the TRISC's, so only one of them may feed a thread. ferrite: each TRISC feeds its own thread and BRISC only moves data (ISA `BabyRISCV/PushTensixInstruction.md`).

**Instruction words.** `opcode << 24 | fields` (ISA `Diagrams/Src/Bits32.lua`; ttsim `data/bh/tensix_isa.json` agrees and names the fields). A TRISC pushes one by storing it to `0xFFE4_0000`, or by executing it: `.ttinsn` is the word rotated left by 2, written straight into the code (opcodes are below `0xC0`, so the low bits are never `0b11` and it cannot be read as RISC-V). Either way a full FIFO stalls the TRISC. The generated programs are mostly `.ttinsn` words (`compiler.tt`, as `.word`s, since clang has no `.ttinsn`); runs that differ per Dst half or frame are branches between two copies. Setup goes through stores instead: a TRISC fuses up to four `.ttinsn`s in a row, and fused ones must not be what a `REPLAY` records.

**Configuration.** Three kinds, all needed:

- *Backend config*: 32-bit words at `0xFFEF_0000`, two banks (thread chooses with `CFG_STATE_ID`), a RISC-V store hits bank 0. The ISA docs name the fields but defer every index, shift and width to tt-metal's `cfg_defines.h`; ours come from ttsim's `tensix_regs.json` and must be checked on silicon.
- *Thread config*: 16-bit registers per thread, written only by `SETC16` from that thread, so they are set in-band at the start of the kernel. Reset values are undocumented, so every one we depend on is written.
- *ADCs and RWCs*: per-thread address counters (`SETADC*`) for unpackers and packer, and register-window counters (`SETRWC`) for the FPU.

TRISC0 writes the backend config, then `fence`, reads back a config word, `fence`, and only then pushes; it initialises the semaphores, waits for its thread to run all that (`tensix_idle`), and sets `READY` in `SYNC`. TRISC1 and TRISC2 push their own thread config and setup (address modes, replay recordings, the packer's counters: nothing that reads the backend config or a semaphore) and then wait for `READY` before anything else: the semaphores' power-on values are not documented. All of it is straight-line stores of constants, generated per kernel (`_pushes`), not loops over tables in L1: with the overlap, a one-tile kernel went from 790 cycles to 490, and a GPT-2 token from 170k to 150k.

**The pipeline** (`fused_kernel`). Each thread has one job, so unpack, math and pack work on different tiles at the same time. L1 holds an array of tile slots; the unpackers' Z counter (Z stride: one tile) and the packer's channel-1 Y counter (that Y stride: one tile) pick the slot, so moving between slots never touches config:

| Thread | Job | Per tile (or matmul step) |
|---|---|---|
| 0 | unpack, and report | `SETADC` each unpacker's Z to its slot, `UNPACR` into SrcA and SrcB; after a frame or a matmul step, wait for the unpackers and `SEMPOST` SEM_UNPACKED |
| 1 | math | per result: set the address modes for its kind (elementwise or matmul; one kernel can mix them), `SEMWAIT` for a free Dst half, aim at it (`SETC16 MATH_Offset` = 0 or 64), zero it if the op accumulates (`ZEROACC` four 16-row blocks); the FPU program (one per matmul step); at the end: wait for the FPU, `SEMPOST` SEM_DST |
| 2 | pack | `SEMWAIT` SEM_DST, aim the packer at the Dst half (ADC ch0 Y) and the L1 slot (ADC ch1 Y), 16 `PACR`, wait for the packer, `SEMGET` SEM_DST, `SEMPOST` SEM_L1 (an intermediate) or SEM_PACKED (an output) |

Handoffs:

- **Unpack to math**: SrcA and SrcB have two banks each, and the hardware does the rest (`UNPACR` waits for a bank the FPU has released, the FPU for one the unpacker has filled).
- **Math to pack**: SEM_DST counts results in Dst waiting to be packed (max 2, one per half). Math waits while both halves are full, pack while none is.
- **Pack to unpack** (fused chains): an intermediate is packed into L1 and unpacked by a later op. Pack posts SEM_L1 for each; before an op that reads an intermediate, unpack takes one for every op packed since it last looked. Packs happen in op order, so that covers whichever earlier op it reads (a plain count would not: the op might read an older intermediate than the one just posted).
- **Between BRISC and the TRISCs**: words in `SYNC`, each written by one core and read with a `fence` first (the L0 data cache never sees another core's writes). BRISC cannot read semaphores, so TRISC0 and TRISC2 relay theirs: they read SEM_UNPACKED or SEM_PACKED (TRISCs see the semaphores at `0xFFE8_0020 + 4i`), take one with a store of 1, and bump a count in `SYNC`.

| word | written by | means |
|---|---|---|
| `FETCHED` | BRISC | fetches done (a frame, or a matmul step's pair); TRISC0 waits for the next before unpacking it |
| `UNPACKED` | TRISC0 | SEM_UNPACKED posts relayed; BRISC refills a slot once the tiles in it two uses ago are unpacked |
| `READY` | TRISC0 | config and semaphores set up |
| `PACKED`, `RING` | TRISC2 | results packed, and where each is (slot) and goes (tile), four entries |
| `SENT` | BRISC | results sent home; TRISC2 packs into an output slot only after its last result there is sent |

**Four programs, one schedule.** A kernel is a program per core, all four running the same loops over units, passes and tiles, each doing its own part of every tile:

- BRISC: wait until the frame's slots are free (`UNPACKED`), fetch the tile's inputs, bump `FETCHED`; for a matmul, the same per step through the stream slots. While it waits, it sends home whatever `PACKED` says is ready, in order, and bumps `SENT`. At the end, until all its results are sent.
- TRISC0: wait for `FETCHED` (relaying SEM_UNPACKED meanwhile), then the unpacks, `drain` before an op that reads an intermediate, and the report.
- TRISC1: the math, alternating Dst halves (it tracks `h` itself).
- TRISC2: the packs, tracking `h` the same way; for the result, first claim its output slot (`output`), then relay SEM_PACKED until every result is counted.

A result goes home as soon as it is packed, and the only waits between the cores are on data: a slot to be refilled, a tile to be unpacked, a result to be sent. No core waits for one that is waiting for it: every wait loop on BRISC also sends, and every one on TRISC0 or TRISC2 also relays.

**Fusion** (`fused_kernel`, `Tensor` is lazy). `+`, `-`, `*` and `@` build a tree; `realize` (or `tolist`) compiles the whole tree into one kernel, so `(a @ b) * d - c` or `(a @ b) + (e @ f)` is one launch. A matmul's own operands are realized first: each of its output tiles needs a whole row and column of them. Equal subtrees become one op. Slot 0 is the runtime's zero tile, slots 1-4 stream matmul tiles; per output tile, a frame of slots holds the elementwise inputs, then one slot per op in post order; the last is the result. BRISC fetches the inputs into the frame; then, op by op, an elementwise op is unpacked from its operands' slots, computed into the next Dst half and packed into its own slot, or a matmul runs its K steps (each: BRISC waits until the stream slots from two steps back are unpacked and fetches B(k, j), and A(i, k) unless A's row is in L1 already; unpack; MVMULs) and is packed. Intermediates live in L1 only, and only the result goes to DRAM. Two frames alternate, so tile n + 1 is fetched while tile n computes, and results go home while both happen. Every intermediate is rounded to bf16 on its way to L1, exactly as it would be on its way to DRAM, so a fused chain is bit for bit the same as one launch per op (`test_fusion` checks). A single op is just a one-op chain. Limits: matmuls only over realized tensors (no fused prologue), at most 15 ops (SEM_L1 is 4 bits).

**The SFPU: relu, exp, reciprocal** (`compiler.SFPU`). A 32-lane fp32 vector unit that works on Dst, issued from the math thread right after the FPU, over the same Dst half. Facts (ISA `VectorUnit.md`, `SFPLOAD.md` and the per-instruction pages; ttsim `tensix.cpp`):

- LRegs 0-7 are writable; 8 = 0.8373, 9 = 0.0, 10 = 1.0, 11 = -1.0 are constants. Writing 12-15 configures load macros instead of computing, so never use them as a destination.
- An `SFPLOAD`/`SFPSTORE` moves 4 Dst rows x 8 columns (even columns at address 4r, odd at 4r + 2). So 32 of them cover a tile, and the address is the instruction's immediate plus the Dst counter plus the math thread's `MATH_Offset`. With fp32 Dst use mod0 3; the 16-bit modes address raw rows.
- We load with address mode 2 (no step) and store with 3 (Dst counter + 2), so one body repeated 32 times walks the tile, and it is recorded in the replay buffer.
- `SFPMAD` is a fused multiply-add with one rounding; denormals flush to zero in and out. Blackhole stalls for the 2-cycle MAD latency by itself (Wormhole needed `SFPNOP`s), with listed exceptions none of our programs hit.
- `SFPSWAP` mod 1 leaves the min in vd and the max in vc. `SFPARECIP` (Blackhole only) is a reciprocal estimate within 0.6%: inf for 0 and denormals, 0 above 2^126. There is no exp instruction in ttsim (`SFPARECIP` mode 2 is rejected).
- Condition codes are shared by all threads: `SFPENCC` mod 10 imm 3 turns lane flags on with all lanes set, imm 2 turns them off. `SFPIADD` with an immediate narrows the enabled lanes.

The programs:

- **relu**: `max(x, 0)` with one `SFPSWAP` against 0.
- **exp**: `(e^(x/4096))^4096` as a degree-4 Taylor series and 12 squarings, only multiply-adds. x is raised to at least -104 first, or the series overflows for very negative x. About 1e-4 relative in fp32, within a bf16 step after packing.
- **reciprocal**: the `SFPARECIP` estimate and two Newton steps `y += y(1 - xy)`, only in lanes where x is a normal number. 0 and inf would make `0 * inf = NaN`, so those lanes keep the estimate (inf and 0).

Packing to bf16 turns NaN into inf, so a missing guard shows only for an inf input. `test_sfpu` feeds ±inf, ±0, a denormal and ±1e30.

**Where a unary op runs.** In place, on the Dst result of the op below it, if nothing else reads that result (`relu(a @ b + c)` is one op with two SFPU programs after it). Otherwise its operand goes through `x + 0` (slot 0 is the runtime's zero tile) into Dst first. In place it sees the fp32 result, not one rounded to bf16, so it can be more accurate than op by op: `exp(x + y)` fused is `exp` of the fp32 sum.

**ttsim's FIFOs.** ttsim executes one instruction per pipe per clock. A TRISC's `.ttinsn` stalls while its thread's FIFO holds more than 52 entries, like silicon's stall on a full FIFO; stores to `0xFFE4_0000` have no such check there (only the short setup uses them), and a 512-entry overflow kills the process. (When BRISC pushed everything, it had to pause after each `REPLAY` for this reason.)

**Replay budget.** The math thread's 32 entries are shared out per kernel: SFPU bodies first (32 replays a tile, longest first), then the matmul and mul phases (4). A run that does not fit is pushed by a loop on TRISC1 (`compiler.Loop`), still correct, just more pushes. exp (21) and reciprocal (12) do not fit together.

**Row reductions** (`rowsum`, `rowmax`; `reduce_step`, `across_rows`, `COMBINE`). Each column tile of the operand is unpacked beside the zero tile and added to zero into Dst. (BRISC used to zero that tile at the start of every such kernel: 512 stores, 1200 cycles, more than a one-tile reduction's whole work. Now the runtime keeps one in every worker, and a GPT-2 token went from 282k cycles to 195k.) The first goes into an accumulator (Dst rows 256-319, or 320-383 for a second reduction in the same pass). Each later one goes 128 rows further on, where an SFPU pass folds it in: load both, `SFPSWAP` for max or a multiply-add by 1.0 for sum, store; 32 passes of 4 words, replayed.

At the end of the row, `across_rows` folds each tile row over its 32 columns:

- Row r sits in Dst rows `(r // 16) * 32 + r % 16` (columns 0-15) and 16 rows on (16-31). So four `SFPLOAD`s (even and odd columns, both faces) hold 4 rows at a time, a row per 8-lane group.
- After folding those four, `SFPSHFT2` mod 3 rotates each 8-lane group by one lane. Rotating by 1, 2 and 4 and folding each time leaves the row's result in all 8 lanes.
- That goes back to all four places, so the row's value fills all 32 columns. This is 8 groups of 21 words, pushed directly since the addresses differ.

The accumulator is packed into a slot. Each accumulator has a semaphore (1 while free): the first step of a row takes it, and the pack gives it back, so the next row cannot overwrite an accumulator still being packed. Sums are fp32 throughout, rounded once.

**Rows in passes, one launch** (`fused_kernel`). If an expression reduces rows, a work unit is a whole row of tiles and each worker owns rows; otherwise a unit is one tile (or, with one matmul, a block of them: "Blocks" below). A row is done in passes, each a sweep over its column tiles. A reduction's result is readable only in the pass after the one that swept its operand, and each pass recomputes the wide values it needs rather than keep whole rows in L1.

So `realize` works some of it out first, on its own: a long matmul (K over 256), and an elementwise value that a reduction reads and something else reads too (a layer norm's input). The second matters when there are few rows: GPT-2's residual stream is a lazy sum of 14 terms by the time a layer norm reads it, which with the norm is more than one kernel holds; split by the 15-op rule into 3 or 4 kernels of one row unit each, all of it ran on one worker, three passes over. Realized first, the sum runs a tile per worker, and the norm is one kernel: a layer is the 11 launches it should be, and a token went from 195k cycles to 170k.

softmax `e * e.rowsum().reciprocal()` with `e = (x - x.rowmax()).exp()` is three passes in one launch:

1. Sweep x into the row max.
2. Compute e tile by tile and sweep it into the row sum; the reciprocal runs in place on the accumulator.
3. Recompute e and multiply by the reciprocal.

Values one tile wide (reduction results, ops on them like `rowsum(x) * (1/n)`, narrow inputs) are computed once per row, in the first tile of the pass that needs them, into slots of their own after the two frames.

Book-keeping happens at run time, in each program, because what a tile does depends on whether it is a pass's first or last:

- Which Dst half each op gets (TRISC1 and TRISC2, each counting the same way).
- How many `SEM_L1` posts an unpack must take before reading an intermediate (TRISC0 takes all packed so far: `drain`).
- Which frame or stream slot is free again (BRISC, counting thread 0's reports the way TRISC0 pushes them).

**Any shape** (`Tensor`, `fused_kernel(shapes=...)` takes each input's rows and columns). A tensor is padded to whole tiles, and the padding has one rule: zeros, except that a dimension of 1 fills its tile with copies (a row vector's tile holds the row in all 32 rows, a column's in all 32 columns, a scalar's everywhere). Uploads follow it, and so does every kernel's result:

- **Results go home tidy.** In a tile at the result's last row or column, the math thread zeroes the padding in Dst before the pack (`mask_columns` with 0, `mask_rows`: SFPU stores into the lanes past the edge, picked as below). Inside a kernel the padding is whatever the ops made of it (`exp(0)` is 1); in DRAM it is zero, so a matmul can sum over a padded K: zero columns of A meet zero rows of B. (Multiplying by a 0/1 mask instead would not do: an inf in the padding times 0 is NaN.) A dimension of 1 stays filled by itself: an op on filled tiles gives filled tiles, a row reduction writes all 32 columns, and a matmul with a filled B gives a filled result. This first ran on BRISC, a store per padding datum before the send: 5000 cycles a tile, 129k for a 16-row layer norm. On the SFPU it is a few dozen instructions.
- **Row reductions mask their last tile.** Padding columns would count in a sum, and with all values negative a padding 0 would win a max. After the last column tile of a row lands in Dst, an SFPU pass (`mask_columns`) stores the identity (0, or -inf for max) into the columns past the edge. Which columns an `SFPSTORE` lane holds is fixed (lane l of a store at an even or odd address: column 2 (l % 8) or one more, plus 16 in the right faces), and `LReg[15]` holds 2l in every lane (ISA `LReg.md`; ttsim agrees), so `LReg[15] & 14` compared with the first padded column (`SFPIADD` with an immediate, flags on) enables exactly the padded lanes, and the store only writes enabled lanes. 32 stores at most, only in a row's last tile.
- **Broadcasting is numpy's**: shapes match, or one side has 1 where the other does not; a number is a 1 x 1 tensor. It is tile reuse, not a hardware broadcast: the fetch for output tile (i, j) takes the 1-wide input's tile (i, 0), (0, j) or (0, 0), and the filling makes that tile right everywhere. A computed operand one tile wide stays in the expression and is worked out once per row; one with a single row (repeated down) is realized first in its own kernel. `rowsum` and `rowmax` give a column (rows x 1); of a column they are the column itself (32 copies would add up). `(R, 1) @ (1, C)` is broadcasting, not a matmul (a filled K would add up 32 times).

**A batch dimension** (`Tensor(..., batch=)`, `shape` is (batch, rows, cols)). A batch is its matrices one after another in DRAM; a batch of 1 repeats across a bigger one (broadcasting, as numpy's leading dimension). Kernels take each input as address and tile strides for batch and row plus an offset, and units run through the batch. Elementwise ops, row reductions and matmul (matrix by matrix, or by one matrix for all) all work on batches. Views: `x[b]` is one matrix, and `x.split(n, axis)` cuts a matrix into a batch of n in place, which is how attention's heads become a batch: `test_batch`, and GPT-2's attention is three batched kernels instead of a loop over twelve heads.

**Views** (`x[r0:r1, c0:c1]`). A slice on tile boundaries shares its owner's DRAM: tile (b, i, j) of the view is tile `b * bstride + i * stride + j + offset` of the owner's, and every kernel takes each input as (address, batch stride, row stride, offset), so views go straight into kernels with no copy. The view keeps its owner alive, and only the owner frees. A view ends on a tile boundary or at the tensor's end, so its padding follows the rule too. GPT-2's q, k and v are views of one qkv result, split into heads by `split`.

**Transpose** (`x.T`, `transpose_kernel`). Result tile (i, j) is input tile (j, i) transposed. BRISC fetches the tile face by face with the two off-diagonal faces swapped (four 512-byte NoC reads), and unpacker 0 transposes each 16x16 face on its way into SrcA (`Haloize_mode`, backend config word 72 bit 8; ISA `UNPACR_Regular.md` swaps the row's low 4 bits with the column; unpacker 1 ignores it). Then `x + 0` through the FPU, which is exact. Since `Haloize_mode` applies to every SrcA load in the kernel, a transpose is a kernel of its own, run when `.T` is asked for rather than fused.

**gelu** is GPT-2's tanh form, built from what is there: `0.5 x (1 + tanh(z))` is `x / (1 + e^(-2z))`, so `x * (1 + exp(x (-k c x^2 - k))).reciprocal()` with `k = 2 sqrt(2/pi)`, `c = 0.044715`. One kernel; within 2% of numpy after the bf16 rounding of each step.

**Upload and download** go a tile at a time through the `array` module (floats to fp32 bytes, then bf16 bits with one list comprehension), about 10x faster than a struct call per value, which matters at GPT-2's 768 x 3072.

**A GPT-2 layer** (`test_gpt2_layer`). GPT-2 small's block: 768 wide, 12 heads of 64, MLP 3072, on 16 tokens, with weights drawn the way GPT-2 initialises them. `gpt2_layer` is written once and runs on both ferrite `Tensor`s and numpy arrays (the reference, float64, same bf16 weights). Layer norms are one launch each; q, k and v are 64-column views of one qkv result; the heads are a batch (`split`): queries and values split by columns, the keys transposed once and split by rows, then one kernel for every head's scores, mask and softmax, one for attention times values, and one through each head's 64 rows of `w_proj`; the 12 heads are summed (the same as concatenating them, with no concat) and added to the residual last, after the bias, so the large value is rounded once. Inside a kernel that sweeps rows in passes, a long matmul (K over 256) runs first on its own, or each pass would recompute it. A token was 148 launches and 0.8M simulated cycles (before multicast launches and split K: now 138 and 150k). The residual reaches 3000 in one channel, where bf16 steps by 16, so the logits drift about 1 from numpy's over 12 blocks: a pick within that of numpy's best is a tie bf16 cannot break (`test_gpt2_model`). 49 launches, about 6 s on ttsim. Every intermediate is bf16, so the block's change to x matches to within 2% of its size (0.03 against 1.7). Breaking the row mask, the tidy padding, the transpose's face swap or the views' offsets each fails it.

**GPT-2 small, the real thing** (`test/gpt2.py`, `test_gpt2_model` when `GPT2=` names the weights). Hugging Face's `openai-community/gpt2` `model.safetensors` (float32, read with numpy and rounded to bf16), tiktoken's `gpt2` tokenizer. `forward` is written once for Tensors and numpy arrays: token and position embeddings, 12 blocks, the last layer norm, and the output layer, which is the token embedding again. The numpy version, in fp32, matches Hugging Face's own GPT-2 to 1e-4 in the logits, so it is the model and not just a copy of our mistakes. On ttsim, greedy from "The meaning of life is", ferrite writes " not the same as the meaning of death", the same 8 tokens as numpy and as Hugging Face, each step's 50257 logits within about 1.5 (they span about 100). What it took:

- `Tensor.frombytes` / `tobytes`: bf16 in row-major order, moved into tile order 16 values at a time with no per-value Python work. All 124M weights are on the chip in 5 s.
- `Tensor.gather(ids)`, the embedding lookup: a BRISC-only kernel that copies each row's two 32-byte runs out of the table's tiles. A NoC read from DRAM keeps its low 6 address bits, so a run comes in as the 64-byte block holding it and is copied into place.
- The output layer multiplies by `wte.T`, transposed on the chip once (37,700 tiles, 1.4 s on ttsim).
- Logits only for the last position: `gather([n - 1])` of the final layer norm, a filled one-row tile.
- No KV cache: up to 32 tokens are one tile row anyway, so a step costs the same either way. 27 s a token on ttsim, about 120,000 matmul steps (one per weight tile); the output layer is 7 s of that. On silicon the same work is a few milliseconds.
- `Tensor._run` waits up to 60 s: on the simulator the biggest kernels take seconds.

**Too big for one kernel.** A kernel holds at most 15 ops (SEM_L1 is 4 bits); the 12 heads' sum alone is 23. `realize` counts the ops a kernel would get and, while there are too many, realizes the operand with the most first, on its own.

**MNIST** (`test_mnist_mlp`): 784 -> 128 -> 10 with relu and biases on a batch of 16, two launches, against numpy in float64: logits within 2%, and the same predicted classes wherever the top two are not a near tie. It found a bug as old as matmul fusion: the kernel took its row width from every input, including a matmul's A, whose columns are K. With K wider than the result (784 against 128) the units were cut wrong and whole result tiles were never computed. The tests had only multiplied with K at most the result's width. Now the width comes from what elementwise ops and reductions sweep.

Bug found on the way: a row reduction whose operand is one tile wide inside a wider kernel was swept like a wide one, adding the same tile once per column tile. It is now reduced in the pass's first tile; `test_any_shape` covers it (and caught the old code).

**rsqrt** (SFPU, x > 0): the bit trick `0x5F3759DF - bits(x) / 2` (`SFPSHFT` by -1 from vc, `SFPIADD` vc - vd with no flag update) within 3.5%, then two Newton steps `y *= 1.5 - x/2 * y^2`. 0 and negatives are not handled.

**Softmax and layer norm** are built from these in `test_softmax_and_layernorm`:

- softmax: `e = (x - x.rowmax()).exp(); e * e.rowsum().reciprocal()`
- layer norm: `d = x - x.rowsum() * (1/n); d * ((d * d).rowsum() * (1/n) + eps).rsqrt() * gamma + beta`

Each is one launch; they match float64 to about 2% (softmax) and 0.05 + 2% (layer norm), the bf16 rounding of each intermediate.

**Replay buffers and address modes keep the pushes small.** Each TRISC pushes one instruction a cycle (four if they come in a row), so words per tile are its code and time per tile. Each thread has a 32-entry replay buffer (`REPLAY` with `load` records the next N instructions; without it, plays them back; it sits after the MOP expander, so it worked from BRISC too). Each kernel's setup records the runs that repeat: one 16-`MVMUL` fidelity phase and one 9-word `ELWMUL` phase in the math thread, 15 `PACR`s in the pack thread. The last instruction of a sequence that also has to release the Src banks is pushed by itself with `clear_dvalid`. Elementwise ops use address modes 0-1 and matmul 4-7, all set once at setup, so a kernel can mix them with no per-op rewrites. Blackhole's `addr_mode` field is 3 bits (bits 16:14 in ttsim's `tensix_isa.json`; the ISA pages are Wormhole's, with 2 bits plus an `ADDR_MOD_SET_Base` that Blackhole dropped). This needs confirming on silicon.

| words pushed, all three threads | before | now |
|---|---|---|
| add or sub tile | 48 | 26 |
| mul tile | 80 | 27 |
| matmul step | 71 | 12 |
| matmul output tile, on top of its steps | 39 | 17 |

What is left is per-tile control (slots, Dst half, semaphores). `test_replay_keeps_pushes_small` holds these numbers. A matmul test multiplies by a diagonal of full-mantissa values, so every output is one product that is exact only if all four fidelity phases ran; it catches a replay one instruction short.

**No push order to get right.** When BRISC fed all three threads through one stream of stores, a full FIFO stalled BRISC, so it could never queue work a thread would block on before pushing what unblocks it. With a TRISC per thread, a stalled TRISC holds up only its own thread, whose work waits on the others anyway.

**What ttsim can and cannot check here.** ttsim runs each Tensix instruction the moment it can, in zero time, and ignores the busy conditions of `STALLWAIT`. Each TRISC now pushes as fast as it can, and TRISC2, with the least to push, runs ahead, so the waits really happen there. Breaking each one on purpose (and running the tests) shows: without the pack's `SEMWAIT`, the math's Dst-half wait or the SEM_L1 wait, ttsim stops on a semaphore taken while empty or posted while full (`test_threads_wait` alone hits all three); without BRISC's wait for `UNPACKED` or TRISC2's wait for `SENT`, results are wrong; without TRISC0's wait for `FETCHED` or the math's wait for `READY`, a test fails or hangs. Not caught on ttsim: the `STALLWAIT` before thread 0's report (unpacks are instant there), TRISC2's wait for `READY` and TRISC0's `tensix_idle` before setting it (ttsim's semaphores start at 0 and its thread 0 is always done by then), and the accumulator semaphores. Speed-up from the overlap cannot be measured on ttsim at all; that waits for silicon.

**Tile layout** (ours; the ISA has no notion of a tile). A 32x32 tile is four 16x16 faces in row-major order (top-left, top-right, bottom-left, bottom-right), each face row-major: element (r, c) is at `(r//16*2 + c//16)*256 + r%16*16 + c%16`. An unpacker copies the 1024 datums linearly into Src rows, so face f is Src rows 16f..16f+15, which is what `MVMUL` wants; Dst and the packer keep the same order. Elementwise ops do not care. Tensors are tiles in row-major tile order, tile i in DRAM bank i % 8 at `base + i//8 * 2048`.

**Matmul, one pair of tiles** (`compiler.matmul`). `MVMUL` does `Dst[8x16] += SrcB[8x16] @ SrcA[16x16]`: SrcB rows from its row counter (a multiple of 8), SrcA rows from its counter (a multiple of 16 in ttsim), Dst rows from the instruction's `dst` field plus the Dst counter. So the left tile goes to SrcB (unpacker 1) and the right tile to SrcA (unpacker 0). Output face (i, j) = sum over k of left face (i, k) @ right face (k, j): 8 face products of 2 `MVMUL`s each, 16 per fidelity phase, 4 phases (SrcA gives 5 mantissa bits a phase, SrcB 7). The `dst` field carries the output rows; the Src counters walk in the order i, k, h (half), j, which needs only four address modes (slots 4-7): SrcA +16 rows (next j), SrcA -16 / SrcB +8 (next h), SrcA +16 / SrcB +8 (next k or i), and the same plus fidelity +1 at the end of a phase. Counters wrap at 64, so that last step lands back on row 0. The last `MVMUL` of the last phase gives both banks back to the unpackers.

**Matmul, whole tensors.** A matmul is an op in a fused kernel (alone, it is a one-op kernel). Each worker owns some output tiles; each is K steps through the pipeline: the left (i, k) and right (k, j) tiles, 64 `MVMUL`s into the same Dst half, which accumulates over k in fp32; then one pack. Error on random bf16 inputs is under 2^-8 of sum |a*b| (1.4e-3 measured on 64x96 @ 96x128).

**A's row stays in L1.** Each matmul in a kernel has a row of slots for A's tiles (K of them, after the row values). A worker's result tiles run along rows, so while they stay in row i only B's tiles are fetched; A's row comes in with the steps of the first tile in a new row (`fresh`, after everything so far is unpacked, which matters only when K is one tile: otherwise the wait for B's slot two steps back already covers it, and ttsim never shows the difference). Unpacker 1 walks the row by itself: the matmul sets its Z to the row's first slot once, and each step's `UNPACR` adds 1 (`AddrMode` bits 0-1, Ch0ZInc). That halves DRAM reads for a matmul with one row of tiles, which is every matmul of GPT-2 decoding a short prompt: each weight tile is read once a token.

**Blocks.** With one matmul and no row passes, a unit is a block of up to 4 x 4 result tiles, done column by column: the block's rows of A and its current column of B stay in L1, unpacker 1 walking A's row (its start worked out at run time and pushed with a store, since it depends on the row) and unpacker 0 walking B's column. Each A and B tile is read once a block rather than once a result tile. Part-blocks at the edges leave out the tiles past the matrix (`test_matmul_blocks` puts a canary after the result). The benchmark's DRAM reads went from 14.9 to 6.9 times its inputs; reading B once per block row is what is left, and sharing tiles between workers (multicast) would be the next step.

**Split K.** A unit is at least a result tile, so a matmul with few of them leaves workers idle however long K is: a token's MLP down-projection, (5 x 3072) @ (3072 x 768), is 24 result tiles of 96 steps each, on 24 of 119 workers. When an unbatched matmul's result tiles times s still fit on the workers and each part keeps at least 8 K tiles, `@` cuts K into s parts (2, 4 or 8; `split` of A's columns and B's rows, views, no copy), multiplies them as a batch, and sums the parts lazily, so the sum fuses with what comes after (the bias, the residual). The down-projection becomes 96 units of 24 steps, 8000 cycles to 3000, and a token 371k cycles to 312k. Each part is rounded to bf16 before the sum, one more rounding than Dst's fp32 sum over all of K; GPT-2 still picks numpy's token 8 times out of 8. On silicon it also spreads the weights' DRAM reads over four times the cores.

**Fuzzing** (`test_fuzz`). Random expressions (elementwise ops with broadcasting, unary ops, row reductions, matmuls, odd shapes, batches), and every kernel they turn into checked against numpy on the same inputs read back from the chip. Checking kernel by kernel keeps bf16 rounding that piles up through a tree from passing for a bug, or hiding one. The test runs 30 expressions; longer runs of the same code (thousands) found two limits the compiler did not respect: a kernel sweeps its rows at one width, so row reductions of tensors of different widths cannot share one (the narrower was reduced as if one tile wide), and it has two accumulators, so a third reduction in a pass crashed. `_expr` now computes such reductions first, on their own.

**Only slots below 256 can be unpacked.** An unpacker's input address takes only the low 8 bits of its Z counter (ISA `ADCs.md`; ttsim agrees, its Z is 13 bits wide but the unpacker sees 8), so frames, row values and A's rows all sit below slot 256 (`UNPACKABLE`). A matmul whose row does not fit streams A through slots 3-4 instead. Two long matmuls in one kernel once wrapped past it and read the wrong tiles; it took a checker running every kernel of a GPT-2 forward against numpy to find.

**Precision.** Dst is kept in fp32 (`Fp32_enabled`); the packer rounds it to bf16 (`(bits + 0x8000) >> 16`, the only fp32-Dst-to-bf16 mode ttsim allows is `Read_32b_data` with In = bf16). With 4 fidelity phases a bf16 x bf16 product is exact before that rounding (bit-exact on 8192 random pairs). `ELWADD`/`ELWSUB` align the two 11-bit mantissas with round-half-up first, so about 0.4% of random sums differ from the fp32 sum by a last-place bf16 step. fp32 tensors would lose 13 mantissa bits in SrcA/SrcB (TF32); full fp32 needs unpack-to-Dst and the SFPU.

**Where the docs and ttsim disagree.** `UNPACR` bit 7 is `MultiContextMode` in the ISA and `ovrd_thread_id` in ttsim; both say it must be 1. Bit 0 is "write 0" in the ISA, `last` (must be 1) in ttsim; we follow ttsim. The Blackhole README says "4x packer", the PACR page says one packer with four Dst read ports. `SETDVALID` is unsupported on Blackhole (use `UNPACR_NOP`).

**ttsim's cycles are its model, not silicon's.** A kernel whose root is a binary op costs about 110 cycles a tile more than one whose root is unary (`x + y` 119 a tile, `exp(x + y)` 8, on one worker): the difference is in how ttsim times the math thread's last `STALLWAIT` of a tile, not in our code. The benchmark and the per-token numbers above count ttsim's cycles; use them to compare versions of our code, not to predict silicon.

**What ttsim cannot check here.** It executes each Tensix instruction atomically and ignores most `STALLWAIT` conditions (FPU, packer, unpacker busy), so a missing wait between units passes on ttsim and fails on silicon. The `STALLWAIT`s here are what the ISA asks for.

**Next for the compiler:**

1. The TRISCs push every instruction themselves. The MOP expander (which BRISC could not use) would turn a whole tile's math into one word, and the loops over units and tiles could run in it.
2. Deeper pipelines: more L1 slots, so DRAM latency hides behind more than one step.
3. Matmul reuse: A's row stays in L1, but B's tiles are fetched once per result row, and every core reads its own copies. Blocks of result tiles per core (B reused across rows), or multicasting tiles to the cores that share them, would cut DRAM reads further.
4. Row passes recompute: softmax computes exp twice and reads x three times. Keeping a row's tiles in L1 between passes (when they fit) would trade L1 for that. Column reductions. With few rows, row ownership leaves workers idle; splitting a row across workers would need a cross-worker reduction. Fusion beyond this: a matmul epilogue could stay in Dst (`(a @ b) + c` by unpacking c and adding into the same half, no pack/unpack of the product), prologues (`(a + b) @ c`, per K step), unary ops, broadcasts. Keeping intermediates in fp32 (an fp32 pack, or moving Dst back into SrcA/SrcB) instead of rounding each to bf16.
5. The SFPU, for fp32, exp, recip, and the rest.
6. Weights in BFP8. A token reads every weight once, so on silicon decode waits on DRAM, and BFP8 (16 values share an exponent byte; a tile is 64 exponent bytes, then 1024 sign-and-magnitude bytes, 1088 in all) halves that. What ttsim shows (`tensix.cpp`): the packer writes that layout when `Pack_out_format` is BFP8 (6) and `Exp_section_size` (config word 68, bits 16-31, in 16-byte units) is 4, so a cast is an `x + 0` kernel like the transpose; a value is `sign * magnitude * 2^(exponent - 133)`; MVMUL takes a BFP8 SrcA with a bf16 SrcB (the Src format comes from the unpacker that filled the bank). The catch is the unpacker: its format is per kernel (config words 64 and 72, byte 0: `RMWCIB0` can switch them inside the unpack thread, after a `STALLWAIT` for the unpackers), and it finds a BFP8 tile's exponents at `base + 64 z` and its data after an exponent section sized by the tile descriptor's dimensions, so Z cannot pick a 2 KiB slot the way it does for bf16. A matmul with BFP8 weights would set the unpacker's base per tile (as LLK does) or give the B stream a region of its own laid out that way. Not built: it needs the card to confirm how silicon addresses it.

## Bring-up on a p150

Everything so far ran on ttsim, which models the chip from Tenstorrent's own knowledge but runs each instruction at once, ignores most busy conditions, and completes NoC requests instantly. The day the card arrives, go in this order; each step only needs the ones before it. The tests open the card themselves when `TTSIM` is not set (`pci.open_device`), so the commands are the simulator's without it, as root.

Before: `lspci -d 1e52:b140` shows the card; the IOMMU is off or in passthrough (`iommu=pt`), since `pci.Card` hands the device physical addresses; one 2 MiB hugepage is free (`echo 4 > /proc/sys/vm/nr_hugepages`). `pci.Card` unbinds tt-kmd itself.

| step | command | what it confirms (and what to suspect if it fails) |
|---|---|---|
| 1 | `pytest test -k "config_space or arc"` | BARs, the TLB config registers (`bh.TLB_CONFIG`, `TLB_FIELDS`), the NoC coordinates of ARC, ARC's boot status, message queue and telemetry tags, and that `POWER_ON` wakes Tensix (else try AICLK go busy, `0x52`) |
| 2 | `-k "dma or l1 or dram"` | the PCIe tile's loopback into host memory (`bh.PCIE`, bus addresses below 2^58), multicast and unicast L1 writes, DRAM bank coordinates and GDDR training |
| 3 | `python3 -m ferrite.hello` | soft reset bits, reset PCs and their override (`bh.TRISC_RESET_PC`), the firmware's jump table, the NIU's request registers, device writes into host memory, and the dispatcher: its multicasts from a core (the rectangle in `NOC_RET_ADDR_HI` with translated coordinates, one acknowledgement per worker, the counter clear) and the workers' atomic increments. A hang here with every worker done (their mailboxes say so) points at the count; one before any worker starts, at the multicast's acknowledgements (the dispatcher waits for exactly one per worker: more, from tiles in the rectangle that are not workers, would hang it rather than let a go overtake the arguments) |
| 4 | `-k "every_core or relaunch or moves_data or queue or hung"` | the launch path end to end; instruction cache invalidation (`bh.CFG_IC_INVALIDATE`, a no-op in ttsim, so only here can a stale kernel show) |
| 5 | `-k "tile_math or tensor_math"` | the backend config indices (`bh.TENSIX_CFG_FIELDS`, from ttsim's `tensix_regs.json`, not the ISA), instruction encodings, `.ttinsn`, semaphores from the TRISCs, unpack and pack of bf16 |
| 6 | `-k "matmul or stream"` | MVMUL, the 3-bit address modes (from ttsim's `tensix_isa.json`; the ISA pages are Wormhole's), fidelity phases, the replay buffer |
| 7 | `-k "sfpu or reductions or any_shape or views"` | SFPU encodings, LReg 15 as 2 x lane, lane flags, `SFPARECIP`, `Haloize_mode` |
| 8 | the whole suite, a few times over | the waits ttsim never makes happen: the `STALLWAIT`s before reports and packs, TRISC2's wait for `READY`, `tensix_idle` before setting it (NOTES above: "What ttsim can and cannot check here"). A failure that comes and goes points here |
| 9 | `GPT2=... pytest test -k gpt2`, then `test/gpt2.py` | everything at once, and the first real timings |

For a hang, `Runtime.wait` names the stuck workers; read their PC snapshots (`0xFFB1_3138`, five words) with `chip.read32` to see where each core sits (ttsim does not model them, so this is untested too).

## Still unknown or untested

- The real-card backend (`pci.Card`) has never run: the p150 has not arrived. It assumes root, the IOMMU off or in passthrough, and a free 2 MiB hugepage (`echo 1 > /proc/sys/vm/nr_hugepages`).
- Whether the all-on power message is enough to wake Tensix on silicon after kmd clock-gated it, or whether `0x52` (AICLK go busy) is needed too.
- Whether the host-to-PCIe-tile loopback in `test_dma` (BAR read that turns into a device read of host memory) behaves on silicon as it does on ttsim. Device-initiated writes from a core's NIU are the path that matters and are untested.
- ISA does not document BAR2 (iATU, DMA engine), the DRAM tile's internal address map, ARC's message protocol, or the power-on value of soft reset. ttsim models them; real silicon may differ.
- ttsim does not model `mhartid`, `fence.i`, the PC snapshot registers, NoC request splitting, or most of BAR2, and it completes every NoC request instantly, so the outstanding-request counters never move there. Instruction cache invalidation is a no-op in ttsim, so `test_relaunch_without_reset` cannot catch a stale-instruction bug.
