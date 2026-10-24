# Notes

How ferrite drives a Blackhole, from the PCIe BARs up to the kernels its compiler emits, where each fact about the chip
comes from, and what is still unknown. Facts the code uses live in `ferrite/bh.py`. Sources: **ISA** =
tt-isa-documentation/BlackholeA0, **sim** = ttsim `src/`, **kmd** = tt-kmd, **umd** = tt-umd.

## The chip from the host

### PCIe function

Vendor 0x1E52, device 0xB140 (sim `pcie.cpp`, kmd). Three 64-bit prefetchable BARs (ISA `PCIExpressTile/README.md`):

| BAR | Size | Contents |
|---|---|---|
| BAR0 | 512 MiB | `0x0000_0000-0x193F_FFFF`: 202 TLB windows of 2 MiB (TLBs 0-201; 0-31 can do strided multicast; 201 is the kernel driver's) |
| | | `0x1FC0_0000`: TLB config, `u32[3]` per TLB for TLBs 0-209, then `u32 strided[32]` at `0x1FC0_09D8` |
| | | `0x1FD0_4000`, `0x1FD1_4000`: the PCIe tile's NoC0 and NoC1 NIU registers |
| | | `0x1FE0_0000-0x1FFF_FFFF`: ARC on PCIe 1 only; reserved on PCIe 0, the one a p150 uses |
| BAR2 | 1 MiB | PCIe controller (DBI): eDMA at `0x100-0x1A8`, iATU at `0x1000` (outbound region r at `0x1000 + 0x200 r`). Not in ISA; offsets from kmd `blackhole.c`, umd, sim |
| BAR4 | 32 GiB | 8 TLB windows of 4 GiB (TLBs 202-209) |

### NoC

17 x 12 grid. Raw NoC0 coordinates (ISA `Diagrams/Src/NoC.lua`, sim `tile.cpp`): DRAM at x=0 (banks 0-3) and x=9 (banks
4-7), ARC (8,0), PCIe 0 (2,0) and PCIe 1 (11,0), Ethernet on y=1, Tensix at x=1..7,10..16, y=2..11. NoC1 numbers the
grid from the other corner: (16-x, 11-y).

Firmware turns on **coordinate translation** at boot (`NIU_CFG_0` bit 14) and programs every tile's tables, so harvested
Tensix columns move to the highest x (ISA `NoC/Coordinates.md`). ferrite uses translated coordinates everywhere:

| Translated (x, y) | What |
|---|---|
| x 1..7, 10..14, y 2..11 | Tensix, 120 on a p150 (x 15, 16 are its 2 harvested columns from firmware 19.5 on, umd; before 19.5 all 140 run) |
| x 17..18, y 12..23 | DRAM: bank b at (17 + b//4, 12 + 3(b%4) + port), port 0..2, all three reaching the same 4 GiB |
| (19, 24) | the PCIe tile connected to the host |
| x 20..31, y 25 | Ethernet |
| y 0 and 1 | not translated in x: ARC is (8, 0) |

A NoC address is (x, y, 64-bit address inside that tile).

### Inside a Tensix tile

As the tile's own cores see it (ISA `TensixTile/BabyRISCV/README.md`):

| Address | What | Over NoC |
|---|---|---|
| `0x0000_0000-0x0017_FFFF` | L1, 1.5 MiB. Code runs only from here | yes |
| `0xFFB0_0000` | the core's data RAM: 8 KiB for BRISC and NCRISC, 4 KiB a TRISC; the stack | no |
| `0xFFB1_2000-0xFFB1_2FFF` | tile control and debug registers: soft reset, reset PCs | yes |
| `0xFFB1_3138-0xFFB1_314B` | each core's PC snapshot | yes |
| `0xFFB2_0000`, `0xFFB3_0000` | NoC0 and NoC1 NIU registers | yes |
| `0xFFE0_0000`... | Tensix coprocessor: GPRs, instruction push, mailboxes, config | no |

MMIO registers take exactly one 4-byte access, from either side.

### DRAM and host memory

Eight 4 GiB banks; the address inside a DRAM tile is the byte offset in its bank (umd soc descriptor; sim models 4080
MiB a bank). GDDR must be trained before first use: telemetry tag 22 has 2 bits a bank (trained, error); umd waits up to
300 s and retrains with ARC message `0xB6`.

The device reaches host memory by NoC requests to the PCIe tile (19, 24) (ISA `PCIExpressTile/README.md`): an address
below 2^58 goes to the host as a bus address (bits 58/59: relaxed ordering, no-snoop); bit 60 goes through the outbound
iATU; `0xF800_0000_0000_0000` is the DBI. So a host buffer at bus address B is NoC address (19, 24, B), with no iATU
setup (kmd's DMA-mask comment says the same). kmd and umd route pinned memory through the bit-60 window instead, so the
direct path rests on firmware having set window 0 up after each boot. If it fails on the card, the fallback is kmd's:
outbound region r at BAR2 + `0x1000 + 0x200 r` (base +`0x08`/`0x0C`, limit +`0x10`/`0x20`, target +`0x14`/`0x18`,
CTRL_1 = `1 << 13`, CTRL_2 = `1 << 31`; kmd `blackhole.c` 86-120, 878-911) and bit 60 on the bus address. ttsim models
that path too (`translate_pci_dma_addr`).

### TLB windows

A TLB turns a range of BAR0 (or BAR4) into a range of one tile's NoC address space: a 96-bit config at `BAR0 +
0x1FC0_0000 + 12 i`, written as three 32-bit uncached stores (odd TLBs are not 8-byte aligned), the address window
aligned (ISA `PCIExpressTile/HostToDeviceTLBs.md`, sim, umd). 2 MiB TLBs:

| Bits | Field |
|---|---|
| 0-42 | NoC address >> 21 |
| 43-48, 49-54 | x_end, y_end (the target, or a multicast rectangle's far corner) |
| 55-60, 61-66 | x_start, y_start (multicast only) |
| 67-68 | noc |
| 69 | multicast |
| 70-71 | ordering: 0 relaxed, 1 strict AXI, 2 posted writes, 3 counted writes |
| 72 | linked (never safe) |
| 73, 75, 76-77 | static VC enable, buddy, VC class |

4 GiB TLBs: address >> 32 in bits 0-31, then x_end 32, y_end 38, x_start 44, y_start 50, noc 56, mcast 58, ordering 59,
linked 61, static VC 62. A read+write window on a static VC "crashes the host" (umd `architecture_tlbs.cpp`), so unicast
uses dynamic VC. ferrite moves one shared window (TLB 0, strict ordering) whenever an access crosses into another 2 MiB
block or tile, and gives each runtime's doorbell a TLB of its own; multicast sets x_start/y_start, mcast and static VC
class 2, as umd does.

### Reset, and what the cores run

Per Tensix tile, `RISCV_DEBUG_REG_SOFT_RESET_0` at `0xFFB1_21B0` (ISA `TensixTile/SoftReset.md`), read-modify-write; a
set bit holds a core: 11 BRISC, 12-14 TRISC0-2, 18 NCRISC. Power-on is `0x47800`, all five held (sim). umd releases with
`0x80047000` (bit 31: staggered start, power only). Leaving reset invalidates the instruction cache, zeroes the GPRs,
sets the PC, and zeroes the core's data RAM over up to 2048 cycles unless `0xFFB1_2224` says not to: load code while the
core is held.

There is no boot ROM. A released core fetches from L1 at its reset PC: BRISC `0x0`; TRISC0/1/2 `0x6000`/`0xA000`/
`0xE000`, or `0xFFB1_2228/222C/2230` if bit n of `0xFFB1_2234` is set; NCRISC `0x12000`, or `0xFFB1_2238` if bit 0 of
`0xFFB1_223C` is set.

The cores are rv32im with Zicsr, Zaamo (no lr/sc), Zba, Zbb and part of F; no C extension (its encoding space holds
`.ttinsn`). ferrite compiles with `clang --target=riscv32-unknown-elf -march=rv32im` and links a flat binary with lld.
Gotchas (ISA `BabyRISCV/`):

- `mhartid` reads 0 on every core (ttsim kills the process on it): tell cores apart by entry point.
- `fence.i` does not flush the instruction cache (ttsim rejects it): new code needs a reset release or a write to
  `RISCV_IC_INVALIDATE` in Tensix config.
- The L0 data cache (4 x 16 B) is not coherent with NoC or host writes: poll L1 with a `fence` in the loop.
- Stores to one region (L1) land in order; across regions a `fence` is not enough: store, load it back, use it, signal.
- Unsupported instructions are undefined behaviour, not traps.

ARC runs the board firmware and set up translation and the PCIe windows; Ethernet core E0 runs Tenstorrent's Ethernet
firmware.

### What tt-kmd did, done by ferrite

| kmd | ferrite (`pci.Card`, `Chip`) |
|---|---|
| enable the device, 58-bit DMA mask, bus mastering; Max Read Request Size 4096 | unbind kmd, sysfs `enable`, memory + bus master in `PCI_COMMAND`, MRRS 4096 |
| map BAR0, BAR2, BAR4; reserve TLB 201 | mmap sysfs `resource0` alone, read and write it a word at a time (never libc's memcpy, which may go a byte at a time on uncached memory), zero the strided multicast registers; every TLB but 201 is ours |
| wait for ARC: `0x80030408` bit 0 (umd: `& 7 == 5`, 300 s after a reset) | wait for `& 7 == 5` |
| ARC `0xA0` (ASIC_STATE0); power policy `0x21`: all off with no fd open (**Tensix clock-gated**), `0x7FFE0F21` on open | send `0xA0`, then `0x7FFF0F21` (everything on, max AI clock), then `0x52` (AICLK go busy), then wait up to 0.2 s for telemetry's AI clock within 5% of 1,350 MHz, as umd does |
| pin host memory, program an outbound iATU region | a locked 2 MiB hugepage, its bus address from `/proc/self/pagemap` (IOMMU off or passthrough), used directly as a NoC address |
| reset: `PCI_COMMAND` bit 6 as a marker, config writes `0x934 = 1` then `0x930 = 0x11`, wait for the device and firmware, restore config space | `pci.reset` (`python3 -m ferrite.chip reset`): the same writes through sysfs `config`, then the kernel's remove and rescan; untested |
| telemetry: `*0x80030434` tags, `*0x80030430` values, tag word `id \| index << 16` | `Chip.telemetry`; Tensix columns from tag 34, GDDR training from tags 22 and 36 (waits up to 300 s) |
| "NoC cleanup": cores back in reset when a process dies | a `Runtime` halts its cores at exit and before it takes its hugepage |

**ARC messages** (kmd `msgqueue.c`, umd): `*0x8003042C` points at a control block, word 0 the base of 4 queues, word 1
`entries | queues << 8`. Queue i (3 = applications) is a 32-byte header (`+0` request wptr, `+4` response rptr, `+0x10`
request rptr, `+0x14` response wptr; pointers wrap at 2N), then N request slots and N response slots of 32 bytes (`u32
header, u32 payload[7]`). Write a slot, bump the request wptr, poke ARC (write `0x10000` to `0x80030100`), poll the
response wptr for up to 1 s; a status below 240 in the low byte is OK. ttsim answers kmd's and umd's messages, kills the
process on an unknown code, and returns zeros (so `0x57`, harvesting, is unusable there).

## The runtime

`ferrite/runtime.py`.

**Cores.** `Runtime(chip, start, end)` takes a rectangle of Tensix in translated coordinates. Its last core runs the
**dispatcher**, the rest are **workers**, also a grid of whole NoC rows (`Runtime.grid`: 9 rows of 12) for kernels that
multicast along rows and columns.

**A worker's L1:**

| Address | What |
|---|---|
| `0x0` | firmware: a jump per core, BRISC at `0x0`, TRISC i at `0x4 + 4i` |
| `0x1000` | mailbox: `[go, kernel, arguments, workers] [done, the dispatcher's x \| y << 6, scratch, index]` |
| `0x1020` | the TRISCs' `[go, done 0, done 1, done 2]` |
| `0x1040` | `SYNC`: 64 bytes for a kernel's cores; its first 8 words zeroed before each launch |
| `0x1080` | `FLAGS`: 32 words for workers to signal each other over the NoC, zeroed by their own worker at the end |
| `0x1100` | two argument buffers (launches alternate) |
| `0x2000` | loaded kernels, 500 KiB: each four entry offsets, then a program per core |
| `0x7F000` | a tile of zeros, 4 KiB, written once at boot (the compiler's slot 0) |
| `0x80000` | kernel data (`L1_DATA`) |

The dispatcher's tile: firmware at `0x0`, config at `0x1000` (`[doorbell, ring lo, hi, ring size, status lo, hi,
workers' rectangle, workers]`), its outbox at `0x1400`, the workers' report count at `0x80000` and the sum of the flits
their reads brought in beside it, command staging from `0x80040`.

**Boot** (host, once): multicast reset and worker firmware over the rectangle, a mailbox per worker (its index, the
dispatcher's coordinates), the dispatcher's firmware and config, a TLB aimed at the doorbell, the TRISC reset PCs, then
multicast release of every BRISC and TRISC.

**Loaded kernels** (`Runtime.load`). A kernel goes into every worker's L1 once, by one multicast, and stays; launches
name its address. Images run wherever they land (clang's medany model, linker relaxation and jump tables off). When the
500 KiB are full, the next load waits for the queue and evicts them all. A kernel's `.bss` is cleared at each launch.

**Commands.** A launch is one command: a 16-byte header `[length, kernel address, argument bytes, workers]`, then the
arguments, 64-byte aligned, in a ring in host memory; then the new write pointer into the doorbell, the host's only PCIe
access, for one launch or a recorded block of them (`Runtime.block`). Commands never straddle the ring's end (a length
of 0 marks the tail unused); pointers are free-running byte counts.

**The dispatcher** (BRISC firmware, about 1.2 KB), in a loop:

1. Poll the doorbell; once it has run every command it read, read all the host has written since (at most `STAGE`, 64
   KiB) in one NoC read from the PCIe tile, and run the commands from its L1.
2. Multicast the arguments to every worker and wait for every acknowledgement.
3. Fire the go: the same 16 bytes from its outbox (launch number, kernel, arguments, workers) into every mailbox.
4. Poll the count of reports until every worker has added to it.
5. Write the launch's times and bytes (`profile`), then `[read pointer, launches done]`, into host memory as posted writes
   on one static VC, which stay in order, so the host never sees a launch done before its times.

The next command's arguments go into the other buffer while a launch runs.

**Multicast from a core**: a write with `NOC_CMD_BRCST_PACKET`, path reserve and static VC class 2 (ttsim insists on
all three), the rectangle in `NOC_RET_ADDR_HI` as `end | start << 12`, `NOC_CMD_BRCST_SRC_INCLUDE` off. Each recipient
acknowledges, which leaves `NIU_MST_REQS_OUTSTANDING_ID` wrong (up once, down once a recipient), so the sender counts
`NIU_MST_WR_ACK_RECEIVED` up to the recipients and then clears the outstanding counters (`0xFFFF` at `NIU + 0x60`),
with nothing else in flight.

**A worker** (firmware for all four cores, about 680 bytes). BRISC presets request initiators 1 and 2 at boot
(`noc_setup`). Each launch it polls `go` with a `fence`, sits out if its index is past the launch's workers, takes its
share of the units, zeroes `SYNC`'s counters, invalidates the instruction caches (Tensix config 185, bits 0-3, then
`fence`, read back, `fence`), writes the TRISCs' go, runs its program, waits for its transfers and the TRISCs' done,
marks the launch done in its mailbox, and adds 1 to the dispatcher's count by a NoC atomic increment (posted, on a static
VC, as tt-metal's). A TRISC polls its
go, runs its program, waits until its Tensix thread has run all it pushed (`tensix_idle`), and writes its done. A queued
launch takes about 150 ttsim cycles.

**Profiles** (`Runtime.profile`). The dispatcher reads its tile's wall clock (`RISCV_DEBUG_REG_WALL_CLOCK_L`, ISA
`DebugTimestamper.md`) at a launch's go and at its last report, into a ring of 1024 entries in host memory, so every
launch has its time on the chip's own clock. Where the NIU counts read data (`NIU_MST_RD_DATA_WORD_RECEIVED`, counter 3,
64-byte flits; ISA `NoC/Counters.md`: a card, and ttsim-chaos, since ttsim as released stops on it), each worker reads it
at go and at done and adds the difference to a sum beside the dispatcher's count, by a second atomic increment on the
same static VC before its report, so the sum is whole once the count is; the dispatcher takes its growth before the next
go, so every launch also has the bytes its workers' reads brought in: DRAM's, other tiles' L1, a tile's copies within
its own. Launches are named by their ops; `profile(since)` sums them by name.

**Replay** (`tensor.jit`). Its first call runs; its second is recorded (every launch, `Runtime.log`, and the DRAM it
takes stays taken, `Runtime.pinned`); each later call copies its inputs into the second's and sends the recorded
launches as one block, one doorbell write, nothing built or compiled. It asks of the function fixed input shapes and
dtypes, nothing else it reads changing, and no reading back; its result and everything it kept are the same tensors
every call. A decode step replays this way (a token's place is data, not a shape): 0.1 ms of host a token, against 21
ms built op by op.

**Waiting** (`Runtime.wait`) polls the status words in host memory, never the chip, with a timeout counted from the
last launch that finished. Only reading back waits; a readback is itself a launch (`readback_kernel`) that copies tiles
into host memory (a tensor of fewer than 32 rows sends only its rows). On timeout it names the workers that have not
finished and, for the first four, which cores, their `SYNC` counters and `FLAGS`, their NoC requests in flight, and on a
card their PCs (`Runtime.report`).

**DRAM** (`dram_alloc`, `dram_free`). A tensor takes the same address range in all 8 banks (tile i in bank i % 8),
first fit, 64-byte aligned. A `Tensor` frees its DRAM in `__del__`, which goes back only once every launch queued before
the free is done.

**Device API**, prepended to every program (`DEVICE`): `noc_me`, `noc_read`, `noc_write`, `noc_get`, `noc_put`,
`noc_wait`, `l1_settle`. `noc_get` and `noc_put` use the preset initiators 1 and 2, writing only the four registers that
change (17 instructions a read). Rules (ISA `NoC/MemoryMap.md`, `NoC/Counters.md`, enforced by ttsim):

- At most 16 KiB a request; source and destination agree in their low 4 bits for L1, low 6 when DRAM is involved.
- Reads must be acknowledged; inline writes to L1 are broken on Blackhole.
- Wait for `NOC_CMD_CTRL` bit 0 before touching an initiator; read it back before reading counters.
- Before the NIU reads what a core just stored: `fence`, load, `fence` (`l1_settle`); after data lands, `fence` before
  reading it. Every `fence` is also a compiler barrier.

Not built: commands other than a launch, NCRISC (in reset), and a barrier between the host's ring writes and the doorbell
(x86 orders them).

## The compiler

`ferrite/compiler.py` emits the kernels; `ferrite/tensor.py` builds the trees of ops they compute.

**Who drives the Tensix.** Three threads, each an in-order instruction FIFO; any thread can issue to any unit. TRISC t
pushes to thread t; BRISC only moves data (a BRISC and a TRISC push in the same cycle drop the TRISC's). An instruction
is `opcode << 24 | fields` (ISA `Diagrams/Src/Bits32.lua`; ttsim `tensix_isa.json` names the fields). A TRISC pushes by
executing it: `.ttinsn`, the word rotated left by 2, written into the code (`compiler.tt`); a full FIFO stalls it. A
word that varies at run time (a Dst half, a slot) is a store of the word plus a C value (`field`); what a `REPLAY`
records goes through stores too, since fused `.ttinsn`s must not be recorded.

**Configuration**, all needed: backend config (32-bit words at `0xFFEF_0000`; indices from ttsim's `tensix_regs.json`,
to check on silicon), thread config (16-bit, `SETC16` from its own thread only, reset values undocumented, so all are
written), and per-thread address counters (`SETADC*`, `SETRWC`). TRISC0 writes the backend config, reads it back between
fences, sets the semaphores, waits for its thread (`tensix_idle`) and sets `READY`; the other TRISCs wait for it.

**Tiles.** A 32x32 tile is four 16x16 faces in row-major order, each row-major: (r, c) at `(r//16*2 + c//16)*256 +
r%16*16 + c%16`, which is what `MVMUL` wants. A tensor's tiles are in row-major tile order, tile i in bank i % 8; tile
(i, j) of matrix b is tile `b // rep * bstride + i * stride + j + offset` (views and repeats).

**Precision.** Dst is fp32; the packer rounds to bf16 half up. Four fidelity phases make a bf16 product exact.
`ELWADD`/`ELWSUB` align mantissas first (about 0.4% of random sums are a last-place step off). fp32 would lose 13 bits
in SrcA/SrcB, so fp32 tensors are unpacked into Dst and computed on the SFPU.

**Where the ISA and ttsim disagree.** `UNPACR` bit 7 is `MultiContextMode` in the ISA and `ovrd_thread_id` in ttsim
(both 1); bit 0 is ttsim's `last` (must be 1). `SETDVALID` is unsupported (use `UNPACR_NOP`). `addr_mode` is 3 bits on
Blackhole (ttsim); the ISA pages are Wormhole's.

### The pipeline

Each thread has one job, so unpack, math and pack work on different tiles at once. The unpackers' Z counter and the
packer's channel-1 Y counter pick an L1 tile slot, so moving between slots never touches config.

| Thread | Per tile (or matmul step) |
|---|---|
| 0, unpack | `SETADC` each unpacker's Z to its slot, `UNPACR` into SrcA and SrcB; report each frame or step (`SEMPOST` SEM_UNPACKED) |
| 1, math | wait for a free Dst half (`SEMWAIT`), aim at it, zero it if the op accumulates; the FPU program; `SEMPOST` SEM_DST |
| 2, pack | `SEMWAIT` SEM_DST, aim at the half and the slot, 16 `PACR`s, `SEMGET` SEM_DST, `SEMPOST` SEM_L1 (an intermediate) or SEM_PACKED (a result) |

Handoffs: SrcA and SrcB have two banks each, and the hardware hands them over; SEM_DST counts results waiting (two Dst
halves); an op that reads an intermediate first takes one SEM_L1 for every op packed since it last looked (`drain`).
BRISC cannot read semaphores, so TRISC0 and TRISC2 relay theirs into words of `SYNC`, each written by one core and read
after a `fence`. A semaphore holds 15, and a post onto a full one breaks the contract, so neither pushes a 16th report
before relaying one.

| word | written by | means |
|---|---|---|
| `FETCHED` | BRISC | frames fetched |
| `ISSUED` | BRISC | matmul steps whose reads are out |
| `UNPACKED` | TRISC0 | reports relayed: a slot may be refilled |
| `READY` | TRISC0 | config and semaphores set up |
| `PACKED`, `RING` | TRISC2 | results packed, where each is and goes |
| `SENT` | BRISC | results sent home |

All four programs run the same loops over units, passes and tiles, each doing its own part of every tile, and every
wait loop also does its other duty (BRISC sends, TRISC0 and TRISC2 relay), so no core waits on one waiting for it. Each
thread's 32-entry replay buffer holds the runs that repeat: a fidelity phase of 16 `MVMUL`s, SFPU bodies, 15 `PACR`s. An
unpacker sees only the low 8 bits of its Z (ISA `ADCs.md`), so everything an unpacker reads sits below slot 256.

### Matmul

`MVMUL` does `Dst[8x16] += SrcB[8x16] @ SrcA[16x16]`, so A goes to SrcB (unpacker 1) and B to SrcA (unpacker 0). A pair
of tiles is 8 face products of 2 `MVMUL`s, 16 a fidelity phase, walked by four address modes (slots 4-7) whose counters
wrap at 64. A result tile is K steps into one Dst half, which sums in fp32.

- **What stays in L1.** A's row stays while a worker's result tiles stay in its row. With one matmul and no passes a
  unit is a block of up to 4 x 4 result tiles, A's rows and B's current column kept in L1.
- **Tiles shared between workers** (`Runtime.grid`). A matmul of more than one tile row gives grid worker (gy, gx) one
  block of the result: each worker of a grid row reads its share of A's rows and multicasts it along the row, each of a
  grid column its share of B's columns down the column into one of three buffers, each share followed by a flag in
  `FLAGS`. Three buffers need no message: a worker posts column c only once its own unpacker is done with c - 2. When
  A's rows and three buffers do not fit below slot 256, K is cut into parts (`shared_k`). A big matmul reads its inputs
  about once.
- **A token's row** (one tile row): every grid worker takes a block of the result's columns, the first grid row reads A
  in shares and multicasts it to all, and B streams.
- **B's matrices picked on the chip** (`matmul(..., index=)`): batch entry b is row b // k of A by the matrix of B its
  int32 index picks. BRISC fetches the indices first (up to 2400 tokens a call) and offsets B's tile addresses: a
  mixture of experts' matmul, the experts the chip's own choice, so a replayed step reads the ones its tokens chose. An
  index past B's matrices picks none: all four cores read it and skip the entry's steps, so its result is the zero Dst
  half packed, with nothing read or multiplied.
- **A's row shared by a batch**: one tile row of A that every matrix of a batch of more than the grid's rows reads (a
  decode step's tokens through each expert) is read once, in shares by the grid's first row, and multicast to the grid
  before the first tile. A row or column kept in L1 is tagged by its own matrix, so a batch whose entries share one
  (a repeat's) does not fetch it again.
- **Split K**: a matmul of one tile row with few result tiles cuts K into 2, 4 or 8 parts as a batch and sums them
  lazily, each part rounded to bf16 first.
- **Reads in flight** (`DEPTH`, 4): step k's reads carry NoC transaction ID k % 4 + 1, B streams through 4 slots, and
  thread 0's TRISC waits for that ID's `NIU_MST_REQS_OUTSTANDING_ID` to reach 0 before unpacking (BRISC reads initiator
  1's `CMD` back and fences before bumping `ISSUED`). The depth is a guess for the card to tune.

**Matmul precision.** The FPU adds each group of 8 products after aligning them at 12 bits (ttsim's `mvmul`), so a
matmul is good to about 2^-12 of its larger terms. An fp32 matmul is bf16x3: each operand hi + lo in bf16 (lo the
operand less itself rounded by `ftob`), hi@hi + hi@lo + lo@hi in fp32.

**Fidelity phases** (`matmul(fidelity=)`, `tensor.FIDELITY`): phase 0 multiplies A's top 7 bits by B's top 5, 1 the same
by B's last 3, 2 A's last bit by B's top 5, 3 the last by the last (ISA `SrcASrcB.md`). Four (the default) are exact;
two drop A's last bit; one keeps 5 bits of B. ttsim costs them the same; on silicon each phase is an `MVMUL` a block.

**Block floats** (`cast("bfp8")`, `cast("bfp4")`). A bfp8 tile is 1088 bytes: 64 exponents, one for each run of 16 (a
face's row), then 1024 bytes of a sign and 7 bits of magnitude, a value being `magnitude * 2^(exponent - 133)`. bfp4 is
3 bits, two values a byte, 576 bytes a tile. The packer casts to bfp8 from fp32 (`Pack_out_format` 6, `Exp_section_size`
4), rounding once, half up; bfp4 is a BRISC kernel of its own (`bfp4_kernel`) rounding to the nearest 3 bits, as
Tenstorrent's host conversion does (the packer's bfp4 truncates). The unpacker makes each value the bf16 it is on its
way into SrcA, so a matmul by a block-float B is bit for bit the bf16 matmul by the same values. It finds slot z's
exponents at `base + 64 z` and its values at `+ 1024 z`, a slot's first datum below 65536, so a block-float B lives in
regions past slot 255 (`B8`), read in two pieces a tile (`getb`); thread 0 switches unpacker 0 to it around the matmul's
steps (`bfp_b`, by `RMWCIB` between `STALLWAIT`s). `put` writes bf16 rows into a block-float tensor, each run rounded as
the cast rounds it: a KV cache in bfp8 or bfp4.

### Fusion and elementwise

**Fusion.** Ops build a tree; `realize` compiles it into one kernel. Per output tile a frame of slots holds the inputs,
then one slot per op; every intermediate is packed into L1, rounded to its dtype, so a fused chain is bit for bit one
launch per op; only the result goes to DRAM. Two frames alternate, so tile n + 1 is fetched while tile n computes. A
matmul's operands are realized first. A kernel holds at most 15 ops (SEM_L1 is 4 bits): `_fit` realizes a part first,
a value two ops read if one fits. A number is a 1 x 1 tensor of its tensor's dtype, uploaded once a runtime.

**The SFPU** (`compiler.SFPU`): a 32-lane fp32 vector unit on Dst, run from the math thread after the FPU (ISA
`VectorUnit.md`, ttsim `tensix.cpp`):

- LRegs 0-7 are writable; 9 = 0.0, 10 = 1.0, 11 = -1.0 are constants; writing 12-15 configures load macros. LReg 15
  holds 2 x lane.
- An `SFPLOAD`/`SFPSTORE` moves 4 Dst rows x 8 columns (even at 4r, odd at 4r + 2); 32 cover a tile. Load with address
  mode 2, store with 3 (Dst + 2), so one body run 32 times walks it.
- `SFPMAD` is a fused multiply-add rounding to nearest even; denormals flush. `SFPSWAP` mod 1 leaves the min in vd, the
  max in vc. `SFPARECIP` is a reciprocal estimate within 0.6%. `SFPGT` mod 8 gives an all-ones mask. Lane flags
  (`SFPENCC`, `SFPIADD` with an immediate, `SFPSETCC`) make every instruction write only enabled lanes.

The programs are tinygrad's primitives (exp2, log2, sin, sqrt, reciprocal; max, cmplt, cmpne; where) plus exp, log and
cos, with their constants in fp32 inside. Each transcendental reduces x by exact steps and finishes with a fitted
polynomial, so fp32 tensors get float32's accuracy (exp 8e-8, log 1.6e-7, sin 1e-7, sqrt 3 ulps). Binary programs see
their second operand 128 Dst rows on. `where` is `keep(c, x) + drop(c, y)`, exact, with no `c * x` to turn an inf into a
NaN. In a bf16 kernel a NaN is inf before the SFPU sees it, so only fp32 kernels check for NaN. A unary op on a result
nothing else reads runs in place in Dst on the unrounded result; otherwise on its operand plus the zero tile.

### Reductions and passes

`sum(axis)`, `max(axis)` (axis 0 the batch, 1 the rows, 2 each row to a column). Each tile along the axis lands in Dst
beside the zero tile: the first in an accumulator (Dst rows 256 or 320), each later one 128 rows on, folded in by an SFPU
pass. A line's last tile is masked to the identity where cut short (`mask_columns`, `mask_rows`). At the end the tile is
folded inside itself so the result fills it and broadcasts: along a row by rotating lane groups (`SFPSHFT2`), down a
column by `SFPTRANSP`. Each accumulator has a semaphore; sums are fp32 throughout.

If an expression reduces, a unit is a whole line of tiles along the axis, done in passes: a reduction is readable only
in the pass after it swept, and each pass recomputes what it needs. Softmax is three passes in one launch (the max, the
sum of e, e times the sum's reciprocal). An input read in more than one pass keeps its line in L1; a value one tile long
along the axis gets a slot of its own, worked out once a line. A kernel reduces along one axis; others are computed
first.

### Attention

`Tensor.attend` (`compiler.attention_kernel`): softmax(q kᵀ + mask) v in one kernel, its scores kept in L1. A unit is up
to eight tile rows of one head's queries. BRISC first reads the unit's places (each row's last key; with a sliding
window, its first), works out the tiles of keys they reach and which some row sees only in part, and tells the TRISCs.
The unit then sweeps those tiles twice, a frame at a time (a tile row of keys, in the second sweep the values too), and
for a tile some row sees only in part BRISC writes the mask's tiles into the frame from the places, -1e9 where a row
must not look. The keys' length is an argument, so one kernel serves every length, and a server one replay for all.

- Each tile of scores is made in Dst: the mask's tile (or the zero tile) added first, then the queries by the keys, read
  transposed, `MVMUL` accumulating onto it.
- The first sweep folds them into each row's largest (in Dst's two accumulators for a unit of up to two tile rows, else
  in L1 between key tiles); the second subtracts the largest, and the SFPU's exp makes p in place.
- p, packed as bf16, is multiplied by the values and by a tile of ones (each row's sum of p), each product packed onto
  its output tile in L1 in fp32: the packer adds to what is there (`Pack_L1_Acc`, config word 71 bit 19), switched in
  and out by `repack`. One fused kernel divides the output by the sums: two launches.
- One sweep (flash attention's) would need the output rescaled in L1 whenever a row's largest grows, which the packer
  cannot do, so the keys are swept twice and the values once. A unit's keys that fit L1 stay there from the first sweep
  for the second (`attention_store`: a bf16 cache's in free slots; a block float's in regions of whole key rows, at most
  64 tiles each, below B8 past the slots in use and past the frames' regions, 104 rows of Qwen3.6's 256-wide heads in
  bfp8), the unpacker switched to each row's region: each key is read once. Within 1% of numpy. A decode step's
  attention at Qwen3.6's shapes reads 1.13 times the cache's keys and values at 8 slots of 2,048 places where it read
  1.65, and 1.04 at 32 of 1,024.

A decode step has few units (a key head's per slot), so when they are few each unit's keys are shared out among up to 8
**parts** on workers of their own (flash decoding), or as many more as keep each part's keys in L1, each part's p v, sums and largest coming home for a few fused kernels
to combine, each weighted by exp of its largest less the largest of all. A part with no keys takes one tile hidden by a
tile of -1e9; a row's largest is held at -1e4 at least, so a row that sees none of a part's keys gets weight 0; each
part's largest is rounded to bf16 in Dst so its p and its weight agree. `llama.attention` uses it for heads of 64 and
wider, but gpt-oss's (its sinks).

### Gated DeltaNet's step

`Tensor.delta` (`compiler.delta_kernel`): a decode step's Gated DeltaNet from its token's projections to its outputs,
or a verify step's token and its draft, in fp32 on the SFPU, each state read once and written once. A unit is a key
head of a sequence and its value heads, so it alone reads and writes its channels of the conv's last inputs. BRISC
reads its value heads' S (kt x vt tiles each) into a frame (two alternate between units), from the states after both
of the last step's tokens or after its first alone as accept says, and the row runs of its channels from the conv's
last 3 inputs, the tokens' own rows of the projections and the 4 taps: q's and k's go into columns 0-3 of a tile (a
channel a row), v's land in rows 0-3 (a channel a column), b, a, dt and decay in place 0. Once it has read them it
writes the inputs each token leaves (each one up a place, its own last; a verify step's first token's into those kept
apart), where the sequence is live: no shift launches, and the inputs never cross DRAM in whole tiles. The math thread
fills Dst from L1 itself, masks what lies outside the filled places to 0 there (so no tile is zeroed in L1), and folds
(`across_rows`, `across_cols`) into broadcast tiles: each channel's 4 inputs by its 4 taps summed across the row (or
down the column) through SiLU, q and k each to length 1 by the sum of their squares folded down (q times kd^-1/2),
beta = sigmoid(b) and e^g = exp(decay softplus(a + dt)) everywhere, k . q. The recurrence is separable by column: for
each value head and column tile j, one SFPU pass over its kt tiles sums u = S^T k and w = S^T q down the column (two
Dst accumulators), d = beta (v - e^g u) stays in Dst, the output's tile is e^g w + d (k . q), and the column's new
tiles e^g S + k d (a first of two tokens' packed into L1 too, the second's S). The SFPU programs are functions the
loops call, so no loop's body is thousands of inline Tensix words long: one that was made ttsim decode an XMOV (why is
not known). BRISC sends the output's row runs into its tile and the new tiles over
the old, but a dead sequence's. A decode step's Gated DeltaNet block is then the qkv matmul, this kernel, the output's
norm and gate and the output matmul: at Qwen3.6's shapes 22 launches with its experts where it was 30, 65 MB read where
it was 87 at one sequence and 174 where 204 at eight; the kernel's RISC-V work is 38K ttsim cycles at one sequence
and 55K at eight, where the six kernels it replaces took 13K and 20K.

### Shapes and movement

**Padding.** A tensor is padded to whole tiles with zeros, except that a dimension of 1 fills its tile with copies (a
row vector's tile holds the row 32 times), so broadcasting reuses the tile as it is. A result's padding is zeroed in Dst
before its last tiles are packed, so a matmul can sum over a padded K. `(R, 1) @ (1, C)` is broadcasting. The rows its
last tile holds are an argument (the mask's words pushed with them), so unless a kernel reduces along rows, reads a
B transposed or picks matrices by an index, one kernel serves every row count in the same tiles, and attention's
takes its query rows the same way: a prompt's last chunk, whatever its length, compiles nothing new past its tile
rows (a kernel takes about 0.2 s of clang).

**Batches and views.** A batch is matrices one after another; a batch of 1 repeats. Every kernel takes each input as
(address, batch stride, row stride, offset), so a view (`x[b]`, slices on tile boundaries, any forward step through the
batch, `split`) goes into kernels with no copy, and a result can be written into a view (`realize(into)`, `merge`).
`repeat_interleave(n)` repeats each matrix in place (grouped-query attention's keys).

**Transpose** (`.T`). A matmul reads a transposed B as it lies: unpacker 0 transposes each face on its way into SrcA
(`Haloize_mode`, config word 72 bit 8, switched by `haloize`) and the math walks the faces transposed (`mmT`, five
address modes). Elsewhere `transpose_kernel` fetches each tile's faces swapped.

**Movement at any granularity** (`reshape`, `permute`, `expand`, `pad`, any slice; `move_kernel`). Each is one map from
the result's elements to the source's (an offset, three strides, per-axis bounds, 0 outside); a chain of them composes
into one launch, except across a reshape. The kernel is BRISC alone: it fetches source tiles into a cache in L1 (127
slots, by tile number mod a prime), takes a whole tile by one NoC read where it can, and copies runs four datums at a
time (bf16 two a word).

**Rows by index.** `gather(ids)` copies rows out of a table (the embedding), a result tile's 64 runs read as 64-byte
blocks all in flight before one wait; `put(ids, rows)` writes rows into a tensor in place, a unit per row put (a KV
cache's token at its place, the place data rather than a shape). A NoC read's latency, which ttsim hides, is then paid
once a tile rather than once a run.

### Dtypes

**fp32** (4 KiB tiles): the math thread unpacks each operand straight into Dst (`fill`, unpacker 0's second context) and
the SFPU does every op, each rounded once, so a fused fp32 tree is numpy's float32. A kernel computes in one dtype; a
bf16 operand of an fp32 op is cast first. **int32**: two's complement through Dst and the LRegs (`SFPSTORE`'s INT32
mode, since its FP32 mode flushes small integers); add and sub by `SFPIADD`, comparisons through sign-magnitude
(`SFPCAST` mode 3). **Casts** pack in another format (two 4 KiB buffers at `OUTS`; the packer switched by `repack`).
**Upload and download** move values between row-major and tile order by strided 8-byte copies, a bank's tiles in one
write.

### Autograd

tinygrad's way: a rule per op (`_backward`), applied in reverse, each rule the same lazy ops, so the backward pass fuses
as the forward does. An fp32 matmul is recorded as one matmul. `test_autograd` checks every op against numpy's central
differences.

## Models

`models/llama.py` is written once for ferrite Tensors and numpy, numpy the reference at every step. It reads Llama's,
Mistral's, Qwen2's and Qwen3's checkpoints, and Mixtral's, Qwen2-MoE's, Qwen3-MoE's and gpt-oss's (experts unpacked from
MXFP4), and Qwen3.5's and Qwen3.6's, each checked against transformers' own model (`test_hf_families`, `test_qwen3_moe`,
`test_qwen3_5_moe`). Weights are read only when uploaded (`Weight`) and cast on the chip to their formats (`precision`:
bf16, bfp8, bfp4, or `tt`, Tenstorrent's performance precision); `llama.arrays` gives numpy the same values. Each block's
q, k and v are one matrix, gate and up another (`fuse`). RoPE is one kernel for q, k and v (`x cos + (x @ turn) sin`).
The tokenizer is tokenizer.json's byte-level BPE in plain Python, tiktoken's and Hugging Face's tokens over the same file.

- **A KV cache of a fixed size**: each token's keys and values `put` at its place, and attention's mask built on the
  chip from the places, so a decode step is the same launches every token and replays. A prompt goes in chunks.
- **Several sequences at once** (`forward`'s `slots`): their caches side by side in one, a key head's query heads of
  each sequence one batch entry, so a key head's rows are read once for all its query heads.
- **A mixture of experts** (`moe`): each token's k best experts by `topk` on the chip (a kernel of BRISC's, a row a
  worker), their SwiGLUs by matmuls that pick the experts' weights where they lie, weighted and summed. A decode step's
  tokens go through each expert some token picked once, every token an entry's rows (`Tensor.touched`: the experts
  picked in order, then none, and each entry's weight from each token, 0 where it picked another), weighted before the
  down projection and the entries summed, so each touched expert's weights are read once and the tokens' row once. A
  prompt's tokens are grouped by expert (`Tensor.group`, a counting sort on BRISC) so each expert's weights are read
  once for every 32 of its tokens, the tiles past those used picking none. Whichever makes fewer tile rows. Nothing
  comes back to the host, so a decode step replays. A synthetic block of 32 experts (2048 wide, 8 picked, bfp4) at 8
  tokens reads 67 MB where a batch entry for each token and choice read 162.
- **Gated DeltaNet** (`delta`, Qwen3.5 and 3.6): one matmul makes q, k, v, z, b and a; a causal conv of 4 taps (its last
  3 inputs kept) and SiLU; each value head keeps a 128 x 128 state, S = e^g S + beta k (v - e^g S^T k)^T, read by its
  query, RMS-normed and gated by SiLU(z), in fp32. A decode step's token, or a verify step's token and its draft, goes
  through the recurrent rule in one kernel of each head (`Tensor.delta`, `step` numpy's), q and k from the conv's row
  as it lies; a prompt goes in pieces of 64 (`chunk`, the UT transform: (I + L)^-1 = (I + P)(I + P^2)(I + P^4)..., P =
  -L, no loop over tokens), everything the state does not enter done for every piece at once, then the state through
  the pieces in turn. The states are written back in place (`assign`), so a replay carries them.
- **Gated attention** (Qwen3.5 and 3.6): each head's output times sigmoid(gate), RoPE on a quarter of each head, q and k
  RMS-normed per head, RMSNorms by 1 + w.
- **Speculative decoding** (`draft`, `forward`'s accept): the multi-token prediction layer is a block past the last;
  from a place's hidden state and the next token it gives a draft. A verify step takes a token and its draft and keeps
  the DeltaNet states twice, after both and after the first; the next step's accept row says which it goes on from,
  the kernel reading that one alone and writing both.
- **Token accuracy** (`llama.accuracy`), Tenstorrent's way: the reference file's (tt_transformers'
  `tests/reference_outputs/<model>.refpt`) first 512 tokens as the prompt, then 500 steps fed the file's next token;
  top-1 where the pick is the reference model's best, top-5 where it is among its five. In numpy it runs as one prefill
  with the weights as the chip holds them, for a model of any size.

**Qwen3.6-35B-A3B** (40 blocks 2048 wide, 30 Gated DeltaNet and 10 attention; 256 experts of 512, 8 chosen, and a
shared one; a vocabulary of 248,320). Real blocks are within 0.22% of transformers in numpy and about 1% a block on
ttsim. Token accuracy in numpy over the whole model: bf16 top-1 99.6%, top-5 100%; every expert matrix in bfp4, the rest
and the KV cache in bfp8 (21.7 GB, one card) 92.0% and 94.2%; the experts' gate and up in bfp4 and the rest in bfp8
(27.2 GB) 93.8% and 96.0%; the experts' gate and up in bfp4 and the rest in bf16 98.2% and 99.6%, so the experts' down
projection in bfp4 costs most of it. Every matrix in bfp8 (two cards) and a bfp4 KV cache are not measured yet: their
runs on this host came out NaN ("Still unknown"). Tenstorrent's performance mode runs Llama-3.1-8B at 90% top-1. The multi-token prediction
layer's draft is the model's own greedy pick at 94.3% of the places past the prompt (92.5% over all 1,022), in numpy
in bf16 over the same text.

## Serving

`models/serve.py`, an OpenAI-compatible server.

- **Slots.** The engine's KV cache has `slots` slots of `places` rows, and each step is one replayed call for every slot,
  a free one running on its last row, which no sequence reaches, its attention seeing the slot's first row alone (one
  tile read) and its Gated DeltaNet states left as they are. Attention in one kernel reads only the rows each
  sequence's place reaches, so one replay serves every length; otherwise (gpt-oss's sinks, heads narrower than 64)
  attention reads each slot's rows up to the power of 2 (32 at least) that holds the longest sequence's, a replay
  recorded for each such length. A prompt is prefilled a chunk between steps (2,048 tokens for a mixture of experts,
  whose chunks read nearly all its experts; 256 otherwise).
- **The prefix cache.** A prompt goes to the free slot holding the longest start of it, and only the rest is prefilled.
- **Sessions kept.** Gated DeltaNet's states are for exactly the tokens they took in, so a session (its tokens, its
  slot's cache rows and its states) is kept when its prompt is in and when it ends, and a prompt that begins with a kept
  session goes on from it in any slot: an agent's next turn prefills only its new tokens. It is copied on the chip, its
  tiles as they are (`assign` of a tensor computed already: `copy_kernel`, BRISC's, a block float's too), into DRAM the
  caches leave (`park` bytes: by default all that is free when the engine starts but 2 GiB), so keeping one and taking
  it back are a few launches and no PCIe; past that the least recently used go to the host (`keep` bytes), read back
  and later written over the BARs, and past that they are gone.
- **Speculative steps** for a model with a prediction layer: a call verifying every slot's token and draft, then one
  drafting the next (each row's best token, on the chip); a pick is sampled as it would be without a draft, so the tokens
  are the model's own.
- **Heads** (`heads`). The output layer writes its logits in fp32, and each call ends with each row's head on the chip:
  its 64 largest logits and their tokens, by `topk` on parts of the row a worker's L1 holds (Qwen3.6's 248,320 as 97 of
  2,560), each part's 64 side by side, then a `topk` of those given their tokens as labels; and the log of the sum of
  exp(x t) over the row at its request's t = 1/temperature, each part's by one fp32 kernel (two sweeps over a part kept
  in L1) and theirs by another. The host reads back under 1 KB a row, not the row: greedy is its first token; top_k up
  to 64, and top_p and min_p whose cut lies inside the head, are exact from it, the sum giving each token's share of the
  whole row; a draw with none of them that lands past the head's mass reads the whole rows back and draws from the rest
  of its row alone. Penalties, shapes and logprobs read the whole row. 8 slots' tokens at temperature 0.7 and top_p 0.95
  take 0.08 ms of the host's a step, where drawing from whole rows took 4.3.
- **Requests.** Penalties (presence, frequency, repetition, as vLLM applies them), temperature, top_k, top_p, min_p,
  seed, stop strings (searched for only where the newest token's text could complete one), logprobs; those a request
  leaves out are the model's own (its `generation_config.json`'s, as vLLM takes them: Qwen3.6's top_k 20, top_p 0.95),
  else OpenAI's. Qwen3.5's and 3.6's
  template written out, tool calls back as OpenAI's `tool_calls`; `tool_choice` and `response_format` (a JSON schema,
  parsed as a prefix) are shapes: a drawn token is taken if the text can still fit, else drawn again without it.
  Reasoning comes apart from the answer (`reasoning_content`).
- **HTTP**: `/v1/models`, `/v1/completions`, `/v1/chat/completions`, whole or streamed. A request holds a slot from its
  arrival to its end, and one that finds none free gets a 429 at once (OpenRouter tries the next provider, and does not
  count 429s against uptime). Usage counts the prompt's cached tokens, and ends a stream. A request with no text for 600
  s gets a 504.

## The cost model

`models/cost.py`: what a model costs to serve, from its config alone. A card holds the weights in a mode's formats, a
GiB for kernels and activations, and sessions (KV cache rows, Gated DeltaNet states). A decode step for n users reads
every matrix but the experts and the embedding, the experts its tokens pick (E (1 - (1 - k/E)^tokens) a block), the
states, and what attention's kernel reads (keys twice, values once); it waits on DRAM or the
matmuls, whichever is longer, then on its launches (`LAUNCHES`, counted on ttsim) or the RISC-V work ttsim counts
(`CONTROL`), whichever is more. A prompt's chunks read the experts their tokens pick and their attention's keys and
values for every eight tile rows of each query head. The card is its specification (512 GB/s at a 0.7 reach, a guess;
664 TFLOPS at one fidelity phase; a microsecond a launch) until the probe measures it (`probe=probe.json`).

For Qwen3.6 on one card (experts bfp4, KV cache bfp8, sessions of 32K tokens): 25 sessions; 93 tokens/s for one user,
237 a card at eight users, 273 at sixteen; a 32K-token prompt in 2.1 s. Decoding speculatively with 94% of drafts taken
(`accept=0.94`; it pays above 44% at one user and 55% at eight): 22 sessions, 125 tokens/s for one user, 61 each and 243
a card at four, 295 a card at eight.

## tinygrad on ferrite

`adapters/tinygrad_ferrite.py`: tinygrad's frontend on ferrite's compiler and runtime. The FERRITE device takes over at
`create_linear_with_vars`, where tinygrad would schedule kernels, and runs the graph of tinygrad's primitives op by op
into ferrite's lazy tensors, giving tinygrad an empty schedule. Movement stays a view (a small ShapeTracker), folded
into ferrite's three axes when an op needs the data; a broadcast multiply summed is a ferrite matmul; what ferrite lacks
(64-bit integers, bit operations, tinygrad's random numbers) runs lazily on tinygrad's CPU device. `TinyJit` is ferrite's
replay. PyTorch runs through tinygrad's own frontend (`extra/torch_backend`). Not on the chip yet: symbolic shapes,
dtypes past float32, bfloat16, int32 and bool.

## ttsim

ttsim runs each Tensix instruction as soon as it can, in zero time, ignores the busy conditions of `STALLWAIT`, and
answers NoC and DRAM requests at once. In one clock each tile steps its RISC-V cores once and runs what Tensix work can
go, so a kernel's cycles are the RISC-V instructions on its critical path: its numbers compare versions of our code,
they do not predict silicon. `pci.Sim.tick` runs 10 clocks a poll. It kills the process on any contract violation.

Breaking a wait on purpose shows what it checks: the pack's `SEMWAIT`, the math's Dst-half wait and the SEM_L1 wait stop
ttsim on a semaphore; BRISC's wait for `UNPACKED` and TRISC2's for `SENT` give wrong results. It cannot check the
`STALLWAIT` before a report, TRISC2's wait for `READY`, `tensix_idle` before it, or the accumulator semaphores: those are
silicon's to check.

**ttsim-chaos** (`test/ttsim-chaos.patch`, against ttsim v1.11.5 at 464814f), each part off unless set:
`TTSIM_CHAOS_NOC=lo,hi` carries out a Tensix tile's NoC request `lo` to `hi` clocks late (counted outstanding until
then; static-VC requests keep their order); `TTSIM_CHAOS_TENSIX=p` and `TTSIM_CHAOS_RISCV=p` make a pipe or a core wait
a turn with probability p; `TTSIM_CHAOS_SEED=n`; `TTSIM_CHAOS_UNHARVESTED=1` runs a p150's 140 Tensix, as firmware
before 19.5 does. It also models the NIU's read-data counter (counter 3, a flit for each 64 bytes a read brings in), and
exports `libttsim_chaos`, by which `pci.Sim` knows it does (`counts`). The suite passes under each; it caught missing
waits for a frame's reads and a matmul step's, and TRISC0 running 16 reports ahead.

## Ethernet links

`Chip.links`, `Chip.send`, `Tensor.to`, `link_kernel`; tested on ttsim's two-chip build (`bh_x2`, a P300's two chips,
`test_two_chips`). Sources: tt-isa-documentation `BlackholeA0/EthernetTile/` (its `ethdump` sample), tt-metal
`eth_fw_api.h`, `dev_mem_map.h`, `tunneling.h`, tt-umd's topology discovery.

- 14 Ethernet tiles on NoC0 y=1, each 512 KiB of L1, two RISC-V cores and one 400 GbE link; a p150 has 8 with ports
  (telemetry tag 35), translated x 24-31, y 25.
- E0 runs Tenstorrent's firmware (training, retraining): its L1's top 64 KiB, boot results at 0x7CC00 (port status 1
  when up), the peer's identity at 0x7CFE0. Never reset E0; E1 is free (held in reset by bit 0x1000 of `SOFT_RESET_0`).
- A transmit queue is registers at 0xFFB90000 + 0x1000 n: source, size and destination (16-byte aligned), then `CMD` 2,
  a write into the peer tile's L1; packets arrive once and in order, so a flag sent after the data lands after it. The
  host can write those registers over the NoC, so `link_kernel` copies DRAM to DRAM with no code on the Ethernet tiles:
  a worker each side, two slots in the tile's L1, a flag per chunk and a credit back.
- About 50 GB/s each way per link, 0.5 us a hop (tt-metal). umd calls the links flaky: poll their status, never wait
  without a timeout.

## Bring-up on a p150

Before: 300 W on one 12V-2x6 cable; `lspci -d 1e52:b140` shows the card, ideally at `LnkSta: Speed 32GT/s, Width x16`;
the IOMMU off or in passthrough (`iommu=pt`), since `pci.Card` hands the device physical addresses; a few 2 MiB
hugepages (`echo 4 > /proc/sys/vm/nr_hugepages`); root. `pci.Card` unbinds tt-kmd itself. Check the firmware with
tt-smi (it needs tt-kmd), and flash the latest bundle with tt-flash if older. The tests open the card when `TTSIM` is
not set. In order, each step needing only the ones before it:

| step | command | confirms (and what to suspect if it fails) |
|---|---|---|
| 0 | `python3 -m ferrite.chip` | ARC up, Tensix powered, the AI clock at 1350 (800 is idle), the Tensix columns, DRAM trained, every telemetry tag; if the NoC does not translate coordinates, `bh.py`'s are wrong for this card |
| 1 | `python3 -m pytest test -k "config_space or arc"` | BARs, TLB config, ARC's status, message queue and telemetry |
| 2 | `-k "dma or l1 or (dram and not freed)"` | the PCIe tile into host memory (bus addresses below 2^58), L1 writes, DRAM coordinates and training |
| 3 | `python3 -m ferrite.hello` | reset bits and PCs, the firmware, the NIU's initiators, device writes into host memory, the dispatcher's ring reads, posted status writes, multicasts and the workers' atomic increments. A hang with every worker done points at the count; one before any starts, at the multicast's acknowledgements |
| 4 | `-k "every_core or relaunch or moves_data or queue or hung"` | the launch path; instruction cache invalidation (a no-op on ttsim). A value wrong in `tolist` but right in `chip.read` points at the readback |
| 5 | `-k "tile_math or tensor_math"` | backend config indices (from ttsim, not the ISA), encodings, `.ttinsn`, semaphores, bf16 unpack and pack |
| 6 | `-k "matmul or stream or bfp"` | `MVMUL`, address modes, fidelity, replay, grid multicasts, bfp8, reads waited for by ID (`DEPTH = 1` if matmuls are wrong now and then) |
| 7 | `-k "sfpu or reduc or softmax or any_shape or views or batch or movement"` | run-time slots, BRISC's 2-byte L1 accesses, SFPU encodings, lane flags, `SFPTRANSP`, `Haloize_mode` |
| 8 | the whole suite, a few times | the waits ttsim never makes happen; a failure that comes and goes points here |
| 9 | `GPT2=... python3 -m pytest test -k gpt2`, then `models/gpt2.py` | everything at once, and the first timings (`Runtime.profile`; all 0 means `bh.WALL_CLOCK`) |
| 10 | `python3 -m ferrite.probe`, `python3 -m ferrite.bench` (and `1 768 3072`, `4 bfp8`) | launch cost, DRAM GB/s by request size and reads in flight (block-float tiles as the matmuls read them, a near and a far bank) and a read's latency, NoC latency, MVMUL, unpack, pack and SFPU rates, BRISC's topk and grouping, PCIe; a matmul's TFLOPS and decode's GB/s. `probe 8 probe.json` feeds the cost model (its bandwidth: the slower of bfp8's and bfp4's tiles as the matmuls read them) |
| 11 | `python3 -m ferrite.probe 256`, and a model of a few GB | uploads; if far under a GB/s, map the windows write-combined (`resource0_wc`) |
| 12 | `test/benchmark.py ttnn ...`, then `ferrite ...` | ferrite against TT-NN on the same card |

Rehearsed in order on ttsim-chaos. For a hang, read `Runtime.wait`'s report against "The pipeline": TRISC0 waits for
`FETCHED`, BRISC for `UNPACKED` and flags, TRISC2 for `SENT`, and a read ID that stays in flight waits on the NoC. A
hung kernel needs no reset (the next `Runtime` resets its cores); a hung NoC or ARC does (`python3 -m ferrite.chip
reset`, untested).

## Still unknown or untested

- `pci.Card` and `pci.reset` have not run on a card; whether ARC's power messages wake Tensix to 1,350 MHz after kmd
  gated it; whether window 0 reaches host memory without kmd's iATU; whether device writes into host memory and the
  dispatcher's ordered posted writes behave as on ttsim.
- The ISA does not document BAR2, the DRAM tile's address map, ARC's protocol or soft reset's power-on value; ttsim
  models them, silicon may differ. ttsim does not model `mhartid`, `fence.i`, the PC snapshots or request splitting,
  and its instruction cache invalidation is a no-op.
- The card's real rates: DRAM bandwidth a step reaches, launch cost, the RISC-V work's speed, one worker's read
  bandwidth, the best `DEPTH`. The cost model guesses all of them.
- **This host's CPU (an i7-12700KF) is unstable under load**: plain Python segfaults about one run in five, on some
  CPUs and not others (`journalctl -k` names the CPU). A run that dies or goes wrong under load may be the host: a
  whole-suite run that died; Qwen3.6's accuracy runs that came out NaN under full load (`gpt2.bfp` overflowing on
  weights that round cleanly on one core; one of three sound on a rerun, the bfp4 KV cache's not in three, one of them
  on the cores the crashes never name: the memory, more likely than a core); gpt-oss-20b's two bf16 runs far from
  numpy with heavy work beside them; one `test_train` readback under ttsim-chaos.
</content>
</invoke>
