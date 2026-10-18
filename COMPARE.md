# ferrite and tt-metal on ttsim

The same ops through ferrite and through Tenstorrent's own stack (tt-metal with TT-NN), on the same simulator, counted
the same way. ttsim runs each Tensix instruction at once and its NoC and DRAM answer at once, so its cycles are not
silicon time; what it does count fairly is what the RISC-V cores do and how many requests go over the NoC. Measured
2026-10-18: ferrite at bc715ac, tt-metal at 9f98dee9 (2026-10-17) with TT-NN in C++, ttsim v1.11.5 (the version that
tt-metal names in `tt_metal/ttsim-version`), Blackhole.

## How it was counted

- **Cycles**: the chip clocks the host asked ttsim for (`libttsim_clock`), averaged over three runs of the op queued
  back to back after a cached one. ferrite counts them itself (`pci.Sim`, which runs ttsim 10 clocks a poll; this
  table first had ferrite's numbers polled every 1000 and rounded up); for tt-metal a shim in front of `libttsim.so`
  adds them up. tt-metal's driver (UMD) clocks once each time it reads the device, so in both stacks
  the host's own work costs nothing and a cycle is device work.
- **Dispatch**: ferrite has one way, its dispatcher core. tt-metal has two: fast dispatch, where cores on the chip
  dispatch as ferrite's does, and slow dispatch, where the host writes every kernel, argument and go itself (at no
  cost in cycles here), which ttsim recommends. Fast dispatch is the like-for-like number; slow dispatch is the
  kernels alone.
- **DRAM reads**: ttsim keeps one NoC counter per core per NoC, read requests (`NIU_MST_RD_RESP_RECEIVED`), summed
  over the worker cores before and after. ferrite reads whole tiles, and TT-NN's matmul makes exactly one request per
  input tile, so a request is a 2 KiB tile.
- **Cores**: ferrite uses 119 workers and a dispatcher of ttsim's 120 Tensix cores; TT-NN uses 11x10 = 110 workers.
- bf16, tile layout, in DRAM. Each op made one program in TT-NN (counted from its program cache) and one launch in
  ferrite.

## Results

| op | ferrite | TT-NN, fast dispatch | TT-NN, slow dispatch | DRAM reads, ferrite | DRAM reads, TT-NN |
|---|---|---|---|---|---|
| matmul 512x768 @ 768x3072 | 17.0k cycles (HiFi4) | 7.9k (HiFi2, its default); 7.8k (HiFi4) | 3.4k; 3.3k | 18,432 tiles, 6.86x the inputs | 2,688 tiles, 1.00x |
| softmax 64x768, along rows | 1.4k | 4.5k | 3.7k | 144 tiles, 3.00x | 72 tiles, 1.50x |
| layer norm 64x768, along rows | 2.5k | 2.4k | 1.4k | 288 tiles, 6.00x | 56 tiles, 1.17x |
| add 32x32 (one tile: launch cost) | 0.4k | 5.3k | 0.8k | 2 tiles | 2 tiles |

Compiling, on this machine (host seconds, not cycles): ferrite builds a kernel with clang and lld in about 0.2 s.
tt-metal builds its firmware when the device opens (0.9 s) and each op's kernels the first time (matmul 0.7 s,
softmax 0.6 s, layer norm 0.8 s); its cache held 115 MB after these ops. Both cache on disk, and a cached run
compiles nothing.

Size: ferrite is about 2,200 lines of Python. tt-metal's stack is about 1.65M lines of C++ and Python not counting
tests: tt-metal 540k, LLK 111k (44k of it Blackhole's), TT-NN 935k, UMD 56k, tt-kmd 10k.

Since then: ferrite's matmuls share their tiles between workers by multicast (NOTES, "Tiles shared between workers"):
the matmul above reads 2,688 tiles, 1.00x its inputs, as TT-NN's does, each worker reading an equal share, in 1.85k
cycles (NOTES, "Cutting the critical path"). A line's inputs stay in L1 between passes (NOTES, "Lines in passes"): the
softmax reads 48 tiles, 1.00x, in 1.4k cycles, the layer norm 54, 1.12x (its constants, once a line), in 0.46k. A
one-tile launch is 0.36k. On ttsim these are RISC-V instructions on the critical path, not silicon time.

## What carries over to silicon

- **DRAM traffic does.** A byte read from DRAM is a byte on the card too. TT-NN's matmul reads every input tile once:
  one core reads it from DRAM and multicasts it to the others that need it. ferrite keeps A's rows and B's column in
  L1 within a core, but every core reads its own copies, 6.9x the inputs (NOTES, "Next for the compiler", item 3).
  TT-NN's softmax and layer norm read each row about once (1.5x and 1.17x), where ferrite's passes read x again (3x
  and 6x; item 4). On silicon, where DRAM bandwidth bounds big matmuls and decode, this is TT-NN's largest lead.
- **Launch work does, in kind.** With dispatch on the chip, TT-NN spends about 4.5k cycles of RISC-V work an op
  (fast against slow dispatch: 5.3k against 0.8k for one tile, 7.9k against 3.4k for the matmul). ferrite's whole
  one-tile op, launch included, is 0.4k: arguments and go multicast to every worker, completion written to host
  memory. Silicon adds NoC latency to both. tt-metal's traces replay commands from DRAM instead of the host; they
  were not measured here.
- **Math time does not.** ttsim runs a queued Tensix instruction at no cost, however it got into the FIFO (pushed,
  replayed or expanded from a MOP), and counts a clock only while a tile's pipes all wait, one RISC-V instruction each
  (NOTES, "How ttsim counts"). So HiFi2 costs what HiFi4 does (3.4k against 3.3k) though HiFi4 is twice the MVMUL
  passes, and the gap between the two matmuls is in their RISC-V code: profiled, ferrite's idle clocks are the cores
  waiting on each other (the unpacker polling BRISC's fetches, BRISC waiting for flags and reports, the packer for
  BRISC to send), not the math thread, which is already ahead. On the card, MVMULs take their real time, and the size
  of either stack's overhead against them waits for silicon.
- **Defaults differ.** TT-NN multiplies bf16 at HiFi2 with the packer accumulating in L1; ferrite always at HiFi4,
  full precision. A fair silicon comparison runs both at HiFi4, or ferrite at HiFi2 too.

So on ttsim: ferrite launches about 13x cheaper and compiles about 3x faster, in about 1/700 of the code; TT-NN reads
DRAM 2 to 7x less and spends far fewer RISC-V instructions waiting between its cores. Of ferrite's "Next for the
compiler" items, these numbers ranked first multicast matmuls (3) and rows kept in L1 between passes (4), both done
since; the MOP expander (1) turned out not to be what the gap is.

## Reproducing

tt-metal on Arch Linux needed `hwloc` and `libc++` (pacman), SFPI 7.88.0's Debian tarball (there is none for Arch;
its sha256 checks), and three local fixes for clang and libc++ 22: `#include <cstdlib>` in the bundled fmt's
`format.h`, a `const` dropped from a map key in `tt_metal/fabric/mesh_graph.cpp`, and the system `ar` and `ranlib`
for CMake. Then, from `~/Desktop/tt`, with a toolchain file for the system's clang 22 and libc++:

    cmake -S tt-metal -B tt-metal/build_Release -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=clang22-libcxx.cmake \
      -DENABLE_DISTRIBUTED=OFF -DBUILD_PROGRAMMING_EXAMPLES=ON -DCMAKE_COMPILE_WARNING_AS_ERROR=OFF
    ninja -C tt-metal/build_Release install

The shim (`sim/shim.c`, built to `sim/libttsim.so` beside a copy of `blackhole_140_arch.yaml` as
`soc_descriptor.yaml`) and both measurement programs (`cmp/ttnn_ops.cpp`, `cmp/ferrite_ops.py`) are in
`~/Desktop/tt`, outside this repository:

    TTSIM=/path/to/libttsim.so python3 cmp/ferrite_ops.py
    TT_METAL_HOME=$PWD/tt-metal TT_METAL_SIMULATOR=$PWD/sim/libttsim.so TT_METAL_DISABLE_SFPLOADMACRO=1 cmp/build/ttnn_ops
    TT_METAL_SLOW_DISPATCH_MODE=1 ... cmp/build/ttnn_ops    # the same, slow dispatch
