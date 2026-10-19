# ferrite and TT-NN

The same ops through ferrite and through Tenstorrent's own stack (tt-metal with TT-NN), on the same simulator, counted
the same way. Measured 2026-10-19: ferrite at aa08e0f, tt-metal at 9f98dee9 (2026-10-17) with TT-NN in C++, ttsim
v1.11.5 (the version tt-metal names in `tt_metal/ttsim-version`), Blackhole. bf16 in tile layout in DRAM, matmuls at
full precision (HiFi4) in both; each op is one launch in ferrite and one program in TT-NN.

| op | ferrite | TT-NN, fast dispatch | TT-NN, slow dispatch | DRAM reads, ferrite | DRAM reads, TT-NN |
|---|---|---|---|---|---|
| matmul 512x768 @ 768x3072 | **1.5k cycles** | 7.8k | 3.3k | 1.00x the inputs | 1.00x |
| matmul 32x768 @ 768x3072 (a token) | **0.58k** | 6.9k | 1.8k | 1.00x | 1.00x |
| the same, weights in bfp8 | **1.5k** | 6.9k | 1.8k | 0.53x the bf16 bytes | 0.53x |
| softmax 64x768, along rows | **1.4k** | 4.5k | 3.7k | **1.00x** | 1.50x |
| layer norm 64x768, along rows | **0.44k** | 2.4k | 1.4k | **1.08x** | 1.17x |
| add 32x32 (one tile: launch cost) | **0.29k** | 5.3k | 0.8k | 1.00x | 1.00x |

ferrite does each op in 3 to 18 times fewer cycles than TT-NN with fast dispatch, which dispatches from cores on the
chip as ferrite does, and in 1.2 to 3 times fewer than TT-NN's kernels alone (slow dispatch, where the host does the
dispatching at no cost here). Both read each input from DRAM about once.

## How it was counted

- **Cycles**: the chip clocks the host asked ttsim for (`libttsim_clock`), averaged over three runs of the op queued
  back to back after a cached one. ferrite counts them itself (`pci.Sim`, 10 clocks a poll); for tt-metal, a shim in
  front of `libttsim.so` adds them up. tt-metal's driver (UMD) clocks once each time it reads the device, so in both
  the host's own work costs nothing and a cycle is the chip's.
- **DRAM reads**: ttsim's count of answered read requests (`NIU_MST_RD_RESP_RECEIVED`), summed over the workers. Both
  read whole 2 KiB tiles; a bfp8 tile is 1,088 bytes, one request in TT-NN and two in ferrite (its exponents, its
  values).
- **Cores**: ferrite uses 119 workers and a dispatcher of ttsim's 120 Tensix cores; TT-NN 11x10 = 110 workers.

## What carries over to a card

- **DRAM traffic does.** A byte read from DRAM is a byte on the card. On a card big matmuls and decode wait on DRAM,
  and there the two read the same bytes.
- **Dispatch does, in kind.** TT-NN with fast dispatch spends about 4.5k cycles of RISC-V work on an op (5.3k against
  0.8k for one tile); ferrite's whole one-tile op is 0.29k: arguments and go multicast to every worker, completion
  written to host memory. A card adds NoC latency to both.
- **Math time does not.** ttsim runs a queued Tensix instruction at no cost and counts a clock only while every pipe of
  a tile waits, one RISC-V instruction each (NOTES, "How ttsim counts"). So these cycles are each stack's control work
  around the math, not the math: HiFi2 costs what HiFi4 does here, though on a card it is half the multiplies. TT-NN
  multiplies at HiFi2 unless told; ferrite at HiFi4 unless told (`tensor.FIDELITY`).
- **The host's work is not counted here,** and on a card it counts. ferrite issuing a GPT-2 token op by op took the
  host 6.5 ms, while the card should need about 0.5 ms; replayed from a recording (`tensor.jit`), a GPT-2 or
  SmolLM-135M token takes the host 0.1 ms. TT-NN's traces replay too; they were not measured here.

Compiling (host seconds): ferrite builds a kernel with clang and lld in about 0.2 s; tt-metal builds its firmware when
the device opens (0.9 s) and each op's kernels the first time (0.6 to 0.8 s). Both cache on disk.

Size: ferrite is about 2,700 lines of Python. tt-metal's stack is about 1.65M lines of C++ and Python, not counting
tests: tt-metal 540k, LLK 111k (44k of it Blackhole's), TT-NN 935k, UMD 56k, tt-kmd 10k.

On 2026-10-18 (ferrite at bc715ac) it was the other way round for DRAM: ferrite's matmul read 6.9 times its inputs, its
softmax 3 and its layer norm 6, and the matmul took 17.0k cycles. Sharing tiles between workers by multicast and keeping
a line's inputs in L1 between passes closed that (NOTES, "Tiles shared between workers", "Lines in passes").

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
`~/Desktop/tt`, outside this repository. Run them one at a time: the shim counts clocks in one file in `/dev/shm`, so
two runs at once each count both.

    TTSIM=/path/to/libttsim.so python3 cmp/ferrite_ops.py
    TT_METAL_RUNTIME_ROOT=$PWD/tt-metal TT_METAL_SIMULATOR=$PWD/sim/libttsim.so TT_METAL_DISABLE_SFPLOADMACRO=1 cmp/build/ttnn_ops
    TT_METAL_SLOW_DISPATCH_MODE=1 ... cmp/build/ttnn_ops    # the same, slow dispatch
