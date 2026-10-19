# ferrite and TT-NN

The same ops, and the same LLM, through ferrite and through Tenstorrent's own stack (tt-metal with TT-NN), on the same
simulator, counted the same way. Measured 2026-10-19: ferrite at 9d20e4e, tt-metal at 9f98dee9 (2026-10-17) with TT-NN
in C++, ttsim v1.11.5 (the version tt-metal names in `tt_metal/ttsim-version`), Blackhole. bf16 in tile layout in DRAM,
matmuls at full precision (HiFi4) in both; each op is one launch in ferrite and one program in TT-NN.

| op | ferrite | TT-NN, fast dispatch | TT-NN, slow dispatch | DRAM reads, ferrite | DRAM reads, TT-NN |
|---|---|---|---|---|---|
| matmul 512x768 @ 768x3072 | **1.56k cycles** | 7.8k | 3.3k | 1.00x the inputs | 1.00x |
| matmul 32x768 @ 768x3072 (a token) | **0.58k** | 6.9k | 1.8k | 1.00x | 1.00x |
| the same, weights in bfp8 | **0.77k** | 6.9k | 1.8k | 0.53x the bf16 bytes | 0.53x |
| softmax 64x768, along rows | **1.22k** | 4.5k | 3.7k | **1.00x** | 1.50x |
| layer norm 64x768, along rows | **0.43k** | 2.4k | 1.4k | **1.08x** | 1.17x |
| add 32x32 (one tile: launch cost) | **0.30k** | 5.3k | 0.8k | 1.00x | 1.00x |

ferrite does each op in 4 to 18 times fewer cycles than TT-NN with fast dispatch, which dispatches from cores on the
chip as ferrite does, and in 2 to 3 times fewer than TT-NN's kernels alone (slow dispatch, where the host does the
dispatching at no cost here). ferrite reads each input from DRAM about once, TT-NN up to 1.5 times.

## End to end: SmolLM-135M

The whole model, on the same simulator: Tenstorrent's own LLM code (tt-metal's `models/tt_transformers`, built and run
as its `simple_text_demo` does, decode traced) against ferrite's (`test/llama.py`, decode replayed by `tensor.jit`).
Both prefill "The meaning of life is" (SmolLM's ids 504 2455 282 1029 314), then decode a token at a time from a KV
cache, and both write the same tokens, " a question that has been".

| | ferrite, bf16 weights | ferrite, bfp8 weights | TT `tt_transformers` |
|---|---|---|---|
| a decode token, replayed | **180k cycles** | **199k** | 1,602k (performance or accuracy mode) |
| the prompt's prefill, 5 tokens | **224k** | **243k** | 2,108k (performance; 3,064k the first time) |
| a decode step for 32 users, a token each | **564k** | **582k** | 1,829k (performance) |
| a decode token at place 1000 (a cache of 1024) | **233k** | **251k** | 1,602k (performance) |
| launches a decode token | **443** | **443** | about 1,360 TT-NN operations, each one or more programs |
| weights read a decode token | 269 MB | 143 MB | 116 MB (performance), 168 MB (accuracy) |
| ... that at a p150's 512 GB/s | 0.53 ms | 0.28 ms | 0.23 ms, 0.33 ms |

ferrite decodes a token in 8 to 9 times fewer simulated cycles than TT's own model code, and prefills in 9 times
fewer, with a third as many launches. TT's two precision modes are within 0.1% of each other here: performance mode
keeps the MLP's up and gate weights in bfp4 at LoFi and the rest in bfp8 at HiFi2, accuracy mode attention and the KV
cache in bf16 at HiFi4 and the MLP in bfp8; the output layer is bfp8 in both. ttsim counts the RISC-V work around the
math, which those settings barely change (ferrite's bfp8 costs it 10%, two reads a tile). TT's operations a token are
82 for one block and 126 for two (counted untraced), so 44 a block where ferrite has 14.

At place 1000 TT's token costs what it does at place 5, and ferrite's 53k more: on ttsim, where tile work is free, a
longer cache costs TT's attention no more control, while ferrite's grows with it (the attention kernel, one line of 32
tiles a head on 9 workers, is 46k of the 233k, and the cache's put 36k, against 22k at 32 places). At that rate ferrite
would reach TT's 1.6M near 27,000 places, far past SmolLM's 2,048.

32 users decoding together (all from the same prompt, each at its own places; `llama.forward`'s starts, TT's batch)
take ferrite 3.2 times fewer cycles a step than TT, 17.6k a token against 57k. Batching gains TT more: it computes 32
rows for one user anyway, so 32 users cost it 14% more than one. Its attention is per user, where ferrite's runs over
every user's cache, masked: no more tile work, but one line of work a head, 9 lines for 119 workers (NOTES, "A
Llama-family model").

What it could mean on a card: if silicon's RISC-V control work is anything like ttsim's count (in-order cores, about
an instruction a clock at 1.35 GHz), TT's 1.6M cycles are 1.2 ms of control a token, five times its 0.23 ms of DRAM
in performance mode, while ferrite's 0.13 to 0.15 ms are under its DRAM time (0.28 ms in bfp8). Then ferrite decodes
as fast as its weights can be read and TT does not. ttsim's NoC and DRAM answer at once, so a card adds latency to
both, and only a card can say how far apart they are; SmolLM-135M is small, so control is a large share of its time,
and in a bigger model DRAM is a larger share for both. Both replay a decode step, so the host's part is small: ferrite's
is 0.1 to 0.2 ms of Python a token (measured), TT's trace was not measured. At 32 users the same reckoning gives
ferrite 0.42 to 0.43 ms of control a step against its weights' 0.28 (bfp8) or 0.53 ms (bf16), and TT 1.36 ms against
0.23 ms: about 60 to 75 thousand tokens a second against 23 thousand, if control is what ttsim counts.

### A bigger one: SmolLM2-360M

The same two programs on SmolLM2-360M (32 blocks 960 wide, 15 query heads sharing 5 key and value heads, Hugging Face's
`HuggingFaceTB/SmolLM2-360M`), which ferrite runs unchanged: both write the same tokens, " to be happy".

| | ferrite, bf16 weights | ferrite, bfp8 | ferrite, bfp4 MLP (TT's layout) | TT `tt_transformers` |
|---|---|---|---|---|
| a decode token, replayed | **193k cycles** | **257k** | **257k** | 1,873k (performance) |
| the prompt's prefill, 5 tokens | **240k** | **300k** | **301k** | 2,386k |
| weights read a decode token | 724 MB | 384 MB | 306 MB | 306 MB (performance), 458 MB (accuracy) |
| ... that at a p150's 512 GB/s | 1.41 ms | 0.75 ms | 0.60 ms | 0.60 ms, 0.89 ms |

ferrite takes 7 to 10 times fewer cycles here too. What the cycles hide shows here: ttsim's count grows with launches,
not with weights (ferrite's 193k against 180k for a model 2.7 times the size), while on a card a decode token's time
grows with the bytes it reads. By the reckoning above, ferrite in bf16 would take 1.41 ms a token, its weights' time,
about what TT's 1.87M cycles of control come to (1.39 ms); in bfp8 0.75 ms, 1.85 times faster; and with the MLP's
gate and up in bfp4, as TT's performance mode keeps them (rounded to nearest, as TT's host conversion rounds; NOTES,
"Weights in BFP8 and BFP4"), it reads what TT reads, 0.60 ms, 2.3 times faster, and writes the same tokens.

### Where DRAM decides: SmolLM2-1.7B

SmolLM2-1.7B (24 blocks 2048 wide, 32 heads, an MLP of 8192, `HuggingFaceTB/SmolLM2-1.7B`), which ferrite also runs
unchanged: all three write the same tokens, " to be happy.".

| | ferrite, bf16 weights | ferrite, bfp4 MLP (TT's layout) | TT `tt_transformers` |
|---|---|---|---|
| a decode token, replayed | **226k cycles** | **442k** | 2,129k (performance) |
| the prompt's prefill, 5 tokens | **261k** | **469k** | 2,275k |
| weights read a decode token | 3,423 MB | 1,416 MB | 1,416 MB (performance), 2,196 MB (accuracy) |
| ... that at a p150's 512 GB/s | 6.68 ms | 2.76 ms | 2.76 ms, 4.29 ms |

The cycles stop deciding it here. TT's 2.13M cycles of control come to 1.58 ms a token, under its weights' 2.76 ms, so
on a card both would wait on DRAM, reading the same bytes in TT's layout: a 1.7B model's speed is how close each comes
to the card's bandwidth, which only a card can measure. (Block floats cost ferrite twice the cycles of bf16 here: two
reads a tile, its exponents and its values, and at 1.7B the weights' reads are most of a token's control.) ferrite's
lead is in the work around the math, so it is largest where that work is: small models, a few users, and the host.

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
  0.8k for one tile); ferrite's whole one-tile op is 0.30k: arguments and go multicast to every worker, completion
  written to host memory. A card adds NoC latency to both.
- **Math time does not.** ttsim runs a queued Tensix instruction at no cost and counts a clock only while every pipe of
  a tile waits, one RISC-V instruction each (NOTES, "How ttsim counts"). So these cycles are each stack's control work
  around the math, not the math: HiFi2 costs what HiFi4 does here, though on a card it is half the multiplies. TT-NN
  multiplies at HiFi2 unless told; ferrite at HiFi4 unless told (`tensor.FIDELITY`).
- **The host's work is not counted here,** and on a card it counts. ferrite issuing a GPT-2 token op by op took the
  host 7.3 ms, while the card should need about 0.5 ms; replayed from a recording (`tensor.jit`), a GPT-2 or
  SmolLM-135M token takes the host 0.1 ms. TT-NN's traces replay too; they were not measured here.

Compiling (host seconds): ferrite builds a kernel with clang and lld in about 0.2 s; tt-metal builds its firmware when
the device opens (0.9 s) and each op's kernels the first time (0.6 to 0.8 s). Both cache on disk.

Size: ferrite is about 2,800 lines of Python. tt-metal's stack is about 1.65M lines of C++ and Python, not counting
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

The shim (`sim/shim.c`, built to `sim/libttsim.so` beside a copy of `blackhole_140_arch.yaml` as `soc_descriptor.yaml`)
and the measurement programs (`cmp/ttnn_ops.cpp`, `cmp/ferrite_ops.py`, and for SmolLM `cmp/ttnn_smollm.py`,
`cmp/ferrite_smollm.py`) are in `~/Desktop/tt`, outside this repository. Run them one at a time: the shim counts clocks
in one file in `/dev/shm`, so two runs at once each count both.

    TTSIM=/path/to/libttsim.so python3 cmp/ferrite_ops.py
    TT_METAL_RUNTIME_ROOT=$PWD/tt-metal TT_METAL_SIMULATOR=$PWD/sim/libttsim.so TT_METAL_DISABLE_SFPLOADMACRO=1 cmp/build/ttnn_ops
    TT_METAL_SLOW_DISPATCH_MODE=1 ... cmp/build/ttnn_ops    # the same, slow dispatch

SmolLM-135M through TT's Python API needs `tools` on the path (its `tracy` module), the token ids given rather than
TT's tokenizer (which fails on this model: no `<unk>` in its vocabulary), and the prefill warmup off (it runs TT's
on-device sampling, which uses an SFPU instruction ttsim does not model, `SFPCONFIG` mode 8):

    TTSIM=/path/to/libttsim.so python3 cmp/ferrite_smollm.py bf16 4    # or bfp8; a third argument, 32, for 32 users
    HF_MODEL=/path/to/SmolLM-135M TT_METAL_RUNTIME_ROOT=$PWD/tt-metal TT_METAL_SIMULATOR=$PWD/sim/libttsim.so \
      TT_METAL_DISABLE_SFPLOADMACRO=1 LD_LIBRARY_PATH=$PWD/tt-metal/build_Release/lib \
      PYTHONPATH=$PWD/tt-metal:$PWD/tt-metal/ttnn:$PWD/tt-metal/tools python3 cmp/ttnn_smollm.py performance 4    # or accuracy
    ... cmp/ttnn_smollm.py performance 3 all 32    # 32 users: decode alone (TT prefills users one at a time)
    START=1000 ... cmp/ttnn_smollm.py performance 3    # from place 1000, the cache empty; ferrite_smollm.py too
    COUNT_OPS=1 ... cmp/ttnn_smollm.py performance 2 1    # TT-NN operations a token, untraced, one block (2 for two)
