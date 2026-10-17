# trickle

trickle is Tenstorrent's software done right: the whole stack for Blackhole, from PCIe up to a compiler, small enough to read. Frameworks (tinygrad, PyTorch) plug in on top through thin adapters; they are not part of trickle.

## Design

From George Hotz's "Advice to Tenstorrent" (github.com/geohot/tt-tiny): the stack is "3 (and only 2 you have to build)".

1. Compiler: a compiler built for Tensix's tiles and dataflow, emitting Tensix instructions directly from the ISA docs. No LLK.
2. Runtime and driver: talk to the chip over PCIe from userspace. A runtime that is just a runtime: memory, DMA, loading, launching, queues.

Nothing from Tenstorrent runs at runtime except the firmware on the card: no tt-kmd, tt-umd, tt-metal or LLK.

## Taste

- Small. Every line earns its place. Delete before you add.
- Flat. Few files, no class hierarchies, no config systems, no abstraction until a second user needs it.
- Readable. The code is the documentation; comments say why, not what.
- Python with ctypes. No dependencies beyond the standard library, except for tests.
- Blackhole only, with its chip-specific facts kept as data in one place (`trickle/bh.py`).
- Every wait has a timeout: a hung chip is reported, never waited on forever.
- Every step has a test that runs on the simulator.

## Layout

- `trickle/bh.py`: Blackhole facts as data. NOTES.md says where each one comes from.
- `trickle/pci.py`: the device interface (config space, BARs, DMA memory, `tick`) with two backends: `Sim` (libttsim.so) and `Card` (sysfs).
- `trickle/chip.py`: NoC reads and writes (unicast and multicast) through a shared TLB window, dedicated TLBs (`map`), waiting with a timeout, ARC messages and telemetry, core reset.
- `trickle/runtime.py`: compiling device C with clang + lld (with the device NoC API prepended), kernels as a program per core (`compile_kernel`), firmware for a worker's BRISC and TRISCs and for the dispatcher, and the command ring in host memory: a launch is one PCIe write, completion is read from host memory.
- `trickle/compiler.py`: Tensix instruction assembler (encodings and config fields as data in `bh.py`), per-tile programs (elementwise, matmul with MVMUL), kernels as four programs that stream tiles through the three Tensix threads (BRISC moves tiles; each TRISC feeds one thread, unpack, math or pack, with `.ttinsn` instructions), SFPU programs (relu, exp, reciprocal, rsqrt), row reductions, broadcasting by tile reuse, fusion of whole op trees into one kernel with intermediates in L1 (rows swept in passes when they reduce, so softmax and layer norm are one launch each; too-big trees split), transpose, the tile layout, and `Tensor`: bf16 tensors of any shape in DRAM (freed when dropped) with numpy broadcasting, tile-aligned views, lazy `+`, `-`, `*`, `/`, `@`, `relu`, `exp`, `reciprocal`, `rsqrt`, `gelu`, `rowsum`, `rowmax`, `.T`, and numbers or `Tensor.full` for constants, on every worker. `frombytes`/`tobytes` for bf16 in bulk, and `gather` for embeddings. Enough for GPT-2 small (`test/gpt2.py`).
- `trickle/hello.py`: fib(n) on every Tensix core, the one-command demo.
- `test/test_sim.py`: every step, on ttsim (or on a card when `TTSIM` is not set).
- `test/gpt2.py`: GPT-2 small (written once for trickle Tensors and numpy), and greedy generation from the real weights, checked against numpy step by step.
- `NOTES.md`: what we learned about the chip, and what is still unknown.

## Running

Needs clang and lld on PATH, and ttsim's `libttsim.so` built for Blackhole (outside this repo: `./make.py src/_out/release_bh/libttsim.so` in tenstorrent/ttsim).

    TTSIM=/path/to/libttsim.so python3 -m trickle.hello
    TTSIM=/path/to/libttsim.so python3 -m pytest test
    TTSIM=/path/to/libttsim.so python3 test/gpt2.py path/to/model.safetensors "The meaning of life is" 8

The tests need numpy; `test/gpt2.py` also tiktoken and the weights (Hugging Face `openai-community/gpt2`, `model.safetensors`), and `GPT2=path/to/model.safetensors` adds the whole model to the tests. Without `TTSIM`, `pci.open_device()` looks for a real Blackhole on the PCI bus.

The simulator is a process-wide singleton that kills the process on any contract violation (bad address, bad TLB config, a semaphore taken while empty): a test run that stops early means the code did something the hardware would not like. pytest hides ttsim's `ERROR:` line; run with `-s` to see it.
