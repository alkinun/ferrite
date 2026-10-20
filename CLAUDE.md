# ferrite

ferrite is Tenstorrent's software done right: the whole stack for Blackhole, from PCIe up to a compiler, small enough to read. tinygrad plugs in on top through a thin adapter, and PyTorch through tinygrad's own frontend; neither is part of ferrite.

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
- Blackhole only, with its chip-specific facts kept as data in one place (`ferrite/bh.py`).
- Every wait has a timeout: a hung chip is reported, never waited on forever.
- Every step has a test that runs on the simulator.

## Layout

- `ferrite/bh.py`: Blackhole's facts as data: addresses, sizes and limits, instruction and config encodings (an instruction's modifier values stay where the compiler uses them). NOTES.md says where each comes from.
- `ferrite/pci.py`: the device interface (config space, BARs, DMA memory, `tick`), with two backends: `Sim` (libttsim.so) and `Card` (sysfs), and a whole-chip reset for a card that hangs.
- `ferrite/chip.py`: NoC reads and writes through TLB windows (unicast and multicast), waiting with a timeout, ARC messages and telemetry, core reset; `python3 -m ferrite.chip` prints what a card says about itself (`reset`: after resetting it).
- `ferrite/runtime.py`: device C compiled with clang and lld (cached in `~/.cache/ferrite`, for a week unused), firmware for the workers and the dispatcher, kernels loaded once and kept until their room is full, the command ring in host memory (a launch, or a recorded block of them, is one PCIe write), DRAM, waiting, and each launch's time on the chip's clock (`profile`).
- `ferrite/compiler.py`: Tensix instructions from the ISA. A fused kernel is four programs (BRISC moves tiles; each TRISC feeds one Tensix thread: unpack, math or pack): elementwise ops and SFPU programs, matmuls (blocks, tiles shared between workers by multicast, reads in flight, bfp8 and bfp4 weights), reductions along any axis in passes, movement, `gather` and `put`.
- `ferrite/tensor.py`: `Tensor`: lazy, any shape, bf16, fp32, int32, bfp8 or bfp4, numpy's broadcasting, views, tinygrad's ops and autograd. Realizing turns a tree of ops into as few kernels as fit; `jit` records a function's launches and replays them.
- `ferrite/hello.py`: fib(n) on every worker, the one-command demo. `ferrite/bench.py`: a matmul's speed and DRAM reads (ROADMAP, Stage 1). `ferrite/probe.py`: the numbers ttsim cannot give (launch cost, DRAM GB/s by reads in flight, NoC latency, MVMUL, unpack, pack and SFPU rates, PCIe), for the card's first day.
- `adapters/tinygrad_ferrite.py`: tinygrad on ferrite, outside it: the FERRITE device (tinygrad's tensors of any rank as views of ferrite's), and a runner for tinygrad's scripts unchanged. PyTorch runs through tinygrad's own frontend (`extra/torch_backend` in tinygrad), not an adapter of ours.
- `hw/mac.py`, `hw/mac.v`, `hw/mac_tb.py`: the first hardware (ROADMAP, Stage 1): a multiply-accumulate unit, bf16 by bf16 into fp32. The model in integers, the Verilog, and the cocotb testbench that holds one to the other bit for bit.
- `test/ttsim-chaos.patch`: ttsim with NoC latency and core jitter (NOTES, "ttsim-chaos"), for the missing waits plain ttsim hides.
- `test/test_sim.py`: every step, on ttsim (or on a card when `TTSIM` is not set). `test/gpt2.py`, `test/llama.py`: GPT-2 small and SmolLM-135M (and SmolLM2; its loader reads Llama 3.1's checkpoints too), written once for ferrite Tensors and numpy, generating from the real weights, checked against numpy step by step. `test/train.py`: a small GPT trained on ferrite in fp32 with Adam, checked against PyTorch. `test/benchmark.py`: Stage 1's benchmarks (ROADMAP: B1, B2, B3) on ferrite or on TT-NN (its B2 and B3 by Tenstorrent's own demo script), with the versions and lines of code that took part, into `benchmark-<stack>.json`.
- `NOTES.md`: how ferrite works, and what is still unknown. `LESSONS.md`: what the work has taught us; add to it as we go, and keep it short and true. `ROADMAP.md`: the five gated stages from here to a chip of our own. `COMPARE.md`: ferrite against TT-NN.

## Running

Needs clang and lld on PATH, and ttsim's `libttsim.so` built for Blackhole (outside this repo: `./make.py src/_out/release_bh/libttsim.so` in tenstorrent/ttsim).

    TTSIM=/path/to/libttsim.so python3 -m ferrite.chip
    TTSIM=/path/to/libttsim.so python3 -m ferrite.hello
    TTSIM=/path/to/libttsim.so python3 -m pytest test
    TTSIM=/path/to/libttsim.so python3 -m ferrite.bench
    TTSIM=/path/to/libttsim.so python3 -m ferrite.probe
    TTSIM=/path/to/libttsim.so python3 test/gpt2.py path/to/model.safetensors "The meaning of life is" 8
    TTSIM=/path/to/libttsim.so python3 test/llama.py path/to/SmolLM-135M "The meaning of life is" 6
    TTSIM=/path/to/libttsim.so python3 test/train.py 300
    TTSIM=/path/to/libttsim.so BOOK=path/to/tale-of-two-cities.txt.bz2 python3 test/benchmark.py ferrite path/to/SmolLM-135M path/to/SmolLM-135M.refpt --quick
    TTSIM=/path/to/libttsim.so PYTHONPATH=path/to/tinygrad python3 adapters/tinygrad_ferrite.py path/to/tinygrad/examples/transformer.py

The tests need numpy; `test/gpt2.py` and `test/llama.py` also tiktoken; `test/gpt2.py` the weights (Hugging Face `openai-community/gpt2`, `model.safetensors`), and `GPT2=path/to/model.safetensors` adds the whole model to the tests; `test/llama.py` wants Hugging Face's `HuggingFaceTB/SmolLM-135M` folder (config.json, tokenizer.json, model.safetensors), and `LLAMA=path/to/folder` adds it to the tests; `test/train.py` and `test_train` want PyTorch, as the reference; `test_tinygrad` wants tinygrad (`TINYGRAD=path/to/tinygrad`, or installed), and the adapter numpy; `test_torch_frontend` wants tinygrad and PyTorch (tinygrad's torch frontend builds a small C++ extension the first time, with the system's C++ compiler and ninja). `test/benchmark.py` wants tiktoken and PyTorch (to read the reference file), and `cloc` to count lines; its `ttnn` side runs in tt-metal's Python environment (`TT_METAL_HOME`). `test_mac_hardware` wants Verilator and cocotb. `FUZZ=n` runs n of `test_fuzz`'s expressions rather than 30. Without `TTSIM`, `pci.open_device()` looks for a real Blackhole on the PCI bus.

The simulator is a process-wide singleton that kills the process on any contract violation (bad address, bad TLB config, a semaphore taken while empty): a test run that stops early means the code did something the hardware would not like. pytest hides ttsim's `ERROR:` line; run with `-s` to see it.
