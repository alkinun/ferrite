# ferrite

ferrite is Tenstorrent's software done right: the whole stack for Blackhole, from PCIe up to a compiler, small enough to read. Frameworks (tinygrad, PyTorch) plug in on top through thin adapters; they are not part of ferrite.

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

- `ferrite/bh.py`: Blackhole facts as data. NOTES.md says where each one comes from.
- `ferrite/pci.py`: the device interface (config space, BARs, DMA memory, `tick`) with two backends: `Sim` (libttsim.so) and `Card` (sysfs).
- `ferrite/chip.py`: NoC reads and writes (unicast and multicast) through a shared TLB window, dedicated TLBs (`map`), waiting with a timeout, ARC messages and telemetry, core reset.
- `ferrite/runtime.py`: compiling device C with clang + lld (with the device NoC API prepended; images cached in `~/.cache/ferrite`), kernels as a program per core (`compile_kernel`), firmware for a worker's BRISC and TRISCs and for the dispatcher, and the command ring in host memory: a launch is one PCIe write, completion is read from host memory.
- `ferrite/compiler.py`: Tensix instruction assembler (encodings and config fields as data in `bh.py`), per-tile programs (elementwise, matmul with MVMUL), kernels as four programs that stream tiles through the three Tensix threads (BRISC moves tiles; each TRISC feeds one thread, unpack, math or pack, with `.ttinsn` instructions), SFPU programs for tinygrad's elementwise primitives (exp2, log2, sin, sqrt, reciprocal; max, cmplt, cmpne; where as two exact picks and an add) and for exp, log, cos and relu, reductions along any axis (the batch, rows or columns), broadcasting by tile reuse, fusion of whole op trees into one kernel with intermediates in L1 (lines of tiles along the reduced axis swept in passes, so softmax and layer norm are one launch each; too-big trees split), transpose, movement at any granularity (reshape, permute, expand, pad, slices at any step: one map per move, copied by a BRISC kernel), the tile layout, and matmuls in blocks with A's rows and B's column kept in L1.
- `ferrite/tensor.py`: `Tensor`, which builds the trees `compiler.py` turns into kernels, and autograd (tinygrad's rules, `backward`, `gradient`, `requires_grad_`): bf16, fp32 or int32 tensors (fp32 and int32 unpacked straight into Dst and computed on the SFPU; casts, and matmuls written as fp32 or of fp32 as bf16x3) of any shape, batched or not, in DRAM (freed when dropped) with numpy broadcasting, tile-aligned views (`x[b]`, slices of whole tiles, `split` for heads and `merge` back, results written into views), split K for matmuls with few result tiles, lazy `+`, `-`, `*`, `/`, `@`, comparisons, `where`, `maximum`, `minimum`, `exp`, `exp2`, `log`, `log2`, `sin`, `cos`, `sqrt`, `rsqrt`, `reciprocal`, `relu`, `gelu`, `sum(axis)`, `max(axis)`, `mean`, `softmax(axis)`, `log_softmax(axis)`, `layernorm(axis)`, `sigmoid`, `silu`, `.T`, `reshape`, `permute`, `expand`, `pad`, `cat` (a chain of moves composed into one), `conv2d` (im2col) and `max_pool2d`, `assign` (an optimizer's update of a weight, in place), and numbers or `Tensor.full` for constants, on every worker. `frombytes`/`tobytes` in bulk, and `gather` for embeddings, by an int32 tensor of indices on the chip. Enough for GPT-2 small (`test/gpt2.py`), SmolLM-135M (`test/llama.py`) and a small CNN (`test_mnist_cnn`), and for training a small GPT (`test/train.py`).
- `ferrite/hello.py`: fib(n) on every Tensix core, the one-command demo.
- `ferrite/bench.py`: a matmul's speed (ROADMAP, stage 3): its time and TFLOPS (simulated cycles on ttsim).
- `test/test_sim.py`: every step, on ttsim (or on a card when `TTSIM` is not set).
- `test/llama.py`: a Llama-family model, SmolLM-135M (RMSNorm, RoPE, SwiGLU, grouped-query attention, a KV cache), the same way: written once, generating from the real weights, checked against numpy step by step.
- `test/gpt2.py`: GPT-2 small (written once for ferrite Tensors and numpy), and greedy generation from the real weights, checked against numpy step by step.
- `adapters/views.py`: ferrite tensors of any rank for the framework adapters: views (shape, strides, offset, a mask) of a ferrite tensor, folded into its three axes when an op needs the data (as it is, a transpose, one move, or block swaps), elementwise operands, reductions over any axes.
- `adapters/tinygrad_ferrite.py`: ferrite as tinygrad's FERRITE device, outside ferrite: tinygrad's tensor graph, taken where tinygrad would schedule it, run op by op on ferrite (movement as views folded into three axes, a broadcast multiply summed as a matmul, what ferrite lacks on tinygrad's CPU), and a runner for tinygrad scripts unchanged.
- `test/train.py`: a small GPT (GPT-2's blocks) trained on ferrite in fp32 with Adam: each step's loss and gradients checked against PyTorch's on the same weights, and the loss curve against PyTorch training by itself.
- `NOTES.md`: what we learned about the chip, and what is still unknown.
- `LESSONS.md`: what the work has taught us, about the chip, the simulator, the compiler and how we work, and what a chip of our own should do differently. Add to it as we go, and keep it short and true.
- `ROADMAP.md`: the stages from here, each with its steps and what done means.

## Running

Needs clang and lld on PATH, and ttsim's `libttsim.so` built for Blackhole (outside this repo: `./make.py src/_out/release_bh/libttsim.so` in tenstorrent/ttsim).

    TTSIM=/path/to/libttsim.so python3 -m ferrite.hello
    TTSIM=/path/to/libttsim.so python3 -m pytest test
    TTSIM=/path/to/libttsim.so python3 -m ferrite.bench
    TTSIM=/path/to/libttsim.so python3 test/gpt2.py path/to/model.safetensors "The meaning of life is" 8
    TTSIM=/path/to/libttsim.so python3 test/llama.py path/to/SmolLM-135M "The meaning of life is" 8
    TTSIM=/path/to/libttsim.so python3 test/train.py 300
    TTSIM=/path/to/libttsim.so PYTHONPATH=path/to/tinygrad python3 adapters/tinygrad_ferrite.py path/to/tinygrad/examples/transformer.py

The tests need numpy; `test/gpt2.py` also tiktoken and the weights (Hugging Face `openai-community/gpt2`, `model.safetensors`), and `GPT2=path/to/model.safetensors` adds the whole model to the tests; `test/llama.py` wants Hugging Face's `HuggingFaceTB/SmolLM-135M` folder (config.json, tokenizer.json, model.safetensors), and `LLAMA=path/to/folder` adds it to the tests; `test/train.py` and `test_train` want PyTorch, as the reference; `test_tinygrad` wants tinygrad (`TINYGRAD=path/to/tinygrad`, or installed), and the adapter numpy. Without `TTSIM`, `pci.open_device()` looks for a real Blackhole on the PCI bus.

The simulator is a process-wide singleton that kills the process on any contract violation (bad address, bad TLB config, a semaphore taken while empty): a test run that stops early means the code did something the hardware would not like. pytest hides ttsim's `ERROR:` line; run with `-s` to see it.
