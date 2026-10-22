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
- Python with ctypes. No dependencies beyond the standard library in `ferrite/`; the tests, `adapters/` and `models/` may use numpy.
- Blackhole only, with its chip-specific facts kept as data in one place (`ferrite/bh.py`).
- Every wait has a timeout: a hung chip is reported, never waited on forever.
- Every step has a test that runs on the simulator.

## Layout

- `ferrite/bh.py`: Blackhole's facts as data: addresses, sizes and limits, instruction and config encodings (an instruction's modifier values stay where the compiler uses them). NOTES.md says where each comes from.
- `ferrite/pci.py`: the device interface (config space, BAR0, DMA memory, `tick`), with two backends: `Sim` (libttsim.so, each chip of a multi-chip build) and `Card` (sysfs), and a whole-chip reset for a card that hangs.
- `ferrite/chip.py`: NoC reads and writes through TLB windows (unicast and multicast), waiting with a timeout, ARC messages and telemetry, core reset, and the Ethernet links (`links`: which are up, to which chip's channel; `send`: a TT-link write across one); `python3 -m ferrite.chip` prints what a card says about itself (`reset`: after resetting it).
- `ferrite/runtime.py`: device C compiled with clang and lld (cached in `~/.cache/ferrite`, for a week unused), firmware for the workers and the dispatcher, kernels loaded once and kept until their room is full, the command ring in host memory (a launch, or a recorded block of them, is one PCIe write), DRAM, waiting, and each launch's time on the chip's clock (`profile`).
- `ferrite/compiler.py`: Tensix instructions from the ISA. A fused kernel is four programs (BRISC moves tiles; each TRISC feeds one Tensix thread: unpack, math or pack): elementwise ops and SFPU programs, matmuls (blocks, tiles shared between workers by multicast, reads in flight, bfp8 and bfp4 weights, B's matrices picked on the chip: a mixture of experts'), reductions along any axis in passes, movement, `gather`, `put`, `topk` and `group` (a prompt's tokens by expert), and a copy across an Ethernet link (`link_kernel`).
- `ferrite/tensor.py`: `Tensor`: lazy, any shape, bf16, fp32, int32, bfp8 or bfp4, numpy's broadcasting, views, tinygrad's ops and autograd. Realizing turns a tree of ops into as few kernels as fit; `jit` records a function's launches and replays them; `to` copies a tensor to another chip across an Ethernet link.
- `ferrite/hello.py`: fib(n) on every worker, the one-command demo. `ferrite/bench.py`: a matmul's speed and DRAM reads (ROADMAP, Stage 1). `ferrite/probe.py`: the numbers ttsim cannot give (launch cost, DRAM GB/s by reads in flight, NoC latency, MVMUL, unpack, pack and SFPU rates, BRISC's topk and grouping for a mixture of experts, PCIe), for the card's first day, into a JSON file for `models/cost.py` when given one.
- `adapters/tinygrad_ferrite.py`: tinygrad on ferrite, outside it: the FERRITE device (tinygrad's tensors of any rank as views of ferrite's), and a runner for tinygrad's scripts unchanged. PyTorch runs through tinygrad's own frontend (`extra/torch_backend` in tinygrad), not an adapter of ours.
- `models/`: the models and the server, on ferrite but outside it (they use numpy, as `adapters/` may). `gpt2.py`, `llama.py`: GPT-2 small and SmolLM-135M (and SmolLM2; its loader reads Llama 3.1's, Mistral's, Qwen2's and Qwen3's checkpoints too, and Mixtral's, Qwen2-MoE's, Qwen3-MoE's and gpt-oss's, a mixture of experts in a block's MLP place, and Qwen3.5's and Qwen3.6's, Gated DeltaNet blocks between gated attention ones), written once for ferrite Tensors and numpy, generating from the real weights, checked against numpy step by step. `cost.py`: what a model costs to serve on a card, from its config alone (what it holds and a decode step reads and launches, sessions a card keeps, tokens/s, dollars a million tokens), the card's rates its specification's or the probe's. `serve.py`: an OpenAI-compatible server for them, from a folder or Hugging Face model id (ROADMAP, Stage 3): requests decode together, a slot of one KV cache each, every step one replayed call, and a prompt goes to the free slot holding the longest start of it, or takes a session saved on the host (its cache rows and Gated DeltaNet states, between an agent's turns), so a shared prefix is not prefilled again; Qwen3.5's tool calls, OpenAI's way, and tool_choice and response_format (a JSON schema) as shapes each token drawn keeps to; OpenAI's sampling parameters and log probabilities; usage counting the cached prompt tokens, as OpenRouter bills them; and a model with a multi-token prediction layer decodes speculatively, a draft verified each step.
- `test/ttsim-chaos.patch`: ttsim with NoC latency and core jitter (NOTES, "ttsim-chaos"), for the missing waits plain ttsim hides, and with a p150's 140 Tensix unharvested, as firmware before 19.5 runs it.
- `test/test_sim.py`: every step, on ttsim (or on a card when `TTSIM` is not set). `test/train.py`: a small GPT trained on ferrite in fp32 with Adam, checked against PyTorch. `test/benchmark.py`: Stage 1's benchmarks (COMPARE.md: B1, B2, B3) on ferrite or on TT-NN (its B2 and B3 by Tenstorrent's own demo script), with the versions and lines of code that took part, into `benchmark-<stack>.json`; `table` sets the two side by side.
- `NOTES.md`: how ferrite works, and what is still unknown. `LESSONS.md`: what the work has taught us; add to it as we go, and keep it short and true. `ROADMAP.md`: the plan, stage by stage, from one card to serving the open models agents run on through OpenRouter (Qwen3.6-35B-A3B first, on a card or two; then the big flash models on linked cards), and the next steps in order. `COMPARE.md`: ferrite against TT-NN, and how the card benchmarks are measured.

## Running

Needs clang and lld on PATH, and ttsim's `libttsim.so` built for Blackhole (outside this repo: `./make.py src/_out/release_bh/libttsim.so` in tenstorrent/ttsim). Everything runs from the repository's root.

    TTSIM=/path/to/libttsim.so python3 -m ferrite.chip
    TTSIM=/path/to/libttsim.so python3 -m ferrite.hello
    TTSIM=/path/to/libttsim.so python3 -m pytest test
    TTSIM=/path/to/libttsim.so python3 -m ferrite.bench
    TTSIM=/path/to/libttsim.so python3 -m ferrite.probe
    TTSIM=/path/to/libttsim.so python3 models/gpt2.py path/to/model.safetensors "The meaning of life is" 8
    TTSIM=/path/to/libttsim.so python3 models/llama.py path/to/SmolLM-135M "The meaning of life is" 6
    TTSIM=/path/to/libttsim.so python3 test/train.py 300
    python3 models/cost.py path/to/Qwen3.6-35B-A3B experts context=32768   # its config.json is enough; probe=probe.json for a card's numbers
    TTSIM=/path/to/libttsim.so python3 models/serve.py path/to/SmolLM-135M 8000   # then: curl localhost:8000/v1/completions -d '{"prompt": "Hi"}'
    TTSIM=/path/to/libttsim.so BOOK=path/to/tale-of-two-cities.txt.bz2 python3 test/benchmark.py ferrite path/to/SmolLM-135M path/to/SmolLM-135M.refpt --quick
    TTSIM=/path/to/libttsim.so PYTHONPATH=path/to/tinygrad python3 adapters/tinygrad_ferrite.py path/to/tinygrad/examples/transformer.py

The tests need numpy; `models/gpt2.py` also tiktoken, and `test_tokenizer` it as the reference (and `models/llama.py`'s token accuracy PyTorch, to read the reference file); `models/gpt2.py` the weights (Hugging Face `openai-community/gpt2`, `model.safetensors`), and `GPT2=path/to/model.safetensors` adds the whole model to the tests; `models/llama.py` wants Hugging Face's `HuggingFaceTB/SmolLM-135M` folder (config.json, tokenizer.json, model.safetensors), and `LLAMA=path/to/folder` adds it to the tests; `test/train.py` and `test_train` want PyTorch, as the reference; `test_tinygrad` and `test_tinygrad_llm` want tinygrad (`TINYGRAD=path/to/tinygrad`, or installed), and the adapter numpy; `test_torch_frontend` wants a tinygrad checkout (`TINYGRAD=`, for its `extra/torch_backend`) and PyTorch (tinygrad's torch frontend builds a small C++ extension the first time, with the system's C++ compiler and ninja). `test/benchmark.py` wants tiktoken and PyTorch (to read the reference file), and `cloc` to count lines; its `ttnn` side runs in tt-metal's Python environment (`TT_METAL_HOME`), and `TT_KMD=path/to/tt-kmd` counts tt-kmd's lines too. `test_llama_frequencies` wants Hugging Face's transformers, as the reference, and `test_qwen3_moe` and `test_hf_families` it and PyTorch; `test_qwen3_5_moe` a transformers that has Qwen3.5-MoE (5.20 has it, 5.1 not). `LLAMA8B=1` adds one block in Llama-3.1-8B's shapes (minutes on ttsim). `FUZZ=n` runs n of `test_fuzz`'s expressions rather than 30. `TTSIM2=path/to/libttsim.so` of ttsim's two-chip build (`./make.py src/_out/release_bh_x2/libttsim.so`) adds `test_two_chips`. Without `TTSIM`, `pci.open_device()` looks for a real Blackhole on the PCI bus.

The simulator is a process-wide singleton that kills the process on any contract violation (bad address, bad TLB config, a semaphore taken while empty): a test run that stops early means the code did something the hardware would not like. pytest hides ttsim's `ERROR:` line; run with `-s` to see it.
