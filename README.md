# ferrite

The whole software stack for Tenstorrent's Blackhole, from PCIe up to a tensor compiler, in about 1,900 lines of
Python code. Nothing from Tenstorrent runs but the card's own firmware: no kernel driver, no tt-umd, no tt-metal, no
LLK. ferrite talks to the card through its PCIe BAR from userspace, loads its own firmware onto every Tensix core, and
compiles tensor expressions straight to Tensix instructions, written from the ISA documentation.

```python
from ferrite import pci
from ferrite.chip import Chip
from ferrite.runtime import Runtime
from ferrite.tensor import Tensor

chip = Chip(pci.open_device())                       # the card, or the simulator when TTSIM is set
rt = Runtime(chip, chip.tensix[0], chip.tensix[-1])  # firmware on every Tensix core, one of them dispatching
x = Tensor.of(rt, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
w = Tensor.of(rt, [[0.5, -1.0], [2.0, 0.25], [-1.0, 1.0]])
print((x @ w).relu().softmax(-1).tolist())           # lazy: one fused kernel, launched when the value is read
```

## Running

Needs Python 3, clang and lld, and Tenstorrent's simulator, ttsim, built for Blackhole (`libttsim.so`); the tests also
need numpy. Without `TTSIM`, ferrite looks for a real Blackhole on the PCI bus (as root); `NOTES.md` has the plan for
bringing one up.

    TTSIM=/path/to/libttsim.so python3 -m ferrite.chip       # what the card says about itself: clock, cores, DRAM
    TTSIM=/path/to/libttsim.so python3 -m ferrite.hello      # every worker computes fib(30); one more dispatches
    TTSIM=/path/to/libttsim.so python3 -m pytest test        # every step, on the simulator
    TTSIM=/path/to/libttsim.so python3 -m ferrite.bench      # a matmul's speed and DRAM reads
    TTSIM=/path/to/libttsim.so python3 -m ferrite.probe      # launch, DRAM, NoC, Tensix unit and PCIe rates (for the card)
    python3 test/benchmark.py ferrite Llama-3.1-8B-Instruct tt_transformers/tests/reference_outputs/Llama-3.1-8B-Instruct.refpt   # B1-B3 (COMPARE.md), or ttnn

## What runs

GPT-2 small and SmolLM-135M (SmolLM2-360M and 1.7B too) with their real weights (Hugging Face's `model.safetensors`),
each word checked against the same model in numpy; where ferrite picks another, its pick is in brackets (here a near
tie, 0.36 apart in numpy's logits). Both decode with a KV cache, each token replayed from a recording, so a token
costs the host 0.1 ms, and both can keep their weight matrices in bfp8 (add `bfp8`), at about half the bytes (SmolLM's
MLP in bfp4 too, as Tenstorrent's own code keeps it: add `bfp4`).

    TTSIM=... python3 models/gpt2.py model.safetensors "The meaning of life is" 8
    The meaning of life is not[ferrite: ' to'] the same as the meaning of death
    TTSIM=... python3 models/llama.py SmolLM-135M "The meaning of life is" 6
    The meaning of life is a question that has been asked

Llama-3.1-8B-Instruct runs whole on the simulator at Tenstorrent's performance precision (MLP weights in bfp4, the rest
in bfp8), each token the one numpy picks. Mixtures of experts run with their experts chosen on the chip: one kernel of
a RISC-V core takes each token's best experts, matmuls read the chosen experts' weights where they lie, and a prompt's
tokens are grouped by expert, so a decode step replays with whatever experts its tokens pick. Qwen3-30B-A3B's first two
blocks, with its real weights, pick numpy's token at every place of a prompt and decode as numpy does. `models/llama.py`
reads Llama's, Mistral's, Qwen2's and Qwen3's checkpoints, and Mixtral's, Qwen2-MoE's, Qwen3-MoE's and gpt-oss's (its
MXFP4 experts too), and Qwen3.5's and Qwen3.6's, whose Gated DeltaNet blocks keep a fixed-size state rather than a
cache, each checked against transformers' own model. Qwen3.6-35B-A3B's real blocks run on the simulator within about 1%
a block of the same model in numpy. Two chips (ttsim's two-chip build) find their
Ethernet links and copy a tensor from one's DRAM to the other's across one, kernels on both chips driving it, with no
code on the Ethernet tiles (`Tensor.to`).

Serving: `models/serve.py` is an OpenAI-compatible server (`/v1/completions`, `/v1/chat/completions`, whole or streamed)
for these models, from a folder or a Hugging Face model id; a thinking model's reasoning (Qwen3's) comes apart from its
answer, and Qwen3.6's tool calls come back as OpenAI's `tool_calls`. Requests decode together, each in a slot of one KV
cache (and of Gated DeltaNet's states), every step one replayed call for all of them, and a prompt goes to the slot
that already holds the longest start of it, or takes a session saved on the host, so a shared prefix (an agent's
conversation so far) is not prefilled again. Qwen3.6 drafts with its multi-token prediction layer, a token and its
draft verified each step, one or two tokens a step, the model's own. SmolLM-135M answers "The meaning of life is" over
HTTP as above.

    TTSIM=... python3 models/serve.py SmolLM-135M 8000
    curl localhost:8000/v1/completions -d '{"prompt": "The meaning of life is", "max_tokens": 6, "temperature": 0}'

Training: a small GPT (GPT-2's blocks, fp32, Adam) learns this README and LESSONS.md a character at a time, each step
replayed from a recording, its loss and gradients the same as PyTorch's for the same weights (`test/train.py 300`).

tinygrad, outside ferrite: its tensor graph runs on ferrite as the FERRITE device, so tinygrad's own scripts run
unchanged (its `examples/transformer.py` learns two-digit addition to 99.99% on the simulator) and `TinyJit` replays a
call's launches as ferrite's `jit` does. So does tinygrad's own LLM code (`tinygrad/llm`'s Transformer), matching
ferrite's own Llama's logits, though not yet at its speed (NOTES: "tinygrad on ferrite"). PyTorch code comes the same
way, through tinygrad's own PyTorch frontend (`extra/torch_backend` in tinygrad: the `"tiny"` device).

    TTSIM=... PYTHONPATH=tinygrad python3 adapters/tinygrad_ferrite.py tinygrad/examples/transformer.py

## Speed

Against Tenstorrent's own stack (tt-metal with TT-NN) on the same simulator: ferrite does the same ops in 4 to 18
times fewer cycles of control work, reading as little DRAM, and runs SmolLM-135M end to end, the same tokens, in 9 to
10 times fewer cycles a token than Tenstorrent's own LLM code (`tt_transformers`), with a quarter as many launches (5
times fewer cycles for 32 users decoding at once), 8 to 11 times fewer for SmolLM2-360M. The simulator times only that
control work, not the math or the DRAM: by 1.7B parameters a token's weights take longer to read than TT's control
work, so there both would wait on DRAM and the lead is in how close each gets to the card's bandwidth. How fast either
is on a real card waits for one; `COMPARE.md` has the numbers and what carries over.

## Code

- `ferrite/pci.py`, `ferrite/chip.py`: the card over PCIe (TLB windows onto the NoC, the ARC management core, DMA into
  host memory), or ttsim's simulated card behind the same interface.
- `ferrite/runtime.py`: firmware for every core, kernels that stay loaded, and a command ring in host memory that one
  core reads by itself, so a launch costs the host one PCIe write, and a recorded block of launches one too.
- `ferrite/tensor.py`: lazy tensors of any shape, bf16, fp32, int32, bfp8 or bfp4, with numpy's broadcasting and
  tinygrad's autograd. A tree of ops becomes as few fused kernels as fit; `jit` records a function's launches and
  replays them.
- `ferrite/compiler.py`: those kernels, as Tensix instructions from the ISA. Each is four programs for four of a Tensix
  tile's five cores: BRISC's moves tiles in and out, and one for each TRISC feeds the unpack, math or pack thread.
- `ferrite/bh.py`: Blackhole's facts as data: addresses, limits, instruction and config encodings.
- `adapters/tinygrad_ferrite.py`: tinygrad on ferrite, and PyTorch through tinygrad's frontend.
- `models/`: GPT-2 and the Llama-family models (`gpt2.py`, `llama.py`), written once for ferrite and numpy, the
  OpenAI-compatible server for them (`serve.py`), and what one costs to serve on a card (`cost.py`).
- `test/`: every step on the simulator (`test_sim.py`), training, and the benchmarks.

## Docs

`NOTES.md`: how it works, and what is still unknown. `COMPARE.md`: ferrite against TT-NN, and how the card benchmarks
are measured. `ROADMAP.md`: the plan, from one card to a box of linked cards serving the open models agents run on, and
the next steps. `LESSONS.md`: what the work has taught us.

## License

Apache-2.0 (`LICENSE`), as tt-metal's.
