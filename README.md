# ferrite

The whole software stack for Tenstorrent's Blackhole, from PCIe up to a tensor compiler, in about 2700 lines of Python.
Nothing from Tenstorrent runs but the card's own firmware: no kernel driver, no tt-umd, no tt-metal, no LLK. ferrite
talks to the card through its PCIe BARs from userspace, loads its own firmware onto every Tensix core, and compiles
tensor expressions straight to Tensix instructions, written from the ISA documentation.

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
need numpy. Without `TTSIM`, ferrite looks for a real Blackhole on the PCI bus (as root). It has not met one yet:
`NOTES.md` has the plan for the first day on a p150.

    TTSIM=/path/to/libttsim.so python3 -m ferrite.hello      # every worker computes fib(30); one more dispatches
    TTSIM=/path/to/libttsim.so python3 -m pytest test        # every step, on the simulator
    TTSIM=/path/to/libttsim.so python3 -m ferrite.bench      # a matmul's speed and DRAM reads

## What runs

GPT-2 small and SmolLM-135M with their real weights (Hugging Face's `model.safetensors`), each word checked against the
same model in numpy; where ferrite picks another, its pick is in brackets (here a near tie, 0.36 apart in numpy's
logits). SmolLM decodes with a KV cache, each token replayed from a recording, so a token costs the host 0.1 ms. Both
can keep their weight matrices in bfp8 (add `bfp8`), at about half the bytes.

    TTSIM=... python3 test/gpt2.py model.safetensors "The meaning of life is" 8
    The meaning of life is not[ferrite: ' to'] the same as the meaning of death
    TTSIM=... python3 test/llama.py SmolLM-135M "The meaning of life is" 6
    The meaning of life is a question that has been asked

Training: a small GPT (GPT-2's blocks, fp32, Adam) learns this README and LESSONS.md a character at a time, each step
replayed from a recording, its loss and gradients the same as PyTorch's for the same weights (`test/train.py 300`).

Frameworks, outside ferrite: tinygrad's tensor graph runs on it as the FERRITE device, so tinygrad's own scripts run
unchanged (its `examples/transformer.py` learns two-digit addition to 99.99% on the simulator); and PyTorch's, as a
`FerriteTensor` whose aten ops run on the chip, so eager PyTorch, its autograd and its optimizers do, and Hugging Face
transformers' models run, generate and fine-tune unchanged.

    TTSIM=... PYTHONPATH=tinygrad python3 adapters/tinygrad_ferrite.py tinygrad/examples/transformer.py

    from adapters.torch_ferrite import ferrite
    model = ferrite(AutoModelForCausalLM.from_pretrained("SmolLM-135M"))
    model.generate(ferrite(ids), max_new_tokens=6)

## Speed

Against Tenstorrent's own stack (tt-metal with TT-NN) on the same simulator: ferrite does the same ops in 3 to 18 times
fewer cycles of control work, reading as little DRAM. The simulator times only that control work, not the math or the
DRAM, so how fast either is on a real card waits for one; `COMPARE.md` has the numbers and what carries over.

## Code

- `ferrite/pci.py`, `ferrite/chip.py`: the card over PCIe (TLB windows onto the NoC, the ARC management core, DMA into
  host memory), or ttsim's simulated card behind the same interface.
- `ferrite/runtime.py`: firmware for every core, kernels that stay loaded, and a command ring in host memory that one
  core reads by itself, so a launch costs the host one PCIe write, and a recorded block of launches one too.
- `ferrite/tensor.py`: lazy tensors of any shape, bf16, fp32, int32 or bfp8, with numpy's broadcasting and tinygrad's
  autograd. A tree of ops becomes one fused kernel; `jit` records a function's launches and replays them.
- `ferrite/compiler.py`: those kernels, as Tensix instructions from the ISA. Each is four programs, one per core of a
  Tensix tile: one moves tiles in and out, and three feed the unpack, math and pack threads.
- `ferrite/bh.py`: every Blackhole fact the code uses, as data.
- `adapters/`: tinygrad and PyTorch on ferrite (and `views.py`, tensors of any rank as views of ferrite's).
- `test/`: every step on the simulator (`test_sim.py`), and the models.

## Docs

`NOTES.md`: how it works, and what is still unknown. `COMPARE.md`: ferrite against TT-NN. `ROADMAP.md`: where it goes,
from here to silicon, many cards and a chip of our own. `LESSONS.md`: what the work has taught us.
