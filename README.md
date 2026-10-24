# ferrite

The whole software stack for Tenstorrent's Blackhole, from PCIe up to a tensor compiler, in Python. Nothing from
Tenstorrent runs but the card's own firmware: no kernel driver, no tt-umd, no tt-metal, no LLK. ferrite talks to the
card through its PCIe BAR from userspace, loads its own firmware onto every Tensix core, and compiles tensor
expressions straight to Tensix instructions, written from the ISA documentation.

```python
from ferrite import pci
from ferrite.chip import Chip
from ferrite.runtime import Runtime
from ferrite.tensor import Tensor

chip = Chip(pci.open_device())              # the card, or the simulator when TTSIM is set
rt = Runtime(chip)                          # firmware on every Tensix core, one of them dispatching
x = Tensor.of(rt, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
w = Tensor.of(rt, [[0.5, -1.0], [2.0, 0.25], [-1.0, 1.0]])
print((x @ w).relu().softmax(-1).tolist())  # lazy: one fused kernel, launched when the value is read
```

## Running

Needs Python 3, clang and lld, and Tenstorrent's simulator, ttsim, built for Blackhole (`libttsim.so`); the tests also
need numpy. Without `TTSIM`, ferrite looks for a real Blackhole on the PCI bus (as root).

    TTSIM=/path/to/libttsim.so python3 -m ferrite.chip       # what the card says about itself
    TTSIM=/path/to/libttsim.so python3 -m ferrite.hello      # every worker computes fib(30)
    TTSIM=/path/to/libttsim.so python3 -m pytest test        # every step, on the simulator
    TTSIM=/path/to/libttsim.so python3 models/llama.py SmolLM-135M "The meaning of life is" 6
    TTSIM=/path/to/libttsim.so python3 models/serve.py SmolLM-135M 8000

## What runs

GPT-2 and the Llama family (Llama 3.1, Mistral, Qwen2 and Qwen3, the mixtures of experts of Mixtral, Qwen2-MoE,
Qwen3-MoE and gpt-oss, and Qwen3.5 and Qwen3.6 with their Gated DeltaNet blocks) with their real weights, each checked
against the same model in numpy and against transformers. Weights in bf16, bfp8 or bfp4; a decode step replayed from a
recording; mixtures of experts with their experts chosen on the chip. `models/serve.py` is an OpenAI-compatible server
for them: requests decode together, sessions are kept on the card between an agent's turns, Qwen3.6 drafts with its
multi-token prediction layer, and tool calls, structured outputs and cached-token usage are OpenAI's and OpenRouter's.
tinygrad runs on ferrite as a device (`adapters/`), and PyTorch through tinygrad's frontend.

## Code

- `ferrite/`: the card over PCIe (`pci.py`, `chip.py`), the runtime (`runtime.py`), lazy tensors (`tensor.py`), the
  compiler (`compiler.py`), Blackhole's facts as data (`bh.py`).
- `models/`: the models, the server, and what one costs to serve on a card.
- `adapters/`: tinygrad on ferrite.
- `test/`: every step on the simulator, training against PyTorch, and benchmarks against TT-NN and of serving.

`NOTES.md` says how it works, where each fact about the chip comes from, how to bring up a card, and what is still
unknown.

## License

Apache-2.0 (`LICENSE`).
