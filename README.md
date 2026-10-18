# ferrite

The whole software stack for Tenstorrent's Blackhole, from PCIe up to a tensor compiler, in about 2100 lines of Python.
No Tenstorrent software runs: no kernel driver, no tt-umd, no tt-metal, no LLK. ferrite talks to the card through its
PCIe BARs from userspace, loads its own firmware onto every core, and compiles tensor expressions straight to Tensix
instructions, written from the ISA documentation.

It runs GPT-2 small with the real weights, so far on the simulator, checking every word against the same model in
numpy (which matches Hugging Face's GPT-2):

    TTSIM=/path/to/libttsim.so python3 test/gpt2.py model.safetensors "The meaning of life is" 8
    The meaning of life is not the same as the meaning of death

Each word's scores stay within about 1 of numpy's (they span about 100); where numpy's top two are closer than that,
bf16 can pick the other one.

## What is there

- `ferrite/pci.py`, `ferrite/chip.py`: the card over PCIe (TLB windows onto the NoC, the ARC management core, DMA into
  host memory), or ttsim's simulated card behind the same interface.
- `ferrite/runtime.py`: firmware for each core, kernels that stay loaded, and a command ring in host memory that one core
  reads by itself, so a launch costs the host one PCIe write.
- `ferrite/tensor.py`: lazy tensors (any shape, batched, bf16, fp32 or int32) whose expressions become one fused kernel
  each: elementwise ops (tinygrad's primitives and what it builds from them: exp, log, sin, cos, sqrt, comparisons,
  where), matmul, sums and maxes along any axis (so softmax and layer norm are one launch), transpose, views,
  reshape, permute, expand, pad and slicing at any offset, and autograd (tinygrad's rules) for all of it.
- `ferrite/compiler.py`: those kernels, as Tensix instructions from the ISA. Each is four programs, one per core of a
  Tensix tile: one moves tiles in and out, and three feed the unpack, math and pack threads.
- `ferrite/bh.py`: every Blackhole fact the code uses, as data. `NOTES.md` says where each comes from.

## Running

Needs Python 3, clang and lld, and Tenstorrent's simulator, ttsim, built for Blackhole (`libttsim.so`). The tests also
need numpy; the GPT-2 demo needs tiktoken and Hugging Face's `openai-community/gpt2` `model.safetensors`.

    TTSIM=/path/to/libttsim.so python3 -m ferrite.hello      # every core computes fib(30)
    TTSIM=/path/to/libttsim.so python3 -m pytest test        # every step, on the simulator
    TTSIM=/path/to/libttsim.so python3 -m ferrite.bench      # a matmul's speed and DRAM traffic

Without `TTSIM`, ferrite looks for a real Blackhole on the PCI bus (as root). It has not met one yet: `NOTES.md` has the
bring-up plan for the first day on a p150.

## Where it is going

`ROADMAP.md`: from here to a general compiler, silicon, frameworks, many cards, and a chip of our own.
`LESSONS.md`: what we have learned on the way.

