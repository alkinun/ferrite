# Roadmap

The problem: AI accelerators other than NVIDIA's are held back by their software, not their silicon. ferrite attacks it
from the software side, one chip at a time, staying small: a whole stack one person can read, built on a few primitive
ops done completely rather than thousands done partly.

Each stage is useful on its own and makes the next one possible. A stage is done when its last line holds; its steps
are ticked as tests show them. NOTES.md has the detail of what is built.

## Stage 0: a stack on the simulator (done)

- [x] Userspace PCIe driver, the NoC, ARC, resets: no Tenstorrent software at runtime
- [x] Firmware on every core, resident kernels, a command ring: a launch is one PCIe write
- [x] A compiler from the ISA docs: fused kernels for elementwise ops, matmul, SFPU ops, row reductions, views, batches
- [x] GPT-2 small with the real weights, token for token the same as numpy (`test/gpt2.py`)

Done: a real model generates text on ttsim with nothing from Tenstorrent but the simulator.

## Stage 1: a general compiler, on the simulator

Any model and training, from a small set of primitives: every model is built from these, and training is each run
backwards.

- [x] Reductions along any axis (sum, max), not only rows (`test_reduce_axes`)
- [x] Movement at any granularity: reshape, permute, expand, pad, slice, not only on tile boundaries (`test_movement`)
- [x] More elementwise ops: log, sqrt, sin, cos, compare, where (`test_elementwise`)
- [x] More dtypes: fp32 tensors (accuracy, gradients), integers (indices) (`test_fp32`, `test_int32`)
- [x] Autograd: a backward for every primitive, checked against numpy's gradients (`test_autograd`)
- [x] A Llama-family model: RMSNorm, RoPE, SwiGLU, grouped-query attention, a KV cache (SmolLM-135M, `test/llama.py`)
- [x] A small CNN: convolution as an unfolded matmul, max pooling (`test_mnist_cnn`)
- [x] Training: a small GPT for a few hundred steps, its loss falling as PyTorch's does (`test/train.py`, `test_train`)

Done: Llama and CNN inference and GPT training run correctly on ttsim.

## Stage 2: silicon

- [ ] The bring-up checklist on a p150 (NOTES.md, "Bring-up on a p150"): hello, the tests, the models
- [ ] Real timing: per-kernel profiles from the card, not the simulator's cycles

Done: everything from stage 1 runs correctly on the card.

## Stage 3: fast on silicon

- [ ] Matmul near the chip's peak, decode near its DRAM bandwidth
- [x] Tiles multicast between cores: a matmul reads each input tile from DRAM once (`test_matmul_multicast`)
- [ ] What the simulator could not measure: several reads in flight, BFP8 weights, fewer fidelity phases where accuracy
      allows
- [ ] Benchmarks against Tenstorrent's own stack, published

Done: competitive numbers, and for every gap, a known reason.

## Stage 4: other people's code

- [x] A tinygrad backend (tinygrad is built on almost the same primitives): `adapters/tinygrad_ferrite.py`, `test_tinygrad`
- [x] A PyTorch adapter: `adapters/torch_ferrite.py`, `test_torch`
- [x] Models and training scripts written for those frameworks run unchanged: tinygrad's `examples/transformer.py`
      trains to 99.99% on ttsim

Done: someone else's model trains on a Tenstorrent card through ferrite.

## Stage 5: scale

- [ ] Several cards over Blackhole's Ethernet links
- [ ] Distributed inference and training
- [ ] A model too big for any one consumer GPU, trained and served on a few cards

Done: a small cluster of cards does what one GPU cannot.

## Stage 6: a chip of our own, on paper

From stage 3 on, alongside: computer architecture, RTL, an FPGA board.

- [ ] An accelerator designed for simple software, from what Blackhole taught us (what was hard to program is designed
      away)
- [ ] Its simulator, and a ferrite-style compiler for it
- [ ] Real models on the simulated chip, with credible performance numbers
- [ ] A prototype on an FPGA

Done: a design whose software already works, which is what makes a team and silicon worth funding.

## Then: choose

With public results in hand: work with Tenstorrent, tinygrad or AMD (the same approach fits any accelerator held back
by its software), or form a team and build the chip.
