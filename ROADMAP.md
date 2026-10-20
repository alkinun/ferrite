# ferrite roadmap

ferrite is the first step toward an AI inference accelerator of our own. This file is the plan from here to that chip:
five stages, each closed by a gate of exit criteria.

## Why software first

Accelerators other than NVIDIA's are held back by their software, not their silicon. A chip that arrives without a stack
as good as CUDA starts out losing, so we build the stack first and the chip after it.

- **Prove it on someone else's silicon.** Tenstorrent's Blackhole is open down to its instruction set: its ISA
  documentation and its simulator, ttsim, are public. ferrite drives it with nothing from Tenstorrent at runtime but the
  card's firmware, in about 3,000 lines of Python. Tenstorrent's own stack spans six layers (tt-kmd, UMD, tt-metal, LLK,
  TT-NN, the model code), about 1.65 million lines without the model code (COMPARE.md). If ferrite runs an
  8-billion-parameter model on the same card within 1.5 times of that stack's speed (Stage 1's gate), with hundreds of
  times less code, the claim stands: a compiler and a runtime, kept clean, are all an accelerator needs. As George
  Hotz's "Advice to Tenstorrent" puts it, "it's 3 (and only 2 you have to build)".
- **Results before claims.** ferrite's case is made with numbers measured on real silicon and published with what it
  takes to reproduce them; numbers from the simulator say so.
- **One stack for every chip.** The same stack drives an FPGA tile we design (Stage 2), our architecture's simulator and
  FPGA prototype (Stage 3), and the chip (Stage 5), so the chip's software works before its silicon exists.

## Where we start (2026-10-20)

Everything so far ran on ttsim. ferrite has never run on silicon; a p150a is ordered.

- **Built:** a userspace PCIe driver, firmware on every Tensix core, a command ring that makes a launch one PCIe write,
  and a compiler from tensor expressions to Tensix instructions, written from the ISA documentation and, where it
  stops, ttsim's data.
- **Runs:** GPT-2 small, SmolLM-135M, SmolLM2-360M and SmolLM2-1.7B with their real weights, in bf16 and in block floats
  (bfp8; for SmolLM also the MLP's gate and up in bfp4, as Tenstorrent's own models keep them), each token checked
  against the same model in numpy, decode replayed from a recording; a small CNN; a small GPT trained in fp32 with
  Adam, step by step against PyTorch; tinygrad's own scripts on its FERRITE device; PyTorch through tinygrad's torch
  frontend. The test suite passes: 68 tests, the models, tinygrad and PyTorch included.
- **Against TT-NN on the same simulator:** 4 to 18 times fewer cycles of control work an op than TT-NN with fast
  dispatch (2 to 3.5 times fewer than with slow dispatch), and 9 to 10 times fewer a SmolLM-135M decode token
  (COMPARE.md). ttsim counts only that control work, not the math or the DRAM, so none of this says yet how fast either
  stack is on a card.
- **Not built:** mixture-of-experts models, attention that scales to long context, more than one card, a Hugging Face
  loader and server, and a backend for any chip but Blackhole.

## Focus

We build an inference card that runs open models, with far more memory per dollar than a GPU, and a software stack small
enough for one engineer to read and for a buyer to audit.

- **For:** organizations that run open models on their own hardware: regulated industries, government and sovereign
  buyers, startups serving their own fine-tuned models, research labs.
- **We win on:** memory capacity per dollar; prompt processing per dollar at long context; open models running from
  their Hugging Face ids with no porting by the user; and a stack small enough to audit end to end.
- **We never build:** training chips, frontier-scale systems, consumer devices, HBM designs, leading-edge nodes,
  hardwired models, or anything that races NVIDIA or AMD on peak compute.
- **Frontends:** tinygrad is the one framework frontend, and PyTorch comes through tinygrad's torch frontend. Customers
  get a Hugging Face loader and an OpenAI-compatible server. Whether the loader runs a model through tinygrad's model
  code or through ferrite's own is the loader decision, made in Stage 1.
- **Training:** ferrite keeps training correct (its own autograd and optimizers, in `test/train.py`, and tinygrad's
  training graphs, which run on it), but training speed is not a goal.

## Rules

1. **One stage at a time.** A stage's work starts only once the stage before it has passed its gate: every exit
   criterion ticked. Nothing else counts.
2. **Targets are fixed before the work.** A stage's exit criteria, and the benchmarks they name, are final the day the
   stage opens; until then they are drafts. After that, a criterion changes only in writing, in this file, with the
   reason, and never because the work fell short of it.
3. **Exact in criteria, open in timing.** A stage takes as long as it takes; it is done when its gate passes, not on a
   date.
4. **A target the hardware cannot reach is still a result.** If a measurement shows the hardware itself cannot reach a
   target (a probe at the hardware's limit, not a slow kernel of ours), the criterion may be rewritten to that limit
   under rule 2, and the gap goes into LESSONS.md as something our chip must design away.
5. **Every January, recheck the bets** at the end of this file. A broken bet changes the plan before more work goes in.
6. **Measured, not projected.** A criterion checks numbers measured as the next section says, or computed from them by a
   formula it gives, unless the criterion says "projects". Every published number comes with what it takes to
   reproduce it.

## Measurement

What the words and numbers in the exit criteria mean.

- **Names.** The p150a is our card. A p150 is either version of it (the p150a, actively cooled, or the p150b, passive:
  the same chip and memory); P150 is how Tenstorrent's tables name it. TT is Tenstorrent. TT-NN is Tenstorrent's
  software stack: tt-metal with its TT-NN library and, for a model, the tt_transformers code that runs it there.
- **Versions.** Every number records the commits of ferrite and tt-metal (and so the UMD it pins), the versions of
  tt-kmd, the card's firmware, tinygrad and PyTorch where they take part, and the host's CPU, OS and kernel. The TT-NN
  numbers a gate compares with are measured on our own card with Tenstorrent's own scripts (tt_transformers'
  `simple_text_demo.py` for models), never taken from Tenstorrent's tables.
- **Decode speed.** Batch 1, unless a benchmark says otherwise. The prompt is the first 512 tokens of Tenstorrent's
  reference file for the model (below); then 200 tokens are generated greedily, end-of-sequence ignored. tokens/s is
  the steps after each stack's warm-up over their wall time on the host's clock. The warm-up is the steps before the
  first replay of a recorded step: TT-NN's first, which compiles and captures its trace (Tenstorrent's script leaves
  it out too), and ferrite's first two, which run and record (`jit`).
- **Prefill speed.** Batch 1. Time to first token: the wall time from a prompt's token ids on the host to its first
  generated token, after a warm-up prefill of the same length, which compiles (Tenstorrent's script does the same, and
  reports its warm-up apart, as compile time). A prompt of n tokens is the first n of A Tale of Two Cities as
  tt_transformers keeps it (`tests/tale-of-two-cities.txt.bz2`), in the model's tokenizer.
- **TT's performance precision.** Each weight matrix and the KV cache in the format, and each matmul at the number of
  fidelity phases, that tt_transformers' performance mode gives it at the pinned tt-metal commit. At 9f98dee9 that is,
  for Llama-3.1-8B: the MLP's gate and up projections in bfp4, multiplied at one phase (LoFi), except in the last of
  its 32 blocks, which keeps them in bfp8 at two (HiFi2); the other weight matrices (attention's, the MLP's down
  projection, the output layer) and the KV cache in bfp8, multiplied at two phases; attention itself at two phases in
  decode and four in prefill.
- **Effective DRAM bandwidth.** Computed: the bytes a decode token reads at the benchmark's formats, times the measured
  tokens/s, over the card's 512 GB/s. Weights count at their stored size (per 1,024 values: bf16 2,048 bytes, bfp8
  1,088, bfp4 576), the token embedding only for the row gathered, the KV cache at the decode's mean context (612.5
  tokens for a 512-token prompt and 200 steps). For Llama-3.1-8B at TT's performance precision that is about 6.20 GB a
  token, so 50% of the bandwidth is 41.3 tokens/s.
- **Token accuracy.** Tenstorrent's method (`simple_text_demo.py`, its `ci-token-matching` run). The reference file
  (`tests/reference_outputs/<model>.refpt` in tt_transformers) holds 1,024 tokens of text and, for each, the Hugging
  Face model's five most likely next tokens. The prompt is its first 512 tokens; then, for 500 steps, the stack is fed
  the file's next token whatever it predicted (teacher forcing). A step counts for top-1 when the stack's most likely
  token is the reference model's most likely, and for top-5 when it is among the reference model's five most likely.
  For a model without a reference file, one is made with Tenstorrent's own script (`tests/generate_reference_hf.py`)
  and the model's own configuration: the script's override of RoPE for Qwen models (YaRN) is left out, since the
  served model does not use it.
- **Matmul time.** bf16 inputs and result in DRAM, sums in fp32, each stack in its own tile layout, both at the same
  number of fidelity phases. The median of 5 measurements, each 20 back-to-back runs after 3 warm-up runs, timed from
  the first launch to the last result. TT-NN's side is `ttnn.matmul` called as its documentation calls it, with the
  fidelity and fp32 accumulation set in its compute kernel config.
- **No more than 1.5 times slower.** A time at most 1.5 times TT-NN's; a speed at least two thirds of TT-NN's.
- **Lines of code.** Code lines as `cloc` counts them (no blank or comment lines). ferrite: `ferrite/` and `adapters/`,
  each given. Tenstorrent: tt-metal (with TT-NN, LLK and the UMD it pins) and tt-kmd, at the recorded versions, without
  tests, models, docs or other third-party code.

## Stage 1: Silicon

Opened 2026-10-20.

ferrite runs correctly and fast on a real p150a, and the numbers are public.

**Benchmarks** (as Measurement defines them)

- **B1, matmul:** (4096 x 4096) @ (4096 x 4096) in bf16, at two fidelity phases (HiFi2, what tt_transformers uses for
  most of its matmuls). Four phases are reported too, not gated.
- **B2, decode:** `meta-llama/Llama-3.1-8B-Instruct` on one p150a at TT's performance precision: tokens/s, effective
  DRAM bandwidth, token accuracy. Batch 32 (32 users decoding together, the case Tenstorrent publishes) is reported too,
  not gated. For scale: at tt-metal 9f98dee9, Tenstorrent's target for this model on a P150 (`model_targets.yaml`) is
  42.0 tokens/s a user at batch 32, and its accuracy table gives top-1 90% and top-5 98%. It gives no batch-1 speed
  (its deprecated PERF.md gave 33.6), and a user's speed at batch 1 is usually at least that at batch 32, so TT-NN is
  likely near the 50% this stage asks of ferrite already.
- **B3, long-context prefill:** time to first token for Llama-3.1-8B at TT's performance precision, with prompts of
  4,096 and 32,768 tokens, on both stacks. Published in Stage 1 as it stands (a prompt a stack cannot run yet is
  published as such); gated in Stage 2.

**Work**

- Before the card arrives: a harsher ttsim (NoC latency, Tensix jitter) to bring out ordering bugs; probes for what
  ttsim cannot measure (launch latency, DRAM GB/s per core, NoC latency, unpack, math and pack rates); a card reset that
  needs no reboot, and a report of what each core of a hung worker waits on.
- Bring the card up with NOTES.md's checklist, in its order. Run TT-NN's Llama-3.1-8B and matmul on the card first: that
  shows the card works and gives the baselines.
- Run Llama-3.1-8B on ferrite: Llama 3's RoPE scaling, its 128,256-token vocabulary and untied output layer, and 16 GB
  of bf16 weights to upload.
- Tune matmul toward the card's peak and decode toward its DRAM bandwidth, measuring what ttsim could not: reads in
  flight, block-float weights, fewer fidelity phases, and the cost of each launch.
- Measure B3 against TT-NN, so Stage 2 starts from a known gap.
- **The loader decision.** Run B2 through ferrite's own model code and through tinygrad's own LLM code
  (`tinygrad/llm`, which covers Llama 3 and Qwen 3, its mixture of experts included) on FERRITE, and decide which the
  Hugging Face loader runs. For a like-for-like comparison, tinygrad's path needs three things first: the same model
  and weights (Llama 3.1's RoPE scaling; tinygrad's LLM code loads GGUF files and casts weights to fp16 by default),
  B2's block-float formats through the adapter (which carries fp32, bf16 and int32 today), and replay of a decode step
  whose position is a symbolic variable (the adapter runs such a call every time).
- Turn the hand check of PyTorch through tinygrad's torch frontend into an automated test
  (`adapters/torch_ferrite.py` is already gone).
- Add a LICENSE: Apache-2.0, as tt-metal's.
- Publish B1, B2 and B3 for both stacks, each stack's lines of code beside them, and a script anyone with a p150 can
  rerun.
- The first hardware design, in simulation: a multiply-accumulate unit in Verilog or Amaranth, tested with Verilator
  and cocotb. bf16 inputs, an fp32 accumulator; each product exact unless it is out of fp32's range (then infinity, or
  zero), each sum rounded to nearest even; denormals in and out flushed to zero; inf x 0 and inf - inf give NaN, and
  every NaN out is 0x7FC00000.

**Exit criteria** (fixed 2026-10-20)

- [ ] On the p150a, three runs in a row of `pytest test` (with `GPT2`, `LLAMA` and `TINYGRAD` set, and PyTorch
      installed) have no failures and no skips but `test_card_wants_root`, which skips as root by design.
- [ ] B2: decode at no less than 41.3 tokens/s (50% of the card's DRAM bandwidth), with top-1 at least 90% and top-5 at
      least 98%.
- [ ] B1 and B2: ferrite is no more than 1.5 times slower than TT-NN on the same card.
- [ ] B1, B2 and B3 are published for both stacks, with each stack's lines of code and the rerun script.
- [ ] The loader decision is written in this file, with both measurements and its reasons.
- [ ] A LICENSE is in the repository, and PyTorch through tinygrad's torch frontend is an automated test that passes on
      the card.
- [ ] The multiply-accumulate unit matches a Python model of that arithmetic bit for bit in cocotb, on a million random
      inputs and on zeros, infinities, NaNs and denormals.

**Not yet:** FPGA work, our own architecture, customers.

## Stage 2: First hardware and first users

ferrite drives a compute tile we designed, and other people run models on ferrite.

**Work**

- Buy a Kintex-7 XC7K70T PCIe card with DDR3 and a documented pinout (about $170 used); Vivado's free tier covers this
  part. Bring up PCIe with LitePCIe.
- Build one compute tile: a matmul array on the DSP blocks, an on-chip scratchpad, DMA to the card's DRAM, control
  registers over PCIe. Design it against LESSONS.md's "For a chip of our own", so the tile is also the first test of
  those ideas.
- Split ferrite into what every chip shares (tensors, fusion, autograd, replay, the host's side of the runtime) and a
  backend per chip: Blackhole is the first, the tile the second.
- Ship a Hugging Face loader and an OpenAI-compatible server on the p150a, through the path the loader decision chose,
  for three families: Llama 3, Qwen 3, and Qwen 3's mixture of experts (Qwen3-30B-A3B: about 18 GB with its experts in
  bfp4, so it fits one card). The models named here are re-pinned to the families' current releases when the stage
  opens.
- Mixture of experts in the compiler: routing (top-k) on the chip, and expert weights picked by data on the chip, so a
  decode step still replays from a recording.
- Attention for long context: one pass over the keys and values with a running max and sum (flash attention's method),
  shared out over every worker.
- Not gated: a second p150a, and one model across both cards over their QSFP-DD links. What it shows feeds the
  card-to-card links in Stage 3's chip spec.
- Decide, with Stage 1's numbers, how ferrite works with Tenstorrent from here.

**Exit criteria**

- [ ] A matmul issued from ferrite runs on our FPGA tile over PCIe and matches a Python model of the tile's arithmetic
      bit for bit.
- [ ] ferrite has one backend interface, with Blackhole and the tile behind it, and Stage 1's suite criterion still
      holds on the p150a.
- [ ] A Hugging Face model id becomes an endpoint of the OpenAI API (`/v1/completions` and `/v1/chat/completions`,
      streaming included, checked with OpenAI's own Python client) for Llama-3.1-8B, Qwen3-8B and Qwen3-30B-A3B, each
      at the formats it is served in: Llama-3.1-8B with Stage 1's token accuracy, the Qwen models with top-1 at least
      85% and top-5 at least 95%.
- [ ] Qwen3-30B-A3B, its experts in bfp4 and its other weight matrices in bfp8, decodes on one p150a at no less than 87
      tokens/s: 40% of the card's DRAM bandwidth at about 2.36 GB a token (40% rather than Llama's 50%: a mixture of
      experts reads many small matrices, picked at run time).
- [ ] B3 at 32,768 tokens (or the longest prompt TT-NN runs on one p150a, if shorter): ferrite is no more than 1.5
      times slower than TT-NN on the same card.
- [ ] At least 3 people outside the project have run a model on ferrite from its public instructions, and said so in
      writing.
- [ ] How ferrite works with Tenstorrent from here is decided and written down, with its reasons.

**Not yet:** multi-tile designs, our own architecture spec, fundraising.

## Stage 3: Architecture v1

Our own architecture exists on paper, in a simulator and on an FPGA, and ten potential customers have seen it.

**Work**

- Write the architecture spec from Blackhole's lessons (LESSONS.md) and the bets below: what was hard to program is
  designed away, and the ISA fits on one page.
- Build its cycle-level simulator, which models time as well as rules (LESSONS.md, "The simulator"), and a ferrite
  backend for it.
- Move to a used Alveo U50 (about £850, roughly $1,100; 8 GB of HBM2, 201 GB/s nominal and 316 GB/s at the peak AMD
  measured): several tiles, an on-chip dispatcher, and the HBM standing in for the chip's LPDDR. From Vivado 2026.1 an
  Alveo card needs Vivado's own Alveo tier; Vivado 2025.2's free edition still covers it.
- Write the target chip's spec: memory, bandwidth, compute, card-to-card links, die area, cost per card, and the cost to
  first silicon (masks, licensed PCIe and LPDDR IP, EDA tools, packaging, boards).
- Have a patent attorney review the architecture for freedom to operate.
- Show ferrite on the p150a and the projected chip to 10 potential customers in the target market.

**Exit criteria**

- [ ] A small LLM (SmolLM-135M) generates text on the U50 through ferrite, token for token as its numpy reference.
- [ ] Measured FPGA speed is within 20% of what the simulator predicted, for a matmul and for that LLM's decode, the
      predictions committed to the repository before the FPGA runs.
- [ ] The target chip's spec projects at least 2 times the tokens per second per dollar of card price of the best card
      on sale, on a mixture-of-experts model named in the spec, serving requests of 32,768-token prompts and 512
      generated tokens. The best card on sale is the best of NVIDIA's, AMD's and Tenstorrent's for that workload, its
      speed measured by us where we have the card and otherwise taken from a named public source, its price as listed
      or quoted, with the source.
- [ ] The cost to first silicon is estimated from written quotes for the IP and the masks.
- [ ] The patent attorney's freedom-to-operate review is written up, with no patent it rates as likely infringed that
      the design neither avoids nor licenses.
- [ ] 10 customer conversations are written up, and at least 3 of those customers say in writing that they would test a
      card.

**Not yet:** tapeout, hiring, fundraising.

## Stage 4: Partner, team, proof

A customer's benchmark becomes the chip's spec, a hardware co-founder joins, and one tile comes back from a fab working.

**Work**

- Sign a design partner; their benchmark becomes the chip's spec.
- Bring in a hardware co-founder who has taped out a chip on a modern process.
- Tape out one compute tile on a shuttle run of wafer.space's GF180MCU service. For scale: its Run 3 (2026) sold a
  quarter slot, 1.94 x 2.53 mm and 1,000 dies, for $3,000 ($2,000 early bird), with design files due in December 2026
  and chips in the second quarter of 2027. Runs keep their own dates, so book one as soon as the stage opens. At 180 nm
  the tile proves our design flow, not the chip's speed.
- Raise the funding for Stage 5.

**Exit criteria**

- [ ] A design partner has signed a letter of intent or a paid pilot, with their benchmark and its targets in it.
- [ ] The target chip's spec projects meeting those targets, and at least 2 times the tokens per second per dollar of
      the best card on sale, compared as in Stage 3, on the partner's benchmark.
- [ ] The hardware co-founder has joined full-time.
- [ ] The wafer.space tile runs a matmul correctly, driven by ferrite through the FPGA.
- [ ] Funding is closed for all of Stage 5, the cost to first silicon included.

**Not yet:** a full chip, a sales team.

## Stage 5: The chip

First boards with our chip reach the design partner and pass their benchmark. Expect 18 to 24 months from the start of
the design to first silicon.

**Work**

- Keep the hardware boring: a mature process, LPDDR memory, licensed PCIe and memory interfaces, RISC-V control cores.
  The novelty lives in the architecture and the software.
- ferrite targets the chip from day one through its simulator, so the software is ready before the silicon.
- Build the board and the card's firmware alongside the chip.

**Exit criteria**

- [ ] First silicon boots and passes ferrite's test suite on the chip's backend.
- [ ] The partner's benchmark meets the targets signed in Stage 4.
- [ ] Boards are shipped to the design partner.

**Not yet:** production, and the funding round that pays for it: they come after this file.

## Bets to recheck every January

The chip is years away, so it is designed for the models that will run when it ships, not for today's. These are the
assumptions it rests on.

| Bet | Sign it broke | What changes |
| --- | --- | --- |
| Generating tokens stays limited by memory, not math | Popular open models become compute-bound at decode | Move die area from memory interfaces to compute |
| Open mixture-of-experts models keep winning | Leading open releases go back to dense models | Bandwidth matters more than capacity; revisit the memory choice |
| Agents keep pushing context length up | Typical prompt lengths stop growing | Less prefill compute, more decode bandwidth |
| 4-bit and lower becomes the standard format | Quality losses keep 8-bit the norm | Wider multipliers and more memory per model |
| Model architectures keep changing | Architectures stay frozen for years | Hardwired chips become a threat; consider a fixed-function block |
| Owning your AI keeps getting more important | Cloud prices fall until on-prem stops paying off for our buyers | Revisit who the card is for |

## Sources

- [Tenstorrent Blackhole cards](https://tenstorrent.com/hardware/blackhole): the p150a at $1,399, with 120 Tensix cores,
  180 MB of SRAM, 32 GB of GDDR6 at 512 GB/s and four QSFP-DD 800G ports.
- [Lightning V2 on Tenstorrent](https://arxiv.org/pdf/2604.03279v1): the P150 at $1,400, and its DRAM at 512 GB/s
  (Tables 1 and 3).
- Tenstorrent's [ISA documentation](https://github.com/tenstorrent/tt-isa-documentation) and
  [ttsim](https://github.com/tenstorrent/ttsim); George Hotz's [Advice to Tenstorrent](https://github.com/geohot/tt-tiny).
- tt_transformers at tt-metal 9f98dee9:
  [model_targets.yaml](https://github.com/tenstorrent/tt-metal/blob/9f98dee9/models/model_targets.yaml) (Llama-3.1-8B
  on a P150: 42.0 tokens/s a user at batch 32; top-1 90%, top-5 98%),
  [PERF.md](https://github.com/tenstorrent/tt-metal/blob/9f98dee9/models/tt_transformers/PERF.md) (deprecated; 33.6
  tokens/s at batch 1),
  [model_config.py](https://github.com/tenstorrent/tt-metal/blob/9f98dee9/models/tt_transformers/tt/model_config.py)
  and
  [performance_decoder_config.json](https://github.com/tenstorrent/tt-metal/blob/9f98dee9/models/tt_transformers/model_params/Llama-3.1-8B-Instruct/performance_decoder_config.json)
  (the performance mode's formats and fidelities),
  [simple_text_demo.py](https://github.com/tenstorrent/tt-metal/blob/9f98dee9/models/tt_transformers/demo/simple_text_demo.py)
  (how speed and token accuracy are measured).
- [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) and
  [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B): the shapes behind this file's byte counts.
- [Kintex-7 XC7K70T PCIe card on eBay](https://www.ebay.de/itm/375987216539);
  [AMD UG973 (2025.2)](https://docs.amd.com/r/2025.2-English/ug973-vivado-release-notes-install-license/Supported-Devices): the
  free ML Standard edition's devices, the XC7K70T and Alveo cards among them;
  [AMD's Vivado licensing](https://www.amd.com/en/products/software/adaptive-socs-and-fpgas/vivado/vivado-licensing-options.html)
  and [a summary of the 2026.1 tiers](https://bard0.com/insights/vivado-2026-1-licensing.html): 7 Series in the free
  Basic tier, Alveo in its own tier.
- [Used Alveo U50 on eBay](https://www.ebay.com/itm/156794377853), and
  [AMD DS965](https://docs.amd.com/r/en-US/ds965-u50/Product-Details): the U50's 8 GB of HBM2, 201 GB/s nominal and
  316 GB/s at its measured peak.
- [wafer.space GF180MCU Run 3](https://www.crowdsupply.com/wafer-space/gf180mcu-run-3): slot sizes, prices and dates.
- [Silicon Analysts: tapeout costs and timelines](https://siliconanalysts.com/for-startups)
- [tinygrad](https://github.com/tinygrad/tinygrad)
