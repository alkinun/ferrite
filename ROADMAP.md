# ferrite roadmap

ferrite is the software Tenstorrent's hardware deserves: a compiler and a runtime for Blackhole, small enough to read,
and fast. This file is the plan from one card to machines of linked cards that we sell: four stages, each closed by a
gate of exit criteria.

## Why

Accelerators other than NVIDIA's are held back by their software, not their silicon. Blackhole is good silicon behind a
heavy stack, so we build the stack it should have had, and sell it on Tenstorrent's own cards.

- **Prove it on the card.** Tenstorrent's Blackhole is open down to its instruction set: its ISA documentation and its
  simulator, ttsim, are public. ferrite drives it with nothing from Tenstorrent at runtime but the card's firmware, in
  about 1,900 lines of Python code. Tenstorrent's own stack spans six layers (tt-kmd, UMD, tt-metal, LLK, TT-NN, the
  model code), about 1.42 million lines of code without its tests, models and docs (COMPARE.md). If ferrite runs an
  8-billion-parameter model on the same card within 1.5 times of that stack's speed (Stage 1's gate), with hundreds of
  times less code, the claim stands: a compiler and a runtime, kept clean, are all an accelerator needs. As George
  Hotz's "Advice to Tenstorrent" puts it, "it's 3 (and only 2 you have to build)".
- **Results before claims.** ferrite's case is made with numbers measured on real silicon and published with what it
  takes to reproduce them; numbers from the simulator say so.
- **The links are the opening.** Each p150 has four 800G QSFP-DD ports that join cards directly, with no switch and no
  host between them: up to 400 GB/s each way, six times its PCIe link. Nothing else at its price has them. Software
  that makes two, four or sixteen cards one device is what Tenstorrent's stack does worst, and what the largest open
  models need.
- **Software, not silicon.** Our hardware is Tenstorrent's cards, bought, not designed. The most we build is machines
  from them.

## Where we start (2026-10-21)

Everything so far ran on ttsim. ferrite has never run on silicon; a p150a is ordered.

- **Built:** a userspace PCIe driver, firmware on every Tensix core, a command ring that makes a launch one PCIe write,
  and a compiler from tensor expressions to Tensix instructions, written from the ISA documentation and, where it
  stops, ttsim's data.
- **Runs:** GPT-2 small, SmolLM-135M, SmolLM2-360M and SmolLM2-1.7B with their real weights, in bf16 and in block floats
  (bfp8; for SmolLM also the MLP's gate and up in bfp4, as Tenstorrent's own models keep them), each token checked
  against the same model in numpy, decode replayed from a recording; a small CNN; a small GPT trained in fp32 with
  Adam, step by step against PyTorch; tinygrad's own scripts on its FERRITE device; PyTorch through tinygrad's torch
  frontend. The test suite passes: 73 tests, the models, tinygrad and PyTorch included.
- **Against TT-NN on the same simulator:** 4 to 18 times fewer cycles of control work an op than TT-NN with fast
  dispatch (2 to 3.5 times fewer than with slow dispatch), and 9 to 10 times fewer a SmolLM-135M decode token
  (COMPARE.md). ttsim counts only that control work, not the math or the DRAM, so none of this says yet how fast either
  stack is on a card.
- **Not built:** mixture-of-experts models, attention that scales to long context, more than one card, and a Hugging
  Face loader and server.

## Focus

Tenstorrent's hardware with the software it deserves: machines of one, two or four Blackhole cards, linked card to card,
that run the largest open models from their Hugging Face ids, for teams that run AI on their own hardware.

- **For:** first, developers and small teams who host their own models: they buy one card, and tell others. Then
  companies that keep their data in-house. Government, sovereign and regulated buyers come last: they buy slowly.
- **Against:** what those buyers would buy instead: NVIDIA's DGX Spark (128 GB at 273 GB/s, $4,699) and its workstation
  GPUs, Apple's Mac Studio, AMD's Strix Halo machines, and Tenstorrent's own QuietBox 2 (four Blackholes, $9,999) with
  Tenstorrent's software.
- **We win on:** fast memory per dollar (four p150as are 128 GB at 2 TB/s for about $5,600 in cards); scaling card to
  card with no switch; open models running from their Hugging Face ids with no porting by the user; and a stack small
  enough to audit end to end.
- **We never build:** chips or cards of our own, training systems, consumer devices, or anything that races NVIDIA on
  peak compute.
- **Tenstorrent** is a partner, not a rival: every machine we sell is their cards sold.
- **Frontends:** tinygrad is the one framework frontend, and PyTorch comes through tinygrad's torch frontend. Customers
  get a Hugging Face loader and an OpenAI-compatible server. Whether the loader runs a model through tinygrad's model
  code or through ferrite's own is the loader decision, made in Stage 1.
- **Training:** ferrite keeps training correct (its own autograd and optimizers, in `test/train.py`, and tinygrad's
  training graphs, which run on it), but training speed is not a goal.

## Rules

1. **One stage at a time.** A stage's work starts only once the stage before it has passed its gate: every exit
   criterion ticked. Nothing else counts.
2. **Targets are fixed before the work.** A stage's exit criteria, and the benchmarks they name, are final the day the
   stage opens; until then they are drafts, and the models they name are re-pinned to their families' current releases
   when it opens. After that, a criterion changes only in writing, in this file, with the reason, and never because the
   work fell short of it.
3. **Exact in criteria, open in timing.** A stage takes as long as it takes; it is done when its gate passes, not on a
   date.
4. **A target the hardware cannot reach is still a result.** If a measurement shows the hardware itself cannot reach a
   target (a probe at the hardware's limit, not a slow kernel of ours), the criterion may be rewritten to that limit
   under rule 2, and the limit goes into LESSONS.md.
5. **Every January, recheck the bets** at the end of this file. A broken bet changes the plan before more work goes in.
6. **Measured, not projected.** A criterion checks numbers measured as the next section says, or computed from them by a
   formula it gives, unless the criterion says "projects". Every published number comes with what it takes to
   reproduce it.

## Measurement

What the words and numbers in the exit criteria mean.

- **Names.** The p150a is our card. A p150 is either version of it (the p150a, actively cooled, or the p150b, passive:
  the same chip and memory); P150 is how Tenstorrent's tables name it. A machine is a computer with one or more p150s,
  linked by QSFP-DD cables. TT is Tenstorrent. TT-NN is Tenstorrent's software stack: tt-metal with its TT-NN library
  and, for a model, the tt_transformers code that runs it there.
- **Versions.** Every number records the commits of ferrite and tt-metal (and so the UMD it pins), the versions of
  tt-kmd, the card's firmware, tinygrad and PyTorch where they take part, and the host's CPU, OS and kernel. The TT-NN
  numbers a gate compares with are measured on our p150s with Tenstorrent's own scripts (tt_transformers'
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
- **Several cards.** A benchmark on n cards takes their bandwidth as n times 512 GB/s. The link's latency is half the
  round trip of 32 bytes written from a Tensix core on one card to one on the other and written back, the median of
  1,000; its bandwidth is 1 GiB moved from one card's DRAM to the other's, over one cable and over every cable between
  them.
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

## Stage 1: One card

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
  needs no reboot, and a report of what each core of a hung worker waits on. Done: ttsim-chaos
  (`test/ttsim-chaos.patch`), `ferrite.probe`, `pci.reset`, `Runtime.report`.
- Bring the card up with NOTES.md's checklist, in its order. Run TT-NN's Llama-3.1-8B and matmul on the card first: that
  shows the card works and gives the baselines.
- Run Llama-3.1-8B on ferrite: Llama 3's RoPE scaling, its 128,256-token vocabulary and untied output layer, and 16 GB
  of bf16 weights to upload.
- Tune matmul toward the card's peak and decode toward its DRAM bandwidth, measuring what ttsim could not: reads in
  flight, block-float weights, fewer fidelity phases, and the cost of each launch.
- Measure B3 against TT-NN, so Stage 2 starts from a known gap.
- **The loader decision.** Run B2 through ferrite's own model code and through tinygrad's own LLM code
  (`tinygrad/llm`, which covers Llama 3 and Qwen 3, its mixture of experts included) on FERRITE, and decide which the
  Hugging Face loader runs. For a like-for-like comparison, tinygrad's path needs five things first (NOTES, "tinygrad on
  ferrite"): bf16 activations, or the adapter multiplying fp32 by bf16 as one bf16 matmul; a KV cache in a dtype the
  chip holds; replay of a decode step whose place is a symbolic variable; a format for each weight; and Llama 3.1's
  RoPE scaling, which tinygrad's LLM code ignores.
- Turn the hand check of PyTorch through tinygrad's torch frontend into an automated test
  (`adapters/torch_ferrite.py` is already gone). Done: `test_torch_frontend`.
- Add a LICENSE: Apache-2.0, as tt-metal's. Done.
- Publish B1, B2 and B3 for both stacks, each stack's lines of code beside them, and a script anyone with a p150 can
  rerun.

**Exit criteria** (fixed 2026-10-20)

- [ ] On the p150a, three runs in a row of `python3 -m pytest test` (with `GPT2`, `LLAMA`, `LLAMA8B` and `TINYGRAD`
      set, and everything CLAUDE.md lists for the tests installed) have no failures and no skips but
      `test_card_wants_root`, which skips as root by design.
- [ ] B2: decode at no less than 41.3 tokens/s (50% of the card's DRAM bandwidth), with top-1 at least 90% and top-5 at
      least 98%.
- [ ] B1 and B2: ferrite is no more than 1.5 times slower than TT-NN on the same card.
- [ ] B1, B2 and B3 are published for both stacks, with each stack's lines of code and the rerun script.
- [ ] The loader decision is written in this file, with both measurements and its reasons.
- [ ] A LICENSE is in the repository, and PyTorch through tinygrad's torch frontend is an automated test that passes on
      the card.

Amended on 2026-10-20, the day the stage opened: the first criterion names the suite's command as it runs
(`python3 -m pytest`, so that `ferrite` is importable) and its dependencies by reference, since the multiply-accumulate
unit's tests brought Verilator, cocotb and Yosys.

Amended on 2026-10-21: the plan no longer designs chips of our own, so the multiply-accumulate unit, the first step
toward one, left the stage's work and its gate. It had already passed (bit for bit against its model in cocotb, on a
million random inputs and on zeros, infinities, NaNs and denormals). Its code and tests went with it, so the suite no
longer needs Verilator, cocotb or Yosys. And the first criterion sets `LLAMA8B` too: `test_llama_8b_shapes`, added
after the stage opened, skips without it, which the criterion would not allow.

**Not yet:** a second card, customers.

## Stage 2: Two cards, first users

Two p150as joined by a QSFP-DD cable run a model too big for one, and other people run models on ferrite.

**Benchmarks** (drafts until the stage opens, as Measurement defines them)

- **B4, link:** the latency and bandwidth between two p150as.
- **B5, two-card decode:** `meta-llama/Llama-3.3-70B-Instruct` across two p150as at TT's performance precision (about
  55.2 GB a token in B2's formats), and B2's model split across both cards.

**Work**

- Buy a second p150a and a QSFP-DD cable (Tenstorrent sells a 0.5 m one for $200).
- Drive the link through Tenstorrent's Ethernet firmware, which runs on the card as ARC's does, and measure B4.
- Split a model across cards: each matmul's weights shared out between them, and each layer's partial results
  exchanged over the link, so each card reads its share of the weights.
- Attention for long context: one pass over the keys and values with a running max and sum (flash attention's method),
  shared out over every worker.
- Ship a Hugging Face loader and an OpenAI-compatible server, on one card or two, through the path the loader decision
  chose.
- Run TT-NN's Llama-3.3-70B on the same two cards if Tenstorrent's code runs it there, for comparison (not gated).

**Exit criteria** (drafts until the stage opens)

- [ ] Stage 1's suite criterion still holds on each card, and B4 is published.
- [ ] B5: Llama-3.3-70B decodes across two p150as at no less than 9.3 tokens/s (50% of their 1,024 GB/s), with top-1 at
      least 90% and top-5 at least 98%.
- [ ] B2's model, split across the two cards, decodes at no less than 1.6 times its speed on one.
- [ ] B3 at 32,768 tokens (or the longest prompt TT-NN runs on one p150a, if shorter): ferrite is no more than 1.5
      times slower than TT-NN on the same card.
- [ ] A Hugging Face model id becomes an endpoint of the OpenAI API (`/v1/completions` and `/v1/chat/completions`,
      streaming included, checked with OpenAI's own Python client) for Llama-3.1-8B and Qwen3-8B on one card and
      Llama-3.3-70B on two, each at the formats it is served in: the Llama models with B2's token accuracy, Qwen3-8B
      with top-1 at least 85% and top-5 at least 95%.
- [ ] At least 3 people outside the project have run a model on ferrite from its public instructions, and said so in
      writing.

**Not yet:** a third card, mixture-of-experts models, selling anything.

## Stage 3: Four cards, the machine

Four p150as in one computer, every pair linked, run the largest open models as one device, and buyers have seen it.

**Work**

- Buy two more p150as, five more cables, and a host with four PCIe x16 slots and power for about 1.5 kW.
- Four cards as one device: a model split four ways over the six links.
- Mixture of experts in the compiler: routing (top-k) on the chip, and expert weights picked by data on the chip, so a
  decode step still replays from a recording.
- Memory past the cards: the experts a model rarely uses kept in the host's memory and computed by its CPU, as
  KTransformers does beside GPUs, so a model bigger than the cards' 128 GB runs.
- The machine: a parts list anyone can buy, and setup in one command.
- Measure what buyers would buy instead, on the same models: DGX Spark, QuietBox 2 with Tenstorrent's software, and a
  workstation GPU.
- Talk to buyers in the market Focus names.

**Exit criteria** (drafts until the stage opens)

- [ ] `Qwen/Qwen3-30B-A3B`, its experts in bfp4 and its other weight matrices in bfp8, decodes on one p150a at no less
      than 87 tokens/s: 40% of the card's bandwidth at about 2.36 GB a token (40% rather than Llama's 50%: a mixture of
      experts reads many small matrices, picked at run time).
- [ ] `openai/gpt-oss-120b` decodes across four p150as at no less than 168 tokens/s (30% of their 2,048 GB/s at about
      3.66 GB a token, its experts in bfp4), and Llama-3.3-70B at no less than 18.5 (50%); the Llama with B2's token
      accuracy, gpt-oss with top-1 at least 85% and top-5 at least 95%.
- [ ] `Qwen/Qwen3-235B-A22B` (about 136 GB with its experts in bfp4, more than the four cards hold) runs on the
      machine with the host's memory, and its speed and token accuracy are published.
- [ ] The machine's tokens/s per dollar on those models is published beside DGX Spark's and QuietBox 2's with
      Tenstorrent's software (measured where we have the machine, otherwise from a named public source, each price as
      listed), and it leads both on at least one of them.
- [ ] Someone outside the project has built the machine from the parts list and run a model on it from the
      instructions.
- [ ] 10 conversations with buyers are written up, and at least 3 of them say in writing that they would buy a machine.

**Not yet:** selling machines.

## Stage 4: Machines sold

People pay for machines of linked p150s running ferrite, and Tenstorrent sells us the cards on terms that leave a
margin.

**Work**

- Build and sell machines of one, two and four p150s with ferrite installed, and support them.
- Agree reseller terms or volume prices with Tenstorrent.
- For bigger buyers, 8 or 16 cards over QSFP-DD across machines: 16 cards with four ports each make a four-dimensional
  hypercube, 512 GB at 8 TB/s with no switch.
- When buyers want more in one machine, talk with Tenstorrent about the p300 (two Blackholes on one card).

**Exit criteria** (drafts until the stage opens)

- [ ] 10 machines are sold to paying customers, each at a positive margin.
- [ ] A reseller or volume agreement with Tenstorrent is signed.

**Not yet:** what comes after this file.

## Bets to recheck every January

ferrite and its machines are built for the models people run now and next. These are the assumptions they rest on.

| Bet | Sign it broke | What changes |
| --- | --- | --- |
| Generating tokens stays limited by memory, not math | Popular open models become compute-bound at decode | Tune for compute; sell on prefill and batch throughput |
| Open mixture-of-experts models keep winning | Leading open releases go back to dense models | Capacity matters less, bandwidth per card more |
| Agents keep pushing context length up | Typical prompt lengths stop growing | Long-context prefill matters less |
| Owning your AI keeps getting more important | Cloud prices fall until on-prem stops paying off for our buyers | Revisit who the machines are for |
| Tenstorrent keeps selling linked p150s at about today's price | The p150 is discontinued, its ports dropped, or its price far higher | Move to the card that replaces it, the p300 or later |
| Tenstorrent's own stack stays heavy | Tenstorrent ships a small, fast stack of its own | Compete on machines and support, or work with them on theirs |

## Sources

- [Tenstorrent Blackhole cards](https://tenstorrent.com/hardware/blackhole): the p150a at $1,399, with 120 Tensix cores,
  180 MB of SRAM, 32 GB of GDDR6 at 512 GB/s and four QSFP-DD 800G ports.
- [Lightning V2 on Tenstorrent](https://arxiv.org/pdf/2604.03279v1): the P150 at $1,400, and its DRAM at 512 GB/s
  (Tables 1 and 3).
- [GPU Smith's p150 datasheet](https://gpusmith.com/hardware/gpus/tenstorrent-blackhole-p150): the QSFP-DD ports link
  Blackhole cards directly, over passive cables up to 2 m; [Tenstorrent's 800G QSFP-DD
  cable](https://shop.tenstorrent.com/products/qsfp-dd-cables-800g), 0.5 m, $200.
- [GPU Smith's QuietBox datasheet](https://gpusmith.com/hardware/systems/tenstorrent-quietbox): QuietBox 2, $9,999, four
  Blackholes on two p300c cards, 128 GB of GDDR6, 256 GB of DDR5 in the host.
- [DGX Spark's prices](https://glukhov.org/hardware/ai/nvidia-dgx-spark-prices/): 128 GB of LPDDR5X at 273 GB/s, $3,999
  at launch and $4,699 from February 2026.
- [KTransformers](https://github.com/kvcache-ai/ktransformers): a mixture of experts' weights split between a GPU and
  the host's CPU and memory.
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
- [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct),
  [Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct),
  [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B),
  [Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B) and
  [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b): the shapes behind this file's byte counts.
- [tinygrad](https://github.com/tinygrad/tinygrad)
