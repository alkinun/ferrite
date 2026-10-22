# ferrite roadmap

Tenstorrent's Blackhole cards are cheap and open down to their instruction set, and their software holds them back.
ferrite is a small stack for them, and the plan is to show what that software costs: make one card fast, serve the
open models agents run on through OpenRouter more cheaply than providers on GPUs can, then join cards through their own
Ethernet ports for the big ones. A stage is done when its checks pass on the cards, with the numbers published
(COMPARE.md says how they are measured). The stages say what done means; "Next" is the order of the work.

## Why this

- **The demand.** OpenRouter's volume (October 2026) is in open mixtures of experts, and agents and code are 58% of
  its spend. A harness takes whichever provider is cheap, healthy and right in its tool calls: several to dozens serve
  each model, small and new ones among them, some at 30 to 60 tokens/s a user.
- **The first model: Qwen3.6-35B-A3B.** 35B parameters with 3B active (Gated DeltaNet and gated attention, 256 experts
  and a shared one), Apache-2.0, 262K tokens of context, thinking and tool calls: 786B tokens a week through OpenRouter
  from ten providers, at $0.05 to $0.25 a million input tokens and $0.70 to $1.80 output, 46 to 186 tokens/s a user. It
  fits one card with its experts in bfp4 (about 22 GB) or two in bfp8 (37 GB): a card or a pair is a whole server, with
  no links between them, and a box of them is that many copies, one failing while the others serve.
- **The edge, on paper.** A card costs about $0.13 an hour to own and run, and at 8 users would make 350 to 500
  output tokens/s: cents a million tokens, where the cheapest provider charges $0.70. Against an H100 rented at $2 an
  hour, a card has per dollar an hour about twice the memory bandwidth (what a generated token costs), four times the
  compute (an uncached prompt token) and six times the memory (sessions' caches between an agent's turns), and a
  provider can start with one card where a GPU wants filling. So: the cheapest, at a speed within the others' range,
  and exact against the reference model.
- **Then the big ones.** The open "flash" models are where most of the volume is: DeepSeek V4.1 Flash (about 42T tokens
  a week), GLM-5.3-Flash (11T) and MiMo-V2.6-Flash (8.6T), each 284B to 320B parameters with 13B to 18B active. A copy
  needs 8 to 16 linked cards, and DeepSeek's and MiMo's experts come in FP4, which a GPU reads at 4 bits and bfp4 cannot
  hold (it moves 17% of their weights; bfp8 holds them exactly, at twice the bytes: NOTES, "Weights in FP4").
- **Not now.** Image and video generation, where the compute edge is largest but demand through OpenRouter is thin
  and Tenstorrent with Prodia already chase it. A chip of our own: what the cards teach about the hardware goes into
  LESSONS.md as it comes.

## Stage 1: One card

ferrite runs correctly and fast on a p150.

- [ ] The test suite passes on the card.
- [ ] The probe's numbers are measured (DRAM GB/s, matmul, unpack, pack and SFPU rates, launch cost, PCIe), and the
  cost model is filled in with them.
- [ ] Llama-3.1-8B decodes at half the card's DRAM bandwidth (41.3 tokens/s), at Tenstorrent's token accuracy.
- [ ] B1 and B2 are no more than 1.5 times slower than TT-NN on the same card, and B1 to B3 are published for both.

## Stage 2: Qwen3.6-35B-A3B

The first model to serve runs on one card, or on two.

- [ ] Its blocks (Gated DeltaNet, gated attention with partial RoPE, its experts) match transformers' with the real
  weights.
- [ ] Its experts in bfp4 keep the bf16 model's token accuracy, so it runs on one card; else it runs on two in bfp8.
- [ ] Its multi-token prediction layer decodes speculatively.
- [ ] The server serves it with tool calls and reasoning, keeping each session's caches and states in host memory
  between its turns, up to 262K tokens of context.
- [ ] Its tokens/s a user and a card, and its cost a million input, cached and output tokens, are measured on the card.

## Stage 3: Serving

Qwen3.6-35B-A3B is served to the public through OpenRouter.

- [ ] Its uptime is what its other providers show (99.5% or better).
- [ ] Its tool calls succeed as often as the other providers' (OpenRouter's Auto Exacto).
- [ ] It is the cheapest of the model's providers, at a speed within their range.
- [ ] It takes OpenRouter's batch jobs.
- [ ] More models of its size are served the same way, each on a card or two.

## Stage 4: Two cards

Two p150s joined by a QSFP-DD cable work as one device.

- [ ] The link's latency and bandwidth are measured.
- [ ] A mixture of experts runs with its experts split across the two cards, its logits the reference's.
- [ ] A model that fits on one card decodes faster split across two.

## Stage 5: The flash models

DeepSeek V4.1 Flash runs on a box of linked p150s and is served beside Qwen3.6.

- [ ] Its experts are held exactly in bfp8, or in bfp4 with the rounding shown harmless to its token accuracy.
- [ ] The whole model has the reference's token accuracy, and decodes speculatively with its multi-token prediction.
- [ ] Its cost a million tokens is measured on the box, and it is served through OpenRouter.
- [ ] A second box serves too, so one can fail without the provider going down.

## Next

The work in order, each step with its test on ttsim, ticked here as it lands. When the card comes, its bring-up goes
first (NOTES, "Bring-up on a p150").

- [x] **Qwen3.6's blocks** in `models/llama.py`: Gated DeltaNet (a chunk of tokens at a time, the state carried between
  chunks, a decode step a chunk of one), gated attention with partial RoPE, its experts with the shared one; real
  blocks within 0.22% of transformers in numpy and within about 1% a block on ttsim (NOTES, "Qwen3.5's and Qwen3.6's
  language models").
- [x] **Sessions**: several sequences at once, each with its own states; each session's cache rows and DeltaNet states
  kept on the host between its turns and brought back for the next; the server's slots for them, and tool calls in
  Qwen's template (NOTES, "Serving").
- [x] **Speculative decoding** with its multi-token prediction layer, a rejected draft's DeltaNet states put back
  (NOTES, "Qwen3.5's and Qwen3.6's language models", "Serving"); how many drafts the real layer gets right waits for
  the whole model.
- [x] **The cost model**: what a model reads and holds a token and how many sessions a card keeps, and with the probe's
  numbers, tokens/s and dollars a million tokens (`models/cost.py`; NOTES, "The cost model").
- [ ] **Attention over a session's own rows** in a batched step, where it reads every session's: four times the step at
  eight users with 32K-token sessions.
- [ ] **Fewer launches a Gated DeltaNet block** than its 86 (the chunk's small ops, a kernel each): a quarter of a
  one-user step.
- [ ] **Qwen3.6's KV cache in bfp8**, its token accuracy kept: a session's rows, most of what a step reads, halved.
- [ ] **One card or two**: its experts in bfp4 against bf16, the token accuracy in numpy over the whole model. Each
  matrix moves as much in bfp4 as Llama-3.1-8B's MLP does in Tenstorrent's performance mode; the whole model is 72 GB
  in bf16, and the token accuracy waits for it on a disk with room.
