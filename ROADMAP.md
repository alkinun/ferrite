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
- **The edge, on paper.** A card costs about $0.13 an hour to own and run, and at 8 users would make 230 to 320
  output tokens/s (the cost model, sessions of 32K and 8K tokens): about $0.15 a million tokens, where the cheapest
  provider charges $0.70. Against an H100 rented at $2 an
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

The work in order, each step with its test on ttsim. A step leaves this list once it lands; git's log and NOTES keep
what it did and what it measured.

Since the stages were set: Qwen3.6's blocks, sessions kept on the host, speculative decoding and the cost model;
attention in one kernel, by the places its rows reach, in parts for a decode step's few units (flash decoding); a KV
cache in bfp8 or bfp4; fewer launches and less control in Gated DeltaNet, cheaper moves, a put by the rows it writes;
and in the server what OpenRouter passes and expects: sampling parameters and log probabilities, tool_choice,
structured outputs, cached tokens in the usage, an early 429 when every slot is busy.

- [ ] **One card or two**: the whole model's token accuracy in numpy, Tenstorrent's way (`cmp/qwen36/accuracy.py`).
  bf16 is top-1 99.6%, top-5 100% against the reference; one card's layouts cost 6 to 8 points of top-1: every expert
  matrix in bfp4 (21.7 GB) 92.0% and 94.2%, the experts' gate and up in bfp4 and down in bfp8 (27.2 GB) 93.8% and
  96.0%, where Tenstorrent's own performance mode runs Llama-3.1-8B at 90% top-1. Every matrix in bfp8 (two cards) and the KV cache in bfp4 wait on a rerun: their first runs came out NaN, it
  seems from this host's instability (NOTES, "Still unknown or untested").
- [ ] **How often the drafts are right** (`cmp/qwen36/drafts.py`, after the accuracy runs): speculative decoding pays
  above about 45% at one user and 55% at eight (NOTES, "The cost model").
- [ ] **The card's bring-up**, when it comes, on a host that no longer crashes: NOTES' "Bring-up on a p150", steps 0
  to 12, then the probe's numbers into the cost model (Stage 1).
- [ ] **Qwen3.6 on the card**: its weights uploaded in the layout the accuracy chose, a decode step's tokens/s and
  cost a million measured against the cost model's (Stage 2's last check), and served on the host to a client of our
  own before OpenRouter's (Stage 3).
- [ ] **Gated DeltaNet's turns in one kernel**: a prompt's state goes through its pieces of 64 at 35 launches a piece,
  some 1,330 a block for a chunk of 2,048 tokens (NOTES, "Qwen3.5's and Qwen3.6's language models").
- [ ] **Attention in one sweep**: the keys read once, not twice, the output rescaled in L1 as a row's largest grows,
  and a prompt's unit sharing its key head's rows among that head's query heads (NOTES, "Attention").
