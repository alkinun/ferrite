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
- [x] **Attention over a session's own rows** in a batched step, a key head's rows read once for all its query heads,
  where they were read once a query head: eight times for Qwen3.6 (NOTES, "A Llama-family model"). Its two moves a
  block cost more than they save at short contexts on ttsim, a datum at a time (NOTES, "Next for the compiler", 6).
- [x] **Fewer launches a Gated DeltaNet block**: a decode or verify step's tokens one at a time by the recurrent rule,
  and its small weights in fp32, 86 launches a block to 20 (27 for several slots), a verify step's 130 to 50; a
  Qwen3.6 step is 1,701 launches (NOTES, "Qwen3.5's and Qwen3.6's language models").
- [x] **A put by the rows it writes**: the KV cache's `put` visited every tile of the cache to find a step's rows, 2.5M
  of a 32-user SmolLM step's 3.4M ttsim cycles at 1,056 places a user; it now works as the rows it writes, 159k
  (NOTES, "Rows by index").
- [x] **Qwen3.6's KV cache in bfp8** (`serve.py`'s last argument): 25 sessions a card where 14 were, 283 tokens/s a card
  at eight users. The format alone keeps Llama-3.1-8B's token accuracy (top-1 99.6% against 100%) and moves Qwen3.6's
  real blocks' logits 0.7%, a quarter of what bfp8 weights do (NOTES, "Qwen3.5's and Qwen3.6's language models"); the
  whole model's token accuracy waits for the whole model, as its experts' formats do.
- [x] **Usage as OpenRouter bills it**: each response's usage counts the prompt's cached tokens (those its slot or a
  saved session held, not prefilled), and a stream ends with it. OpenRouter sends an agent's next requests back to the
  provider that cached its last, when cached tokens cost less there (NOTES, "Serving").
- [x] **The sampling parameters OpenRouter passes**: top_k, min_p, the presence, frequency and repetition penalties, and
  log probabilities (logprobs, top_logprobs), on the host from each token's logits, a token's sampling 1.3 ms where its
  whole sort of Qwen3.6's vocabulary was 6 (NOTES, "Serving").
- [x] **tool_choice**: none, required or a named function, as a shape the answer keeps to: a token drawn is taken if
  the text it makes still fits, else drawn again without it (NOTES, "Serving").
- [x] **Structured outputs**: answers to a JSON schema (response_format) as a shape, its prefix parsed as it comes; no
  start of one leads nowhere, and whitespace and numbers are bounded, which a shape's model would otherwise run on in
  (NOTES, "Serving").
- [x] **Less control a Gated DeltaNet block**: its conv's inputs laid out as the step's tokens are, a decode step's
  conv one kernel over them, so a token of a step adds 17k ttsim cycles to a block where it added 26k, and an
  eight-user step's control is 4.3 ms where it was 5.9 (NOTES, "Qwen3.5's and Qwen3.6's language models", "The cost
  model"); most of the rest is each slot's heads moved into batch entries and out.
- [x] **Cheaper moves**: a run four datums at a time and bf16's two to a word, a one-row tile's zeros its row's alone,
  and the tile cache's record kept between launches: a 32-user SmolLM step's moves 405k ttsim cycles where 586k, a
  Qwen3.6 step's control for eight users 3.3 ms where 3.7 (NOTES, "Next for the compiler", 6). What is left is each
  result tile's own work: the zero tile, the fetch, a row's copies down the tile, the write home.
- [x] **Attention over what a slot holds**: a step's attention reads each slot's first rows alone, up to the power of 2
  that holds the longest sequence's (a replay recorded for each), and a prompt's chunk its own slot's up to its last
  place, where every call read every slot whole: a slot can be as long as Qwen3.6's 262K places without every step
  paying for them (NOTES, "Serving").
- [x] **The cost of a prompt and of attention** (`models/cost.py`'s `prompt` and `scores`): a chunk of a prompt reads
  the experts its tokens pick, nearly all of every block's at the server's 256 tokens; and attention's scores go
  through DRAM six times, as much again as the cache rows a step reads (2.0 GB of an eight-user step's 11.5: 226
  tokens/s a card where the model said 268), and a chunk reads each key head's rows once a query head and 128 queries:
  a 32K-token prompt of Qwen3.6 takes 6.7 s, where its experts and rows alone would take 1.1 (NOTES, "The cost model").
- [x] **Attention in one kernel** (`Tensor.attend`): softmax(q kᵀ + mask) v with its scores kept in L1, two sweeps
  over a frame of keys at a time, the output added onto in L1 by the packer in fp32; a unit of two tile rows of queries
  shares its keys and values. At Qwen3.6's shapes a decode step's attention reads 1.7 times its cache rows where the
  softmax and the matmul read 2.5, with a fifth of the RISC-V work, and a prompt's chunk 59 times where they read 117
  (NOTES, "Attention").
- [x] **Attention by places, a prompt's in units of eight tile rows**: each unit reads the tiles of keys its rows'
  places reach, and the mask only where some row sees a tile in part, so each slot of a step reads its own rows however
  long the others' are; a prompt's unit takes 256 queries of a head, their running maximums in L1 and their sums made
  as products by a tile of ones. At Qwen3.6's shapes a decode step's attention reads 1.5 times its rows where 1.7, and
  a prompt's chunk 17 times where 59; the cost model's eight users make 237 tokens/s a card where 224, and a 32K-token
  prompt takes 2.1 s where 6.9 (NOTES, "Attention").
- [x] **A KV cache in bfp4** (`put` writes it, attention reads it in place, bit for bit the bf16 one by the same
  values): 42 sessions a card where bfp8 keeps 25, and 284 tokens/s a card at eight users where 237, if the whole
  model's token accuracy holds with it, which tonight's run of it says (NOTES, "Qwen3.5's and Qwen3.6's language
  models").
- [ ] **One card or two**: its experts in bfp4 against bf16, the token accuracy in numpy over the whole model. Each
  matrix moves as much in bfp4 as Llama-3.1-8B's MLP does in Tenstorrent's performance mode, but on the real blocks 2
  and 3 every MLP matrix in bfp4 moves the logits 16% where gate and up alone move them 8% (with down in bfp8 the
  model is about 27 GB, still one card); the whole model is 72 GB in bf16, and the token accuracy waits for it on a
  disk with room.
