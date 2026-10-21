# ferrite roadmap

Tenstorrent's Blackhole cards are fast, affordable and open down to their instruction set, but the software that drives
them is heavy. ferrite is a small stack for them. The plan: make one card fast, join cards through their QSFP-DD ports
into one device, and serve large open models to the public. A stage is done when its checks pass on the cards, with the
numbers published (COMPARE.md says how they are measured).

## Stage 1: One card

ferrite runs correctly and fast on a p150.

- [ ] The test suite passes on the card.
- [ ] Llama-3.1-8B decodes at half the card's DRAM bandwidth (41.3 tokens/s), at Tenstorrent's token accuracy.
- [ ] B1 and B2 are no more than 1.5 times slower than TT-NN on the same card, and B1 to B3 are published for both.

## Stage 2: Two cards

Two p150s joined by a QSFP-DD cable work as one device.

- [ ] The link's latency and bandwidth are measured.
- [ ] A model too big for one card (a 70B) runs across both, and one that fits runs faster split across two.
- [ ] Long prompts (B3) are no more than 1.5 times slower than TT-NN.

## Stage 3: Four cards

Four linked p150s in one machine serve many users at once.

- [ ] Mixture-of-experts models run on the chip, up to ones that need all four cards.
- [ ] An OpenAI-compatible server takes a Hugging Face model id and batches many requests, with a prefix cache.

## Stage 4: Serving

The four-card machine serves an open model to the public as an OpenRouter provider.

- [ ] It keeps the uptime OpenRouter routes normally at (95%).
- [ ] Its speed and its cost per million tokens are published beside the model's other providers.

## Stage 5: Eight cards

Eight linked p150s serve mixture-of-experts models of 200 to 400 billion parameters.

- [ ] One such model is served on OpenRouter at a speed and price that hold up against its other providers.
