# What a model costs to serve on Blackhole cards (ROADMAP, Next), from its config alone (llama.config): what a card
# holds of it (its weights in the formats it runs in; each session's KV cache rows and Gated DeltaNet states), what a
# decode step reads, computes and launches (its weights, the experts its tokens pick, its sessions' rows and states),
# and so how many sessions a card keeps, tokens/s for a user and for a card, and dollars a million tokens. The card's
# rates are its specification's until the probe has measured them (python3 -m ferrite.probe 8 probe.json, then
# probe=probe.json here); the launches a block takes are ferrite's own, counted on ttsim. A model of what ferrite would
# do: the card says how far off it is.
#   python3 models/cost.py path/to/model [bf16|bfp8|bfp4|experts] [key=value ...]   # keys: CARD's and USE's
import json, math, os, sys
from types import SimpleNamespace
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ferrite import bh
import llama

BYTES = {"bf16": 2.0, "bfp8": 1088 / 1024, "bfp4": 576 / 1024}  # a value's, with its share of its tile's exponents
MODES = {"bf16": {}, "bfp8": {"*": "bfp8"},  # each kind of matrix's format, Tenstorrent's performance precision's, and with
         "bfp4": {"*": "bfp8", "gate_up": "bfp4"},  # every expert matrix in bfp4 (Qwen3.6 on a card). The embedding stays
         "experts": {"*": "bfp8", "gate_up": "bfp4", "down": "bfp4"}}  # bf16 (a token's row is gathered), and so do the routers
LAUNCHES = {"attention": 12, "experts": 18, "deltanet": 27, "deltanet verify": 57, "rest": 11, "draft": 45}  # a decode step's
# for several slots: an attention block's (with its norms), a mixture of experts' more, a Gated DeltaNet block's (and one
# taking a token and a draft at once), the rest (the embedding's gather, the step's masks, the final norm and the output
# layer), the multi-token prediction layer's call. Counted on ttsim at Qwen3.6's shapes, four slots (cmp/qwen36/launches.py).
CONTROL = {"deltanet": (20600, 12400), "attention": (18100, 7600), "rest": (2900, 1200),  # a decode step's RISC-V work as
           "deltanet verify": (44200, 13600), "attention verify": (30200, 6400), "rest verify": (6100, 800)}  # ttsim counts it,
# and a verify step's: a Gated DeltaNet block's (with its experts), an attention block's (with its experts), the rest's, so
# many cycles and so many more a token (cmp/qwen36/control.py at Qwen3.6's shapes, one slot and eight: moves, most of it)
WORKERS = 119  # a p150's 120 Tensix tiles but the dispatcher
CARD = dict(dram=len(bh.DRAM) * 4 * 2**30, reserve=2**30,  # a p150: 8 GDDR6 banks of 4 GiB, a GiB of it for kernels and activations
            bandwidth=bh.DRAM_GBPS * 1e9, reach=0.7,  # its 512 GB/s, and the share of it a decode step reaches (a guess)
            tflops=664e12, phases=2,  # bfp8 matmuls at one fidelity phase (16 cycles a step), and the phases they take
            launch=1e-6, pcie=25e9,  # a launch on the card (NOTES, "Next for the compiler"), and PCIe 4.0 x16's bytes/s
            price=1299, host=600, years=3, watts=300, host_watts=75, kwh=0.15)  # the p150a at its launch, a share of a host,
            # three years; the card's power and its host's share, at $0.15 a kWh
USE = dict(context=32768, kv="bf16", accept=0.0)  # each session's places, its slot's, all of them filled (a step reads each slot's
# rows up to the power of 2 that holds the longest session's), and its KV cache's format; the share of drafts taken

def probed(card, path):  # the probe's numbers (python3 -m ferrite.probe 8 path) in place of the specification's
  p = json.load(open(path))
  return card | dict(bandwidth=max(v for k, v in p.items() if k.startswith("DRAM reads") and k.endswith("all workers")) * 1e9, reach=1.0,
                     tflops=2 * 32**3 / p["matmul step, 1 phase, unpacks included"] * WORKERS * bh.AICLK,
                     launch=p["launch, 100 queued: chip each"] / bh.AICLK, pcie=min(p["upload over PCIe"], p["read back over PCIe"]) * 1e9)

def shape(c, mode="bf16"):
  """A model's matrices in mode's formats, as a decode step meets them: the bytes every step reads (all but the experts
  and the embedding), one expert's bytes and the blocks with experts, the multi-token prediction layer's bytes but its
  experts (and its blocks), the embedding's (held, a row read), and the values a token's matmuls take."""
  d, hd, fmt = c.dim, llama.head(c), lambda kind: BYTES[MODES[mode].get(kind, MODES[mode].get("*", "bf16"))]
  attention = (d * c.heads * hd * (2 if c.gated else 1) + 2 * d * c.kv * hd + c.heads * hd * d) * fmt("attention")
  delta = (d * (2 * c.kh * c.kd + 2 * c.vh * c.vd + 2 * c.vh) + c.vh * c.vd * d) * fmt("deltanet") if c.kh else 0
  e, f, m, sh = c.experts or 0, c.ffn or 0, c.moe or 0, c.shared or 0
  mixture = 3 * d * sh * fmt("shared") + d * e * BYTES["bf16"]  # a block's shared expert and router
  moe = [l for l in range(c.layers) if e and l not in c.dense]
  every = (sum(delta if llama.linear(c, l) else attention for l in range(c.layers)) + len(moe) * mixture
           + (c.layers - len(moe)) * d * f * (2 * fmt("gate_up") + fmt("down")) + c.vocab * d * fmt("head"))
  draft = c.mtp * (attention + mixture + 2 * d * d * fmt("fc"))
  values = (sum((d * (2 * c.kh * c.kd + 2 * c.vh * c.vd + 2 * c.vh) + c.vh * c.vd * d) if llama.linear(c, l) else attention / fmt("attention") for l in range(c.layers))
            + len(moe) * (3 * d * (sh + (c.k or 0) * m) + d * e) + (c.layers - len(moe)) * 3 * d * f + c.vocab * d)
  return SimpleNamespace(every=every, expert=d * m * (2 * fmt("gate_up") + fmt("down")), moe=len(moe), draft=draft, embed=c.vocab * d * BYTES["bf16"],
                         bytes=every + (len(moe) + c.mtp) * e * d * m * (2 * fmt("gate_up") + fmt("down")) + draft + c.vocab * d * BYTES["bf16"],
                         values=values, held=values + (len(moe) + c.mtp) * (e - (c.k or 0)) * 3 * d * m + c.mtp * (attention / fmt("attention") + 2 * d * d)
                         + c.vocab * d * (0 if c.tied else 1))

def session(c, use, speculative=False):
  """A session's bytes: its KV cache rows, the model's (each attention block's, a sliding window's at most) and the
  multi-token prediction layer's, and its Gated DeltaNet states (twice when speculating: after a draft and before)."""
  window = lambda l: c.sliding if c.sliding and (not c.layer_types or c.layer_types[l] == "sliding_attention") else use["context"]
  row = 2 * c.kv * llama.head(c) * BYTES[use["kv"]]
  kv = sum(row * min(use["context"], window(l)) for l in range(c.layers) if not llama.linear(c, l))
  state = sum(4 * (c.vh * c.kd * c.vd + (c.conv - 1) * (2 * c.kh * c.kd + c.vh * c.vd)) for l in range(c.layers) if llama.linear(c, l))
  return SimpleNamespace(kv=kv, draft=c.mtp * row * use["context"], state=state * (1 + speculative), bytes=kv + c.mtp * row * use["context"] + state * (1 + speculative))

def touched(c, tokens): return c.experts * (1 - (1 - c.k / c.experts) ** tokens) if c.experts else 0  # a block's experts some token picks

def scores(c, entries, queries, rows):  # the bytes an attention block's scores move through DRAM, as ferrite computes them: a
  # tile of 32 queries by 32 rows for each of `entries` heads (or key heads, their query heads' rows side by side) is
  # written by the first matmul and read by the softmax, with the mask's, then the softmax's written and read by the
  # second matmul once for each block of 4 of its result's column tiles (a head's 256 values: twice)
  return (4 + -(-llama.head(c) // 128)) * entries * -(-queries // 32) * -(-rows // 32) * 2048

def step(c, s, use, card, users, speculative=False, cards=1):
  """A decode step for `users` sessions at once, a token each, or a token and a draft each then the next drafts (two
  calls): the bytes it reads, its flops, launches, control and seconds (waiting on DRAM or on the matmuls, whichever is
  longer, then on the launches or the kernels' own RISC-V work, if the card's counts as ttsim's does, whichever is
  more: the small kernels that do it read little), and tokens/s a user."""
  n, one, d = users * (1 + speculative), session(c, use, speculative), c.dim
  attention = sum(not llama.linear(c, l) for l in range(c.layers))
  reads = s.every + s.moe * touched(c, n) * s.expert + n * one.kv + users * one.state * (1.5 if speculative else 1)  # states read and written, twice
  reads += attention * scores(c, c.kv * users, c.heads // c.kv * (1 + speculative), use["context"])                # when speculating; scores
  flops = 2 * n * s.values + n * sum(4 * c.heads * llama.head(c) * use["context"] for l in range(c.layers) if not llama.linear(c, l))
  linear = sum(llama.linear(c, l) for l in range(c.layers))
  launches = (c.layers - linear) * LAUNCHES["attention"] + linear * LAUNCHES["deltanet verify" if speculative else "deltanet"]
  launches += s.moe * LAUNCHES["experts"] + LAUNCHES["rest"]
  cycles = lambda kind, blocks: blocks * (lambda a, b: a + b * n)(*CONTROL[kind + " verify" if speculative else kind])
  control = (cycles("attention", c.layers - linear + c.mtp * speculative) + cycles("deltanet", linear) + cycles("rest", 1 + speculative)) / bh.AICLK
  if speculative:  # the drafting call: the layer over two places a user, and the output layer again
    reads += s.draft + c.mtp * touched(c, 2 * users) * s.expert + c.vocab * d * BYTES["bfp8"] + 2 * users * one.draft
    reads += c.mtp * scores(c, c.kv * users, 2 * c.heads // c.kv, use["context"])
    flops += 2 * 2 * users * (s.draft / BYTES["bfp8"] + c.k * 3 * d * (c.moe or 0) + c.vocab * d)
    launches += LAUNCHES["draft"]
  seconds = max(reads / (card["bandwidth"] * card["reach"] * cards), flops / (card["tflops"] / card["phases"] * cards)) + max(launches * card["launch"], control)
  return SimpleNamespace(reads=reads, seconds=seconds, launches=launches, control=control, flops=flops, rate=(1 + use["accept"] if speculative else 1) / seconds)

def prompt(c, s, use, card, chunk, cards=1):
  """A prompt that fills a session (use's context), prefilled chunk tokens at a time as serve.py's engine takes it: each
  chunk reads every matrix but the experts and the embedding, the experts its tokens pick, and its session's rows up to
  the power of 2 that holds its last place (each key head's once for each of its query heads and each block of 4 tiles
  of 32 queries, as the matmuls' blocks read them), moves its scores through DRAM, and computes its tokens' matmuls (the
  output layer's for its last alone) and their scores over those rows; with a multi-token prediction layer, that
  layer's over each place too. Its seconds (each chunk waiting on DRAM or on the matmuls, whichever is longer; launches and control left out),
  tokens/s, and the most bytes a chunk's scores take (each attention block's, and their softmax)."""
  hd, d, n = llama.head(c), c.dim, use["context"]
  blocks = sum(not llama.linear(c, l) for l in range(c.layers)) + c.mtp
  seconds, most = 0.0, 0
  for at in range(0, n, chunk):
    m = min(chunk, n - at)
    rows = min(n, 1 << max(5, (at + m - 1).bit_length()))
    reads = s.every + s.moe * touched(c, m) * s.expert + blocks * (-(-m // 128) * c.heads * hd * 2 * BYTES[use["kv"]] * rows + scores(c, c.heads, m, rows))
    flops = 2 * m * (s.values - c.vocab * d) + 2 * c.vocab * d + blocks * 4 * m * c.heads * hd * rows
    if c.mtp:
      reads += s.draft + c.mtp * touched(c, m) * s.expert + c.vocab * d * BYTES["bfp8"]
      flops += 2 * m * (s.draft / BYTES["bfp8"] + c.k * 3 * d * c.moe + c.vocab * d)
    seconds += max(reads / (card["bandwidth"] * card["reach"] * cards), flops / (card["tflops"] / card["phases"] * cards))
    most = max(most, 2 * c.heads * m * rows * BYTES["bf16"])
  return SimpleNamespace(seconds=seconds, rate=n / seconds, scores=most)

def hourly(card): return (card["price"] + card["host"]) / (card["years"] * 8766) + (card["watts"] + card["host_watts"]) / 1000 * card["kwh"]

def report(folder, mode="bf16", **keys):
  """The cost model's table for a model's folder (its config.json), mode's formats, and CARD's and USE's keys."""
  card, use = dict(CARD), dict(USE)
  for k, v in keys.items():
    if k != "probe": (card if k in card else use)[k] = v if k == "kv" else float(v)
  if "probe" in keys: card = probed(card, keys["probe"])
  use["context"] = int(use["context"])
  c, name = llama.config(folder), os.path.basename(os.path.normpath(folder))
  s = shape(c, mode)
  cards = math.ceil(s.bytes / (card["dram"] - card["reserve"]))
  room, hour = cards * (card["dram"] - card["reserve"]) - s.bytes, hourly(card) * cards
  out = [f"{name}: {s.held / 1e9:.1f}B parameters, {s.values / 1e9:.2f}B a token; in {mode} {s.bytes / 1e9:.1f} GB on {cards} card{'s' * (cards > 1)}, "
         f"{room / 1e9:.1f} GB to spare, ${hour:.3f} an hour"]
  for speculative in (False, True) if c.mtp else (False,):
    one = session(c, use, speculative)
    most = max(int(room // one.bytes), 1)
    out.append(f"  a session of {use['context']} tokens: {one.kv / 1e9:.2f} GB of KV cache rows ({one.kv / use['context'] / 1e3:.1f} KB a token, {use['kv']})"
               + (f" and {one.state / 1e6:.0f} MB of Gated DeltaNet states" if one.state else "") + f": {most} on the card{'s' * (cards > 1)}")
    out.append(f"  decode, {'a token and a draft' if speculative else 'a token'} a user a step" + (f", {use['accept']:.0%} of drafts taken" if speculative else "") + ":")
    out.append("     users   reads, GB   step, ms   launches   control, ms   tokens/s a user   a card   $ a million out" + ("   drafts to take, to pay" if speculative else ""))
    for users in [u for u in (1, 2, 4, 8, 16, 32, 64, 128) if u <= most]:
      t = step(c, s, use, card, users, speculative, cards)
      pays = f"{step(c, s, use, card, users, True, cards).seconds / step(c, s, use, card, users, False, cards).seconds - 1:23.0%}" if speculative else ""
      out.append(f"  {users:8d} {t.reads / 1e9:11.2f} {t.seconds * 1e3:10.1f} {t.launches:10d} {t.control * 1e3:13.1f} {t.rate:17.1f} {users * t.rate:8.0f} "
                 f"{hour / (users * t.rate * 3600) * 1e6:17.3f}{pays}")
  out.append(f"  a prompt of {use['context']} tokens, prefilled a chunk at a time:")
  out.append("     chunk   seconds   tokens/s a card   $ a million in   a chunk's scores, GB")
  for chunk in (256, 1024, 2048, 4096):
    p = prompt(c, s, use, card, chunk, cards)
    out.append(f"  {chunk:8d} {p.seconds:9.2f} {p.rate:17,.0f} {hour / (p.rate * 3600) * 1e6:16.4f} {p.scores / 1e9:22.2f}")
  out.append(f"  a session back from the host: {session(c, use).bytes / card['pcie'] * 1e3:.0f} ms over PCIe at {card['pcie'] / 1e9:.0f} GB/s")
  return "\n".join(out)

if __name__ == "__main__":
  args = [a for a in sys.argv[2:] if "=" not in a]
  print(report(sys.argv[1], args[0] if args else "bf16", **dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)))
