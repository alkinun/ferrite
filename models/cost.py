# What a model costs to serve on Blackhole cards, from its config alone (llama.config): what a card
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
from ferrite.compiler import DEPTH, attention_parts, attention_store
import llama

BYTES = {"bf16": 2.0, "bfp8": 1088 / 1024, "bfp4": 576 / 1024}  # a value's, with its share of its tile's exponents
MODES = {"bf16": {}, "bfp8": {"*": "bfp8"},  # each kind of matrix's format, Tenstorrent's performance precision's, and with
         "bfp4": {"*": "bfp8", "gate_up": "bfp4"},  # every expert matrix in bfp4 (Qwen3.6 on a card). The embedding stays
         "experts": {"*": "bfp8", "gate_up": "bfp4", "down": "bfp4", "draft head": "bfp4"}}  # bf16 (a token's row is gathered), and
# so do the routers; with a draft head, the multi-token prediction layer drafts by a copy of the output layer in its format
LAUNCHES = {"attention": 21, "experts": 17, "deltanet": 5, "rest": 10,  # a decode step's: an attention block's (its norms,
            "attention verify": 21, "deltanet verify": 5, "rest verify": 17, "draft": 54}  # projections, RoPE, the cache's put,
# attention in parts and their sum, its output and gate), a mixture of experts' more, a Gated DeltaNet block's (its
# projections, the delta kernel, its output's norm and gate), the rest (the embedding's gather, the places' columns, the
# final norm, the output layer, sampling's heads), each a verify step's too, and the multi-token prediction layer's call.
# Counted on ttsim at Qwen3.6's shapes, eight slots (cmp/qwen36/step.py).
ACTIVATIONS = 12e6, 45e6  # bytes a decode step reads past its weights, experts, states and attention's rows: a block's
# (activations in 32-row tiles, each expert's outputs) and the rest's (the logits sampling reads), at Qwen3.6's shapes
CONTROL = {  # ttsim cycles a block's launches (with its experts), and the rest's, take past their reads at 265 bytes a cycle
  "attention": ((1, 42000), (8, 21800), (32, 24000)), "deltanet": ((1, 38400), (8, 13500), (32, 17900)),  # (358 GB/s at
  "rest": ((1, 180000), (8, 837000), (32, 1918000)),                                                       # 1.35 GHz), by
  "attention verify": ((2, 46900), (16, 29800)), "deltanet verify": ((2, 63600), (16, 28200)),            # the step's tokens:
  "rest verify": ((2, 449000), (16, 1702000)), "draft": ((2, 50300), (16, 32800))}                         # the RISC-V work a
# card waits on beyond DRAM (cmp/qwen36/step.py's "past reads" at Qwen3.6's shapes; between counts, a straight line)
WORKERS = 119  # a p150's 120 Tensix tiles but the dispatcher
CARD = dict(dram=len(bh.DRAM) * 4 * 2**30, reserve=2**30,  # a p150: 8 GDDR6 banks of 4 GiB, a GiB of it for kernels and activations
            bandwidth=bh.DRAM_GBPS * 1e9, reach=0.7,  # its 512 GB/s, and the share of it a decode step reaches (a guess)
            tflops=664e12, phases=2,  # bfp8 matmuls at one fidelity phase (16 cycles a step), and the phases they take
            launch=1e-6, pcie=25e9,  # a launch on the card (a guess), and PCIe 4.0 x16's bytes/s
            price=1299, host=600, years=3, watts=300, host_watts=75, kwh=0.15)  # the p150a at its launch, a share of a host,
            # three years; the card's power and its host's share, at $0.15 a kWh
USE = dict(context=32768, kv="bf16", accept=0.0)  # each session's places, its slot's, all of them filled (a step reads each slot's
# rows up to the power of 2 that holds the longest session's), and its KV cache's format; the share of drafts taken

def probed(card, path):  # the probe's numbers (python3 -m ferrite.probe 8 path) in place of the specification's
  p = json.load(open(path))
  tiles = [p[f"DRAM reads of {f} tiles in two requests, {DEPTH} in flight: all workers"] for f in ("bfp8", "bfp4")]  # (as the matmuls read them)
  return card | dict(bandwidth=min(tiles) * 1e9, reach=1.0,
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
  copy = c.mtp * c.vocab * d * fmt("draft head") if "draft head" in MODES[mode] else 0  # (the draft's own output layer)
  values = (sum((d * (2 * c.kh * c.kd + 2 * c.vh * c.vd + 2 * c.vh) + c.vh * c.vd * d) if llama.linear(c, l) else attention / fmt("attention") for l in range(c.layers))
            + len(moe) * (3 * d * (sh + (c.k or 0) * m) + d * e) + (c.layers - len(moe)) * 3 * d * f + c.vocab * d)
  return SimpleNamespace(every=every, expert=d * m * (2 * fmt("gate_up") + fmt("down")), moe=len(moe), draft=draft, embed=c.vocab * d * BYTES["bf16"],
                         head=copy or c.vocab * d * fmt("head"),  # (the output layer a draft reads)
                         bytes=every + (len(moe) + c.mtp) * e * d * m * (2 * fmt("gate_up") + fmt("down")) + draft + copy + c.vocab * d * BYTES["bf16"],
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

def attention(c, entries, queries, rows, kv):  # the bytes an attention block reads, as ferrite's kernel reads them
  # (Tensor.attend): for every eight tile rows of 32 queries of each of `entries` heads (or key heads, a token's query
  # heads side by side), the `rows` keys and values its places reach, the values once and the keys once, but twice
  # those past what a part keeps in L1 between its sweeps (attention_store), the keys shared among parts as attend shares them
  ht, kt, wt = llama.head(c) // 32, -(-rows // 32), kv if kv in ("bfp8", "bfp4") else None
  units = entries * -(-queries // 256)
  parts = attention_parts(ht, queries, wt, kt, units, WORKERS)
  return units * (2 * kt + max(0, kt - parts * attention_store(ht, queries, wt, parts)[1])) * 32 * llama.head(c) * BYTES[kv]

def past(kind, n):  # CONTROL's cycles for kind at n tokens: a straight line between its counts, flat past them
  points = CONTROL[kind]
  if n <= points[0][0] or n >= points[-1][0]: return points[0][1] if n <= points[0][0] else points[-1][1]
  (a, x), (b, y) = next((p, q) for p, q in zip(points, points[1:]) if p[0] <= n <= q[0])
  return x + (y - x) * (n - a) / (b - a)

def step(c, s, use, card, users, speculative=False, cards=1):
  """A decode step for `users` sessions at once, a token each, or a token and a draft each then the next drafts (two
  calls): the bytes it reads (and the states' it writes), its flops, launches, control and seconds: waiting on DRAM or
  on the matmuls, whichever is longer, then on its launches, then on the RISC-V work its kernels do past their reads
  (if the card's counts as ttsim's does), and tokens/s a user."""
  n, one, d = users * (1 + speculative), session(c, use, speculative), c.dim
  blocks = sum(not llama.linear(c, l) for l in range(c.layers))
  reads = s.every + s.moe * touched(c, n) * s.expert + users * one.state * (1.5 if speculative else 2)  # states read and written once (a
  reads += blocks * attention(c, c.kv * users, c.heads // c.kv * (1 + speculative), use["context"], use["kv"])  # verify's: one of two, both)
  flops = 2 * n * s.values + n * sum(4 * c.heads * llama.head(c) * use["context"] for l in range(c.layers) if not llama.linear(c, l))
  linear, v = sum(llama.linear(c, l) for l in range(c.layers)), " verify" * speculative
  reads += c.layers * ACTIVATIONS[0] + ACTIVATIONS[1]
  launches = blocks * LAUNCHES["attention" + v] + linear * LAUNCHES["deltanet" + v] + s.moe * LAUNCHES["experts"] + LAUNCHES["rest" + v]
  control = blocks * past("attention" + v, n) + linear * past("deltanet" + v, n) + past("rest" + v, n)
  if speculative:  # the drafting call: the layer over two places a user, and the output layer again
    reads += s.draft + c.mtp * touched(c, 2 * users) * s.expert + s.head + c.mtp * ACTIVATIONS[0]
    reads += c.mtp * attention(c, c.kv * users, 2 * c.heads // c.kv, use["context"], use["kv"])
    flops += 2 * 2 * users * (s.draft / BYTES["bfp8"] + c.k * 3 * d * (c.moe or 0) + c.vocab * d)
    launches += LAUNCHES["draft"]
    control += c.mtp * past("draft", n)
  control /= bh.AICLK
  seconds = max(reads / (card["bandwidth"] * card["reach"] * cards), flops / (card["tflops"] / card["phases"] * cards)) + launches * card["launch"] + control
  return SimpleNamespace(reads=reads, seconds=seconds, launches=launches, control=control, flops=flops, rate=(1 + use["accept"] if speculative else 1) / seconds)

def prompt(c, s, use, card, chunk, cards=1):
  """A prompt that fills a session (use's context), prefilled chunk tokens at a time as serve.py's engine takes it: each
  chunk reads every matrix but the experts and the embedding, the experts its tokens pick, and its session's rows up to
  its last place (`attention`: a key head's for every eight tile rows of 32 queries of each of its query heads), and
  computes its tokens' matmuls (the output layer's for its last alone) and their scores over those rows; with a
  multi-token prediction layer, that layer's over each place too. Its seconds (each chunk waiting on DRAM or on the
  matmuls, whichever is longer; launches and control left out) and tokens/s."""
  hd, d, n = llama.head(c), c.dim, use["context"]
  blocks = sum(not llama.linear(c, l) for l in range(c.layers)) + c.mtp
  seconds = 0.0
  for at in range(0, n, chunk):
    m = min(chunk, n - at)
    rows = at + m
    reads = s.every + s.moe * touched(c, m) * s.expert + blocks * attention(c, c.heads, m, rows, use["kv"])
    flops = 2 * m * (s.values - c.vocab * d) + 2 * c.vocab * d + blocks * 4 * m * c.heads * hd * rows
    if c.mtp:
      reads += s.draft + c.mtp * touched(c, m) * s.expert + s.head
      flops += 2 * m * (s.draft / BYTES["bfp8"] + c.k * 3 * d * c.moe + c.vocab * d)
    seconds += max(reads / (card["bandwidth"] * card["reach"] * cards), flops / (card["tflops"] / card["phases"] * cards))
  return SimpleNamespace(seconds=seconds, rate=n / seconds)

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
  out.append("     chunk   seconds   tokens/s a card   $ a million in")
  for chunk in (256, 1024, 2048, 4096):
    p = prompt(c, s, use, card, chunk, cards)
    out.append(f"  {chunk:8d} {p.seconds:9.2f} {p.rate:17,.0f} {hour / (p.rate * 3600) * 1e6:16.4f}")
  kept = session(c, use).bytes  # (a copy in DRAM: read and written)
  out.append(f"  a session taken back: {2 * kept / (card['bandwidth'] * card['reach']) * 1e3:.1f} ms kept on the chip, "
             f"{kept / card['pcie'] * 1e3:.0f} ms from the host over PCIe at {card['pcie'] / 1e9:.0f} GB/s")
  return "\n".join(out)

if __name__ == "__main__":
  args = [a for a in sys.argv[2:] if "=" not in a]
  print(report(sys.argv[1], args[0] if args else "bf16", **dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)))
