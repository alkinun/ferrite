# Stage 1's benchmarks on ferrite (ROADMAP: B1, B2 and B3, each measured as its Measurement section says), with the
# versions of everything that took part. Prints them and writes them to benchmark-ferrite.json.
#   python3 test/benchmark.py path/to/Llama-3.1-8B-Instruct path/to/Llama-3.1-8B-Instruct.refpt [--quick]
# The folder holds Hugging Face's config.json, tokenizer.json and safetensors; the reference file is tt_transformers'
# tests/reference_outputs/<model>.refpt. B3's prompts are the start of A Tale of Two Cities, which tt_transformers keeps
# beside them (tests/tale-of-two-cities.txt.bz2; BOOK=path for another place). On ttsim (TTSIM set) the times are its
# clock, not silicon's, and --quick shrinks every size and the model to two blocks: only a check of the script.
import bz2, json, os, platform, statistics, subprocess, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ferrite import bh, pci
from ferrite.chip import Chip
from ferrite.runtime import Runtime
from ferrite.tensor import Tensor, jit
import gpt2, llama

ROOT = os.path.join(os.path.dirname(__file__), "..")

def versions(chip):
  git = lambda *a: subprocess.run(["git", "-C", ROOT, *a], capture_output=True, text=True).stdout.strip()
  cloc = lambda d: json.loads(subprocess.run(["cloc", "--json", "--quiet", d], cwd=ROOT, capture_output=True, text=True).stdout)["SUM"]["code"]
  firmware = chip.telemetry().get(bh.TAG_FIRMWARE, 0)
  first = lambda path, key: next((l.split(key, 1)[1].strip().strip('"') for l in open(path) if l.startswith(key)), None)
  v = {"ferrite": git("rev-parse", "--short", "HEAD") + (" with changes" if git("status", "--porcelain", "--untracked-files=no") else ""),
       "firmware": ".".join(str(firmware >> s & 255) for s in (24, 16, 8)),
       "host": {"cpu": first("/proc/cpuinfo", "model name\t:"), "os": first("/etc/os-release", "PRETTY_NAME="),
                "kernel": platform.release(), "python": platform.python_version(), "numpy": np.__version__}}
  if isinstance(chip.pci, pci.Card):
    link = lambda f: open(f"{chip.pci.sys}/{f}").read().strip()
    v["card"] = f"{chip.pci.sys}, PCIe {link('current_link_speed')} x{link('current_link_width')}"
  else: v["card"] = "ttsim"
  try: v["lines of code"] = {d: cloc(d) for d in ("ferrite", "adapters")}  # code lines as cloc counts them
  except (OSError, ValueError, KeyError): v["lines of code"] = "not counted: cloc is not installed"
  return v

def bf16(rng, shape): return (rng.standard_normal(shape, np.float32).view(np.uint32) >> 16).astype(np.uint16).tobytes()

def matmul(rt, n, phases, clock, runs=(3, 5, 20)):
  """Seconds for (n x n) @ (n x n) in bf16 at `phases` fidelity phases: after runs[0] warm-up runs, the median of
  runs[1] measurements, each of runs[2] back to back from the first launch to the last result."""
  rng = np.random.default_rng(0)
  a, b = (Tensor.frombytes(rt, n, n, bf16(rng, (n, n))) for _ in range(2))
  for _ in range(runs[0]): a.matmul(b, fidelity=phases).realize()
  rt.wait()
  times = []
  for _ in range(runs[1]):
    start = clock()
    for _ in range(runs[2]): a.matmul(b, fidelity=phases).realize()
    rt.wait()
    times.append((clock() - start) / runs[2])
  return statistics.median(times)

def decode(rt, t, c, prompt, steps, users, clock):
  """A user's tokens/s, `users` decoding together greedily (end-of-sequence ignored) after the same prompt, prefilled
  into each one's rows of the cache: the steps after the warm-up (the first two: jit runs, then records) over their
  time, each from the token ids on the host to the next ones there."""
  n, on = len(prompt), lambda row: Tensor.of(rt, [row], "int32")
  size = -(-(n + steps) // 32) * 32
  kv, starts = llama.kv_cache(c, size * users, llama.zeros(rt, llama.cache_format(c))), [size * u for u in range(users)]
  tokens = []
  for u in range(users):  # the prompt into each user's rows (batch 1: as a prompt, prefill), and its next token
    if users == 1: out = llama.forward(on(prompt), on(list(range(n))), t, c, kv)
    else: out = llama.forward(on(prompt), on(list(range(n))), t, c, kv, on([starts[u]] * n))[n - 1:n, :]
    tokens.append(int(llama.logits(out).argmax()))
  step, times = jit(lambda ids, places, starts=None: llama.forward(ids, places, t, c, kv, starts)), []
  for i in range(steps):
    start = clock()
    ids, places = on(tokens), on([n + i] * users)
    tokens = [int(x) for x in gpt2.values(step(ids, places) if users == 1 else step(ids, places, on(starts))).argmax(-1)]
    times.append(clock() - start)
  return (steps - 2) / sum(times[2:])

def prefill(rt, t, c, folder, prompt, clock):
  """Seconds to the first token after prompt, from its token ids on the host, after a warm-up prefill of it (which
  compiles, as Tenstorrent's script's does): RoPE's tables for its places, the cache just big enough."""
  n, on = len(prompt), lambda row: Tensor.of(rt, [row], "int32")
  t = t | llama.upload(rt, {k: w for k, w in llama.load(folder, n)[1].items() if k in ("cos", "sin")})
  kv = llama.kv_cache(c, -(-n // 32) * 32, llama.zeros(rt, llama.cache_format(c)))
  for _ in range(2):
    start = clock()
    int(llama.logits(llama.forward(on(prompt), on(list(range(n))), t, c, kv)).argmax())
  return clock() - start

def bytes_per_token(t, c, context):  # what a decode token reads (ROADMAP, Measurement): every weight at its stored size
  # but the embedding's and RoPE's tables, a row of each, and the KV cache at the decode's mean context
  size = {"bf16": 2, "bfp8": 1088 / 1024, "bfp4": 576 / 1024}
  rows = {"embed": c.dim, "cos": t["cos"].cols, "sin": t["sin"].cols}
  weights = sum(size[x.dtype] * rows.get(k, x.rows * x.cols) for k, x in t.items())
  return weights + context * c.layers * 2 * c.kv * (c.dim // c.heads) * size[llama.cache_format(c)]

def main():
  folder, refpt, quick = sys.argv[1], sys.argv[2], "--quick" in sys.argv
  chip = Chip(pci.open_device())
  rt, clock, sim = Runtime(chip, chip.tensix[0], chip.tensix[-1]), chip.pci.now, isinstance(chip.pci, pci.Sim)
  out = {"versions": versions(chip), "clock": "ttsim's, not silicon's: a check of the script" if sim else "the host's"}
  hz = bh.AICLK if sim else chip.telemetry()[bh.TAG_AICLK] * 1e6  # ttsim's clock counts at bh.AICLK, whatever it says
  # B1: matmul, (4096 x 4096) @ (4096 x 4096), at two fidelity phases (gated) and four
  n = 256 if quick else 4096
  for phases in (2, 4):
    s = matmul(rt, n, phases, clock, (1, 2, 2) if quick else (3, 5, 20))
    peak = len(chip.tensix) * bh.MVMUL_FLOPS * hz / phases
    out[f"B1 matmul {n}^3 bf16, {phases} phases"] = {"ms": s * 1e3, "TFLOPS": 2 * n**3 / s / 1e12, "of the peak": 2 * n**3 / s / peak}
  # B2: decode after the reference file's first 512 tokens at TT's performance precision; batch 1 gated, 32 reported
  first, after, top5 = llama.reference(refpt)
  steps, accuracy_steps, lengths = (8, 8, (64, 128)) if quick else (200, 500, (4096, 32768))
  c, weights = llama.load(folder, len(first) + max(steps, accuracy_steps))
  if quick: c.layers, first = 2, first[-32:]  # (the reference's next tokens still follow its whole prompt)
  weights = {k: w for k, w in weights.items() if not k.startswith("layers.") or int(k.split(".")[1]) < c.layers}
  c.precision = llama.precision(c, "tt")
  t = llama.upload(rt, weights, c.precision)
  out["B2 model"] = {"folder": os.path.basename(os.path.normpath(folder)), "blocks": c.layers, "precision": "tt", "prompt": len(first), "steps": steps}
  tps = decode(rt, t, c, first, steps, 1, clock)
  read = bytes_per_token(t, c, len(first) + (steps + 1) / 2)
  out["B2 decode, batch 1"] = {"tokens/s": tps, "GB a token": read / 1e9, "effective DRAM bandwidth": tps * read / 512e9}
  users = 2 if quick else 32
  tps = decode(rt, t, c, first, steps, users, clock)
  out[f"B2 decode, batch {users}"] = {"tokens/s a user": tps, "tokens/s": tps * users}
  top1, top5_share, _ = llama.accuracy(rt, t, c, first, after, top5, accuracy_steps)
  out["B2 token accuracy"] = {"top-1": top1, "top-5": top5_share, "steps": accuracy_steps}
  # B3: time to first token, prompts of 4,096 and 32,768 tokens; a prompt ferrite cannot run yet is reported as such
  book = bz2.open(os.environ.get("BOOK") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(refpt))), "tale-of-two-cities.txt.bz2"), "rt", encoding="utf-8").read()
  ids = llama.tokenizer(folder).encode(book)
  for n in lengths:
    try: out[f"B3 prefill, {n} tokens"] = {"seconds to first token": prefill(rt, t, c, folder, ids[:n], clock)}
    except Exception as e: out[f"B3 prefill, {n} tokens"] = {"does not run yet": f"{type(e).__name__}: {e}"}
  json.dump(out, open("benchmark-ferrite.json", "w"), indent=1)
  for k, v in out.items(): print(f"{k}: {v}")

if __name__ == "__main__": main()
