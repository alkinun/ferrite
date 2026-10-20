# Stage 1's benchmarks (ROADMAP: B1, B2 and B3, each measured as its Measurement section says) on ferrite or on TT-NN,
# on the same card, with the versions of everything that took part. Prints them and writes them to benchmark-<stack>.json.
#   python3 test/benchmark.py ferrite path/to/Llama-3.1-8B-Instruct path/to/Llama-3.1-8B-Instruct.refpt [--quick]
#   python3 test/benchmark.py ttnn path/to/Llama-3.1-8B-Instruct path/to/Llama-3.1-8B-Instruct.refpt [--quick]
# The folder holds Hugging Face's files for the model; the reference file is tt_transformers'
# tests/reference_outputs/<model>.refpt. B3's prompts are the start of A Tale of Two Cities, which tt_transformers keeps
# beside them (tests/tale-of-two-cities.txt.bz2; BOOK=path for another place).
# ttnn runs in tt-metal's Python environment, TT_METAL_HOME its checkout. B1 is timed as TT-NN's tutorial times a
# matmul; B2 and B3 are the numbers of Tenstorrent's own script (simple_text_demo.py, its performance runs named below,
# given our prompts and step counts), its logs kept beside them (benchmark-ttnn-<run>.log). TT_KMD=path/to/tt-kmd
# counts tt-kmd's lines of code too. On ttsim (TTSIM set for ferrite, TT_METAL_SIMULATOR for TT-NN) the times are its
# clock, not silicon's, and --quick shrinks every size and the model to two blocks: only a check of the script. TT's
# script does not run on ttsim (its prefill warm-up uses an SFPU mode ttsim lacks), so there only TT-NN's B1 does.
import bz2, glob, json, os, platform, re, statistics, subprocess, sys, tempfile, time
from types import SimpleNamespace
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ferrite import bh, pci
import gpt2, llama

ROOT = os.path.join(os.path.dirname(__file__), "..")
SIZES = {False: SimpleNamespace(n=4096, runs=(3, 5, 20), steps=200, accuracy=500, users=32, lengths=(4096, 32768), blocks=None),
         True: SimpleNamespace(n=256, runs=(1, 2, 2), steps=8, accuracy=8, users=2, lengths=(64, 128), blocks=2)}

def git(folder):  # a checkout's commit, and whether it has changes
  run = lambda *a: subprocess.run(["git", "-C", folder, *a], capture_output=True, text=True).stdout.strip()
  return run("rev-parse", "--short", "HEAD") + (" with changes" if run("status", "--porcelain", "--untracked-files=no") else "")

def lines(folder, keep):  # code lines as cloc counts them, in the files git tracks there (and in its submodules) that keep keeps
  files = subprocess.run(["git", "-C", folder, "ls-files", "--recurse-submodules"], capture_output=True, text=True).stdout.splitlines()
  with tempfile.NamedTemporaryFile("w") as f:
    f.write("\n".join(os.path.join(folder, p) for p in files if keep(p)))
    f.flush()
    try: return json.loads(subprocess.run(["cloc", "--json", "--quiet", f"--list-file={f.name}"], capture_output=True, text=True).stdout)["SUM"]["code"]
    except (OSError, ValueError, KeyError): return "not counted: cloc is not installed"

def tenstorrent(path):  # Tenstorrent's code as ROADMAP counts it: no tests, models, docs (Markdown included), CI or editor
  parts = path.split("/")  # settings (dot folders), or third-party code but the UMD tt-metal pins
  return (not {"tests", "test", "models", "docs"} & set(parts) and not any(p.startswith(".") for p in parts) and not path.endswith(".md")
          and all(parts[i + 1] == "umd" for i, p in enumerate(parts[:-1]) if p == "third_party"))

def host():
  first = lambda path, key: next((l.split(key, 1)[1].strip().strip('"') for l in open(path) if l.startswith(key)), None)
  return {"cpu": first("/proc/cpuinfo", "model name\t:"), "os": first("/etc/os-release", "PRETTY_NAME="), "kernel": platform.release(),
          "python": platform.python_version(), "numpy": np.__version__}

def card(sim):  # which card, and its PCIe link
  if sim: return "ttsim"
  bdf = pci.find_card()
  read = lambda f: open(f"/sys/bus/pci/devices/{bdf}/{f}").read().strip()
  return f"{bdf}, PCIe {read('current_link_speed')} x{read('current_link_width')}"

def timed(run, wait, clock, runs):  # seconds a run: after runs[0] warm-up runs, the median of runs[1] measurements, each
  for _ in range(runs[0]): run()     # of runs[2] runs back to back, from the first launch to the last result
  wait()
  times = []
  for _ in range(runs[1]):
    start = clock()
    for _ in range(runs[2]): run()
    wait()
    times.append((clock() - start) / runs[2])
  return statistics.median(times)

def weights(c):  # each weight as llama.upload makes it, its rows, columns and format, from the model's sizes alone
  hd, qkv = c.dim // c.heads, (c.heads + 2 * c.kv) * (c.dim // c.heads)
  shapes = {"embed": (c.vocab, c.dim), "embed.T": (c.dim, c.vocab), "norm": (1, c.dim), "cos": (c.places, qkv), "sin": (c.places, qkv), "turn": (hd, hd)}
  for l in range(c.layers):
    shapes |= {f"layers.{l}.{k}": s for k, s in (("input_layernorm", (1, c.dim)), ("post_attention_layernorm", (1, c.dim)), ("qkv_proj", (c.dim, qkv)),
                                                 ("o_proj", (c.dim, c.dim)), ("gate_up_proj", (c.dim, 2 * c.ffn)), ("down_proj", (c.ffn, c.dim)))}
  return {k: SimpleNamespace(rows=r, cols=n, dtype=llama.weight_format(k, getattr(c, "precision", None))) for k, (r, n) in shapes.items()}

def bytes_per_token(t, c, context):  # what a decode token reads (ROADMAP, Measurement): every weight at its stored size
  # but the embedding's and RoPE's tables, a row of each, and the KV cache at the decode's mean context
  size = {"bf16": 2, "bfp8": 1088 / 1024, "bfp4": 576 / 1024}
  rows = {"embed": c.dim, "cos": t["cos"].cols, "sin": t["sin"].cols}
  return (sum(size[x.dtype] * rows.get(k, x.rows * x.cols) for k, x in t.items())
          + context * c.layers * 2 * c.kv * (c.dim // c.heads) * size[llama.cache_format(c)])

def model(folder, s, places):  # the model's config at TT's performance precision, and its weights, read when wanted
  c, w = llama.load(folder, places)
  if s.blocks: c.layers = s.blocks
  c.precision = llama.precision(c, "tt")
  return c, {k: v for k, v in w.items() if not k.startswith("layers.") or int(k.split(".")[1]) < c.layers}

def ferrite(folder, prompt, after, top5, book, s):
  from ferrite.chip import Chip
  from ferrite.runtime import Runtime
  from ferrite.tensor import Tensor, jit
  chip = Chip(pci.open_device())
  rt, clock, on = Runtime(chip, chip.tensix[0], chip.tensix[-1]), chip.pci.now, lambda row: Tensor.of(rt, [row], "int32")
  firmware = chip.telemetry().get(bh.TAG_FIRMWARE, 0)
  out = {"versions": {"ferrite": git(ROOT), "firmware": ".".join(str(firmware >> b & 255) for b in (24, 16, 8)), "card": card(isinstance(chip.pci, pci.Sim)),
                      "host": host(), "lines of code": {d: lines(ROOT, lambda p: p.startswith(d + "/")) for d in ("ferrite", "adapters")}},
         "clock": "ttsim's, not silicon's: a check of the script" if isinstance(chip.pci, pci.Sim) else "the host's"}

  def decode(t, c, users):
    """A user's tokens/s, `users` decoding together greedily (end-of-sequence ignored) after the same prompt, prefilled
    into each one's rows of the cache: the steps after the warm-up (the first two: jit runs, then records) over their
    time, each from the token ids on the host to the next ones there."""
    n, size = len(prompt), -(-(len(prompt) + s.steps) // 32) * 32
    kv, starts = llama.kv_cache(c, size * users, llama.zeros(rt, llama.cache_format(c))), [size * u for u in range(users)]
    tokens = []
    for u in range(users):  # batch 1 as a prompt (prefill), more as rows of a batch, and each one's next token
      if users == 1: last = llama.forward(on(prompt), on(list(range(n))), t, c, kv)
      else: last = llama.forward(on(prompt), on(list(range(n))), t, c, kv, on([starts[u]] * n))[n - 1:n, :]
      tokens.append(int(llama.logits(last).argmax()))
    step, times = jit(lambda ids, places, starts=None: llama.forward(ids, places, t, c, kv, starts)), []
    for i in range(s.steps):
      start = clock()
      ids, places = on(tokens), on([n + i] * users)
      tokens = [int(x) for x in gpt2.values(step(ids, places) if users == 1 else step(ids, places, on(starts))).argmax(-1)]
      times.append(clock() - start)
    return (s.steps - 2) / sum(times[2:])

  def prefill(t, c, ids):
    """Seconds to the first token after ids, from them on the host, after a warm-up prefill of them (RoPE's tables
    uploaded for their places, the cache just big enough)."""
    t = t | llama.upload(rt, {k: w for k, w in llama.load(folder, len(ids))[1].items() if k in ("cos", "sin")})
    kv = llama.kv_cache(c, -(-len(ids) // 32) * 32, llama.zeros(rt, llama.cache_format(c)))
    for _ in range(2):
      start = clock()
      int(llama.logits(llama.forward(on(ids), on(list(range(len(ids)))), t, c, kv)).argmax())
    return clock() - start

  rng = np.random.default_rng(0)
  a, b = (Tensor.frombytes(rt, s.n, s.n, (rng.standard_normal((s.n, s.n), np.float32).view(np.uint32) >> 16).astype(np.uint16).tobytes()) for _ in range(2))
  for phases in (2, 4):
    seconds = timed(lambda: a.matmul(b, fidelity=phases).realize(), rt.wait, clock, s.runs)
    out[f"B1 matmul {s.n}^3 bf16, {phases} phases"] = {"ms": seconds * 1e3, "TFLOPS": 2 * s.n**3 / seconds / 1e12}
  del a, b  # their DRAM back before the model's
  c, w = model(folder, s, len(prompt) + max(s.steps, s.accuracy))
  t = llama.upload(rt, w, c.precision)
  tps = decode(t, c, 1)
  read = bytes_per_token(t, c, len(prompt) + (s.steps + 1) / 2)
  out["B2 decode, batch 1"] = {"tokens/s": tps, "GB a token": read / 1e9, "effective DRAM bandwidth": tps * read / (bh.DRAM_GBPS * 1e9)}
  tps = decode(t, c, s.users)
  out[f"B2 decode, batch {s.users}"] = {"tokens/s a user": tps, "tokens/s": tps * s.users}
  top1, five, _ = llama.accuracy(rt, t, c, prompt, after, top5, s.accuracy)
  out["B2 token accuracy"] = {"top-1": top1, "top-5": five, "steps": s.accuracy}
  ids = llama.tokenizer(folder).encode(book)
  for n in s.lengths:  # a prompt ferrite cannot run yet is published as such
    try: out[f"B3 prefill, {n} tokens"] = {"seconds to first token": prefill(t, c, ids[:n])}
    except Exception as e: out[f"B3 prefill, {n} tokens"] = {"does not run yet": f"{type(e).__name__}: {e}"}
  return out

DEMO = "models/tt_transformers/demo/simple_text_demo.py"
LOGGED = {"tokens/s a user": (r"@ ([\d.]+) tok/s/user", 1), "tokens/s": (r"\(([\d.]+) tok/s throughput\)", 1),
          "seconds to first token": (r"Time to First Token \(TTFT\): ([\d.]+)ms", 1e-3), "prompt tokens": (r"Encoded prompt lengths:\s*\[?(\d+)", 1),
          "top-1": (r"Top1 Accuracy: ([\d.]+)%", 1e-2), "top-5": (r"Top5 Accuracy: ([\d.]+)%", 1e-2)}

def ttnn(folder, prompt, after, top5, book, s):
  import torch
  from transformers import AutoTokenizer
  tt, kmd, sim = os.environ["TT_METAL_HOME"], os.environ.get("TT_KMD"), "TT_METAL_SIMULATOR" in os.environ
  read = lambda path: open(path).read().strip() if os.path.exists(path) else None
  out = {"versions": {"tt-metal": git(tt), "umd": git(f"{tt}/tt_metal/third_party/umd"), "tt-kmd": read("/sys/module/tenstorrent/version"),
                      "firmware": next(map(read, glob.glob("/sys/class/tenstorrent/*/tt_fw_bundle_ver")), None), "card": card(sim),
                      "torch": torch.__version__, "host": host(),
                      "lines of code": {"tt-metal": lines(tt, tenstorrent), "tt-kmd": lines(kmd, tenstorrent) if kmd else "not counted: TT_KMD not set"}},
         "clock": "ttsim's, not silicon's: a check of the script" if sim else "the host's"}

  def matmul(n, phases):
    """Seconds, as TT-NN's tutorial (ttnn/tutorials/ttnn_intro.ipynb) calls and times a matmul: its fidelity, sums in
    fp32 (fp32_dest_acc_en, packer_l1_acc), inputs and result bf16 in DRAM."""
    import ttnn
    device = ttnn.open_device(device_id=0)
    try:
      torch.manual_seed(0)
      a, b = (ttnn.from_torch(torch.randn((n, n), dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device) for _ in range(2))
      config = ttnn.WormholeComputeKernelConfig(math_fidelity=getattr(ttnn.MathFidelity, f"HiFi{phases}"), math_approx_mode=False,
                                                fp32_dest_acc_en=True, packer_l1_acc=True)
      return timed(lambda: ttnn.matmul(a, b, compute_kernel_config=config), lambda: ttnn.synchronize_device(device), time.monotonic, s.runs)
    finally: ttnn.close_device(device)

  def demo(label, run, want, *flags, text=None):
    """Tenstorrent's script, its run `run` at TT's performance precision, with flags over its settings (and text, a
    prompt, as its input): what it logs of want, and why it did not pass if it did not."""
    with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
      if text is not None: json.dump([{"prompt": text}], f); f.flush(); flags += ("--input_prompts", f.name, "--instruct", "0")
      if s.blocks: flags += ("--num_layers", str(s.blocks))
      p = subprocess.run([sys.executable, "-m", "pytest", DEMO, "-k", f"performance and {run}", "-s", "-p", "no:cacheprovider", *flags],
                         cwd=tt, env=os.environ | {"HF_MODEL": folder}, capture_output=True, text=True)
    log = p.stdout + p.stderr
    open(f"benchmark-ttnn-{label}.log", "w").write(log)
    got = {k: float(m[-1]) * LOGGED[k][1] for k in want + ("prompt tokens",) if (m := re.findall(LOGGED[k][0], log))}
    if p.returncode or set(want) - set(got):
      got["did not pass"] = [l for l in log.splitlines() if l.startswith(("E ", "FAILED", "SKIPPED")) or "ERROR:" in l][-3:] or log.splitlines()[-1:]
    return got

  for phases in (2, 4):
    seconds = matmul(s.n, phases)
    out[f"B1 matmul {s.n}^3 bf16, {phases} phases"] = {"ms": seconds * 1e3, "TFLOPS": 2 * s.n**3 / seconds / 1e12}
  tok = AutoTokenizer.from_pretrained(folder)
  text = tok.decode(prompt)  # as TT's token accuracy makes its prompt from the reference's tokens (prepare_ref_tokens)
  decoding = ("--stop_at_eos", "0", "--max_generated_tokens", str(s.steps))
  out["B2 decode, batch 1"] = demo("b2-batch-1", "batch-1", ("tokens/s",), *decoding, text=text)
  c, _ = model(folder, s, 1)
  if "tokens/s" in out["B2 decode, batch 1"]:
    read = bytes_per_token(weights(c), c, len(prompt) + (s.steps + 1) / 2)
    out["B2 decode, batch 1"] |= {"GB a token": read / 1e9, "effective DRAM bandwidth": out["B2 decode, batch 1"]["tokens/s"] * read / (bh.DRAM_GBPS * 1e9)}
  out[f"B2 decode, batch {s.users}"] = demo(f"b2-batch-{s.users}", "batch-32 and not log-probs", ("tokens/s a user", "tokens/s"), *decoding,
                                            *(("--batch_size", str(s.users)) if s.users != 32 else ()), text=text)
  out["B2 token accuracy"] = demo("b2-accuracy", "ci-token-matching", ("top-1", "top-5"), "--max_generated_tokens", str(s.accuracy))
  ids = tok.encode(book, add_special_tokens=False)
  for n in s.lengths:  # the script adds the beginning-of-text token to the n - 1 of the book; room for twice n
    out[f"B3 prefill, {n} tokens"] = demo(f"b3-{n}", "long-context-64k", ("seconds to first token",), "--mode", "prefill",
                                          "--max_seq_len", str(2 * n), "--max_generated_tokens", "1", text=tok.decode(ids[:n - 1]))
  return out

def main():
  stack, folder, refpt, s = sys.argv[1], sys.argv[2], sys.argv[3], SIZES["--quick" in sys.argv]
  assert stack in ("ferrite", "ttnn"), "the stack to measure: ferrite or ttnn"
  prompt, after, top5 = llama.reference(refpt)
  if s.blocks: prompt = prompt[-32:]  # (the reference's next tokens still follow its whole prompt)
  path = os.environ.get("BOOK") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(refpt))), "tale-of-two-cities.txt.bz2")
  out = {"ferrite": ferrite, "ttnn": ttnn}[stack](folder, prompt, after, top5, bz2.open(path, "rt", encoding="utf-8").read(), s)
  json.dump(out, open(f"benchmark-{stack}.json", "w"), indent=1)
  for k, v in out.items(): print(f"{k}: {v}")

if __name__ == "__main__": main()
