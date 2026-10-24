# Four benchmarks on ferrite or on TT-NN, on the same card, with the versions of everything that took part (ferrite's,
# tt-metal's and the UMD it pins, tt-kmd's, the card's firmware, the host's), and the lines of code as cloc counts them.
# Prints them and writes them to benchmark-<stack>.json.
#   B1, a matmul: (4096 x 4096) @ (4096 x 4096) in bf16, sums in fp32, at two fidelity phases and at four; the median of 5
#     measurements, each 20 runs back to back after 3 warm-up runs, from the first launch to the last result.
#   B2, decode: Llama-3.1-8B-Instruct at TT's performance precision (llama.precision's tt), for one user and for 32: after
#     the reference file's first 512 tokens, 200 greedy steps, tokens/s over the steps after each stack's warm-up (until
#     its step replays); the effective DRAM bandwidth (the bytes a token reads, bytes_per_token, times tokens/s, over the
#     card's 512 GB/s: 6.20 GB a token, so 41.3 tokens/s is half); and Tenstorrent's token accuracy (llama.accuracy: the
#     prompt, then 500 steps fed the file's next token, top-1 and top-5 against the reference model's five most likely).
#   B3, long prefill: the time to first token for B2's model with prompts of 4,096 and 32,768 tokens, after a warm-up.
#   B4, serving: B2's model behind serve.py's server, 32 agents at once over HTTP (load), each four turns that share a
#     system prompt and grow by their answers and a tool's result; tokens/s a user and for the card, the time to first
#     token, and the prompt tokens the server had cached. ferrite's alone: TT-NN serves through vLLM, not here, but load
#     takes any OpenAI-compatible server's URL (python3 test/benchmark.py load URL [agents]).
# table compares them, B1's and B2's speed against ferrite being no more than 1.5 times slower than TT-NN.
#   python3 test/benchmark.py ferrite path/to/Llama-3.1-8B-Instruct path/to/Llama-3.1-8B-Instruct.refpt [--quick]
#   python3 test/benchmark.py ttnn path/to/Llama-3.1-8B-Instruct path/to/Llama-3.1-8B-Instruct.refpt [--quick]
#   python3 test/benchmark.py table benchmark-ferrite.json benchmark-ttnn.json   # the two side by side, as markdown
#   python3 test/benchmark.py load http://localhost:8000 [agents]   # B4's load against a server already running
# The folder holds Hugging Face's files for the model; the reference file is tt_transformers'
# tests/reference_outputs/<model>.refpt. B3's prompts are the start of A Tale of Two Cities, which tt_transformers keeps
# beside them (tests/tale-of-two-cities.txt.bz2; BOOK=path for another place).
# ttnn runs in tt-metal's Python environment, TT_METAL_HOME its checkout. B1 is timed as TT-NN's tutorial times a
# matmul; B2 and B3 are the numbers of Tenstorrent's own script (simple_text_demo.py, its performance runs named below,
# given our prompts and step counts), its logs kept beside them (benchmark-ttnn-<run>.log). TT_KMD=path/to/tt-kmd
# counts tt-kmd's lines of code too. On ttsim (TTSIM set for ferrite, TT_METAL_SIMULATOR for TT-NN) the times are its
# clock, not silicon's, and --quick shrinks every size and the model to two blocks: only a check of the script. TT's
# script does not run on ttsim (its prefill warm-up uses an SFPU mode ttsim lacks), so there only TT-NN's B1 does.
import bz2, glob, json, os, platform, re, shutil, statistics, subprocess, sys, tempfile, threading, time, urllib.request
from types import SimpleNamespace
import numpy as np
sys.path[:0] = [os.path.join(os.path.dirname(__file__), ".."), os.path.join(os.path.dirname(__file__), "..", "models")]
from ferrite import bh, pci
import cost, gpt2, llama

ROOT = os.path.join(os.path.dirname(__file__), "..")
SIZES = {False: SimpleNamespace(n=4096, runs=(3, 5, 20), steps=200, accuracy=500, users=32, lengths=(4096, 32768), blocks=None,
                               turns=SimpleNamespace(system=1500, task=400, tool=200, out=256, n=4)),  # (B4's: words, an answer's tokens, turns)
         True: SimpleNamespace(n=256, runs=(1, 2, 2), steps=8, accuracy=8, users=2, lengths=(64, 128), blocks=2,
                               turns=SimpleNamespace(system=24, task=6, tool=6, out=8, n=2))}

def git(folder):  # a checkout's commit, and whether it has changes
  run = lambda *a: subprocess.run(["git", "-C", folder, *a], capture_output=True, text=True).stdout.strip()
  return run("rev-parse", "--short", "HEAD") + (" with changes" if run("status", "--porcelain", "--untracked-files=no") else "")

def lines(folder, keep):  # code lines as cloc counts them, in the files git tracks there (and in its submodules) that keep keeps
  if not shutil.which("cloc"): return "not counted: cloc is not installed"
  files = subprocess.run(["git", "-C", folder, "ls-files", "--recurse-submodules"], capture_output=True, text=True).stdout.splitlines()
  with tempfile.NamedTemporaryFile("w") as f:
    f.write("\n".join(os.path.join(folder, p) for p in files if keep(p)))
    f.flush()
    out = subprocess.run(["cloc", "--json", "--quiet", f"--list-file={f.name}"], capture_output=True, text=True).stdout
  try: return json.loads(out)["SUM"]["code"]
  except (ValueError, KeyError): return f"not counted: cloc found no code in {folder}"

def tenstorrent(path):  # Tenstorrent's code as we count it: no tests, models, docs (Markdown included), CI or editor
  parts = path.split("/")  # settings (dot folders), or third-party code but the UMD tt-metal pins
  return (not {"tests", "test", "models", "docs"} & set(parts) and not any(p.startswith(".") for p in parts) and not path.endswith(".md")
          and all(parts[i + 1] == "umd" for i, p in enumerate(parts[:-1]) if p == "third_party"))

def host():
  first = lambda path, key: next((l.split(key, 1)[1].strip().strip('"') for l in open(path) if l.startswith(key)), None)
  return {"cpu": first("/proc/cpuinfo", "model name\t:"), "os": first("/etc/os-release", "PRETTY_NAME="), "kernel": platform.release(),
          "python": platform.python_version(), "numpy": np.__version__}

def card(sim):  # which card, and its PCIe link
  if sim: return "ttsim"
  bdf = pci.find_cards()[0]
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

def load(url, agents, turns, text):
  """B4: agents at once against an OpenAI-compatible server at url (/v1/completions, streamed, greedy, no stop), each
  turns.n turns of turns.out tokens: a system prompt they all share (text's first turns.system words), then its own task
  (turns.task words, from its own part of text), each next turn its prompt so far, its answer and a tool's result
  (turns.tool more words). Tokens/s a user (each turn's tokens after its first over their time, the median), tokens/s
  for all of them (every token over the wall time), the time to first token (the median and the 90th percentile), and
  the share of prompt tokens the server had cached, from its usage."""
  words, done, failed = text.split(), [], []
  piece = lambda at, n: " " + " ".join(words[at:at + n])
  def agent(a):
    at = turns.system + a * (turns.task + turns.n * turns.tool)  # (its part of the text)
    prompt = piece(0, turns.system) + piece(at, turns.task)
    try:
      for k in range(turns.n):
        body = json.dumps({"prompt": prompt, "max_tokens": turns.out, "temperature": 0, "stream": True}).encode()
        start, stamps, answer, usage = time.perf_counter(), [], "", None
        with urllib.request.urlopen(urllib.request.Request(url + "/v1/completions", body, {"Content-Type": "application/json"}), timeout=3600) as r:
          for line in r:
            if not line.startswith(b"data: {"): continue
            chunk = json.loads(line[6:])
            usage = chunk.get("usage") or usage
            if chunk["choices"] and chunk["choices"][0]["text"]: stamps.append(time.perf_counter()); answer += chunk["choices"][0]["text"]
        done.append((start, stamps, usage))
        prompt += answer + piece(at + turns.task + k * turns.tool, turns.tool)
    except Exception as e: failed.append(f"agent {a}: {type(e).__name__}: {e}")
  threads = [threading.Thread(target=agent, args=(a,)) for a in range(agents)]
  for t in threads: t.start()
  for t in threads: t.join()
  assert not failed, failed
  first = sorted(stamps[0] - start for start, stamps, _ in done)
  each = statistics.median((u["completion_tokens"] - 1) / (s[-1] - s[0]) for _, s, u in done if s[-1] > s[0])
  made, span = sum(u["completion_tokens"] for *_, u in done), max(s[-1] for _, s, _ in done) - min(start for start, *_ in done)
  cached = sum(u["prompt_tokens_details"]["cached_tokens"] for *_, u in done) / sum(u["prompt_tokens"] for *_, u in done)
  return {"agents": agents, "turns": len(done), "tokens/s a user": each, "tokens/s": made / span, "seconds to first token": statistics.median(first),
          "seconds to first token, 90th percentile": first[int(0.9 * (len(first) - 1))], "prompt tokens cached": cached}

def weights(c):  # each weight as llama.upload makes it, its rows, columns and format, from the model's sizes alone
  hd = llama.head(c)
  qkv = (c.heads + 2 * c.kv) * hd
  shapes = {"embed": (c.vocab, c.dim), "embed.T": (c.dim, c.vocab), "norm": (1, c.dim), "cos": (c.places, qkv), "sin": (c.places, qkv),
            "turn": (hd, hd)}
  block = {"input_layernorm": (1, c.dim), "post_attention_layernorm": (1, c.dim), "qkv_proj": (c.dim, qkv), "o_proj": (c.heads * hd, c.dim),
           "gate_up_proj": (c.dim, 2 * c.ffn), "down_proj": (c.ffn, c.dim)}
  for l in range(c.layers): shapes |= {f"layers.{l}.{k}": s for k, s in block.items()}
  return {k: SimpleNamespace(rows=r, cols=n, dtype=llama.weight_format(k, c.precision)) for k, (r, n) in shapes.items()}

def bytes_per_token(t, c, context):
  """What a decode token reads (B2): every weight at its stored size but the embedding's and RoPE's tables, a row of
  each, and the KV cache at the decode's mean context."""
  rows = {"embed": c.dim, "cos": t["cos"].cols, "sin": t["sin"].cols}
  return (sum(cost.BYTES[x.dtype] * rows.get(k, x.rows * x.cols) for k, x in t.items())
          + context * c.layers * 2 * c.kv * llama.head(c) * cost.BYTES[llama.cache_format(c)])

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
  rt, clock, on, sim = Runtime(chip), chip.pci.now, lambda row: Tensor.of(rt, [row], "int32"), isinstance(chip.pci, pci.Sim)
  tables = lambda n: llama.upload(rt, {k: w for k, w in llama.load(folder, n)[1].items() if k in ("cos", "sin")})  # RoPE's, for n places
  firmware = chip.telemetry().get(bh.TAG_FIRMWARE, 0)
  code = {d: lines(ROOT, lambda p: p.startswith(d + "/")) for d in ("ferrite", "adapters")}
  versions = {"ferrite": git(ROOT), "firmware": ".".join(str(firmware >> b & 255) for b in (24, 16, 8)), "card": card(sim),
              "AI clock MHz": chip.aiclk, "host": host(), "lines of code": code}
  out = {"versions": versions, "clock": "ttsim's, not silicon's: a check of the script" if sim else "the host's"}

  def decode(t, c, users):
    """A user's tokens/s, `users` decoding together greedily (end-of-sequence ignored) after the same prompt, prefilled
    into each one's rows of the cache: the steps after the warm-up (the first two: jit runs, then records) over their
    time, each from the token ids on the host to the next ones there."""
    n, size = len(prompt), -(-(len(prompt) + s.steps) // 32) * 32
    kv, tokens = llama.kv_cache(c, size * users, llama.zeros(rt, llama.cache_format(c)), users), []
    for u in range(users):  # each one's prompt into its part of the cache (prefill), and its next token
      tokens.append(int(llama.logits(llama.forward(on(prompt), on(list(range(n))), t, c, llama.slot(c, kv, u, users))).argmax()))
    step, times = jit(lambda ids, places: llama.forward(ids, places, t, c, kv, users if users > 1 else None)), []
    for i in range(s.steps):
      start = clock()
      ids, places = on(tokens), on([n + i] * users)
      tokens = [int(x) for x in gpt2.values(step(ids, places)).argmax(-1)]
      times.append(clock() - start)
    return (s.steps - 2) / sum(times[2:])

  def prefill(t, c, ids):
    """Seconds to the first token after ids, from them on the host, after a warm-up prefill of them (RoPE's tables
    uploaded for their places, the cache just big enough), and the chunk the prompt went in (llama.prefill): as many
    tokens as keep attention's scores within 2 GiB, all of them if they fit."""
    t, size = t | tables(len(ids)), -(-len(ids) // 32) * 32
    kv, chunk = llama.kv_cache(c, size, llama.zeros(rt, llama.cache_format(c))), min(len(ids), max(32, (1 << 31) // (2 * c.heads * size) // 32 * 32))
    for _ in range(2):
      start = clock()
      int(llama.logits(llama.prefill(ids, t, c, kv, chunk, on)).argmax())
    return clock() - start, chunk

  def matmul(phases):  # B1 (its tensors' DRAM back before the model's)
    rng = np.random.default_rng(0)
    a, b = (Tensor.frombytes(rt, s.n, s.n, (rng.standard_normal((s.n, s.n), np.float32).view(np.uint32) >> 16).astype(np.uint16).tobytes())
            for _ in range(2))
    seconds = timed(lambda: a.matmul(b, fidelity=phases).realize(), rt.wait, clock, s.runs)
    return {"ms": seconds * 1e3, "TFLOPS": 2 * s.n**3 / seconds / 1e12}
  for phases in (2, 4): out[f"B1 matmul {s.n}^3 bf16, {phases} phases"] = matmul(phases)
  c, w = model(folder, s, len(prompt) + max(s.steps, s.accuracy))
  t = llama.upload(rt, w, c.precision)
  tps = decode(t, c, 1)
  read = bytes_per_token(t, c, len(prompt) + (s.steps + 1) / 2)
  out["B2 decode, batch 1"] = {"tokens/s": tps, "GB a token": read / 1e9, "effective DRAM bandwidth": tps * read / (bh.DRAM_GBPS * 1e9)}
  tps = decode(t, c, s.users)
  out[f"B2 decode, batch {s.users}"] = {"tokens/s a user": tps, "tokens/s": tps * s.users}
  top1, five, _ = llama.accuracy(rt, t, c, prompt, after, top5, s.accuracy)
  out["B2 token accuracy"] = {"top-1": top1, "top-5": five, "steps": s.accuracy}
  ids = llama.Tokenizer(folder).encode(book)
  for n in s.lengths:  # a prompt ferrite cannot run is published as such
    try: out[f"B3 prefill, {n} tokens"] = dict(zip(("seconds to first token", "chunk"), prefill(t, c, ids[:n])))
    except Exception as e: out[f"B3 prefill, {n} tokens"] = {"does not run": f"{type(e).__name__}: {e}"}
  import serve  # B4: a slot for each agent, as many places as two tokens a word and every answer take
  places = -(-(2 * (s.turns.system + s.turns.task + s.turns.n * s.turns.tool) + s.turns.n * s.turns.out) // 32) * 32
  engine = serve.Engine(rt, t | tables(places), c, llama.Tokenizer(folder), s.users, places)
  threading.Thread(target=engine.run, daemon=True).start()
  httpd = serve.server(engine, os.path.basename(folder.rstrip("/")), port=0)
  threading.Thread(target=httpd.serve_forever, daemon=True).start()
  out[f"B4 serving, {s.users} agents"] = load(f"http://localhost:{httpd.server_address[1]}", s.users, s.turns, book)
  httpd.shutdown()
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
                      "lines of code": {"tt-metal": lines(tt, tenstorrent),
                                        "tt-kmd": lines(kmd, tenstorrent) if kmd else "not counted: TT_KMD not set"}},
         "clock": "ttsim's, not silicon's: a check of the script" if sim else "the host's"}

  def matmul(n, phases):
    """Seconds, as TT-NN's tutorial (ttnn/tutorials/ttnn_intro.ipynb) calls and times a matmul: its fidelity, sums in
    fp32 (fp32_dest_acc_en, packer_l1_acc), inputs and result bf16 in DRAM."""
    import ttnn
    device = ttnn.open_device(device_id=0)
    try:
      torch.manual_seed(0)
      upload = lambda x: ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
      a, b = (upload(torch.randn((n, n), dtype=torch.bfloat16)) for _ in range(2))
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
    b2 = out["B2 decode, batch 1"]
    b2 |= {"GB a token": read / 1e9, "effective DRAM bandwidth": b2["tokens/s"] * read / (bh.DRAM_GBPS * 1e9)}
  out[f"B2 decode, batch {s.users}"] = demo(f"b2-batch-{s.users}", "batch-32 and not log-probs", ("tokens/s a user", "tokens/s"), *decoding,
                                            *(("--batch_size", str(s.users)) if s.users != 32 else ()), text=text)
  out["B2 token accuracy"] = demo("b2-accuracy", "ci-token-matching", ("top-1", "top-5"), "--max_generated_tokens", str(s.accuracy))
  ids = tok.encode(book, add_special_tokens=False)
  for n in s.lengths:  # the script adds the beginning-of-text token to the n - 1 of the book; room for twice n
    out[f"B3 prefill, {n} tokens"] = demo(f"b3-{n}", "long-context-64k", ("seconds to first token",), "--mode", "prefill",
                                          "--max_seq_len", str(2 * n), "--max_generated_tokens", "1", text=tok.decode(ids[:n - 1]))
  return out

def table(ferrite, ttnn):
  """The two stacks' results side by side, as markdown: each benchmark's numbers, and for B1 and B2 the gate's ratio
  (ferrite no more than 1.5 times slower than TT-NN: a time at most 1.5 times, a speed at least two thirds)."""
  show = lambda v: ", ".join(f"{k} {x:.4g}" if isinstance(x, float) else f"{k} {x}" for k, x in v.items()) if isinstance(v, dict) else str(v)
  rows = ["| | ferrite | TT-NN | TT-NN's speed over ferrite's |", "|---|---|---|---|"]
  for key in [k for k in {**ferrite, **ttnn} if k.startswith("B")]:
    f, t = ferrite.get(key, {}), ttnn.get(key, {})
    speed = ("ms", -1) if key.startswith("B1") else ("tokens/s", 1) if key == "B2 decode, batch 1" else None  # (a time's ratio inverted)
    ratio = (t[speed[0]] / f[speed[0]]) ** speed[1] if speed and speed[0] in f and speed[0] in t else None
    gate = "" if ratio is None else f"{ratio:.2f} ({'within' if ratio <= 1.5 else 'past'} 1.5)"
    rows.append(f"| {key} | {show(f)} | {show(t)} | {gate} |")
  for key in ("lines of code", "clock"): rows.append(f"| {key} | {show(ferrite['versions'].get(key) if key != 'clock' else ferrite.get(key))} | "
                                                    f"{show(ttnn['versions'].get(key) if key != 'clock' else ttnn.get(key))} | |")
  return "\n".join(rows)

def main():
  if sys.argv[1] == "table": return print(table(*(json.load(open(p)) for p in sys.argv[2:4])))
  if sys.argv[1] == "load":
    path = os.environ.get("BOOK") or "tale-of-two-cities.txt.bz2"
    book = bz2.open(path, "rt", encoding="utf-8").read()
    return print(load(sys.argv[2].rstrip("/"), int(sys.argv[3]) if len(sys.argv) > 3 else 32, SIZES[False].turns, book))
  stack, folder, refpt, s = sys.argv[1], sys.argv[2], sys.argv[3], SIZES["--quick" in sys.argv]
  assert stack in ("ferrite", "ttnn"), "the stack to measure: ferrite or ttnn"
  prompt, after, top5 = llama.reference(refpt)
  if s.blocks: prompt = prompt[-32:]  # (the reference's next tokens still follow its whole prompt)
  path = os.environ.get("BOOK") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(refpt))), "tale-of-two-cities.txt.bz2")
  out = {"ferrite": ferrite, "ttnn": ttnn}[stack](folder, prompt, after, top5, bz2.open(path, "rt", encoding="utf-8").read(), s)
  json.dump(out, open(f"benchmark-{stack}.json", "w"), indent=1)
  for k, v in out.items(): print(f"{k}: {v}")

if __name__ == "__main__": main()
