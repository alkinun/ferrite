# An OpenAI-compatible server for a Llama-family model on ferrite (ROADMAP, Stage 3). Requests join and leave a batch
# that decodes together: each sequence has a slot of one KV cache, and every step is one replayed call for all the
# slots. A prompt goes to the free slot whose last sequence began the same way, so what they share (a system prompt, a
# conversation so far) is not prefilled again: a prefix cache with nothing to copy.
#   TTSIM=/path/to/libttsim.so python3 test/serve.py model [port] [slots] [places] [bf16|bfp8|bfp4|tt]   # a folder or a model id
#   curl localhost:8000/v1/completions -d '{"prompt": "The meaning of life is", "max_tokens": 8, "temperature": 0}'
import json, os, queue, shutil, sys, threading, time, urllib.error, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ferrite.tensor import Tensor, jit
import gpt2, llama

class Request:
  """A prompt's tokens and how to go on from them: greedy at temperature 0, else sampled from the softmax at that
  temperature with its least likely tokens past top_p left out, until a token in eos, a stop string or max_tokens.
  Each piece of new text goes into out as soon as no stop string could still begin in it, then None; finish says why.
  One whose client has gone (gone) ends at its next token, or takes no slot."""
  def __init__(self, tokenizer, ids, max_tokens=16, temperature=1.0, top_p=1.0, stop=(), eos=(), seed=None):
    self.tok, self.ids, self.max_tokens, self.temperature, self.top_p = tokenizer, list(ids), max_tokens, temperature, top_p
    self.stop, self.eos, self.rng = [stop] if isinstance(stop, str) else list(stop or ()), set(eos), np.random.default_rng(seed)
    self.made, self.said, self.out, self.finish, self.gone = [], "", queue.Queue(), None, False

  def pick(self, logits):
    if self.temperature == 0: return int(logits.argmax())
    p = np.exp((logits - logits.max()) / self.temperature)
    order = np.argsort(-p)
    keep = order[:np.searchsorted(np.cumsum(p[order]) / p.sum(), self.top_p) + 1]
    return int(self.rng.choice(keep, p=p[keep] / p[keep].sum()))

  def take(self, logits, room):  # the next token from its logits (room: whether its slot has a place for one more): done?
    if self.gone: self.finish = "gone"; self.out.put(None); return True
    token = self.pick(logits)
    if token not in self.eos: self.made.append(token)
    text = self.tok.decode(self.made)
    cut = min((i for i in (text.find(s) for s in self.stop) if i >= 0), default=-1)
    if token in self.eos or cut >= 0 or len(self.made) == self.max_tokens or not room:
      self.say(text if cut < 0 else text[:cut])
      self.finish = "stop" if token in self.eos or cut >= 0 else "length"
      self.out.put(None)
      return True
    hold = max((k for s in self.stop for k in range(1, len(s)) if text.endswith(s[:k])), default=0)  # a stop string's start
    self.say(text[:len(text) - hold].rstrip("\ufffd"))  # (or a character whose bytes are still coming)
    return False

  def say(self, text):
    if len(text) > len(self.said): self.out.put(text[len(self.said):]); self.said = text

class Engine:
  """Sequences decoding together on the chip, a slot of `places` rows of one KV cache each. Every step runs every slot,
  one that is free or still prefilling on its last row, which no sequence reaches, so the step's shapes never change
  and it is replayed (jit). A prompt is prefilled `chunk` tokens at a time, from what its slot holds of it already, a
  chunk between steps, so a long one slows the others' tokens rather than stopping them."""
  def __init__(self, rt, t, c, tokenizer, slots=4, places=512, chunk=256):
    self.rt, self.t, self.c, self.tok, self.places, self.chunk = rt, t, c, tokenizer, places, chunk
    self.kv = llama.kv_cache(c, slots * places, llama.zeros(rt, llama.cache_format(c)))
    self.starts, self.held, self.busy = [places * u for u in range(slots)], [[] for _ in range(slots)], [None] * slots
    self.used = [0] * slots  # when each slot last took a request: of those sharing as much, the one used longest ago goes
    self.step = jit(lambda ids, places, starts: llama.forward(ids, places, t, c, self.kv, starts))
    self.waiting, self.prefilled = queue.Queue(), 0  # (tokens prefilled in all: what the prefix cache saves shows here)

  def on(self, row): return Tensor.of(self.rt, [row], "int32")

  def admit(self, r):  # into the free slot holding the longest start of its prompt (its last token runs again for its logits)
    if r.gone: return
    shared = lambda u: next((i for i, (a, b) in enumerate(zip(self.held[u], r.ids)) if a != b), min(len(self.held[u]), len(r.ids)))
    u = max((u for u, b in enumerate(self.busy) if b is None), key=lambda u: (shared(u), -self.used[u]))
    self.used[u], self.busy[u] = max(self.used) + 1, r
    self.held[u] = r.ids[:min(shared(u), len(r.ids) - 1)]

  def fill(self, u):  # the next chunk of slot u's prompt; the last one's last logits give the first new token
    r, at = self.busy[u], len(self.held[u])
    if r.gone: return self.take(u, None)
    ids = r.ids[at:at + self.chunk]
    out = llama.forward(self.on(ids), self.on(list(range(at, at + len(ids)))), self.t, self.c, self.kv, self.on([self.starts[u]] * len(ids)))
    self.held[u] += ids
    self.prefilled += len(ids)
    if len(self.held[u]) == len(r.ids): self.take(u, gpt2.values(out[len(ids) - 1:len(ids), :])[0])

  def advance(self):  # one step for every slot: each decoding one's last token in, its next one out
    live = [r if r and r.made else None for r in self.busy]
    if not any(live): return
    ids = [r.made[-1] if r else 0 for r in live]
    places = [len(h) if r else self.places - 1 for h, r in zip(self.held, live)]
    logits = gpt2.values(self.step(self.on(ids), self.on(places), self.on(self.starts)))
    for u, r in enumerate(live):
      if r: self.held[u].append(r.made[-1]); self.take(u, logits[u])

  def take(self, u, logits):  # a slot's request takes its next token; a finished one leaves its slot (and its rows' keys)
    if self.busy[u].take(logits, len(self.held[u]) < self.places - 1): self.busy[u] = None

  def run(self):  # the engine's thread: admit while a slot is free, a chunk of a prompt, a step; on an error, every request ends
    try:
      while True:
        if not any(self.busy): self.admit(self.waiting.get())
        while None in self.busy and not self.waiting.empty(): self.admit(self.waiting.get())
        if filling := [u for u, r in enumerate(self.busy) if r and not r.made]: self.fill(filling[0])
        self.advance()
    except BaseException:
      for r in self.busy + list(self.waiting.queue):
        if r: r.finish = "error"; r.out.put(None)
      raise

def chat(template, messages):  # messages as the prompt the model was trained on, for the two kinds of template it can be
  if "<|start_header_id|>" in template:  # Llama 3's
    turn = lambda m: f"<|start_header_id|>{m['role']}<|end_header_id|>\n\n{m['content']}<|eot_id|>"
    return "<|begin_of_text|>" + "".join(map(turn, messages)) + "<|start_header_id|>assistant<|end_header_id|>\n\n"
  if "<|im_start|>" in template:  # ChatML: SmolLM's instruct models', Qwen's
    return "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages) + "<|im_start|>assistant\n"
  raise ValueError("a chat template other than Llama 3's or ChatML")

def server(engine, name, template="", eos=(), port=8000, wait=600):
  """An HTTP server for the engine (its thread must be running): /v1/models, and /v1/completions and
  /v1/chat/completions, whole or streamed (server-sent events). A piece of text not ready in `wait` seconds ends the
  request: a hung chip is reported, not waited on."""
  class Handler(BaseHTTPRequestHandler):
    def reply(self, code, body):
      data = json.dumps(body).encode()
      self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(data))); self.end_headers()
      self.wfile.write(data)
    def do_GET(self):
      if self.path != "/v1/models": return self.reply(404, {"error": {"message": f"no {self.path}"}})
      self.reply(200, {"object": "list", "data": [{"id": name, "object": "model", "owned_by": "ferrite"}]})
    def do_POST(self):
      is_chat = self.path == "/v1/chat/completions"
      if not is_chat and self.path != "/v1/completions": return self.reply(404, {"error": {"message": f"no {self.path}"}})
      try:
        q = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        ids = engine.tok.encode(chat(template, q["messages"]), special=True) if is_chat else engine.tok.encode(q["prompt"])
      except (ValueError, KeyError, TypeError) as e: return self.reply(400, {"error": {"message": f"{type(e).__name__}: {e}"}})
      if len(ids) >= engine.places - 1: return self.reply(400, {"error": {"message": f"{len(ids)} tokens: a slot holds {engine.places - 2}"}})
      r = Request(engine.tok, ids, q.get("max_tokens") or (engine.places if is_chat else 16), q.get("temperature", 1.0), q.get("top_p", 1.0),
                  q.get("stop"), eos, q.get("seed"))
      engine.waiting.put(r)
      head = {"id": f"ferrite-{id(r)}", "object": "chat.completion" if is_chat else "text_completion", "created": int(time.time()), "model": name}
      piece = lambda text, done: ({"index": 0, "delta": {"content": text} if text else {}} if is_chat else {"index": 0, "text": text}) | {"finish_reason": done}
      if q.get("stream"):
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
        send = lambda body: (self.wfile.write(f"data: {json.dumps(body)}\n\n".encode()), self.wfile.flush())
        head["object"] += ".chunk" if is_chat else ""
        try:
          if is_chat: send(head | {"choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
          while (text := r.out.get(timeout=wait)) is not None: send(head | {"choices": [piece(text, None)]})
          send(head | {"choices": [piece("", r.finish)]})
          return self.wfile.write(b"data: [DONE]\n\n")
        except (OSError, queue.Empty): r.gone = True  # the client went, or no text came: its slot is let go
        return
      try: text = "".join(iter(lambda: r.out.get(timeout=wait), None))
      except queue.Empty: return self.reply(504, {"error": {"message": f"no text in {wait}s: the chip hangs"}})
      if r.finish == "error": return self.reply(500, {"error": {"message": "the engine stopped"}})
      choice = {"index": 0, "message": {"role": "assistant", "content": text}} if is_chat else {"index": 0, "text": text, "logprobs": None}
      self.reply(200, head | {"choices": [choice | {"finish_reason": r.finish}],
                              "usage": {"prompt_tokens": len(ids), "completion_tokens": len(r.made), "total_tokens": len(ids) + len(r.made)}})
  return ThreadingHTTPServer(("", port), Handler)

def fetch(model, cache=os.path.expanduser("~/.cache/ferrite/hub")):
  """A model's folder: model itself if it is one, else Hugging Face's files for that model id, downloaded once into
  cache (from HF_ENDPOINT if set, with HF_TOKEN for a gated model): its config and tokenizer, the configs it has of
  generation and chat (and a chat template in a file of its own), and its weights, in one file or in the shards its
  index names. A finished download is marked (.done), and not asked for again."""
  if os.path.isdir(model): return model
  folder = os.path.join(cache, model)
  if os.path.exists(os.path.join(folder, ".done")): return folder
  os.makedirs(folder, exist_ok=True)
  def get(name, needed=True):
    path = os.path.join(folder, name)
    if os.path.exists(path): return path
    token = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"} if "HF_TOKEN" in os.environ else {}
    url = f"{os.environ.get('HF_ENDPOINT', 'https://huggingface.co')}/{model}/resolve/main/{name}"
    try:
      with urllib.request.urlopen(urllib.request.Request(url, headers=token), timeout=60) as r, open(path + ".part", "wb") as f: shutil.copyfileobj(r, f)
    except urllib.error.HTTPError as e:
      if needed or e.code != 404: raise
      return None
    os.replace(path + ".part", path)
    return path
  for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json", "chat_template.jinja"):
    get(name, name in ("config.json", "tokenizer.json"))
  if index := get("model.safetensors.index.json", False):
    for shard in sorted(set(llama.read_json(index)["weight_map"].values())): get(shard)
  else: get("model.safetensors")
  open(os.path.join(folder, ".done"), "w").close()
  return folder

def main():
  from ferrite import pci
  from ferrite.chip import Chip
  from ferrite.runtime import Runtime
  folder, args = fetch(sys.argv[1]), sys.argv[2:] + [None] * 4
  port, slots, places, mode = int(args[0] or 8000), int(args[1] or 4), int(args[2] or 512), args[3] or "bf16"
  chip = Chip(pci.open_device())
  rt, start = Runtime(chip, chip.tensix[0], chip.tensix[-1]), time.time()
  c, weights = llama.load(folder, places)
  c.precision = llama.precision(c, mode)
  engine = Engine(rt, llama.upload(rt, weights, c.precision), c, llama.Tokenizer(folder), slots, places)
  config = lambda f: llama.read_json(os.path.join(folder, f)) if os.path.exists(os.path.join(folder, f)) else {}
  eos = (config("generation_config.json") or config("config.json")).get("eos_token_id")
  threading.Thread(target=engine.run, daemon=True).start()
  print(f"{os.path.basename(folder.rstrip('/'))} on the chip in {time.time() - start:.0f}s ({mode}), {slots} slots of {places} places, port {port}", flush=True)
  jinja = os.path.join(folder, "chat_template.jinja")  # where transformers 5 keeps it
  template = config("tokenizer_config.json").get("chat_template") or (open(jinja, encoding="utf-8").read() if os.path.exists(jinja) else "")
  server(engine, os.path.basename(folder.rstrip("/")), template, eos if isinstance(eos, list) else [eos] if eos is not None else [], port).serve_forever()

if __name__ == "__main__": main()
