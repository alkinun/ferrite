# An OpenAI-compatible server for a Llama-family model on ferrite. Requests join and leave a batch
# that decodes together: each sequence has a slot of one KV cache (and of Gated DeltaNet's states), and every step is
# one replayed call for all the slots. A prompt goes to the free slot whose last sequence began the same way, or takes
# a session saved on the host that did, so what they share (a system prompt, an agent's conversation so far) is not
# prefilled again.
#   TTSIM=/path/to/libttsim.so python3 models/serve.py model [port] [slots] [places] [bf16|bfp8|bfp4|tt] [bf16|bfp8|bfp4]   # a folder or a
#   model id; the weights' formats (llama.precision), then the KV cache's
#   curl localhost:8000/v1/completions -d '{"prompt": "The meaning of life is", "max_tokens": 8, "temperature": 0}'
import collections, json, math, os, queue, re, shutil, sys, threading, time, urllib.error, urllib.request, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ferrite.tensor import Tensor, jit
import gpt2, llama

class Request:
  """A prompt's tokens and how to go on from them (pick), until a token in eos, a stop string or max_tokens. Each piece
  of new text goes into out as soon as no stop string could still begin in it, then None once the engine is done with
  it (end); finish says why. One whose client has gone (gone) ends at its next token, or takes no slot. cached: its
  prompt's tokens that its slot, or a session saved on the host, held already, which were not prefilled. logprobs: for
  each token made, its log probability and those of the logprobs most likely (probs), from the model's logits as they
  are; marks: how many it had at each piece of text. shape: what its text must keep to (pick), or None. room: the slot it
  holds from its offer to its end (Engine.offer), if it was offered."""
  def __init__(self, tokenizer, ids, max_tokens=16, temperature=1.0, top_p=1.0, stop=(), eos=(), seed=None, top_k=0, min_p=0.0,
               presence=0.0, frequency=0.0, repetition=1.0, logprobs=None, shape=None):
    self.tok, self.ids, self.max_tokens, self.temperature, self.top_p = tokenizer, list(ids), max_tokens, temperature, top_p
    self.stop, self.eos, self.rng = [stop] if isinstance(stop, str) else list(stop or ()), set(eos), np.random.default_rng(seed)
    self.top_k, self.min_p, self.presence, self.frequency, self.repetition, self.logprobs = top_k, min_p, presence, frequency, repetition, logprobs
    self.shape = shape
    self.made, self.said, self.out, self.finish, self.gone, self.cached, self.probs, self.marks = [], "", queue.Queue(), None, False, 0, [], []
    self.room = None

  def pick(self, logits):
    """The next token: the logits less the penalties (presence and frequency by the tokens made so far, repetition by
    the prompt's and those made, as vLLM applies them), then drawn (draw). With a shape (a function of the text made: 0
    if it cannot begin one, 1 if it can, 2 if it is a whole one), the token drawn is taken if the text it makes can
    still begin one (an end of sequence, if the text is a whole one), else drawn again with that one left out: no mask
    over 248K tokens a step, and a check or two when the model is in the shape already."""
    if self.presence or self.frequency or self.repetition != 1:
      made = np.bincount(self.made, minlength=len(logits))[:len(logits)]
      logits = logits - self.presence * (made > 0) - self.frequency * made
      seen = (made > 0) | (np.bincount(self.ids, minlength=len(logits))[:len(logits)] > 0)
      logits = np.where(seen, np.where(logits > 0, logits / self.repetition, logits * self.repetition), logits)
    if self.shape is None: return self.draw(logits)
    logits = np.array(logits, np.float32)
    for _ in range(32):
      if self.fits(token := self.draw(logits)): return token
      logits[token] = -np.inf
    return next((int(t) for t in np.argsort(-logits) if logits[t] > -np.inf and self.fits(int(t))), None)  # the likeliest that
    # fits, else None: the end (a whole shape, for a model without an end of sequence; else a shape that leads nowhere)

  def fits(self, token):  # whether the text with token keeps to the shape (or, token an end of sequence, the text is whole)
    return self.shape(self.tok.decode(self.made)) == 2 if token in self.eos else self.shape(self.tok.decode(self.made + [token])) > 0

  def draw(self, logits):
    """Greedy at temperature 0, else sampled at that temperature from the top_k most likely (0: all), the fewest of
    those whose probabilities sum to top_p, and of those the ones at least min_p of the most likely's. The fewest are
    sought among the 1,024 most likely first: a sort of a vocabulary of 248K is 5 ms of the host's a token, where a
    decode step of eight users takes the card 28."""
    if self.temperature == 0: return int(logits.argmax())
    p, keep = np.exp((logits - logits.max()) / self.temperature), None  # (None: every token)
    if self.top_k or self.top_p < 1:
      keep = np.argpartition(-p, (n := min(self.top_k or 1024, len(p))) - 1)[:n]
      keep = keep[np.argsort(-p[keep])]
      if not self.top_k and p[keep].sum() < self.top_p * p.sum(): keep = np.argsort(-p)  # (a flatter distribution's)
      if self.top_p < 1: keep = keep[:np.searchsorted(np.cumsum(p[keep]), self.top_p * (p[keep] if self.top_k else p).sum()) + 1]
    if self.min_p: keep = np.flatnonzero(p >= self.min_p * p.max()) if keep is None else keep[p[keep] >= self.min_p * p.max()]
    w = p if keep is None else p[keep]
    i = min(int(np.searchsorted(np.cumsum(w), self.rng.random() * w.sum(), side="right")), len(w) - 1)
    return i if keep is None else int(keep[i])

  def take(self, logits, room):  # the next token from its logits (room: whether its slot has a place for one more): done?
    if self.gone: self.finish = "gone"; return True
    token = self.pick(logits)
    end = token is None or token in self.eos
    if not end:
      self.made.append(token)
      if self.logprobs is not None:  # (the model's own: its logits before any penalty or temperature, as vLLM gives them)
        lp = logits - logits.max()
        lp = lp - np.log(np.exp(lp).sum())
        top = np.argpartition(-lp, self.logprobs)[:self.logprobs] if self.logprobs else []
        self.probs.append((token, float(lp[token]), [(int(t), float(lp[t])) for t in sorted(top, key=lambda t: -lp[t])]))
    text = self.tok.decode(self.made)
    cut = min((i for i in (text.find(s) for s in self.stop) if i >= 0), default=-1)
    if end or cut >= 0 or len(self.made) == self.max_tokens or not room:
      self.say(text if cut < 0 else text[:cut])
      self.finish = "stop" if end or cut >= 0 else "length"
      return True
    hold = max((k for s in self.stop for k in range(1, len(s)) if text.endswith(s[:k])), default=0)  # a stop string's start
    self.say(text[:len(text) - hold].rstrip("\ufffd"))  # (or a character whose bytes are still coming)
    return False

  def say(self, text):
    if len(text) > len(self.said): self.marks.append(len(self.probs)); self.out.put(text[len(self.said):]); self.said = text

  def end(self):
    if self.room: self.room.release(); self.room = None  # (its slot free for the next offer)
    self.out.put(None)

class Engine:
  """Sequences decoding together on the chip, a slot of `places` rows of one KV cache each (and Gated DeltaNet's states
  each). Every step runs every slot, one that is free or still prefilling on its last row, which no sequence reaches,
  and leaving its states as they are, so the step's shapes never change and it is replayed (jit). Its attention reads
  each slot's first rows alone, as many as the power of 2 (32 at least) that holds the longest sequence's, a replay
  recorded for each, so a step costs what its sequences hold, not their slots' size. A prompt is prefilled `chunk`
  tokens at a time, from what its slot holds of it already, a chunk between steps, so a long one slows the others'
  tokens rather than stopping them.
  A session (its tokens, its slot's rows of each cache and its states) is saved on the host when its prompt is in and
  again when its request ends, up to `keep` bytes, the least recently used going first, so an agent's next turn,
  whose prompt is its last prompt and answer and more, starts where the last left off whichever slot it gets. A KV
  cache's rows serve any start of what they hold; Gated DeltaNet's states are for exactly what they took in.
  A model with a multi-token prediction layer (Qwen3.5's) decodes speculatively: each step verifies every slot's next
  token and a draft after it in one call (llama.forward's accept), takes the second token too where the model's pick
  for it is the draft (a pick sampled as it would be: the tokens are the model's own at any temperature), and drafts
  the next in another call (llama.draft); a turned-down draft's Gated DeltaNet states are put back by the next step."""
  def __init__(self, rt, t, c, tokenizer, slots=4, places=512, chunk=256, keep=4 << 30):
    assert places % 32 == 0, "slots of whole tiles"
    self.rt, self.t, self.c, self.tok, self.places, self.chunk, self.keep = rt, t, c, tokenizer, places, chunk, keep
    self.spec = getattr(c, "mtp", 0) > 0 and "mtp.fc" in t  # (drafts from the multi-token prediction layer)
    self.kv = llama.kv_cache(c, slots * places, llama.zeros(rt, llama.cache_format(c)), slots, self.spec)
    self.slot = [llama.slot(c, self.kv, u, slots) for u in range(slots)]  # the caches as slot u's alone
    self.linear = any(llama.linear(c, l) for l in range(c.layers))  # (its states then say where a slot can go on from)
    self.held, self.busy = [[] for _ in range(slots)], [None] * slots
    self.used = [0] * slots  # when each slot last took a request: of those sharing as much, the one used longest ago goes
    self.replays, self.hidden = {}, {}  # (a call's replay for each length it reads, and verify's hidden states for each)
    self.drafts, self.taken = [0] * slots, [1.0] * slots  # each slot's draft, and whether its last was taken (its states then go on from both)
    self.drafted = self.agreed = 0  # (drafts compared with the model's pick, and taken: what speculation saves shows here)
    self.waiting, self.prefilled = queue.Queue(), 0  # (tokens prefilled in all: what the prefix cache saves shows here)
    self.room = threading.BoundedSemaphore(slots)  # a slot for each offered request, from its offer to its end
    self.saved = collections.OrderedDict()  # tokens -> (each block's bytes: a KV cache's rows, or Gated DeltaNet's states)

  def on(self, row, dtype="int32"): return Tensor.of(self.rt, [row], dtype)

  def offer(self, r):  # r queued if a slot is free for it, held until r ends; else False, at once, rather than a wait
    if not self.room.acquire(blocking=False): return False
    r.room = self.room
    self.waiting.put(r)
    return True

  def length(self, last):  # the rows of each slot a call reads when its sequences' last place is last
    return min(self.places, 1 << max(5, last.bit_length()))

  def replay(self, f, n):  # f's replay (jit) for calls reading n rows of each slot
    if (f, n) not in self.replays: self.replays[f, n] = jit(lambda *xs: f(*xs, length=n))
    return self.replays[f, n]

  def reach(self, held, ids):  # how much of a prompt can go on from what held is the state after (its last token runs again)
    if self.linear: return len(held) if len(held) < len(ids) and list(ids[:len(held)]) == list(held) else 0
    return min(next((i for i, (a, b) in enumerate(zip(held, ids)) if a != b), min(len(held), len(ids))), len(ids) - 1)

  def admit(self, r):  # into the free slot that holds the most of its prompt, or one that takes the saved session that does
    if r.gone: return r.end()
    free = [u for u, b in enumerate(self.busy) if b is None]
    u = max(free, key=lambda u: (self.reach(self.held[u], r.ids), -self.used[u]))
    best = max(self.saved, key=lambda k: self.reach(k, r.ids), default=())
    if self.reach(best, r.ids) > self.reach(self.held[u], r.ids): self.restore(u := min(free, key=lambda u: self.used[u]), best)
    self.used[u], self.busy[u] = max(self.used) + 1, r
    n = self.reach(self.held[u], r.ids)
    if self.linear and n < len(self.held[u]): self.restore(u)  # its states from the start again
    elif not self.taken[u]:  # its states after a turned-down draft, those after the token before it, where a prompt goes on from
      for l, s in enumerate(self.slot[u]):
        if llama.linear(self.c, l): s[0].assign(s[2]), s[1].assign(s[3])
      self.taken[u] = 1.0
    self.held[u], r.cached = r.ids[:n], n

  def save(self, u):  # slot u's session to the host: its tokens, its rows of each KV cache (whole tiles) and its states (after a
    key, rows = tuple(self.held[u]), -(-len(self.held[u]) // 32) * 32  # turned-down draft, those after the token before it)
    if not key or key in self.saved or not self.keep: return
    self.saved[key] = [tuple(x.tobytes() for x in (s[:2] if self.taken[u] else s[2:])) if llama.linear(self.c, l) else s[:, :rows, :].tobytes()
                       for l, s in enumerate(self.slot[u])]
    size = lambda v: sum(len(b) for x in v for b in (x if isinstance(x, tuple) else [x]))
    while sum(size(v) for v in self.saved.values()) > self.keep: self.saved.popitem(last=False)

  def restore(self, u, key=()):  # a saved session into slot u, its tokens what the slot holds (none: its states all 0)
    if key: self.saved.move_to_end(key)
    rows = -(-len(key) // 32) * 32
    for l, s in enumerate(self.slot[u]):
      data = self.saved[key][l] if key else None
      if llama.linear(self.c, l):
        for i, x in enumerate(s[:2]): x.assign(Tensor.frombytes(self.rt, x.rows, x.cols, data[i] if data else bytes(4 * x.batch * x.rows * x.cols), x.batch, "fp32"))
      elif data: s.put(self.on(list(range(rows))), Tensor.frombytes(self.rt, rows, s.cols, data, s.batch))
    self.held[u], self.taken[u] = list(key), 1.0

  def fill(self, u):  # the next chunk of slot u's prompt; the last one's last logits give the first new token (and, speculating,
    r, at = self.busy[u], len(self.held[u])  # the multi-token prediction layer over the chunk's places, the last its first draft)
    if r.gone: return self.take(u, None)
    ids = r.ids[at:at + self.chunk]
    places, n = self.on(list(range(at, at + len(ids)))), self.length(at + len(ids) - 1)
    out = llama.forward(self.on(ids), places, self.t, self.c, self.slot[u], hidden=self.spec, length=n)
    out, h = out if self.spec else (out, None)
    self.held[u] += ids
    self.prefilled += len(ids)
    if len(self.held[u]) == len(r.ids): self.save(u); self.take(u, gpt2.values(out)[0], 1 + self.spec)
    if self.spec and self.busy[u] is r:  # each place's next token: the prompt's, and past its end the one just taken
      drafts = gpt2.values(llama.draft(self.on((r.ids + r.made)[at + 1:at + len(ids) + 1]), places, h, self.t, self.c, self.slot[u], length=n, every=False))
      if r.made: self.drafts[u], self.taken[u] = self.guess(u, drafts[-1]), 1.0

  def advance(self):  # one step for every slot: each decoding one's last token in, its next one out
    live = [r if r and r.made else None for r in self.busy]
    if not any(live): return
    if self.spec: return self.speculate(live)
    ids = [r.made[-1] if r else 0 for r in live]
    places = [len(h) if r else self.places - 1 for h, r in zip(self.held, live)]
    step = self.replay(self.decode, self.length(max(p for p, r in zip(places, live) if r)))
    logits = gpt2.values(step(self.on(ids), self.on(places), self.on([float(bool(r)) for r in live], "fp32")))
    for u, r in enumerate(live):
      if r: self.held[u].append(r.made[-1]); self.take(u, logits[u])

  def take(self, u, logits, room=1):  # a slot's request takes its next token; a finished one leaves its slot, its session saved
    r = self.busy[u]                   # before it ends (so whoever waits on it finds the chip idle)
    if r.take(logits, len(self.held[u]) + room < self.places): self.save(u); self.busy[u] = None; r.end()
    return self.busy[u] is None

  def decode(self, ids, places, live, length):  # (replay's) a token of every slot
    return llama.forward(ids, places, self.t, self.c, self.kv, len(self.busy), live=live, length=length)

  def check(self, ids, places, live, accept, length):  # (speculating, replay's) each slot's token and draft taken in, after its
    logits, self.hidden[length] = llama.forward(ids, places, self.t, self.c, self.kv, len(self.busy), live=live, accept=accept, hidden=True, length=length)
    return logits  # last step's states (accept): their logits, and their hidden states kept for guesses

  def guesses(self, ids, places, taken, length):  # (replay's) the multi-token prediction layer over each slot's two last places
    n, d = len(self.busy), self.c.dim             # (the second the first again where its draft was turned down): its logits
    h = self.hidden[length].reshape(n, 2, d)
    first = h[:, 0:1, :]
    h = first.cat(first + taken.reshape(n, 1, 1) * (h[:, 1:2, :] - first), axis=1).reshape(2 * n, d).cast("bf16")  # (one or the other, exactly)
    return llama.draft(ids, places, h, self.t, self.c, self.kv, n, length)

  def guess(self, u, logits): return int(logits.argmax())  # slot u's draft, from the multi-token prediction layer's logits

  def speculate(self, live):  # one step for every slot: its token and its draft verified, the draft taken if the model's pick
    n, at = len(live), [len(h) for h in self.held]  # agrees, then the next drafts
    ids = [x for u, r in enumerate(live) for x in ([r.made[-1], self.drafts[u]] if r else [0, 0])]
    places = [p for u, r in enumerate(live) for p in ([at[u], at[u] + 1] if r else [self.places - 1] * 2)]
    m = self.length(max(at[u] + 1 for u, r in enumerate(live) if r))  # (the drafts' places too)
    logits = gpt2.values(self.replay(self.check, m)(self.on(ids), self.on(places), self.on([float(bool(r)) for r in live], "fp32"), self.on(self.taken, "fp32")))
    pairs = [[0, 0, self.places - 1, self.places - 1] for _ in range(n)]  # each slot's next tokens and their places, for the drafts
    for u, r in enumerate(live):
      if not r: continue
      self.held[u].append(r.made[-1])
      self.taken[u] = 0.0
      if self.take(u, logits[2 * u], 2): continue
      self.drafted += 1
      if r.made[-1] != self.drafts[u]: pairs[u] = [r.made[-1], r.made[-1], at[u], at[u]]; continue
      self.held[u].append(r.made[-1])
      self.taken[u], self.agreed = 1.0, self.agreed + 1
      if not self.take(u, logits[2 * u + 1], 2): pairs[u] = [self.drafts[u], r.made[-1], at[u], at[u] + 1]
    if not any(self.busy[u] is r for u, r in enumerate(live) if r): return
    out = gpt2.values(self.replay(self.guesses, m)(self.on([t for p in pairs for t in p[:2]]), self.on([q for p in pairs for q in p[2:]]), self.on(self.taken, "fp32")))
    for u, r in enumerate(live):
      if r and self.busy[u] is r: self.drafts[u] = self.guess(u, out[2 * u + 1])

  def run(self):  # the engine's thread: admit while a slot is free, a chunk of a prompt, a step; on an error, every request ends
    try:
      while True:
        if not any(self.busy): self.admit(self.waiting.get())
        while None in self.busy and not self.waiting.empty(): self.admit(self.waiting.get())
        if filling := [u for u, r in enumerate(self.busy) if r and not r.made]: self.fill(filling[0])
        self.advance()
    except BaseException:
      for r in self.busy + list(self.waiting.queue):
        if r: r.finish = "error"; r.end()
      raise

def chat(template, messages, thinking=True, effort="medium", tools=None):  # messages as the prompt the model was trained on, for the kinds of
  if "<tool_call>\n<function=" in template: return qwen(messages, thinking, tools)  # template it can be
  if tools: raise ValueError("tools, with a chat template other than Qwen3.5's")
  if "<|channel|>" in template:  # gpt-oss's (harmony): its own system message, a first system message as a developer's, the turns
    system = (f"You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\nCurrent date: {time.strftime('%Y-%m-%d')}\n\n"
              f"Reasoning: {effort}\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.")
    head, turns = f"<|start|>system<|message|>{system}<|end|>", messages
    if messages and messages[0]["role"] in ("system", "developer"):
      head, turns = head + f"<|start|>developer<|message|># Instructions\n\n{messages[0]['content']}\n\n<|end|>", messages[1:]
    said = lambda m: "<|start|>user<|message|>" if m["role"] == "user" else "<|start|>assistant<|channel|>final<|message|>"
    return head + "".join(f"{said(m)}{m['content']}<|end|>" for m in turns) + "<|start|>assistant"
  if "<|start_header_id|>" in template:  # Llama 3's
    turn = lambda m: f"<|start_header_id|>{m['role']}<|end_header_id|>\n\n{m['content']}<|eot_id|>"
    return "<|begin_of_text|>" + "".join(map(turn, messages)) + "<|start_header_id|>assistant<|end_header_id|>\n\n"
  if "<|im_start|>" in template:  # ChatML: SmolLM's instruct models', Qwen's; Qwen3's not thinking, its reasoning given as none
    prompt = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages) + "<|im_start|>assistant\n"
    return prompt if thinking or "<think>" not in template else prompt + "<think>\n\n</think>\n\n"
  raise ValueError("a chat template other than gpt-oss's, Llama 3's or ChatML")

CALLING = ("\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n"
           "<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\n"
           "that can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the "
           "specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required "
           "parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function "
           "call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do "
           "not tell the user about function calls\n</IMPORTANT>")  # Qwen3.5's template's words on how to call a tool

def qwen(messages, thinking=True, tools=None):
  """Qwen3.5's and Qwen3.6's chat template (its chat_template.jinja, for text): the tools described in the system turn,
  each assistant turn after the last question with its reasoning (an agent's tool calls since: the next prompt then
  goes on from the last one and its answer), tool calls as XML, tool results in a user turn of their own."""
  text = lambda m: (m.get("content") or "").strip()
  question = [i for i, m in enumerate(messages) if m["role"] == "user" and not (text(m).startswith("<tool_response>") and text(m).endswith("</tool_response>"))]
  if not question: raise ValueError("no question among the messages")
  system = text(messages[0]) if messages[0]["role"] == "system" else None
  out = ""
  if tools: out = "<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>" + "".join(
    "\n" + json.dumps(t, ensure_ascii=False) for t in tools) + "\n</tools>" + CALLING + (f"\n\n{system}" if system else "") + "<|im_end|>\n"
  elif system is not None: out = f"<|im_start|>system\n{system}<|im_end|>\n"
  for i, m in enumerate(messages):
    if m["role"] == "user": out += f"<|im_start|>user\n{text(m)}<|im_end|>\n"
    elif m["role"] == "assistant":
      content, reasoning = text(m), m.get("reasoning_content")
      if not isinstance(reasoning, str):
        reasoning = content.split("</think>")[0].rstrip("\n").split("<think>")[-1].lstrip("\n") if "</think>" in content else ""
        content = content.split("</think>")[-1].lstrip("\n")
      out += "<|im_start|>assistant\n" + (f"<think>\n{reasoning.strip()}\n</think>\n\n" if i > question[-1] else "") + content
      for j, call in enumerate(m.get("tool_calls") or []):
        f = call.get("function", call)
        args = json.loads(f.get("arguments") or "{}") if isinstance(f.get("arguments"), str) else f.get("arguments") or {}
        out += ("\n\n" if content.strip() else "") if j == 0 else "\n"
        out += f"<tool_call>\n<function={f['name']}>\n" + "".join(
          f"<parameter={k}>\n{v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)}\n</parameter>\n" for k, v in args.items()) + "</function>\n</tool_call>"
      out += "<|im_end|>\n"
    elif m["role"] == "tool":
      if i == 0 or messages[i - 1]["role"] != "tool": out += "<|im_start|>user"
      out += f"\n<tool_response>\n{text(m)}\n</tool_response>"
      if i == len(messages) - 1 or messages[i + 1]["role"] != "tool": out += "<|im_end|>\n"
    elif m["role"] != "system" or i: raise ValueError(f"a {m['role']} message there")
  return out + "<|im_start|>assistant\n" + ("<think>\n" if thinking else "<think>\n\n</think>\n\n")

def calls(text, tools=()):
  """An answer's tool calls in Qwen3.5's XML, OpenAI's way: (the text before them, the calls), each call's arguments
  as JSON, a value taken as the tool's schema says (a string as it is, anything else read as JSON where it is)."""
  head, mark, rest = text.partition("<tool_call>")
  schema = {t.get("function", t).get("name"): t.get("function", t).get("parameters", {}).get("properties", {}) for t in tools or ()}
  def value(name, key, v):
    if schema.get(name, {}).get(key, {}).get("type", "string") == "string": return v
    try: return json.loads(v)
    except ValueError: return v
  found = [(name, {k: value(name, k, v) for k, v in re.findall(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", body, re.S)})
           for name, body in re.findall(r"<function=([^>\n]+)>(.*?)</function>", mark + rest, re.S)]
  return head.strip(), [{"id": f"call_{uuid.uuid4().hex[:24]}", "type": "function", "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}
                        for name, args in found]

def chosen(choice, tools):
  """The shape tool_choice gives an answer (Request's: a function of its text, 0 if it cannot begin one, 1 if it can,
  2 if it is a whole one), or None for any (auto). none: no tool call; required: one or more calls of the tools, and
  nothing before them; a function by name: calls of it alone. A thinking model's answer is its text after its
  reasoning (answer)."""
  if choice in (None, "auto"): return None
  if choice == "none": return lambda text: 0 if "<tool_call>" in text else 2
  names = [t.get("function", t).get("name") for t in tools or ()]
  if choice != "required": names = [n for n in names if n == choice["function"]["name"]]
  if not names: raise ValueError(f"tool_choice: no such tool among the tools: {choice}")
  can = lambda part, whole: whole.startswith(part) or part.startswith(whole)  # (part, as far as it has come, is or begins whole)
  def fits(text):  # (at once: whitespace first could go on for ever)
    if not can(text, "<tool_call>\n<function=") or not all(any(can(part, n + ">") for n in names) for part in text.split("<function=")[1:]): return 0
    return 2 if text.rstrip().endswith("</tool_call>") and calls(text, tools)[1] else 1
  return fits

class Short(Exception): pass  # the text ends inside a value: a start of one, so far
class Bad(Exception): pass  # the text cannot begin one

def formatted(form):
  """The shape response_format gives an answer (chosen's kind), or None for any text: json_object, a JSON object;
  json_schema, a JSON value of its schema. Of JSON Schema, type (or several), properties, required,
  additionalProperties, items, minItems and maxItems, enum and const (as json.dumps writes them), anyOf and oneOf, $ref
  into $defs or definitions; the rest (a string's pattern or format, a number's bounds) is not kept to. Between its
  tokens is one space, tab or newline at most, and a number has 32 characters at most: either could go on for ever (a
  model's whitespace does, held to a shape: vLLM's Qwen models have). A comma comes only where a key or an item may
  follow it, so no start of a shape leads nowhere."""
  kind = (form or {}).get("type", "text")
  if kind == "text": return None
  if kind not in ("json_object", "json_schema"): raise ValueError(f"response_format: no {kind}")
  root = {"type": "object"} if kind == "json_object" else form["json_schema"]["schema"]
  def fits(text):
    i = 0
    def ws():
      nonlocal i
      start = i
      while i < len(text) and text[i] in " \t\n\r": i += 1
      if i - start > 1: raise Bad
    def peek():
      ws()
      if i == len(text): raise Short
      return text[i]
    def word(w):  # a literal (true, false, null), as far as the text goes
      nonlocal i
      if not w.startswith(text[i:i + len(w)]): raise Bad
      if i + len(w) > len(text): raise Short
      i += len(w)
    def string(allowed=None):  # its value (Short: the text ends inside it); Bad once it begins none of allowed, as JSON
      nonlocal i
      start, i = i, i + 1
      while i < len(text) and text[i] != '"':
        if ord(text[i]) < 32: raise Bad
        if text[i] == "\\":
          escape = text[i + 1:i + 2]
          if escape and escape not in '"\\/bfnrtu' or escape == "u" and not re.fullmatch("[0-9a-fA-F]*", text[i + 2:i + 6]): raise Bad
          i += 6 if escape == "u" else 2
        else: i += 1
      if allowed is not None and not any(json.dumps(a, ensure_ascii=False).startswith(text[start:i + 1]) for a in allowed): raise Bad
      if i >= len(text): raise Short
      i += 1
      return json.loads(text[start:i])
    def value(sc):
      nonlocal i
      sc = sc or {}
      while "$ref" in sc: sc = root["$defs" if "$defs" in sc["$ref"] else "definitions"][sc["$ref"].rsplit("/", 1)[1]]
      if "anyOf" in sc or "oneOf" in sc:  # (the first whole one; a start, if any could still be one)
        start, short = i, False
        for alt in sc.get("anyOf") or sc["oneOf"]:
          try: return value({k: v for k, v in sc.items() if k not in ("anyOf", "oneOf")} | alt)
          except Short: short = True
          except Bad: pass
          i = start
        raise Short if short else Bad
      c, kinds = peek(), sc.get("type") or ["object", "array", "string", "number", "boolean", "null"]
      kinds = [kinds] if isinstance(kinds, str) else kinds
      if "enum" in sc or "const" in sc:  # one of them, as json.dumps writes it
        texts = [json.dumps(v, ensure_ascii=False) for v in sc.get("enum", [sc.get("const")])]
        if done := [t for t in texts if text.startswith(t, i)]: i += len(max(done, key=len)); return
        raise Short if any(t.startswith(text[i:]) for t in texts) else Bad
      if c == '"' and "string" in kinds: string()
      elif c in "-0123456789" and {"number", "integer"} & set(kinds):
        start = i
        while i < len(text) and text[i] in "-+0123456789.eE": i += 1
        number = text[start:i]
        if len(number) > 32: raise Bad
        if not re.fullmatch(r"-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][-+]?[0-9]+)?", number):  # else a start of one, which a digit
          begun = re.fullmatch(r"-?((0|[1-9][0-9]*)(\.[0-9]*|(\.[0-9]+)?[eE][-+]?[0-9]*)?)?", number)  # more makes whole
          if i == len(text) and begun and len(number) < 32 and ("number" in kinds or not re.search("[.eE]", number)): raise Short
          raise Bad
        if "number" not in kinds and re.search("[.eE]", number): raise Bad
      elif c in "tfn":
        if ("null" if c == "n" else "boolean") not in kinds: raise Bad
        word({"t": "true", "f": "false", "n": "null"}[c])
      elif c == "{" and "object" in kinds:
        i, seen = i + 1, set()
        props, extra = sc.get("properties", {}), sc.get("additionalProperties", True)
        while True:
          if peek() == "}" and set(sc.get("required", ())) <= seen: i += 1; return
          if seen and (text[i] != "," or extra is False and set(props) <= seen): raise Bad  # (a comma only if a key may follow)
          if seen: i += 1
          if peek() != '"': raise Bad
          key = string(None if extra is not False else [k for k in props if k not in seen])
          if key in seen or key not in props and extra is False: raise Bad
          if peek() != ":": raise Bad
          i += 1
          value(props.get(key, extra if isinstance(extra, dict) else {}))
          seen.add(key)
      elif c == "[" and "array" in kinds:
        i, n = i + 1, 0
        while True:
          if peek() == "]" and n >= sc.get("minItems", 0): i += 1; return
          if n and text[i] != ",": raise Bad
          if n: i += 1
          if n >= sc.get("maxItems", math.inf): raise Bad
          value(sc.get("items", {}))
          n += 1
      else: raise Bad
    try:
      value(root)
      ws()
      return 2 if i == len(text) else 0
    except Short: return 1
    except Bad: return 0
  return fits

def answer(text):  # a thinking model's text as its reasoning and its answer (none until it comes): Qwen3's around </think>,
  if text.startswith("<|channel|>"):  # or gpt-oss's analysis channel, until its <|end|>, and its final one
    if "<|message|>" not in text: return "", ""  # (the channel's name still coming)
    head, _, content = text.partition("<|channel|>final<|message|>")
    return head.removeprefix("<|channel|>analysis<|message|>").partition("<|end|>")[0].strip(), content
  reasoning, _, content = text.partition("</think>")
  return reasoning.removeprefix("<think>").strip(), content.lstrip()

def server(engine, name, template="", eos=(), port=8000, wait=600):
  """An HTTP server for the engine (its thread must be running): /v1/models, and /v1/completions and
  /v1/chat/completions, whole or streamed (server-sent events). A model that thinks (its template has <think>, or
  gpt-oss's channels) gives its reasoning apart from its answer (reasoning_content), unless chat_template_kwargs'
  enable_thinking is false (Qwen3's); gpt-oss reasons as hard as reasoning_effort says (low, medium or high). Given
  tools (Qwen3.5's template), the answer's tool calls come as OpenAI's tool_calls, a stream's after its text. Its
  usage counts the prompt's tokens that were cached (prompt_tokens_details), as OpenRouter bills them, and ends a stream
  too, a chunk of its own, as OpenRouter's does. A piece of text not ready in `wait` seconds ends the request: a hung
  chip is reported, not waited on."""
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
        thinking, tools = (q.get("chat_template_kwargs") or {}).get("enable_thinking", True) is not False, is_chat and q.get("tools")
        prompt = chat(template, q["messages"], thinking, q.get("reasoning_effort") or "medium", tools) if is_chat else q["prompt"]
        ids = engine.tok.encode(prompt, special=is_chat)
        top = (q.get("top_logprobs") or 0 if q.get("logprobs") else None) if is_chat else q.get("logprobs")  # (a completion's: how many)
        if top is not None and not 0 <= top <= 20: raise ValueError("logprobs: at most 20 a token")
        apart = is_chat and (thinking and "<think>" in template or "<|channel|>" in template)
        choice, form = q.get("tool_choice"), formatted(q.get("response_format"))
        fits = chosen(choice, tools)
        if form and choice in (None, "auto", "none"):  # its format, or with tools it may call (auto) a call
          call = chosen("required", tools) if tools and choice != "none" else None
          fits = form if call is None else (lambda text: max(form(text), call(text)))
        shape = fits and apart and (lambda text: fits(answer(text)[1])) or fits  # (a thinking model's on its answer alone)
        r = Request(engine.tok, ids, q.get("max_tokens") or (engine.places if is_chat else 16), q.get("temperature", 1.0), q.get("top_p", 1.0),
                    q.get("stop"), eos, q.get("seed"), max(q.get("top_k") or 0, 0), q.get("min_p") or 0.0, q.get("presence_penalty") or 0.0,
                    q.get("frequency_penalty") or 0.0, q.get("repetition_penalty") or 1.0, top, shape)
      except (ValueError, KeyError, TypeError) as e: return self.reply(400, {"error": {"message": f"{type(e).__name__}: {e}"}})
      if len(ids) >= engine.places - 1: return self.reply(400, {"error": {"message": f"{len(ids)} tokens: a slot holds {engine.places - 2}"}})
      if not engine.offer(r): return self.reply(429, {"error": {"message": "every slot is busy", "code": 429}})  # (OpenRouter asks for one at
      # once under load, rather than a wait: it tries the request's next provider, and does not count it against uptime)
      head = {"id": f"ferrite-{id(r)}", "object": "chat.completion" if is_chat else "text_completion", "created": int(time.time()), "model": name}
      seen = {"text": "", "reasoning_content": "", "content": ""}
      def parts(text):  # a chat's text as a thinking model's reasoning and answer, apart, and only the answer's text before its tool calls
        out = dict(zip(("reasoning_content", "content"), answer(text))) if apart else {"content": text}
        if tools:  # (and before what may be the start of one)
          shown = out["content"].partition("<tool_call>")[0]
          out["content"] = shown[:len(shown) - max((k for k in range(1, 11) if shown.endswith("<tool_call>"[:k])), default=0)]
        return out
      def delta(text):  # a chat's next piece: what is new of each part
        seen["text"] += text
        new = parts(seen["text"])
        out = {k: v[len(seen[k]):] for k, v in new.items() if len(v) > len(seen[k])}
        seen.update(new)
        return out
      called = lambda text: calls(answer(text)[1] if apart else text, tools) if tools else (None, [])  # the answer's text and tool calls
      piece = lambda text, done: ({"index": 0, "delta": delta(text)} if is_chat else {"index": 0, "text": text}) | {"finish_reason": done}
      word, raw = lambda t: engine.tok.decode([t]), lambda t: list(engine.tok.bytes[t])
      def chances(made):  # tokens' log probabilities as OpenAI gives them: a chat's for its content, a completion's as lists
        if is_chat: return {"content": [{"token": word(t), "logprob": p, "bytes": raw(t), "top_logprobs": [{"token": word(a), "logprob": b, "bytes": raw(a)}
                                                                                                              for a, b in top]} for t, p, top in made]}
        return {"tokens": [word(t) for t, _, _ in made], "token_logprobs": [p for _, p, _ in made], "top_logprobs": [{word(a): b for a, b in top} for _, _, top in made]}
      def logprobs(k=None):  # its tokens' log probabilities, or those of its k-th piece's
        if r.logprobs is None: return {"logprobs": None}
        return {"logprobs": chances(r.probs if k is None else r.probs[r.marks[k - 1] if k else 0:r.marks[k]])}
      usage = lambda: {"prompt_tokens": len(ids), "completion_tokens": len(r.made), "total_tokens": len(ids) + len(r.made),
                       "prompt_tokens_details": {"cached_tokens": r.cached}}
      if q.get("stream"):
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
        send = lambda body: (self.wfile.write(f"data: {json.dumps(body)}\n\n".encode()), self.wfile.flush())
        head["object"] += ".chunk" if is_chat else ""
        try:
          if is_chat: send(head | {"choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
          for k, text in enumerate(iter(lambda: r.out.get(timeout=wait), None)):  # (each piece with its tokens' log probabilities)
            send(head | {"choices": [piece(text, None) | logprobs(k)]})
          if made := called(seen["text"])[1]:
            send(head | {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": i} | c for i, c in enumerate(made)]}, "finish_reason": None}]})
          send(head | {"choices": [piece("", "tool_calls" if made else r.finish)]})
          send(head | {"choices": [], "usage": usage()})
          return self.wfile.write(b"data: [DONE]\n\n")
        except (OSError, queue.Empty): r.gone = True  # the client went, or no text came: its slot is let go
        return
      try: text = "".join(iter(lambda: r.out.get(timeout=wait), None))
      except queue.Empty: return self.reply(504, {"error": {"message": f"no text in {wait}s: the chip hangs"}})
      if r.finish == "error": return self.reply(500, {"error": {"message": "the engine stopped"}})
      message, (content, made) = parts(text), called(text)
      if made: message |= {"content": content or None, "tool_calls": made}
      choice = ({"index": 0, "message": {"role": "assistant"} | message} if is_chat else {"index": 0, "text": text}) | logprobs()
      self.reply(200, head | {"choices": [choice | {"finish_reason": "tool_calls" if made else r.finish}], "usage": usage()})
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
  folder, args = fetch(sys.argv[1]), sys.argv[2:] + [None] * 5
  port, slots, places, mode, kv = int(args[0] or 8000), int(args[1] or 4), int(args[2] or 512), args[3] or "bf16", args[4]
  chip = Chip(pci.open_device())
  rt, start = Runtime(chip, chip.tensix[0], chip.tensix[-1]), time.time()
  c, weights = llama.load(folder, places)
  c.precision = llama.precision(c, mode, kv)
  chunk = 2048 if c.experts else 256  # a mixture of experts' chunk reads nearly all of every block's experts (cost.prompt)
  engine = Engine(rt, llama.upload(rt, weights, c.precision), c, llama.Tokenizer(folder), slots, places, chunk)
  config = lambda f: llama.read_json(os.path.join(folder, f)) if os.path.exists(os.path.join(folder, f)) else {}
  eos = (config("generation_config.json") or config("config.json")).get("eos_token_id")
  threading.Thread(target=engine.run, daemon=True).start()
  print(f"{os.path.basename(folder.rstrip('/'))} on the chip in {time.time() - start:.0f}s ({mode}, a {llama.cache_format(c)} KV cache), {slots} slots of {places} places, "
        f"prompts {chunk} tokens at a time, port {port}", flush=True)
  jinja = os.path.join(folder, "chat_template.jinja")  # where transformers 5 keeps it
  template = config("tokenizer_config.json").get("chat_template") or (open(jinja, encoding="utf-8").read() if os.path.exists(jinja) else "")
  server(engine, os.path.basename(folder.rstrip("/")), template, eos if isinstance(eos, list) else [eos] if eos is not None else [], port).serve_forever()

if __name__ == "__main__": main()
