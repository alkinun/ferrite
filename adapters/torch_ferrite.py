# ferrite under PyTorch: a ferrite tensor in torch's clothes. FerriteTensor is a torch.Tensor subclass with no storage of
# its own, holding a ferrite tensor of any rank (adapters/views.py). Every aten op on one comes to __torch_dispatch__ and
# becomes ferrite's lazy ops at once, which ferrite fuses until something reads the data back: so eager PyTorch, its
# autograd and its optimizers run on the chip, and weights and optimizer state stay there between steps. An op ferrite
# has no version of is decomposed into simpler aten ops by PyTorch's own decompositions, and failing that runs on the
# CPU, its tensors brought over. PyTorch has no device of ours without C++, so a FerriteTensor says it is on the CPU.
#   from adapters.torch_ferrite import ferrite; model, x = ferrite(model), ferrite(x); ... loss.item(), y.cpu()
import math, os, sys
import numpy as np
import torch
from torch._decomp import core_aten_decompositions, decomposition_table
from torch.utils._pytree import tree_map
sys.path.insert(0, os.path.dirname(__file__))
from views import View, Value, broadcast, common_groups, download, materialize, operand, reduce, spread, upload

aten = torch.ops.aten
prims = torch.ops.prims
DECOMPOSITIONS = core_aten_decompositions()  # into core aten ops (the rest of PyTorch's decompositions reach prims, also handled)
DTYPES = {torch.float32: "fp32", torch.bfloat16: "bf16", torch.int32: "int32", torch.int64: "int32", torch.bool: None}  # int64 as int32

class FerriteTensor(torch.Tensor):
  @staticmethod
  def __new__(cls, v, requires_grad=False):  # v: a Value whose dtype is torch's
    t = torch.Tensor._make_wrapper_subclass(cls, v.shape, dtype=v.dtype, device="cpu", requires_grad=requires_grad)
    t.v = v
    return t
  def __repr__(self, **kwargs): return f"FerriteTensor({self.cpu()})"
  def cpu(self, memory_format=None): return to_cpu(self)  # PyTorch would see a tensor on the CPU already
  def numpy(self): return to_cpu(self).numpy()
  @classmethod
  def __torch_dispatch__(cls, func, types, args=(), kwargs=None): return dispatch(func, args, kwargs or {})

def ferrite(x):
  """x on the chip as a FerriteTensor (one already is returned as it is), or a module with its parameters and buffers moved
  there, those shared between modules (tied weights) still shared."""
  if isinstance(x, torch.nn.Module):
    moved = {}
    for m in x.modules():
      for name, p in m.named_parameters(recurse=False):
        if id(p) not in moved: moved[id(p)] = torch.nn.Parameter(ferrite(p.detach()), p.requires_grad)
        setattr(m, name, moved[id(p)])
      for name, b in m.named_buffers(recurse=False): setattr(m, name, ferrite(b))
    return x
  if isinstance(x, FerriteTensor): return x
  a = x.detach().contiguous()
  a = a.view(torch.int16).numpy().view(np.uint16) if a.dtype is torch.bfloat16 else a.numpy()
  return FerriteTensor(Value(upload(a, DTYPES[x.dtype] or "int32"), View(tuple(x.shape)), x.dtype))

def to_cpu(t):  # a FerriteTensor as a torch CPU tensor
  a = download(t.v)
  if t.dtype is torch.bfloat16: return torch.from_numpy(a.view(np.int16).copy()).view(torch.bfloat16)
  return torch.from_numpy(np.array(a)).to(t.dtype)

def wrap(t, shape, dtype): return FerriteTensor(Value(t, View(tuple(shape)), dtype))
def view(x, v): return FerriteTensor(Value(x.v.t, v, x.dtype))
def chip(x): return hasattr(x, "eq")  # a ferrite tensor, not a number

def ew(f, args, dtype=None):
  """An elementwise op on FerriteTensors and numbers (a plain tensor goes up first, a 0-d one is a number), numpy's
  broadcasting: f of ferrite tensors, folded alike, or numbers."""
  num = lambda a: a.item() if a.dim() == 0 else ferrite(a).v
  vals = [a.v if isinstance(a, FerriteTensor) else num(a) if isinstance(a, torch.Tensor) else a for a in args]
  shape = tuple(torch.broadcast_shapes(*[v.shape for v in vals if isinstance(v, Value)]))
  vals = [broadcast(v, shape) if isinstance(v, Value) else v for v in vals]
  groups, dtype = common_groups(shape, vals), dtype or next(v.dtype for v in vals if isinstance(v, Value))
  out = f(*[operand(v, shape, groups) if isinstance(v, Value) else v for v in vals])
  if DTYPES[dtype] == "int32" and out.dtype == "fp32": out = out.cast("int32")  # ferrite multiplies int32 in fp32
  return FerriteTensor(Value(spread(out, shape, groups), View(shape), dtype))

def dims(d, n):  # torch's dim argument as a list of axes
  return list(range(n)) if d is None or d == [] else [k % n for k in ([d] if isinstance(d, int) else d)]

def reduction(op, x, dim=None, keepdim=False):
  axes = dims(dim, x.dim())
  shape = [1 if k in axes else n for k, n in enumerate(x.shape)] if keepdim else [n for k, n in enumerate(x.shape) if k not in axes]
  return wrap(reduce(x.v, axes, op), shape, x.dtype)

def reshape(x, shape):
  shape = list(shape)
  if -1 in shape: shape[shape.index(-1)] = x.numel() // -math.prod(shape)
  v = x.v.view.reshape(shape)
  return view(x, v) if v is not None else wrap(materialize(x.v), shape, x.dtype)  # a reshape the view cannot follow moves first

def select(x, dim, i):  # one place along dim, the axis gone
  d = dim % x.dim()
  return reshape(sliced(x, d, i % x.shape[d], i % x.shape[d] + 1), x.shape[:d] + x.shape[d + 1:])

def sliced(x, dim, start=None, end=None, step=1):
  d = dim % x.dim()
  start, end, _ = slice(start, end).indices(x.shape[d])
  x = view(x, x.v.view.shrink([(start, max(start, end)) if k == d else (0, n) for k, n in enumerate(x.shape)]))
  if step == 1: return x
  v = x.v.view if not x.v.view.mask else View(x.shape)  # padding and a step: moved first
  t = x.v.t if not x.v.view.mask else materialize(x.v)
  return FerriteTensor(Value(t, View([-(-n // step) if k == d else n for k, n in enumerate(v.shape)],
                                     [s * step if k == d else s for k, s in enumerate(v.strides)], v.offset), x.dtype))

def matmul(a, b):  # mm and bmm (PyTorch makes every matmul one of them): ferrite's, each operand as (batch, rows, columns)
  three = lambda x: materialize(x.v, [list(range(x.dim() - 2)), [x.dim() - 2], [x.dim() - 1]])
  return wrap(three(a) @ three(b), a.shape[:-1] + (b.shape[-1],), a.dtype)

def cat(xs, dim=0):  # tinygrad's way: each padded with zeros into its place, and summed
  xs = [x for x in xs if x.numel()]
  d, n, at, out = dim % xs[0].dim(), sum(x.shape[dim] for x in xs), 0, None
  for x in xs:
    placed = view(x, x.v.view.pad([(at, n - at - x.shape[d]) if k == d else (0, 0) for k in range(x.dim())]))
    out, at = placed if out is None else ew(lambda p, q: p + q, [out, placed]), at + x.shape[d]
  return out

def pad(x, widths, value=0):  # torch's (before, after) pairs from the last axis back: zeros outside by the view, then value there
  w = [(0, 0)] * (x.dim() - len(widths) // 2) + [tuple(widths[i:i + 2]) for i in range(len(widths) - 2, -1, -2)]
  if min(min(w)) < 0: return on_cpu(aten.constant_pad_nd.default, (x, widths, value), {})  # a crop as well
  out, ones = view(x, x.v.view.pad(w)), constant(x, 1)
  return ew(lambda p, m: p - m * value + value, [out, view(ones, ones.v.view.pad(w))]) if value else out

def constant(like, value, shape=None, dtype=None):  # a value spread over a shape: one value uploaded, and a view
  shape, dtype = tuple(shape if shape is not None else like.shape), dtype or like.dtype
  return FerriteTensor(Value(upload(np.full((1,), value), DTYPES[dtype] or "int32"), View((1,) * len(shape)).expand(shape), dtype))

def to_copy(a, dtype=None, device=None, **kwargs):  # to the CPU, or another dtype on the chip
  if device is not None and torch.device(device).type == "cpu": return to_cpu(a).to(dtype or a.dtype)
  if dtype is None or dtype == a.dtype: return FerriteTensor(a.v)
  return ew(lambda p: p.cast(DTYPES[dtype]) if DTYPES[dtype] else p.ne(0), [a], dtype)

def transpose(a, d0, d1):
  order = list(range(a.dim()))
  order[d0 % a.dim()], order[d1 % a.dim()] = order[d1 % a.dim()], order[d0 % a.dim()]
  return view(a, a.v.view.permute(order))

def expand(a, shape, implicit=False):
  full = (1,) * (len(shape) - a.dim()) + tuple(a.shape)
  return view(a, broadcast(a.v, [m if n == -1 else n for n, m in zip(shape, full)]).view)

def index(x, indices):
  """x indexed along one axis by a tensor of ids (an embedding lookup, a cache's positions): ferrite's gather, of that
  axis's rows of the rest, the ids a row of int32, the result's axes put where PyTorch puts them; more, the CPU's."""
  d = [k for k, i in enumerate(indices) if i is not None]
  if len(d) != 1 or not isinstance(indices[d[0]], FerriteTensor): return on_cpu(aten.index.Tensor, (x, indices), {})
  d, ids, rest = d[0], indices[d[0]], [k for k in range(x.dim()) if k != d[0]]
  rows = materialize(x.v, [[], [d], rest]).gather(materialize(ids.v, [[], [], list(range(ids.dim()))]))
  out, n = wrap(rows, ids.shape + tuple(x.shape[k] for k in rest), x.dtype), ids.dim()
  return view(out, out.v.view.permute(list(range(n, n + d)) + list(range(n)) + list(range(n + d, out.dim()))))

def most(a, dim=None, keepdim=False):  # max of it all (generate asks if any sequence is unfinished); with dim, the CPU's (indices too)
  if isinstance(dim, torch.Tensor): return OPS[aten.maximum](a, dim)
  return reduction("max", a) if dim is None else on_cpu(aten.max.dim, (a, dim, keepdim), {})

def places(x, d): return ferrite(torch.arange(x.shape[d], dtype=torch.int32).reshape([-1 if k == d else 1 for k in range(x.dim())]))

def gather(x, dim, index, sparse_grad=False):  # one element along dim per row (cross entropy's pick): a one-hot, multiplied in
  d = dim % x.dim()
  if index.shape[d] != 1: return on_cpu(aten.gather.default, (x, dim, index), {})
  return reduction("sum", ew(lambda p, q, i: p * q.eq(i).cast(p.dtype), [x, places(x, d), index], x.dtype), d, keepdim=True)

def scatter_value(x, dim, index, value):  # value at index along dim (cross entropy's one-hot): picked where the place is index
  d = dim % x.dim()
  if isinstance(value, torch.Tensor): return on_cpu(aten.scatter.src, (x, dim, index, value), {})  # values to scatter: the CPU's
  if index.shape[d] != 1: return on_cpu(aten.scatter.value, (x, dim, index, value), {})
  return ew(lambda p, q, i: q.eq(i).where(value, p), [x, places(x, d), index], x.dtype)

def attention_backward(grad, q, k, v, out, lse, dropout_p, is_causal, attn_mask=None, scale=None):
  """PyTorch's attention kernel's backward (it has no decomposition), written in PyTorch's ops, which come back here: the
  scores' softmax again (the forward's decomposition keeps the weights where the kernel keeps their logsumexp, so lse is
  not read), then the chain rule through the two matmuls. With fewer key and value heads than query heads (grouped-query
  attention), each serves a group of query heads in turn: spread over them, and their gradients summed back."""
  if dropout_p: return on_cpu(aten._scaled_dot_product_flash_attention_for_cpu_backward.default, (grad, q, k, v, out, lse, dropout_p, is_causal),
                              {"attn_mask": attn_mask, "scale": scale})  # its mask was random: the CPU's
  scale, g = scale or 1 / math.sqrt(q.shape[-1]), q.shape[-3] // k.shape[-3]
  if g > 1: k, v = (x.unsqueeze(-3).expand(*x.shape[:-2], g, *x.shape[-2:]).flatten(-4, -3) for x in (k, v))
  s = q @ k.transpose(-2, -1) * scale
  if is_causal: s = s + ferrite(torch.full(s.shape[-2:], -math.inf).triu(1))
  if attn_mask is not None: s = s + attn_mask if attn_mask.dtype is not torch.bool else torch.where(attn_mask, s, -math.inf)
  p = s.softmax(-1)
  ds = p * (grad @ v.transpose(-2, -1) - (grad * out).sum(-1, keepdim=True))
  dk, dv = ds.transpose(-2, -1) @ q * scale, p.transpose(-2, -1) @ grad
  if g > 1: dk, dv = (d.unflatten(-3, (-1, g)).sum(-3) for d in (dk, dv))
  return ds @ k * scale, dk, dv

def embedding_backward(grad, ids, rows, padding_idx, scale_grad_by_freq):  # the rows' one-hot, transposed, times grad
  if scale_grad_by_freq: return on_cpu(aten.embedding_dense_backward.default, (grad, ids, rows, padding_idx, True), {})
  hot = places(torch.empty(rows, 1), 0) == ids.reshape(1, -1)
  if padding_idx >= 0: hot = hot & (ids.reshape(1, -1) != padding_idx)
  return hot.to(grad.dtype) @ grad.reshape(ids.numel(), -1)

def erf(p):  # Abramowitz and Stegun 7.1.26, within 1.5e-7: 1 - t q(t) e^(-x^2), t = 1 / (1 + 0.3275911 |x|), odd
  a = p.maximum(-p)
  t = (a * 0.3275911 + 1).reciprocal()
  q = ((((t * 1.061405429 - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t
  e = 1 - q * (-(a * a)).exp()
  return (p < 0).where(-e, e)

def variance(x, dim=None, correction=1, keepdim=False):
  n = math.prod(x.shape[k] for k in dims(dim, x.dim()))
  d = ew(lambda p, m: p - m, [x, mean(x, dim, keepdim=True)])
  return ew(lambda q: q * (1 / max(n - correction, 1)), [reduction("sum", ew(lambda p: p * p, [d]), dim, keepdim)])

def mean(x, dim=None, keepdim=False, dtype=None):
  s = reduction("sum", x, dim, keepdim)
  return ew(lambda p: p * (s.numel() / x.numel()), [s])

def broadcast_in_dim(a, shape, kept):  # a's axes at kept in shape, 1 elsewhere, then spread
  return expand(reshape(a, [a.shape[kept.index(k)] if k in kept else 1 for k in range(len(shape))]), shape)

def unary(f): return lambda a: ew(f, [a])
def logical(f, op): return lambda a, b: ew(f, [a, b]) if a.dtype is torch.bool else on_cpu(op, (a, b), {})  # bools; ints the CPU's
POWERS = {1: lambda p: p * 1, 2: lambda p: p * p, 3: lambda p: p * p * p, 0.5: lambda p: p.sqrt(), -0.5: lambda p: p.rsqrt(),
          -1: lambda p: p.reciprocal()}
def quotient(a, b): return t if (t := torch.result_type(a, b)).is_floating_point else torch.get_default_dtype()  # of a true division
COMPARE = {aten.lt: lambda p, q: p < q, aten.gt: lambda p, q: p > q, aten.le: lambda p, q: p <= q, aten.ge: lambda p, q: p >= q,
           aten.eq: lambda p, q: p.eq(q) if chip(p) else q.eq(p), aten.ne: lambda p, q: p.ne(q) if chip(p) else q.ne(p)}
OPS = {  # aten op -> what it is on FerriteTensors (their in-place versions come from these)
  aten.add: lambda a, b, alpha=1: ew(lambda p, q: p + q * alpha if alpha != 1 else p + q, [a, b]),
  aten.sub: lambda a, b, alpha=1: ew(lambda p, q: p - q * alpha if alpha != 1 else p - q, [a, b]),
  aten.rsub: lambda a, b, alpha=1: ew(lambda p, q: q * alpha - p if alpha != 1 else q - p, [a, b]),
  aten.mul: lambda a, b: ew(lambda p, q: p * q, [a, b]),
  aten.div: lambda a, b: ew(lambda p, q: p * q.reciprocal() if chip(q) else p * (1 / q), [a, b], quotient(a, b)),
  aten.maximum: lambda a, b: ew(lambda p, q: p.maximum(q) if chip(p) else q.maximum(p), [a, b]),
  aten.minimum: lambda a, b: ew(lambda p, q: p.minimum(q) if chip(p) else q.minimum(p), [a, b]),
  aten.where: lambda c, a, b: ew(lambda c, p, q: c.where(p, q), [c, a, b], a.dtype if isinstance(a, torch.Tensor) else b.dtype),
  **{op: (lambda f: lambda a, b: ew(f, [a, b], torch.bool))(f) for op, f in COMPARE.items()},
  aten.neg: unary(lambda p: -p), aten.exp: unary(lambda p: p.exp()), aten.log: unary(lambda p: p.log()), aten.sqrt: unary(lambda p: p.sqrt()),
  aten.rsqrt: unary(lambda p: p.rsqrt()), aten.reciprocal: unary(lambda p: p.reciprocal()), aten.sin: unary(lambda p: p.sin()),
  aten.cos: unary(lambda p: p.cos()), aten.relu: unary(lambda p: p.relu()), aten.sigmoid: unary(lambda p: p.sigmoid()),
  aten.erf: unary(erf), aten.abs: unary(lambda p: p.maximum(-p)), aten.mean: mean, aten.var: variance,
  aten.sum: lambda a, dim=None, keepdim=False, dtype=None: reduction("sum", a, dim, keepdim),
  aten.amax: lambda a, dim=None, keepdim=False: reduction("max", a, dim, keepdim),
  aten.mm: matmul, aten.bmm: matmul, aten.cat: cat,
  aten.view: reshape, aten._unsafe_view: reshape, aten.reshape: reshape, aten.expand: expand, aten.transpose: transpose,
  aten.permute: lambda a, order: view(a, a.v.view.permute(dims(order, a.dim()))),
  aten.t: lambda a: view(a, a.v.view.permute(list(range(a.dim()))[::-1])),
  aten.unsqueeze: lambda a, d: reshape(a, a.shape[:d % (a.dim() + 1)] + (1,) + a.shape[d % (a.dim() + 1):]),
  aten.squeeze: lambda a, d=None: reshape(a, [n for k, n in enumerate(a.shape) if n != 1 or (d is not None and k not in dims(d, a.dim()))]),
  aten.slice: sliced, aten.select: select, aten.index: index, aten.constant_pad_nd: pad,
  aten.unflatten: lambda a, d, sizes: reshape(a, a.shape[:d % a.dim()] + tuple(sizes) + a.shape[d % a.dim() + 1:]),
  aten.embedding_dense_backward: embedding_backward,
  aten.flatten: lambda a, start=0, end=-1: reshape(a, a.shape[:start % max(a.dim(), 1)] + (-1,) + a.shape[end % max(a.dim(), 1) + 1:]),
  aten.clone: lambda a, memory_format=None: FerriteTensor(a.v), aten.detach: lambda a: FerriteTensor(a.v), aten.lift_fresh: lambda a: a,
  aten.zeros_like: lambda a, **k: constant(a, 0, dtype=k.get("dtype")), aten.ones_like: lambda a, **k: constant(a, 1, dtype=k.get("dtype")),
  aten.full_like: lambda a, value, **k: constant(a, value, dtype=k.get("dtype")),
  aten.new_zeros: lambda a, size, **k: constant(a, 0, size, k.get("dtype")), aten.new_ones: lambda a, size, **k: constant(a, 1, size, k.get("dtype")),
  aten.new_full: lambda a, size, value, **k: constant(a, value, size, k.get("dtype")),
  aten._to_copy: to_copy, aten.to: lambda a, dtype=None, *rest, **k: to_copy(a, dtype if isinstance(dtype, torch.dtype) else k.get("dtype")),
  aten.gather: gather, aten.scatter: scatter_value, aten._scaled_dot_product_flash_attention_for_cpu_backward: attention_backward,
  aten.bitwise_not: lambda a: ew(lambda p: p.eq(0), [a]) if a.dtype is torch.bool else on_cpu(aten.bitwise_not.default, (a,), {}),
  aten.bitwise_and: logical(lambda p, q: p.minimum(q), aten.bitwise_and.Tensor), aten.bitwise_or: logical(lambda p, q: p.maximum(q), aten.bitwise_or.Tensor),
  aten.max: most,
  aten._local_scalar_dense: lambda a: download(a.v).reshape(-1)[0].item(),
}

PRIMS = {prims.view_of: lambda a: FerriteTensor(a.v), prims.pow: lambda a, e: ew(POWERS[e], [a]) if e in POWERS else on_cpu(prims.pow.default, (a, e), {}),
         prims.sum: lambda a, dims, output_dtype=None: reduction("sum", a, dims), prims.var: lambda a, dims, correction=1, output_dtype=None:
         variance(a, dims, correction), prims.broadcast_in_dim: broadcast_in_dim, prims.convert_element_type: lambda a, dtype: to_copy(a, dtype)}

def dispatch(func, args, kwargs):
  """func on ferrite: its own version, or the in-place one made from it (the result kept, and computed now, so a training
  loop does not grow an endless lazy graph), or PyTorch's decomposition, or the CPU."""
  packet, name = func.overloadpacket, func.overloadpacket.__name__
  args = tree_map(lambda x: ferrite(x) if type(x) is torch.Tensor and x.dim() else x, args)  # a model's own CPU tensors (positions)
  if packet in OPS and not (packet is aten.div and kwargs.get("rounding_mode")): return OPS[packet](*args, **kwargs)
  if name.endswith("_") and isinstance(args[0], FerriteTensor) and (base := getattr(aten, name[:-1], None)) in OPS:
    return keep(args[0], OPS[base](*args, **kwargs))
  if name == "copy_": return keep(args[0], args[1])  # into a FerriteTensor, from anywhere
  if name in ("fill_", "zero_"): return keep(args[0], constant(args[0], args[1] if name == "fill_" else 0))
  if func in DECOMPOSITIONS: return DECOMPOSITIONS[func](*args, **kwargs)
  if func in decomposition_table: return decomposition_table[func](*args, **kwargs)
  if packet in PRIMS: return PRIMS[packet](*args, **kwargs)
  if str(func).startswith("prims.") and getattr(aten, name, None) in OPS: return OPS[getattr(aten, name)](*args, **kwargs)  # as aten's
  return on_cpu(func, args, kwargs)

def keep(t, x):  # x into the FerriteTensor t, in place: its whole shape, computed now
  x = x if isinstance(x, FerriteTensor) else ferrite(x)
  t.v = Value(materialize(broadcast(x.v, tuple(t.shape))).realize(), View(tuple(t.shape)), t.dtype)
  return t

def on_cpu(func, args, kwargs):
  """func by PyTorch on the CPU, its FerriteTensors brought over, and its tensors sent back (in place: its first)."""
  there = lambda x: to_cpu(x) if isinstance(x, FerriteTensor) else x
  out = func(*tree_map(there, args), **tree_map(there, kwargs))
  if func.overloadpacket.__name__.endswith("_") and isinstance(args[0], FerriteTensor): return keep(args[0], out)
  return tree_map(lambda x: ferrite(x) if isinstance(x, torch.Tensor) and not isinstance(x, FerriteTensor) else x, out)
