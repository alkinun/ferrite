# The compiler: tensor ops emitted as Tensix instructions straight from the ISA, no LLK. A tensor lives in DRAM as
# bf16 32x32 tiles, tile i in bank i % 8. Elementwise ops are lazy: a chain of them becomes one fused kernel, which
# fetches each input tile once, keeps every intermediate in L1 and writes only the result to DRAM. Each worker's BRISC
# runs a generated kernel: it pulls operand tiles into L1 over the NoC and streams them through the three Tensix
# threads, unpack, math and pack running at the same time on different tiles, then writes results back.
#
# Tile layout: four 16x16 faces in row-major order (top-left, top-right, bottom-left, bottom-right), each face
# row-major. The ISA has no notion of a tile; this is the layout the FPU's 16x16 matrix multiply wants. An unpacker
# reads a whole tile into a SrcA/SrcB bank, so face f lands in rows 16f..16f+15, and Dst holds the result the same way.
import struct
from trickle import bh
from trickle.runtime import DATA, KERNEL, compile_rv32

TILE = 32 * 32 * 2  # bytes in a bf16 tile: one SrcA/SrcB bank (64 rows of 16), 64 Dst rows
SLOTS = DATA  # L1 tile slots: slot i at SLOTS + i * TILE, picked by the unpackers' Z and the packer's channel 1 Y counters
UNPACKED, PACKED = 63, 62  # thread 0 GPRs BRISC polls: matmul steps unpacked, output tiles packed
DST_HALF = 64  # Dst rows of one result in fp32; Dst holds two, so math fills one half while pack drains the other
SEM_DST, SEM_PACKED, SEM_L1 = 1 << 0, 1 << 1, 1 << 2  # results in Dst to pack, outputs packed but not reported, and
                                                       # intermediates packed into L1 but not yet waited for

def ins(name, **fields):
  opcode, layout = bh.TENSIX_OPS[name]
  word = opcode << 24
  for k, v in fields.items():
    first, width = layout[k]
    assert 0 <= v < 1 << width, f"{name} {k}={v} does not fit in {width} bits"
    word |= v << first
  return word

def config(**fields):  # backend config {word index: value}, every bit of a touched word set
  regs = {}
  for k, v in fields.items():
    index, first, width = bh.TENSIX_CFG_FIELDS[k]
    assert 0 <= v < 1 << width, f"{k}={v} does not fit in {width} bits"
    regs[index] = regs.get(index, 0) | v << first
  return regs

# Unpackers read one whole tile each (1024 datums) into a SrcA/SrcB bank; Z picks the slot (the Z stride is one tile).
# The FPU writes Dst in fp32. The packer reads 4 Dst rows per PACR and rounds to bf16 on the way to L1; its channel 1
# Y picks the slot (that Y stride is one tile). Base addresses are in 16-byte units, minus 1.
CONFIG = config(
  Dstacc=bh.BF16, Fp32_enabled=1, Zero_Flag_disabled_src=0,
  PCK0_Ystride=32, PCK0_ZW=0, PCK0_Ystride_1=TILE, PCK0_ZW_1=0, PCK0_Base_1=0, Read_32b_data=1, Edge_mask=0xFFFF,
  pack_reads_per_xy_plane=16, UNP0_add_dest_addr_cntr=0, UNP0_Ystride_1=0, UNP0_ZW_1=0, UNP1_ZW_1=0,
  SEC0_format=bh.BF16, SEC0_x_dim=1024, SEC0_y_dim=1, SEC0_z_dim=1, SEC0_out_format=bh.BF16, SEC0_uncompressed=1,
  SEC0_base=SLOTS // 16 - 1, SEC0_dest_addr=64, SEC0_tile_x_dim=1024, SEC0_offset=0,  # 64: SrcA rows are biased by 4
  SEC1_format=bh.BF16, SEC1_x_dim=1024, SEC1_y_dim=1, SEC1_z_dim=1, SEC1_out_format=bh.BF16, SEC1_uncompressed=1,
  SEC1_base=SLOTS // 16 - 1, L1_Dest_addr=SLOTS // 16 - 1, Disable_zero_compress=1, Pack_out_format=bh.BF16,
  Pack_in_format=bh.BF16, PCK_dest_offset=0)

def thread_setup(ab, dst):
  """SETC16s for a thread: the basics, and address modes 0-3 as (SrcA step, SrcB step) and (Dst step, fidelity step)."""
  regs = {r: 0 for r in bh.THREAD_CFG.values()} | {bh.THREAD_CFG["SRCA_SET"]: bh.SRCA_SET_BY_ADDR}
  for slot in range(4):
    regs[bh.ADDR_MOD_AB + slot] = ab[slot][0] | ab[slot][1] << 8
    regs[bh.ADDR_MOD_DST + slot] = dst[slot][0] | dst[slot][1] << bh.FIDELITY_INCR
    regs[bh.ADDR_MOD_BIAS + slot] = 0
  regs[bh.ADDR_MOD_PACK] = 4  # each PACR moves on 4 Dst rows
  return [ins("SETC16", setc16_reg=r, setc16_value=v) for r, v in regs.items()]

# The three threads. Thread 0 unpacks and reports, thread 1 does the math, thread 2 packs. Between them:
#   unpack -> math: SrcA/SrcB each have two banks; UNPACR waits for a free bank, the FPU for a full one, by hardware.
#   math -> pack: SEM_DST counts results in Dst waiting to be packed; math waits while both halves are full, pack while
#     there is none.
#   pack -> unpack, inside a fused kernel: an intermediate is packed into L1 and unpacked again by a later op. Pack
#     posts SEM_L1 for each; unpack takes one per op packed since it last looked, before it reads any intermediate.
#   pack -> BRISC: BRISC can only read thread 0's GPRs, so pack posts SEM_PACKED and thread 0 relays it into a GPR.
# BRISC queues work for a thread only behind what unblocks it: unpack, then math, then pack. Its path into a thread
# is a few entries deep, and a full FIFO stalls BRISC.
UNPACKERS = bh.UNPACK0 | bh.UNPACK1
SETUP_UNPACK = [ins("SEMINIT", sem_sel=SEM_DST, max_value=2), ins("SEMINIT", sem_sel=SEM_PACKED, max_value=15),
                ins("SEMINIT", sem_sel=SEM_L1, max_value=15), ins("SETADCXY", cnt_set_mask=UNPACKERS, bit_mask=0b1111),
                ins("SETADCZW", cnt_set_mask=UNPACKERS, bit_mask=0b1111), ins("SETADCXX", cnt_set_mask=UNPACKERS, x_end2=1023)]
SETUP_PACK = [ins("SETADCXY", cnt_set_mask=bh.PACK, bit_mask=0b1111), ins("SETADCZW", cnt_set_mask=bh.PACK, bit_mask=0b1111),
              ins("SETADCXX", cnt_set_mask=bh.PACK, x_end2=15)]  # 16 datums a Dst row
def unpack(a, b):  # slot a into SrcA and slot b into SrcB, each bank then handed to the FPU
  return [ins("SETADC", cnt_set_mask=bh.UNPACK0, dimension_index=2, value=a), ins("SETADC", cnt_set_mask=bh.UNPACK1, dimension_index=2, value=b),
          ins("UNPACR", last=1, set_dat_valid=1, ovrd_thread_id=1), ins("UNPACR", last=1, set_dat_valid=1, ovrd_thread_id=1, unpack_block_selection=1)]
def math_begin(half, zero):  # wait for a free Dst half, aim the FPU at it, and zero it if the math accumulates
  return ([ins("SEMWAIT", stall_res=bh.STALL_ALL, sem_sel=SEM_DST, wait_sem_cond=2),
           ins("SETC16", setc16_reg=bh.THREAD_CFG["MATH_Offset"], setc16_value=DST_HALF * half)] +
          [ins("ZEROACC", clear_mode=1, use_32_bit_mode=1, where=4 * half + b) for b in range(4) if zero])  # 16-row blocks
MATH_END = [ins("STALLWAIT", stall_res=bh.STALL_SYNC, wait_res=bh.WAIT_FPU), ins("SEMPOST", sem_sel=SEM_DST)]
def pack(half, slot, done):  # a Dst half into an L1 slot, then free the half and post done
  return ([ins("SEMWAIT", stall_res=bh.STALL_ALL, sem_sel=SEM_DST, wait_sem_cond=1),
           ins("SETADC", cnt_set_mask=bh.PACK, dimension_index=1, value=DST_HALF * half),  # channel 0 Y: Dst row
           ins("SETADC", cnt_set_mask=bh.PACK, dimension_index=1, channel_index=1, value=slot),  # channel 1 Y: L1 slot
           ins("STALLWAIT", stall_res=bh.STALL_PACK, wait_res=bh.WAIT_FPU)] +
          [ins("PACR", last=int(i == 15)) for i in range(16)] +  # all four Dst read ports: 4 rows a PACR
          [ins("STALLWAIT", stall_res=bh.STALL_SYNC, wait_res=bh.WAIT_PACK), ins("SEMGET", sem_sel=SEM_DST), ins("SEMPOST", sem_sel=done)])
RELAY = [ins("SEMWAIT", stall_res=bh.STALL_ALL, sem_sel=SEM_PACKED, wait_sem_cond=1), ins("SEMGET", sem_sel=SEM_PACKED)]
UNPACKED_REPORT = [ins("STALLWAIT", stall_res=bh.STALL_THCON, wait_res=bh.WAIT_UNPACK)]  # then BRISC's SETGPR(UNPACKED)

ELEMENTWISE_THREAD = thread_setup(ab=[(8, 8), (0, 0), (0, 0), (0, 0)], dst=[(8, 0), (0, 1), (0, 0), (0, 0)])
def elementwise(op):
  """Math for one tile of SrcA op SrcB: 8 rows of 16 per instruction, address mode 0 steps 8 rows. ELWMUL always
  accumulates and multiplies a slice of the mantissas per fidelity phase: on a zeroed Dst, all 4 phases, the last
  instruction of each in address mode 1, which steps the phase."""
  fpu = {"add": "ELWADD", "sub": "ELWSUB", "mul": "ELWMUL"}[op]
  if op != "mul": return [ins("SETRWC", bit_mask=0b1111)] + [ins(fpu, clear_dvalid=3 * (i == 7)) for i in range(8)]
  return [ins("SETRWC", bit_mask=0b1111)] + [w for phase in range(4) for w in [ins("SETRWC", bit_mask=0b0111)] +
          [ins(fpu, addr_mode=int(i == 7), clear_dvalid=3 * (phase == 3 and i == 7)) for i in range(8)]]

MATMUL_THREAD = thread_setup(ab=[(16, 0), (48, 8), (16, 8), (16, 8)], dst=[(0, 0), (0, 0), (0, 0), (0, 1)])
def matmul():
  """Math for one step of Dst += A @ B on a pair of tiles, A in SrcB and B in SrcA: MVMUL does
  Dst[8x16] += SrcB[8x16] @ SrcA[16x16]. Output face (i, j) adds A face (i, k) times B face (k, j) for k = 0, 1, in two
  MVMULs of 8 rows (h). In the order below SrcA steps by (16, -16, 16, -16, ...) faces and SrcB by (0, 8, 0, 8, ...)
  rows, so four address modes cover it, and the last one wraps both back to row 0 (counters wrap at 64) and moves on
  to the next fidelity phase."""
  step = [ins("SETRWC", bit_mask=0b1111)]
  for phase in range(4):  # SrcA gives 5 mantissa bits a phase, SrcB 7; four phases make the whole bf16 product
    for i, k, h, j in [(i, k, h, j) for i in (0, 1) for k in (0, 1) for h in (0, 1) for j in (0, 1)]:
      last = (i, k, h, j) == (1, 1, 1, 1)
      step.append(ins("MVMUL", dst=32 * i + 16 * j + 8 * h, addr_mode=0 if j == 0 else 1 if h == 0 else 3 if last else 2,
                      clear_dvalid=3 * (phase == 3 and last)))
  return step

def _array(name, words): return f"static const unsigned {name}[] = {{{', '.join(f'{w:#x}' for w in words)}}};"
def _arrays(name, programs, t="unsigned"): return f"static const {t} {name}[{len(programs)}][{len(programs[0])}] = {{{', '.join('{' + ', '.join(f'{w:#x}' for w in p) + '}' for p in programs)}}};"

def _kernel(thread, arrays, body):
  """A BRISC kernel: backend config, the semaphores and every thread's setup, then body. body can push(t, p) a program
  to thread t, wait for unpacking with UNPACKED, and writeback(n, slot, base, tile) the n-th output (1-based)."""
  setgpr = ins("SETGPR", reg_index_16b=0)
  return compile_rv32(f"""
{_array("cfg_index", CONFIG)} {_array("cfg_value", CONFIG.values())} {_array("thread", thread)} {_array("relay", RELAY)}
{_array("setup_unpack", SETUP_UNPACK)} {_array("setup_pack", SETUP_PACK)} {_array("bank", [x | y << 6 for x, y in bh.DRAM])}
{arrays}
#define PUSH(t) (*(volatile unsigned *)({bh.TENSIX_PUSH:#x} + 0x10000 * (t)))
#define GPR(r) (((volatile unsigned *){bh.TENSIX_GPR:#x})[r])
#define CFG ((volatile unsigned *){bh.TENSIX_CFG:#x})
#define SLOT(i) ({SLOTS:#x} + (i) * {TILE})
#define push(t, p) for (unsigned i = 0; i < sizeof p / 4; i++) PUSH(t) = p[i]
#define setgpr(r, v) PUSH(0) = {setgpr:#x} | 2 * (r) | ((v) & 0xFFFF) << 8
#define reached(r, v) while ((GPR(r) & 0xFFFF) < (v)) {{}}  // 16-bit counts: a launch handles under 65536 of each
// tile t of a tensor at DRAM address base: bank t % 8, row t / 8 of the bank
static void get(unsigned base, unsigned t, unsigned slot) {{ noc_read(bank[t % 8] & 63, bank[t % 8] >> 6, base + t / 8 * {TILE}, SLOT(slot), {TILE}); }}
static void writeback(unsigned n, unsigned slot, unsigned base, unsigned t) {{  // once the n-th output is packed, send it home
  push(0, relay); setgpr({PACKED}, n);
  reached({PACKED}, n);
  noc_write(SLOT(slot), bank[t % 8] & 63, bank[t % 8] >> 6, base + t / 8 * {TILE}, {TILE}); noc_wait();
}}
void kmain(volatile unsigned *a) {{
  for (unsigned i = 0; i < sizeof cfg_index / 4; i++) CFG[cfg_index[i]] = cfg_value[i];
  GPR({UNPACKED}) = GPR({PACKED}) = 0;
  asm volatile("fence" ::: "memory");  // config and GPR writes are out; reading back from each region lands after them
  (void)CFG[cfg_index[0]]; (void)GPR({PACKED});
  asm volatile("fence" ::: "memory");  // and has answered, so no instruction pushed below can see stale config
  push(0, thread); push(0, setup_unpack); setgpr({PACKED}, 0xFFFF);
  reached({PACKED}, 0xFFFF);  // semaphores are initialised before the other threads can touch them
  setgpr({PACKED}, 0);
  push(1, thread); push(2, thread); push(2, setup_pack);
{body}
}}
""", KERNEL)

def fused_kernel(expr, order="op"):
  """A kernel for an elementwise expression: nested (op, x, y) tuples with leaves as input numbers. args: [first tile,
  tiles, out, input 0, input 1, ...]. A tile's slots: inputs, then one per op in post order (equal subexpressions are
  one op); the last op is the result. Each op unpacks its operands from their slots, computes into a Dst half, and
  packs into its slot, so intermediates never leave L1. Two frames of slots alternate between output tiles. Every
  intermediate is rounded to bf16 on the way to L1, exactly as if each op ran on its own.
  order is how BRISC queues the work: "op" (unpack, math, pack, op by op) is the only one safe on silicon. The others
  make a thread wait for real on ttsim, which otherwise runs every instruction the moment it can: "pack_first" queues
  each pack before its math, "thread" queues a whole tile for thread 0, then 1, then 2. On silicon they would stall
  BRISC on a full FIFO before it pushes what unblocks it; ttsim's FIFOs are deep enough."""
  ops, inputs = [], 0
  def visit(e):
    nonlocal inputs
    if isinstance(e, int): inputs = max(inputs, e + 1)
    elif e not in ops: visit(e[1]); visit(e[2]); ops.append(e)
  visit(expr)
  assert len(ops) <= 15, "SEM_L1 counts at most 15 intermediates"
  frame = inputs + len(ops)
  def slot(f, e): return f * frame + (e if isinstance(e, int) else inputs + ops.index(e))
  programs = []
  for f in (0, 1):
    pushes, seen = [], 0  # ops whose intermediates the unpack thread has waited for
    for j, (op, x, y) in enumerate(ops):
      wait = []
      if not (isinstance(x, int) and isinstance(y, int)):  # reads an intermediate: wait for every op packed before it
        wait, seen = [ins("SEMWAIT", stall_res=bh.STALL_ALL, sem_sel=SEM_L1, wait_sem_cond=1), ins("SEMGET", sem_sel=SEM_L1)] * (j - seen), j
      half = (f * len(ops) + j) % 2
      steps = [(0, wait + unpack(slot(f, x), slot(f, y))), (1, math_begin(half, op == "mul") + elementwise(op) + MATH_END),
               (2, pack(half, slot(f, ops[j]), SEM_PACKED if j == len(ops) - 1 else SEM_L1))]
      if order == "pack_first": steps = steps[2:] + steps[:2]
      pushes += [(t, w) for t, words in steps for w in words]
    programs.append(sorted(pushes, key=lambda p: p[0]) if order == "thread" else pushes)
  out = inputs + len(ops) - 1
  return _kernel(ELEMENTWISE_THREAD, _arrays("words", [[w for _, w in p] for p in programs]) + _arrays("threads", [[t for t, _ in p] for p in programs], "unsigned char"), f"""
  for (unsigned n = 0; n < a[1]; n++) {{
    unsigned t = a[0] + n, f = n & 1;
    for (unsigned i = 0; i < {inputs}; i++) get(a[3 + i], t, f * {frame} + i);
    noc_wait();
    for (unsigned i = 0; i < sizeof words[f] / 4; i++) PUSH(threads[f][i]) = words[f][i];
    if (n) writeback(n, (f ^ 1) * {frame} + {out}, a[2], t - 1);  // the frame two tiles back is free again after this
  }}
  if (a[1]) writeback(a[1], ((a[1] - 1) & 1) * {frame} + {out}, a[2], a[0] + a[1] - 1);""")

def matmul_kernel():
  """args: [first output tile, tiles, A, B, C, K tiles, N tiles]. Each output tile is K steps: A's tile (i, k) to SrcB
  and B's tile (k, j) to SrcA, alternating between two slot pairs, MVMULs into the same Dst half, then one pack."""
  return _kernel(MATMUL_THREAD, _arrays("unpack", [unpack(s, 2 + s) + UNPACKED_REPORT for s in (0, 1)]) +
                 _arrays("math_begin", [math_begin(h, True) for h in (0, 1)]) + _array("step", matmul()) +
                 _array("math_end", MATH_END) + _arrays("pack", [pack(h, 4 + h, SEM_PACKED) for h in (0, 1)]), """
  for (unsigned n = 0, s = 0; n < a[1]; n++) {
    unsigned t = a[0] + n;
    for (unsigned k = 0; k < a[5]; k++, s++) {
      if (s >= 2) reached(%d, s - 1);  // the slots' tiles from two steps ago have been unpacked
      get(a[3], k * a[6] + t %% a[6], s & 1); get(a[2], t / a[6] * a[5] + k, 2 + (s & 1)); noc_wait();
      push(0, unpack[s & 1]); setgpr(%d, s + 1);
      if (k == 0) push(1, math_begin[n & 1]);
      push(1, step);
      if (k == a[5] - 1) push(1, math_end);
    }
    push(2, pack[n & 1]);
    if (n) writeback(n, 4 + ((n - 1) & 1), a[4], t - 1);  // before the next pack reuses its slot
  }
  if (a[1]) writeback(a[1], 4 + ((a[1] - 1) & 1), a[4], a[0] + a[1] - 1);""" % (UNPACKED, UNPACKED))

def to_bf16(x):
  bits = struct.unpack("<I", struct.pack("<f", x))[0]
  return (bits + 0x7FFF + (bits >> 16 & 1)) >> 16  # round to nearest even
def from_bf16(b): return struct.unpack("<f", struct.pack("<I", b << 16))[0]
def tile_index(r, c): return (r // 16 * 2 + c // 16) * 256 + r % 16 * 16 + c % 16  # where (r, c) of a tile sits in it

class Tensor:
  """rows x cols of bf16: in DRAM as 32x32 tiles in row-major tile order, tile i in bank i % 8; or, until realized, an
  elementwise op on other tensors. Realizing compiles the whole tree of unrealized ops under it into one kernel."""
  def __init__(self, rt, rows, cols, op=None, srcs=()):
    assert rows % 32 == 0 and cols % 32 == 0, "tensors are whole tiles"
    self.rt, self.rows, self.cols, self.tiles, self.op, self.srcs = rt, rows, cols, rows * cols // 1024, op, srcs
    self.addr = None if op else rt.dram_alloc(-(-self.tiles // len(bh.DRAM)) * TILE)

  @classmethod
  def of(cls, rt, rows):
    t = cls(rt, len(rows), len(rows[0]))
    for i in range(t.tiles):
      r0, c0, tile = i // (t.cols // 32) * 32, i % (t.cols // 32) * 32, [0] * 1024
      for r in range(32):
        for c in range(32): tile[tile_index(r, c)] = to_bf16(rows[r0 + r][c0 + c])
      t.rt.chip.write(*bh.DRAM[i % 8], t.addr + i // 8 * TILE, struct.pack("<1024H", *tile))
    return t

  def tolist(self):
    self.realize()
    out = [[0.0] * self.cols for _ in range(self.rows)]
    for i in range(self.tiles):
      r0, c0 = i // (self.cols // 32) * 32, i % (self.cols // 32) * 32
      tile = struct.unpack("<1024H", self.rt.chip.read(*bh.DRAM[i % 8], self.addr + i // 8 * TILE, TILE))
      for r in range(32):
        for c in range(32): out[r0 + r][c0 + c] = from_bf16(tile[tile_index(r, c)])
    return out

  def realize(self):
    if self.op is None: return self
    inputs = []
    def expr(t):  # the tree of unrealized ops under t, realized tensors as numbers into inputs
      if t.op is None:
        if all(t is not u for u in inputs): inputs.append(t)
        return next(i for i, u in enumerate(inputs) if u is t)
      return (t.op, expr(t.srcs[0]), expr(t.srcs[1]))
    e = expr(self)
    kernel = kernels.get(e) or kernels.setdefault(e, fused_kernel(e))
    self.addr, self.op, self.srcs = self.rt.dram_alloc(-(-self.tiles // len(bh.DRAM)) * TILE), None, ()
    return self._run(kernel, self.addr, *[t.addr for t in inputs])

  def _run(self, kernel, *args):  # this tensor's tiles split evenly over the workers
    cores = self.rt.cores
    share = {xy: (i * self.tiles // len(cores), (i + 1) * self.tiles // len(cores)) for i, xy in enumerate(cores)}
    self.rt.launch(kernel, lambda x, y: (share[x, y][0], share[x, y][1] - share[x, y][0], *args))
    self.rt.wait()
    return self

  def _elementwise(self, op, other):
    assert (self.rows, self.cols) == (other.rows, other.cols), "shapes must match"
    return Tensor(self.rt, self.rows, self.cols, op, (self, other))
  def __add__(self, other): return self._elementwise("add", other)
  def __sub__(self, other): return self._elementwise("sub", other)
  def __mul__(self, other): return self._elementwise("mul", other)

  def __matmul__(self, other):
    assert self.cols == other.rows, f"({self.rows}, {self.cols}) @ ({other.rows}, {other.cols})"
    a, b, out = self.realize(), other.realize(), Tensor(self.rt, self.rows, other.cols)
    kernel = kernels.get("matmul") or kernels.setdefault("matmul", matmul_kernel())
    return out._run(kernel, a.addr, b.addr, out.addr, self.cols // 32, other.cols // 32)

kernels = {}  # expression or "matmul" -> compiled kernel image
