# The compiler: tensor ops emitted as Tensix instructions straight from the ISA, no LLK. A tensor lives in DRAM as
# bf16 32x32 tiles, tile i in bank i % 8. Ops are lazy: a tree of elementwise ops and matmuls becomes one fused kernel,
# which fetches each input tile once, keeps every intermediate in L1 and writes only the result to DRAM. Each worker's BRISC
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

# The math thread's address modes, as (SrcA step, SrcB step) and (Dst step, fidelity step): slots 0-1 for elementwise
# ops, 4-7 for matmul, so one kernel can mix them without rewriting any between ops.
ADDR_MODES = {0: ((8, 8), (8, 0)), 1: ((0, 0), (0, 1)), 4: ((16, 0), (0, 0)), 5: ((48, 8), (0, 0)), 6: ((16, 8), (0, 0)), 7: ((16, 8), (0, 1))}
THREAD = [ins("SETC16", setc16_reg=r, setc16_value=v) for r, v in  # every thread: the basics, written since reset values are undocumented
          ({r: 0 for r in bh.THREAD_CFG.values()} | {bh.THREAD_CFG["SRCA_SET"]: bh.SRCA_SET_BY_ADDR, bh.ADDR_MOD_PACK: 4} |
           {bh.ADDR_MOD_BIAS + m: 0 for m in range(8)} |  # address mode 0 of the packer moves on 4 Dst rows a PACR
           {bh.ADDR_MOD_AB + m: a | b << 8 for m, ((a, b), _) in ADDR_MODES.items()} |
           {bh.ADDR_MOD_DST + m: d | f << bh.FIDELITY_INCR for m, (_, (d, f)) in ADDR_MODES.items()}).items()]

# Each thread has a 32-entry replay buffer: REPLAY with load records the next instructions, without it plays them
# back. The setup records the runs that repeat, so a tile costs BRISC a few words instead of dozens.
def replay(index, count): return ins("REPLAY", index=index, count=count)
def record(index, words): return [ins("REPLAY", load=1, index=index, count=len(words))] + words
MM, MUL, PACRS = 0, 16, 0  # what the setup records where: math thread a matmul phase at 0 and a mul phase at 16,
                           # pack thread 15 PACRs at 0

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
              ins("SETADCXX", cnt_set_mask=bh.PACK, x_end2=15)] + record(PACRS, [ins("PACR") for _ in range(15)])  # 16 datums a Dst row
def unpack(a, b):  # slot a into SrcA and slot b into SrcB, each bank then handed to the FPU
  return [ins("SETADC", cnt_set_mask=bh.UNPACK0, dimension_index=2, value=a), ins("SETADC", cnt_set_mask=bh.UNPACK1, dimension_index=2, value=b),
          ins("UNPACR", last=1, set_dat_valid=1, ovrd_thread_id=1), ins("UNPACR", last=1, set_dat_valid=1, ovrd_thread_id=1, unpack_block_selection=1)]
def math_begin(half, zero):  # wait for a free Dst half, aim the FPU at it, and zero it if the math accumulates
  return ([ins("SEMWAIT", stall_res=bh.STALL_ALL, sem_sel=SEM_DST, wait_sem_cond=2),
           ins("SETC16", setc16_reg=bh.THREAD_CFG["MATH_Offset"], setc16_value=DST_HALF * half)] +
          [ins("ZEROACC", clear_mode=1, use_32_bit_mode=1, where=4 * half + b) for b in range(4) if zero])  # 16-row blocks
MATH_END = [ins("STALLWAIT", stall_res=bh.STALL_SYNC, wait_res=bh.WAIT_FPU), ins("SEMPOST", sem_sel=SEM_DST)]
def pack(half, slot, done):  # a Dst half into an L1 slot, then free the half and post done
  return [ins("SEMWAIT", stall_res=bh.STALL_ALL, sem_sel=SEM_DST, wait_sem_cond=1),
          ins("SETADC", cnt_set_mask=bh.PACK, dimension_index=1, value=DST_HALF * half),  # channel 0 Y: Dst row
          ins("SETADC", cnt_set_mask=bh.PACK, dimension_index=1, channel_index=1, value=slot),  # channel 1 Y: L1 slot
          ins("STALLWAIT", stall_res=bh.STALL_PACK, wait_res=bh.WAIT_FPU),
          replay(PACRS, 15), ins("PACR", last=1),  # 16 PACRs, 4 Dst rows each through all four read ports
          ins("STALLWAIT", stall_res=bh.STALL_SYNC, wait_res=bh.WAIT_PACK), ins("SEMGET", sem_sel=SEM_DST), ins("SEMPOST", sem_sel=done)]
RELAY = [ins("SEMWAIT", stall_res=bh.STALL_ALL, sem_sel=SEM_PACKED, wait_sem_cond=1), ins("SEMGET", sem_sel=SEM_PACKED)]
UNPACKED_REPORT = [ins("STALLWAIT", stall_res=bh.STALL_THCON, wait_res=bh.WAIT_UNPACK)]  # then BRISC's SETGPR(UNPACKED)

def matmul_phase():
  """One fidelity phase of Dst += A @ B on a pair of tiles, A in SrcB and B in SrcA: MVMUL does
  Dst[8x16] += SrcB[8x16] @ SrcA[16x16]. Output face (i, j) adds A face (i, k) times B face (k, j) for k = 0, 1, in two
  MVMULs of 8 rows (h). In the order below SrcA steps by (16, -16, 16, -16, ...) faces and SrcB by (0, 8, 0, 8, ...)
  rows, so four address modes (4-7) cover it, and the last one wraps both back to row 0 (counters wrap at 64) and
  moves on to the next fidelity phase."""
  phase = []
  for i, k, h, j in [(i, k, h, j) for i in (0, 1) for k in (0, 1) for h in (0, 1) for j in (0, 1)]:
    last = (i, k, h, j) == (1, 1, 1, 1)
    phase.append(ins("MVMUL", dst=32 * i + 16 * j + 8 * h, addr_mode=4 + (0 if j == 0 else 1 if h == 0 else 3 if last else 2)))
  return phase
MUL_PHASE = [ins("SETRWC", bit_mask=0b0111)] + [ins("ELWMUL", addr_mode=int(i == 7)) for i in range(8)]  # mode 1 steps the phase
SETUP_MATH = record(MM, matmul_phase()) + record(MUL, MUL_PHASE)

def last(word): return word | 3 << 22  # the same ELW*/MVMUL with clear_dvalid: both Src banks go back to the unpackers
def matmul():  # one matmul step: four phases (SrcA gives 5 mantissa bits a phase, SrcB 7: four make the whole product)
  return [ins("SETRWC", bit_mask=0b1111)] + [replay(MM, 16)] * 3 + [replay(MM, 15), last(matmul_phase()[-1])]

def elementwise(op):
  """Math for one tile of SrcA op SrcB: 8 rows of 16 per instruction, address mode 0 steps 8 rows. ELWMUL always
  accumulates and multiplies a slice of the mantissas per fidelity phase: on a zeroed Dst, all 4 phases."""
  if op == "mul": return [ins("SETRWC", bit_mask=0b1111)] + [replay(MUL, 9)] * 3 + [replay(MUL, 8), last(MUL_PHASE[-1])]
  fpu = {"add": "ELWADD", "sub": "ELWSUB"}[op]
  return [ins("SETRWC", bit_mask=0b1111)] + [ins(fpu) for _ in range(7)] + [last(ins(fpu))]

def _array(name, words, t="unsigned"): return f"static const {t} {name}[] = {{{', '.join(f'{w:#x}' for w in words)}}};"
def _arrays(name, programs): return f"static const unsigned {name}[{len(programs)}][{len(programs[0])}] = {{{', '.join('{' + ', '.join(f'{w:#x}' for w in p) + '}' for p in programs)}}};"

def _kernel(arrays, body):
  """A BRISC kernel: backend config, the semaphores and every thread's setup, then body. body can push(t, p) a program
  to thread t, step() a matmul, and writeback(n, slot, base, tile) the n-th output (1-based)."""
  setgpr = ins("SETGPR", reg_index_16b=0)
  return compile_rv32(f"""
{_array("cfg_index", CONFIG)} {_array("cfg_value", CONFIG.values())} {_array("thread", THREAD)} {_array("relay", RELAY)}
{_array("setup_unpack", SETUP_UNPACK)} {_array("setup_math", SETUP_MATH)} {_array("setup_pack", SETUP_PACK)}
{_array("bank", [x | y << 6 for x, y in bh.DRAM])}
{_arrays("mm_unpack", [unpack(s, 2 + s) + UNPACKED_REPORT for s in (0, 1)])} {_array("mm", matmul())} {_array("math_end", MATH_END)}
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
static unsigned steps;  // matmul steps so far: their B and A tiles alternate between slots 0, 1 and 2, 3
static void step(unsigned a, unsigned a_tile, unsigned b, unsigned b_tile) {{  // A's tile goes to SrcB, B's to SrcA
  if (steps >= 2) reached({UNPACKED}, steps - 1);  // the slots' tiles from two steps ago have been unpacked
  get(b, b_tile, steps & 1); get(a, a_tile, 2 + (steps & 1)); noc_wait();
  push(0, mm_unpack[steps & 1]); steps++; setgpr({UNPACKED}, steps);
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
  push(1, thread); push(1, setup_math); push(2, thread); push(2, setup_pack);
{body}
}}
""", KERNEL)

def fused_kernel(expr, nt, order="op"):
  """A kernel for an expression: nested tuples, (op, x, y) for add, sub and mul, ("matmul", a, b, K tiles) where a and
  b are inputs, and plain numbers for inputs. nt: the result's width in tiles. args: [first tile, tiles, out, input 0,
  input 1, ...]. Slots 0-3 stream matmul tiles; then a frame per output tile holds the inputs, then one slot per op in
  post order (equal subexpressions are one op), the last op being the result. An elementwise op unpacks its operands
  from their slots, a matmul streams its K pairs of tiles through slots 0-3; either computes into a Dst half and packs
  into its own slot, so intermediates never leave L1. Two frames alternate between output tiles. Every intermediate
  is rounded to bf16 on the way to L1, exactly as if each op ran on its own.
  order is how BRISC queues the work: "op" (unpack, math, pack, op by op) is the only one safe on silicon. The others
  make a thread wait for real on ttsim, which otherwise runs every instruction the moment it can: "pack_first" queues
  each pack before its math, "thread" queues a whole tile for thread 0, then 1, then 2 (no matmuls). On silicon they
  would stall BRISC on a full FIFO before it pushes what unblocks it; ttsim's FIFOs are deep enough."""
  ops, inputs, fetched = [], 0, set()
  def visit(e):
    nonlocal inputs
    if isinstance(e, int): inputs = max(inputs, e + 1); fetched.add(e)
    elif e[0] == "matmul": inputs = max(inputs, e[1] + 1, e[2] + 1)
    else: visit(e[1]); visit(e[2])
    if not isinstance(e, int) and e not in ops: ops.append(e)
  visit(expr)
  assert len(ops) <= 15, "SEM_L1 counts at most 15 intermediates"
  frame = inputs + len(ops)
  def slot(f, e): return 4 + f * frame + (e if isinstance(e, int) else inputs + ops.index(e))
  arrays, bodies = [], []
  def words(ws): arrays.append(_array(f"w{len(arrays)}", ws)); return f"w{len(arrays) - 1}"
  for f in (0, 1):
    stmts, seen = [], 0  # (thread, C statement); seen: ops whose intermediates the unpack thread has waited for
    for j, op in enumerate(ops):
      half, pack_stmt = (f * len(ops) + j) % 2, (2, f"push(2, {words(pack((f * len(ops) + j) % 2, slot(f, op), SEM_PACKED if j == len(ops) - 1 else SEM_L1))});")
      if op[0] == "matmul":
        _, ia, ib, kt = op
        begin = words(math_begin(half, True))
        stmts += [(0, f"for (unsigned k = 0; k < {kt}; k++) {{ step(a[{3 + ia}], t / {nt} * {kt} + k, a[{3 + ib}], k * {nt} + t % {nt});"
                      f" if (k == 0) push(1, {begin}); push(1, mm); }} push(1, math_end);"), pack_stmt]
        continue
      name, x, y = op
      wait = []
      if not (isinstance(x, int) and isinstance(y, int)):  # reads an intermediate: wait for every op packed before it
        wait, seen = [ins("SEMWAIT", stall_res=bh.STALL_ALL, sem_sel=SEM_L1, wait_sem_cond=1), ins("SEMGET", sem_sel=SEM_L1)] * (j - seen), j
      steps = [(0, f"push(0, {words(wait + unpack(slot(f, x), slot(f, y)))});"),
               (1, f"push(1, {words(math_begin(half, name == 'mul') + elementwise(name) + MATH_END)});"), pack_stmt]
      stmts += steps[2:] + steps[:2] if order == "pack_first" else steps
    if order == "thread":
      assert all(op[0] != "matmul" for op in ops), "thread order cannot interleave a matmul's fetches"
      stmts.sort(key=lambda s: s[0])
    bodies.append(" ".join([f"get(a[{3 + i}], t, {slot(f, i)});" for i in sorted(fetched)] + ["noc_wait();"] + [c for _, c in stmts]))
  out = lambda f: slot(f, ops[-1])
  return _kernel("\n".join(arrays), f"""
  for (unsigned n = 0; n < a[1]; n++) {{
    unsigned t = a[0] + n;
    if (n & 1) {{ {bodies[1]} }}
    else {{ {bodies[0]} }}
    if (n) writeback(n, (n & 1) ? {out(0)} : {out(1)}, a[2], t - 1);  // the frame two tiles back is free again after this
  }}
  if (a[1]) writeback(a[1], (a[1] & 1) ? {out(0)} : {out(1)}, a[2], a[0] + a[1] - 1);""")

def to_bf16(x):
  bits = struct.unpack("<I", struct.pack("<f", x))[0]
  return (bits + 0x7FFF + (bits >> 16 & 1)) >> 16  # round to nearest even
def from_bf16(b): return struct.unpack("<f", struct.pack("<I", b << 16))[0]
def tile_index(r, c): return (r // 16 * 2 + c // 16) * 256 + r % 16 * 16 + c % 16  # where (r, c) of a tile sits in it

class Tensor:
  """rows x cols of bf16: in DRAM as 32x32 tiles in row-major tile order, tile i in bank i % 8; or, until realized, an
  op on other tensors. Realizing compiles the whole tree of unrealized ops under it into one kernel; a matmul's own
  operands are realized first, since each of its output tiles needs a whole row and column of them."""
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
      if t.op == "matmul":  # its operands are whole matrices, not tiles: realize them on their own first
        a, b = (s.realize() for s in t.srcs)
        return ("matmul", expr(a), expr(b), a.cols // 32)
      return (t.op, expr(t.srcs[0]), expr(t.srcs[1]))
    key = (expr(self), self.cols // 32)
    kernel = kernels.get(key) or kernels.setdefault(key, fused_kernel(*key))
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
    return Tensor(self.rt, self.rows, other.cols, "matmul", (self, other))

kernels = {}  # (expression, width in tiles) -> compiled kernel image
