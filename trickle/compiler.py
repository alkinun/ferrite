# The compiler: tensor ops emitted as Tensix instructions straight from the ISA, no LLK. A tensor lives in DRAM as
# bf16 32x32 tiles, tile i in bank i % 8. Ops are lazy: a tree of elementwise ops, matmuls, SFPU ops (relu, exp,
# reciprocal, rsqrt) and row reductions becomes one fused kernel, which fetches each input tile once, keeps every
# intermediate in L1 and writes only the result to DRAM. A kernel is four programs for each worker tile: BRISC fetches operand tiles into L1 over the
# NoC and sends results home, and each TRISC feeds one Tensix thread its instructions (unpack, math, pack), the three
# running at the same time on different tiles.
#
# Tile layout: four 16x16 faces in row-major order (top-left, top-right, bottom-left, bottom-right), each face
# row-major, so datum (r, c) is at (r // 16 * 2 + c // 16) * 256 + r % 16 * 16 + c % 16. The ISA has no notion of a
# tile; this is the layout the FPU's 16x16 matrix multiply wants. An unpacker reads a whole tile into a SrcA/SrcB bank,
# so face f lands in rows 16f..16f+15, and Dst holds the result the same way.
import array, math, struct
from trickle import bh
from trickle.runtime import DATA, SYNC, compile_kernel

TILE = 32 * 32 * 2  # bytes in a bf16 tile: one SrcA/SrcB bank (64 rows of 16), 64 Dst rows
SLOTS = DATA  # L1 tile slots: slot i at SLOTS + i * TILE, picked by the unpackers' Z and the packer's channel 1 Y counters
DST_HALF = 64  # Dst rows of one result in fp32; Dst holds two, so math fills one half while pack drains the other
SEM_DST, SEM_PACKED, SEM_L1 = 1 << 0, 1 << 1, 1 << 2  # results in Dst to pack, outputs packed but not reported, and
                                                       # intermediates packed into L1 but not yet waited for
SEM_UNPACKED = 1 << 5  # frames (and matmul steps) unpacked but not reported
ACCS = ((256, 1 << 3), (320, 1 << 4))  # row reductions accumulate here in Dst (scratch 128 rows on), each with a
                                       # semaphore that is 1 while it is free: taken by the first step, given back by the pack

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
CONFIG_FIELDS = dict(
  Dstacc=bh.BF16, Fp32_enabled=1, Zero_Flag_disabled_src=0,
  PCK0_Ystride=32, PCK0_ZW=0, PCK0_Ystride_1=TILE, PCK0_ZW_1=0, PCK0_Base_1=0, Read_32b_data=1, Edge_mask=0xFFFF,
  pack_reads_per_xy_plane=16, UNP0_add_dest_addr_cntr=0, UNP0_Ystride_1=0, UNP0_ZW_1=0, UNP1_ZW_1=0,
  SEC0_format=bh.BF16, SEC0_x_dim=1024, SEC0_y_dim=1, SEC0_z_dim=1, SEC0_out_format=bh.BF16, SEC0_uncompressed=1,
  SEC0_base=SLOTS // 16 - 1, SEC0_dest_addr=64, SEC0_tile_x_dim=1024, SEC0_offset=0,  # 64: SrcA rows are biased by 4
  SEC1_format=bh.BF16, SEC1_x_dim=1024, SEC1_y_dim=1, SEC1_z_dim=1, SEC1_out_format=bh.BF16, SEC1_uncompressed=1,
  SEC1_base=SLOTS // 16 - 1, L1_Dest_addr=SLOTS // 16 - 1, Disable_zero_compress=1, Pack_out_format=bh.BF16,
  Pack_in_format=bh.BF16, PCK_dest_offset=0)
CONFIG = config(**CONFIG_FIELDS)
TRANSPOSING = config(**CONFIG_FIELDS, SEC0_Haloize_mode=1)  # unpacker 0 transposes each 16x16 face into SrcA

# The math thread's address modes, as (SrcA step, SrcB step) and (Dst step, fidelity step): 0-1 for elementwise ops,
# 2-3 for SFPU loads and stores (the store steps Dst by 2), 4-7 for matmul, so ops mix with no rewrites between them.
ADDR_MODES = {0: ((8, 8), (8, 0)), 1: ((0, 0), (0, 1)), 2: ((0, 0), (0, 0)), 3: ((0, 0), (2, 0)),
              4: ((16, 0), (0, 0)), 5: ((48, 8), (0, 0)), 6: ((16, 8), (0, 0)), 7: ((16, 8), (0, 1))}
THREAD = [ins("SETC16", setc16_reg=r, setc16_value=v) for r, v in  # every thread: the basics, written since reset values are undocumented
          ({r: 0 for r in bh.THREAD_CFG.values()} | {bh.THREAD_CFG["SRCA_SET"]: bh.SRCA_SET_BY_ADDR, bh.ADDR_MOD_PACK: 4} |
           {bh.ADDR_MOD_BIAS + m: 0 for m in range(8)} |  # address mode 0 of the packer moves on 4 Dst rows a PACR
           {bh.ADDR_MOD_AB + m: a | b << 8 for m, ((a, b), _) in ADDR_MODES.items()} |
           {bh.ADDR_MOD_DST + m: d | f << bh.FIDELITY_INCR for m, (_, (d, f)) in ADDR_MODES.items()}).items()]

# Each thread has a 32-entry replay buffer: REPLAY with load records the next instructions, without it plays them
# back, so a run that repeats costs its TRISC one word. The pack thread's always holds 15 PACRs; the math thread's is
# shared out per kernel between the sequences that kernel uses, those replayed most first.
def replay(index, count): return ins("REPLAY", index=index, count=count)
def record(index, words): return [ins("REPLAY", load=1, index=index, count=len(words))] + words
PACRS = 0
class Loop(list):  # words a TRISC pushes `times` times over, in a loop rather than written out
  def __init__(self, times, words): super().__init__(words); self.times = times

# The three threads, each fed by its own TRISC: thread 0 unpacks, thread 1 does the math, thread 2 packs. Between them:
#   unpack -> math: SrcA/SrcB each have two banks; UNPACR waits for a free bank, the FPU for a full one, by hardware.
#   math -> pack: SEM_DST counts results in Dst waiting to be packed; math waits while both halves are full, pack while
#     there is none.
#   pack -> unpack, inside a fused kernel: an intermediate is packed into L1 and unpacked again by a later op. Pack
#     posts SEM_L1 for each; unpack takes one per op packed since it last looked, before it reads any intermediate.
#   unpack, pack -> BRISC: thread 0 posts SEM_UNPACKED once it has unpacked a frame's tiles (or a matmul step's pair),
#     the pack thread SEM_PACKED once it has packed a result; their TRISCs count those into SYNC, where BRISC looks.
UNPACKERS = bh.UNPACK0 | bh.UNPACK1
SETUP_UNPACK = [ins("SEMINIT", sem_sel=SEM_DST, max_value=2), ins("SEMINIT", sem_sel=SEM_PACKED, max_value=15),
                ins("SEMINIT", sem_sel=SEM_L1, max_value=15), ins("SEMINIT", sem_sel=SEM_UNPACKED, max_value=15),
                *[ins("SEMINIT", sem_sel=sem, init_value=1, max_value=1) for _, sem in ACCS],
                ins("SETADCXY", cnt_set_mask=UNPACKERS, bit_mask=0b1111),
                ins("SETADCZW", cnt_set_mask=UNPACKERS, bit_mask=0b1111), ins("SETADCXX", cnt_set_mask=UNPACKERS, x_end2=1023)]
SETUP_PACK = [ins("SETADCXY", cnt_set_mask=bh.PACK, bit_mask=0b1111), ins("SETADCZW", cnt_set_mask=bh.PACK, bit_mask=0b1111),
              ins("SETADCXX", cnt_set_mask=bh.PACK, x_end2=15)] + record(PACRS, [ins("PACR") for _ in range(15)])  # 16 datums a Dst row
def unpack(a, b):  # slot a into SrcA and slot b into SrcB, each bank then handed to the FPU
  return [ins("SETADC", cnt_set_mask=bh.UNPACK0, dimension_index=2, value=a), ins("SETADC", cnt_set_mask=bh.UNPACK1, dimension_index=2, value=b),
          ins("UNPACR", last=1, set_dat_valid=1, ovrd_thread_id=1), ins("UNPACR", last=1, set_dat_valid=1, ovrd_thread_id=1, unpack_block_selection=1)]
REPORT = [ins("STALLWAIT", stall_res=bh.STALL_SYNC, wait_res=bh.WAIT_UNPACK), ins("SEMPOST", sem_sel=SEM_UNPACKED)]
def math_begin(half, zero):  # wait for a free Dst half, aim the FPU at it, and zero it if the math accumulates
  return ([ins("SEMWAIT", stall_res=bh.STALL_ALL, sem_sel=SEM_DST, wait_sem_cond=2),
           ins("SETC16", setc16_reg=bh.THREAD_CFG["MATH_Offset"], setc16_value=DST_HALF * half)] +
          [ins("ZEROACC", clear_mode=1, use_32_bit_mode=1, where=4 * half + b) for b in range(4) if zero])  # 16-row blocks
MATH_END = [ins("STALLWAIT", stall_res=bh.STALL_SYNC, wait_res=bh.WAIT_FPU | bh.WAIT_SFPU), ins("SEMPOST", sem_sel=SEM_DST)]
def pack(row, slot, *done):  # 64 Dst rows from row into an L1 slot, then free them and post done
  return [ins("SEMWAIT", stall_res=bh.STALL_ALL, sem_sel=SEM_DST, wait_sem_cond=1),
          ins("SETADC", cnt_set_mask=bh.PACK, dimension_index=1, value=row),  # channel 0 Y: Dst row
          ins("SETADC", cnt_set_mask=bh.PACK, dimension_index=1, channel_index=1, value=slot),  # channel 1 Y: L1 slot
          ins("STALLWAIT", stall_res=bh.STALL_PACK, wait_res=bh.WAIT_FPU | bh.WAIT_SFPU),
          replay(PACRS, 15), ins("PACR", last=1),  # 16 PACRs, 4 Dst rows each through all four read ports
          ins("STALLWAIT", stall_res=bh.STALL_SYNC, wait_res=bh.WAIT_PACK), ins("SEMGET", sem_sel=SEM_DST)] + [ins("SEMPOST", sem_sel=d) for d in done]
TAKE_L1 = [ins("SEMWAIT", stall_res=bh.STALL_ALL, sem_sel=SEM_L1, wait_sem_cond=1), ins("SEMGET", sem_sel=SEM_L1)]

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

def to_bf16(x):
  bits = struct.unpack("<I", struct.pack("<f", x))[0]
  return (bits + 0x7FFF + (bits >> 16 & 1)) >> 16  # round to nearest even
def from_bf16(b): return struct.unpack("<f", struct.pack("<I", b << 16))[0]

# SFPU programs: a setup, then a body run 32 times; each pass loads 32 datums of the Dst half (4 rows x 8 even or odd
# columns) into LReg 0 and stores the result back, the store stepping Dst by 2 (address modes 2 and 3; fp32 Dst, mod0 3).
def sfp_load(): return ins("SFPLOAD", vd=0, mod0=3, addr_mode=2)
def sfp_store(r): return ins("SFPSTORE", vd=r, mod0=3, addr_mode=3)
def sfp_bits(r, bits): return [ins("SFPLOADI", vd=r, mod0=8, imm16=bits >> 16), ins("SFPLOADI", vd=r, mod0=10, imm16=bits & 0xFFFF)]
def sfp_const(r, x): return sfp_bits(r, struct.unpack("<I", struct.pack("<f", x))[0])  # a full fp32 constant into LReg r
def mad(d, a, b, c, negate_a=0): return ins("SFPMAD", vd=d, va=a, vb=b, vc=c, mod1=negate_a)  # d = (-)a * b + c
def mul(d, a, b): return ins("SFPMUL", vd=d, va=a, vb=b, vc=9)
ZERO, ONE = 9, 10  # constant LRegs
SFPU = {
  # max(x, 0): SFPSWAP's min/max mode leaves the smaller value in vd and the larger in vc
  "relu": ([], [sfp_load(), ins("SFPMOV", vd=1, vc=ZERO), ins("SFPSWAP", vd=1, vc=0, mod1=1), sfp_store(0)]),
  # e^x = (e^(x/4096))^4096: a degree-4 Taylor series at |x/4096| < 0.026, then 12 squarings. Every step is a
  # multiply-add, so it needs no exponent tricks; accuracy is ~1e-4, plenty for bf16. x is first raised to at least
  # -104 (e^-104 is below the smallest normal float, so still 0), or the series itself overflows for very negative x;
  # large positive x overflows to inf in the squarings as it should
  "exp": (sfp_const(1, 2**-12) + sfp_const(2, 1 / 24) + sfp_const(3, 1 / 6) + sfp_const(4, 0.5),
          [sfp_load(), ins("SFPLOADI", vd=7, mod0=0, imm16=to_bf16(-104.0)), ins("SFPSWAP", vd=7, vc=0, mod1=1),
           mul(5, 0, 1), mad(6, 5, 2, 3), mad(6, 6, 5, 4), mad(6, 6, 5, ONE), mad(6, 6, 5, ONE)] +
          [mul(6, 6, 6) for _ in range(12)] + [sfp_store(6)]),
  # 1/x: SFPARECIP's 7-bit estimate, then two Newton steps y += y * (1 - x * y). Only where x is a normal number:
  # 0 and denormals (estimate inf) and inf (estimate 0) would turn into 0 * inf = NaN, so those lanes keep the estimate
  "recip": ([], [sfp_load(), ins("SFPARECIP", vd=1, vc=0), ins("SFPEXEXP", vd=3, vc=0),  # LReg 3: unbiased exponent
                 ins("SFPENCC", mod1=10, imm2=3),  # lanes enabled by flags, all set
                 ins("SFPIADD", vd=4, vc=3, mod1=1, imm12=-128 & 0xFFF),  # keep lanes with exponent < 128 ...
                 ins("SFPIADD", vd=4, vc=3, mod1=9, imm12=126),  # ... and >= -126
                 mad(2, 0, 1, ONE, negate_a=1), mad(1, 1, 2, 1), mad(2, 0, 1, ONE, negate_a=1), mad(1, 1, 2, 1),
                 ins("SFPENCC", mod1=10, imm2=2), sfp_store(1)]),  # every lane again
  # 1/sqrt(x) for x > 0: the bit trick (0x5F3759DF - bits / 2, within 3.5%), then two Newton steps y *= 1.5 - x/2 y^2
  "rsqrt": (sfp_const(2, struct.unpack("<f", struct.pack("<I", 0x5F3759DF))[0]) + sfp_const(3, 1.5) + sfp_const(4, 0.5),
            [sfp_load(), ins("SFPSHFT", vd=1, vc=0, mod1=5, imm12=-1 & 0xFFF),  # LReg 1 = bits of x >> 1
             ins("SFPIADD", vd=1, vc=2, mod1=6), mul(5, 0, 4)] +  # LReg 1 = magic - LReg 1 as integers; LReg 5 = x/2
            [w for _ in range(2) for w in (mul(6, 1, 1), mad(6, 5, 6, 3, negate_a=1), mul(1, 1, 6))] + [sfp_store(1)]),
}
# Reductions along rows: x + 0 lands in Dst (the first column tile in the result's half, the rest 128 rows further
# on, where the SFPU folds them in). Then each row is folded across its 32 columns and the result written to all of
# them, so the tile broadcasts along rows as is.
def fold(kind, d, t):  # LReg d = d op t (for max, t is left with the min)
  return ins("SFPSWAP", vd=t, vc=d, mod1=1) if kind == "rowmax" else mad(d, t, ONE, d)
COMBINE = {k: [ins("SFPLOAD", vd=0, mod0=3, addr_mode=2), ins("SFPLOAD", vd=1, mod0=3, addr_mode=2, imm10=128), fold(k, 0, 1), sfp_store(0)]
           for k in ("rowsum", "rowmax")}
def across_rows(kind):
  """Fold every tile row across its 32 columns. Row r of a tile is Dst row (r // 16) * 32 + r % 16 (columns 0-15)
  and 16 rows on (16-31); an SFPLOAD takes 4 such rows' even or odd columns, one row per 8-lane group. Folding the
  four loads leaves 8 lanes a row, and rotating the group by 1, 2 and 4 lanes and folding each time leaves the
  row's result in every lane, which goes back to all four places."""
  rot = lambda d, s: ins("SFPSHFT2", vd=d, vc=s, mod1=3)
  p = [ins("STALLWAIT", stall_res=bh.STALL_SFPU, wait_res=bh.WAIT_FPU), ins("SETRWC", bit_mask=0b0100)]
  for base in (0, 4, 8, 12, 32, 36, 40, 44):
    at = [base, base + 2, base + 16, base + 18]  # even and odd columns, left and right face
    p += [ins("SFPLOAD", vd=r, mod0=3, addr_mode=2, imm10=a) for r, a in enumerate(at)]
    p += [fold(kind, 0, 1), fold(kind, 2, 3), fold(kind, 0, 2)]
    p += [rot(1, 0), fold(kind, 0, 1), rot(1, 0), rot(1, 1), fold(kind, 0, 1), rot(1, 0), rot(1, 1), rot(1, 1), rot(1, 1), fold(kind, 0, 1)]
    p += [ins("SFPSTORE", vd=0, mod0=3, addr_mode=2, imm10=a) for a in at]
  return p
def mask_columns(kind, n, d):
  """Columns n and on of the tile at Dst row d to the reduction's identity (0, or -inf for max), so the padding past a
  tensor's last column does not count. An SFPSTORE writes 4 rows of 8 even or odd columns, lane l at column 2 (l % 8)
  (+ 1 if odd); LReg 15 holds 2l, so LReg 7 = LReg 15 & 14 is twice the lane's place in its row, and each store goes to
  the lanes whose column is n or more."""
  fill = ZERO if kind == "rowsum" else 5
  p = [ins("STALLWAIT", stall_res=bh.STALL_SFPU, wait_res=bh.WAIT_FPU), ins("SETRWC", bit_mask=0b0100)]
  p += ([] if kind == "rowsum" else sfp_const(5, -math.inf)) + sfp_bits(7, 14) + [ins("SFPAND", vd=7, vc=15)]
  for base, right, odd in [(b, r, o) for b in (0, 4, 8, 12, 32, 36, 40, 44) for r in (0, 1) for o in (0, 1)]:
    t = max(0, min(8, -(-(n - 16 * right - odd) // 2)))  # the first lane (of 8) at column n or more
    if t < 8: p += [ins("SFPENCC", mod1=10, imm2=3), ins("SFPIADD", vd=6, vc=7, mod1=9, imm12=-2 * t & 0xFFF),  # lanes t+
                    ins("SFPSTORE", vd=fill, mod0=3, addr_mode=2, imm10=d + base + 16 * right + 2 * odd)]  # right faces: 16 rows on
  return p + [ins("SFPENCC", mod1=10, imm2=2)]
def mask_rows(m):
  """Rows m and on of the tile at the math thread's Dst half to 0, past a result's last row. A store covers 4 rows, lane l
  in row l // 8 of them, and LReg 15 holds 2l, so LReg 15 - 16k >= 0 picks the rows from k on."""
  p = [ins("STALLWAIT", stall_res=bh.STALL_SFPU, wait_res=bh.WAIT_FPU), ins("SETRWC", bit_mask=0b0100)]
  for r in range(0, 32, 4):  # tile rows r..r+3: Dst rows (r // 16) * 32 + r % 16 on, left face, and 16 more, right face
    k = max(0, m - r)
    if k < 4: p += [w for d in (0, 2, 16, 18) for w in (ins("SFPENCC", mod1=10, imm2=3), ins("SFPIADD", vd=6, vc=15, mod1=9, imm12=-16 * k & 0xFFF),
                                                     ins("SFPSTORE", vd=ZERO, mod0=3, addr_mode=2, imm10=r // 16 * 32 + r % 16 + d))]
  return p + [ins("SFPENCC", mod1=10, imm2=2)]
MUL_PHASE = [ins("SETRWC", bit_mask=0b0111)] + [ins("ELWMUL", addr_mode=int(i == 7)) for i in range(8)]  # mode 1 steps the phase
MATH_RUNS = {"mm": matmul_phase(), "mul": MUL_PHASE} | {u: body for u, (_, body) in SFPU.items()} | COMBINE

def allocate(names):
  """{run: replay index} for the math thread: SFPU bodies (32 replays a tile, longest first), then matmul and mul
  phases (4). What does not fit is pushed word by word."""
  slots, at = {}, 0
  for name in sorted(names, key=lambda n: (n in ("mm", "mul"), -len(MATH_RUNS[n]))):
    if at + len(MATH_RUNS[name]) <= 32: slots[name], at = at, at + len(MATH_RUNS[name])
  return slots
def run(slots, name, count=None):  # the first count words of a run: one REPLAY if it was recorded, else the words
  words = MATH_RUNS[name][:count]
  return [replay(slots[name], len(words))] if name in slots else words
def runs(slots, name, times): return run(slots, name) * times if name in slots else [Loop(times, MATH_RUNS[name])]
def setup_math(slots): return [ins("SFPENCC", mod1=10, imm2=2)] + [w for name, at in slots.items() for w in record(at, MATH_RUNS[name])]

def last(word): return word | 3 << 22  # the same ELW*/MVMUL with clear_dvalid: both Src banks go back to the unpackers
def matmul(slots):  # one matmul step: four phases (SrcA gives 5 mantissa bits a phase, SrcB 7: four make the whole product)
  return [ins("SETRWC", bit_mask=0b1111)] + run(slots, "mm") * 3 + run(slots, "mm", 15) + [last(matmul_phase()[-1])]

def elementwise(op, slots, dst=0):
  """Math for one tile of SrcA op SrcB at Dst row dst: 8 rows of 16 per instruction, address mode 0 steps 8 rows.
  ELWMUL always accumulates and multiplies a slice of the mantissas per fidelity phase: on a zeroed Dst, 4 phases."""
  if op == "mul": return [ins("SETRWC", bit_mask=0b1111)] + run(slots, "mul") * 3 + run(slots, "mul", 8) + [last(MUL_PHASE[-1])]
  fpu = {"add": "ELWADD", "sub": "ELWSUB"}[op]
  return [ins("SETRWC", bit_mask=0b1111)] + [ins(fpu, dst=dst) for _ in range(7)] + [last(ins(fpu, dst=dst))]

def at(row): return ins("SETC16", setc16_reg=bh.THREAD_CFG["MATH_Offset"], setc16_value=row)  # aim FPU and SFPU at row
def reduce_step(kind, slots, acc, first, edge=0):
  """One column tile of a row reduction, which has SrcA = the tile and SrcB = zeros: the first lands in the
  accumulator once it is free; later ones land in scratch and the SFPU folds them in. edge: how many of the tile's
  columns are the tensor's, if it is the row's last and cut short."""
  row, sem = ACCS[acc]
  if first: return ([ins("SEMWAIT", stall_res=bh.STALL_ALL, sem_sel=sem, wait_sem_cond=1), ins("SEMGET", sem_sel=sem), at(row)] +
                    elementwise("add", slots) + (mask_columns(kind, edge, 0) if edge else []))
  return ([at(row)] + elementwise("add", slots, dst=128) + (mask_columns(kind, edge, 128) if edge else []) +
          [ins("STALLWAIT", stall_res=bh.STALL_SFPU, wait_res=bh.WAIT_FPU), ins("SETRWC", bit_mask=0b0100)] + runs(slots, kind, 32))
def reduce_finish(kind, slots, acc, posts):  # fold each row across and run the SFPU ops on the result (then MATH_END)
  return [ins("SEMWAIT", stall_res=bh.STALL_ALL, sem_sel=SEM_DST, wait_sem_cond=2), at(ACCS[acc][0])] + across_rows(kind) + sfpu(posts, slots)

def sfpu(posts, slots):  # the SFPU ops, in order, over the Dst half the FPU just wrote
  return [w for u in posts for w in [ins("STALLWAIT", stall_res=bh.STALL_SFPU, wait_res=bh.WAIT_FPU), ins("SETRWC", bit_mask=0b0100)] +
          SFPU[u][0] + runs(slots, u, 32)]

def _array(name, words): return f"static const unsigned {name}[] = {{{', '.join(f'{w:#x}' for w in words)}}};"
def halves(programs): return f"if (h) {{ {tt(programs[1])} }} else {{ {tt(programs[0])} }}"  # one program per Dst half, by h

def tt(words):
  """C for a TRISC to push words to its thread as .ttinsn instructions: each word rotated left by 2 and written into
  the code, so the TRISC stalls while the thread's FIFO is full. A Loop becomes a C loop."""
  c, run = [], []
  for w in list(words) + [None]:
    if isinstance(w, int):
      assert w < 0xC0000000, f"{w:#x} is no Tensix instruction"
      run.append(f"{(w << 2 | w >> 30) & 0xFFFFFFFF:#x}")
      continue
    if run: c.append(f'asm volatile(".word {", ".join(run)}" ::: "memory");')
    run = []
    if w is not None: c.append(f'_Pragma("clang loop unroll(disable)") for (unsigned r_ = 0; r_ < {w.times}; r_++) {{ {tt(w)} }}')
  return " ".join(c)

ZERO_SLOT, FRAMES = 4, 5  # slots 0-3 stream matmul tiles, 4 holds a zero tile, then two frames, then row values

# SYNC (zeroed by the firmware for each launch) holds what BRISC and the TRISCs tell each other, each word written by
# one of them: FETCHED, how many times BRISC has fetched tiles into L1 (a frame, or a matmul step's pair); UNPACKED, how
# many of those thread 0 has unpacked, from TRISC0; READY, set by TRISC0 once the backend config and semaphores are
# set up; PACKED, how many results the pack thread has packed, from TRISC2, with where each goes in RING (four
# entries [slot, tile]); SENT, how many of those BRISC has sent home.
SYNC_WORDS = f"""
enum {{ FETCHED, UNPACKED, READY, PACKED, SENT, RING = 8 }};
#define SYNC ((volatile unsigned *){SYNC:#x})
static unsigned peek(unsigned i) {{ asm volatile("fence" ::: "memory"); return SYNC[i]; }}  // fresh: the L0 cache sees no other core's writes
"""
TRISC = SYNC_WORDS + f"""
#define SEM(s) (((volatile unsigned *){bh.TENSIX_SEM:#x})[__builtin_ctz(s)])  // read: its value; write 0: post, 1: take one
// Setup goes through stores, not .ttinsn: fused .ttinsn instructions can corrupt what REPLAY records
static void push(const unsigned *p, unsigned n) {{ for (unsigned i = 0; i < n; i++) *(volatile unsigned *){bh.TENSIX_PUSH:#x} = p[i]; }}
#define PUSH(p) push(p, sizeof p / 4)
"""

def _brisc(zero, total, body):  # moves tiles: inputs into L1 ahead of thread 0, results home behind the pack thread
  return SYNC_WORDS + f"""
#define SLOT(i) ({SLOTS:#x} + (i) * {TILE})
{_array("bank", [x | y << 6 for x, y in bh.DRAM])}
static unsigned fetched, events, frame_done[2], stream_done[2], steps, sent, out;
// tile t of a tensor at DRAM address base: bank t % 8, row t / 8 of the bank
static void get(unsigned base, unsigned t, unsigned slot) {{ noc_read(bank[t % 8] & 63, bank[t % 8] >> 6, base + t / 8 * {TILE}, SLOT(slot), {TILE}); }}
// the same with its two off-diagonal faces swapped, half of transposing it (unpacker 0 does the rest)
static void flipped(unsigned base, unsigned t, unsigned slot) {{
  for (unsigned f = 0; f < 4; f++) noc_read(bank[t % 8] & 63, bank[t % 8] >> 6, base + t / 8 * {TILE} + f * {TILE // 4}, SLOT(slot) + (f == 1 ? 2 : f == 2 ? 1 : f) * {TILE // 4}, {TILE // 4});
}}
static void ready(void) {{ noc_wait(); SYNC[FETCHED] = ++fetched; }}
static void send(void) {{  // the results packed so far go home, in the order they were packed
  for (; sent < peek(PACKED); SYNC[SENT] = ++sent) {{
    unsigned slot = peek(RING + sent % 4 * 2), t = peek(RING + sent % 4 * 2 + 1);
    noc_write(SLOT(slot), bank[t % 8] & 63, bank[t % 8] >> 6, out + t / 8 * {TILE}, {TILE}); noc_wait();
  }}
}}
// A slot is fetched into again once thread 0 has unpacked what was there: the frame two tiles back, the matmul pair two
// steps back. Thread 0 reports each frame and step in turn; events counts them the same way. Results go home meanwhile.
static void reached(unsigned n) {{ while (peek(UNPACKED) < n) send(); }}
static void step(unsigned a, unsigned a_tile, unsigned b, unsigned b_tile) {{  // A's tile goes to SrcB, B's to SrcA
  reached(stream_done[steps & 1]);
  get(b, b_tile, steps & 1); get(a, a_tile, 2 + (steps & 1)); ready();
  stream_done[steps++ & 1] = ++events;
}}
void kmain(volatile unsigned *a) {{
  out = a[2];
  {f"for (unsigned i = 0; i < {TILE // 4}; i++) ((volatile unsigned *)SLOT({ZERO_SLOT}))[i] = 0; l1_settle(SLOT({ZERO_SLOT}));" if zero else ""}
  unsigned tiles = 0;
{body}
  while (sent < {total}) send();
}}"""

def _unpacker(body, cfg=CONFIG):  # TRISC0: the backend config and semaphores, then unpacking, each tile once BRISC has fetched it
  return TRISC + f"""
#define CFG ((volatile unsigned *){bh.TENSIX_CFG:#x})
{_array("cfg_index", cfg)} {_array("cfg_value", cfg.values())} {_array("thread", THREAD)} {_array("setup", SETUP_UNPACK)}
static unsigned needed, relayed, posted, drained, steps;
static void relay(void) {{ for (; SEM({SEM_UNPACKED}); SYNC[UNPACKED] = ++relayed) SEM({SEM_UNPACKED}) = 1; }}  // thread 0's reports, for BRISC
static void fetched(void) {{ needed++; do relay(); while (peek(FETCHED) < needed); }}  // until BRISC has fetched the next tiles
static void drain(void) {{ for (; drained < posted; drained++) {{ {tt(TAKE_L1)} }} }}  // until everything packed so far has landed
void kmain(volatile unsigned *a) {{
  for (unsigned i = 0; i < sizeof cfg_index / 4; i++) CFG[cfg_index[i]] = cfg_value[i];
  asm volatile("fence" ::: "memory");  // the config writes are out; a read from the same block is processed after them
  (void)CFG[cfg_index[0]];
  asm volatile("fence" ::: "memory");  // and has answered, so no instruction pushed below can see them stale
  PUSH(thread); PUSH(setup);
  tensix_idle();  // the semaphores are set before the other threads can touch them
  SYNC[READY] = 1;
  unsigned tiles = 0;
{body}
}}"""

def _math(slots, body):  # TRISC1
  return TRISC + f"""
{_array("thread", THREAD)} {_array("setup", setup_math(slots))}
void kmain(volatile unsigned *a) {{
  while (!peek(READY)) {{}}
  PUSH(thread); PUSH(setup);
  unsigned tiles = 0, h = 0;  // h: the Dst half the next op gets
{body}
}}"""

def _packer(body):  # TRISC2
  return TRISC + f"""
{_array("thread", THREAD)} {_array("setup", SETUP_PACK)}
static unsigned outputs, relayed, used[2];
static void relay(void) {{ for (; SEM({SEM_PACKED}); SYNC[PACKED] = ++relayed) SEM({SEM_PACKED}) = 1; }}  // packed results, for BRISC
// The next result is packed into slot (output slot s: one per frame, or one for all) and goes home as tile t. Once the
// last result packed there has gone home, tell BRISC in RING; at most two wait to go home, one per output slot.
static void output(unsigned s, unsigned slot, unsigned t) {{
  while (peek(SENT) < used[s]) relay();
  SYNC[RING + outputs % 4 * 2] = slot; SYNC[RING + outputs % 4 * 2 + 1] = t;
  used[s] = ++outputs;
}}
void kmain(volatile unsigned *a) {{
  while (!peek(READY)) {{}}
  PUSH(thread); PUSH(setup);
  unsigned tiles = 0, h = 0;  // h: the Dst half the next op gets
{body}
  while (relayed < outputs) relay();
}}"""

def fused_kernel(expr, shapes):
  """A kernel for an expression: nested tuples, (op, x, y) for add, sub and mul, (op, x) for relu, exp, recip, rsqrt,
  rowsum and rowmax, ("matmul", a, b) where a and b are inputs, and plain numbers for inputs. shapes: each input's
  (rows, columns), laid out as Tensor says; a dimension of 1 repeats across the others (broadcasting).
  args: [first unit, units, out, then for each input its DRAM address, tile stride and tile offset (Tensor's)]. A unit
  is a tile of the result, or, if the expression reduces rows, a whole row of tiles, so the reduction can stay on chip.

  Every op computes into Dst and is packed into an L1 slot: elementwise ops from their operands' slots, a matmul from
  its K pairs of tiles streamed through slots 0-3. A unary op runs on the SFPU over Dst: in place, on the result of the
  op below it if nothing else reads that, or else on its operand plus a zero tile. A row reduction folds each column
  tile of its operand into an accumulator in Dst, then across the row (see across_rows), and packs one tile.
  A row is done in passes: a pass is a sweep over its column tiles, and a reduction's result can only be read in the
  pass after the one that swept it. Each pass recomputes what it needs (softmax computes exp twice) rather than keep
  whole rows in L1. Values one tile wide (reductions, ops on them, narrow inputs) are worked out once per row, in the
  first tile of the pass that needs them, into slots of their own.
  Two frames of slots alternate between tiles: inputs, then one slot per op. Every intermediate is rounded to bf16 on
  its way to L1, exactly as if each op ran on its own; in-place SFPU ops see the fp32 result.
  The four programs run the same loops over units, passes and tiles, each doing its own part of every tile."""
  uses, seen_exprs = {}, set()
  def count(e):  # how many ops read each subexpression
    if isinstance(e, int) or e in seen_exprs: return
    seen_exprs.add(e)
    for c in (() if e[0] == "matmul" else e[1:]):
      uses[c] = uses.get(c, 0) + 1
      count(c)
  count(expr)
  tiles = [(-(-r // 32), -(-c // 32)) for r, c in shapes]
  ops, refs = [], {}  # ops: dicts with kind, x, y, kt, posts, shape, width; refs: expr -> input number or ("op", j)
  size = lambda r: shapes[r] if isinstance(r, int) else ops[r[1]]["shape"]  # (rows, columns)
  width = lambda r: tiles[r][1] if isinstance(r, int) else ops[r[1]]["width"]  # in tiles
  def visit(e):
    if isinstance(e, int): return e
    if e in refs: return refs[e]
    if e[0] == "matmul":
      op = dict(kind="matmul", x=e[1], y=e[2], kt=tiles[e[1]][1], shape=(shapes[e[1]][0], shapes[e[2]][1]), width=tiles[e[2]][1])
    elif e[0] in COMBINE:
      x = visit(e[1])
      op = dict(kind=e[0], x=x, shape=(size(x)[0], 1), width=1)
    elif e[0] in SFPU:
      x = visit(e[1])
      if isinstance(x, tuple) and uses[e[1]] == 1:  # in place, on the result of the op below
        ops[x[1]]["posts"].append(e[0])
        refs[e] = x
        return x
      op = dict(kind="add", x=x, y="zero", shape=size(x), width=width(x), posts=[e[0]])  # operand + 0 into Dst, then the SFPU
    else:
      x, y = visit(e[1]), visit(e[2])
      op = dict(kind=e[0], x=x, y=y, shape=tuple(map(max, size(x), size(y))), width=max(width(x), width(y)))
    ops.append({"posts": [], **op})
    refs[e] = ("op", len(ops) - 1)
    return refs[e]
  root = visit(expr)[1]
  operands = lambda op: [r for r in (op.get("x"), op.get("y")) if r not in (None, "zero")] if op["kind"] != "matmul" else []
  # w: the row's width in tiles, what ops sweep (a matmul's A can be wider: its columns are K, streamed, not swept)
  inputs, w = len(shapes), max([op["width"] for op in ops] + [width(r) for op in ops for r in operands(op)])
  reduced = lambda r: isinstance(r, tuple) and ops[r[1]]["kind"] in COMBINE
  wide = lambda r: r != "zero" and width(r) == w and not reduced(r)  # varies along the row; otherwise one tile a row
  rowwise = any(op["kind"] in COMBINE for op in ops)
  assert len(ops) <= 15, "SEM_L1 counts at most 15 intermediates"
  # passes: a wide op runs in the pass after every reduction it depends on has finished; a narrow one is ready then
  ready, sweep = {}, {}  # ready: first pass a value can be read in; sweep: pass a reduction sweeps its operand in
  for j, op in enumerate(ops):
    deps = [ready[r[1]] if isinstance(r, tuple) else 0 for r in operands(op)]
    if op["kind"] in COMBINE: sweep[j] = max(deps, default=0); ready[j] = sweep[j] + 1
    else: ready[j] = max(deps, default=0)
  last = sweep[root] if root in sweep else ready[root]
  slots = allocate({"mm" for op in ops if op["kind"] == "matmul"} | {"mul" for op in ops if op["kind"] == "mul"} |
                   {u for op in ops for u in op["posts"]} | {op["kind"] for op in ops if op["kind"] in COMBINE})
  frame = inputs + len(ops)
  def slot(f, r):  # an input or wide op lives in the frame, a narrow op in its own slot after both frames
    if r == "zero": return ZERO_SLOT
    if isinstance(r, int): return FRAMES + f * frame + r
    return FRAMES + (f * frame + inputs + r[1] if wide(r) else 2 * frame + r[1])
  def tile(i):  # which tile of input i goes with this unit's row i_ and column j
    rows, cols = tiles[i]
    return f"{'i_' if rows > 1 else 0} * a[{4 + 3 * i}] + {'j' if cols > 1 else 0} + a[{5 + 3 * i}]"
  B, U, M, P = range(4)  # the programs: BRISC moves tiles; TRISC 0, 1, 2 feed the unpack, math and pack threads
  rows, cols = size(("op", root))
  def math_end(j):  # the math thread's last words for op j: the root's padding in Dst goes to 0 first, in a tile at the edge
    if j != root: return tt(MATH_END)
    edge = [f"if ({f'j == {w - 1}' if wide(('op', root)) else 1}) {{ {tt(mask_columns('rowsum', cols % 32, 0))} }}"] * (cols > 1 and cols % 32 > 0)
    edge += [f"if (i_ == {-(-rows // 32) - 1}) {{ {tt(mask_rows(rows % 32))} }}"] * (rows > 1 and rows % 32 > 0)
    return " ".join(edge + [tt(MATH_END)])
  def output(j, f):  # for the pack thread: where op j's result goes, if it is the root
    me = ("op", j)
    return f"output({f if wide(me) else 0}, {slot(f, me)}, {f'i_ * {w} + j' if wide(me) else 'i_'}); " if j == root else ""
  passes = []
  for p in range(last + 1):
    reductions = [j for j in sweep if sweep[j] == p]
    sinks = [ops[j]["x"] for j in reductions] + ([("op", root)] if p == last and wide(("op", root)) else [])
    swept = []  # the wide ops this pass computes, operands first
    def need(r):
      if isinstance(r, tuple) and wide(r) and r[1] not in swept:
        for o in operands(ops[r[1]]): need(o)
        swept.append(r[1])
    for r in sinks: need(r)
    once = [j for j, op in enumerate(ops) if not wide(("op", j)) and op["kind"] not in COMBINE and ready[j] == p]
    bodies = []
    for f in (0, 1):
      c = [[], [], [], []]  # each program's C for this tile
      fetch = sorted({r for j in once + swept + reductions for r in operands(ops[j]) if isinstance(r, int)})
      c[B] += [f"reached(frame_done[{f}]);"] + [f"get(a[{3 + 3 * r}], {tile(r)}, {slot(f, r)});" for r in fetch] + ["ready();"]
      c[U].append("fetched();")
      for j in once + swept:
        op, me = ops[j], ("op", j)
        cond, posted = "if (k == 0) " if j in once else "", "" if j == root else " posted++;"
        c[P].append(f"{cond}{{ {output(j, f)}{halves([pack(DST_HALF * h, slot(f, me), SEM_PACKED if j == root else SEM_L1) for h in (0, 1)])} h ^= 1; }}")
        if op["kind"] == "matmul":
          kt, bw = op["kt"], tiles[op["y"]][1]  # B's width in tiles
          x, y = 3 + 3 * op["x"], 3 + 3 * op["y"]  # A's and B's address, stride, offset in a
          c[B].append(f"{cond}for (unsigned m = 0; m < {kt}; m++) step(a[{x}], i_ * a[{x + 1}] + m + a[{x + 2}], a[{y}], m * a[{y + 1}] + {'j' if bw > 1 else 0} + a[{y + 2}]);")
          c[U].append(f"{cond}{{ for (unsigned m = 0; m < {kt}; m++) {{ fetched(); "
                      f"if (steps++ & 1) {{ {tt(unpack(1, 3) + REPORT)} }} else {{ {tt(unpack(0, 2) + REPORT)} }} }}{posted} }}")
          c[M].append(f"{cond}{{ {halves([math_begin(h, True) for h in (0, 1)])} for (unsigned m = 0; m < {kt}; m++) {{ {tt(matmul(slots))} }} "
                      f"{tt(sfpu(op['posts'], slots))} {math_end(j)} h ^= 1; }}")
          continue
        drain = "drain(); " if any(isinstance(r, tuple) for r in operands(op)) else ""
        c[U].append(f"{cond}{{ {drain}{tt(unpack(slot(f, op['x']), slot(f, op['y'])))}{posted} }}")
        c[M].append(f"{cond}{{ {halves([math_begin(h, op['kind'] == 'mul') + elementwise(op['kind'], slots) + sfpu(op['posts'], slots) for h in (0, 1)])} {math_end(j)} h ^= 1; }}")
      for acc, j in enumerate(reductions):
        assert acc < len(ACCS), "two reductions a pass"
        op, me, x, kind = ops[j], ("op", j), ops[j]["x"], ops[j]["kind"]
        drain = "drain(); " if isinstance(x, tuple) else ""
        edge = size(x)[1] % 32 if size(x)[1] > 1 else 0  # the operand's columns in its last tile, if that is cut short
        end = "j == j1 - 1" if wide(x) else "k == 0"  # the row's last tile (a narrow operand is one tile a row)
        def step(first):  # the last tile's padding must not count
          plain, cut = tt(reduce_step(kind, slots, acc, first)), tt(reduce_step(kind, slots, acc, first, edge))
          return f"if ({end}) {{ {cut} }} else {{ {plain} }}" if edge else plain
        finish, posted = f'{tt(reduce_finish(kind, slots, acc, op["posts"]))} {math_end(j)}', "" if j == root else " posted++;"
        pk = tt(pack(ACCS[acc][0], slot(f, me), SEM_PACKED if j == root else SEM_L1, ACCS[acc][1]))
        if wide(x):  # swept over the row
          c[U].append(f"{drain}{tt(unpack(slot(f, x), ZERO_SLOT))} if ({end}) {{{posted} }}")
          c[M].append(f"if (k == 0) {{ {step(True)} }} else {{ {step(False)} }} if ({end}) {{ {finish} }}")
        else:  # reduced in the pass's first tile
          c[U].append(f"if ({end}) {{ {drain}{tt(unpack(slot(f, x), ZERO_SLOT))}{posted} }}")
          c[M].append(f"if ({end}) {{ {step(True)} {finish} }}")
        c[P].append(f"if ({end}) {{ {output(j, f)}{pk} }}")
      c[U].append(tt(REPORT))
      c[B].append(f"frame_done[{f}] = ++events;")
      bodies.append(c)
    passes.append((bool(swept or reductions), bodies))
  def program(c):  # the loops all four programs share, around each one's part of a tile
    sweeps = "\n".join(f"""    for (unsigned j = j0, k = 0; j < {'j1' if full else 'j0 + 1'}; j++, k++) {{
      if (tiles++ & 1) {{ {" ".join(b[1][c])} }} else {{ {" ".join(b[0][c])} }}
    }}""" for full, b in passes)
    return f"""  for (unsigned u = 0; u < a[1]; u++) {{
    unsigned unit = a[0] + u, i_ = {'unit' if rowwise else f'unit / {w}'}, j0 = {'0' if rowwise else f'unit % {w}'}, j1 = {w if rowwise else 'j0 + 1'};
{sweeps}
  }}"""
  total = f"a[1] * {w if rowwise and wide(('op', root)) else 1}"  # results this worker sends home
  zero = rowwise or any(op.get("y") == "zero" for op in ops)
  return compile_kernel(_brisc(zero, total, program(B)), _unpacker(program(U)), _math(slots, program(M)), _packer(program(P)))


def gather_kernel():
  """Rows of a tensor into a new one, args [first unit, units, out, the tensor's address, tile stride, tile offset, the
  result's tile columns, rows, their indices...]. BRISC alone: for each result tile, each row's two 16-datum runs (left
  and right face) come from wherever that row sits in the tensor's tiles. A NoC read from DRAM must keep its address's
  low 6 bits, so each run comes in as the 64-byte block holding it and is copied into place from there."""
  return compile_kernel(f"""
#define SLOT {SLOTS:#x}
#define BLOCK {SLOTS + TILE:#x}
{_array("bank", [x | y << 6 for x, y in bh.DRAM])}
void kmain(volatile unsigned *a) {{
  for (unsigned u = 0; u < a[1]; u++) {{
    unsigned t = a[0] + u, i = t / a[6], j = t % a[6];
    for (unsigned r = 0; r < 32; r++) {{
      unsigned row = a[7] == 1 ? 0 : i * 32 + r;  // one row fills the tile
      for (unsigned face = r / 16 * 2; face < r / 16 * 2 + 2; face++) {{
        volatile unsigned *to = (volatile unsigned *)(SLOT + face * 512 + r % 16 * 32);
        if (row >= a[7]) {{ for (unsigned k = 0; k < 8; k++) to[k] = 0; continue; }}
        unsigned id = a[8 + row], from = id / 32 * a[4] + j + a[5], at = (id % 32 / 16 * 2 + face % 2) * 512 + id % 16 * 32;
        noc_read(bank[from % 8] & 63, bank[from % 8] >> 6, a[3] + from / 8 * {TILE} + at / 64 * 64, BLOCK, 64);
        noc_wait();
        for (unsigned k = 0; k < 8; k++) to[k] = ((volatile unsigned *)(BLOCK + at % 64))[k];
      }}
    }}
    noc_write(SLOT, bank[t % 8] & 63, bank[t % 8] >> 6, a[2] + t / 8 * {TILE}, {TILE});
    noc_wait();
  }}
}}""")

def transpose_kernel(rows, cols):
  """x.T for x of (rows, cols), args [first unit, units, out, x's address, tile stride, tile offset]: result tile (i, j)
  is x's tile (j, i) transposed. BRISC fetches that tile with its off-diagonal faces swapped, and unpacker 0 transposes
  each face on its way into SrcA (Haloize_mode), so x + 0 through the FPU lands the transpose in Dst. Haloize_mode is
  backend config, so it holds for every SrcA load in the kernel: a transpose is a kernel of its own."""
  ct, frame = -(-rows // 32), lambda f: (FRAMES + 2 * f, FRAMES + 2 * f + 1)  # the result's tile columns; x's slot and the result's
  def loop(*per_frame):  # the same loop over result tiles t for all four programs, frame f alternating
    return f"""  for (unsigned u = 0; u < a[1]; u++) {{
    unsigned t = a[0] + u;
    if (u & 1) {{ {per_frame[1]} }} else {{ {per_frame[0]} }}
  }}"""
  return compile_kernel(
    _brisc(True, "a[1]", loop(*[f"reached(frame_done[{f}]); flipped(a[3], t % {ct} * a[4] + t / {ct} + a[5], {frame(f)[0]}); ready(); frame_done[{f}] = ++events;" for f in (0, 1)])),
    _unpacker(loop(*[f"fetched(); {tt(unpack(frame(f)[0], ZERO_SLOT) + REPORT)}" for f in (0, 1)]), TRANSPOSING),
    _math({}, loop(*[halves([math_begin(h, False) + elementwise("add", {}) + MATH_END for h in (0, 1)]) + " h ^= 1;"] * 2)),  # padding stays 0
    _packer(loop(*[f"output({f}, {frame(f)[1]}, t); " + halves([pack(DST_HALF * h, frame(f)[1], SEM_PACKED) for h in (0, 1)]) + " h ^= 1;" for f in (0, 1)])))

class Tensor:
  """rows x cols of bf16: in DRAM as 32x32 tiles in row-major tile order, tile i in bank i % 8; or, until realized, an
  op on other tensors. Realizing compiles the whole tree of unrealized ops under it into one kernel; a matmul's own
  operands are realized first, since each of its output tiles needs a whole row and column of them.
  Any shape: the last tile row and column are padded, with zeros, except that a dimension of 1 fills its tile with
  copies (a row vector's tile holds the row 32 times), so broadcasting reuses the tile as it is. Every kernel leaves
  its result that way too, so a matmul can sum over a padded K.
  A view (x[r0:r1, c0:c1] on tile boundaries) shares its owner's DRAM: its tile (i, j) is tile i * stride + j + offset."""
  def __init__(self, rt, rows, cols, op=None, srcs=()):
    self.rt, self.rows, self.cols, self.op, self.srcs = rt, rows, cols, op, srcs
    self.tiles, self.stride, self.offset, self.owner = -(-rows // 32) * -(-cols // 32), -(-cols // 32), 0, None
    self.addr = None if op else rt.dram_alloc(-(-self.tiles // len(bh.DRAM)) * TILE)

  def __del__(self):  # its DRAM goes back once nothing refers to it (launches are waited for, so none still reads it)
    if self.addr is not None and self.owner is None: self.rt.dram_free(self.addr)

  def _where(self, i, j):  # tile (i, j) in DRAM: (x, y, address)
    t = i * self.stride + j + self.offset
    return (*bh.DRAM[t % 8], self.addr + t // 8 * TILE)

  @classmethod
  def full(cls, rt, rows, cols, value): return cls.of(rt, [[value] * cols for _ in range(rows)])

  @classmethod
  def of(cls, rt, rows):
    """From a list of rows of numbers, rounded to bf16 (to nearest even)."""
    bits = array.array("I", array.array("f", [v for row in rows for v in row]).tobytes())
    return cls.frombytes(rt, len(rows), len(rows[0]), array.array("H", [(b + 0x7FFF + (b >> 16 & 1)) >> 16 for b in bits]))

  @classmethod
  def frombytes(cls, rt, rows, cols, data):
    """From bf16 in row-major order (2 bytes a value, little-endian). Runs of 16 values are copied into tile order as
    they are, nothing per value, which is what makes a model's weights quick to load."""
    t, ct, data, pad = cls(rt, rows, cols), -(-cols // 32), memoryview(data).cast("B"), bytes(32)
    assert len(data) == 2 * rows * cols, f"{len(data)} bytes for {rows} x {cols}"
    for i in range(t.tiles):
      r0, c0, tile = i // ct * 32, i % ct * 32, []
      for face in range(4):  # 16 rows of 16, face by face (the tile layout, at the top)
        c = c0 + face % 2 * 16
        for r in range(r0 + face // 2 * 16, r0 + face // 2 * 16 + 16):
          r = 0 if rows == 1 else r  # a dimension of 1 fills the tile; the rest pads with 0
          run = b"" if r >= rows else bytes(data[2 * r * cols:2 * r * cols + 2]) * 16 if cols == 1 else bytes(data[2 * (r * cols + c):2 * (r * cols + min(c + 16, cols))])
          tile.append(run + pad[len(run):])
      t.rt.chip.write(*t._where(i // ct, i % ct), b"".join(tile))
    return t

  def tobytes(self):
    """bf16 in row-major order, as frombytes takes it."""
    self.realize()
    out, ct = bytearray(2 * self.rows * self.cols), -(-self.cols // 32)
    for i in range(self.tiles):
      r0, c0, tile = i // ct * 32, i % ct * 32, self.rt.chip.read(*self._where(i // ct, i % ct), TILE)
      for face in range(4):
        c, n = c0 + face % 2 * 16, min(16, self.cols - c0 - face % 2 * 16)
        for k, r in enumerate(range(r0 + face // 2 * 16, min(r0 + face // 2 * 16 + 16, self.rows))):
          if n > 0: out[2 * (r * self.cols + c):2 * (r * self.cols + c + n)] = tile[face * 512 + k * 32:face * 512 + k * 32 + 2 * n]
    return bytes(out)

  def tolist(self):
    values = array.array("f", array.array("I", [b << 16 for b in array.array("H", self.tobytes())]).tobytes()).tolist()
    return [values[r * self.cols:(r + 1) * self.cols] for r in range(self.rows)]

  def gather(self, ids):
    """Rows ids of this tensor, as a tensor: an embedding lookup. A data-movement kernel (see gather_kernel)."""
    assert all(0 <= i < self.rows for i in ids)
    out = Tensor(self.rt, len(ids), self.cols)
    return out._run(_cached("gather", gather_kernel), out.addr, self.realize().addr, self.stride, self.offset, -(-self.cols // 32), len(ids), *ids)

  def __getitem__(self, key):
    """x[r0:r1, c0:c1], a view of whole tiles: starts on tile boundaries, and ends too unless at the tensor's end. No
    copy; kernels read it in place."""
    (r0, r1, _), (c0, c1, _) = (k.indices(n) for k, n in zip(key, (self.rows, self.cols)))
    assert r0 % 32 == c0 % 32 == 0 and (r1 % 32 == 0 or r1 == self.rows) and (c1 % 32 == 0 or c1 == self.cols), "views are whole tiles"
    assert (r1 - r0 > 1 or self.rows == 1) and (c1 - c0 > 1 or self.cols == 1), "a dimension of 1 must fill its tile"
    v = Tensor(self.rt, r1 - r0, c1 - c0, "view")
    v.op, v.addr, v.owner, v.stride = None, self.realize().addr, self.owner or self, self.stride
    v.offset = self.offset + r0 // 32 * self.stride + c0 // 32
    return v

  @property
  def T(self):
    """The transpose, computed now, in a kernel of its own (see transpose_kernel)."""
    out = Tensor(self.rt, self.cols, self.rows)
    return out._run(_cached(("transpose", self.rows, self.cols), lambda: transpose_kernel(self.rows, self.cols)), out.addr, self.realize().addr, self.stride, self.offset)

  def realize(self):
    if self.op is None: return self
    while _ops(self, set()) > 15:  # too big for one kernel (SEM_L1 counts 15): its biggest part goes first, on its own
      max((s for s in self.srcs if s.op), key=lambda s: _ops(s, set())).realize()
    inputs = []
    e = _expr(self, self.rows, inputs)
    key = (e, tuple((u.rows, u.cols) for u in inputs))
    kernel = _cached(key, lambda: fused_kernel(*key))
    self.addr, self.op, self.srcs = self.rt.dram_alloc(-(-self.tiles // len(bh.DRAM)) * TILE), None, ()
    rowwise = any(isinstance(x, tuple) and x[0] in COMBINE for x in _subexprs(e))  # units are whole rows
    return self._run(kernel, self.addr, *[v for t in inputs for v in (t.addr, t.stride, t.offset)], units=-(-self.rows // 32) if rowwise else None)

  def _run(self, kernel, *args, units=None):  # this tensor's tiles, or rows, split evenly over as many workers as have any
    units = units or self.tiles
    cores = self.rt.cores[:units]
    share = {xy: (i * units // len(cores), (i + 1) * units // len(cores)) for i, xy in enumerate(cores)}
    self.rt.launch(kernel, lambda x, y: (share[x, y][0], share[x, y][1] - share[x, y][0], *args), len(cores))
    self.rt.wait(60)  # a model's biggest kernels take seconds on the simulator (milliseconds on a chip)
    return self

  def _elementwise(self, op, other):
    """numpy's broadcasting: shapes match, or one side has 1 where the other does not. A number is a 1 x 1 tensor."""
    if not isinstance(other, Tensor): other = Tensor.full(self.rt, 1, 1, other)
    shape = []
    for a, b in ((self.rows, other.rows), (self.cols, other.cols)):
      assert a == b or 1 in (a, b), f"cannot broadcast ({self.rows}, {self.cols}) with ({other.rows}, {other.cols})"
      shape.append(max(a, b))
    return Tensor(self.rt, *shape, op, (self, other))
  def relu(self): return Tensor(self.rt, self.rows, self.cols, "relu", (self,))
  def exp(self): return Tensor(self.rt, self.rows, self.cols, "exp", (self,))
  def reciprocal(self): return Tensor(self.rt, self.rows, self.cols, "recip", (self,))
  def rsqrt(self): return Tensor(self.rt, self.rows, self.cols, "rsqrt", (self,))
  def gelu(self):  # GPT-2's: 0.5 x (1 + tanh(z)), z = sqrt(2/pi) (x + 0.044715 x^3), which is x / (1 + e^-2z)
    k = 2 * math.sqrt(2 / math.pi)
    return self * (1 + (self * (self * self * (-0.044715 * k) - k)).exp()).reciprocal()
  def rowsum(self): return self if self.cols == 1 else Tensor(self.rt, self.rows, 1, "rowsum", (self,))
  def rowmax(self): return self if self.cols == 1 else Tensor(self.rt, self.rows, 1, "rowmax", (self,))
  def __add__(self, other): return self._elementwise("add", other)
  def __sub__(self, other): return self._elementwise("sub", other)
  def __mul__(self, other): return self._elementwise("mul", other)
  def __truediv__(self, other): return self * (1 / other if isinstance(other, (int, float)) else other.reciprocal())
  __radd__, __rmul__ = __add__, __mul__
  def __rsub__(self, other): return Tensor.full(self.rt, 1, 1, other) - self
  def __neg__(self): return self * -1.0

  def __matmul__(self, other):
    assert self.cols == other.rows, f"({self.rows}, {self.cols}) @ ({other.rows}, {other.cols})"
    if self.cols == 1: return self * other  # (R, 1) @ (1, C) is broadcasting
    return Tensor(self.rt, self.rows, other.cols, "matmul", (self, other))

def _ops(t, seen):  # how many ops a kernel for t would hold, at most (an SFPU op on another's result adds none)
  if t.op is None or id(t) in seen: return 0
  seen.add(id(t))
  return 1 + (0 if t.op == "matmul" else sum(_ops(s, seen) for s in t.srcs))

def _expr(t, rows, inputs):  # the tree of unrealized ops under t, realized tensors as numbers into inputs
  if t.op is None:
    if all(t is not u for u in inputs): inputs.append(t)
    return next(i for i, u in enumerate(inputs) if u is t)
  if t.rows != rows: return _expr(t.realize(), rows, inputs)  # one row, repeated down: compute it once
  if t.op == "matmul":  # its operands are whole matrices, not tiles: realize them on their own first
    a, b = (s.realize() for s in t.srcs)
    return ("matmul", _expr(a, rows, inputs), _expr(b, rows, inputs))
  return (t.op, *(_expr(s, rows, inputs) for s in t.srcs))

def _subexprs(e):
  yield e
  if isinstance(e, tuple):
    for c in e[1:]: yield from _subexprs(c)

kernels = {}  # (expression, input shapes), or a kernel's name and shape -> compiled kernel image
def _cached(key, compile):
  if key not in kernels: kernels[key] = compile()
  return kernels[key]
