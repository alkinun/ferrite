# The compiler: tensor ops emitted as Tensix instructions straight from the ISA, no LLK. A tensor lives in DRAM as
# bf16 32x32 tiles, tile i in bank i % 8. Ops are lazy: a tree of elementwise ops, matmuls, SFPU ops (relu, exp,
# reciprocal, rsqrt) and reductions along an axis becomes one fused kernel, which fetches each input tile once, keeps every
# intermediate in L1 and writes only the result to DRAM. A kernel is four programs for each worker tile: BRISC fetches
# operand tiles into L1 over the NoC and sends results home, and each TRISC feeds one Tensix thread its instructions
# (unpack, math, pack), the three running at the same time on different tiles.
#
# Tile layout: four 16x16 faces in row-major order (top-left, top-right, bottom-left, bottom-right), each face
# row-major, so datum (r, c) is at (r // 16 * 2 + c // 16) * 256 + r % 16 * 16 + c % 16. The ISA has no notion of a
# tile; this is the layout the FPU's 16x16 matrix multiply wants. An unpacker reads a whole tile into a SrcA/SrcB bank,
# so face f lands in rows 16f..16f+15, and Dst holds the result the same way.
import array, math, struct
from ferrite import bh
from ferrite.runtime import DATA, SYNC, ZEROS, compile_kernel

TILE = 32 * 32 * 2  # bytes in a bf16 tile: one SrcA/SrcB bank (64 rows of 16), 64 Dst rows
SLOTS = ZEROS  # L1 tile slots: slot i at SLOTS + i * TILE (slot 0 is the runtime's zeros), picked by the unpackers'
               # Z and the packer's channel 1 Y counters
DST_HALF = 64  # Dst rows of one result in fp32; Dst holds two, so math fills one half while pack drains the other
SEM_DST, SEM_PACKED, SEM_L1 = 1 << 0, 1 << 1, 1 << 2  # results in Dst to pack, outputs packed but not reported, and
                                                       # intermediates packed into L1 but not yet waited for
SEM_UNPACKED = 1 << 5  # frames (and matmul steps) unpacked but not reported
ACCS = ((256, 1 << 3), (320, 1 << 4))  # reductions accumulate here in Dst (scratch 128 rows on), each with a
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
def setadc(unpacker, slot): return ins("SETADC", cnt_set_mask=unpacker, dimension_index=2, value=slot)  # its Z: the slot
def unpack(a, b):  # slot a into SrcA and slot b into SrcB, each bank then handed to the FPU
  return [setadc(bh.UNPACK0, a), setadc(bh.UNPACK1, b), ins("UNPACR", last=1, set_dat_valid=1, ovrd_thread_id=1),
          ins("UNPACR", last=1, set_dat_valid=1, ovrd_thread_id=1, unpack_block_selection=1)]
REPORT = [ins("STALLWAIT", stall_res=bh.STALL_SYNC, wait_res=bh.WAIT_UNPACK), ins("SEMPOST", sem_sel=SEM_UNPACKED)]
def haloize(on):  # unpacker 0 transposes each 16x16 face into SrcA from its next unpack on (or stops): the config write
  word, bit, _ = bh.TENSIX_CFG_FIELDS["SEC0_Haloize_mode"]  # waits for unpacker 0's unpacks, the next unpack for the write
  return [ins("STALLWAIT", stall_res=bh.STALL_CFG, wait_res=bh.WAIT_UNPACK0),
          ins(f"RMWCIB{bit // 8}", cfg_reg_addr=word, data=on << bit % 8, mask=1 << bit % 8), ins("STALLWAIT", stall_res=bh.STALL_UNPACK, wait_res=bh.WAIT_CFG)]
def step_unpack(s):  # a matmul step: B's tile from stream slot s into SrcA, and A's next tile in its row into SrcB
  return [setadc(bh.UNPACK0, s), ins("UNPACR", last=1, set_dat_valid=1, ovrd_thread_id=1),
          ins("UNPACR", last=1, set_dat_valid=1, ovrd_thread_id=1, unpack_block_selection=1, addr_mode=1)] + REPORT  # mode 1: Z += 1
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
          ins("STALLWAIT", stall_res=bh.STALL_SYNC, wait_res=bh.WAIT_PACK), ins("SEMGET", sem_sel=SEM_DST)] + \
         [ins("SEMPOST", sem_sel=d) for d in done]
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
# Reductions along an axis (0: the batch, to one matrix; 1: rows, to one row; 2: columns, to one column): each tile
# along it is x + 0 into Dst (the first in the result's half, the rest 128 rows further on, where the SFPU folds them
# in). Then each row or column of the tile is folded across its 32 and the result written to all of them, so the tile
# broadcasts along that axis as is (over the batch, the tiles were folded as they came and there is nothing left).
FOLDS = {f"{f}{ax}": ax for f in ("sum", "max") for ax in range(3)}  # kind -> the axis it folds
def fold(kind, d, t):  # LReg d = d op t (for max, t is left with the min)
  return ins("SFPSWAP", vd=t, vc=d, mod1=1) if kind[:3] == "max" else mad(d, t, ONE, d)
COMBINE = {k: [ins("SFPLOAD", vd=0, mod0=3, addr_mode=2), ins("SFPLOAD", vd=1, mod0=3, addr_mode=2, imm10=128), fold(k, 0, 1), sfp_store(0)]
           for k in FOLDS}
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
def across_cols(kind):
  """Fold every tile column across its 32 rows. Column c of a tile is lane c % 16 // 2 of the loads at the even (c even)
  or odd addresses of its two faces (the left ones for c < 16), a row of it in each 8-lane group: the 8 loads fold into
  one register, three copies and SFPTRANSP put each of its lane groups across a register of its own, and folding those
  leaves the column's result in every lane, which goes back to all 8 places."""
  p = [ins("STALLWAIT", stall_res=bh.STALL_SFPU, wait_res=bh.WAIT_FPU), ins("SETRWC", bit_mask=0b0100)]
  for half, odd in [(h, o) for h in (0, 16) for o in (0, 2)]:
    at = [half + odd + base for base in (0, 4, 8, 12, 32, 36, 40, 44)]  # top face rows 0-15, bottom face
    p += [ins("SFPLOAD", vd=0, mod0=3, addr_mode=2, imm10=at[0])]
    p += [w for a in at[1:] for w in (ins("SFPLOAD", vd=1, mod0=3, addr_mode=2, imm10=a), fold(kind, 0, 1))]
    p += [ins("SFPMOV", vd=r, vc=0) for r in (1, 2, 3)] + [ins("SFPTRANSP")] + [fold(kind, 0, r) for r in (1, 2, 3)]
    p += [ins("SFPSTORE", vd=0, mod0=3, addr_mode=2, imm10=a) for a in at]
  return p
ACROSS = {0: lambda kind: [], 1: across_cols, 2: across_rows}
def masking(kind):  # the SFPU after the FPU, and the reduction's identity in a register: 0 (LReg 9), or -inf in LReg 5 for max
  p = [ins("STALLWAIT", stall_res=bh.STALL_SFPU, wait_res=bh.WAIT_FPU), ins("SETRWC", bit_mask=0b0100)]
  return (p, ZERO) if kind[:3] == "sum" else (p + sfp_const(5, -math.inf), 5)
def mask_columns(kind, n, d=0):
  """Columns n and on of the tile at Dst row d to the reduction's identity (0, or -inf for max), so the padding past a
  tensor's last column does not count. An SFPSTORE writes 4 rows of 8 even or odd columns, lane l at column 2 (l % 8)
  (+ 1 if odd); LReg 15 holds 2l, so LReg 7 = LReg 15 & 14 is twice the lane's place in its row, and each store goes to
  the lanes whose column is n or more."""
  p, fill = masking(kind)
  p += sfp_bits(7, 14) + [ins("SFPAND", vd=7, vc=15)]
  for base, right, odd in [(b, r, o) for b in (0, 4, 8, 12, 32, 36, 40, 44) for r in (0, 1) for o in (0, 1)]:
    t = max(0, min(8, -(-(n - 16 * right - odd) // 2)))  # the first lane (of 8) at column n or more
    if t < 8: p += [ins("SFPENCC", mod1=10, imm2=3), ins("SFPIADD", vd=6, vc=7, mod1=9, imm12=-2 * t & 0xFFF),  # lanes t+
                    ins("SFPSTORE", vd=fill, mod0=3, addr_mode=2, imm10=d + base + 16 * right + 2 * odd)]  # right faces: 16 rows on
  return p + [ins("SFPENCC", mod1=10, imm2=2)]
def mask_rows(kind, m, d=0):
  """Rows m and on of the tile at Dst row d to the identity, past a tensor's last row. A store covers 4 rows, lane l in
  row l // 8 of them, and LReg 15 holds 2l, so LReg 15 - 16k >= 0 picks the rows from k on."""
  p, fill = masking(kind)
  for r in range(0, 32, 4):  # tile rows r..r+3: Dst rows (r // 16) * 32 + r % 16 on, left face, and 16 more, right face
    k = max(0, m - r)
    if k < 4: p += [w for o in (0, 2, 16, 18) for w in (ins("SFPENCC", mod1=10, imm2=3), ins("SFPIADD", vd=6, vc=15, mod1=9, imm12=-16 * k & 0xFFF),
                                                     ins("SFPSTORE", vd=fill, mod0=3, addr_mode=2, imm10=d + r // 16 * 32 + r % 16 + o))]
  return p + [ins("SFPENCC", mod1=10, imm2=2)]
MASK = {1: mask_rows, 2: mask_columns}  # the padding past the last tile along an axis
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
  """One tile of a reduction, which has SrcA = the tile and SrcB = zeros: the first lands in the accumulator once it is
  free; later ones land in scratch and the SFPU folds them in. edge: how many of the tile's rows or columns along the
  axis are the tensor's, if it is the line's last and cut short."""
  row, sem = ACCS[acc]
  mask = lambda d: MASK[FOLDS[kind]](kind, edge, d) if edge else []
  if first: return ([ins("SEMWAIT", stall_res=bh.STALL_ALL, sem_sel=sem, wait_sem_cond=1), ins("SEMGET", sem_sel=sem), at(row)] +
                    elementwise("add", slots) + mask(0))
  return ([at(row)] + elementwise("add", slots, dst=128) + mask(128) +
          [ins("STALLWAIT", stall_res=bh.STALL_SFPU, wait_res=bh.WAIT_FPU), ins("SETRWC", bit_mask=0b0100)] + runs(slots, kind, 32))
def reduce_finish(kind, slots, acc, posts):  # fold across the tile and run the SFPU ops on the result (then MATH_END)
  return [ins("SEMWAIT", stall_res=bh.STALL_ALL, sem_sel=SEM_DST, wait_sem_cond=2), at(ACCS[acc][0])] + ACROSS[FOLDS[kind]](kind) + sfpu(posts, slots)

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

ZERO_SLOT, FRAMES = 0, 5  # slot 0 is a zero tile, 1-4 stream a matmul's B and A, then two frames, row values, A's rows
UNPACKABLE = 256  # an unpacker sees only the low 8 bits of its Z counter (ISA ADCs.md): the slots it reads are below this

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
static inline void push_word(unsigned w) {{ *(volatile unsigned *){bh.TENSIX_PUSH:#x} = w; }}
"""
def _pushes(words): return " ".join(f"push_word({w:#x});" for w in words)  # straight-line, with no table to load from

def _brisc(total, body):  # moves tiles: inputs into L1 ahead of thread 0, results home behind the pack thread
  return SYNC_WORDS + f"""
#define SLOT(i) ({SLOTS:#x} + (i) * {TILE})
{_array("bank", [x | y << 6 for x, y in bh.DRAM])}
static unsigned fetched, events, frame_done[2], stream_done[2], steps, sent, out, rows[16];  // rows: i + 1 of what each place in L1 holds (fresh)
// tile t of a tensor at DRAM address base: bank t % 8, row t / 8 of the bank
#define BANK(t) bank[(t) % 8] & 63, bank[(t) % 8] >> 6  // the NoC x, y of tile t's bank
static void get(unsigned base, unsigned t, unsigned slot) {{ noc_read(BANK(t), base + t / 8 * {TILE}, SLOT(slot), {TILE}); }}
// the same with its two off-diagonal faces swapped, half of transposing it (unpacker 0 does the rest)
static void flipped(unsigned base, unsigned t, unsigned slot) {{
  for (unsigned f = 0; f < 4; f++)
    noc_read(BANK(t), base + t / 8 * {TILE} + f * {TILE // 4}, SLOT(slot) + (f == 1 ? 2 : f == 2 ? 1 : f) * {TILE // 4}, {TILE // 4});
}}
static void ready(void) {{ noc_wait(); SYNC[FETCHED] = ++fetched; }}
static void send(void) {{  // the results packed so far go home, in the order they were packed
  for (; sent < peek(PACKED); SYNC[SENT] = ++sent) {{
    unsigned slot = peek(RING + sent % 4 * 2), t = peek(RING + sent % 4 * 2 + 1);
    noc_write(SLOT(slot), BANK(t), out + t / 8 * {TILE}, {TILE}); noc_wait();
  }}
}}
// A slot is fetched into again once thread 0 has unpacked what was there: the frame two tiles back, B's slot two steps
// back. Thread 0 reports each frame and step in turn; events counts them the same way. Results go home meanwhile.
static void reached(unsigned n) {{ while (peek(UNPACKED) < n) send(); }}
// A matmul step: B's tile (for SrcA) into b_slot and A's into a_slot, each a stream slot (B 1-2, A 3-4) or its place in
// the row or column kept in L1, unless it is there already (KEPT)
#define KEPT 0xFFFFFFFF
static void step(unsigned bt, unsigned b, unsigned b_tile, unsigned b_slot, unsigned a, unsigned a_tile, unsigned a_slot) {{
  reached(stream_done[steps & 1]);
  if (b_slot != KEPT) {{ if (bt) flipped(b, b_tile, b_slot); else get(b, b_tile, b_slot); }}  // bt: B is read transposed
  if (a_slot != KEPT) get(a, a_tile, a_slot);
  ready();
  stream_done[steps++ & 1] = ++events;
}}
// A row of A (or column of B) stays in L1, in place q, while the result tiles use it; when tag i, which row, changes,
// the new one comes in once the old is unpacked
static unsigned fresh(unsigned q, unsigned i) {{
  if (rows[q] == i + 1) return 0;
  reached(events);
  return rows[q] = i + 1;
}}
void kmain(volatile unsigned *a) {{
  out = a[2];
  unsigned tiles = 0;
{body}
  while (sent < {total}) send();
}}"""

def _unpacker(body, cfg=CONFIG):  # TRISC0: the backend config and semaphores, then unpacking, each tile once BRISC has fetched it
  return TRISC + f"""
#define CFG ((volatile unsigned *){bh.TENSIX_CFG:#x})
static unsigned needed, relayed, posted, drained, steps;
static void relay(void) {{ for (; SEM({SEM_UNPACKED}); SYNC[UNPACKED] = ++relayed) SEM({SEM_UNPACKED}) = 1; }}  // thread 0's reports, for BRISC
static void fetched(void) {{ needed++; do relay(); while (peek(FETCHED) < needed); }}  // until BRISC has fetched the next tiles
static void drain(void) {{ for (; drained < posted; drained++) {{ {tt(TAKE_L1)} }} }}  // until everything packed so far has landed
void kmain(volatile unsigned *a) {{
  {" ".join(f"CFG[{i}] = {v:#x};" for i, v in cfg.items())}
  asm volatile("fence" ::: "memory");  // the config writes are out; a read from the same block is processed after them
  (void)CFG[{next(iter(cfg))}];
  asm volatile("fence" ::: "memory");  // and has answered, so no instruction pushed below can see them stale
  {_pushes(THREAD + SETUP_UNPACK)}
  tensix_idle();  // the semaphores are set before the other threads can touch them
  SYNC[READY] = 1;
  unsigned tiles = 0;
{body}
}}"""

def _math(slots, body):  # TRISC1
  return TRISC + f"""
void kmain(volatile unsigned *a) {{
  {_pushes(THREAD + setup_math(slots))}
  while (!peek(READY)) {{}}  // the backend config and the semaphores
  unsigned tiles = 0, h = 0;  // h: the Dst half the next op gets
{body}
}}"""

def _packer(body):  # TRISC2
  return TRISC + f"""
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
  {_pushes(THREAD + SETUP_PACK)}
  while (!peek(READY)) {{}}
  unsigned tiles = 0, h = 0;  // h: the Dst half the next op gets
{body}
  while (relayed < outputs) relay();
}}"""

def fused_kernel(expr, shapes):
  """A kernel for an expression: nested tuples, (op, x, y) for add, sub and mul, (op, x) for relu, exp, recip, rsqrt and
  the reductions (FOLDS: sum0, max2, ...), ("matmul", a, b) where a is an input and b an input or ("T", input), read
  transposed, and plain numbers for inputs. shapes: each input's (batch, rows, columns), laid out as Tensor says; a
  dimension of 1 repeats across the others (broadcasting).
  args: [first unit, units, then the result's and each input's DRAM address, tile strides for batch and row, and tile
  offset (Tensor's)]. A unit is a tile of the result, or a block of up to 4 x 4 of them (one matmul and nothing that
  reduces), or, if the expression reduces along an axis, a whole line of tiles along it (a row, a column, or one tile
  position through the batch), so the reduction can stay on chip.

  Every op computes into Dst and is packed into an L1 slot: elementwise ops from their operands' slots, a matmul from
  its K pairs of tiles, B's streamed through slots 1-2 and A's row kept in L1 while the result tiles stay in its row
  (if it fits below slot 256, else streamed through slots 3-4). A unary op runs on the SFPU over Dst: in place, on the
  result of the op below it if nothing else reads that, or else on its operand plus the zero tile (slot 0). A reduction
  folds each tile of its operand along the axis into an accumulator in Dst, then across the tile (across_rows,
  across_cols), and packs one tile.
  A line is done in passes: a pass is a sweep over its tiles, and a reduction's result can only be read in the pass
  after the one that swept it. Each pass recomputes what it needs (softmax computes exp twice) rather than keep whole
  lines in L1. Values one tile long along the axis (reductions, ops on them, inputs of 1 there) are worked out once per
  line, in the first tile of the pass that needs them, into slots of their own.
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
  ext = lambda s, ax: s[ax] if ax == 0 else -(-s[ax] // 32)  # a shape's extent along an axis: matrices, or tiles
  ops, refs = [], {}  # ops: dicts with kind, x, y, kt, posts, shape; refs: expr -> input number or ("op", j)
  size = lambda r: shapes[r] if isinstance(r, int) else ops[r[1]]["shape"]  # (batch, rows, columns)
  def visit(e):
    if isinstance(e, int): return e
    if e in refs: return refs[e]
    if e[0] == "matmul":  # B is an input, or ("T", input): that input transposed
      bt, y = (True, e[2][1]) if isinstance(e[2], tuple) else (False, e[2])
      a = shapes[e[1]]
      op = dict(kind="matmul", x=e[1], y=y, bt=bt, kt=ext(a, 2), shape=(max(a[0], shapes[y][0]), a[1], shapes[y][1 if bt else 2]))
    elif e[0] in COMBINE:
      x = visit(e[1])
      op = dict(kind=e[0], x=x, shape=tuple(1 if ax == FOLDS[e[0]] else s for ax, s in enumerate(size(x))))
    elif e[0] in SFPU:
      x = visit(e[1])
      if isinstance(x, tuple) and uses[e[1]] == 1:  # in place, on the result of the op below
        ops[x[1]]["posts"].append(e[0])
        refs[e] = x
        return x
      op = dict(kind="add", x=x, y="zero", shape=size(x), posts=[e[0]])  # operand + 0 into Dst, then the SFPU
    else:
      x, y = visit(e[1]), visit(e[2])
      op = dict(kind=e[0], x=x, y=y, shape=tuple(map(max, size(x), size(y))))
    ops.append({"posts": [], **op})
    refs[e] = ("op", len(ops) - 1)
    return refs[e]
  root = visit(expr)[1]
  operands = lambda op: [r for r in (op.get("x"), op.get("y")) if r not in (None, "zero")] if op["kind"] != "matmul" else []
  reduced = lambda r: isinstance(r, tuple) and ops[r[1]]["kind"] in COMBINE
  axes = {FOLDS[op["kind"]] for op in ops if op["kind"] in COMBINE}
  assert len(axes) <= 1, "a kernel reduces along one axis"
  reducing, axis = bool(axes), next(iter(axes), 2)  # the axis swept, in passes; otherwise units are tiles, j along a row
  # n: the result's extent along each axis, what ops sweep along the one they reduce (a matmul's A can be wider: its
  # columns are K, streamed, not swept)
  inputs = len(shapes)
  n = [max([ext(op["shape"], ax) for op in ops] + [ext(size(r), ax) for op in ops for r in operands(op)]) for ax in range(3)]
  nb, rt, w = n
  L = n[axis]  # the sweep's length
  assert all(ext(size(r), axis) in (1, L) for op in ops for r in operands(op)), "a kernel sweeps its lines at one length"
  swept = lambda r: r != "zero" and ext(size(r), axis) == L and not reduced(r)  # varies along the sweep; otherwise one tile a line
  assert len(ops) <= 15, "SEM_L1 counts at most 15 intermediates"
  # passes: a swept op runs in the pass after every reduction it depends on has finished; a narrow one is ready then
  ready, sweep = {}, {}  # ready: first pass a value can be read in; sweep: pass a reduction sweeps its operand in
  for j, op in enumerate(ops):
    deps = [ready[r[1]] if isinstance(r, tuple) else 0 for r in operands(op)]
    if op["kind"] in COMBINE: sweep[j] = max(deps, default=0); ready[j] = sweep[j] + 1
    else: ready[j] = max(deps, default=0)
  last = sweep[root] if root in sweep else ready[root]
  slots = allocate({"mm" for op in ops if op["kind"] == "matmul"} | {"mul" for op in ops if op["kind"] == "mul"} |
                   {u for op in ops for u in op["posts"]} | {op["kind"] for op in ops if op["kind"] in COMBINE})
  frame, matmuls = inputs + len(ops), [j for j, op in enumerate(ops) if op["kind"] == "matmul"]
  rows_at, top = [], FRAMES + 2 * frame + len(ops)  # each matmul's row of A in L1, while they fit; else A streams too
  assert top <= UNPACKABLE and len(matmuls) <= 16, "too many tiles in a kernel"
  for j in matmuls:
    fits = top + ops[j]["kt"] <= UNPACKABLE
    rows_at.append(top if fits else None)
    top += ops[j]["kt"] if fits else 0
  me = ("op", root)
  rows, cols = size(me)[1:]  # the result's
  # With one matmul and no passes, a unit is a block of bp x bq result tiles, column by column: the block's bp rows of
  # A and its current column of B stay in L1, so each is read from DRAM once a block, not once a tile
  bp = min(4, rt) if len(matmuls) == 1 and not reducing else 1
  while bp > 1 and FRAMES + 2 * frame + len(ops) + (bp + 1) * ops[matmuls[0]]["kt"] > UNPACKABLE: bp -= 1
  bq = min(4, w) if bp > 1 else 1
  if bp > 1: rows_at = [FRAMES + 2 * frame + len(ops)]  # A's bp rows, then B's column
  kept_b = rows_at[0] + bp * ops[matmuls[0]]["kt"] if bp > 1 else None
  def slot(f, r):  # an input or swept op lives in the frame, a narrow op in its own slot after both frames
    if r == "zero": return ZERO_SLOT
    if isinstance(r, int): return FRAMES + f * frame + r
    return FRAMES + (f * frame + inputs + r[1] if swept(r) else 2 * frame + r[1])
  arg = lambda i: 6 + 4 * i  # input i's [address, batch stride, row stride, offset] in a, after the result's
  def where(r, base):  # which tile of r (an input, or the result: its [address, batch stride, row stride, offset] are at a[base]) is matrix b, row i_, column j's
    s = size(r)
    return f"{f'b * a[{base + 1}] + ' if s[0] > 1 else ''}{'i_' if ext(s, 1) > 1 else 0} * a[{base + 2}] + {'j' if ext(s, 2) > 1 else 0} + a[{base + 3}]"
  tile = lambda i: where(i, arg(i))
  B, U, M, P = range(4)  # the programs: BRISC moves tiles; TRISC 0, 1, 2 feed the unpack, math and pack threads
  def math_end(j):  # the math thread's last words for op j: the root's padding in Dst goes to 0 first, in a tile at the edge
    if j != root: return tt(MATH_END)
    at_end = lambda ax: f"{('i_', 'j')[ax - 1]} == {ext(size(me), ax) - 1}"
    edge = [f"if ({at_end(2)}) {{ {tt(mask_columns('sum', cols % 32))} }}"] * (cols > 1 and cols % 32 > 0)
    edge += [f"if ({at_end(1)}) {{ {tt(mask_rows('sum', rows % 32))} }}"] * (rows > 1 and rows % 32 > 0)
    return " ".join(edge + [tt(MATH_END)])
  def output(j, f):  # for the pack thread: where op j's result goes, if it is the root
    return f"output({f if swept(me) else 0}, {slot(f, me)}, {where(me, 2)}); " if j == root else ""
  passes = []
  for p in range(last + 1):
    reductions = [j for j in sweep if sweep[j] == p]
    sinks = [ops[j]["x"] for j in reductions] + ([me] if p == last and swept(me) else [])
    swept_ops = []  # the swept ops this pass computes, operands first
    def need(r):
      if isinstance(r, tuple) and swept(r) and r[1] not in swept_ops:
        for o in operands(ops[r[1]]): need(o)
        swept_ops.append(r[1])
    for r in sinks: need(r)
    once = [j for j, op in enumerate(ops) if not swept(("op", j)) and op["kind"] not in COMBINE and ready[j] == p]
    bodies = []
    for f in (0, 1):
      c = [[], [], [], []]  # each program's C for this tile
      fetch = sorted({r for j in once + swept_ops + reductions for r in operands(ops[j]) if isinstance(r, int)})
      c[B] += [f"reached(frame_done[{f}]);"] + [f"get(a[{arg(r)}], {tile(r)}, {slot(f, r)});" for r in fetch] + ["ready();"]
      c[U].append("fetched();")
      for j in once + swept_ops:
        op, me_j = ops[j], ("op", j)
        cond, posted = "if (k == 0) " if j in once else "", "" if j == root else " posted++;"
        pk = halves([pack(DST_HALF * h, slot(f, me_j), SEM_PACKED if j == root else SEM_L1) for h in (0, 1)])
        c[P].append(f"{cond}{{ {output(j, f)}{pk} h ^= 1; }}")
        if op["kind"] == "matmul":
          kt, bw, q, bt = op["kt"], ext(op["shape"], 2), matmuls.index(j), int(op["bt"])  # B's width in tiles; which matmul
          x, y = arg(op["x"]), arg(op["y"])
          bj = 'j' if bw > 1 else 0  # B's tile (m, j), or for B transposed the input's tile (j, m)
          in_batch = f"b * a[{y + 1}] + " if shapes[op["y"]][0] > 1 else ""
          b_tile = f"{in_batch}{bj} * a[{y + 2}] + m + a[{y + 3}]" if bt else f"{in_batch}m * a[{y + 2}] + {bj} + a[{y + 3}]"
          on, off = (tt(haloize(1)), tt(haloize(0))) if bt else ("", "")
          a_tile = f"{f'b * a[{x + 1}] + ' if shapes[op['x']][0] > 1 else ''}i_ * a[{x + 2}] + m + a[{x + 3}]"
          if kept_b is not None:  # A's bp rows and B's column in L1: unpacker 0 walks the column, 1 the row of A
            c[B].append(f"{cond}{{ unsigned na = fresh(i_ % {bp}, b * {rt} + i_), nb = fresh(15, b * {w} + j); for (unsigned m = 0; m < {kt}; m++) "
                        f"step({bt}, a[{y}], {b_tile}, nb ? {kept_b} + m : KEPT, a[{x}], {a_tile}, na ? {rows_at[q]} + i_ % {bp} * {kt} + m : KEPT); }}")
            walk = [ins("UNPACR", last=1, set_dat_valid=1, ovrd_thread_id=1, addr_mode=1),
                    ins("UNPACR", last=1, set_dat_valid=1, ovrd_thread_id=1, unpack_block_selection=1, addr_mode=1)] + REPORT
            c[U].append(f"{cond}{{ push_word({setadc(bh.UNPACK1, 0):#x} + {rows_at[q]} + i_ % {bp} * {kt}); {tt([setadc(bh.UNPACK0, kept_b)])} "
                        f"{on} for (unsigned m = 0; m < {kt}; m++) {{ fetched(); {tt(walk)} }} {off}{posted} }}")
          elif rows_at[q] is None:  # A's tile streams in beside B's, each step
            c[B].append(f"{cond}for (unsigned m = 0; m < {kt}; m++) step({bt}, a[{y}], {b_tile}, 1 + (steps & 1), a[{x}], {a_tile}, 3 + (steps & 1));")
            c[U].append(f"{cond}{{ {on} for (unsigned m = 0; m < {kt}; m++) {{ fetched(); "
                        f"if (steps++ & 1) {{ {tt(unpack(2, 4) + REPORT)} }} else {{ {tt(unpack(1, 3) + REPORT)} }} }} {off}{posted} }}")
          else:
            c[B].append(f"{cond}{{ unsigned new = fresh({q}, b * {rt} + i_); for (unsigned m = 0; m < {kt}; m++) "
                        f"step({bt}, a[{y}], {b_tile}, 1 + (steps & 1), a[{x}], {a_tile}, new ? {rows_at[q]} + m : KEPT); }}")
            c[U].append(f"{cond}{{ {tt([setadc(bh.UNPACK1, rows_at[q])])} {on} for (unsigned m = 0; m < {kt}; m++) {{ fetched(); "
                        f"if (steps++ & 1) {{ {tt(step_unpack(2))} }} else {{ {tt(step_unpack(1))} }} }} {off}{posted} }}")
          c[M].append(f"{cond}{{ {halves([math_begin(h, True) for h in (0, 1)])} for (unsigned m = 0; m < {kt}; m++) {{ {tt(matmul(slots))} }} "
                      f"{tt(sfpu(op['posts'], slots))} {math_end(j)} h ^= 1; }}")
          continue
        drain = "drain(); " if any(isinstance(r, tuple) for r in operands(op)) else ""
        c[U].append(f"{cond}{{ {drain}{tt(unpack(slot(f, op['x']), slot(f, op['y'])))}{posted} }}")
        math = halves([math_begin(h, op['kind'] == 'mul') + elementwise(op['kind'], slots) + sfpu(op['posts'], slots) for h in (0, 1)])
        c[M].append(f"{cond}{{ {math} {math_end(j)} h ^= 1; }}")
      for acc, j in enumerate(reductions):
        assert acc < len(ACCS), "two reductions a pass"
        op, me_j, x, kind = ops[j], ("op", j), ops[j]["x"], ops[j]["kind"]
        drain = "drain(); " if isinstance(x, tuple) else ""
        edge = size(x)[axis] % 32 if axis and size(x)[axis] > 1 else 0  # the operand's rows or columns in its last tile, if cut short
        end = f"k == {L - 1}" if swept(x) else "k == 0"  # the line's last tile (a narrow operand is one tile a line)
        def step(first):  # the last tile's padding must not count
          plain, cut = tt(reduce_step(kind, slots, acc, first)), tt(reduce_step(kind, slots, acc, first, edge))
          return f"if ({end}) {{ {cut} }} else {{ {plain} }}" if edge else plain
        finish, posted = f'{tt(reduce_finish(kind, slots, acc, op["posts"]))} {math_end(j)}', "" if j == root else " posted++;"
        pk = tt(pack(ACCS[acc][0], slot(f, me_j), SEM_PACKED if j == root else SEM_L1, ACCS[acc][1]))
        if swept(x):  # swept along the line
          c[U].append(f"{drain}{tt(unpack(slot(f, x), ZERO_SLOT))} if ({end}) {{{posted} }}")
          c[M].append(f"if (k == 0) {{ {step(True)} }} else {{ {step(False)} }} if ({end}) {{ {finish} }}")
        else:  # reduced in the pass's first tile
          c[U].append(f"if ({end}) {{ {drain}{tt(unpack(slot(f, x), ZERO_SLOT))}{posted} }}")
          c[M].append(f"if ({end}) {{ {step(True)} {finish} }}")
        c[P].append(f"if ({end}) {{ {output(j, f)}{pk} }}")
      c[U].append(tt(REPORT))
      c[B].append(f"frame_done[{f}] = ++events;")
      bodies.append(c)
    passes.append((bool(swept_ops or reductions), bodies))
  br, bc = -(-rt // bp), -(-w // bq)  # blocks down and across a matrix
  def program(c):  # the loops all four programs share, around each one's part of a tile
    tile_c = lambda b: f"if (tiles++ & 1) {{ {' '.join(b[1][c])} }} else {{ {' '.join(b[0][c])} }}"
    if not reducing:  # a unit is a block: its tiles column by column, those past the matrix's edge left out
      return f"""  for (unsigned u = 0; u < a[1]; u++) {{
    unsigned unit = a[0] + u, b = unit / {br * bc}, i0 = unit % {br * bc} / {bc} * {bp}, j0 = unit % {bc} * {bq};
    for (unsigned jj = 0; jj < {bq}; jj++) for (unsigned ii = 0; ii < {bp}; ii++) {{
      unsigned i_ = i0 + ii, j = j0 + jj, k = 0;
      if (i_ >= {rt} || j >= {w}) continue;
      {tile_c(passes[0][1])}
    }}
  }}"""
    index = ("b", "i_", "j")  # a tile's place: matrix, row, column. A unit fixes the two the sweep does not walk
    (outer, _), (inner, m) = [(index[ax], n[ax]) for ax in range(3) if ax != axis]
    sweeps = "\n".join(f"""    for (unsigned {index[axis]} = 0, k = 0; k < {L if full else 1}; {index[axis]}++, k++) {{
      {tile_c(b)}
    }}""" for full, b in passes)
    return f"""  for (unsigned u = 0; u < a[1]; u++) {{
    unsigned unit = a[0] + u, {outer} = unit / {m}, {inner} = unit % {m};
{sweeps}
  }}"""
  total = f"a[1] * {L if swept(me) else 1}" if reducing else "tiles"  # results this worker sends home
  image = compile_kernel(_brisc(total, program(B)), _unpacker(program(U)), _math(slots, program(M)), _packer(program(P)))
  return image, (nb * rt * w // L if reducing else nb * br * bc)  # and how many units the result is

def gather_kernel():
  """Rows of a tensor into a new one, args [first unit, units, out, the tensor's address, tile stride, tile offset, the
  result's tile columns, rows, their indices...]. BRISC alone: for each result tile, each row's two 16-datum runs (left
  and right face) come from wherever that row sits in the tensor's tiles. A NoC read from DRAM must keep its address's
  low 6 bits, so each run comes in as the 64-byte block holding it and is copied into place from there. One row fills
  its tile: it comes in once and is doubled up inside L1 by NoC copies from the tile to itself."""
  return compile_kernel(f"""
#define SLOT {DATA:#x}
#define BLOCK {DATA + TILE:#x}
{_array("bank", [x | y << 6 for x, y in bh.DRAM])}
static void copy(unsigned from, unsigned to, unsigned n) {{ noc_read(noc_me() & 63, noc_me() >> 6, from, to, n); }}
void kmain(volatile unsigned *a) {{
  for (unsigned u = 0; u < a[1]; u++) {{
    unsigned t = a[0] + u, i = t / a[6], j = t % a[6], rows = a[7] == 1 ? 1 : 32;
    copy({ZEROS:#x}, SLOT, {TILE});  // the rows past the end stay zero
    noc_wait();
    for (unsigned r = 0; r < rows; r++) {{
      unsigned row = i * 32 + r;
      for (unsigned face = r / 16 * 2; face < r / 16 * 2 + 2; face++) {{
        volatile unsigned *to = (volatile unsigned *)(SLOT + face * 512 + r % 16 * 32);
        if (row >= a[7]) continue;
        unsigned id = a[8 + row], from = id / 32 * a[4] + j + a[5], at = (id % 32 / 16 * 2 + face % 2) * 512 + id % 16 * 32;
        noc_read(bank[from % 8] & 63, bank[from % 8] >> 6, a[3] + from / 8 * {TILE} + at / 64 * 64, BLOCK, 64);
        noc_wait();
        for (unsigned k = 0; k < 8; k++) to[k] = ((volatile unsigned *)(BLOCK + at % 64))[k];
      }}
    }}
    if (rows == 1) {{  // row 0 of faces 0 and 1 down their 16 rows, then faces 0-1 onto 2-3
      l1_settle(SLOT);
      for (unsigned n = 32; n < 512; n *= 2) {{ copy(SLOT, SLOT + n, n); copy(SLOT + 512, SLOT + 512 + n, n); noc_wait(); }}
      copy(SLOT, SLOT + 1024, 1024);
      noc_wait();
    }}
    noc_write(SLOT, bank[t % 8] & 63, bank[t % 8] >> 6, a[2] + t / 8 * {TILE}, {TILE});
    noc_wait();
  }}
}}""")

def transpose_kernel(rows, cols):
  """x.T for x of (rows, cols), each matrix of a batch, args [first unit, units, out, x's address, tile strides for batch
  and row, tile offset]: result tile (i, j) is x's tile (j, i) transposed. BRISC fetches that tile with its off-diagonal
  faces swapped, and unpacker 0 transposes each face on its way into SrcA (Haloize_mode, on in this kernel's config
  throughout), so x + 0 through the FPU lands the transpose in Dst. (A matmul reading B transposed does the same with
  the mode switched on around its K loop: haloize.)"""
  ct, frame = -(-rows // 32), lambda f: (FRAMES + 2 * f, FRAMES + 2 * f + 1)  # the result's tile columns; x's slot and the result's
  per = ct * -(-cols // 32)  # tiles a matrix
  def loop(*per_frame):  # the same loop over result tiles t for all four programs, frame f alternating
    return f"""  for (unsigned u = 0; u < a[1]; u++) {{
    unsigned t = a[0] + u;
    if (u & 1) {{ {per_frame[1]} }} else {{ {per_frame[0]} }}
  }}"""
  return compile_kernel(
    _brisc("a[1]", loop(*[f"reached(frame_done[{f}]); flipped(a[3], t / {per} * a[4] + t % {ct} * a[5] + t % {per} / {ct} + a[6], {frame(f)[0]}); ready(); "
                                f"frame_done[{f}] = ++events;" for f in (0, 1)])),
    _unpacker(loop(*[f"fetched(); {tt(unpack(frame(f)[0], ZERO_SLOT) + REPORT)}" for f in (0, 1)]), TRANSPOSING),
    _math({}, loop(*[halves([math_begin(h, False) + elementwise("add", {}) + MATH_END for h in (0, 1)]) + " h ^= 1;"] * 2)),  # padding stays 0
    _packer(loop(*[f"output({f}, {frame(f)[1]}, t); " + halves([pack(DST_HALF * h, frame(f)[1], SEM_PACKED) for h in (0, 1)]) + " h ^= 1;"
                   for f in (0, 1)])))

class Tensor:
  """batch x rows x cols of bf16 (batch 1 unless said): in DRAM as 32x32 tiles, matrix after matrix, each in row-major
  tile order, tile i in bank i % 8; or, until realized, an op on other tensors. Realizing compiles the whole tree of
  unrealized ops under it into one kernel; a matmul's own operands are realized first, since each of its output tiles
  needs a whole row and column of them (a transposed B is read transposed rather than realized).
  Any shape: the last tile row and column are padded, with zeros, except that a dimension of 1 fills its tile with
  copies (a row vector's tile holds the row 32 times), so broadcasting reuses the tile as it is. Every kernel leaves
  its result that way too, so a matmul can sum over a padded K.
  A view (x[r0:r1, c0:c1] on tile boundaries, x[b], x.split(n)) shares its owner's DRAM: its tile (b, i, j) is tile
  b * bstride + i * stride + j + offset."""
  def __init__(self, rt, rows, cols, op=None, srcs=(), batch=1):
    self.rt, self.rows, self.cols, self.batch, self.op, self.srcs = rt, rows, cols, batch, op, srcs
    tr, tc = -(-rows // 32), -(-cols // 32)  # tile rows and columns
    self.tiles, self.bstride, self.stride, self.offset, self.owner = batch * tr * tc, tr * tc, tc, 0, None
    self.addr = None if op else rt.dram_alloc(-(-self.tiles // len(bh.DRAM)) * TILE)

  def __del__(self):  # its DRAM goes back once nothing refers to it (and no queued launch can still use it)
    if self.addr is not None and self.owner is None: self.rt.dram_free(self.addr)

  def _where(self, b, i, j):  # tile (i, j) of matrix b in DRAM: (x, y, address)
    t = b * self.bstride + i * self.stride + j + self.offset
    return (*bh.DRAM[t % 8], self.addr + t // 8 * TILE)

  @classmethod
  def full(cls, rt, rows, cols, value): return cls.of(rt, [[value] * cols for _ in range(rows)])

  @classmethod
  def of(cls, rt, data):
    """From a list of rows of numbers, or a list of those (a batch), rounded to bf16 (to nearest even)."""
    matrices = data if isinstance(data[0][0], (list, tuple)) else [data]
    bits = array.array("I", array.array("f", [v for m in matrices for row in m for v in row]).tobytes())
    bf16 = array.array("H", [(b + 0x7FFF + (b >> 16 & 1)) >> 16 for b in bits])
    return cls.frombytes(rt, len(matrices[0]), len(matrices[0][0]), bf16, len(matrices))

  @classmethod
  def frombytes(cls, rt, rows, cols, data, batch=1):
    """From bf16 in row-major order (2 bytes a value, little-endian), matrix after matrix. Runs of 16 values are copied
    into tile order as they are, nothing per value, which is what makes a model's weights quick to load."""
    t, tr, tc, data, pad = cls(rt, rows, cols, batch=batch), -(-rows // 32), -(-cols // 32), memoryview(data).cast("B"), bytes(32)
    assert len(data) == 2 * batch * rows * cols, f"{len(data)} bytes for {batch} x {rows} x {cols}"
    for b, i in ((b, i) for b in range(batch) for i in range(tr * tc)):
      m, r0, c0, tile = data[2 * b * rows * cols:2 * (b + 1) * rows * cols], i // tc * 32, i % tc * 32, []
      for face in range(4):  # 16 rows of 16, face by face (the tile layout, at the top)
        c = c0 + face % 2 * 16
        for r in range(r0 + face // 2 * 16, r0 + face // 2 * 16 + 16):
          r = 0 if rows == 1 else r  # a dimension of 1 fills the tile; the rest pads with 0
          at = 2 * (r * cols + c)
          run = b"" if r >= rows else bytes(m[2 * r:2 * r + 2]) * 16 if cols == 1 else bytes(m[at:at + 2 * max(0, min(16, cols - c))])
          tile.append(run + pad[len(run):])
      t.rt.chip.write(*t._where(b, i // tc, i % tc), b"".join(tile))
    return t

  def tobytes(self):
    """bf16 in row-major order, matrix after matrix, as frombytes takes it. Waits for the launches queued so far."""
    self.realize()
    self.rt.wait()
    out, rows, cols, ct = bytearray(2 * self.batch * self.rows * self.cols), self.rows, self.cols, -(-self.cols // 32)
    for b, i in ((b, i) for b in range(self.batch) for i in range(self.tiles // self.batch)):
      r0, c0, tile, base = i // ct * 32, i % ct * 32, self.rt.chip.read(*self._where(b, i // ct, i % ct), TILE), b * rows * cols
      for face in range(4):
        c, n = c0 + face % 2 * 16, min(16, cols - c0 - face % 2 * 16)
        for k, r in enumerate(range(r0 + face // 2 * 16, min(r0 + face // 2 * 16 + 16, rows))):
          if n > 0: out[2 * (base + r * cols + c):2 * (base + r * cols + c + n)] = tile[face * 512 + k * 32:face * 512 + k * 32 + 2 * n]
    return bytes(out)

  def tolist(self):  # rows of numbers, or a list of those if this is a batch
    values = array.array("f", array.array("I", [b << 16 for b in array.array("H", self.tobytes())]).tobytes()).tolist()
    m = [[values[(b * self.rows + r) * self.cols:(b * self.rows + r + 1) * self.cols] for r in range(self.rows)] for b in range(self.batch)]
    return m if self.batch > 1 else m[0]

  def gather(self, ids):
    """Rows ids of this tensor, as a tensor: an embedding lookup. A data-movement kernel (see gather_kernel)."""
    assert self.batch == 1 and all(0 <= i < self.rows for i in ids)
    out = Tensor(self.rt, len(ids), self.cols)
    return out._run(_cached("gather", gather_kernel), out.addr, self.realize().addr, self.stride, self.offset, -(-self.cols // 32), len(ids), *ids)

  def _view(self, rows, cols, batch, offset, bstride=0, stride=None):
    v = Tensor(self.rt, rows, cols, "view", batch=batch)
    v.op, v.addr, v.owner = None, self.realize().addr, self.owner or self
    v.offset, v.bstride, v.stride = self.offset + offset, bstride, self.stride if stride is None else stride
    return v

  def __getitem__(self, key):
    """x[b], one matrix of a batch; or x[r0:r1, c0:c1] of a matrix, a view of whole tiles: starts on tile boundaries,
    and ends too unless at the tensor's end. No copy; kernels read it in place."""
    if isinstance(key, int): return self._view(self.rows, self.cols, 1, key * self.bstride)
    assert self.batch == 1
    (r0, r1, _), (c0, c1, _) = (k.indices(n) for k, n in zip(key, (self.rows, self.cols)))
    assert r0 % 32 == c0 % 32 == 0 and (r1 % 32 == 0 or r1 == self.rows) and (c1 % 32 == 0 or c1 == self.cols), "views are whole tiles"
    assert (r1 - r0 > 1 or self.rows == 1) and (c1 - c0 > 1 or self.cols == 1), "a dimension of 1 must fill its tile"
    return self._view(r1 - r0, c1 - c0, 1, r0 // 32 * self.stride + c0 // 32)

  def split(self, n, axis=1):
    """The matrix cut into n equal parts along axis (1: columns, 0: rows) as a batch of n, in place: attention's heads.
    The parts are whole tiles."""
    assert self.batch == 1
    if axis: assert self.cols % (32 * n) == 0; return self._view(self.rows, self.cols // n, n, 0, self.cols // n // 32)
    assert self.rows % (32 * n) == 0
    return self._view(self.rows // n, self.cols, n, 0, self.rows // n // 32 * self.stride)

  def merge(self):
    """The batch side by side as one matrix, the inverse of split: x.split(n).merge() is x. The kernel computing it
    writes each matrix into its columns (realize into a view); one already computed is copied there."""
    out = Tensor(self.rt, self.rows, self.batch * self.cols)
    (self if self.op not in (None, "T") else self + 0.0).realize(out.split(self.batch))  # a transpose cannot write a view
    return out

  @property
  def T(self):
    """The transpose (of each matrix of a batch). A matmul reads it as its B straight from this tensor, transposing
    tiles as they come in; anything else realizes it in a kernel of its own (transpose_kernel)."""
    return Tensor(self.rt, self.cols, self.rows, "T", (self,), self.batch)

  def realize(self, into=None):
    """Compute it now, into DRAM of its own, or into the view `into` (see merge). Returns it, now realized."""
    if self.op is None: return self
    if self.op == "T":
      assert into is None, "a transpose writes a tensor of its own"
      x, self.op, self.srcs = self.srcs[0].realize(), None, ()
      self.addr = self.rt.dram_alloc(-(-self.tiles // len(bh.DRAM)) * TILE)
      kernel = _cached(("transpose", x.rows, x.cols), lambda: transpose_kernel(x.rows, x.cols))
      return self._run(kernel, self.addr, x.addr, x.bstride if x.batch > 1 else 0, x.stride, x.offset)
    if any(t.op in COMBINE for t in _lazy(self, set())):  # rows are swept in passes, each recomputing what it reads:
      for t in [t for t in _lazy(self, set()) if t.op == "matmul" and t.srcs[0].cols > 256]: t.realize()  # long matmuls once
      # and an elementwise value that a reduction and something else both read: once, on as many workers as it has tiles
      reads = [id(s) for t in _lazy(self, set()) for s in t.srcs]
      for x in [t.srcs[0] for t in _lazy(self, set()) if t.op in COMBINE]:
        if reads.count(id(x)) > 1 and all(t.op not in COMBINE and t.op != "matmul" for t in _lazy(x, set())): x.realize()
    while _ops(self, set()) > 15:  # too big for one kernel (SEM_L1 counts 15): down the heaviest branch, the first part
      t = self                    # that fits goes first, on its own, so what is on top (a layer norm, say) stays whole
      while _ops(t, set()) > 15: t = max((s for s in t.srcs if s.op), key=lambda s: _ops(s, set()))
      t.realize()
    inputs = []
    e = _tree(self, inputs)
    key = (e, tuple(u.shape for u in inputs))
    kernel, units = _cached(key, lambda: fused_kernel(*key))
    if into: self.addr, self.owner, self.bstride, self.stride, self.offset = into.addr, into.owner or into, into.bstride, into.stride, into.offset
    else: self.addr = self.rt.dram_alloc(-(-self.tiles // len(bh.DRAM)) * TILE)
    self.op, self.srcs = None, ()
    args = [v for t in [self] + inputs for v in (t.addr, t.bstride if t.batch > 1 else 0, t.stride, t.offset)]
    return self._run(kernel, *args, units=units)

  def _run(self, kernel, *args, units=None):
    """Queue kernel over units of this tensor (tiles unless told), split evenly over as many workers as have any. Nothing waits for
    it: launches run in order, so the next kernel reads this result, and reading it back (tobytes) waits."""
    self.rt.launch(kernel, args, units or self.tiles)
    return self

  def _elementwise(self, op, other):
    """numpy's broadcasting: shapes match, or one side has 1 where the other does not. A number is a 1 x 1 tensor."""
    if not isinstance(other, Tensor): other = Tensor.full(self.rt, 1, 1, other)
    shape = []
    for a, b in ((self.batch, other.batch), (self.rows, other.rows), (self.cols, other.cols)):
      assert a == b or 1 in (a, b), f"cannot broadcast {self.shape} with {other.shape}"
      shape.append(max(a, b))
    return Tensor(self.rt, shape[1], shape[2], op, (self, other), shape[0])
  @property
  def shape(self): return (self.batch, self.rows, self.cols)
  def _unary(self, op): return Tensor(self.rt, self.rows, self.cols, op, (self,), self.batch)
  def relu(self): return self._unary("relu")
  def exp(self): return self._unary("exp")
  def reciprocal(self): return self._unary("recip")
  def rsqrt(self): return self._unary("rsqrt")
  def gelu(self):  # GPT-2's: 0.5 x (1 + tanh(z)), z = sqrt(2/pi) (x + 0.044715 x^3), which is x / (1 + e^-2z)
    k = 2 * math.sqrt(2 / math.pi)
    return self * (1 + (self * (self * self * (-0.044715 * k) - k)).exp()).reciprocal()
  def softmax(self):  # along each row, max first so exp stays in range
    e = (self - self.max(2)).exp()
    return e / e.sum(2)
  def layernorm(self, eps=1e-5):  # each row to mean 0 and variance 1
    c = self - self.sum(2) * (1 / self.cols)
    return c * ((c * c).sum(2) * (1 / self.cols) + eps).rsqrt()
  def sum(self, axis=None): return self._reduce("sum", axis)
  def max(self, axis=None): return self._reduce("max", axis)
  def _reduce(self, f, axis):
    """numpy's, keeping the axis as 1: over the batch (0, one matrix), down the rows (1, one row) or along each row
    (2, one column); over all three if None. Each axis is a kernel of its own, sweeping lines of tiles along it."""
    if axis is None: return self._reduce(f, 0)._reduce(f, 1)._reduce(f, 2)
    shape = [1 if ax == axis % 3 else s for ax, s in enumerate(self.shape)]
    return self if shape == list(self.shape) else Tensor(self.rt, shape[1], shape[2], f"{f}{axis % 3}", (self,), shape[0])
  def __add__(self, other): return self._elementwise("add", other)
  def __sub__(self, other): return self._elementwise("sub", other)
  def __mul__(self, other): return self._elementwise("mul", other)
  def __truediv__(self, other): return self * (1 / other if isinstance(other, (int, float)) else other.reciprocal())
  __radd__, __rmul__ = __add__, __mul__
  def __rsub__(self, other): return Tensor.full(self.rt, 1, 1, other) - self
  def __neg__(self): return self * -1.0

  def __matmul__(self, other):  # matrix by matrix through a batch; a batch of 1 repeats
    assert self.cols == other.rows and (self.batch == other.batch or 1 in (self.batch, other.batch)), f"{self.shape} @ {other.shape}"
    if self.cols == 1: return self * other  # (R, 1) @ (1, C) is broadcasting
    kt, tiles = self.cols // 32, -(-self.rows // 32) * -(-other.cols // 32)
    s = max([s for s in (2, 4, 8) if self.batch == other.batch == 1 and self.cols % (32 * s) == 0 and kt // s >= 4 and
             s * tiles <= len(self.rt.cores)], default=1)
    if s > 1:  # too few result tiles to keep the workers busy (a token's MLP): K cut in s, a batch, and the parts summed
      parts = self.split(s) @ other.split(s, 0)
      return sum((parts[i] for i in range(1, s)), parts[0])
    return Tensor(self.rt, self.rows, other.cols, "matmul", (self, other), max(self.batch, other.batch))

def _lazy(t, seen):  # the unrealized ops under t, t first (a matmul's operands and a transpose's are whole tensors)
  if t.op is None or id(t) in seen: return
  seen.add(id(t))
  yield t
  for s in () if t.op in ("matmul", "T") else t.srcs: yield from _lazy(s, seen)

def _ops(t, seen):  # how many ops a kernel for t would hold, at most (an SFPU op on another's result adds none)
  if t.op is None or id(t) in seen: return 0
  seen.add(id(t))
  return 1 + (0 if t.op in ("matmul", "T") else sum(_ops(s, seen) for s in t.srcs))

def _tree(t, inputs):
  """The expression a kernel for t computes (fused_kernel's), realized tensors as numbers into inputs. The kernel
  sweeps along the axis of t's reductions (the first found; any along another axis compute first, on their own)."""
  axis = next((FOLDS[u.op] for u in _lazy(t, set()) if u.op in COMBINE), 2)
  return _expr(t, t.shape, inputs, [axis, None, set()])

def _ext(t, ax): return t.shape[ax] if ax == 0 else -(-t.shape[ax] // 32)  # t's extent along an axis: matrices, or tiles

def _expr(t, shape, inputs, kernel):
  """The tree of unrealized ops under t, for a kernel of shape (t's). kernel: [the axis it sweeps, the sweep's length
  (in tiles; the first seen, if t is one tile along it), the reductions it holds (two accumulators)]. Computed first, on
  their own: an op of 1 along an axis the kernel keeps (repeated there: compute it once), and a reduction along another
  axis, of another length, or past the second."""
  if t.op is None:
    if all(t is not u for u in inputs): inputs.append(t)
    return next(i for i, u in enumerate(inputs) if u is t)
  axis = kernel[0]
  if any(t.shape[ax] != shape[ax] for ax in range(3) if ax != axis): return _expr(t.realize(), shape, inputs, kernel)
  if t.op in COMBINE:
    if FOLDS[t.op] != axis: return _expr(t.realize(), shape, inputs, kernel)
    n = _ext(t.srcs[0], axis)
    if n > 1 and kernel[1] is None: kernel[1] = n
    if (n > 1 and n != kernel[1]) or (id(t) not in kernel[2] and len(kernel[2]) == len(ACCS)): return _expr(t.realize(), shape, inputs, kernel)
    kernel[2].add(id(t))
  if t.op == "T": return _expr(t.realize(), shape, inputs, kernel)  # a kernel of its own, unless a matmul's B (below)
  if t.op == "matmul":  # its operands are whole matrices, not tiles: realize them on their own first
    a, b = t.srcs
    b = ("T", _expr(b.srcs[0].realize(), shape, inputs, kernel)) if b.op == "T" else _expr(b.realize(), shape, inputs, kernel)
    return ("matmul", _expr(a.realize(), shape, inputs, kernel), b)
  return (t.op, *(_expr(s, shape, inputs, kernel) for s in t.srcs))

kernels = {}  # (expression, input shapes) -> (image, units a matrix), or a kernel's name and shape -> image
def _cached(key, compile):
  if key not in kernels: kernels[key] = compile()
  return kernels[key]
