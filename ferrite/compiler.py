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

ITEMSIZE = {"bf16": 2, "fp32": 4, "int32": 4}  # the dtypes, by bytes a value; a tile is 1024 values
def tile_bytes(dtype): return 1024 * ITEMSIZE[dtype]  # a bf16 tile is one SrcA/SrcB bank (64 rows of 16), 64 Dst rows
def slot_base(dtype): return DATA - tile_bytes(dtype)  # L1 tile slots of a kernel computing in dtype: slot i at base + i *
                                                        # its tile, so slot 0 is (part of) the runtime's zeros. The unpackers'
                                                        # Z and the packer's channel 1 Y counters pick them
OUTS = 0x17E000  # the result of a kernel that packs in another dtype than it computes in: two 4 KiB tiles, one per frame
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

FORMAT = {"bf16": bh.BF16, "fp32": bh.FP32, "int32": bh.INT32}
def kernel_config(dtype="bf16", out="bf16", transposing=False):
  """A kernel's backend config. Unpackers read one whole tile each (1024 datums): bf16 into a SrcA/SrcB bank; fp32
  straight into Dst (unpacker 0 in context 1, where the 32-bit Dst row is channel 1's Y). Z picks the slot (the Z stride
  is one tile). The FPU writes Dst in fp32. The packer reads 4 Dst rows per PACR (its Dst row stride is in bytes of out)
  and writes them as out (fp32 to bf16 rounds); its channel 1 Y picks the slot (that Y stride is one tile of out). Base
  addresses are in 16-byte units, minus 1, and unpacker 0's rows are biased by 4 (64 datums). transposing: unpacker 0
  transposes each 16x16 face into SrcA."""
  base, fmt, pack = slot_base(dtype) // 16 - 1, FORMAT[dtype], FORMAT[out]
  return config(Dstacc=pack, Fp32_enabled=1, Zero_Flag_disabled_src=0, Read_32b_data=1, Edge_mask=0xFFFF, pack_reads_per_xy_plane=16,
                PCK0_Ystride=16 * ITEMSIZE[out], PCK0_ZW=0, PCK0_Ystride_1=tile_bytes(out), PCK0_ZW_1=0, PCK0_Base_1=0, PCK_dest_offset=0,
                L1_Dest_addr=(slot_base(dtype) if out == dtype else OUTS) // 16 - 1, Disable_zero_compress=1, Pack_out_format=pack,
                Pack_in_format=pack, UNP0_add_dest_addr_cntr=0, UNP0_Ystride_1=64, UNP0_ZW_1=0, UNP1_ZW_1=0,
                SEC0_format=fmt, SEC0_x_dim=1024, SEC0_y_dim=1, SEC0_z_dim=1, SEC0_out_format=fmt, SEC0_uncompressed=1, SEC0_base=base,
                SEC0_dest_addr=64, SEC0_tile_x_dim=1024, SEC0_offset=0, SEC0_Haloize_mode=int(transposing),
                SEC0_uncompressed_1=1, SEC0_to_dst_1=1, SEC0_base_1=base, SEC0_dest_addr_1=64, SEC0_tile_x_dim_1=1024, SEC0_offset_1=0,
                SEC1_format=fmt, SEC1_x_dim=1024, SEC1_y_dim=1, SEC1_z_dim=1, SEC1_out_format=fmt, SEC1_uncompressed=1, SEC1_base=base)

# The math thread's address modes, as (SrcA step, SrcB step) and (Dst step, fidelity step): 0-1 for elementwise ops,
# 2-3 for SFPU loads and stores (the store steps Dst by 2), 4-7 for matmul, so ops mix with no rewrites between them.
ADDR_MODES = {0: ((8, 8), (8, 0)), 1: ((0, 0), (0, 1)), 2: ((0, 0), (0, 0)), 3: ((0, 0), (2, 0)),
              4: ((16, 0), (0, 0)), 5: ((48, 8), (0, 0)), 6: ((16, 8), (0, 0)), 7: ((16, 8), (0, 1))}
def thread(dst=False):  # a thread's config, written since reset values are undocumented; dst: it unpacks into Dst
  return [ins("SETC16", setc16_reg=r, setc16_value=v) for r, v in
          ({r: 0 for r in bh.THREAD_CFG.values()} | {bh.ADDR_MOD_PACK: 4} | {bh.ADDR_MOD_BIAS + m: 0 for m in range(8)} |  # the packer's
           {bh.ADDR_MOD_AB + m: a | b << 8 for m, ((a, b), _) in ADDR_MODES.items()} |    # mode 0 moves on 4 Dst rows a PACR
           {bh.ADDR_MOD_DST + m: d | f << bh.FIDELITY_INCR for m, (_, (d, f)) in ADDR_MODES.items()} |
           ({bh.THREAD_CFG["UNPACK_MISC"]: 1} if dst else {bh.THREAD_CFG["SRCA_SET"]: bh.SRCA_SET_BY_ADDR})).items()]

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
def counters(mask, n):  # a thread's address counters for these units to 0, each instruction moving n datums
  return [ins("SETADCXY", cnt_set_mask=mask, bit_mask=0b1111), ins("SETADCZW", cnt_set_mask=mask, bit_mask=0b1111),
          ins("SETADCXX", cnt_set_mask=mask, x_end2=n - 1)]
SETUP_UNPACK = [ins("SEMINIT", sem_sel=SEM_DST, max_value=2), ins("SEMINIT", sem_sel=SEM_PACKED, max_value=15),
                ins("SEMINIT", sem_sel=SEM_L1, max_value=15), ins("SEMINIT", sem_sel=SEM_UNPACKED, max_value=15),
                *[ins("SEMINIT", sem_sel=sem, init_value=1, max_value=1) for _, sem in ACCS], *counters(UNPACKERS, 1024)]
SETUP_PACK = counters(bh.PACK, 16) + record(PACRS, [ins("PACR") for _ in range(15)])  # 16 datums a Dst row
def setadc(unpacker, slot): return ins("SETADC", cnt_set_mask=unpacker, dimension_index=2, value=slot)  # its Z: the slot
def unpack(a, b):  # slot a into SrcA and slot b into SrcB, each bank then handed to the FPU
  return [setadc(bh.UNPACK0, a), setadc(bh.UNPACK1, b), ins("UNPACR", last=1, set_dat_valid=1, ovrd_thread_id=1),
          ins("UNPACR", last=1, set_dat_valid=1, ovrd_thread_id=1, unpack_block_selection=1)]
REPORT = [ins("STALLWAIT", stall_res=bh.STALL_SYNC, wait_res=bh.WAIT_UNPACK), ins("SEMPOST", sem_sel=SEM_UNPACKED)]
def fill(slot, row):  # an fp32 tile from an L1 slot straight into Dst rows row on, by unpacker 0 in context 1 (the math thread's)
  return [setadc(bh.UNPACK0, slot), ins("SETADC", cnt_set_mask=bh.UNPACK0, dimension_index=1, channel_index=1, value=row),
          ins("UNPACR", last=1, ovrd_thread_id=1)]
FILLED = [ins("STALLWAIT", stall_res=bh.STALL_SFPU | bh.STALL_SYNC, wait_res=bh.WAIT_UNPACK0)]  # the SFPU and the sync unit wait for the fills
def haloize(on):  # unpacker 0 transposes each 16x16 face into SrcA from its next unpack on (or stops): the config write
  word, bit, _ = bh.TENSIX_CFG_FIELDS["SEC0_Haloize_mode"]  # waits for unpacker 0's unpacks, the next unpack for the write
  return [ins("STALLWAIT", stall_res=bh.STALL_CFG, wait_res=bh.WAIT_UNPACK0),
          ins(f"RMWCIB{bit // 8}", cfg_reg_addr=word, data=on << bit % 8, mask=1 << bit % 8),
          ins("STALLWAIT", stall_res=bh.STALL_UNPACK, wait_res=bh.WAIT_CFG)]
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
def sfp_load(r=0, at=0): return ins("SFPLOAD", vd=r, mod0=3, addr_mode=2, imm10=at)
def sfp_store(r, mod0=3): return ins("SFPSTORE", vd=r, mod0=mod0, addr_mode=3)
RAW = 4  # SFPSTORE's INT32 mode stores the bits as they are; its FP32 mode flushes denormals, the small positive integers
def sfp_bits(r, bits): return [ins("SFPLOADI", vd=r, mod0=8, imm16=bits >> 16), ins("SFPLOADI", vd=r, mod0=10, imm16=bits & 0xFFFF)]
def sfp_const(r, x): return sfp_bits(r, struct.unpack("<I", struct.pack("<f", x))[0])  # a full fp32 constant into LReg r
def sfp_bf16(r, x):  # one word, for a constant bf16 holds exactly
  assert from_bf16(to_bf16(x)) == x, f"{x} is not a bf16"
  return ins("SFPLOADI", vd=r, mod0=0, imm16=to_bf16(x))
def mad(d, a, b, c, neg=0): return ins("SFPMAD", vd=d, va=a, vb=b, vc=c, mod1=neg)  # d = a * b + c; neg 1: -a, 2: -c
def mul(d, a, b): return ins("SFPMUL", vd=d, va=a, vb=b, vc=9)
ZERO, ONE, MINUS_ONE = 9, 10, 11  # constant LRegs
FLAGS_ON, FLAGS_RESET, FLAGS_OFF = ins("SFPENCC", mod1=10, imm2=3), ins("SFPENCC", mod1=0), ins("SFPENCC", mod1=10, imm2=2)  # lane flags
NAN, INF = 0x7FC00000, 0x7F800000
def at_least(e, n, t): return ins("SFPIADD", vd=t, vc=e, mod1=9, imm12=-n & 0xFFF)  # flags on: keep the lanes where integer LReg e >= n
def below(e, n, t): return ins("SFPIADD", vd=t, vc=e, mod1=1, imm12=-n & 0xFFF)  # or < n (LReg t is overwritten)
CONSTS = {0.0: ZERO, 1.0: ONE, -1.0: MINUS_ONE}
def poly(d, x, cs, temp):  # LReg d = cs[0] + cs[1] x + ... by Horner, each coefficient into temp as it is needed (0, +-1 are LRegs)
  words = sfp_const(d, cs[-1])
  for c in cs[-2::-1]: words += ([] if c in CONSTS else sfp_const(temp, c)) + [mad(d, d, x, CONSTS.get(c, temp))]
  return words

# The transcendentals reach fp32's own accuracy (about 1e-7), so an fp32 tensor gets numpy's float32 within a step or two:
# each reduces x to a small range by exact steps, and a polynomial (fit offline to minimax) does the rest.
EXP_P = [1.0, 1.0, 0.49999991, 0.16666421, 0.041668355, 0.008374773, 0.001382954]  # e^r on [-ln2/2, ln2/2], 8e-8
LOG2_Q = [1.442695, -0.72134733, 0.4809106, -0.36070517, 0.2879189, -0.23891294, 0.21565668, -0.20746617, 0.12623915]  # 1.3e-7
SIN_P = [1.0, -0.16666658, 0.0083330506, -0.00019809045, 2.6051646e-06]  # sin(r) / r in r^2, r in [-pi/2, pi/2], 1.4e-7

def exp(base2):
  """2^x, or e^x: n = round(t), t = x log2(e) (adding and taking away 1.5 2^23 rounds to an integer, whose bits it
  leaves in the sum's low ones), r = (x - n) ln 2, or for e^x x - n ln 2 in two parts (Cody-Waite: n times 0.693359375 is
  exact), e^r by a polynomial, times 2^floor(n/2) and 2^ceil(n/2), each built from its exponent bits: the multiplies
  overflow to inf, or flush to 0, exactly where the result does. x is first held to where |t| <= 150 (NaN to an end)."""
  (lim, t), ln2 = (150.0, 0) if base2 else (104.0, 5), math.log(2)
  return (sfp_const(1, ln2) if base2 else sfp_const(1, 1 / ln2) + sfp_const(2, 0.693359375) + sfp_const(3, ln2 - 0.693359375),
          [sfp_load(), sfp_bf16(4, -lim), ins("SFPSWAP", vd=4, vc=0, mod1=1), sfp_bf16(4, lim), ins("SFPSWAP", vd=0, vc=4, mod1=1),  # x within +-lim
           *([] if base2 else [mul(5, 0, 1)]), sfp_bf16(6, 1.5 * 2**23), mad(4, t, ONE, 6), mad(7, 4, ONE, 6, neg=2),  # LReg 7: n
           ins("SFPIADD", vd=6, vc=4, mod1=6),  # LReg 6: n as an integer, bits(t + 1.5 2^23) - bits(1.5 2^23)
           *([mad(5, 7, ONE, 0, neg=1), mul(5, 5, 1)] if base2 else [mad(5, 7, 2, 0, neg=1), mad(5, 7, 3, 5, neg=1)]),  # LReg 5: r
           *poly(4, 5, EXP_P, 7), ins("SFPSHFT", vd=0, vc=6, mod1=7, imm12=-1 & 0xFFF), ins("SFPIADD", vd=6, vc=6, mod1=5, imm12=1),
           ins("SFPSHFT", vd=6, mod1=3, imm12=-1 & 0xFFF),  # LReg 0, 6: floor(n/2), ceil(n/2), then each 2^ that, by its bits
           *[w for r in (0, 6) for w in (ins("SFPIADD", vd=r, vc=r, mod1=5, imm12=127), ins("SFPSHFT", vd=r, mod1=1, imm12=23), mul(4, 4, r))],
           sfp_store(4)])

def log(scale):
  """log2(x) scale: x = 2^e m with m in [sqrt(1/2), sqrt(2)) (from [1, 2), halved and e one more where above sqrt 2), so
  log2(x) = e + u q(u), u = m - 1, q a polynomial; relative near 1, where u is small and exact. Then the edges: inf and
  NaN give themselves, a negative number NaN, zero (and a denormal, flushed) -inf."""
  return (sfp_const(5, math.sqrt(2)) + (sfp_const(7, scale) if scale != 1 else []),
          [sfp_load(), ins("SFPEXEXP", vd=1, vc=0, mod1=1), ins("SFPSETEXP", vd=2, vc=0, mod1=1, imm12=127),  # LReg 1: e, biased; 2: m
           FLAGS_ON, ins("SFPGT", vd=2, vc=5, mod1=1), ins("SFPSETEXP", vd=2, vc=2, mod1=1, imm12=126), ins("SFPIADD", vd=1, vc=1, mod1=5, imm12=1),
           FLAGS_OFF, ins("SFPCAST", vd=1, vc=1), sfp_bf16(6, -127.0), mad(1, 1, ONE, 6), mad(2, 2, ONE, MINUS_ONE),
           *poly(3, 2, LOG2_Q, 6), mad(3, 3, 2, 1), *([mul(3, 3, 7)] if scale != 1 else []),
           ins("SFPEXEXP", vd=1, vc=0, mod1=1), FLAGS_ON, at_least(1, 255, 4), ins("SFPMOV", vd=3, vc=0), FLAGS_RESET,
           ins("SFPSETCC", vc=0), *sfp_bits(3, NAN), FLAGS_RESET, below(1, 1, 4), *sfp_bits(3, INF | 1 << 31), FLAGS_OFF, sfp_store(3)])

def sin(phase):
  """sin(x + pi phase): k = round(x / pi + phase), r = x - k pi in two parts (Cody-Waite: k times 3.140625 is exact, and
  so is what is left of x, being within a factor of 2 of it) + pi phase, also in two, so r is in [-pi/2, pi/2] with no
  cancellation near a zero; sin(r) = r p(r^2), p a polynomial, and k's low bit, moved to the sign, flips it. phase 0.5
  makes it cos. From |x| = 2^24 on, where an fp32 x is a whole number of no particular turn and the rounding trick
  fails, it gives 0; inf and NaN give NaN."""
  hi = 1.5707964  # pi/2 in fp32, and what it leaves out
  return (sfp_const(1, 1 / math.pi),
          [sfp_load(), sfp_bf16(7, phase), mad(4, 0, 1, 7), sfp_bf16(5, 1.5 * 2**23), mad(6, 4, ONE, 5), mad(4, 6, ONE, 5, neg=2),  # LReg 4: k
           ins("SFPIADD", vd=5, vc=6, mod1=6), sfp_bf16(2, 3.140625), mad(6, 4, 2, 0, neg=1),  # LReg 5: k as an integer; 6: x - 3.140625 k
           *(sfp_const(3, hi) + [mad(6, 6, ONE, 3)] if phase else []), *sfp_const(3, math.pi - 3.140625), mad(6, 4, 3, 6, neg=1),
           *(sfp_const(3, math.pi / 2 - hi) + [mad(6, 6, ONE, 3)] if phase else []),  # LReg 6: r
           ins("SFPSHFT", vd=5, mod1=1, imm12=31), ins("SFPEXEXP", vd=0, vc=0, mod1=1), mul(7, 6, 6), *poly(4, 7, SIN_P, 3), mul(4, 4, 6),
           ins("SFPXOR", vd=4, vc=5), FLAGS_ON, at_least(0, 127 + 24, 5), ins("SFPMOV", vd=4, vc=ZERO), FLAGS_RESET, at_least(0, 255, 5),
           *sfp_bits(4, NAN), FLAGS_OFF, sfp_store(4)])

SFPU = {
  # max(x, 0): SFPSWAP's min/max mode leaves the smaller value in vd and the larger in vc; a pick, stored raw for int32
  "relu": ([], [sfp_load(), ins("SFPMOV", vd=1, vc=ZERO), ins("SFPSWAP", vd=1, vc=0, mod1=1), sfp_store(0, RAW)]),
  "exp": exp(False), "exp2": exp(True), "log": log(math.log(2)), "log2": log(1), "sin": sin(0.0), "cos": sin(0.5),
  # 1/x: SFPARECIP's 7-bit estimate, then two Newton steps y += y * (1 - x * y). Only where x is a normal number:
  # 0 and denormals (estimate inf) and inf (estimate 0) would turn into 0 * inf = NaN, so those lanes keep the estimate
  "recip": ([], [sfp_load(), ins("SFPARECIP", vd=1, vc=0), ins("SFPEXEXP", vd=3, vc=0),  # LReg 3: unbiased exponent
                 FLAGS_ON, below(3, 128, 4), at_least(3, -126, 4),  # the lanes with exponent in [-126, 128)
                 mad(2, 0, 1, ONE, neg=1), mad(1, 1, 2, 1), mad(2, 0, 1, ONE, neg=1), mad(1, 1, 2, 1), FLAGS_OFF, sfp_store(1)]),
  # sqrt(x) = x / sqrt(x): 1/sqrt(x) by the bit trick (0x5F3759DF - bits / 2, within 3.5%) and two Newton steps
  # y *= 1.5 - x/2 y^2, then times x (x + 0 first, so -0 is 0). 0, inf and NaN give themselves, a negative number NaN
  "sqrt": (sfp_const(2, struct.unpack("<f", struct.pack("<I", 0x5F3759DF))[0]) + sfp_const(3, 1.5) + sfp_const(4, 0.5),
           [sfp_load(), mad(0, 0, ONE, ZERO), ins("SFPSHFT", vd=1, vc=0, mod1=5, imm12=-1 & 0xFFF),  # LReg 1 = bits of x >> 1
            ins("SFPIADD", vd=1, vc=2, mod1=6), mul(5, 0, 4)] +  # LReg 1 = magic - LReg 1 as integers; LReg 5 = x/2
           [w for _ in range(2) for w in (mul(6, 1, 1), mad(6, 5, 6, 3, neg=1), mul(1, 1, 6))] +
           [mul(1, 1, 0), ins("SFPEXEXP", vd=6, vc=0, mod1=1), FLAGS_ON, at_least(6, 255, 5), ins("SFPMOV", vd=1, vc=0), FLAGS_RESET,
            below(6, 1, 5), ins("SFPMOV", vd=1, vc=0), FLAGS_RESET, ins("SFPSETCC", vc=0), *sfp_bits(1, NAN), FLAGS_OFF, sfp_store(1)]),
}
# Binary ops on the SFPU: x and y each added to zero into Dst (x at the op's Dst half, y 128 rows on), loaded into LRegs
# 0 and 1, the result stored over x. Comparisons give 1.0 or 0 (all ones or 0 from SFPGT, and 1.0's bits), after x + 0
# makes -0 into 0. keep and drop are the halves of where: x if c else 0, and y if not c else 0 (c: not 0), as bits, so
# that an inf in the one not picked does not make a NaN, as c * x would.
def binary(*body): return [sfp_load(0), sfp_load(1, 128), *body]
BINARY = {"max", "cmplt", "cmpne", "keep", "drop"}
FP32 = {"add": "fadd", "sub": "fsub", "mul": "fmul"}  # in an fp32 kernel the SFPU does these too, one rounding each, as fp32 does
INT32 = {"add": "iadd", "sub": "isub", "max": "imax", "cmplt": "icmplt", "cmpne": "icmpne", "keep": "keep", "drop": "drop"}  # what int32 does itself
CASTS = {("int32", "fp32"): "itof", ("int32", "bf16"): "itof", ("fp32", "int32"): "ftoi", ("bf16", "int32"): "ftoi"}  # SFPU programs of a cast
SFPU |= {
  "max": ([], binary(ins("SFPSWAP", vd=1, vc=0, mod1=1), sfp_store(0))),
  "cmplt": ([], binary(mad(0, 0, ONE, ZERO), mad(1, 1, ONE, ZERO), ins("SFPGT", vd=1, vc=0, mod1=8), ins("SFPAND", vd=1, vc=ONE), sfp_store(1))),
  "cmpne": ([], binary(mad(0, 0, ONE, ZERO), mad(1, 1, ONE, ZERO), ins("SFPMOV", vd=2, vc=0), ins("SFPGT", vd=2, vc=1, mod1=8),
                       ins("SFPGT", vd=1, vc=0, mod1=8), ins("SFPOR", vd=1, vc=2), ins("SFPAND", vd=1, vc=ONE), sfp_store(1))),
  "keep": ([], binary(ins("SFPABS", vd=2, vc=0, mod1=1), ins("SFPGT", vd=2, vc=ZERO, mod1=8), ins("SFPAND", vd=1, vc=2), sfp_store(1, RAW))),
  "drop": ([], binary(ins("SFPABS", vd=2, vc=0, mod1=1), ins("SFPGT", vd=2, vc=ZERO, mod1=8), ins("SFPNOT", vd=2, vc=2),
                      ins("SFPAND", vd=1, vc=2), sfp_store(1, RAW))),
  "fadd": ([], binary(mad(0, 0, ONE, 1), sfp_store(0))), "fsub": ([], binary(mad(0, 1, ONE, 0, neg=1), sfp_store(0))),
  "fmul": ([], binary(mul(0, 0, 1), sfp_store(0))),
  # int32, two's complement in L1, Dst and LRegs alike (they move it raw). Ordering goes through sign-magnitude
  # (SFPCAST 3, its own inverse), the order SFPGT and SFPSWAP use; a comparison is 1, not 1.0
  "iadd": ([], binary(ins("SFPIADD", vd=0, vc=1, mod1=4), sfp_store(0, RAW))),
  "isub": ([], binary(ins("SFPIADD", vd=1, vc=0, mod1=6), sfp_store(1, RAW))),
  "imax": ([], binary(*[ins("SFPCAST", vd=r, vc=r, mod1=3) for r in (0, 1)], ins("SFPSWAP", vd=1, vc=0, mod1=1), ins("SFPCAST", vd=0, vc=0, mod1=3),
                      sfp_store(0, RAW))),
  "icmplt": ([], binary(*[ins("SFPCAST", vd=r, vc=r, mod1=3) for r in (0, 1)], ins("SFPGT", vd=1, vc=0, mod1=8),
                        ins("SFPLOADI", vd=2, mod0=2, imm16=1), ins("SFPAND", vd=1, vc=2), sfp_store(1, RAW))),
  "icmpne": ([], binary(ins("SFPMOV", vd=2, vc=0), ins("SFPGT", vd=2, vc=1, mod1=8), ins("SFPGT", vd=1, vc=0, mod1=8), ins("SFPOR", vd=1, vc=2),
                        ins("SFPLOADI", vd=2, mod0=2, imm16=1), ins("SFPAND", vd=1, vc=2), sfp_store(1, RAW))),
  # casts: an integer to sign-magnitude, then to float (to nearest, exact below 2^24); a float to the integer toward 0,
  # by shifting its mantissa (hidden bit in) by its exponent - 23: 0 where the exponent is negative, negated if negative
  "itof": ([], [sfp_load(), ins("SFPCAST", vd=0, vc=0, mod1=3), ins("SFPCAST", vd=0, vc=0), sfp_store(0)]),
  "ftoi": ([], [sfp_load(), ins("SFPEXEXP", vd=1, vc=0), ins("SFPEXMAN", vd=2, vc=0), ins("SFPIADD", vd=1, vc=1, mod1=5, imm12=-23 & 0xFFF),
                ins("SFPSHFT", vd=2, vc=1), FLAGS_ON, below(1, -23, 4), ins("SFPMOV", vd=2, vc=ZERO), FLAGS_RESET, ins("SFPSETCC", vc=0),
                ins("SFPIADD", vd=2, vc=ZERO, mod1=6), FLAGS_OFF, sfp_store(2, RAW)]),
}
# Reductions along an axis (0: the batch, to one matrix; 1: rows, to one row; 2: columns, to one column): each tile
# along it is x + 0 into Dst (the first in the result's half, the rest 128 rows further on, where the SFPU folds them
# in). Then each row or column of the tile is folded across its 32 and the result written to all of them, so the tile
# broadcasts along that axis as is (over the batch, the tiles were folded as they came and there is nothing left).
FOLDS = {f"{f}{ax}": ax for f in ("sum", "max") for ax in range(3)}  # kind -> the axis it folds
def fold(kind, d, t):  # LReg d = d op t (for max, t is left with the min)
  return ins("SFPSWAP", vd=t, vc=d, mod1=1) if kind[:3] == "max" else mad(d, t, ONE, d)
COMBINE = {k: binary(fold(k, 0, 1), sfp_store(0)) for k in FOLDS}
def across_rows(kind):
  """Fold every tile row across its 32 columns. Row r of a tile is Dst row (r // 16) * 32 + r % 16 (columns 0-15)
  and 16 rows on (16-31); an SFPLOAD takes 4 such rows' even or odd columns, one row per 8-lane group. Folding the
  four loads leaves 8 lanes a row, and rotating the group by 1, 2 and 4 lanes and folding each time leaves the
  row's result in every lane, which goes back to all four places."""
  rot = lambda d, s: ins("SFPSHFT2", vd=d, vc=s, mod1=3)
  p = [ins("STALLWAIT", stall_res=bh.STALL_SFPU, wait_res=bh.WAIT_FPU), ins("SETRWC", bit_mask=0b0100)]
  for base in (0, 4, 8, 12, 32, 36, 40, 44):
    at = [base, base + 2, base + 16, base + 18]  # even and odd columns, left and right face
    p += [sfp_load(r, a) for r, a in enumerate(at)]
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
    p += [sfp_load(0, at[0])]
    p += [w for a in at[1:] for w in (sfp_load(1, a), fold(kind, 0, 1))]
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
    if t < 8: p += [FLAGS_ON, at_least(7, 2 * t, 6),  # lanes t+
                    ins("SFPSTORE", vd=fill, mod0=3, addr_mode=2, imm10=d + base + 16 * right + 2 * odd)]  # right faces: 16 rows on
  return p + [FLAGS_OFF]
def mask_rows(kind, m, d=0):
  """Rows m and on of the tile at Dst row d to the identity, past a tensor's last row. A store covers 4 rows, lane l in
  row l // 8 of them, and LReg 15 holds 2l, so LReg 15 - 16k >= 0 picks the rows from k on."""
  p, fill = masking(kind)
  for r in range(0, 32, 4):  # tile rows r..r+3: Dst rows (r // 16) * 32 + r % 16 on, left face, and 16 more, right face
    k = max(0, m - r)
    if k < 4: p += [w for o in (0, 2, 16, 18) for w in (FLAGS_ON, at_least(15, 16 * k, 6),
                                                     ins("SFPSTORE", vd=fill, mod0=3, addr_mode=2, imm10=d + r // 16 * 32 + r % 16 + o))]
  return p + [FLAGS_OFF]
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
def setup_math(slots): return [FLAGS_OFF] + [w for name, at in slots.items() for w in record(at, MATH_RUNS[name])]

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
def reduce_step(kind, slots, acc, first, edge=0, src=None):
  """One tile of a reduction, which comes into Dst as SrcA = the tile plus SrcB = zeros, or for fp32 from slot src by a
  fill: the first lands in the accumulator once it is free; later ones land in scratch and the SFPU folds them in.
  edge: how many of the tile's rows or columns along the axis are the tensor's, if it is the line's last and cut short."""
  row, sem = ACCS[acc]
  mask = lambda d: MASK[FOLDS[kind]](kind, edge, d) if edge else []
  put = lambda d: fill(src, row + d) + FILLED if src is not None else elementwise("add", slots, dst=d)
  if first: return ([ins("SEMWAIT", stall_res=bh.STALL_ALL, sem_sel=sem, wait_sem_cond=1), ins("SEMGET", sem_sel=sem), at(row)] +
                    put(0) + mask(0))
  return ([at(row)] + put(128) + mask(128) +
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
static unsigned posted, drained;  // intermediates owed to L1, and those waited for
static void drain(void) {{ for (; drained < posted; drained++) {{ {tt(TAKE_L1)} }} }}  // until everything packed so far has landed
"""
def _pushes(words): return " ".join(f"push_word({w:#x});" for w in words)  # straight-line, with no table to load from

def _brisc(total, body, dtype="bf16", out="bf16"):  # moves tiles: inputs into L1 ahead of the unpackers, results home behind the packer
  tb, to = tile_bytes(dtype), tile_bytes(out)
  return SYNC_WORDS + f"""
#define SLOT(i) ({slot_base(dtype):#x} + (i) * {tb})  // where the unpackers read slot i
#define OUT_SLOT(i) ({slot_base(dtype) if out == dtype else OUTS:#x} + (i) * {to})  // and where the packer writes it
{_array("bank", [x | y << 6 for x, y in bh.DRAM])}
static unsigned fetched, events, frame_done[2], stream_done[2], steps, sent, out, rows[16];  // rows: i + 1 of what each place in L1 holds (fresh)
// tile t of a tensor at DRAM address base: bank t % 8, row t / 8 of the bank
#define BANK(t) bank[(t) % 8] & 63, bank[(t) % 8] >> 6  // the NoC x, y of tile t's bank
static void get(unsigned base, unsigned t, unsigned slot) {{ noc_read(BANK(t), base + t / 8 * {tb}, SLOT(slot), {tb}); }}
// the same with its two off-diagonal faces swapped, half of transposing it (unpacker 0 does the rest)
static void flipped(unsigned base, unsigned t, unsigned slot) {{
  for (unsigned f = 0; f < 4; f++)
    noc_read(BANK(t), base + t / 8 * {tb} + f * {tb // 4}, SLOT(slot) + (f == 1 ? 2 : f == 2 ? 1 : f) * {tb // 4}, {tb // 4});
}}
static void ready(void) {{ noc_wait(); SYNC[FETCHED] = ++fetched; }}
static void send(void) {{  // the results packed so far go home, in the order they were packed
  for (; sent < peek(PACKED); SYNC[SENT] = ++sent) {{
    unsigned slot = peek(RING + sent % 4 * 2), t = peek(RING + sent % 4 * 2 + 1);
    noc_write(OUT_SLOT(slot), BANK(t), out + t / 8 * {to}, {to}); noc_wait();
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

def _unpacker(body, cfg):  # TRISC0: the backend config and semaphores, then unpacking, each tile once BRISC has fetched it
  return TRISC + f"""
#define CFG ((volatile unsigned *){bh.TENSIX_CFG:#x})
static unsigned needed, relayed, steps;
static void relay(void) {{ for (; SEM({SEM_UNPACKED}); SYNC[UNPACKED] = ++relayed) SEM({SEM_UNPACKED}) = 1; }}  // the unpack reports, for BRISC
static void fetched(void) {{ needed++; do relay(); while (peek(FETCHED) < needed); }}  // until BRISC has fetched the next tiles
void kmain(volatile unsigned *a) {{
  {" ".join(f"CFG[{i}] = {v:#x};" for i, v in cfg.items())}
  asm volatile("fence" ::: "memory");  // the config writes are out; a read from the same block is processed after them
  (void)CFG[{next(iter(cfg))}];
  asm volatile("fence" ::: "memory");  // and has answered, so no instruction pushed below can see them stale
  {_pushes(thread() + SETUP_UNPACK)}
  tensix_idle();  // the semaphores are set before the other threads can touch them
  SYNC[READY] = 1;
  unsigned tiles = 0;
{body}
  while (relayed < needed) relay();  // every report taken, whichever thread unpacked
}}"""

def _math(slots, body, dst=False):  # TRISC1; dst: it unpacks into Dst itself, so it waits for BRISC's fetches too
  return TRISC + f"""
static unsigned arrived_;
static void arrived(void) {{ arrived_++; while (peek(FETCHED) < arrived_) {{}} }}  // until BRISC has fetched the next tiles
void kmain(volatile unsigned *a) {{
  {_pushes(thread(dst) + (counters(bh.UNPACK0, 1024) if dst else []) + setup_math(slots))}
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
  {_pushes(thread() + SETUP_PACK)}
  while (!peek(READY)) {{}}
  unsigned tiles = 0, h = 0;  // h: the Dst half the next op gets
{body}
  while (relayed < outputs) relay();
}}"""

def fused_kernel(expr, shapes, dtype="bf16", out=None):
  """A kernel for an expression: nested tuples, (op, x, y) for add, sub, mul and the binary SFPU ops, (op, x) for the
  unary ones, the reductions (FOLDS: sum0, max2, ...) and cast, ("matmul", a, b) where a is an input and b an input or
  ("T", input), read transposed, and plain numbers for inputs. shapes: each input's (batch, rows, columns), laid out as
  Tensor says; a dimension of 1 repeats across the others (broadcasting). dtype: what the inputs are and the kernel
  computes in; out: what it writes (only a kernel of one op, a cast or a matmul, writes another).
  bf16 goes through SrcA and SrcB and the FPU. fp32 and int32 cannot (they hold 19 bits), so the math thread puts each
  operand straight into Dst (fill) and the SFPU does all the math, add and sub too; there is no matmul.
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
  out, dst = out or dtype, dtype != "bf16"  # dst: the math thread fills Dst itself and the SFPU does the math
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
    elif e[0] == "cast":  # into Dst and out again in the other format, through an SFPU program between floats and ints
      x = visit(e[1])
      op = dict(kind="add", x=x, y="zero", shape=size(x), posts=[CASTS[dtype, out]] if (dtype, out) in CASTS else [])
    elif e[0] in SFPU and e[0] not in BINARY:
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
  assert out == dtype or len(ops) == 1, "only a kernel of one op packs in another dtype"
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
  sfpu_name = lambda op: {"fp32": FP32.get(op["kind"], op["kind"]), "int32": INT32.get(op["kind"])}.get(dtype, op["kind"])  # a binary op's program
  sfpu_op = lambda op: op["kind"] in BINARY or (dst and op.get("y", "zero") != "zero")
  assert dtype != "int32" or all(sfpu_name(op) for op in ops if sfpu_op(op)), "an int32 kernel adds, subtracts, compares and picks"
  slots = allocate({"mm" for op in ops if op["kind"] == "matmul"} | {"mul" for op in ops if op["kind"] == "mul" and not dst} |
                   {u for op in ops for u in op["posts"]} | {op["kind"] for op in ops if op["kind"] in COMBINE} |
                   {sfpu_name(op) for op in ops if sfpu_op(op)})
  frame, matmuls = inputs + len(ops), [j for j, op in enumerate(ops) if op["kind"] == "matmul"]
  assert not (dst and matmuls), "only bf16 kernels have matmuls"
  rows_at, top = [], FRAMES + 2 * frame + len(ops)  # each matmul's row of A in L1, while they fit; else A streams too
  assert top <= UNPACKABLE and slot_base(dtype) + top * tile_bytes(dtype) <= OUTS and len(matmuls) <= 16, "too many tiles in a kernel"
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
  def where(r, base):  # which tile of r (an input or the result, its [address, strides, offset] at a[base]) goes with b, i_, j
    s = size(r)
    b, i, j = f"b * a[{base + 1}] + " if s[0] > 1 else "", "i_" if ext(s, 1) > 1 else 0, "j" if ext(s, 2) > 1 else 0
    return f"{b}{i} * a[{base + 2}] + {j} + a[{base + 3}]"
  tile = lambda i: where(i, arg(i))
  B, U, M, P = range(4)  # the programs: BRISC moves tiles; TRISC 0, 1, 2 feed the unpack, math and pack threads
  def math_end(j):  # the math thread's last words for op j: the root's padding in Dst goes to 0 first, in a tile at the edge
    if j != root: return tt(MATH_END)
    at_end = lambda ax: f"{('i_', 'j')[ax - 1]} == {ext(size(me), ax) - 1}"
    edge = [f"if ({at_end(2)}) {{ {tt(mask_columns('sum', cols % 32))} }}"] * (cols > 1 and cols % 32 > 0)
    edge += [f"if ({at_end(1)}) {{ {tt(mask_rows('sum', rows % 32))} }}"] * (rows > 1 and rows % 32 > 0)
    return " ".join(edge + [tt(MATH_END)])
  def pslot(f, r):  # where the packer puts r: its slot, or for a result in another dtype, one at OUTS
    return (f if swept(me) else 0) if out != dtype and r == me else slot(f, r)
  def output(j, f):  # for the pack thread: where op j's result goes, if it is the root
    return f"output({f if swept(me) else 0}, {pslot(f, me)}, {where(me, 2)}); " if j == root else ""
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
      if dst: c[M].append("arrived();")
      for j in once + swept_ops:
        op, me_j = ops[j], ("op", j)
        cond, posted = "if (k == 0) " if j in once else "", "" if j == root else " posted++;"
        pk = halves([pack(DST_HALF * h, pslot(f, me_j), SEM_PACKED if j == root else SEM_L1) for h in (0, 1)])
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
            c[B].append(f"{cond}{{ unsigned na = fresh(i_ % {bp}, b * {rt} + i_), nb = fresh(15, b * {w} + j); "
                        f"for (unsigned m = 0; m < {kt}; m++) step({bt}, a[{y}], {b_tile}, nb ? {kept_b} + m : KEPT, "
                        f"a[{x}], {a_tile}, na ? {rows_at[q]} + i_ % {bp} * {kt} + m : KEPT); }}")
            walk = [ins("UNPACR", last=1, set_dat_valid=1, ovrd_thread_id=1, addr_mode=1),
                    ins("UNPACR", last=1, set_dat_valid=1, ovrd_thread_id=1, unpack_block_selection=1, addr_mode=1)] + REPORT
            c[U].append(f"{cond}{{ push_word({setadc(bh.UNPACK1, 0):#x} + {rows_at[q]} + i_ % {bp} * {kt}); {tt([setadc(bh.UNPACK0, kept_b)])} "
                        f"{on} for (unsigned m = 0; m < {kt}; m++) {{ fetched(); {tt(walk)} }} {off}{posted} }}")
          elif rows_at[q] is None:  # A's tile streams in beside B's, each step
            c[B].append(f"{cond}for (unsigned m = 0; m < {kt}; m++) "
                        f"step({bt}, a[{y}], {b_tile}, 1 + (steps & 1), a[{x}], {a_tile}, 3 + (steps & 1));")
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
        x, y = slot(f, op["x"]), slot(f, op["y"])
        prog = sfpu([sfpu_name(op)], slots) if sfpu_op(op) else []
        if dst:  # x into the op's Dst half, y 128 rows on, then the SFPU: all on the math thread
          fills = lambda h: fill(x, DST_HALF * h) + (fill(y, DST_HALF * h + 128) if op["y"] != "zero" else []) + FILLED
          math = halves([math_begin(h, False) + fills(h) + prog + sfpu(op['posts'], slots) for h in (0, 1)])
          c[M].append(f"{cond}{{ {drain}{math} {math_end(j)} h ^= 1;{posted} }}")
          continue
        if op["kind"] in BINARY:  # x and y each into Dst for the SFPU: x in the op's half, y 128 rows on
          unpacks, fpu = unpack(x, ZERO_SLOT) + unpack(y, ZERO_SLOT), elementwise("add", slots) + elementwise("add", slots, dst=128)
        else: unpacks, fpu = unpack(x, y), elementwise(op["kind"], slots)
        c[U].append(f"{cond}{{ {drain}{tt(unpacks)}{posted} }}")
        math = halves([math_begin(h, op['kind'] == 'mul') + fpu + prog + sfpu(op['posts'], slots) for h in (0, 1)])
        c[M].append(f"{cond}{{ {math} {math_end(j)} h ^= 1; }}")
      for acc, j in enumerate(reductions):
        assert acc < len(ACCS), "two reductions a pass"
        op, me_j, x, kind = ops[j], ("op", j), ops[j]["x"], ops[j]["kind"]
        drain = "drain(); " if isinstance(x, tuple) else ""
        edge = size(x)[axis] % 32 if axis and size(x)[axis] > 1 else 0  # the operand's rows or columns in its last tile, if cut short
        end = f"k == {L - 1}" if swept(x) else "k == 0"  # the line's last tile (a narrow operand is one tile a line)
        src = slot(f, x) if dst else None
        def step(first):  # the last tile's padding must not count
          plain, cut = tt(reduce_step(kind, slots, acc, first, 0, src)), tt(reduce_step(kind, slots, acc, first, edge, src))
          return f"if ({end}) {{ {cut} }} else {{ {plain} }}" if edge else plain
        finish, posted = f'{tt(reduce_finish(kind, slots, acc, op["posts"]))} {math_end(j)}', "" if j == root else " posted++;"
        pk = tt(pack(ACCS[acc][0], pslot(f, me_j), SEM_PACKED if j == root else SEM_L1, ACCS[acc][1]))
        unpacks = "" if dst else tt(unpack(slot(f, x), ZERO_SLOT))  # an fp32 step fills Dst itself
        mine = M if dst else U  # who drains and counts
        if swept(x):  # swept along the line
          c[mine].append(f"{drain}{unpacks} if ({end}) {{{posted} }}")
          c[M].append(f"if (k == 0) {{ {step(True)} }} else {{ {step(False)} }} if ({end}) {{ {finish} }}")
        else:  # reduced in the pass's first tile
          c[mine].append(f"if ({end}) {{ {drain}{unpacks}{posted} }}")
          c[M].append(f"if ({end}) {{ {step(True)} {finish} }}")
        c[P].append(f"if ({end}) {{ {output(j, f)}{pk} }}")
      c[M if dst else U].append(tt(REPORT))
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
  image = compile_kernel(_brisc(total, program(B), dtype, out), _unpacker(program(U), kernel_config(dtype, out)),
                         _math(slots, program(M), dst), _packer(program(P)))
  return image, (nb * rt * w // L if reducing else nb * br * bc)  # and how many units the result is

def gather_kernel(dtype):
  """Rows of a tensor into a new one, args [first unit, units, out, the tensor's address, tile stride, tile offset, the
  result's tile columns, rows, the int32 row of their indices' address and tile offset]. BRISC alone: for each result
  tile, the indices' tile in, then each row's two 16-datum runs (left and right face) from wherever that row sits in
  the tensor's tiles. A NoC read from DRAM must keep its address's
  low 6 bits, so each run comes in as the 64-byte block holding it and is copied into place from there. One row fills
  its tile: it comes in once and is doubled up inside L1 by NoC copies from the tile to itself."""
  t, face, run = tile_bytes(dtype), tile_bytes(dtype) // 4, 16 * ITEMSIZE[dtype]  # bytes in a tile, a face, a face's row
  return compile_kernel(f"""
#define SLOT {DATA:#x}
#define BLOCK {DATA + t:#x}
#define IDS {DATA + t + 64:#x}
{_array("bank", [x | y << 6 for x, y in bh.DRAM])}
static void copy(unsigned from, unsigned to, unsigned n) {{ noc_read(noc_me() & 63, noc_me() >> 6, from, to, n); }}
static unsigned id(unsigned r) {{ return ((volatile unsigned *)IDS)[r / 16 * 256 + r % 16]; }}  // index r of this tile's 32, in row 0
void kmain(volatile unsigned *a) {{
  for (unsigned u = 0; u < a[1]; u++) {{
    unsigned t = a[0] + u, i = t / a[6], j = t % a[6], rows = a[7] == 1 ? 1 : 32, it = a[9] + i;  // it: the indices' tile
    noc_read(bank[it % 8] & 63, bank[it % 8] >> 6, a[8] + it / 8 * 4096, IDS, 4096);
    copy({ZEROS:#x}, SLOT, {t});  // the rows past the end stay zero
    noc_wait();
    for (unsigned r = 0; r < rows; r++) {{
      unsigned row = i * 32 + r;
      for (unsigned face = r / 16 * 2; face < r / 16 * 2 + 2; face++) {{
        volatile unsigned *to = (volatile unsigned *)(SLOT + face * {face} + r % 16 * {run});
        if (row >= a[7]) continue;
        unsigned n = id(r), from = n / 32 * a[4] + j + a[5], at = (n % 32 / 16 * 2 + face % 2) * {face} + n % 16 * {run};
        noc_read(bank[from % 8] & 63, bank[from % 8] >> 6, a[3] + from / 8 * {t} + at / 64 * 64, BLOCK, 64);
        noc_wait();
        for (unsigned k = 0; k < {run // 4}; k++) to[k] = ((volatile unsigned *)(BLOCK + at % 64))[k];
      }}
    }}
    if (rows == 1) {{  // row 0 of faces 0 and 1 down their 16 rows, then faces 0-1 onto 2-3
      l1_settle(SLOT);
      for (unsigned n = {run}; n < {face}; n *= 2) {{ copy(SLOT, SLOT + n, n); copy(SLOT + {face}, SLOT + {face} + n, n); noc_wait(); }}
      copy(SLOT, SLOT + {2 * face}, {2 * face});
      noc_wait();
    }}
    noc_write(SLOT, bank[t % 8] & 63, bank[t % 8] >> 6, a[2] + t / 8 * {t}, {t});
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
    _brisc("a[1]", loop(*[f"reached(frame_done[{f}]); flipped(a[3], t / {per} * a[4] + t % {ct} * a[5] + t % {per} / {ct} + a[6], "
                          f"{frame(f)[0]}); ready(); frame_done[{f}] = ++events;" for f in (0, 1)])),
    _unpacker(loop(*[f"fetched(); {tt(unpack(frame(f)[0], ZERO_SLOT) + REPORT)}" for f in (0, 1)]), kernel_config(transposing=True)),
    _math({}, loop(*[halves([math_begin(h, False) + elementwise("add", {}) + MATH_END for h in (0, 1)]) + " h ^= 1;"] * 2)),  # padding stays 0
    _packer(loop(*[f"output({f}, {frame(f)[1]}, t); " + halves([pack(DST_HALF * h, frame(f)[1], SEM_PACKED) for h in (0, 1)]) + " h ^= 1;"
                   for f in (0, 1)])))

MOVE_SLOTS = 127  # x's tiles cached in L1 by a move, by tile number mod a prime, so power-of-two strides spread out

def move_kernel(dtype):
  """A tensor moved around: reshape, permute, expand, pad, slice. args [first unit, units, out, x's address, tile
  strides for batch and row, tile offset, x's rows and columns, the result's rows and columns, then the map: an offset
  and strides for batch, row and column, and each axis's bounds lo and hi]. Element (b, r, c) of the result, inside
  the bounds, is element offset + b sb + r sr + c sc of x, counted in row-major order; outside them, 0.
  BRISC alone, a datum at a time: a NoC read cannot put 2 bytes just anywhere, so x's tiles come whole into a cache in
  L1 as they are needed. A result tile starts as a copy of the zero tile, gets the datums inside the bounds, and goes
  home. Where a datum is in x takes two divisions, unless it is the element after the last one read (as in a reshape
  or a slice of whole rows), which is a step."""
  return compile_kernel(f"""
#define OUT {DATA:#x}
#define CACHE {DATA + tile_bytes(dtype):#x}
typedef {"unsigned short" if ITEMSIZE[dtype] == 2 else "unsigned"} datum;
{_array("bank", [x | y << 6 for x, y in bh.DRAM])}
static unsigned held[{MOVE_SLOTS}];  // tile number + 1 of what each cache slot holds (0: nothing yet)
static volatile datum *fetch(unsigned base, unsigned t) {{  // x's tile t, in L1
  unsigned s = t % {MOVE_SLOTS};
  if (held[s] == t + 1) return (volatile datum *)(CACHE + s * {tile_bytes(dtype)});
  noc_read(bank[t % 8] & 63, bank[t % 8] >> 6, base + t / 8 * {tile_bytes(dtype)}, CACHE + s * {tile_bytes(dtype)}, {tile_bytes(dtype)});
  noc_wait();
  held[s] = t + 1;
  return (volatile datum *)(CACHE + s * {tile_bytes(dtype)});
}}
static unsigned at(unsigned r, unsigned c) {{ return (r / 16 * 2 + c / 16) * 256 + r % 16 * 16 + c % 16; }}  // datum (r, c) in a tile
// The places [from, to) of tile k along an axis n long whose element is inside [lo, hi); a dimension of 1 fills its tile
static int from(int k, int n, int lo) {{ return n == 1 || lo <= 32 * k ? 0 : lo - 32 * k; }}
static int to(int k, int n, int lo, int hi) {{
  if (n == 1) return lo <= 0 && 0 < hi ? 32 : 0;
  int end = (hi < n ? hi : n) - 32 * k;
  return end < 32 ? end : 32;
}}
void kmain(volatile unsigned *a) {{
  unsigned base = a[3], bstride = a[4], stride = a[5], offset = a[6], xr = a[7], xc = a[8], rows = a[9], cols = a[10];
  unsigned tr = (rows + 31) / 32, tc = (cols + 31) / 32;
  int e0 = a[11], sb = a[12], sr = a[13], sc = a[14], lo_b = a[15], lo_r = a[16], lo_c = a[17], hi_b = a[18], hi_r = a[19], hi_c = a[20];
  volatile datum *out = (volatile datum *)OUT, *tile = 0;
  unsigned next = ~0u, row = 0, xj = 0, seen = ~0u, xb = 0, xi = 0, held_t = ~0u;  // the element after the last one read, and where it is
  for (unsigned u = 0; u < a[1]; u++) {{
    unsigned t = a[0] + u;
    int b = t / (tr * tc), i = t % (tr * tc) / tc, j = t % tc;
    noc_read(noc_me() & 63, noc_me() >> 6, {ZEROS:#x}, OUT, {tile_bytes(dtype)});  // padding and what lies outside the bounds stay 0
    noc_wait();
    int r1 = lo_b <= b && b < hi_b ? to(i, rows, lo_r, hi_r) : 0, c0 = from(j, cols, lo_c), c1 = to(j, cols, lo_c, hi_c);
    for (int r = from(i, rows, lo_r); r < r1; r++) {{
      int y = rows == 1 ? 0 : i * 32 + r, e_row = e0 + b * sb + y * sr;
      for (int c = c0; c < c1; c++) {{
        unsigned e = e_row + (cols == 1 ? 0 : j * 32 + c) * sc, t = 0;
        if (e != next) {{ row = e / xc; xj = e % xc; }} else if (++xj == xc) {{ xj = 0; row++; }}
        if (row != seen) {{ seen = row; xb = row / xr; xi = row % xr; }}
        next = e + 1;
        if ((t = xb * bstride + xi / 32 * stride + xj / 32 + offset) != held_t) {{ tile = fetch(base, t); held_t = t; }}
        out[at(r, c)] = tile[at(xi % 32, xj % 32)];
      }}
    }}
    noc_write(OUT, bank[t % 8] & 63, bank[t % 8] >> 6, a[2] + t / 8 * {tile_bytes(dtype)}, {tile_bytes(dtype)});
    noc_wait();
  }}
}}""")

class Tensor:
  """batch x rows x cols of bf16, fp32 or int32 (dtype; batch 1 unless said): in DRAM as 32x32 tiles, matrix after matrix, each in row-major
  tile order, tile i in bank i % 8; or, until realized, an op on other tensors. Realizing compiles the whole tree of
  unrealized ops under it into one kernel; a matmul's own operands are realized first, since each of its output tiles
  needs a whole row and column of them (a transposed B is read transposed rather than realized).
  Any shape: the last tile row and column are padded, with zeros, except that a dimension of 1 fills its tile with
  copies (a row vector's tile holds the row 32 times), so broadcasting reuses the tile as it is. Every kernel leaves
  its result that way too, so a matmul can sum over a padded K.
  A view (x[b], slices on tile boundaries, x.split(n)) shares its owner's DRAM: its tile (b, i, j) is tile
  b * bstride + i * stride + j + offset. Any other movement (reshape, permute, expand, pad, other slices) is a copy,
  made by move_kernel from a map over the source's elements."""
  def __init__(self, rt, rows, cols, op=None, srcs=(), batch=1, dtype="bf16"):
    self.rt, self.rows, self.cols, self.batch, self.op, self.srcs, self.dtype = rt, rows, cols, batch, op, srcs, dtype
    tr, tc = -(-rows // 32), -(-cols // 32)  # tile rows and columns
    self.tiles, self.bstride, self.stride, self.offset, self.owner = batch * tr * tc, tr * tc, tc, 0, None
    self.addr = None if op else self._alloc()
  def _alloc(self): return self.rt.dram_alloc(-(-self.tiles // len(bh.DRAM)) * tile_bytes(self.dtype))

  def __del__(self):  # its DRAM goes back once nothing refers to it (and no queued launch can still use it)
    if self.addr is not None and self.owner is None: self.rt.dram_free(self.addr)

  def _where(self, b, i, j):  # tile (i, j) of matrix b in DRAM: (x, y, address)
    t = b * self.bstride + i * self.stride + j + self.offset
    return (*bh.DRAM[t % 8], self.addr + t // 8 * tile_bytes(self.dtype))

  @classmethod
  def full(cls, rt, rows, cols, value, dtype="bf16"): return cls.of(rt, [[value] * cols for _ in range(rows)], dtype)

  @classmethod
  def of(cls, rt, data, dtype="bf16"):
    """From a list of rows of numbers, or a list of those (a batch), rounded to dtype (bf16: to nearest even)."""
    matrices = data if isinstance(data[0][0], (list, tuple)) else [data]
    values = array.array("i" if dtype == "int32" else "f", [v for m in matrices for row in m for v in row])
    if dtype == "bf16": values = array.array("H", [(b + 0x7FFF + (b >> 16 & 1)) >> 16 for b in array.array("I", values.tobytes())])
    return cls.frombytes(rt, len(matrices[0]), len(matrices[0][0]), values, len(matrices), dtype)

  @classmethod
  def frombytes(cls, rt, rows, cols, data, batch=1, dtype="bf16"):
    """From dtype's values in row-major order (little-endian), matrix after matrix. Runs of 16 values are copied into
    tile order as they are, nothing per value, which is what makes a model's weights quick to load."""
    t, n, tr, tc, data = cls(rt, rows, cols, batch=batch, dtype=dtype), ITEMSIZE[dtype], -(-rows // 32), -(-cols // 32), memoryview(data).cast("B")
    assert len(data) == n * batch * rows * cols, f"{len(data)} bytes for {batch} x {rows} x {cols} of {dtype}"
    for b, i in ((b, i) for b in range(batch) for i in range(tr * tc)):
      m, r0, c0, tile = data[n * b * rows * cols:n * (b + 1) * rows * cols], i // tc * 32, i % tc * 32, []
      for face in range(4):  # 16 rows of 16, face by face (the tile layout, at the top)
        c = c0 + face % 2 * 16
        for r in range(r0 + face // 2 * 16, r0 + face // 2 * 16 + 16):
          r = 0 if rows == 1 else r  # a dimension of 1 fills the tile; the rest pads with 0
          at = n * (r * cols + c)
          run = b"" if r >= rows else bytes(m[n * r:n * r + n]) * 16 if cols == 1 else bytes(m[at:at + n * max(0, min(16, cols - c))])
          tile.append(run + bytes(16 * n - len(run)))
      t.rt.chip.write(*t._where(b, i // tc, i % tc), b"".join(tile))
    return t

  def tobytes(self):
    """dtype's values in row-major order, matrix after matrix, as frombytes takes them. Waits for the launches queued."""
    self.realize()
    self.rt.wait()
    n, rows, cols, ct = ITEMSIZE[self.dtype], self.rows, self.cols, -(-self.cols // 32)
    out = bytearray(n * self.batch * rows * cols)
    for b, i in ((b, i) for b in range(self.batch) for i in range(self.tiles // self.batch)):
      r0, c0, base = i // ct * 32, i % ct * 32, b * rows * cols
      tile = self.rt.chip.read(*self._where(b, i // ct, i % ct), tile_bytes(self.dtype))
      for face in range(4):
        c, k = c0 + face % 2 * 16, min(16, cols - c0 - face % 2 * 16)
        for row, r in enumerate(range(r0 + face // 2 * 16, min(r0 + face // 2 * 16 + 16, rows))):
          at = (face * 256 + row * 16) * n
          if k > 0: out[n * (base + r * cols + c):n * (base + r * cols + c + k)] = tile[at:at + n * k]
    return bytes(out)

  def tolist(self):  # rows of numbers, or a list of those if this is a batch
    raw = self.tobytes()
    if self.dtype == "bf16": raw = array.array("I", [b << 16 for b in array.array("H", raw)]).tobytes()
    values = array.array("i" if self.dtype == "int32" else "f", raw).tolist()
    m = [[values[(b * self.rows + r) * self.cols:(b * self.rows + r + 1) * self.cols] for r in range(self.rows)] for b in range(self.batch)]
    return m if self.batch > 1 else m[0]

  def gather(self, ids):
    """Rows ids of this tensor, as a tensor: an embedding lookup, by a data-movement kernel (see gather_kernel). ids is
    an int32 tensor of one row, read on the chip (so it can be computed there; nothing checks its bounds), or a list."""
    if not isinstance(ids, Tensor):
      assert all(0 <= i < self.rows for i in ids)
      ids = Tensor.of(self.rt, [list(ids)], "int32")
    assert self.batch == 1 and ids.dtype == "int32" and ids.shape[:2] == (1, 1), "rows of a matrix, by a row of int32"
    out, d, ids = Tensor(self.rt, ids.cols, self.cols, dtype=self.dtype), self.dtype, ids.realize()
    return out._run(_cached(("gather", d), lambda: gather_kernel(d)), out.addr, self.realize().addr, self.stride, self.offset,
                    -(-self.cols // 32), ids.cols, ids.addr, ids.offset)

  def _view(self, rows, cols, batch, offset, bstride=0, stride=None):
    v = Tensor(self.rt, rows, cols, "view", batch=batch, dtype=self.dtype)
    v.op, v.addr, v.owner = None, self.realize().addr, self.owner or self
    v.offset, v.bstride, v.stride = self.offset + offset, bstride, self.stride if stride is None else stride
    return v

  def __getitem__(self, key):
    """x[b], one matrix of a batch; x[rows, cols] of every matrix, or x[batch, rows, cols], with slices (any step). A
    slice of whole tiles (starting on tile boundaries, ending on one or at the end, step 1) is a view, read in place;
    any other is a copy (_move)."""
    if isinstance(key, int): return self._view(self.rows, self.cols, 1, key * self.bstride)
    assert isinstance(key, tuple) and len(key) in (2, 3) and all(isinstance(k, slice) for k in key), "x[b], x[rows, cols] or x[batch, rows, cols]"
    ranges = [range(*k.indices(n)) for k, n in zip((slice(None),) * (3 - len(key)) + tuple(key), self.shape)]
    shape = tuple(map(len, ranges))
    assert all(shape), f"an empty slice of {self.shape}"
    if all(g.step == 1 and (ax == 0 or g.start % 32 == 0 and (g.stop % 32 == 0 or g.stop == n) and (len(g) > 1 or n == 1))
           for ax, (g, n) in enumerate(zip(ranges, self.shape))):  # whole tiles; a dimension of 1 must fill its tile
      b, r, c = (g.start for g in ranges)
      return self._view(shape[1], shape[2], shape[0], b * self.bstride + r // 32 * self.stride + c // 32, self.bstride)
    st = self._strides()
    return self._move(shape, sum(g.start * s for g, s in zip(ranges, st)), tuple(g.step * s for g, s in zip(ranges, st)))

  # Movement: each is one map from the result's elements to this tensor's (_move), copied by one launch of move_kernel
  def _strides(self): return (self.rows * self.cols, self.cols, 1)  # of its elements, counted in row-major order
  def _move(self, shape, offset, strides, lo=(0, 0, 0), hi=None):
    """The tensor of shape (batch, rows, cols) whose element (b, r, c) is this one's offset + b strides[0] + r
    strides[1] + c strides[2] (counted in row-major order) inside lo..hi on each axis, and 0 outside."""
    t = Tensor(self.rt, shape[1], shape[2], "move", (self,), shape[0], self.dtype)
    t.map = (offset, *strides, *lo, *(hi or shape))
    return t
  def reshape(self, *shape):  # up to three sizes, missing ones in front being 1
    shape = (1,) * (3 - len(shape)) + shape
    assert math.prod(shape) == math.prod(self.shape), f"cannot reshape {self.shape} to {shape}"
    return self if shape == self.shape else self._move(shape, 0, (shape[1] * shape[2], shape[2], 1))
  def permute(self, *order):  # the axes in a new order; swapping rows and columns is the transpose
    assert sorted(order) == [0, 1, 2]
    if order == (0, 2, 1): return self.T
    return self if order == (0, 1, 2) else self._move(tuple(self.shape[k] for k in order), 0, tuple(self._strides()[k] for k in order))
  def expand(self, *shape):  # dimensions of 1 repeated, missing ones in front being 1
    shape = (1,) * (3 - len(shape)) + shape
    assert all(n in (m, 1) for n, m in zip(self.shape, shape)), f"cannot expand {self.shape} to {shape}"
    return self if shape == self.shape else self._move(shape, 0, tuple(s if n == m else 0 for s, n, m in zip(self._strides(), self.shape, shape)))
  def pad(self, widths):  # zeros: (before, after) for each of the last len(widths) axes, as numpy's
    widths = ((0, 0),) * (3 - len(widths)) + tuple(widths)
    lo, shape = tuple(w[0] for w in widths), tuple(n + w[0] + w[1] for n, w in zip(self.shape, widths))
    return self._move(shape, -sum(w * s for w, s in zip(lo, self._strides())), self._strides(), lo, tuple(w + n for w, n in zip(lo, self.shape)))

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
    out = Tensor(self.rt, self.rows, self.batch * self.cols, dtype=self.dtype)
    (self if self.op not in (None, "T", "move") else self + 0.0).realize(out.split(self.batch))  # nor can these write a view
    return out

  @property
  def T(self):
    """The transpose (of each matrix of a batch). A matmul reads it as its B straight from this tensor, transposing
    tiles as they come in; anything else realizes it in a kernel of its own (transpose_kernel). The unpacker transposes
    only on its way into SrcA, so fp32 moves instead."""
    if self.dtype == "fp32": return self._move((self.batch, self.cols, self.rows), 0, (self.rows * self.cols, 1, self.cols))
    return Tensor(self.rt, self.cols, self.rows, "T", (self,), self.batch)

  def realize(self, into=None):
    """Compute it now, into DRAM of its own, or into the view `into` (see merge). Returns it, now realized."""
    if self.op is None: return self
    if self.op in ("T", "move"):  # data movement, into a tensor of its own
      assert into is None, "a transpose or a move writes a tensor of its own"
      x, op, self.op, self.srcs, d = self.srcs[0].realize(), self.op, None, (), self.dtype
      self.addr = self._alloc()
      where = (self.addr, x.addr, x.bstride if x.batch > 1 else 0, x.stride, x.offset)
      if op == "T": return self._run(_cached(("transpose", x.rows, x.cols), lambda: transpose_kernel(x.rows, x.cols)), *where)
      return self._run(_cached(("move", d), lambda: move_kernel(d)), *where, x.rows, x.cols, self.rows, self.cols,
                       *(v & 0xFFFFFFFF for v in self.map))
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
    e, dtype = _tree(self, inputs)
    key = (e, tuple(u.shape for u in inputs), dtype, self.dtype)
    kernel, units = _cached(key, lambda: fused_kernel(*key))
    if into: self.addr, self.owner, self.bstride, self.stride, self.offset = into.addr, into.owner or into, into.bstride, into.stride, into.offset
    else: self.addr = self._alloc()
    self.op, self.srcs = None, ()
    args = [v for t in [self] + inputs for v in (t.addr, t.bstride if t.batch > 1 else 0, t.stride, t.offset)]
    return self._run(kernel, *args, units=units)

  def _run(self, kernel, *args, units=None):
    """Queue kernel over units of this tensor (tiles unless told), split evenly over as many workers as have any. Nothing waits for
    it: launches run in order, so the next kernel reads this result, and reading it back (tobytes) waits."""
    self.rt.launch(kernel, args, units or self.tiles)
    return self

  @property
  def shape(self): return (self.batch, self.rows, self.cols)

  # Elementwise: tinygrad's primitives (exp2, log2, sin, sqrt, reciprocal; add, mul, max, cmplt, cmpne; where), and the
  # rest built from them. exp, log and cos are SFPU programs too, with log2(e), ln 2 and the quarter turn kept in fp32
  # inside: as numbers they would be bf16 tensors, and exp(10) as exp2(10 * log2(e)) would be 2% off.
  def _unary(self, op):  # int32 goes through fp32, but for relu (a max with 0)
    if self.dtype == "int32" and op != "relu": return self.float()._unary(op)
    return Tensor(self.rt, self.rows, self.cols, op, (self,), self.batch, self.dtype)
  def _binop(self, op, x, reverse=False):
    """numpy's broadcasting: shapes match, or one side has 1 where the other does not. A number is a 1 x 1 tensor of
    this one's dtype (fp32 for a float with int32). Different dtypes meet in fp32, and so does int32 for what it does
    not do itself (mul: exact below 2^24)."""
    if not isinstance(x, Tensor): x = Tensor.full(self.rt, 1, 1, x, "fp32" if self.dtype == "int32" and isinstance(x, float) else self.dtype)
    a, b = (x, self) if reverse else (self, x)
    if a.dtype != b.dtype or (a.dtype == "int32" and op not in INT32): a, b = a.float(), b.float()
    assert all(n == m or 1 in (n, m) for n, m in zip(a.shape, b.shape)), f"cannot broadcast {a.shape} with {b.shape}"
    batch, rows, cols = map(max, a.shape, b.shape)
    return Tensor(self.rt, rows, cols, op, (a, b), batch, a.dtype)
  def cast(self, dtype): return self if dtype == self.dtype else Tensor(self.rt, self.rows, self.cols, "cast", (self,), self.batch, dtype)
  def float(self): return self.cast("fp32")
  def bfloat16(self): return self.cast("bf16")
  def exp2(self): return self._unary("exp2")
  def log2(self): return self._unary("log2")
  def sin(self): return self._unary("sin")
  def sqrt(self): return self._unary("sqrt")
  def reciprocal(self): return self._unary("recip")
  def exp(self): return self._unary("exp")
  def log(self): return self._unary("log")
  def cos(self): return self._unary("cos")
  def relu(self): return self._unary("relu")
  def rsqrt(self): return self.sqrt().reciprocal()
  def maximum(self, x): return self._binop("max", x)
  def minimum(self, x): return -((-self).maximum(-x))
  def where(self, x, y):  # x where self is not 0, else y; the condition, if of another dtype, as 0 or 1 of theirs
    dtype = next((t.dtype for t in (x, y) if isinstance(t, Tensor)), self.dtype)
    c = self if self.dtype == dtype else self.ne(0).cast(dtype)
    return c._binop("keep", x) + c._binop("drop", y)
  def __add__(self, x): return self._binop("add", x)
  def __sub__(self, x): return self._binop("sub", x)
  def __mul__(self, x): return self._binop("mul", x)
  def __rsub__(self, x): return self._binop("sub", x, True)
  __radd__, __rmul__ = __add__, __mul__
  def __truediv__(self, x): return self * (1 / x if isinstance(x, (int, float)) else x.reciprocal())
  def __neg__(self): return 0 - self if self.dtype == "int32" else self * -1  # int32 subtracts exactly, multiplies through fp32
  def __lt__(self, x): return self._binop("cmplt", x)
  def __gt__(self, x): return self._binop("cmplt", x, True)
  def __ge__(self, x): return 1 - (self < x)
  def __le__(self, x): return 1 - (self > x)
  def ne(self, x): return self._binop("cmpne", x)
  def eq(self, x): return 1 - self.ne(x)
  __ne__, __eq__ = ne, eq
  def __hash__(self): return id(self)  # == is elementwise, so a tensor hashes by identity, as in tinygrad
  def gelu(self):  # GPT-2's: 0.5 x (1 + tanh(z)), z = sqrt(2/pi) (x + 0.044715 x^3), which is x / (1 + e^-2z)
    k = 2 * math.sqrt(2 / math.pi)
    return self * (1 + (self * (self * self * (-0.044715 * k) - k)).exp()).reciprocal()

  # Reductions, and what is built from them
  def sum(self, axis=None): return self._reduce("sum", axis)
  def max(self, axis=None): return self._reduce("max", axis)
  def _reduce(self, f, axis):
    """numpy's, keeping the axis as 1: over the batch (0, one matrix), down the rows (1, one row) or along each row
    (2, one column); over all three if None. Each axis is a kernel of its own, sweeping lines of tiles along it."""
    if self.dtype == "int32": return self.float()._reduce(f, axis).cast("int32")  # exact below 2^24
    if axis is None: return self._reduce(f, 0)._reduce(f, 1)._reduce(f, 2)
    shape = [1 if ax == axis % 3 else s for ax, s in enumerate(self.shape)]
    return self if shape == list(self.shape) else Tensor(self.rt, shape[1], shape[2], f"{f}{axis % 3}", (self,), shape[0], self.dtype)
  def mean(self, axis=None):
    s = self.sum(axis)
    return s * (math.prod(s.shape) / math.prod(self.shape))
  def softmax(self, axis=-1):  # max first, so exp stays in range
    e = (self - self.max(axis)).exp()
    return e / e.sum(axis)
  def layernorm(self, axis=-1, eps=1e-5):  # to mean 0 and variance 1 along axis
    c = self - self.mean(axis)
    return c * ((c * c).mean(axis) + eps).rsqrt()

  def __matmul__(self, x): return self.matmul(x)
  def matmul(self, x, dtype=None):
    """Matrix by matrix through a batch (a batch of 1 repeats), summed in fp32 in Dst and written as dtype: bf16, or fp32
    if either operand is, unless said. fp32 operands go as bf16x3: each is hi + lo, both bf16, and hi@hi + hi@lo + lo@hi
    sum in fp32 (lo@lo, 2^-16 of the rest, is left out), about 16 bits of each operand where fp32 has 24."""
    assert "int32" not in (self.dtype, x.dtype), "the FPU multiplies floats"
    dtype = dtype or ("fp32" if "fp32" in (self.dtype, x.dtype) else "bf16")
    if "fp32" in (self.dtype, x.dtype):
      (ah, al), (bh, bl) = ((t.bfloat16(), (t.float() - t.bfloat16()).bfloat16()) for t in (self, x))
      return (ah.matmul(bh, "fp32") + (ah.matmul(bl, "fp32") + al.matmul(bh, "fp32"))).cast(dtype)
    assert self.cols == x.rows and (self.batch == x.batch or 1 in (self.batch, x.batch)), f"{self.shape} @ {x.shape}"
    if self.cols == 1: return (self * x).cast(dtype)  # (R, 1) @ (1, C) is broadcasting
    kt, tiles = self.cols // 32, -(-self.rows // 32) * -(-x.cols // 32)
    s = max([s for s in (2, 4, 8) if self.batch == x.batch == 1 and self.cols % (32 * s) == 0 and kt // s >= 4 and
             s * tiles <= len(self.rt.cores)], default=1)
    if s > 1:  # too few result tiles to keep the workers busy (a token's MLP): K cut in s, a batch, and the parts summed
      parts = self.split(s).matmul(x.split(s, 0), dtype)
      return sum((parts[i] for i in range(1, s)), parts[0])
    return Tensor(self.rt, self.rows, x.cols, "matmul", (self, x), max(self.batch, x.batch), dtype)

WHOLE = ("matmul", "T", "move", "cast")  # ops whose operands are whole tensors, realized first, not tiles in the same kernel
def _lazy(t, seen):  # the unrealized ops under t, t first
  if t.op is None or id(t) in seen: return
  seen.add(id(t))
  yield t
  for s in () if t.op in WHOLE else t.srcs: yield from _lazy(s, seen)

def _ops(t, seen):  # how many ops a kernel for t would hold, at most (an SFPU op on another's result adds none)
  if t.op is None or id(t) in seen: return 0
  seen.add(id(t))
  return 1 + (0 if t.op in WHOLE else sum(_ops(s, seen) for s in t.srcs))

def _tree(t, inputs):
  """The expression a kernel for t computes (fused_kernel's), realized tensors as numbers into inputs, and the dtype it
  computes in: t's, or for a cast or a matmul its operands'. The kernel sweeps along the axis of t's reductions (the
  first found; any along another axis compute first, on their own)."""
  dtype = t.srcs[0].dtype if t.op in ("cast", "matmul") else t.dtype
  axis = next((FOLDS[u.op] for u in _lazy(t, set()) if u.op in COMBINE), 2)
  n = _ext(t, axis)  # the sweep is as long as t along the axis, unless t is one tile there: then its first reduction says
  return _expr(t, t.shape, inputs, [axis, n if n > 1 else None, set(), dtype, t]), dtype

def _ext(t, ax): return t.shape[ax] if ax == 0 else -(-t.shape[ax] // 32)  # t's extent along an axis: matrices, or tiles

def _expr(t, shape, inputs, kernel):
  """The tree of unrealized ops under t, for a kernel of shape (t's). kernel: [the axis it sweeps, the sweep's length
  (in tiles; the first seen, if t is one tile along it), the reductions it holds (two accumulators), the dtype it
  computes in, its root]. Computed first, on their own: an op of another dtype, a cast, a matmul inside an fp32 kernel, an
  op of 1 along an axis the kernel keeps (repeated there: compute it once), and a reduction along another axis, of
  another length, or past the second."""
  if t.op is None:
    assert t.dtype == kernel[3], f"a {t.dtype} input to a {kernel[3]} kernel"
    if all(t is not u for u in inputs): inputs.append(t)
    return next(i for i, u in enumerate(inputs) if u is t)
  if t is kernel[4] and t.op == "cast": return ("cast", _expr(t.srcs[0].realize(), shape, inputs, kernel))
  if t is not kernel[4] and (t.dtype != kernel[3] or t.op == "cast" or (t.op == "matmul" and kernel[3] != "bf16")):
    return _expr(t.realize(), shape, inputs, kernel)
  axis = kernel[0]
  if any(t.shape[ax] != shape[ax] for ax in range(3) if ax != axis): return _expr(t.realize(), shape, inputs, kernel)
  if t.op in COMBINE:
    if FOLDS[t.op] != axis: return _expr(t.realize(), shape, inputs, kernel)
    n = _ext(t.srcs[0], axis)
    if n > 1 and kernel[1] is None: kernel[1] = n
    if (n > 1 and n != kernel[1]) or (id(t) not in kernel[2] and len(kernel[2]) == len(ACCS)): return _expr(t.realize(), shape, inputs, kernel)
    kernel[2].add(id(t))
  if t.op in ("T", "move"): return _expr(t.realize(), shape, inputs, kernel)  # kernels of their own (a matmul's B may be read transposed, below)
  if t.op == "matmul":  # its operands are whole matrices, not tiles: realize them on their own first
    a, b = t.srcs
    b = ("T", _expr(b.srcs[0].realize(), shape, inputs, kernel)) if b.op == "T" else _expr(b.realize(), shape, inputs, kernel)
    return ("matmul", _expr(a.realize(), shape, inputs, kernel), b)
  return (t.op, *(_expr(s, shape, inputs, kernel) for s in t.srcs))

kernels = {}  # (expression, input shapes) -> (image, units a matrix), or a kernel's name and shape -> image
def _cached(key, compile):
  if key not in kernels: kernels[key] = compile()
  return kernels[key]
