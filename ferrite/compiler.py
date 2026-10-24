# The compiler: tensor ops emitted as Tensix instructions straight from the ISA, no LLK. A tree of elementwise ops,
# matmuls, SFPU programs and reductions (tensor.py builds it) becomes one fused kernel, which fetches each input tile
# once, keeps every intermediate in L1 and writes only the result to DRAM. A kernel is four programs for each worker
# tile: BRISC fetches operand tiles into L1 over the NoC and sends results home, and each TRISC feeds one Tensix thread
# its instructions (unpack, math, pack), the three running at the same time on different tiles.
#
# Tile layout: four 16x16 faces in row-major order (top-left, top-right, bottom-left, bottom-right), each face
# row-major, so datum (r, c) is at (r // 16 * 2 + c // 16) * 256 + r % 16 * 16 + c % 16. The ISA has no notion of a
# tile; this is the layout the FPU's 16x16 matrix multiply wants. An unpacker reads a whole tile into a SrcA/SrcB bank,
# so face f lands in rows 16f..16f+15, and Dst holds the result the same way.
import math, struct
from ferrite import bh
from ferrite.runtime import DATA, FLAGS, SYNC, SYNC_NAMES, ZEROS, compile_kernel

ITEMSIZE = {"bf16": 2, "fp32": 4, "int32": 4}  # the dtypes, by bytes a value; a tile is 1024 values
# A bf16 tile is one SrcA/SrcB bank (64 rows of 16), 64 Dst rows. A block float's tile (BFP: the bytes of its values) is 64
# exponents, one for each 16 values (a face's row), then the 1024 values, each a sign and 7 bits of magnitude (bfp8,
# m * 2^(exponent - 133)) or 3 (bfp4, two to a byte, the low one first: bfp8's top 3, which the unpacker shifts back)
BFP = {"bfp8": 1024, "bfp4": 512}
def tile_bytes(dtype): return 64 + BFP[dtype] if dtype in BFP else 1024 * ITEMSIZE[dtype]
def ext(shape, ax): return shape[ax] if ax == 0 else -(-shape[ax] // 32)  # extent along an axis: matrices, or tiles
def slot_base(dtype): return DATA - tile_bytes(dtype)  # L1 tile slots of a kernel computing in dtype: slot i at base + i *
                                                        # its tile, so slot 0 is (part of) the runtime's zeros. The unpackers'
                                                        # Z and the packer's channel 1 Y counters pick them
OUTS = bh.L1_SIZE - 2 * 4096  # the result of a kernel that packs in another dtype than it computes in: two 4 KiB tiles, one per frame
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

FORMAT = {"bf16": bh.BF16, "fp32": bh.FP32, "int32": bh.INT32, "bfp8": bh.BFP8, "bfp4": bh.BFP4}
def kernel_config(dtype="bf16", out="bf16", transposing=False):
  """A kernel's backend config. Unpackers read one whole tile each (1024 datums): bf16 into a SrcA/SrcB bank; fp32
  straight into Dst (unpacker 0 in context 1, where the 32-bit Dst row is channel 1's Y). Z picks the slot (the Z stride
  is one tile). The FPU writes Dst in fp32. The packer reads 4 Dst rows per PACR (its Dst row stride is in bytes of
  out, or of fp32 for a block float) and writes them as out (fp32 to bf16 rounds); its channel 1 Y picks the slot (that
  Y stride is one tile of out). Base addresses are in 16-byte units, minus 1, and unpacker 0's rows are biased by 4 (64
  datums). transposing: unpacker 0 transposes each 16x16 face into SrcA."""
  base, fmt, pack = slot_base(dtype) // 16 - 1, FORMAT[dtype], FORMAT[out]
  late = bh.FP32 if out in BFP else pack  # the packer's own format: a block float from fp32 rounds each value once, from bfp8 twice
  return config(Dstacc=late, Fp32_enabled=1, Zero_Flag_disabled_src=0, Read_32b_data=1, Edge_mask=0xFFFF, pack_reads_per_xy_plane=16,
                PCK0_Ystride=16 * ITEMSIZE["fp32" if late == bh.FP32 else out], PCK0_ZW=0, PCK0_Ystride_1=tile_bytes(out),
                PCK0_ZW_1=0, PCK0_Base_1=0, PCK_dest_offset=0,
                L1_Dest_addr=(slot_base(dtype) if out == dtype else OUTS) // 16 - 1, Disable_zero_compress=1, Pack_out_format=pack,
                Exp_section_size=4,  # a block float's 64 exponents, in 16 bytes
                Pack_in_format=late, UNP0_add_dest_addr_cntr=0, UNP0_Ystride_1=64, UNP0_ZW_1=0, UNP1_ZW_1=0,
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
          ({r: 0 for r in bh.THREAD_CFG.values()} | {bh.ADDR_MOD_BIAS + m: 0 for m in range(8)} |
           {bh.ADDR_MOD_PACK: 4} |  # the packer's mode 0 moves on 4 Dst rows a PACR
           {bh.ADDR_MOD_AB + m: a | b << 8 for m, ((a, b), _) in ADDR_MODES.items()} |
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
#   In an fp32 or int32 kernel the math thread fills Dst itself, so it takes SEM_L1 and posts SEM_UNPACKED instead.
UNPACKERS = bh.UNPACK0 | bh.UNPACK1
def counters(mask, n):  # a thread's address counters for these units to 0, each instruction moving n datums
  return [ins("SETADCXY", cnt_set_mask=mask, bit_mask=0b1111), ins("SETADCZW", cnt_set_mask=mask, bit_mask=0b1111),
          ins("SETADCXX", cnt_set_mask=mask, x_end2=n - 1)]
SETUP_UNPACK = [ins("SEMINIT", sem_sel=SEM_DST, max_value=2), ins("SEMINIT", sem_sel=SEM_PACKED, max_value=15),
                ins("SEMINIT", sem_sel=SEM_L1, max_value=15), ins("SEMINIT", sem_sel=SEM_UNPACKED, max_value=15),
                *[ins("SEMINIT", sem_sel=sem, init_value=1, max_value=1) for _, sem in ACCS], *counters(UNPACKERS, 1024)]
SETUP_PACK, PACK_RUNS = counters(bh.PACK, 16), record(PACRS, [ins("PACR") for _ in range(15)])  # 16 datums a Dst row
def field(word, v): return word + v if isinstance(v, int) else (word, v)  # v into the word's low field: a number, or a C
                                                                             # expression, pushed with the word at run time
def setadc(unpacker, slot): return field(ins("SETADC", cnt_set_mask=unpacker, dimension_index=2, value=0), slot)  # its Z: the slot
def unpack(a, b):  # slot a into SrcA and slot b into SrcB, each bank then handed to the FPU
  return [setadc(bh.UNPACK0, a), setadc(bh.UNPACK1, b), ins("UNPACR", last=1, set_dat_valid=1, ovrd_thread_id=1),
          ins("UNPACR", last=1, set_dat_valid=1, ovrd_thread_id=1, unpack_block_selection=1)]
REPORT = [ins("STALLWAIT", stall_res=bh.STALL_SYNC, wait_res=bh.WAIT_UNPACK), ins("SEMPOST", sem_sel=SEM_UNPACKED)]
def fill(slot, row):  # an fp32 tile from an L1 slot straight into Dst rows row on, by unpacker 0 in context 1 (the math thread's)
  return [setadc(bh.UNPACK0, slot), field(ins("SETADC", cnt_set_mask=bh.UNPACK0, dimension_index=1, channel_index=1, value=0), row),
          ins("UNPACR", last=1, ovrd_thread_id=1)]
FILLED = [ins("STALLWAIT", stall_res=bh.STALL_SFPU | bh.STALL_SYNC, wait_res=bh.WAIT_UNPACK0)]  # the SFPU and the sync unit wait for the fills
def reconfigure(**fields):  # unpacker 0's config fields rewritten from thread 0, a byte at a time (RMWCIB keeps the rest
  # of the word): the writes wait for unpacker 0's unpacks so far, its next unpack for the writes
  writes = [ins(f"RMWCIB{b}", cfg_reg_addr=word, data=v << first >> 8 * b & 255, mask=(1 << width) - 1 << first >> 8 * b & 255)
            for (word, first, width), v in ((bh.TENSIX_CFG_FIELDS[k], v) for k, v in fields.items())
            for b in range(first // 8, (first + width + 7) // 8)]
  return ([ins("STALLWAIT", stall_res=bh.STALL_CFG, wait_res=bh.WAIT_UNPACK0)] + writes +
          [ins("STALLWAIT", stall_res=bh.STALL_UNPACK, wait_res=bh.WAIT_CFG)])
def repack(old, new):  # the packer's config from old to new (kernel_config's), from the pack thread: the bytes that
  # differ, written once the packer is idle, and its next pack held until they are
  writes = [ins(f"RMWCIB{b}", cfg_reg_addr=w, data=new[w] >> 8 * b & 255, mask=255) for w in sorted(new) for b in range(4)
            if (old[w] ^ new[w]) >> 8 * b & 255]
  return ([ins("STALLWAIT", stall_res=bh.STALL_CFG, wait_res=bh.WAIT_PACK)] + writes +
          [ins("STALLWAIT", stall_res=bh.STALL_PACK, wait_res=bh.WAIT_CFG)])
def haloize(on): return reconfigure(SEC0_Haloize_mode=on)  # unpacker 0 transposes each 16x16 face into SrcA (or stops)
def bfp_b(wt, slots, region=0, base=None):  # unpacker 0 reads block floats wt from a region of `slots` slots at B8 (a
  # matmul's B: every slot's exponents, then their values; or at base), each value into SrcA as the bf16 it is; with 0,
  fmt, base = (FORMAT[wt], base or B8 + region * slots * tile_bytes(wt)) if slots else (bh.BF16, slot_base("bf16"))  # bf16 from
  return reconfigure(SEC0_format=fmt, SEC0_out_format=fmt, SEC0_z_dim=max(slots, 1), SEC0_base=base // 16 - 1)       # the slots again
def step_unpack(s):  # a matmul step: B's tile from stream slot s into SrcA, and A's next tile in its row into SrcB
  return [setadc(bh.UNPACK0, s), ins("UNPACR", last=1, set_dat_valid=1, ovrd_thread_id=1),
          ins("UNPACR", last=1, set_dat_valid=1, ovrd_thread_id=1, unpack_block_selection=1, addr_mode=1)] + REPORT  # mode 1: Z += 1
WALK = [ins("UNPACR", last=1, set_dat_valid=1, ovrd_thread_id=1, addr_mode=1),  # a step along B's column and A's row, both
        ins("UNPACR", last=1, set_dat_valid=1, ovrd_thread_id=1, unpack_block_selection=1, addr_mode=1)] + REPORT  # kept in L1
def aim(row): return field(ins("SETC16", setc16_reg=bh.THREAD_CFG["MATH_Offset"], setc16_value=0), row)  # the FPU and SFPU at Dst row on
# In the programs, C's h is the first Dst row of the half the next op gets, 0 or DST_HALF: h ^= DST_HALF moves on
def math_begin(zero):  # wait for a free Dst half, aim the FPU at it, and zero it if the math accumulates
  return ([ins("SEMWAIT", stall_res=bh.STALL_ALL, sem_sel=SEM_DST, wait_sem_cond=2), aim("h")] +
          [field(ins("ZEROACC", clear_mode=1, use_32_bit_mode=1, where=b), "h / 16") for b in range(4) if zero])  # 16-row blocks
MATH_END = [ins("STALLWAIT", stall_res=bh.STALL_SYNC, wait_res=bh.WAIT_FPU | bh.WAIT_SFPU), ins("SEMPOST", sem_sel=SEM_DST)]
def pack(row, slot, *done):  # 64 Dst rows from row into an L1 slot, then free them and post done
  return [ins("SEMWAIT", stall_res=bh.STALL_ALL, sem_sel=SEM_DST, wait_sem_cond=1),
          field(ins("SETADC", cnt_set_mask=bh.PACK, dimension_index=1, value=0), row),  # channel 0 Y: Dst row
          field(ins("SETADC", cnt_set_mask=bh.PACK, dimension_index=1, channel_index=1, value=0), slot),  # channel 1 Y: L1 slot
          ins("STALLWAIT", stall_res=bh.STALL_PACK, wait_res=bh.WAIT_FPU | bh.WAIT_SFPU),
          replay(PACRS, 15), ins("PACR", last=1),  # 16 PACRs, 4 Dst rows each through all four read ports
          ins("STALLWAIT", stall_res=bh.STALL_SYNC, wait_res=bh.WAIT_PACK), ins("SEMGET", sem_sel=SEM_DST)] + \
         [ins("SEMPOST", sem_sel=d) for d in done]
TAKE_L1 = [ins("SEMWAIT", stall_res=bh.STALL_ALL, sem_sel=SEM_L1, wait_sem_cond=1), ins("SEMGET", sem_sel=SEM_L1)]

def matmul_phase(transposed=False):
  """One fidelity phase of Dst += A @ B on a pair of tiles, A in SrcB and B in SrcA: MVMUL does
  Dst[8x16] += SrcB[8x16] @ SrcA[16x16]. Output face (i, j) adds A face (i, k) times B face (k, j) for k = 0, 1, in two
  MVMULs of 8 rows (h). In the order below SrcA steps by (16, -16, 16, -16, ...) faces and SrcB by (0, 8, 0, 8, ...)
  rows, so four address modes (4-7) cover it, and the last one wraps both back to row 0 (counters wrap at 64) and
  moves on to the next fidelity phase.
  transposed: B came in as it lies, each face transposed by unpacker 0 (haloize), so B^T's face (k, j) is SrcA's face
  2j + k: SrcA steps by 32 for the next j or h, by -16 for the next k and by 16 for the next i, five modes where four
  did (TRANSPOSED: slots 4-6 and 2 rewritten around the matmul's steps), and no read is split to move faces about."""
  phase = []
  for i, k, h, j in [(i, k, h, j) for i in (0, 1) for k in (0, 1) for h in (0, 1) for j in (0, 1)]:
    last = (i, k, h, j) == (1, 1, 1, 1)
    mode = 4 + (0 if j == 0 else 1 if h == 0 else 3 if last else 2)
    if transposed and mode == 6 and k == 1: mode = 2  # the next i, where the next k is 6
    phase.append(ins("MVMUL", dst=32 * i + 16 * j + 8 * h, addr_mode=mode))
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
CONSTS = bh.LREG_CONSTS  # constant LRegs
ZERO, ONE, MINUS_ONE = CONSTS[0.0], CONSTS[1.0], CONSTS[-1.0]
CC_ON, CC_ALL, CC_OFF = ins("SFPENCC", mod1=10, imm2=3), ins("SFPENCC", mod1=0), ins("SFPENCC", mod1=10, imm2=2)  # lane flags: on, all set, off
NAN, INF = 0x7FC00000, 0x7F800000
def nans(x, t, inf, mask=False):  # LReg t where LReg x is NaN: |x| past inf's bits (LReg inf holds them) in SFPGT's
  # order, as all ones (mask) or as lane flags, for an SFPMOV to give those lanes NaN (between CC_ON and CC_OFF)
  return [*sfp_bits(inf, INF), ins("SFPABS", vd=t, vc=x, mod1=1), ins("SFPGT", vd=t, vc=inf, mod1=8 if mask else 1)]
def at_least(e, n, t): return ins("SFPIADD", vd=t, vc=e, mod1=9, imm12=-n & 0xFFF)  # flags on: keep the lanes where integer LReg e >= n
def below(e, n, t): return ins("SFPIADD", vd=t, vc=e, mod1=1, imm12=-n & 0xFFF)  # or < n (LReg t is overwritten)
def poly(d, x, cs, temp):  # LReg d = cs[0] + cs[1] x + ... by Horner, each coefficient into temp as it is needed (0, +-1 are LRegs)
  words = sfp_const(d, cs[-1])
  for c in cs[-2::-1]: words += ([] if c in CONSTS else sfp_const(temp, c)) + [mad(d, d, x, CONSTS.get(c, temp))]
  return words

# The transcendentals reach fp32's own accuracy (about 1e-7), so an fp32 tensor gets numpy's float32 within a step or two:
# each reduces x to a small range by exact steps, and a polynomial (fit offline to minimax) does the rest.
EXP_P = [1.0, 1.0, 0.49999991, 0.16666421, 0.041668355, 0.008374773, 0.001382954]  # e^r on [-ln2/2, ln2/2], 8e-8
LOG2_Q = [1.442695, -0.72134733, 0.4809106, -0.36070517, 0.2879189, -0.23891294, 0.21565668, -0.20746617, 0.12623915]  # 1.3e-7
SIN_P = [1.0, -0.16666658, 0.0083330506, -0.00019809045, 2.6051646e-06]  # sin(r) / r in r^2, r in [-pi/2, pi/2], 1.4e-7

def exp(base2, nan=False):
  """2^x, or e^x: n = round(t), t = x log2(e) (adding and taking away 1.5 2^23 rounds to an integer, whose bits it
  leaves in the sum's low ones), r = (x - n) ln 2, or for e^x x - n ln 2 in two parts (Cody-Waite: n times 0.693359375 is
  exact), e^r by a polynomial, times 2^floor(n/2) and 2^ceil(n/2), each built from its exponent bits: the multiplies
  overflow to inf, or flush to 0, exactly where the result does. x is first held to where |t| <= 150, which sends a
  NaN to an end; nan (an fp32 kernel's: in bf16 a NaN is inf before the SFPU sees it) puts it back (x again from Dst)."""
  (lim, t), ln2 = (150.0, 0) if base2 else (104.0, 5), math.log(2)
  return (sfp_const(1, ln2) if base2 else sfp_const(1, 1 / ln2) + sfp_const(2, 0.693359375) + sfp_const(3, ln2 - 0.693359375),
          [sfp_load(), sfp_bf16(4, -lim), ins("SFPSWAP", vd=4, vc=0, mod1=1), sfp_bf16(4, lim), ins("SFPSWAP", vd=0, vc=4, mod1=1),  # x within +-lim
           *([] if base2 else [mul(5, 0, 1)]), sfp_bf16(6, 1.5 * 2**23), mad(4, t, ONE, 6), mad(7, 4, ONE, 6, neg=2),  # LReg 7: n
           ins("SFPIADD", vd=6, vc=4, mod1=6),  # LReg 6: n as an integer, bits(t + 1.5 2^23) - bits(1.5 2^23)
           *([mad(5, 7, ONE, 0, neg=1), mul(5, 5, 1)] if base2 else [mad(5, 7, 2, 0, neg=1), mad(5, 7, 3, 5, neg=1)]),  # LReg 5: r
           *poly(4, 5, EXP_P, 7), ins("SFPSHFT", vd=0, vc=6, mod1=7, imm12=-1 & 0xFFF), ins("SFPIADD", vd=6, vc=6, mod1=5, imm12=1),
           ins("SFPSHFT", vd=6, mod1=3, imm12=-1 & 0xFFF),  # LReg 0, 6: floor(n/2), ceil(n/2), then each 2^ that, by its bits
           *[w for r in (0, 6) for w in (ins("SFPIADD", vd=r, vc=r, mod1=5, imm12=127), ins("SFPSHFT", vd=r, mod1=1, imm12=23), mul(4, 4, r))],
           *([sfp_load(0), CC_ON, *nans(0, 6, 7), ins("SFPMOV", vd=4, vc=0), CC_OFF] if nan else []), sfp_store(4)])

def recip(nan=False):
  """1/x: SFPARECIP's 7-bit estimate, then two Newton steps y += y * (1 - x * y). Only where x is a normal number: 0 and
  denormals (estimate inf) and inf (estimate 0) would turn into 0 * inf = NaN, so those lanes keep the estimate, and
  with nan (fp32's) a NaN's (0) is NaN again."""
  return ([], [sfp_load(), ins("SFPARECIP", vd=1, vc=0), ins("SFPEXEXP", vd=3, vc=0),  # LReg 3: unbiased exponent
               CC_ON, below(3, 128, 4), at_least(3, -126, 4),  # the lanes with exponent in [-126, 128)
               mad(2, 0, 1, ONE, neg=1), mad(1, 1, 2, 1), mad(2, 0, 1, ONE, neg=1), mad(1, 1, 2, 1),
               *([CC_ALL, *nans(0, 3, 2), ins("SFPMOV", vd=1, vc=0)] if nan else []), CC_OFF, sfp_store(1)])

def log(scale):
  """log2(x) scale: x = 2^e m with m in [sqrt(1/2), sqrt(2)) (from [1, 2), halved and e one more where above sqrt 2), so
  log2(x) = e + u q(u), u = m - 1, q a polynomial; relative near 1, where u is small and exact. Then the edges: inf and
  NaN give themselves, a negative number NaN, zero (and a denormal, flushed) -inf."""
  return (sfp_const(5, math.sqrt(2)) + (sfp_const(7, scale) if scale != 1 else []),
          [sfp_load(), ins("SFPEXEXP", vd=1, vc=0, mod1=1), ins("SFPSETEXP", vd=2, vc=0, mod1=1, imm12=127),  # LReg 1: e, biased; 2: m
           CC_ON, ins("SFPGT", vd=2, vc=5, mod1=1), ins("SFPSETEXP", vd=2, vc=2, mod1=1, imm12=126), ins("SFPIADD", vd=1, vc=1, mod1=5, imm12=1),
           CC_OFF, ins("SFPCAST", vd=1, vc=1), sfp_bf16(6, -127.0), mad(1, 1, ONE, 6), mad(2, 2, ONE, MINUS_ONE),
           *poly(3, 2, LOG2_Q, 6), mad(3, 3, 2, 1), *([mul(3, 3, 7)] if scale != 1 else []),
           ins("SFPEXEXP", vd=1, vc=0, mod1=1), CC_ON, at_least(1, 255, 4), ins("SFPMOV", vd=3, vc=0), CC_ALL,
           ins("SFPSETCC", vc=0), *sfp_bits(3, NAN), CC_ALL, below(1, 1, 4), *sfp_bits(3, INF | 1 << 31), CC_OFF, sfp_store(3)])

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
           ins("SFPXOR", vd=4, vc=5), CC_ON, at_least(0, 127 + 24, 5), ins("SFPMOV", vd=4, vc=ZERO), CC_ALL, at_least(0, 255, 5),
           *sfp_bits(4, NAN), CC_OFF, sfp_store(4)])

SFPU = {
  # max(x, 0): SFPSWAP's min/max mode leaves the smaller value in vd and the larger in vc; a pick, stored raw for int32
  "relu": ([], [sfp_load(), ins("SFPMOV", vd=1, vc=ZERO), ins("SFPSWAP", vd=1, vc=0, mod1=1), sfp_store(0, RAW)]),
  "exp": exp(False), "exp2": exp(True), "log": log(math.log(2)), "log2": log(1), "sin": sin(0.0), "cos": sin(0.5),
  "fexp": exp(False, True), "fexp2": exp(True, True), "recip": recip(), "frecip": recip(True),
  # sqrt(x) = x / sqrt(x): 1/sqrt(x) by the bit trick (0x5F3759DF - bits / 2, within 3.5%) and three Newton steps
  # y *= 1.5 - x/2 y^2 (two leave 5e-6, 80 ulps), then times x (x + 0 first, so -0 is 0), within 3 ulps. 0, inf and NaN
  # give themselves, a negative number NaN
  "sqrt": (sfp_const(2, struct.unpack("<f", struct.pack("<I", 0x5F3759DF))[0]) + sfp_const(3, 1.5) + sfp_const(4, 0.5),
           [sfp_load(), mad(0, 0, ONE, ZERO), ins("SFPSHFT", vd=1, vc=0, mod1=5, imm12=-1 & 0xFFF),  # LReg 1 = bits of x >> 1
            ins("SFPIADD", vd=1, vc=2, mod1=6), mul(5, 0, 4)] +  # LReg 1 = magic - LReg 1 as integers; LReg 5 = x/2
           [w for _ in range(3) for w in (mul(6, 1, 1), mad(6, 5, 6, 3, neg=1), mul(1, 1, 6))] +
           [mul(1, 1, 0), ins("SFPEXEXP", vd=6, vc=0, mod1=1), CC_ON, at_least(6, 255, 5), ins("SFPMOV", vd=1, vc=0), CC_ALL,
            below(6, 1, 5), ins("SFPMOV", vd=1, vc=0), CC_ALL, ins("SFPSETCC", vc=0), *sfp_bits(1, NAN), CC_OFF, sfp_store(1)]),
}
# Binary ops on the SFPU: x and y each added to zero into Dst (x at the op's Dst half, y 128 rows on), loaded into LRegs
# 0 and 1, the result stored over x. Comparisons give 1.0 or 0 (all ones or 0 from SFPGT, and 1.0's bits), after x + 0
# makes -0 into 0. keep and drop are the halves of where: x if c else 0, and y if not c else 0 (c: not 0), as bits, so
# that an inf in the one not picked does not make a NaN, as c * x would.
def binary(*body): return [sfp_load(0), sfp_load(1, 128), *body]
BINARY = {"max", "cmplt", "cmpne", "keep", "drop"}
# In an fp32 kernel the SFPU does add, sub and mul too, one rounding each, as fp32 does, and NaN goes as IEEE has it
# through exp, the reciprocal and comparisons, which bf16's take no time for: a bf16 NaN is inf before the SFPU sees it
FP32_OPS = {"add": "fadd", "sub": "fsub", "mul": "fmul", "exp": "fexp", "exp2": "fexp2", "recip": "frecip", "cmplt": "fcmplt",
            "cmpne": "fcmpne"}
INT32_OPS = {"add": "iadd", "sub": "isub", "max": "imax", "cmplt": "icmplt", "cmpne": "icmpne", "keep": "keep", "drop": "drop"}  # what int32 does itself
CASTS = {("int32", "fp32"): "itof", ("int32", "bf16"): "itof", ("fp32", "int32"): "ftoi", ("bf16", "int32"): "ftoi"}  # SFPU programs of a cast
SFPU |= {
  "max": ([], binary(ins("SFPSWAP", vd=1, vc=0, mod1=1), sfp_store(0))),
  "cmplt": ([], binary(mad(0, 0, ONE, ZERO), mad(1, 1, ONE, ZERO), ins("SFPGT", vd=1, vc=0, mod1=8), ins("SFPAND", vd=1, vc=ONE), sfp_store(1))),
  "cmpne": ([], binary(mad(0, 0, ONE, ZERO), mad(1, 1, ONE, ZERO), ins("SFPMOV", vd=2, vc=0), ins("SFPGT", vd=2, vc=1, mod1=8),
                       ins("SFPGT", vd=1, vc=0, mod1=8), ins("SFPOR", vd=1, vc=2), ins("SFPAND", vd=1, vc=ONE), sfp_store(1))),
  "fcmplt": ([], binary(mad(0, 0, ONE, ZERO), mad(1, 1, ONE, ZERO), *nans(0, 2, 4, True), *nans(1, 3, 4, True), ins("SFPOR", vd=2, vc=3),
                       ins("SFPGT", vd=1, vc=0, mod1=8), ins("SFPNOT", vd=2, vc=2), ins("SFPAND", vd=1, vc=2), ins("SFPAND", vd=1, vc=ONE),
                       sfp_store(1))),
  "fcmpne": ([], binary(mad(0, 0, ONE, ZERO), mad(1, 1, ONE, ZERO), *nans(0, 3, 4, True), ins("SFPMOV", vd=2, vc=0),
                       ins("SFPGT", vd=2, vc=1, mod1=8), ins("SFPGT", vd=1, vc=0, mod1=8), ins("SFPOR", vd=1, vc=2), ins("SFPOR", vd=1, vc=3),
                       ins("SFPAND", vd=1, vc=ONE), sfp_store(1))),
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
                ins("SFPSHFT", vd=2, vc=1), CC_ON, below(1, -23, 4), ins("SFPMOV", vd=2, vc=ZERO), CC_ALL, ins("SFPSETCC", vc=0),
                ins("SFPIADD", vd=2, vc=ZERO, mod1=6), CC_OFF, sfp_store(2, RAW)]),
  "one": ([], [sfp_store(ONE)]),  # 1.0 everywhere (a tile of ones)
  "none": (sfp_const(1, -1e9), [sfp_store(1)]),  # -1e9 everywhere (a mask that hides every key)
  "floor": (sfp_const(3, -1e4), [sfp_load(), ins("SFPMOV", vd=1, vc=3), ins("SFPSWAP", vd=1, vc=0, mod1=1), sfp_store(0)]),  # max(x, -1e4)
  # an fp32 value rounded to bf16 as the packer rounds it (half up: the bits plus 2^15, the low 16 cleared), still fp32
  "ftob": ([ins("SFPLOADI", vd=2, mod0=2, imm16=0x8000), *sfp_bits(3, 0xFFFF0000)],
           [sfp_load(), ins("SFPIADD", vd=0, vc=2, mod1=4), ins("SFPAND", vd=0, vc=3), sfp_store(0, RAW)]),
}
# Reductions along an axis (0: the batch, to one matrix; 1: rows, to one row; 2: columns, to one column): each tile
# along it is x + 0 into Dst (the first in the reduction's accumulator, the rest 128 rows past it, where the SFPU folds
# them in). Then each row or column of the tile is folded across its 32 and the result written to all of them, so the tile
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
  p, identity = masking(kind)
  p += sfp_bits(7, 14) + [ins("SFPAND", vd=7, vc=15)]
  for base, right, odd in [(b, r, o) for b in (0, 4, 8, 12, 32, 36, 40, 44) for r in (0, 1) for o in (0, 1)]:
    t = max(0, min(8, -(-(n - 16 * right - odd) // 2)))  # the first lane (of 8) at column n or more
    if t < 8: p += [CC_ON, at_least(7, 2 * t, 6),  # lanes t+
                    ins("SFPSTORE", vd=identity, mod0=3, addr_mode=2, imm10=d + base + 16 * right + 2 * odd)]  # right faces: 16 rows on
  return p + [CC_OFF]
def mask_rows(kind, m, d=0):
  """Rows m and on of the tile at Dst row d to the identity, past a tensor's last row. A store covers 4 rows, lane l in
  row l // 8 of them, and LReg 15 holds 2l, so LReg 15 - 16k >= 0 picks the rows from k on."""
  p, identity = masking(kind)
  for r in range(0, 32, 4):  # tile rows r..r+3: Dst rows (r // 16) * 32 + r % 16 on, left face, and 16 more, right face
    k = max(0, m - r)
    if k < 4: p += [w for o in (0, 2, 16, 18) for w in (CC_ON, at_least(15, 16 * k, 6),
                                                     ins("SFPSTORE", vd=identity, mod0=3, addr_mode=2, imm10=d + r // 16 * 32 + r % 16 + o))]
  return p + [CC_OFF]
def mask_rows_at(m, d=0):
  """mask_rows("sum", m, d) for a C expression m, the rows the last tile holds (1 to 31), known when the kernel runs:
  each group of 4 rows' words pushed with the immediates its place gives them."""
  p, identity = masking("sum")
  cut, store = at_least(15, 0, 6), ins("SFPSTORE", vd=identity, mod0=3, addr_mode=2, imm10=0)
  imm12, imm10 = bh.TENSIX_OPS["SFPIADD"][1]["imm12"][0], bh.TENSIX_OPS["SFPSTORE"][1]["imm10"][0]
  return (f"{tt(p)} for (unsigned r = 0; r < 32; r += 4) if ({m} < r + 4) {{ unsigned k = {m} > r ? {m} - r : 0; "
          f"for (unsigned o = 0; o < 4; o++) {{ push_word({CC_ON:#x}); push_word({cut:#x} | (-16 * k & 0xFFF) << {imm12}); "
          f"push_word({store:#x} | ({d} + r / 16 * 32 + r % 16 + o % 2 * 2 + o / 2 * 16) << {imm10}); }} }} {tt([CC_OFF])}")
MASK = {1: mask_rows, 2: mask_columns}  # the padding past the last tile along an axis
MUL_PHASE = [ins("SETRWC", bit_mask=0b0111)] + [ins("ELWMUL", addr_mode=int(i == 7)) for i in range(8)]  # mode 1 steps the phase
MATH_RUNS = {"mm": matmul_phase(), "mmT": matmul_phase(True), "mul": MUL_PHASE} | {u: body for u, (_, body) in SFPU.items()} | COMBINE

def allocate(names):
  """{run: replay index} for the math thread: SFPU bodies (32 replays a tile, longest first), then matmul and mul
  phases (4). What does not fit is pushed word by word."""
  slots, at = {}, 0
  for name in sorted(names, key=lambda n: (n in ("mm", "mmT", "mul"), -len(MATH_RUNS[n]), n)):  # by name last: the same kernel in every process
    if at + len(MATH_RUNS[name]) <= 32: slots[name], at = at, at + len(MATH_RUNS[name])
  return slots
def run(slots, name, count=None):  # the first count words of a run: one REPLAY if it was recorded, else the words
  words = MATH_RUNS[name][:count]
  return [replay(slots[name], len(words))] if name in slots else words
def runs(slots, name, times): return [Loop(times, run(slots, name))]  # a C loop: the math thread runs ahead of its pushes anyway
def math_runs(slots): return [w for name, at in slots.items() for w in record(at, MATH_RUNS[name])]  # recorded at setup

def release(word):  # the same ELW*/MVMUL with clear_dvalid set: both Src banks go back to the unpackers
  first, width = bh.TENSIX_OPS["MVMUL"][1]["clear_dvalid"]
  return word | ((1 << width) - 1) << first
def phased(slots, name, phases):  # a run of MATH_RUNS, once each fidelity phase, the last word releasing both Src banks
  words = MATH_RUNS[name]
  return [ins("SETRWC", bit_mask=0b1111)] + run(slots, name) * (phases - 1) + run(slots, name, len(words) - 1) + [release(words[-1])]
# A transposed B's walk (matmul_phase): (SrcA step, SrcB step) of the slots it rewrites, and what they hold otherwise
TRANSPOSED = {4: (32, 0), 5: (32, 8), 6: (48, 8), 2: (16, 8)}
def walk(transposed):  # the math thread's address modes set for a transposed B's steps, or back
  return [ins("SETC16", setc16_reg=bh.ADDR_MOD_AB + m, setc16_value=a | b << 8)
          for m, (a, b) in ((m, TRANSPOSED[m] if transposed else ADDR_MODES[m][0]) for m in sorted(TRANSPOSED))]

def matmul(slots, phases=4, transposed=False):
  """One matmul step: its first `phases` fidelity phases (SETRWC starts each step at phase 0). A phase multiplies 7 of
  A's 8 significand bits (SrcB) or its last one by 5 of B's (SrcA) or its last 3 (ISA SrcASrcB.md): phase 0 is A's top 7
  by B's top 5, 1 the same by B's last 3, 2 A's last bit by B's top 5, 3 the last by the last. So four make the whole
  product, three leave out under 2^-11 of it, two drop A's last bit (truncating, 2^-7 at most), one keeps 5 bits of B."""
  return phased(slots, "mmT" if transposed else "mm", phases)

def elementwise(op, slots, dst=0):
  """Math for one tile of SrcA op SrcB at Dst row dst: 8 rows of 16 per instruction, address mode 0 steps 8 rows.
  ELWMUL always accumulates and multiplies a slice of the mantissas per fidelity phase: on a zeroed Dst, 4 phases."""
  if op == "mul": return phased(slots, "mul", 4)
  fpu = {"add": "ELWADD", "sub": "ELWSUB"}[op]
  return [ins("SETRWC", bit_mask=0b1111)] + [ins(fpu, dst=dst) for _ in range(7)] + [release(ins(fpu, dst=dst))]

def reduce_step(kind, slots, acc, first, edge=0, src=None):
  """One tile of a reduction, which comes into Dst as SrcA = the tile plus SrcB = zeros, or for fp32 from slot src by a
  fill: the first lands in the accumulator once it is free; later ones land in scratch and the SFPU folds them in.
  edge: how many of the tile's rows or columns along the axis are the tensor's, if it is the line's last and cut short."""
  row, sem = ACCS[acc]
  mask = lambda d: MASK[FOLDS[kind]](kind, edge, d) if edge else []
  put = lambda d: fill(src, row + d) + FILLED if src is not None else elementwise("add", slots, dst=d)
  if first: return ([ins("SEMWAIT", stall_res=bh.STALL_ALL, sem_sel=sem, wait_sem_cond=1), ins("SEMGET", sem_sel=sem), aim(row)] +
                    put(0) + mask(0))
  return ([aim(row)] + put(128) + mask(128) +
          [ins("STALLWAIT", stall_res=bh.STALL_SFPU, wait_res=bh.WAIT_FPU), ins("SETRWC", bit_mask=0b0100)] + runs(slots, kind, 32))
def reduce_finish(kind, slots, acc, posts):  # fold across the tile and run the SFPU ops on the result (then MATH_END)
  wait = ins("SEMWAIT", stall_res=bh.STALL_ALL, sem_sel=SEM_DST, wait_sem_cond=2)
  return [wait, aim(ACCS[acc][0])] + ACROSS[FOLDS[kind]](kind) + sfpu(posts, slots)

def sfpu(posts, slots):  # the SFPU ops, in order, over the Dst half the FPU just wrote
  return [w for u in posts for w in [ins("STALLWAIT", stall_res=bh.STALL_SFPU, wait_res=bh.WAIT_FPU), ins("SETRWC", bit_mask=0b0100)] +
          SFPU[u][0] + runs(slots, u, 32)]

def _array(name, words): return f"static const unsigned {name}[] = {{{', '.join(f'{w:#x}' for w in words)}}};"
BANKS = _array("bank", [bh.noc_xy(x, y) for x, y in bh.DRAM]) + f"""
#define BANK(t) bank[(t) % {len(bh.DRAM)}] & 63, bank[(t) % {len(bh.DRAM)}] >> 6  // tile t's bank's x, y
#define AT(base, t, size) ((base) + (t) / {len(bh.DRAM)} * (size))  // and its address there, of a tensor at base: tiles of size bytes"""

def tt(words):
  """C for a TRISC to push words to its thread as .ttinsn instructions: each word rotated left by 2 and written into
  the code, so the TRISC stalls while the thread's FIFO is full. A Loop becomes a C loop, and a (word, expression) a
  store of the word plus the expression's value, worked out at run time."""
  c, pending = [], []
  for w in list(words) + [None]:
    if isinstance(w, int):
      assert w < 0xC0000000, f"{w:#x} is no Tensix instruction"
      pending.append(f"{(w << 2 | w >> 30) & 0xFFFFFFFF:#x}")
      continue
    if pending: c.append(f'asm volatile(".word {", ".join(pending)}" ::: "memory");')
    pending = []
    if isinstance(w, tuple): c.append(f"push_word({w[0]:#x} + ({w[1]}));")
    elif w is not None: c.append(f'_Pragma("clang loop unroll(disable)") for (unsigned r_ = 0; r_ < {w.times}; r_++) {{ {tt(w)} }}')
  return " ".join(c)

FLAGS_B = 16  # FLAGS: A's flags (a grid row's workers), then from here B's (a grid column's)
DEPTH = 4  # matmul steps whose reads may be in flight at once, each through a stream slot of B's (and of A's, if it streams)
ZERO_SLOT, FRAMES = 0, 1 + 2 * DEPTH  # slot 0 is a zero tile, then B's stream slots and A's, two frames, the narrow ops, and
                                      # what stays in L1: A's rows, B's column or buffers, kept inputs
UNPACKABLE = bh.UNPACKABLE  # the slots an unpacker reads are below this
B8 = slot_base("bf16") + UNPACKABLE * tile_bytes("bf16")  # a matmul's block-float B, past the slots: regions (one, or mc's three
# buffers), each the exponents of its slots, then their values (the unpacker's layout), at most B8_SLOTS slots: the
# unpacker finds a slot's exponents by its first datum, which must stay below bh.BFP_FIRST_DATUM
B8_SLOTS = bh.BFP_FIRST_DATUM // 1024

def shared_k(rt, grid, added=0):  # the longest K, in tiles, of a matmul of rt tile rows, with `added` inputs added to it,
  inputs, ops = 2 + added, 1 + added  # whose tiles the grid's workers share by multicast (mc): A's rows and B's three
  return (UNPACKABLE - FRAMES - 2 * (inputs + ops) - ops) // (-(-rt // min(len(grid[1]), rt)) + 3)  # buffers past the frames

# SYNC (its counters zeroed by the firmware for each launch) holds what BRISC and the TRISCs tell each other, each word
# written by one of them: FETCHED, how many frames BRISC has fetched into L1; ISSUED, how many matmul steps it has
# issued the reads of; UNPACKED, how many frames and steps thread 0 has unpacked, from TRISC0; READY, set by TRISC0
# once the backend config and semaphores are set up; PACKED, how many results the pack thread has packed, from TRISC2,
# with where each goes in RING (four entries [slot, tile]); SENT, how many of those BRISC has sent home.
SYNC_WORDS = f"""
enum {{ {", ".join(name.upper() for name in SYNC_NAMES)}, RING = 8 }};
#define SYNC ((volatile unsigned *){SYNC:#x})
static unsigned peek(unsigned i) {{ asm volatile("fence" ::: "memory"); return SYNC[i]; }}  // fresh: the L0 cache sees no other core's writes
"""
TRISC = SYNC_WORDS + f"""
#define SEM(s) (((volatile unsigned *){bh.TENSIX_SEM:#x})[__builtin_ctz(s)])  // read: its value; write 0: post, 1: take one
// What REPLAY records (fused .ttinsn instructions can corrupt the recording) and words worked out at run time go through stores
static inline void push_word(unsigned w) {{ *(volatile unsigned *){bh.TENSIX_PUSH:#x} = w; }}
static unsigned posted, drained;  // intermediates owed to L1, and those waited for
static void drain(void) {{ for (; drained < posted; drained++) {{ {tt(TAKE_L1)} }} }}  // until everything packed so far has landed
"""
# A Tensix kernel's programs read their arguments through a plain pointer: they are fixed for the launch (and in this
# core's view once the firmware has fenced), so the compiler may keep them in registers and out of loops
KMAIN = "void kmain(volatile unsigned *args) {\n  const unsigned *a = (const unsigned *)args;"
def _pushes(words): return " ".join(f"push_word({w:#x});" for w in words)  # straight-line, with no table to load from

def _brisc(total, body, dtype="bf16", out="bf16", decls="", tail="", b8=0, wt="bfp8", outs=OUTS):  # moves tiles: inputs into L1
  # ahead of the unpackers, results home behind the packer; decls: statics and tables the body needs; tail: its last words;
  # b8: slots at B8 of block floats wt (a matmul's B); outs: where results in another dtype are packed
  tb, to = tile_bytes(dtype), tile_bytes(out)
  return SYNC_WORDS + f"""
#define SLOT(i) ({slot_base(dtype):#x} + (i) * {tb})  // where the unpackers read slot i
#define OUT_SLOT(i) ({slot_base(dtype) if out == dtype else outs:#x} + (i) * {to})  // and where the packer writes it
{BANKS}
static unsigned fetched, events, frame_done[2], stream_done[{DEPTH}], steps, sent, out;
static unsigned rows[16];  // i + 1 of what each place in L1 holds (fresh)
// tile t of a tensor at DRAM address base (BANK, AT)
static void get(unsigned base, unsigned t, unsigned slot) {{ noc_get(BANK(t), AT(base, t, {tb}), SLOT(slot), {tb}); }}
// the same with its two off-diagonal faces swapped, half of transposing it (unpacker 0 does the rest)
static void flipped(unsigned base, unsigned t, unsigned slot) {{
  for (unsigned f = 0; f < 4; f++)
    noc_get(BANK(t), AT(base, t, {tb}) + f * {tb // 4}, SLOT(slot) + (f == 1 ? 2 : f == 2 ? 1 : f) * {tb // 4}, {tb // 4});
}}
static void ready(void) {{ noc_wait(); SYNC[FETCHED] = ++fetched; }}
static void send(void) {{  // the results packed so far go home, in the order they were packed
  for (; sent < peek(PACKED); SYNC[SENT] = ++sent) {{
    unsigned slot = peek(RING + sent % 4 * 2), t = peek(RING + sent % 4 * 2 + 1);
    noc_put(OUT_SLOT(slot), BANK(t), AT(out, t, {to}), {to}); noc_wait();
  }}
}}
// A slot is fetched into again once thread 0 has unpacked what was there: the frame two tiles back, a stream slot DEPTH
// steps back. Thread 0 reports each frame and step in turn; events counts them the same way. Results go home meanwhile.
static void reached(unsigned n) {{ while (peek(UNPACKED) < n) send(); }}
// block-float tile t into slot s at B8, slot s % b8 of region s / b8: its 64 exponents among the region's exponents, its
// values among theirs
static void getb(unsigned base, unsigned t, unsigned s) {{
  unsigned at = AT(base, t, {tile_bytes(wt)}), r = {B8:#x} + s / {max(b8, 1)} * {b8 * tile_bytes(wt)}, z = s % {max(b8, 1)};
  noc_get(BANK(t), at, r + 64 * z, 64);
  noc_get(BANK(t), at + 64, r + {64 * b8} + {BFP[wt]} * z, {BFP[wt]});
}}
// A matmul step: B's tile (for SrcA) into b_slot and A's into a_slot, each a stream slot or its place in the row or column
// kept in L1, unless it is there already (KEPT). how: B as it is (0), transposed (1), a block float (2). Step k's reads carry
// transaction id k % DEPTH + 1, its stream slots': BRISC counts the steps whose reads are out (ISSUED), and thread 0
// waits for that id's reads to land itself, so up to DEPTH steps' reads are in flight. A frame's reads carry id 0
#define KEPT 0xFFFFFFFF
static unsigned read_id;  // the transaction id initiator 1 puts on reads
static void tagged(unsigned id) {{ if (id != read_id) {{ while (NIU1(CMD) & 1) {{}} NIU1(PACKET_TAG) = (read_id = id) << 10; }} }}
static void step(unsigned how, unsigned b, unsigned b_tile, unsigned b_slot, unsigned a, unsigned a_tile, unsigned a_slot) {{
  unsigned s = steps % {DEPTH};
  reached(stream_done[s]);  // the step that last used slot s has been unpacked
  tagged(s + 1);
  if (b_slot != KEPT) {{ if (how == 2) getb(b, b_tile, b_slot); else get(b, b_tile, b_slot); }}  // a transposed B as it lies
  if (a_slot != KEPT) get(a, a_tile, a_slot);
  (void)NIU1(CMD); asm volatile("fence" ::: "memory");  // the NIU has counted the reads before thread 0 hears of them
  stream_done[s] = ++events;
  SYNC[ISSUED] = ++steps;
}}
// A row of A (or column of B) stays in L1, in place q, while the result tiles use it; when tag i, which row, changes,
// the new one comes in once the old is unpacked
static unsigned fresh(unsigned q, unsigned i) {{
  if (rows[q] == i + 1) return 0;
  reached(events);
  return rows[q] = i + 1;
}}
static void kept_step(void) {{ SYNC[ISSUED] = ++steps; ++events; }}  // a matmul step whose tiles are in L1 already
// Tiles shared between workers by multicast. FLAGS: how many shares each worker of this one's grid row (of A's rows),
// then of its grid column (of B's columns) has sent it. Each worker zeroes its own at the end (unflag): by then it has
// seen every flag of the launch, and none is written again before the next go
#define FLAGS ((volatile unsigned *){FLAGS:#x})
static unsigned col_end[3];  // events at the end of the last column in each B buffer
static void await(unsigned i, unsigned n, unsigned v) {{  // until flags i to i + n - 1 reach v; results go home meanwhile
  for (unsigned k = i; k < i + n; k++)
    for (;;) {{ asm volatile("fence" ::: "memory"); if (FLAGS[k] >= v) break; send(); }}
}}
static void unflag(unsigned a, unsigned b) {{
  for (unsigned k = 0; k < a; k++) FLAGS[k] = 0;
  for (unsigned k = 0; k < b; k++) FLAGS[{FLAGS_B} + k] = 0;
  l1_settle({FLAGS:#x});
}}
// n bytes of this L1 to the same place in the `tiles` other tiles of rect (end | start << 12), until all have them. A
// multicast leaves the outstanding count wrong (up once, down once a tile), so nothing else may be in flight: cleared after
static void mcast(unsigned l1, unsigned n, unsigned rect, unsigned tiles) {{
  noc_wait();
  unsigned acks = NIU(WR_ACKS) + (n + MAX_PACKET - 1) / MAX_PACKET * tiles;
  l1_settle(l1);
  for (unsigned k; n; l1 += k, n -= k) noc_issue(l1, noc_me(), l1, rect, MCAST, k = n < MAX_PACKET ? n : MAX_PACKET);
  while (NIU(WR_ACKS) != acks) {{}}
  NIU(CLEAR_OUTSTANDING) = 0xFFFF;
}}
// This worker's part of what a row or column of the grid shares, n bytes at l1 (read in already): to the same place in
// the `tiles` other tiles of rect, then flag, set to v, the same way. Alone, it waits for the reads
static void share(unsigned l1, unsigned n, unsigned rect, unsigned tiles, unsigned flag, unsigned v) {{
  if (tiles && n) mcast(l1, n, rect, tiles);
  else noc_wait();
  FLAGS[flag] = v;
  if (tiles) mcast((unsigned)&FLAGS[flag], 4, rect, tiles);
}}
{decls}
{KMAIN}
  out = a[2];
  unsigned tiles = 0;
{body}
  {tail}
  while (sent < {total}) send();
}}"""

def _unpacker(body, cfg, decls=""):  # TRISC0: the backend config and semaphores, then unpacking, each tile once BRISC has fetched it
  return TRISC + decls + f"""
#define CFG ((volatile unsigned *){bh.TENSIX_CFG:#x})
static unsigned needed, relayed, stepped;
static void relay(void) {{ for (; SEM({SEM_UNPACKED}); SYNC[UNPACKED] = ++relayed) SEM({SEM_UNPACKED}) = 1; }}  // the unpack reports, for BRISC
static void fetched(void) {{ needed++; do relay(); while (peek(FETCHED) < needed); }}  // until BRISC has fetched the next frame
static unsigned landed(void) {{  // until the next matmul step's tiles are in: BRISC has issued its reads, and those (its
  unsigned z = stepped++ % {DEPTH};  // transaction id's) have landed. Its stream slot
  // and no more than the 15 reports SEM_UNPACKED holds are pushed but not relayed: a step kept in L1 waits for no
  // fetch, so thread 0 could fall further behind, and a post onto a full semaphore breaks the contract
  do relay(); while (peek(ISSUED) < stepped || needed + stepped - relayed > 15);
  while (NIU(OUTSTANDING + 4 + 4 * z)) relay();
  return z;
}}
{KMAIN}
  {" ".join(f"CFG[{i}] = {v:#x};" for i, v in cfg.items())}
  asm volatile("fence" ::: "memory");  // the config writes are out; a read from the same block is processed after them
  (void)CFG[{next(iter(cfg))}];
  asm volatile("fence" ::: "memory");  // and has answered, so no instruction pushed below can see them stale
  {tt(thread() + SETUP_UNPACK)}
  tensix_idle();  // the semaphores are set before the other threads can touch them
  SYNC[READY] = 1;
  unsigned tiles = 0;
{body}
  while (relayed < needed) relay();  // every report taken, whichever thread unpacked
}}"""

def _math(slots, body, dst=False, decls=""):  # TRISC1; dst: it unpacks into Dst itself, so it waits for BRISC's fetches too
  return TRISC + decls + f"""
static unsigned needed;
static void fetched(void) {{ needed++; while (peek(FETCHED) < needed) {{}} }}  // until BRISC has fetched the next tiles
{KMAIN}
  {tt(thread(dst) + (counters(bh.UNPACK0, 1024) if dst else []) + [CC_OFF])} {_pushes(math_runs(slots))}
  while (!peek(READY)) {{}}  // the backend config and the semaphores
  unsigned tiles = 0, h = 0;  // h: the Dst half the next op gets, by its first row
{body}
}}"""

def _packer(body, slots=2):  # TRISC2; slots: output slots, each refilled once its last result has gone home
  return TRISC + f"""
static unsigned outputs, relayed, used[{slots}];
static void relay(void) {{ for (; SEM({SEM_PACKED}); SYNC[PACKED] = ++relayed) SEM({SEM_PACKED}) = 1; }}  // packed results, for BRISC
// The next result is packed into slot (output slot s: one per frame, or one for all) and goes home as tile t. Once the
// last result packed there has gone home, and RING has room (at most four wait to go home), tell BRISC in RING.
static void output(unsigned s, unsigned slot, unsigned t) {{
  while (peek(SENT) < used[s] || peek(SENT) + 4 <= outputs) relay();
  SYNC[RING + outputs % 4 * 2] = slot; SYNC[RING + outputs % 4 * 2 + 1] = t;
  used[s] = ++outputs;
}}
{KMAIN}
  {tt(thread() + SETUP_PACK)} {_pushes(PACK_RUNS)}
  while (!peek(READY)) {{}}
  unsigned tiles = 0, h = 0;  // h: the Dst half the next op gets, by its first row
{body}
  while (relayed < outputs) relay();
}}"""

def fused_kernel(expr, shapes, dtype="bf16", out=None, grid=None, reps=None):
  """A kernel for an expression: nested tuples, (op, x, y) for add, sub, mul and the binary SFPU ops, (op, x) for the
  unary ones, the reductions (FOLDS: sum0, max2, ...) and cast, and ("matmul", a, b[, phases]) at `phases` fidelity
  phases (4 unless said), where a is an input and b an input, ("T", input) for one read transposed, or ("bfp8", input)
  or ("bfp4", input) for a block float (unpacker 0 makes it bf16 on its way into SrcA); an input is a number, its place
  in shapes. shapes: each input's (batch, rows, columns), laid out as Tensor says; a dimension of 1 repeats across the
  others (broadcasting). dtype: what the inputs are and the kernel computes in; out: what its root is written as (a
  cast, or a matmul written as fp32: the packer is switched to it for the root's pack); grid: the workers as
  Runtime.grid has them, for a matmul to share its tiles between them; reps: each input's repeat
  (Tensor.repeat_interleave: its matrix b is the one stored at b // rep), 1 unless said.
  bf16 goes through SrcA and SrcB and the FPU. fp32 and int32 cannot (they hold 19 bits), so the math thread puts each
  operand straight into Dst (fill) and the SFPU does all the math, add and sub too; there is no matmul.
  args: [first unit, units, then the result's and each input's DRAM address, tile strides for batch and row, and tile
  offset (Tensor's), then the rows the result's last tile row holds (0: 32)]: rows past them are zeros whatever the
  shapes say, so one kernel serves every row count in the same tiles. A unit is a tile of the result, or a block of them (one matmul and nothing that reduces: up to 4
  x 4, or given a grid, a worker's block of it), or, if the expression reduces along an axis, a whole line of tiles
  along it (a row, a column, or one tile position through the batch), so the reduction can stay on chip.

  Every op computes into Dst and is packed into an L1 slot: elementwise ops from their operands' slots, a matmul from
  its K pairs of tiles, B's streamed through DEPTH slots (up to DEPTH steps' reads in flight) and A's row kept in L1
  while the result tiles stay in its row (if it fits below slot 256, else streamed too). A matmul with more than one row
  of tiles, given a grid, reads each input tile from DRAM once: worker (gy, gx) of the grid computes one block, each
  worker of a grid row reads its share of the block row's rows of A and multicasts it along the row, and each of a grid
  column its share of each of the column's columns of B, down the column into one of three buffers, flagging each share
  (FLAGS) once every recipient has it. A unary op runs on the SFPU over Dst: in place, on the result of the op below it
  if nothing else reads that, or else on its operand plus the zero tile (slot 0). A reduction folds each tile of its
  operand along the axis into an accumulator in Dst, then across the tile (across_rows, across_cols), and packs one
  tile.
  A line is done in passes: a pass is a sweep over its tiles, and a reduction's result can only be read in the pass
  after the one that swept it. Each pass recomputes what it needs (softmax computes exp twice) rather than keep whole
  lines of it in L1, but reads inputs from DRAM once: one read in more than one pass keeps its line of tiles in L1 from
  the first, while they fit. Values one tile long along the axis (reductions, ops on them, inputs of 1 there) are
  worked out once per line, in the first tile of the pass that needs them, into slots of their own.
  Two frames of slots alternate between tiles: inputs, then one slot per op. Every intermediate goes to L1 in the
  kernel's dtype (bf16 rounded), exactly as if each op ran on its own; in-place SFPU ops see Dst's result unrounded.
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
  ops, refs = [], {}  # ops: dicts with kind, x, y, posts, shape (a matmul's also how, bt, wt, kt, phases); refs: expr -> input number or ("op", j)
  size = lambda r: shapes[r] if isinstance(r, int) else ops[r[1]]["shape"]  # (batch, rows, columns)
  def visit(e):
    if isinstance(e, int): return e
    if e in refs: return refs[e]
    if e[0] == "matmul":  # B is an input, or (how, input): ("T", input) that input transposed, ("bfp8", input) a block float,
      how, y, *ids = e[2] if isinstance(e[2], tuple) else ("", e[2])  # ("Tbfp8", input) a block float transposed (a bfp8 key
      how, ids = how.removeprefix("I"), ids[0] if ids else None  # cache); ("I" + how, input, ids) for each batch entry b the matrix
      a, bt, wt = shapes[e[1]], how.startswith("T"), how.removeprefix("T") or None  # of B the int32 input ids picks (b // k, b % k)
      batch = shapes[ids][1] * shapes[ids][2] if ids is not None else max(a[0], shapes[y][0])
      op = dict(kind="matmul", x=e[1], y=y, ids=ids, how=2 if wt else int(bt), bt=bt, wt=wt, kt=ext(a, 2),  # how: B fetched as a bf16 tile (0, 1) or a block float's (2)
                shape=(batch, a[1], shapes[y][1 if bt else 2]), phases=e[3] if len(e) > 3 else 4)
    elif e[0] in COMBINE:
      x = visit(e[1])
      op = dict(kind=e[0], x=x, shape=tuple(1 if ax == FOLDS[e[0]] else s for ax, s in enumerate(size(x))))
    elif e[0] == "cast" or e[0] in SFPU and e[0] not in BINARY:  # a unary op, or the root's cast (packed as out, after an
      x = visit(e[1])                                              # SFPU program between floats and ints, if need be)
      posts = ([CASTS[dtype, out]] if (dtype, out) in CASTS else []) if e[0] == "cast" else [FP32_OPS.get(e[0], e[0]) if dtype == "fp32" else e[0]]
      if isinstance(x, tuple) and uses[e[1]] == 1:  # in place, on the result of the op below (a bf16 cast rounds it
        ops[x[1]]["posts"] += ["ftob"] * (e[0] == "cast" and dtype == "bf16") + posts  # first, as its own pack would)
        refs[e] = x
        return x
      op = dict(kind="add", x=x, y="zero", shape=size(x), posts=posts)  # operand + 0 into Dst, then the SFPU
    else:
      x, y = visit(e[1]), visit(e[2])
      op = dict(kind=e[0], x=x, y=y, shape=tuple(map(max, size(x), size(y))))
    ops.append({"posts": [], **op})
    refs[e] = ("op", len(ops) - 1)
    return refs[e]
  root = visit(expr)[1]
  # a root packed as another dtype than the rest: the packer switched to it for the root's pack and back, unless it is alone
  config = kernel_config(dtype, out if len(ops) == 1 else dtype)
  root_config = kernel_config(dtype, out)
  around = (tt(repack(config, root_config)), tt(repack(root_config, config))) if out != dtype and len(ops) > 1 else ("", "")
  operands = lambda op: [r for r in (op.get("x"), op.get("y")) if r not in (None, "zero")] if op["kind"] != "matmul" else []
  reduced = lambda r: isinstance(r, tuple) and ops[r[1]]["kind"] in COMBINE
  axes = {FOLDS[op["kind"]] for op in ops if op["kind"] in COMBINE}
  assert len(axes) <= 1, "a kernel reduces along one axis"
  reducing, axis = bool(axes), next(iter(axes), 2)  # the axis swept, in passes; otherwise 2, and units are blocks of tiles
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
  ops_of = {"fp32": FP32_OPS, "int32": INT32_OPS}.get(dtype, {})
  sfpu_name = lambda op: ops_of.get(op["kind"], op["kind"] if dtype != "int32" else None)  # a binary op's program
  sfpu_op = lambda op: op["kind"] in BINARY or (dst and op.get("y", "zero") != "zero")
  assert dtype != "int32" or all(sfpu_name(op) for op in ops if sfpu_op(op)), "an int32 kernel adds, subtracts, compares and picks"
  slots = allocate({"mmT" if op["bt"] else "mm" for op in ops if op["kind"] == "matmul"} | {"mul" for op in ops if op["kind"] == "mul" and not dst} |
                   {u for op in ops for u in op["posts"]} | {op["kind"] for op in ops if op["kind"] in COMBINE} |
                   {sfpu_name(op) for op in ops if sfpu_op(op)})
  frame, matmuls = inputs + len(ops), [j for j, op in enumerate(ops) if op["kind"] == "matmul"]
  assert not (dst and matmuls), "only bf16 kernels have matmuls"
  base = FRAMES + 2 * frame + len(ops)  # the first slot past the frames and the narrow ops
  rows_at, top = [], base  # each matmul's row of A in L1, while they fit; else A streams too
  assert top <= UNPACKABLE and slot_base(dtype) + top * tile_bytes(dtype) <= OUTS and len(matmuls) <= 16, "too many tiles in a kernel"
  for j in matmuls:
    fits = top + ops[j]["kt"] <= UNPACKABLE
    rows_at.append(top if fits else None)
    top += ops[j]["kt"] if fits else 0
  me = ("op", root)
  rows, cols = size(me)[1:]  # the result's
  # With one matmul and no passes, a unit is a block of bp x bq result tiles, column by column: the block's bp rows of
  # A and its current column of B stay in L1, so each is read from DRAM once a block, not once a tile. Given a grid,
  # the blocks are the grid's (mc: its columns and rows in use), and the tiles go round by multicast
  mc = None
  if grid and len(matmuls) == 1 and not reducing and nb == 1 and rt > 1 and ext(ops[matmuls[0]]["shape"], 2) == w:
    kt = ops[matmuls[0]]["kt"]
    bp, bq = -(-rt // min(len(grid[1]), rt)), -(-w // min(len(grid[0]), w))
    n_grid = (-(-w // bq), -(-rt // bp))
    room = kt <= B8_SLOTS if ops[matmuls[0]]["how"] == 2 else base + (bp + 3) * kt <= UNPACKABLE  # B's three buffers
    if base + bp * kt <= UNPACKABLE and room and n_grid[0] * n_grid[1] > 1 and max(n_grid) <= FLAGS_B: mc = n_grid
  # With one row of tiles (a token's matmul), the grid's workers each take a block of the result's columns and the first
  # grid row reads A's tiles in shares and multicasts them to all (bcast), so each comes from DRAM once, not once a
  # worker; B, which no two workers share, streams as it would. Each matrix of a batch (K split in parts) gets whole grid
  # rows of its own, its blocks padded out to them
  bcast = None
  if (grid and not mc and len(matmuls) == 1 and not reducing and rt == 1 and ext(ops[matmuls[0]]["shape"], 2) == w and
      rows_at[0] is not None and nb <= len(grid[1])):
    per = len(grid[1]) // nb * len(grid[0])  # workers for each matrix
    bp, bq = 1, -(-w // min(per, w))
    blocks = -(-w // bq)
    gx_n = len(grid[0]) if nb > 1 else min(len(grid[0]), blocks)
    if nb * blocks > 1: bcast = (gx_n, -(-blocks // gx_n))
  # One tile row of A that every matrix of a bigger batch reads (a decode step's tokens, through each expert some token
  # picked): read once, in shares by the grid's first row and multicast to the whole grid, rather than once a worker (a
  # worker outside the grid reads it itself). Every grid worker takes part, so the batch has a unit for each
  shared = (grid and not mc and not bcast and len(matmuls) == 1 and not reducing and rt == 1 and nb > 1 and rows_at[0] is not None
            and size(ops[matmuls[0]]["x"])[0] == 1 and nb * w >= len(grid[0]) * len(grid[1]))
  if not mc and not bcast:
    bp = min(4, rt) if len(matmuls) == 1 and not reducing else 1
    while bp > 1 and (base + (bp + 1) * ops[matmuls[0]]["kt"] > UNPACKABLE or ops[matmuls[0]]["how"] == 2 and ops[matmuls[0]]["kt"] > B8_SLOTS):
      bp -= 1
    bq = min(4, w) if bp > 1 else 1
  if bp > 1 or mc: rows_at = [base]  # A's bp rows, then B's column (or its three buffers)
  kept_b = rows_at[0] + bp * ops[matmuls[0]]["kt"] if bp > 1 or mc else None
  # a block-float B goes to slots of its own at B8 (a region for each of mc's three buffers, or one for a block's column,
  # or for DEPTH to stream through), unpacker 0 switched to them for its steps
  b8 = (ops[matmuls[0]]["kt"] if kept_b is not None else DEPTH) if any(ops[j]["how"] == 2 for j in matmuls) else 0
  wts = {ops[j]["wt"] for j in matmuls if ops[j]["wt"]}  # the block floats B comes in: one kind a kernel
  assert len(wts) <= 1, "one block float a kernel"
  wt = next(iter(wts), "bfp8")
  # blocks down and across a matrix (the grid's)
  br, bc = (mc[1], len(grid[0])) if mc else (1, bcast[0] * bcast[1]) if bcast else (-(-rt // bp), -(-w // bq))
  decls, tail = "", ""  # BRISC's tables and statics for a multicast matmul or kept inputs, and its last words
  if mc:
    (gx_n, gy_n), (xs, ys) = mc, grid
    decls = (_array("row_rect", [bh.noc_rect((xs[0], y), (xs[gx_n - 1], y)) for y in ys[:gy_n]]) +
             _array("col_rect", [bh.noc_rect((x, ys[0]), (x, ys[gy_n - 1])) for x in xs[:gx_n]]))
    tail = f"unflag({gx_n}, {gy_n});"
  def plan(p):  # pass p's reductions, the swept ops it computes (operands first), and the narrow ops it works out once
    reductions, swept_ops = [j for j in sweep if sweep[j] == p], []
    def need(r):
      if isinstance(r, tuple) and swept(r) and r[1] not in swept_ops:
        for o in operands(ops[r[1]]): need(o)
        swept_ops.append(r[1])
    for r in [ops[j]["x"] for j in reductions] + ([me] if p == last and swept(me) else []): need(r)
    return reductions, swept_ops, [j for j, op in enumerate(ops) if not swept(("op", j)) and op["kind"] not in COMBINE and ready[j] == p]
  plans = [plan(p) for p in range(last + 1)]
  reads = [{r for j in sum(pl, []) for r in operands(ops[j]) if isinstance(r, int)} for pl in plans]  # the inputs each pass reads
  # In passes, an input one tile along the sweep gets a slot of its own, fetched once a line, and one read in more than one
  # pass keeps its line of tiles in L1 (while they fit below slot 256), fetched in the first: neither comes again from DRAM
  kept, passes_of, narrow = {}, lambda r: [p for p in range(last + 1) if r in reads[p]], lambda r: ext(shapes[r], axis) == 1
  for r in range(inputs) if reducing else ():
    if narrow(r) and passes_of(r) and top < UNPACKABLE: kept[r], top = top, top + 1
    elif not narrow(r) and len(passes_of(r)) > 1 and top + L <= UNPACKABLE: kept[r], top = top, top + L
  last_read = lambda keep: max([passes_of(r)[-1] for r in kept if narrow(r) == keep], default=None)  # the last pass reading such
  # In passes, a matmul computed in more than one (scores under a softmax) keeps the line's tiles of its B in L1 from the
  # first, as a kept input does, rather than read them again each pass (one such matmul, A's row kept, while they fit)
  line_b, computed = None, [None]
  for j in matmuls:
    mine = [p for p, pl in enumerate(plans) if j in pl[1]]
    if (reducing and line_b is None and len(mine) > 1 and ops[j]["how"] != 2 and rows_at[matmuls.index(j)] is not None and
        top + L * ops[j]["kt"] <= UNPACKABLE): line_b, computed, lines_at, top = j, mine, top, top + L * ops[j]["kt"]
  if line_b is not None: decls += f"static unsigned line_done[{L}];\n"
  single = {}  # without passes, an input of one tile in all (a number) gets a slot of its own too, fetched once a worker
  if kept_b is not None: top = max(top, kept_b + (3 if mc else 1) * ops[matmuls[0]]["kt"])  # past A's rows and B's buffers
  for r in range(inputs) if not reducing else ():
    if all(ext(shapes[r], ax) == 1 for ax in range(3)) and top < UNPACKABLE: single[r], top = top, top + 1
  def slot(r):  # an input or swept op lives in the tile's frame (C's fr: 0 or 1), unless kept; a narrow op in a slot of its
    if r == "zero": return ZERO_SLOT  # own after both frames
    if r in single: return single[r]
    if isinstance(r, int) and r in kept: return kept[r] if narrow(r) else f"{kept[r]} + k"
    if isinstance(r, int) or swept(r): return f"{FRAMES + (r if isinstance(r, int) else inputs + r[1])} + fr * {frame}"
    return FRAMES + 2 * frame + r[1]
  arg = lambda i: 6 + 4 * i  # input i's [address, batch stride, row stride, offset] in a, after the result's
  def which(r):  # which of r's matrices goes with the result's b: b, a repeat's b / rep, or 0 of a batch of 1
    if size(r)[0] == 1: return "0"
    return f"b / {reps[r]}" if reps and isinstance(r, int) and reps[r] > 1 else "b"
  def matrix(r, base):  # the batch's part of r's tile number
    return "" if size(r)[0] == 1 else f"{which(r)} * a[{base + 1}] + "
  def where(r, base):  # which tile of r (an input or the result, its [address, strides, offset] at a[base]) goes with b, i_, j
    s = size(r)
    i, j = "i_" if ext(s, 1) > 1 else 0, "j" if ext(s, 2) > 1 else 0
    return f"{matrix(r, base)}{i} * a[{base + 2}] + {j} + a[{base + 3}]"
  tile = lambda i: where(i, arg(i))
  B, U, M, P = range(4)  # the programs: BRISC moves tiles; TRISC 0, 1, 2 feed the unpack, math and pack threads
  def math_end(j):  # the math thread's last words for op j: the root's padding in Dst goes to 0 first, in a tile at the edge
    if j != root: return tt(MATH_END)
    at_end = lambda ax: f"{('i_', 'j')[ax - 1]} == {ext(size(me), ax) - 1}"
    edge = [f"if ({at_end(2)}) {{ {tt(mask_columns('sum', cols % 32))} }}"] * (cols > 1 and cols % 32 > 0)
    edge += [f"if ({at_end(1)} && a[{arg(len(shapes))}]) {{ {mask_rows_at(f'a[{arg(len(shapes))}]')} }}"] * (rows > 1)  # (an argument)
    return " ".join(edge + [tt(MATH_END)])
  def pslot(r):  # where the packer puts r: its slot, or for a result in another dtype, one at OUTS
    return ("fr" if swept(me) else 0) if out != dtype and r == me else slot(r)
  def output(j):  # for the pack thread: where op j's result goes, if it is the root
    return f"output({'fr' if swept(me) else 0}, {pslot(me)}, {where(me, 2)}); " if j == root else ""
  def fetch(p, r):  # BRISC's fetch of input r in pass p: into the frame, or into its kept slot in the first pass reading it,
    if r in single: return f"if (tiles == 1) get(a[{arg(r)}], {tile(r)}, {single[r]});"  # with the first tile (tiles counts it already)
    if r not in kept: return f"get(a[{arg(r)}], {tile(r)}, {slot(r)});"  # once the line before has been read there
    if p != passes_of(r)[0]: return ""
    return (f"if (k == 0) {{ reached(narrow_done); get(a[{arg(r)}], {tile(r)}, {kept[r]}); }}" if narrow(r) else
            f"reached(kept_done[k]); get(a[{arg(r)}], {tile(r)}, {kept[r]} + k);")
  if kept: decls += "static unsigned narrow_done;  // events once the line before's last read of a kept slot is done\n"
  if any(not narrow(r) for r in kept): decls += f"static unsigned kept_done[{L}];\n"
  picked = [ops[j]["ids"] for j in matmuls if ops[j]["ids"] is not None]  # B's matrices picked by an int32 tensor, in L1 below OUTS
  indices = ""
  if picked:
    (r,), k, ib = picked, shapes[picked[0]][2], tile_bytes("int32")
    it = -(-shapes[r][1] // 32)  # (their tiles)
    assert shapes[r][0] == 1 and k <= 32 and OUTS - it * ib >= B8 + 3 * B8_SLOTS * tile_bytes("bfp8"), "the indices in a column of tiles below OUTS"
    indices = (f"#define IDS {OUTS - it * ib:#x}\nstatic unsigned ids(unsigned b) {{ unsigned t = b / {k}, j = b % {k};  // index (t, j)\n"
               f"  return ((volatile unsigned *)IDS)[t / 32 * 1024 + (t % 32 / 16 * 2 + j / 16) * 256 + t % 16 * 16 + j % 16]; }}\n")
  passes = []
  for p, (reductions, swept_ops, once) in enumerate(plans):
    c = [[], [], [], []]  # each program's C for a tile
    c[B] += ["reached(frame_done[fr]);"] + [fetch(p, r) for r in sorted(reads[p])] + ["ready();"]
    c[U].append("fetched();")
    if dst or picked: c[M].append("fetched();")  # (a picked matmul's math reads the indices, there once the first frame is)
    for j in once + swept_ops:
      op, me_j = ops[j], ("op", j)
      cond, posted = "if (k == 0) " if j in once else "", "" if j == root else " posted++;"
      pk = tt(pack("h", pslot(me_j), SEM_PACKED if j == root else SEM_L1))
      pk = f"{around[0]} {pk} {around[1]}" if j == root else pk
      c[P].append(f"{cond}{{ {output(j)}{pk} h ^= {DST_HALF}; }}")
      if op["kind"] == "matmul":
        kt, bw, q, how = op["kt"], ext(op["shape"], 2), matmuls.index(j), op["how"]  # B's width in tiles; which matmul
        bt, bs = op["bt"], 0 if how == 2 else 1  # B transposed; B's first stream slot (a block float's are at B8)
        x, y = arg(op["x"]), arg(op["y"])
        bj = 'j' if bw > 1 else 0  # B's tile (m, j), or for B transposed the input's tile (j, m)
        in_batch = f"ids(b) * a[{y + 1}] + " if op["ids"] is not None else matrix(op["y"], y)  # B's matrix: picked, or the batch's
        b_which = "ids(b)" if op["ids"] is not None else which(op["y"])  # (and which: a row or column kept in L1 is tagged by its own)
        live = f"if (ids(b) < {shapes[op['y']][0]}) " if op["ids"] is not None else ""  # an entry picking no matrix: zeros, nothing read
        b_tile = f"{in_batch}{bj} * a[{y + 2}] + m + a[{y + 3}]" if bt else f"{in_batch}m * a[{y + 2}] + {bj} + a[{y + 3}]"
        on = tt(haloize(1) * bt + (bfp_b(wt, b8) if how == 2 else []))  # unpacker 0 transposing faces, reading a block float
        off = tt(haloize(0) * bt + (bfp_b(wt, 0) if how == 2 else []))
        if mc and how == 2: on = tt(haloize(1) * bt) + " " + " else ".join(f"if (jj % 3 == {r}) {{ {tt(bfp_b(wt, b8, r))} }}" for r in range(3))
        a_tile = f"{matrix(op['x'], x)}i_ * a[{x + 2}] + m + a[{x + 3}]"
        buf = mc and f"{0 if how == 2 else kept_b} + jj % 3 * {kt}"  # with mc, B's buffer for column jj (a block float's in B8)
        if mc:  # A's rows and B's columns come by multicast (see mc), column jj into B's buffer jj % 3
          gx_n, gy_n = mc
          gx, gy, a_n = f"unit % {bc}", f"unit / {bc}", f"(i0 + {bp} < {rt} ? {bp} : {rt} - i0) * {kt}"
          part = lambda n, k, parts: (f"{k} * {n} / {parts}", f"({k} + 1) * {n} / {parts}")  # worker k's share of n tiles
          (a0, a1), (b0, b1) = part(a_n, gx, gx_n), part(kt, gy, gy_n)
          get_a = (f"for (unsigned e = {a0}; e < {a1}; e++) "
                   f"{{ unsigned i_ = i0 + e / {kt}, m = e % {kt}; get(a[{x}], {a_tile}, {rows_at[q]} + e); }} "
                   f"share(SLOT({rows_at[q]} + {a0}), ({a1} - {a0}) * {tile_bytes(dtype)}, row_rect[{gy}], {gx_n - 1}, {gx}, 1);")
          get_b = f"for (unsigned m = {b0}; m < {b1}; m++) {('get', 'get', 'getb')[how]}(a[{y}], {b_tile}, {buf} + m); "
          share_b = lambda l1, n: f"share({l1}, ({b1} - {b0}) * {n}, col_rect[{gx}], {gy_n - 1}, {FLAGS_B} + {gy}, jj + 1);"
          if how == 2:  # in region jj % 3 of B8: the share's exponents, then its values
            region = f"{B8} + jj % 3 * {kt * tile_bytes(wt)}"
            get_b += (f"if ({gy_n - 1} && {b1} > {b0}) mcast({region} + 64 * ({b0}), ({b1} - {b0}) * 64, col_rect[{gx}], {gy_n - 1}); " +
                      share_b(f"{region} + {64 * kt} + {BFP[wt]} * ({b0})", BFP[wt]))
          else: get_b += share_b(f"SLOT({buf} + {b0})", tile_bytes(dtype))
          c[B].append(f"{{ if (ii == 0) {{ "  # a column's first tile: its column of B, and in the first column A's rows
                      f"if (jj == 0) {{ {get_a} }} "
                      # done with column jj - 2 here: this column's flag lets the others fill its buffer with jj + 1
                      f"if (jj >= 2) reached(col_end[(jj + 1) % 3]); "
                      f"{get_b} await({FLAGS_B}, {gy_n}, jj + 1); if (jj == 0) await(0, {gx_n}, 1); }} "
                      f"for (unsigned m = 0; m < {kt}; m++) kept_step(); if (ii + 1 == {bp} || i_ + 1 == {rt}) col_end[jj % 3] = events; }}")
        elif kept_b is not None:  # A's bp rows and B's column in L1
          c[B].append(f"{cond}{live}{{ unsigned na = fresh(i_ % {bp}, {which(op['x'])} * {rt} + i_), nb = fresh(15, {b_which} * {w} + j); "
                      f"for (unsigned m = 0; m < {kt}; m++) step({how}, a[{y}], {b_tile}, nb ? {0 if how == 2 else kept_b} + m : KEPT, "
                      f"a[{x}], {a_tile}, na ? {rows_at[q]} + i_ % {bp} * {kt} + m : KEPT); tagged(0); }}")
        if kept_b is not None:  # unpacker 0 walks B's column, 1 the row of A
          column = 0 if how == 2 else buf if mc else kept_b  # a block-float B's from the start of its region
          c[U].append(f"{cond}{{ {tt([setadc(bh.UNPACK1, f'{rows_at[q]} + i_ % {bp} * {kt}'), setadc(bh.UNPACK0, column)])} "
                      f"{on} {live}for (unsigned m = 0; m < {kt}; m++) {{ landed(); {tt(WALK)} }} {off}{posted} }}")
        elif rows_at[q] is None:  # A's tile streams in beside B's, each step
          c[B].append(f"{cond}{{ {live}for (unsigned m = 0; m < {kt}; m++) "
                      f"step({how}, a[{y}], {b_tile}, {bs} + steps % {DEPTH}, a[{x}], {a_tile}, {1 + DEPTH} + steps % {DEPTH}); tagged(0); }}")
          c[U].append(f"{cond}{{ {on} {live}for (unsigned m = 0; m < {kt}; m++) {{ unsigned z = landed(); "
                      f"{tt(unpack(f'{bs} + z', f'{1 + DEPTH} + z') + REPORT)} }} {off}{posted} }}")
        elif j == line_b:  # B's line in L1 across passes: fetched in the first pass that computes it, once the line before
          line = f"{lines_at} + k * {kt}"  # has been done with it; A's row kept as below
          if p == computed[0]:
            c[B].append(f"{{ reached(line_done[k]); unsigned new = fresh({q}, {which(op['x'])} * {rt} + i_); for (unsigned m = 0; m < {kt}; m++) "
                        f"step({how}, a[{y}], {b_tile}, {line} + m, a[{x}], {a_tile}, new ? {rows_at[q]} + m : KEPT); tagged(0); }}")
          else: c[B].append(f"for (unsigned m = 0; m < {kt}; m++) kept_step();")
          c[U].append(f"{{ {tt([setadc(bh.UNPACK1, rows_at[q]), setadc(bh.UNPACK0, line)])} {on} "
                      f"for (unsigned m = 0; m < {kt}; m++) {{ landed(); {tt(WALK)} }} {off}{posted} }}")
        else:
          new = "0" if bcast or shared else f"fresh({q}, {which(op['x'])} * {rt} + i_)"  # (bcast's, shared's: A's row is in already)
          c[B].append(f"{cond}{live}{{ unsigned new = {new}; for (unsigned m = 0; m < {kt}; m++) "
                      f"step({how}, a[{y}], {b_tile}, {bs} + steps % {DEPTH}, a[{x}], {a_tile}, new ? {rows_at[q]} + m : KEPT); tagged(0); }}")
          c[U].append(f"{cond}{{ {tt([setadc(bh.UNPACK1, rows_at[q])])} {on} {live}for (unsigned m = 0; m < {kt}; m++) {{ "
                      f"unsigned z = landed(); {tt(step_unpack(f'{bs} + z'))} }} {off}{posted} }}")
        c[M].append(f"{cond}{{ {tt(math_begin(True) + walk(True) * bt)} "  # a transposed B's walk, then the modes back
                    f"{live}for (unsigned m = 0; m < {kt}; m++) {{ {tt(matmul(slots, op['phases'], bt))} }} "
                    f"{tt(walk(False) * bt + sfpu(op['posts'], slots))} {math_end(j)} h ^= {DST_HALF}; }}")
        continue
      drain = "drain(); " if any(isinstance(r, tuple) for r in operands(op)) else ""
      x, y = slot(op["x"]), slot(op["y"])
      prog = sfpu([sfpu_name(op)], slots) if sfpu_op(op) else []
      if dst:  # x into the op's Dst half, y 128 rows on, then the SFPU: all on the math thread
        fills = fill(x, "h") + (fill(y, "h + 128") if op["y"] != "zero" else []) + FILLED
        math = tt(math_begin(False) + fills + prog + sfpu(op['posts'], slots))
        c[M].append(f"{cond}{{ {drain}{math} {math_end(j)} h ^= {DST_HALF};{posted} }}")
        continue
      if op["kind"] in BINARY:  # x and y each into Dst for the SFPU: x in the op's half, y 128 rows on
        unpacks, fpu = unpack(x, ZERO_SLOT) + unpack(y, ZERO_SLOT), elementwise("add", slots) + elementwise("add", slots, dst=128)
      else: unpacks, fpu = unpack(x, y), elementwise(op["kind"], slots)
      c[U].append(f"{cond}{{ {drain}{tt(unpacks)}{posted} }}")
      math = tt(math_begin(op['kind'] == 'mul') + fpu + prog + sfpu(op['posts'], slots))
      c[M].append(f"{cond}{{ {math} {math_end(j)} h ^= {DST_HALF}; }}")
    for acc, j in enumerate(reductions):
      assert acc < len(ACCS), "two reductions a pass"
      op, me_j, x, kind = ops[j], ("op", j), ops[j]["x"], ops[j]["kind"]
      drain = "drain(); " if isinstance(x, tuple) else ""
      edge = size(x)[axis] % 32 if axis and size(x)[axis] > 1 else 0  # the operand's rows or columns in its last tile, if cut short
      end = f"k == {L - 1}" if swept(x) else "k == 0"  # the line's last tile (a narrow operand is one tile a line)
      src = slot(x) if dst else None
      def step(first):  # the last tile's padding must not count
        plain, cut = tt(reduce_step(kind, slots, acc, first, 0, src)), tt(reduce_step(kind, slots, acc, first, edge, src))
        return f"if ({end}) {{ {cut} }} else {{ {plain} }}" if edge else plain
      finish, posted = f'{tt(reduce_finish(kind, slots, acc, op["posts"]))} {math_end(j)}', "" if j == root else " posted++;"
      pk = tt(pack(ACCS[acc][0], pslot(me_j), SEM_PACKED if j == root else SEM_L1, ACCS[acc][1]))
      pk = f"{around[0]} {pk} {around[1]}" if j == root else pk
      unpacks = "" if dst else tt(unpack(slot(x), ZERO_SLOT))  # an fp32 step fills Dst itself
      mine = M if dst else U  # who drains and counts
      if swept(x):  # swept along the line
        c[mine].append(f"{drain}{unpacks} if ({end}) {{{posted} }}")
        c[M].append(f"if (k == 0) {{ {step(True)} }} else {{ {step(False)} }} if ({end}) {{ {finish} }}")
      else:  # reduced in the pass's first tile
        c[mine].append(f"if ({end}) {{ {drain}{unpacks}{posted} }}")
        c[M].append(f"if ({end}) {{ {step(True)} {finish} }}")
      c[P].append(f"if ({end}) {{ {output(j)}{pk} }}")
    c[M if dst else U].append(tt(REPORT))
    c[B].append("frame_done[fr] = ++events;" + " narrow_done = events;" * (p == last_read(True)) +
                " kept_done[k] = events;" * (p == last_read(False)) + " line_done[k] = events;" * (p == computed[-1]))
    passes.append((bool(swept_ops or reductions), c))
  def program(c):  # the loops all four programs share, around each one's part of a tile
    tile_c = lambda b: f"{{ unsigned fr = tiles++ & 1; {' '.join(b[c])} }}"  # the tile's frame, by turns
    if not reducing:  # a unit is a block: its tiles column by column, those past the matrix's edge left out
      return f"""  unsigned unit = a[0], b = unit / {br * bc}, i0 = unit % {br * bc} / {bc} * {bp}, j0 = unit % {bc} * {bq};
  for (unsigned u = 0; u < a[1]; u++, unit++) {{
    for (unsigned jj = 0; jj < {bq}; jj++) for (unsigned ii = 0; ii < {bp}; ii++) {{
      unsigned i_ = i0 + ii, j = j0 + jj, k = 0;
      if (i_ >= {rt} || j >= {w}) continue;
      {tile_c(passes[0][1])}
    }}
    if ((j0 += {bq}) == {bc * bq}) {{ j0 = 0; if ((i0 += {bp}) == {br * bp}) {{ i0 = 0; b++; }} }}  // the next unit, without dividing
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
  first = ""
  if bcast:  # before its first tile, each worker waits for A's row, which the first grid row of its matrix's reads and multicasts
    (gx_n, gy_n), (xs, ys), kt, x = bcast, grid, ops[matmuls[0]]["kt"], arg(ops[matmuls[0]]["x"])
    rects = [bh.noc_rect((xs[0], ys[g]), (xs[gx_n - 1], ys[g + gy_n - 1])) for g in range(0, nb * gy_n, gy_n)]  # each matrix's rows
    decls, tail = _array("rect", rects) + "\n", f"unflag({gx_n}, 0);"
    in_batch = matrix(ops[matmuls[0]]["x"], x)
    first = (f"  {{ unsigned b = a[0] / {bc}, gx = a[0] % {gx_n}, e = gx * {kt} / {gx_n}, end = (gx + 1) * {kt} / {gx_n};\n"
             f"    if (a[0] % {bc} < {gx_n}) {{ for (unsigned m = e; m < end; m++) get(a[{x}], {in_batch}m + a[{x + 3}], {rows_at[0]} + m);\n"
             f"      share(SLOT({rows_at[0]} + e), (end - e) * {tile_bytes(dtype)}, rect[b], {bc - 1}, gx, 1); }}\n    await(0, {gx_n}, 1);\n  }}\n")
  if shared:  # before its first tile, each grid worker waits for A's row, which the grid's first row reads and multicasts
    (xs, ys), kt, x = grid, ops[matmuls[0]]["kt"], arg(ops[matmuls[0]]["x"])
    decls, tail = _array("grid_x", xs) + _array("grid_y", ys) + "\n", f"unflag({len(xs)}, 0);"
    get_a = lambda e, end: f"for (unsigned m = {e}; m < {end}; m++) get(a[{x}], m + a[{x + 3}], {rows_at[0]} + m);"
    first = (f"  {{ unsigned me = noc_me(), gx = {len(xs)}, gy = {len(ys)};\n"
             f"    for (unsigned i = 0; i < {len(xs)}; i++) if (grid_x[i] == (me & 63)) gx = i;\n"
             f"    for (unsigned i = 0; i < {len(ys)}; i++) if (grid_y[i] == me >> 6) gy = i;\n"
             f"    if (gy == {len(ys)}) {{ {get_a(0, kt)} noc_wait(); }}\n"
             f"    else {{ unsigned e = gx * {kt} / {len(xs)}, end = (gx + 1) * {kt} / {len(xs)};\n"
             f"      if (gy == 0) {{ {get_a('e', 'end')} share(SLOT({rows_at[0]} + e), (end - e) * {tile_bytes(dtype)}, "
             f"{bh.noc_rect((xs[0], ys[0]), (xs[-1], ys[-1]))}, {len(xs) * len(ys) - 1}, gx, 1); }}\n"
             f"      await(0, {len(xs)}, 1); }}\n  }}\n")
  if picked:  # the indices' tiles, 32 tokens each, come in first, where nothing else goes (past the slots and a block float's buffers)
    first = (f"  for (unsigned i = 0; i < {it}; i++) {{ unsigned t = a[{arg(r) + 3}] + i * a[{arg(r) + 2}];  // the indices' tile row i\n"
             f"    noc_get(BANK(t), AT(a[{arg(r)}], t, {ib}), IDS + i * {ib}, {ib}); }}\n  noc_wait();\n" + first)
  brisc = _brisc(total, first + program(B), dtype, out, decls + indices, tail, b8, wt)
  image = compile_kernel(brisc, _unpacker(program(U), config, indices), _math(slots, program(M), dst, indices), _packer(program(P)))
  return image, (nb * rt * w // L if reducing else nb * br * bc)  # and how many units the result is

def attention_store(ht, rows, wt, parts):
  """Where a unit's keys can stay in L1 between attention_kernel's two sweeps: for a block float, the bases of regions
  of whole key rows (at most 64 slots each, the unpacker's), below B8 past the slots in use and past the frames' regions;
  for bf16, the first free slot. And how many key rows fit."""
  R = min(8, -(-rows // 32))
  used = 3 + 2 * (R + (0 if wt else 2 * ht)) + 2 * R + R * ht  # (ones, none, the frames, p, the largest, the queries)
  top = bh.L1_SIZE - R * (ht + 1 + (parts > 1)) * tile_bytes("fp32") - R * tile_bytes("int32") - 64  # (attention_kernel's ranges)
  if not wt: return used, (UNPACKABLE - used) // ht
  per = 64 // ht  # key rows a region
  size = per * ht * tile_bytes(wt)
  windows = ((slot_base("bf16") + used * tile_bytes("bf16") + 63) // 64 * 64, B8), (B8 + 2 * 2 * ht * tile_bytes(wt), top)
  bases = [b for lo, hi in windows for b in range(lo, hi - size + 1, size)]
  return bases, len(bases) * per

def attention_parts(ht, rows, wt, kt, units, workers):
  """The parts Tensor.attend shares each unit's kt tiles of keys among unless told: for units of one or two tile rows
  of queries (a decode step's), as many as the workers allow, at most 8 and each of 8 tiles at least (combining them
  costs a few launches), or as many more as keep each part's keys in L1 (attention_store); else one."""
  if -(-rows // 32) > 2: return 1
  room = lambda P: attention_store(ht, rows, wt, P)[1]
  P = max(1, min(8, kt // 8, workers // units))
  return max(P, min(8, -(-kt // room(2)))) if -(-kt // P) > room(P) else P

def attention_kernel(ht, rows, rep, wt=None, phases=4, window=0, parts=1, keep=False):
  """Attention in one kernel, its scores kept on the chip: softmax(q k^T + mask) v for a batch of heads, args [first
  unit, units, then the result's, q's, k's, v's and the places' DRAM address, tile strides for batch and row, and tile
  offset, then the keys' tiles kt and q's rows]. q is (E, rows, 32 ht), k and v (E / rep, 32 kt, 32 ht), bf16 or the block float wt
  (a KV cache, each of its matrices read for rep of q's), and the places (E or 1, rows, 1, int32): each row's last
  key, and with a window the first is window - 1 before it, the mask hiding (-1e9) the keys outside them. A unit is up
  to eight tile rows of one head's queries, which share each tile of keys and values. BRISC reads its rows' places
  first and works out the tiles of keys they reach, and which of those some row sees only in part, for which it builds
  the mask's tiles from the places: two sweeps over those tiles, a frame of them at a time (a key tile row's ht tiles,
  the second sweep's values too, and the mask tiles it needs). Each sweep makes each tile of scores in Dst, the mask
  added in first (or 0) and the queries' row by the key tiles' (read transposed) multiplied onto it. The first folds it
  into each row's largest so far, as a reduction folds its tiles (in Dst for a unit of one or two tile rows; for more,
  a tile in L1 brought into Dst beside it and packed back, bf16), then across the tile; the second adds the mask less
  that largest instead, so exp makes p, packs it into L1, and multiplies it by the key tile's row of values and by a
  tile of ones (each row's sum of p, in all its columns), the packer adding each product onto its tile of the output in
  L1, in fp32. The result, (E, rows, 32 ht + 1) in fp32, is each head's p v, then each row's sum of p, so
  softmax(...) v is the one over the other. Each value tile comes from DRAM once a unit, each key tile once a sweep,
  and no score leaves L1. parts: each unit's tiles of keys shared out among so many workers (a decode step's few units
  over a long cache), the result (E parts, rows, 32 ht + 33): each part's p v and sums, and its rows' largest scores in
  a tile after them, for the caller to combine. A part with no tile of keys takes the unit's first, every key hidden by
  a tile of -1e9, and a row's largest is held at -1e4 at least, so a part in which a row sees no key gives it p of 0
  and a weight of 0; and it is rounded to bf16 in Dst, so the one its p is made against is the one the caller weighs
  it by. keep: a unit's first key rows, as many as attention_store holds, read in the first sweep, stay in L1 for the
  second, so each comes from DRAM once a unit."""
  assert ht >= 2, "a head of 64 columns at least"
  ra = -(-rows // 32)
  R = min(8, ra)  # tile rows a unit
  per, fs = -(-ra // R), R + (0 if wt else 2 * ht)  # units a head; slots a frame (the masks, then bf16 keys and values)
  ones, none, fr0 = 1, 2, 3  # a tile of ones and one of -1e9 (the SFPU's, packed once a launch), then the frames
  assert parts == 1 or R <= 2, "parts of units of one or two tile rows (decode's)"
  T = ht + 1 + (parts > 1)  # a row's output tiles: p v, the sums, and with parts its largest scores
  mask = lambda r: f"{fr0} + fr * {fs} + {r}"  # frame fr's mask tile for row r, and where the unpackers find its key tile m
  key = lambda m: f"{m}" if wt else f"{fr0} + fr * {fs} + {R} + {m}"  # and value tile j (a block float's: in the frame's
  val = lambda j: f"{ht} + {j}" if wt else f"{fr0} + fr * {fs} + {R + ht} + {j}"  # region at B8)
  into = lambda z: f"fr * {2 * ht} + {z}" if wt else z  # where BRISC puts one (getb's region and slot)
  p_slot, m_slot, q_row = fr0 + 2 * fs, fr0 + 2 * fs + R, fr0 + 2 * fs + 2 * R  # each row's p, its largest, its queries
  oacc = bh.L1_SIZE - R * T * tile_bytes("fp32")  # the output's tiles, each row's p v's then its sums, in fp32
  places = oacc - R * tile_bytes("int32")  # the unit's places, and (two units') key tiles: first, last, the full ones'
  ranges = places - 64                     # first and end, and whether its part has none
  assert ranges >= B8 + 2 * 2 * ht * tile_bytes(wt or "bf16") and q_row + R * ht <= UNPACKABLE, "a unit's tiles in L1"
  if keep:  # a unit's first key rows (n = kt - k0 below room): a block float's row n in region n / rows_a, slot n % rows_a * ht
    store, room = attention_store(ht, rows, wt, parts)  # on; a bf16's at slot store + n * ht on
    rows_a = 64 // ht
  kept = f"(kt - k0 < {room})" if keep else "0"  # (a key row in the store)
  cfg = kernel_config() | config(Pack_L1_Acc=0)
  first = kernel_config("bf16", "fp32") | config(L1_Dest_addr=oacc // 16 - 1, Pack_L1_Acc=0)  # the output's tiles, in fp32
  onto = first | config(Pack_L1_Acc=1)  # and added onto
  slots = allocate({"mmT", "mm", "max2", "exp", "one"} | ({"none", "floor", "ftob"} if parts > 1 else set()))
  held = R <= 2  # each row's largest kept in Dst (ACCS's two accumulators), else in L1 between key tiles
  acc, group = ACCS[0][0], ACCS[0][1]  # (row r's at acc + 64 r, a later tile's 128 rows on; taken for a sweep)
  tile = lambda i, b, r, c: f"({b}) * a[{i + 1}] + ({r}) * a[{i + 2}] + {c} + a[{i + 3}]"  # tile (b, r, c) of the tensor at a[i]
  each = lambda body: f"for (unsigned r = 0; r < nr; r++) {{ {body} }}"
  rng = f"(volatile unsigned *){ranges:#x} + (u & 1) * 8"
  read = f"asm volatile(\"fence\" ::: \"memory\"); volatile unsigned *g = {rng}; unsigned k0 = g[0], k1 = g[1], f0 = g[2], f1 = g[3], empty = g[4];"
  units = lambda start, pass0, between, pass1, end: f"""  for (unsigned u = 0; u < a[1]; u++) {{
    unsigned w_ = (a[0] + u) / {parts}, part = (a[0] + u) % {parts}, e = w_ / {per}, i0 = w_ % {per} * {R}, nr = {ra} - i0 < {R} ? {ra} - i0 : {R};
    {start}
    for (unsigned kt = k0; kt <= k1; kt++) {{ unsigned fr = tiles++ & 1; {pass0} }}
    {between}
    for (unsigned kt = k0; kt <= k1; kt++) {{ unsigned fr = tiles++ & 1; {pass1} }}
    {end}
  }}"""
  part = "(!empty && (kt < f0 || kt >= f1))"  # a tile of keys some row sees only in part: its mask tiles are read
  ob = f"e * {parts} + part" if parts > 1 else "e"  # the result's batch entry
  # BRISC: a tile of ones, once; then for each unit its places, from them its tiles of keys (k0 to k1) and the whole
  # ones (f0 to f1, not included), told to the TRISCs as a frame fetched
  start = (f"unsigned lo = ~0u, hi = 0; " +
           each(f"noc_get(BANK({tile(18, 'e', 'i0 + r', 0)}), AT(a[18], {tile(18, 'e', 'i0 + r', 0)}, 4096), {places:#x} + r * 4096, 4096);") +
           " noc_wait(); " +
           each(f"volatile unsigned *t = (volatile unsigned *)({places:#x} + r * 4096); "
                f"for (unsigned q = 0; q < 32 && (i0 + r) * 32 + q < a[23]; q++) {{ unsigned p = t[q / 16 * 512 + q % 16 * 16]; lo = p < lo ? p : lo; hi = p > hi ? p : hi; }}") +
           f" volatile unsigned *g = {rng}; unsigned k0 = {f'lo + 1 > {window} ? (lo + 1 - {window}) / 32 : 0' if window else '0'}, "
           f"k1 = hi / 32 < a[22] - 1 ? hi / 32 : a[22] - 1, f0 = {f'hi + 1 > {window} ? (hi + 1 - {window} + 31) / 32 : 0' if window else '0'}, f1 = (lo + 1) / 32; " +
           (f"unsigned n_ = k1 - k0 + 1, a0 = k0 + n_ * part / {parts}, a1 = k0 + n_ * (part + 1) / {parts}, empty = a0 == a1; "  # this part's
            "if (empty) k1 = k0; else k0 = a0, k1 = a1 - 1; " if parts > 1 else "unsigned empty = 0; ") +                   # share of them
           "g[0] = k0, g[1] = k1, g[2] = f0, g[3] = f1, g[4] = empty; asm volatile(\"fence\" ::: \"memory\"); ready(); ++events;")
  getk = "getb" if wt else "get"
  framed = f"for (unsigned m = 0; m < {ht}; m++) {getk}(a[10], {tile(10, f'e / {rep}', 'kt', 'm')}, {into(key('m'))});"  # (into the frame)
  if not keep: stored = ""
  elif wt: stored = (f"for (unsigned m = 0, n = kt - k0, z = n % {rows_a} * {ht}; m < {ht}; m++, z++) {{ unsigned t = {tile(10, f'e / {rep}', 'kt', 'm')}, "
                     f"at = AT(a[10], t, {tile_bytes(wt)}); noc_get(BANK(t), at, store[n / {rows_a}] + 64 * z, 64); "
                     f"noc_get(BANK(t), at + 64, store[n / {rows_a}] + {64 * rows_a * ht} + {BFP[wt]} * z, {BFP[wt]}); }}")
  else: stored = f"for (unsigned m = 0; m < {ht}; m++) get(a[10], {tile(10, f'e / {rep}', 'kt', 'm')}, {store} + (kt - k0) * {ht} + m);"
  keys = lambda values: f"if (!{kept}) {{ {framed} }}" + (f" else {{ {stored} }}" if not values else "")  # (a row in the store: the first sweep's)
  fetch = lambda values: (f"reached(frame_done[fr]); if {part} {{ " + each(f"masked({mask('r')}, r, kt);") + " } " + keys(values) +
                          (f" for (unsigned j = 0; j < {ht}; j++) {getk}(a[14], {tile(14, f'e / {rep}', 'kt', 'j')}, {into(val('j'))});" if values else "") +
                          f" l1_settle(SLOT({mask(0)})); ready(); frame_done[fr] = ++events;")
  queries = "if (kt == k0) { reached(events); " + each(f"for (unsigned m = 0; m < {ht}; m++) get(a[6], {tile(6, 'e', 'i0 + r', 'm')}, {q_row} + r * {ht} + m);") + " }"
  brisc = units(start, f"{queries} {fetch(False)}", "", fetch(True), f"expect += nr * {T};")
  region = lambda words: f"if (fr) {{ {tt(words(1))} }} else {{ {tt(words(0))} }}"  # unpacker 0 at frame fr's block floats
  on_k, off_k = region(lambda f: haloize(1) + (bfp_b(wt, 2 * ht, f) if wt else [])), tt(haloize(0) + (bfp_b(wt, 0) if wt else []))
  first_key = key(0)
  if keep and wt:  # (a stored row's region, by a switch: its base is in the instructions)
    on_k = (f"if {kept} {{ {tt(haloize(1))} switch ((kt - k0) / {rows_a}) {{ " +
            " ".join(f"case {g}: {{ {tt(bfp_b(wt, rows_a * ht, base=b))} }} break;" for g, b in enumerate(store)) + f" }} }} else {{ {on_k} }}")
    first_key = f"{kept} ? (kt - k0) % {rows_a} * {ht} : {key(0)}"
  elif keep: first_key = f"{kept} ? {store} + (kt - k0) * {ht} : {key(0)}"
  on_v, off_v = (region(lambda f: bfp_b(wt, 2 * ht, f)), tt(bfp_b(wt, 0))) if wt else ("", "")
  pair = [ins("UNPACR", last=1, set_dat_valid=1, ovrd_thread_id=1, addr_mode=1),  # a key tile into SrcA, a query tile into SrcB,
          ins("UNPACR", last=1, set_dat_valid=1, ovrd_thread_id=1, unpack_block_selection=1, addr_mode=1)]  # each Z then on by 1
  walk_k = f"{on_k} {tt([setadc(bh.UNPACK1, f'{q_row} + r * {ht}'), setadc(bh.UNPACK0, first_key)])} {tt([Loop(ht, pair)])} {off_k}"
  the_mask = f"empty ? {none} : {part} ? {mask('r')} : {ZERO_SLOT}"
  largest = f"{m_slot} + r"
  take = f"for (; drained < upto; drained++) {{ {tt(TAKE_L1)} }}"  # (the last tile's largests, packed)
  unpacker = units(f"fetched(); {read} {tt(REPORT)}",
                   f"fetched(); " + each(f"{tt(unpack(the_mask, ZERO_SLOT))} {walk_k}") + f" {tt(REPORT)}" if held else
                   f"fetched(); unsigned upto = posted; " +
                   each(f"if (kt > k0) {{ {take} {tt(unpack(largest, ZERO_SLOT))} }} {tt(unpack(the_mask, ZERO_SLOT))} {walk_k} posted++;") + f" {tt(REPORT)}",
                   "posted += nr;" if held else f"drain(); " + each(f"{tt(unpack(largest, ZERO_SLOT))} posted++;"),  # (each row's largest, across its tile)
                   f"fetched(); drain(); " + each(f"{tt(unpack(the_mask, largest))} {walk_k} posted++;") + " drain(); "  # each row's p, packed
                   f"{on_v} " + each(f"for (unsigned j = 0; j < {ht}; j++) {{ {tt(unpack(val('j'), f'{p_slot} + r'))} }}") + f" {off_v} " +
                   each(tt(unpack(ones, f"{p_slot} + r"))) + f" {tt(REPORT)}", "")  # (the sums' products after, the ones bf16)
  told = "fetched(); " + read + " needed += 2 * (k1 - k0 + 1);"  # (the unit's frames, which the math and pack threads need not wait for)
  scores = lambda op, row: (f"{tt([aim(row)] + elementwise(op, slots) + walk(True))} "
                            f"for (unsigned m = 0; m < {ht}; m++) {{ {tt(matmul(slots, phases, True))} }} {tt(walk(False))}")
  fold = tt([ins("STALLWAIT", stall_res=bh.STALL_SFPU, wait_res=bh.WAIT_FPU), ins("SETRWC", bit_mask=0b0100), aim("h")] + runs(slots, "max2", 32))
  begin = tt(math_begin(False))
  product = f"{tt(math_begin(True))} {tt(matmul(slots, phases))} {tt(MATH_END)} h ^= {DST_HALF};"  # p by a value tile, or by the ones
  room = tt([ins("SEMWAIT", stall_res=bh.STALL_ALL, sem_sel=SEM_DST, wait_sem_cond=2)])  # a result's place to pack from
  take_acc = tt([ins("SEMWAIT", stall_res=bh.STALL_ALL, sem_sel=group, wait_sem_cond=1), ins("SEMGET", sem_sel=group)])
  fold_acc = tt([ins("STALLWAIT", stall_res=bh.STALL_SFPU, wait_res=bh.WAIT_FPU), ins("SETRWC", bit_mask=0b0100), aim(f"{acc} + 64 * r")] + runs(slots, "max2", 32))
  math = units(told,
               f"if (kt == k0) {{ {take_acc} }} " + each(f"if (kt == k0) {{ {scores('add', f'{acc} + 64 * r')} }} "
                                                        f"else {{ {scores('add', f'{acc + 128} + 64 * r')} {fold_acc} }}") if held else
               each(f"{begin} if (kt > k0) {{ {tt(elementwise('add', slots))} {scores('add', 'h + 128')} {fold} }} "
                    f"else {{ {scores('add', 'h')} }} {tt(MATH_END)} h ^= {DST_HALF};"),
               each(f"{room} {tt([aim(f'{acc} + 64 * r')] + across_rows('max2') + (sfpu(['floor', 'ftob'], slots) if parts > 1 else []) + MATH_END)}" +
                    (f" {room} {tt([ins('SEMPOST', sem_sel=SEM_DST)])}" if parts > 1 else "")) if held else  # (the largest packed twice)
               each(f"{begin} {tt(elementwise('add', slots) + across_rows('max2') + MATH_END)} h ^= {DST_HALF};"),
               each(f"{begin} {scores('sub', 'h')} {tt(sfpu(['exp'], slots))} {tt(MATH_END)} h ^= {DST_HALF};") + " " +
               each(f"for (unsigned j = 0; j < {ht}; j++) {{ {product} }}") + " " + each(product),
               "")
  out = lambda j: f"r * {T} + {j}"  # row r's output tile j (its sums at j = ht, its largest at ht + 1)
  largest_out = (f" while (peek(SENT) < outputs) relay(); {tt(repack(cfg, first) + pack(f'{acc} + 64 * r', out(ht + 1)) + repack(first, cfg))}"
                 if parts > 1 else "")  # (each part's largest, home with its p v)
  packed = lambda slot: tt(pack("h", slot, SEM_L1)) + f" h ^= {DST_HALF};"
  packer = units(f"needed++; while (peek(FETCHED) < needed) relay(); {read} needed += 2 * (k1 - k0 + 1);",
                 "" if held else each(packed(largest)),
                 each(tt(pack(f"{acc} + 64 * r", largest, SEM_L1)) + largest_out) + f" {tt([ins('SEMPOST', sem_sel=group)])}" if held else each(packed(largest)),
                 each(packed(f"{p_slot} + r")) +
                 f" if (kt == k0) {{ while (peek(SENT) < outputs) relay(); {tt(repack(cfg, first))} }} else {{ {tt(repack(cfg, onto))} }} " +
                 each(f"for (unsigned j = 0; j < {ht}; j++) {{ {tt(pack('h', out('j')))} h ^= {DST_HALF}; }}") + " " +
                 each(f"{tt(pack('h', out(ht)))} h ^= {DST_HALF};") +
                 f" if (kt == k0) {{ {tt(repack(first, cfg))} }} else {{ {tt(repack(onto, cfg))} }}",
                 each(f"for (unsigned j = 0; j < {T}; j++) {{ while (peek(SENT) + 4 <= outputs) relay(); "  # all of it goes home
                      f"SYNC[RING + outputs % 4 * 2] = {out('j')}; SYNC[RING + outputs % 4 * 2 + 1] = {tile(2, ob, 'i0 + r', 'j')}; outputs++; "
                      f"{tt([ins('SEMPOST', sem_sel=SEM_PACKED)])} }}"))
  made = [("one", ones)] + [("none", none)] * (parts > 1)  # the tile of ones (for the sums) and of -1e9 (an empty part's mask),
  math = "".join(f"{tt(math_begin(False) + sfpu([p], slots) + MATH_END)} h ^= {DST_HALF};" for p, _ in made) + math  # once a launch:
  packer = "".join(f"{tt(pack('h', z, SEM_L1))} h ^= {DST_HALF};" for _, z in made) + packer                     # made by the SFPU
  hide = to_bf16(-1e9) * 0x10001  # (two of -1e9, bf16, a word)
  decls = "static unsigned expect;" + (_array("store", store) if keep and wt else "") + f"""
static void masked(unsigned slot, unsigned r, unsigned kt) {{  // tile row r's mask for key tile kt from its places: -1e9 where
  volatile unsigned *places = (volatile unsigned *)({places:#x} + r * 4096), *m = (volatile unsigned *)SLOT(slot);  // a row sees
  for (unsigned q = 0; q < 32; q++) {{  // not, two keys a word                                                          // no key
    int p = places[q / 16 * 512 + q % 16 * 16], first = {f"p - {window - 1}" if window else "0"};
    for (unsigned c = 0; c < 32; c += 2) {{
      int j = kt * 32 + c;
      m[(q / 16 * 2 + c / 16) * 128 + q % 16 * 8 + c % 16 / 2] = (j > p || j < first ? {hide & 0xFFFF:#x} : 0) | (j + 1 > p || j + 1 < first ? {hide & 0xFFFF0000:#x}u : 0);
    }}
  }}
}}"""
  return compile_kernel(_brisc("expect", brisc, "bf16", "fp32", decls, b8=2 * ht if wt else 0, wt=wt or "bfp8", outs=oacc),
                        _unpacker(f"posted += {len(made)};" + unpacker, cfg), _math(slots, math), _packer("static unsigned needed;" + packer))

def delta_kernel(kt, vt, vh, rep, m=1):
  """Gated DeltaNet's decode step for each sequence (Tensor.delta), from its token's projections to its outputs, or a
  verify step's token and its draft, in fp32, each state read once: args [first unit, units, then the states', the
  output's (a head a matrix, a sequence's m tokens rows in turn) and live's (a sequence a column) DRAM address, tile
  strides for batch and row, and tile offset; with m of 2 then the states' after a first token and accept's (a sequence
  a column); then x's (a token a row: its q's, k's, v's, z's, b's and a's side by side, a sequence's m in turn), the
  conv's last inputs' (3 matrices, oldest first, a sequence a row; with m of 2 then those after a first token), the
  taps' (4 matrices of a row), dt's and decay's (a value head a column)]. A unit is a key head of a sequence and its rep
  value heads, so it alone reads and writes its channels of the conv's inputs. BRISC reads its states, and the row
  runs of its channels from the conv's inputs, the tokens' own rows of x and the taps: q's and k's into columns 0-3 of
  a tile (a channel a row), v's into rows 0-3 (a channel a column), b, a, dt and decay into place 0. It writes the
  inputs a token leaves (each one up a place, its own last; with m of 2 the first's into those after a first token)
  once it has read them, where the sequence is live. The math thread masks what lies outside those places to 0 in Dst
  (no tile need be zeroed in L1) and folds: a channel's 4 inputs by its 4 taps summed across the row (or down the
  column), through SiLU, as a broadcast tile; q and k each to length 1 (q times kd^-1/2) by the sum of their squares
  folded down; beta = sigmoid(b), e^g = exp(decay softplus(a + dt)). The rule is then separable by column: for each
  value head and column tile j of its S, and each token in turn, one SFPU pass sums u = S^T k and w = S^T q down the
  column's tiles, then d = beta (v - e^g u), the token's output e^g w + d (k . q), and the column's new tiles e^g S +
  k d, the next token's S. Each token's states go home (the last's to the states, a first of two's to the others), a
  dead sequence's not."""
  kh, kd, vd = vh // rep, 32 * kt, 32 * vt
  qk, vw = kh * kd, vh * vd
  S_ = lambda e, i, j: f"(F0 + ({e}) * {kt * vt} + ({i}) * {vt} + ({j}))"  # value head e's S, tile (i, j)
  T0 = rep * kt * vt  # then the taps of the key head's q's and k's channels (columns), of each value head's v's (rows),
  TQ, TK = (lambda i, o=o: f"(F0 + {T0 + o} + ({i}))" for o in (0, kt))  # and each value head's dt and decay
  TV = lambda e, j: f"(F0 + {T0 + 2 * kt} + ({e}) * {vt} + ({j}))"
  DT, DC = (lambda e, o=o: f"(F0 + {T0 + 2 * kt + rep * vt + o} + ({e}))" for o in (0, rep))
  I0, per = T0 + 2 * kt + rep * vt + 2 * rep, 2 * kt + rep * vt + 2 * rep  # then each token's: its inputs (q's, k's, v's), b and a
  IQ, IK = (lambda t, i, o=o: f"(F0 + {I0 + o} + ({t}) * {per} + ({i}))" for o in (0, kt))
  IV = lambda t, e, j: f"(F0 + {I0 + 2 * kt} + ({t}) * {per} + ({e}) * {vt} + ({j}))"
  BB, AA = (lambda t, e, o=o: f"(F0 + {I0 + 2 * kt + rep * vt + o} + ({t}) * {per} + ({e}))" for o in (0, rep))
  F = I0 + m * per  # a frame's slots
  mine = 2 * kt + rep * vt + 2 * rep + 1  # a token's broadcast tiles, packed: keys', queries', values', then e^g, beta, k . q
  KC, QC, VV, E, BT, KQ = (lambda t, o=o: 1 + 2 * F + t * mine + o for o in (0, kt, 2 * kt, 2 * kt + rep * vt, 2 * kt + rep * vt + rep, mine - 1))
  COL = 1 + 2 * F + m * mine  # a first of two's new tiles of the column, the second's S
  OUT = COL + kt * (m - 1)  # an output slot for each Dst half
  sources = 7 + m  # rows a unit reads of the conv: the 3 last inputs, the tokens' own, the 4 taps
  STAGE = slot_base("fp32") + (OUT + 2) * 4096  # BRISC's: each source's runs of q's and k's channels, then the scalars' 64-byte blocks
  scalars = 2 + 2 * rep + 2 * m * rep  # (live, accept; dt and decay; each token's b and a)
  assert STAGE + sources * 256 * kt + 64 * scalars <= OUTS and OUT + 2 <= UNPACKABLE, "a unit's tiles in L1"
  R2, R3, R4, R5, R6, R7 = 128, 192, 256, 320, 384, 448  # Dst rows (fp32: a tile's 64): fills, the two sums, e^g, d
  load = lambda r, at: ins("SFPLOAD", vd=r, mod0=3, addr_mode=2, imm10=at)
  store = lambda r, at, last=True: ins("SFPSTORE", vd=r, mod0=3, addr_mode=3 if last else 2, imm10=at)
  sfpu_body = lambda words: [ins("STALLWAIT", stall_res=bh.STALL_SFPU, wait_res=bh.WAIT_FPU), ins("SETRWC", bit_mask=0b0100), Loop(32, words)]
  programs = {  # each at Dst row 0 (aim(0)), its result into row r
    "kq first": lambda r: [load(0, R2), load(1, R3), mul(2, 0, 1), store(2, R4)],
    "kq": lambda r: [load(0, R2), load(1, R3), load(2, R4), mad(2, 0, 1, 2), store(2, R4)],
    "sums first": lambda r: [load(0, R2), load(1, R3), load(2, R6), mul(3, 1, 0), mul(4, 2, 0), store(3, R4, False), store(4, R5)],
    "sums": lambda r: [load(0, R2), load(1, R3), load(2, R6), load(3, R4), load(4, R5), mad(3, 1, 0, 3), mad(4, 2, 0, 4), store(3, R4, False), store(4, R5)],
    "d": lambda r: [load(0, R2), load(1, R3), load(2, R6), load(3, R4), mad(4, 2, 3, 1, neg=1), mul(5, 0, 4), store(5, R7)],  # beta (v - e u)
    "out": lambda r: [load(0, R6), load(1, R5), load(2, R7), load(3, R2), mul(4, 0, 1), mad(5, 2, 3, 4), store(5, r)],  # e w + d (k . q)
    "state": lambda r: [load(0, R2), load(1, R3), load(2, R6), load(3, R7), mul(4, 2, 0), mad(5, 1, 3, 4), store(5, r)],  # e S + k d
    "copy": lambda r: [load(0, R4), store(0, r)],
    "mul": lambda r: [load(0, R2), load(1, R3), mul(2, 0, 1), store(2, r)],  # inputs by their taps
    "add": lambda r: [load(0, R2), load(1, R3), mad(2, 0, ONE, 1), store(2, r)],  # a + dt
    "aside": lambda r: [load(0, r), store(0, R7)],  # x, kept for SiLU's x sigmoid(x) and softplus's max(x, 0)
    "neg": lambda r: [load(0, r), mul(1, 0, MINUS_ONE), store(1, r)],
    "-abs": lambda r: [load(0, r), ins("SFPABS", vd=1, vc=0, mod1=1), mul(2, 1, MINUS_ONE), store(2, r)],
    "1+": lambda r: [load(0, r), mad(1, 0, ONE, ONE), store(1, r)],
    "times x": lambda r: [load(0, r), load(1, R7), mul(2, 0, 1), store(2, r)],
    "+ max(x, 0)": lambda r: [load(0, R7), ins("SFPMOV", vd=1, vc=ZERO), ins("SFPSWAP", vd=1, vc=0, mod1=1), load(2, r), mad(3, 2, ONE, 0), store(3, r)],
    "times R4": lambda r: [load(0, r), load(1, R4), mul(2, 0, 1), store(2, r)],  # by decay, or a q's or k's tile by its scale
    "squares first": lambda r: [load(0, R2), mul(1, 0, 0), store(1, R4)],
    "squares": lambda r: [load(0, R2), load(1, R4), mad(1, 0, 0, 1), store(1, R4)],
    "+ eps": lambda r: [load(0, R4), *sfp_const(5, 1e-6), mad(1, 0, ONE, 5), store(1, R4)],
    "q's scale": lambda r: [load(0, R4), *sfp_const(5, kd ** -0.5), mul(1, 0, 5), store(1, R4)]}
  name = lambda n: "p_" + "".join(ch if ch.isalnum() else "_" for ch in n)  # (each program a function: its words once in the program,
  run = lambda n: f"{name(n)}(0);"                                             # and the loops that call them short)
  to_h = lambda n: f"{name(n)}(h);"
  free = tt([ins("STALLWAIT", stall_res=bh.STALL_UNPACK, wait_res=bh.WAIT_SFPU)])  # fills wait for the SFPU to be done with their rows
  fills = lambda *pairs: free + " " + tt([w for slot, row in pairs for w in fill(slot, row)] + FILLED)
  half = tt([ins("SEMWAIT", stall_res=bh.STALL_ALL, sem_sel=SEM_DST, wait_sem_cond=2)])  # a Dst half free for a result
  done = f"{tt(MATH_END)} h ^= {DST_HALF};"
  at = lambda row: tt([aim(row)])
  folds = lambda *kinds: " ".join(f"{k}();" for k in kinds)  # (each fold a function: its words once in the program)
  lib = lambda *names: f"{at('h')} {' '.join(f'{name(n)}();' for n in names)} {at(0)}"  # the SFPU's own programs on the tile at h
  silu = f"{to_h('aside')} {to_h('neg')} {lib('fexp')} {to_h('1+')} {lib('frecip')} {to_h('times x')}"
  masks = {"columns 4 on": mask_columns("sum", 4), "rows 4 on": mask_rows("sum", 4), "all but place 0": mask_rows("sum", 1) + mask_columns("sum", 1)}
  only = lambda mask: f"{at('h')} {name(mask)}();"  # the tile at h but the places filled, 0
  conv = lambda ins, taps, mask, kind: f"{at(0)} {fills((ins, R2), (taps, R3))} {to_h('mul')} {only(mask)} {folds(kind)} {at(0)} {silu}"
  def result(words, slot=None, home=None, keep=None):  # (math's, pack's) words for a result in the Dst half h: packed into
    math = f"{half} {words} {done}" + (" posted++;" if slot is not None else "")  # slot (an intermediate, its L1 kept until
    if home is None: return math, f"{tt(pack('h', slot, SEM_L1))} h ^= {DST_HALF};"  # it has gone home, if it goes: keep,
    to, keep = (slot, keep) if slot is not None else (f"{OUT} + h / {DST_HALF}", f"h / {DST_HALF}")  # its output slot's
    return math, f"output({keep}, {to}, {home}); {tt(pack('h', to, SEM_PACKED, *([SEM_L1] if slot is not None else [])))} h ^= {DST_HALF};"
  loop = lambda var, n, pair: tuple(f"for (unsigned {var} = 0; {var} < {n}; {var}++) {{ {w} }}" for w in pair)  # (math's, pack's), n times
  def unit_length(c, scale):  # the kt tiles of q's or k's at slot c to length 1 (and scaled): the sum of their squares folded down,
    return [(f"drain(); {at(0)} {fills((c, R2))} {run('squares first')} for (unsigned i = 1; i < {kt}; i++) {{ {fills((f'{c} + i', R2))} "
             f"{run('squares')} }} {at(R4)} {folds('sum1')} {at(0)} {run('+ eps')} {at(R4)} {name('sqrt')}(); {name('frecip')}(); {at(0)}"
             + (f" {run(scale)}" if scale else ""), ""),  # its 1 / sqrt in R4, then each tile by it
            loop("i", kt, result(f"{fills((f'{c} + i', 'h'))} {to_h('times R4')}", f"{c} + i"))]
  steps = []
  for t in range(m):  # each token's broadcast tiles: the conv's, the gates', and its k . q
    steps += [loop("i", kt, result(conv(IQ(t, "i"), TQ("i"), "columns 4 on", "sum2"), f"{QC(t)} + i")),
              loop("i", kt, result(conv(IK(t, "i"), TK("i"), "columns 4 on", "sum2"), f"{KC(t)} + i")),
              *unit_length(QC(t), "q's scale"), *unit_length(KC(t), None),
              loop("e", rep, loop("j", vt, result(conv(IV(t, "e", "j"), TV("e", "j"), "rows 4 on", "sum1"), f"{VV(t)} + e * {vt} + j"))),
              loop("e", rep, result(f"{at('h')} {fills((BB(t, 'e'), 'h'))} {only('all but place 0')} {folds('sum2', 'sum1')} {at(0)} "
                                    f"{to_h('neg')} {lib('fexp')} {to_h('1+')} {lib('frecip')}", f"{BT(t)} + e")),  # beta = sigmoid(b)
              loop("e", rep, result(f"{at(0)} {fills((DC('e'), R4))} {at(R4)} {name('all but place 0')}(); {folds('sum2', 'sum1')} "
                                    f"{at(0)} {fills((AA(t, 'e'), R2), (DT('e'), R3))} {to_h('add')} {only('all but place 0')} {folds('sum2', 'sum1')} "
                                    f"{at(0)} {to_h('aside')} {to_h('-abs')} {lib('fexp')} {to_h('1+')} {lib('log')} {to_h('+ max(x, 0)')} "
                                    f"{to_h('times R4')} {lib('fexp')}", f"{E(t)} + e")),  # e^g = exp(decay softplus(a + dt))
              (f"drain(); {at(0)} {fills((KC(t), R2), (QC(t), R3))} {run('kq first')} for (unsigned i = 1; i < {kt}; i++) "
               f"{{ {fills((f'{KC(t)} + i', R2), (f'{QC(t)} + i', R3))} {run('kq')} }} {at(R4)} {folds('sum1')} {at(0)}", ""),
              result(f"{free} {to_h('copy')}", KQ(t)), ("drain();", "")]
  column = []  # each value head's column tile j of S, through each token in turn
  for t in range(m):
    col = (lambda i: S_("e", i, "j")) if t == 0 else (lambda i: f"{COL} + ({i})")
    column += [("drain();" if t else "", ""),
               (f"{fills((col(0), R2), (KC(t), R3), (QC(t), R6))} {run('sums first')} "
                f"for (unsigned i = 1; i < {kt}; i++) {{ {fills((col('i'), R2), (f'{KC(t)} + i', R3), (f'{QC(t)} + i', R6))} {run('sums')} }} "
                f"{at(R4)} {folds('sum1')} {at(R5)} {folds('sum1')} {at(0)} "
                f"{fills((f'{BT(t)} + e', R2), (f'{VV(t)} + e * {vt} + j', R3), (f'{E(t)} + e', R6))} {run('d')}", ""),
               result(f"{fills((KQ(t), R2))} {to_h('out')}", home=f"{1 << 30:#x} | (unit * {rep} + e) << 8 | {t} << 6 | j"),
               loop("i", kt, result(f"{fills((col('i'), R2), (f'{KC(t)} + i', R3))} {to_h('state')}", f"{COL} + i" if t < m - 1 else None,
                                    f"{(m - 1 - t) << 31:#x} | (unit * {rep} + e) << 8 | (i * {vt} + j)", "2 + i"))]
  steps.append(loop("e", rep, loop("j", vt, (" ".join(c for c, _ in column), " ".join(c for _, c in column)))))
  math, packs = [w for w, _ in steps], [w for _, w in steps]
  units = lambda body: f"""  for (unsigned u = 0; u < a[1]; u++) {{
    unsigned unit = a[0] + u, F0 = 1 + (u & 1) * {F};
    {body}
  }}"""
  # BRISC: for each unit, once its frame is free, its states, its channels' runs and its values, put in place; then the
  # conv's inputs it leaves; meanwhile, results go home as they are packed
  tile = lambda i, b, r, c: f"({b}) * A[({i}) + 1] + ({r}) * A[({i}) + 2] + ({c}) + A[({i}) + 3]"  # tile (b, r, c) of the tensor at A[i]
  first = 14 if m > 1 else None  # (where in args: the states after a first token, then x, the conv's inputs, the taps, dt and decay)
  X = 22 if m > 1 else 14
  C, C2, TAPS = X + 4, X + 8, X + 4 + 4 * m
  DTA, DCA = TAPS + 4, TAPS + 8
  run_at = lambda r: f"((({r}) % 32) / 16 * 2 + f) * 1024 + (({r}) % 16) * 64"  # row r's run in face f of an fp32 tile, in bytes
  sc = lambda k: f"STAGE + {sources * 256 * kt} + 64 * ({k})"  # scalar k's 64-byte block
  scalar = lambda i, row, col, k: (f"{{ unsigned t_ = {tile(i, 0, f'({row}) / 32', f'({col}) / 32')}, o_ = (({row}) % 32 / 16 * 2 + ({col}) % 32 / 16) * 1024 + "
                                   f"(({row}) % 16 * 16 + ({col}) % 16) * 4; noc_get(BANK(t_), AT(A[{i}], t_, 4096) + o_ / 64 * 64, {sc(k)}, 64); word[{k}] = o_ % 64 / 4; }}")
  value = lambda k: f"((volatile unsigned *)({sc(k)}))[word[{k}]]"
  staged = lambda src, g: f"(STAGE + ({src}) * {256 * kt} + ({g}) * {128 * kt})"  # source src's runs of q's (g 0) or k's channels (1)
  brisc = SYNC_WORDS + ROWS + f"""
#define SLOT(i) ({slot_base("fp32"):#x} + (i) * 4096)
#define STAGE {STAGE:#x}
static const unsigned *A;
static unsigned fetched, events, frame_done[2], sent, alive[4], word[{scalars}];
static void ready(void) {{ noc_wait(); SYNC[FETCHED] = ++fetched; }}
static void send(void) {{  // the results packed so far go home: an output's row runs, a live sequence's state's tiles
  for (; sent < peek(PACKED); SYNC[SENT] = ++sent) {{
    unsigned slot = peek(RING + sent % 4 * 2), t = peek(RING + sent % 4 * 2 + 1), unit = t >> 8 & 0x3FFFFF, s = unit / {vh}, h = unit % {vh};
    if (t >> 30 & 1) for (unsigned f = 0; f < 2; f++) {{  // the output's tile of the head's token: its row's runs
      unsigned r = s * {m} + (t >> 6 & 3), o = {tile(6, 'h', 'r / 32', 't & 63')}, off = {run_at('r')};
      noc_put(SLOT(slot) + off, BANK(o), AT(A[6], o, 4096) + off, 64);
    }} else if (alive[unit / {rep} % 4]) {{  // the states', or the first token's
      unsigned at = t >> 31 ? {first or 2} : 2, o = unit * A[at + 1] + (t & 255) / {vt} * A[at + 2] + (t & 255) % {vt} + A[at + 3];
      noc_put(SLOT(slot), BANK(o), AT(A[at], o, 4096), 4096);
    }}
    noc_wait();
  }}
}}
static void reached(unsigned n) {{ while (peek(UNPACKED) < n) send(); }}
// source src's row (the conv's 3 last inputs, from the states at A[c]; the tokens' own rows of x; the 4 taps): its runs of
// the channels in n tiles from col on, both faces of each, into L1 at to (a tile's step bytes on, a face's half on)
static void get_runs(unsigned src, unsigned s, unsigned c, unsigned col, unsigned n, unsigned to, unsigned step, unsigned half) {{
  unsigned i = src < 3 ? c : src < {3 + m} ? {X} : {TAPS}, b = src < 3 ? src : src < {3 + m} ? 0 : src - {3 + m}, f = 0;
  unsigned row = src < 3 ? s : src < {3 + m} ? s * {m} + src - 3 : 0, t = {tile('i', 'b', 'row / 32', 'col / 32')}, off = {run_at('row')};
  for (unsigned j = 0; j < n; j++, t++, to += step) {{
    noc_get(BANK(t), AT(A[i], t, 4096) + off, to, 64);
    noc_get(BANK(t), AT(A[i], t, 4096) + off + 1024, to + half, 64);
  }}
}}
static void put_runs(unsigned at, unsigned d, unsigned s, unsigned col, unsigned n, unsigned from, unsigned step, unsigned half) {{  // into
  unsigned t = {tile('at', 'd', 's / 32', 'col / 32')}, f = 0, off = {run_at('s')};                                          // input d
  for (unsigned j = 0; j < n; j++, t++, from += step) {{                                                                     // of the
    noc_put(from, BANK(t), AT(A[at], t, 4096) + off, 64);                                                                    // states
    noc_put(from + half, BANK(t), AT(A[at], t, 4096) + off + 1024, 64);                                                      // at A[at]
  }}
}}
void kmain(volatile unsigned *args) {{
  A = (const unsigned *)args;
  for (unsigned u = 0; u < A[1]; u++) {{
    unsigned unit = A[0] + u, F0 = 1 + (u & 1) * {F}, s = unit / {kh}, hk = unit % {kh};
    unsigned cq = hk * {kd}, ck = {qk} + hk * {kd}, cv = {2 * qk} + hk * {rep * vd};  // its channels: q's, k's, its value heads' v's
    reached(frame_done[u & 1]);
    {scalar(10, 0, 's', 0)} {scalar(18, 0, 's', 1) if m > 1 else ""}
    noc_wait();
    alive[unit % 4] = {value(0)} != 0;
    unsigned taken = {value(1) if m > 1 else 1}, from = taken ? 2 : {first or 2}, c = taken ? {C} : {C2};  // (the states the last step's draft says)
    for (unsigned e = 0; e < {rep}; e++) for (unsigned i = 0; i < {kt}; i++) for (unsigned j = 0; j < {vt}; j++) {{
      unsigned t = (s * {vh} + hk * {rep} + e) * A[from + 1] + i * A[from + 2] + j + A[from + 3];
      noc_get(BANK(t), AT(A[from], t, 4096), SLOT({S_('e', 'i', 'j')}), 4096);
    }}
    for (unsigned src = 0; src < {sources}; src++) for (unsigned g = 0; g < 2; g++)  // q's and k's, staged
      get_runs(src, s, c, g ? ck : cq, {kt}, {staged('src', 'g')}, 128, 64);
    for (unsigned r = 0; r < 4; r++) {{  // v's into rows 0-3 of their tiles: each token's inputs, and the taps
      for (unsigned t = 0; t < {m}; t++) get_runs(t + r, s, c, cv, {rep * vt}, SLOT({IV('t', 0, 0)}) + r * 64, 4096, 1024);
      get_runs({3 + m} + r, s, c, cv, {rep * vt}, SLOT({TV(0, 0)}) + r * 64, 4096, 1024);
    }}
    for (unsigned e = 0; e < {rep}; e++) {{
      unsigned h = hk * {rep} + e;
      {scalar(DTA, 0, 'h', '2 + e')} {scalar(DCA, 0, 'h', f'{2 + rep} + e')}
      for (unsigned t = 0; t < {m}; t++) {{
        {scalar(X, f's * {m} + t', f'{2 * qk + 2 * vw} + h', f'{2 + 2 * rep} + t * {2 * rep} + e')}
        {scalar(X, f's * {m} + t', f'{2 * qk + 2 * vw + vh} + h', f'{2 + 3 * rep} + t * {2 * rep} + e')}
      }}
    }}
    noc_wait();
    for (unsigned g = 0; g < 2; g++) for (unsigned i = 0; i < {kt}; i++) {{  // q's and k's, a channel a row: input or tap j in column j
      unsigned *taps = (unsigned *)SLOT(g ? {TK('i')} : {TQ('i')}), *ins[{m}];
      const unsigned *from = (const unsigned *){staged(0, 'g')} + i * 32;  // (the runs of source 0, tile i; a source's {64 * kt} words on)
      for (unsigned t = 0; t < {m}; t++) ins[t] = (unsigned *)SLOT(g ? {IK('t', 'i')} : {IQ('t', 'i')});
      for (unsigned r = 0; r < 32; r++) {{
        unsigned at = (r / 16 * 2) * 256 + r % 16 * 16, o = r / 16 * 16 + r % 16;  // its place in the tile, and in tile i's runs
        for (unsigned j = 0; j < 4; j++) {{
          taps[at + j] = from[({3 + m} + j) * {64 * kt} + o];
          for (unsigned t = 0; t < {m}; t++) ins[t][at + j] = from[(t + j) * {64 * kt} + o];
        }}
      }}
    }}
    for (unsigned e = 0; e < {rep}; e++) {{
      ((volatile unsigned *)SLOT({DT('e')}))[0] = {value('2 + e')}, ((volatile unsigned *)SLOT({DC('e')}))[0] = {value(f'{2 + rep} + e')};
      for (unsigned t = 0; t < {m}; t++)
        ((volatile unsigned *)SLOT({BB('t', 'e')}))[0] = {value(f'{2 + 2 * rep} + t * {2 * rep} + e')}, ((volatile unsigned *)SLOT({AA('t', 'e')}))[0] = {value(f'{2 + 3 * rep} + t * {2 * rep} + e')};
    }}
    if (alive[unit % 4]) for (unsigned t = 0; t < {m}; t++) for (unsigned d = 0; d < 3; d++) {{  // the inputs token t leaves: sources
      unsigned at = t == {m - 1} ? {C} : {C2}, src = t + 1 + d, tv = src > 3 ? src - 3 : 0;      // t + 1 to t + 3
      for (unsigned g = 0; g < 2; g++) put_runs(at, d, s, g ? ck : cq, {kt}, {staged('src', 'g')}, 128, 64);
      put_runs(at, d, s, cv, {rep * vt}, SLOT({IV('tv', 0, 0)}) + (src - tv) * 64, 4096, 1024);  // (from token tv's tiles)
    }}
    l1_settle(SLOT(F0));
    ready(); frame_done[u & 1] = ++events;
  }}
  while (sent < A[1] * {rep * vt * m * (kt + 1)}) send();
}}"""
  cfg = kernel_config("fp32", "fp32")
  fold = "".join(f"__attribute__((noinline)) static void {k}(void) {{ {tt(ACROSS[FOLDS[k]](k))} }}\n" for k in ("sum1", "sum2"))  # down a tile's columns, along its rows
  fold += "".join(f"__attribute__((noinline)) static void {name(n)}(void) {{ {tt(sfpu([n], {}))} }}\n" for n in ("fexp", "frecip", "log", "sqrt"))
  fold += "".join(f"__attribute__((noinline)) static void {name(n)}(void) {{ {tt(words)} }}\n" for n, words in masks.items())
  fold += "".join(f"__attribute__((noinline)) static void {name(n)}(unsigned h) {{ if (h) {{ {tt(sfpu_body(p(DST_HALF)))} }} else {{ {tt(sfpu_body(p(0)))} }} }}\n"
                  for n, p in programs.items())  # (into the result's row: 0, or the other half's)
  return compile_kernel(brisc, _unpacker(units("fetched();"), cfg), _math({}, units("fetched(); " + " ".join(math) + " " + tt(REPORT)), True, fold),
                        _packer(units(" ".join(packs)), 2 + kt))

ROWS = BANKS + """
static void copy(unsigned from, unsigned to, unsigned n) {  // (this tile's coordinates read once a launch: an NIU register read)
  static unsigned me;
  if (!me) me = noc_me();
  noc_get(me & 63, me >> 6, from, to, n);
}
static void row_down(unsigned tile, unsigned item) {  // a tile of item-byte values: row 0 of faces 0 and 1 down their 16 rows,
  unsigned face = 256 * item;                         // then faces 0-1 onto 2-3
  l1_settle(tile);
  for (unsigned n = 16 * item; n < face; n *= 2) { copy(tile, tile + n, n); copy(tile + face, tile + face + n, n); noc_wait(); }
  copy(tile, tile + 2 * face, 2 * face);
  noc_wait();
}"""  # C shared by the kernels of BRISC's alone: the banks, copies inside this L1, and a row filling its tile

def gather_kernel(dtype):
  """Rows of a tensor into a new one, args [first unit, units, out, the tensor's address, tile stride, tile offset, the
  result's tile columns, rows, the int32 row of their indices' address and tile offset]. BRISC alone: for each result
  tile, the indices' tile in, then each row's two 16-datum runs (left and right face) from wherever that row sits in
  the tensor's tiles. A NoC read from DRAM must keep its address's low 6 bits, so each run comes in as the 64-byte
  block holding it, all of a tile's in flight at once, and is copied into place from there. One row fills its tile: it
  comes in once and is doubled up inside L1 by NoC copies from the tile to itself."""
  tb, fb, run = tile_bytes(dtype), tile_bytes(dtype) // 4, 16 * ITEMSIZE[dtype]  # bytes in a tile, a face, a face's row
  return compile_kernel(f"""
#define SLOT {DATA:#x}
#define IDS {DATA + tb:#x}
#define BLOCKS {DATA + tb + 4096:#x}
{ROWS}
static unsigned id(unsigned r) {{ return ((volatile unsigned *)IDS)[r / 16 * 256 + r % 16]; }}  // index r of this tile's 32, in row 0
static unsigned place(unsigned n, unsigned f) {{ return (n % 32 / 16 * 2 + f) * {fb} + n % 16 * {run}; }}  // row n's run in face f, in its tile
void kmain(volatile unsigned *a) {{
  for (unsigned u = 0; u < a[1]; u++) {{
    unsigned t = a[0] + u, i = t / a[6], j = t % a[6], rows = a[7] - i * 32 < 32 ? a[7] - i * 32 : 32, it = a[9] + i;  // it: the indices' tile
    noc_get(BANK(it), AT(a[8], it, {tile_bytes("int32")}), IDS, {tile_bytes("int32")});
    copy({ZEROS:#x}, SLOT, {tb});  // the rows past the end stay zero
    noc_wait();
    for (unsigned r = 0; r < rows; r++) for (unsigned f = 0; f < 2; f++) {{  // row r's run in face f: its block into BLOCKS's 2 r + f
      unsigned n = id(r), from = n / 32 * a[4] + j + a[5];
      noc_get(BANK(from), AT(a[3], from, {tb}) + place(n, f) / 64 * 64, BLOCKS + (2 * r + f) * 64, 64);
    }}
    noc_wait();
    for (unsigned r = 0; r < rows; r++) for (unsigned f = 0; f < 2; f++) {{
      volatile unsigned *to = (volatile unsigned *)(SLOT + (r / 16 * 2 + f) * {fb} + r % 16 * {run}), *by = (volatile unsigned *)(BLOCKS + (2 * r + f) * 64 + place(id(r), f) % 64);
      for (unsigned k = 0; k < {run // 4}; k++) to[k] = by[k];
    }}
    if (a[7] == 1) row_down(SLOT, {ITEMSIZE[dtype]});
    noc_write(SLOT, BANK(t), AT(a[2], t, {tb}), {tb});
    noc_wait();
  }}
}}""")

def copy_kernel(dtype):
  """Tiles as they are, from a tensor or view into another of its shape (assign's of what is computed already: a block
  float's too), args [first unit, units, then the result's and x's address, tile strides for batch and row, and tile
  offset, then the tile rows and columns]. BRISC alone, a unit a tile, each read while the one before is written."""
  tb = tile_bytes(dtype)
  return compile_kernel(f"""
{BANKS}
void kmain(volatile unsigned *a) {{
  unsigned tr = a[10], tc = a[11];
  for (unsigned u = 0; u < a[1]; u++) {{
    unsigned t = a[0] + u, b = t / (tr * tc), i = t / tc % tr, j = t % tc, slot = {DATA:#x} + u % 2 * {tb};
    unsigned from = b * a[7] + i * a[8] + j + a[9], to = b * a[3] + i * a[4] + j + a[5];
    noc_get(BANK(from), AT(a[6], from, {tb}), slot, {tb});
    noc_wait();  // (the tile before's write too, from the other slot)
    noc_write(slot, BANK(to), AT(a[2], to, {tb}), {tb});
  }}
  noc_wait();
}}""")

def put_kernel(dtype):
  """Rows of x into rows of a tensor, in place (gather's inverse), args [first unit, units, then the tensor's and x's
  address, tile strides for batch and row, and tile offset, the int32 row of the n rows' places' address and tile offset,
  n, the tensor's tile columns and tile rows]. BRISC alone: a unit is a row of x in one of the tensor's matrices and tile
  columns, so the work goes as the rows put, not as the tensor's tiles. Each worker reads the places (32 to a tile) and
  chains the rows that land in each tile row, first to last; the unit of a tile row's first row reads its tile once
  (in flight with its first row's tile of x), each row's tile of x once for every row in it, copies the rows' two runs
  (left and right face) in order, so a later row to the same place stays, and writes the tile back: no two workers
  write one tile. 32 rows that fill a tile in order (a prompt's, from a tile's first place) make it their tile of x as
  it is. A place past the tensor's rows is left out.
  Into a block float (a KV cache in bfp8 or bfp4), x is bf16 and each run of 16 is rounded as a cast rounds it: to its
  largest exponent, bfp8's 7 bits of magnitude half up (as the packer does), bfp4's 3 to the nearest (as bfp4_kernel
  does), a 0 without its sign (which the unpacker would read as -inf)."""
  bfp = dtype in BFP
  t, s, run = tile_bytes(dtype), tile_bytes("bf16" if bfp else dtype), 16 * ITEMSIZE["bf16" if bfp else dtype]  # bytes in a tile of the tensor, of x, in x's run
  copy = ("for (unsigned f = 0; f < 2; f++) pack(FROM + at(i, f), TILE, r, f);" if bfp else
          f"for (unsigned f = 0; f < 2; f++) for (unsigned k = 0; k < {run // 4}; k++) "
          f"((volatile unsigned *)(TILE + at(r, f)))[k] = ((volatile unsigned *)(FROM + at(i, f)))[k];")
  whole = (f"noc_get(BANK(from), AT(a[6], from, {s}), FROM, {s}); noc_wait(); "
           f"for (unsigned r = 0; r < 32; r++) for (unsigned f = 0; f < 2; f++) pack(FROM + at(r, f), TILE, r, f);" if bfp else
           f"noc_get(BANK(from), AT(a[6], from, {t}), TILE, {t}); noc_wait();")
  return compile_kernel(f"""
#define TILE {DATA:#x}
#define FROM {DATA + 4096:#x}
#define IDS {DATA + 8192:#x}
#define PLACES {DATA + 12288:#x}
{BANKS}
static unsigned id(unsigned r) {{ return ((volatile unsigned *)IDS)[r / 16 * 256 + r % 16]; }}  // place r of the 32, in row 0
static unsigned at(unsigned r, unsigned f) {{ return (r % 32 / 16 * 2 + f) * {s // 4} + r % 16 * {run}; }}  // row r's run in face f, in x
static unsigned nibble(unsigned b, unsigned top) {{  // bf16 b's 3 bits under exponent top, rounded to the nearest, and its sign
  unsigned e = b >> 7 & 255, shift = top - e, m = (b & 127) | (e ? 128 : 0), q = shift < 4 ? (m + (16u << shift)) >> (shift + 5) : 0;
  if (q > 7) q = 7;
  return q ? (b >> 15 & 1) << 3 | q : 0;
}}
// the run of 16 bf16 values at src as row r's run in face f of the block-float tile at tile: its exponent, then its values
static void pack(unsigned src, unsigned tile, unsigned r, unsigned f) {{
  volatile unsigned short *v = (volatile unsigned short *)src;
  unsigned top = 0, n = (r % 32 / 16 * 2 + f) * 16 + r % 16;
  for (unsigned k = 0; k < 16; k++) {{ unsigned e = v[k] >> 7 & 255; if (e > top) top = e; }}
  ((volatile unsigned char *)tile)[n] = top;
  for (unsigned k = 0; k < 16; k++) {{
    {"if (k % 2) continue; ((volatile unsigned char *)tile)[64 + n * 8 + k / 2] = nibble(v[k], top) | nibble(v[k + 1], top) << 4;" if dtype == "bfp4" else
     "unsigned b = v[k], e = b >> 7 & 255, m = (b & 127) | (e ? 128 : 0), shift = top - e < 8 ? top - e : 8, q = (m + (1u << shift)) >> (shift + 1); "
     "if (q > 127) q = 127; ((volatile unsigned char *)tile)[64 + n * 16 + k] = q ? (b >> 15) << 7 | q : 0;"}
  }}
}}
void kmain(volatile unsigned *a) {{
  unsigned n = a[12], ct = a[13], tr = a[14];
  // each row's place, the next row to land in its tile row (~0: none), and each tile row's first row
  volatile unsigned *place = (volatile unsigned *)PLACES, *next = place + n, *first = next + n;
  for (unsigned i = 0; i < n; i += 32) {{
    unsigned it = a[11] + i / 32;
    noc_get(BANK(it), AT(a[10], it, {tile_bytes("int32")}), IDS, {tile_bytes("int32")});
    noc_wait();
    for (unsigned k = 0; k < 32 && i + k < n; k++) place[i + k] = id(k);
  }}
  for (unsigned i = 0; i < tr; i++) first[i] = ~0u;
  for (unsigned i = n; i-- > 0;) if (place[i] / 32 < tr) {{ next[i] = first[place[i] / 32]; first[place[i] / 32] = i; }}
  for (unsigned u = 0; u < a[1]; u++) {{
    unsigned unit = a[0] + u, b = unit / (n * ct), i0 = unit / ct % n, j = unit % ct, ti = place[i0] / 32;
    if (ti >= tr || first[ti] != i0) continue;  // the tile of an earlier row's unit, or none
    unsigned to = b * a[3] + ti * a[4] + j + a[5], held = ~0u, whole = i0 % 32 == 0 && i0 + 32 <= n;
    for (unsigned k = 0; whole && k < 32; k++) whole = place[i0 + k] == 32 * ti + k;
    if (whole) {{ unsigned from = b * a[7] + i0 / 32 * a[8] + j + a[9]; {whole} }}
    else noc_get(BANK(to), AT(a[2], to, {t}), TILE, {t});  // (in flight with the first row's tile of x)
    for (unsigned i = i0; i != ~0u; i = next[i]) {{
      if (whole && i < i0 + 32) continue;
      unsigned r = place[i], from = b * a[7] + i / 32 * a[8] + j + a[9];
      if (from != held) noc_get(BANK(from), AT(a[6], from, {s}), FROM, {s}), held = from;
      noc_wait();
      {copy}
    }}
    noc_write(TILE, BANK(to), AT(a[2], to, {t}), {t});
    noc_wait();
  }}
}}""")

def topk_kernel(k, cols, dtype="bf16", parts=1, labels=False):
  """Each row's k largest of its cols and their columns, the first of equals first, args [first unit, units, the
  values' address, the indices', then x's address, tile strides for batch and row, tile offset, and rows, then the
  labels' address, tile strides and offset]. BRISC alone, a unit one row of a tile row of a part of x, 32 to a tile
  row, so a decode step's few rows go to as many workers: the row's runs come in (for more than 8 rows, the part's tile
  row, once for a worker's units of it), and the row is scanned once (one row of x: by the tile row's first unit, which writes the whole tiles), run by run,
  its k best so far a heap with the worst at its root (of equal values, the later column), which a better value
  replaces: once the worst is a float of no sign, a value is checked against it as a signed integer, else by its key
  (bf16 or fp32 as an unsigned integer once its sign orders it, negatives flipped, -0 as 0); then sorted, best first. parts: each row cut into so many parts of cols columns, whole tiles, each part's k
  best side by side in the results ((batch, rows, parts k)), each index its column in the whole row. labels: each index
  taken from an int32 tensor of x's shape at the column picked, read 64 bytes about it (a part's best indices, for a
  topk over the parts' best). Each unit writes its row of each of the results' tiles (bf16 or fp32, and int32), both
  faces, zeros past the k columns and all zeros past x's rows, so a tile is written whole by its 32 units and no two
  write the same bytes; but a dimension of 1 fills its tile with copies, as every tensor's does (one k of 1 the row with
  its column, one row of x every row)."""
  ct, kt, item, bits = -(-cols // 32), -(-k // 32), ITEMSIZE[dtype], 8 * ITEMSIZE[dtype]  # x's tiles a part, the results' a part
  tb = 1024 * item
  assert parts == 1 or (k % 32 == 0 and cols % 32 == 0), "parts of whole tiles, side by side in whole tiles"
  assert DATA + ct * tb + tb + 4096 + 64 * k <= OUTS, "a part's tile row in L1"
  return compile_kernel(f"""
#define X {DATA:#x}
#define VALUES {DATA + ct * tb:#x}
#define IDS {DATA + ct * tb + tb:#x}
#define LABELS {DATA + ct * tb + tb + 4096:#x}
{ROWS}
typedef {"unsigned short" if item == 2 else "unsigned"} datum;
typedef {"short" if item == 2 else "int"} signed_datum;
static unsigned at(unsigned r, unsigned c) {{ return (r / 16 * 2 + c / 16) * 256 + r % 16 * 16 + c % 16; }}  // datum (r, c) in a tile
// The k best so far as a heap, the worst at the root (of equal keys, the later column), so a better value takes its place
static int worse(const unsigned *top, const unsigned *col, unsigned i, unsigned j) {{ return top[i] < top[j] || (top[i] == top[j] && col[i] > col[j]); }}
static void swap(unsigned *top, unsigned *col, unsigned i, unsigned j) {{ unsigned t = top[i], c = col[i]; top[i] = top[j], col[i] = col[j], top[j] = t, col[j] = c; }}
static void down(unsigned *top, unsigned *col, unsigned i, unsigned n) {{
  for (unsigned m = i;; i = m) {{
    unsigned l = 2 * i + 1;
    if (l < n && worse(top, col, l, m)) m = l;
    if (l + 1 < n && worse(top, col, l + 1, m)) m = l + 1;
    if (m == i) return;
    swap(top, col, i, m);
  }}
}}
static unsigned key(unsigned v) {{  // the order of the floats, as unsigned integers
  const unsigned sign = 1u << {bits - 1}, all = {(1 << bits) - 1:#x};
  return v == sign ? sign : v & sign ? all & ~v : v | sign;
}}
void kmain(volatile unsigned *a) {{
  datum *x = (datum *)X;
  volatile datum *values = (volatile datum *)VALUES;  // a result tile of each: this unit's row in it, at its place
  volatile unsigned *ids = (volatile unsigned *)IDS;
  unsigned rows = a[8], tr = (rows + 31) / 32, held = ~0u, done = ~0u, top[{k}], col[{k}], kept = 0;  // done: the row whose k best
  for (unsigned u = 0; u < a[1]; u++) {{                                                                // top and col hold (copies' too)
    unsigned t = (a[0] + u) / 32, r = (a[0] + u) % 32, i = t % tr, p = t / tr % {parts}, b = t / tr / {parts};  // row r of x's
    unsigned first = b * a[5] + i * a[6] + p * {ct} + a[7];                                                    // tile row i, part p, matrix b
    unsigned n = rows == 1 ? 1 : rows - i * 32 < 32 ? rows - i * 32 : 32, s = rows == 1 ? 0 : r;  // x's rows here; the row r holds
    unsigned face = r / 16 * 512 + r % 16 * 16;  // the row's run in the left face, in datums
    if (rows == 1 && r) continue;  // (one row of x: the tile row's first unit writes all its copies)
    if (s < n && t * 32 + s != done) {{
      done = t * 32 + s, kept = 0;
      if (t != held || rows <= 8) {{  // a few rows: each one's own runs, where they lie in its tiles; more: the tile row, once
        for (unsigned j = 0; j < {ct}; j++)                                                                    // a worker
          if (rows <= 8) for (unsigned f = 0; f < 2; f++) {{
            unsigned o = (s / 16 * 2 + f) * {256 * item} + s % 16 * {16 * item};
            noc_get(BANK(first + j), AT(a[4], first + j, {tb}) + o, X + j * {tb} + o, {16 * item});
          }}
          else noc_get(BANK(first + j), AT(a[4], first + j, {tb}), X + j * {tb}, {tb});
        noc_wait();
        held = t;
      }}
      const signed_datum *row = (const signed_datum *)x + (s / 16 * 2) * 256 + s % 16 * 16;  // its run of 16 in face 0 of tile 0
      unsigned last = 0;  // the worst kept's key, and while it is a float of no sign, its bits: a later value is no better
      int raw = -1;       // than it unless above it as a signed integer, one compare where the key takes four
      for (unsigned c0 = 0; c0 < {cols}; c0 += 16) {{  // the row's run of 16 in each face of each tile, in order
        const signed_datum *run = row + c0 / 32 * 1024 + c0 % 32 / 16 * 256;
        if (raw >= 0 && c0 + 16 <= {cols}) {{  // (a whole run no better than the worst kept: one branch for its 16)
          int better = 0;
          for (unsigned c = 0; c < 16; c++) better |= run[c] > raw;
          if (!better) continue;
        }}
        for (unsigned c = c0, end = c0 + 16 < {cols} ? c0 + 16 : {cols}; c < end; c++) {{
          signed_datum d = run[c - c0];
          if (kept == {k} && (raw >= 0 ? d <= raw : key((datum)d) <= last)) continue;  // no better than the worst kept
          unsigned v = key((datum)d), i;
          if (kept < {k}) for (top[i = kept++] = v, col[i] = c; i && worse(top, col, i, (i - 1) / 2); i = (i - 1) / 2) swap(top, col, i, (i - 1) / 2);
          else top[0] = v, col[0] = c, down(top, col, 0, {k});  // (it takes the worst's place)
          if (kept == {k}) last = top[0], raw = last >= {1 << (bits - 1):#x}u ? (int)(last & {(1 << (bits - 1)) - 1:#x}) : -1;
        }}
      }}
      for (unsigned n = kept; n > 1; n--) swap(top, col, 0, n - 1), down(top, col, 0, n - 1);  // the worst to the end in turn: best first
      {f"""for (unsigned j = 0; j < {k}; j++) {{  // each pick's label: the 64 bytes about it
        unsigned c = col[j], lt = b * a[10] + i * a[11] + c / 32 + a[12], o = at(s, c % 32) * 4;
        noc_get(BANK(lt), AT(a[9], lt, 4096) + o / 64 * 64, LABELS + j * 64, 64);
      }}
      noc_wait();""" if labels else ""}
    }}
    for (unsigned q = 0; q < {kt}; q++) {{  // the unit's row of each result tile: its k picks there, 0 past them
      for (unsigned c = 0; c < 32; c++) {{
        unsigned j = q * 32 + c, v = 0, id = 0;
        if (s < n && j < {k}) v = x[col[j] / 32 * 1024 + at(s, col[j] % 32)], id = {f"((volatile unsigned *)(LABELS + j * 64))[at(s, col[j] % 32) % 16]" if labels else f"p * {cols} + col[j]"};
        if (s < n && {k} * {parts} == 1) v = x[col[0] / 32 * 1024 + at(s, col[0] % 32)], id = {"((volatile unsigned *)LABELS)[at(s, col[0] % 32) % 16]" if labels else "col[0]"};  // (copies across)
        values[at(r, c)] = v, ids[at(r, c)] = id;
      }}
      unsigned o = (b * tr + i) * {parts * kt} + p * {kt} + q;  // the result tile
      if (rows == 1) {{  // (whole, its row doubled down by copies inside L1)
        row_down(VALUES, {item}), row_down(IDS, 4);
        noc_write(VALUES, BANK(o), AT(a[2], o, {tb}), {tb}), noc_write(IDS, BANK(o), AT(a[3], o, 4096), 4096);
      }}
      else for (unsigned h = 0; h < 2; h++) {{  // its left and right faces' runs, from the same places in L1
        unsigned d = face + h * 256;
        noc_write(VALUES + d * {item}, BANK(o), AT(a[2], o, {tb}) + d * {item}, {16 * item});
        noc_write(IDS + d * 4, BANK(o), AT(a[3], o, 4096) + d * 4, 64);
      }}
      noc_wait();
    }}
  }}
}}""")

def group_slots(rows, k, n):  # Tensor.group's tiles of slots, enough for any choices (one for each 32, and one more for each
  return -(-rows * k // 32) + min(n, rows * k), -(-rows // 32) * 32  # value that can come), and the length of a run of places

def group_kernel(rows, k, n):
  """A mixture of experts' tokens grouped by expert (Tensor.group), args [first unit, units, the order's address, the
  tiles', the places', then the choices' address, tile stride for rows, and tile offset]. BRISC alone, on one worker:
  the choices (rows x k, values below n) come in, each value's are counted, its run of slots starts where the runs of
  the values before it end, each rounded up to a whole tile, and each choice takes the next slot of its value's run.
  The order (a row of the token each slot holds), the tiles (a column of each tile's value) and the places (a row of k
  runs, each choice's slot) are built in L1 as plain arrays and sent a tile at a time, a row's tile holding the row
  32 times and a column's the column, as every tensor's does."""
  t, rp = group_slots(rows, k, n)
  fill = DATA + -(-rows // 32) * 4096  # past the choices: each value's count, then its next slot
  out = (fill + 4 * (n + 33 * t + k * rp) + 63) // 64 * 64
  assert out + 4096 <= OUTS, "the choices and their arrays in L1"
  return compile_kernel(f"""
#define IDS {DATA:#x}
#define FILL {fill:#x}
#define ORDER (FILL + {4 * n})
#define TILES (ORDER + {4 * 32 * t})
#define PLACES (TILES + {4 * t})
#define OUT {out:#x}
{ROWS}
static unsigned at(unsigned r, unsigned c) {{ return (r / 16 * 2 + c / 16) * 256 + r % 16 * 16 + c % 16; }}  // datum (r, c) in a tile
static unsigned choice(unsigned r, unsigned j) {{ return ((volatile unsigned *)IDS)[r / 32 * 1024 + at(r % 32, j)]; }}
static void send(unsigned base, unsigned i) {{ noc_write(OUT, BANK(i), AT(base, i, 4096), 4096); noc_wait(); }}
void kmain(volatile unsigned *a) {{
  volatile unsigned *fill = (volatile unsigned *)FILL, *order = (volatile unsigned *)ORDER;
  volatile unsigned *tiles = (volatile unsigned *)TILES, *places = (volatile unsigned *)PLACES, *out = (volatile unsigned *)OUT;
  if (!a[1]) return;
  for (unsigned i = 0; i < {-(-rows // 32)}; i++) {{ unsigned u = a[7] + i * a[6]; noc_get(BANK(u), AT(a[5], u, 4096), IDS + i * 4096, 4096); }}
  for (unsigned e = 0; e < {n}; e++) fill[e] = 0;
  for (unsigned s = 0; s < {32 * t}; s++) order[s] = 0;
  for (unsigned i = 0; i < {t}; i++) tiles[i] = {n};  // (none: a matmul picking by them skips the tiles past those used)
  for (unsigned s = 0; s < {k * rp}; s++) places[s] = 0;
  noc_wait();
  for (unsigned r = 0; r < {rows}; r++) for (unsigned j = 0; j < {k}; j++) fill[choice(r, j)]++;
  for (unsigned e = 0, s = 0; e < {n}; e++) {{  // each value's run: its count rounded up to whole tiles, after the runs before it
    unsigned count = fill[e];
    fill[e] = s;
    for (unsigned end = s + (count + 31) / 32 * 32; s < end; s += 32) tiles[s / 32] = e;
  }}
  for (unsigned r = 0; r < {rows}; r++)
    for (unsigned j = 0; j < {k}; j++) {{ unsigned s = fill[choice(r, j)]++; order[s] = r; places[j * {rp} + r] = s; }}
  for (unsigned i = 0; i < {t}; i++) {{  // the order and the places, rows: each tile's row 0, copied down it; the tiles, a column
    for (unsigned c = 0; c < 32; c++) out[at(0, c)] = order[i * 32 + c];
    row_down(OUT, 4), send(a[2], i);
  }}
  for (unsigned i = 0; i < {-(-t // 32)}; i++) {{
    for (unsigned r = 0; r < 32; r++) for (unsigned c = 0; c < 32; c++) out[at(r, c)] = i * 32 + r < {t} ? tiles[i * 32 + r] : {n};
    send(a[3], i);
  }}
  for (unsigned i = 0; i < {k * rp // 32}; i++) {{
    for (unsigned c = 0; c < 32; c++) out[at(0, c)] = places[i * 32 + c];
    row_down(OUT, 4), send(a[4], i);
  }}
}}""")

def touched_kernel(rows, k, n):
  """A mixture of experts' choices as the experts some row picks (Tensor.touched), args [first unit, units, the
  experts' address, the weights', then the choices' address, tile stride for rows and tile offset, then the choices'
  weights' (bf16) the same]. BRISC alone, a unit an entry: each worker reads the choices (rows x k, values below n) and
  their weights once, marks the experts picked, and lists them in increasing order, then n (none) past them; entry s
  writes its tile of each of its weights' tile rows, row r the weight row r gave expert s of the list (0 if it picked
  it not), across the row, and the entries' first of each 32 their tile of the experts' column, each its expert across
  its row, as a column's tile holds it."""
  tr, slots = -(-rows // 32), min(rows * k, n)
  return compile_kernel(f"""
#define IDS {DATA:#x}
#define WEIGHTS {DATA + tr * 4096:#x}
#define LIST {DATA + tr * 6144:#x}
#define OUT {(DATA + tr * 6144 + 4 * n + 4 * slots + 63) // 64 * 64:#x}
{ROWS}
static unsigned at(unsigned r, unsigned c) {{ return (r / 16 * 2 + c / 16) * 256 + r % 16 * 16 + c % 16; }}  // datum (r, c) in a tile
static unsigned choice(unsigned r, unsigned j) {{ return ((const unsigned *)IDS)[r / 32 * 1024 + at(r % 32, j)]; }}  // (L1 the NoC wrote,
static unsigned weight(unsigned r, unsigned j) {{ return ((const unsigned short *)WEIGHTS)[r / 32 * 1024 + at(r % 32, j)]; }}  // past noc_wait's fence)
void kmain(volatile unsigned *a) {{
  unsigned *list = (unsigned *)LIST, *mark = list + {slots}, *out = (unsigned *)OUT;
  if (!a[1]) return;
  for (unsigned i = 0; i < {tr}; i++) {{
    unsigned u = a[6] + i * a[5], v = a[9] + i * a[8];
    noc_get(BANK(u), AT(a[4], u, 4096), IDS + i * 4096, 4096), noc_get(BANK(v), AT(a[7], v, 2048), WEIGHTS + i * 2048, 2048);
  }}
  copy({ZEROS:#x}, OUT, 4096);  // (a weights' tile's rows past the choices' stay 0)
  for (unsigned e = 0; e < {n}; e++) mark[e] = 0;
  noc_wait();
  for (unsigned r = 0; r < {rows}; r++) for (unsigned j = 0; j < {k}; j++) mark[choice(r, j)] = 1;
  unsigned count = 0;
  for (unsigned e = 0; e < {n}; e++) if (mark[e]) list[count++] = e;
  for (; count < {slots}; count++) list[count] = {n};
  for (unsigned u = 0; u < a[1]; u++) {{
    unsigned s = a[0] + u, e = list[s];
    for (unsigned i = 0; i < {tr}; i++) {{  // its weights' tile row i: each row's weight for e, across the row
      for (unsigned r = 0; r < 32 && i * 32 + r < {rows}; r++) {{
        unsigned w = 0;
        for (unsigned j = 0; j < {k}; j++) if (choice(i * 32 + r, j) == e) w = weight(i * 32 + r, j);
        for (unsigned *row = out + (r / 16 * 2) * 128 + r % 16 * 8, c = 0; c < 8; c++) row[c] = row[128 + c] = w | w << 16;  // (its runs in both faces)
      }}
      unsigned t = s * {tr} + i;
      l1_settle(OUT);
      noc_write(OUT, BANK(t), AT(a[3], t, 2048), 2048), noc_wait();
    }}
    if (s % 32) continue;
    for (unsigned r = 0; r < 32; r++) for (unsigned *row = out + (r / 16 * 2) * 256 + r % 16 * 16, e = s + r < {slots} ? list[s + r] : {n}, c = 0; c < 16; c++)
      row[c] = row[256 + c] = e;                                                                                           // the experts'
    l1_settle(OUT);                                                                                                        // tile of entries
    noc_write(OUT, BANK(s / 32), AT(a[2], s / 32, 4096), 4096), noc_wait();                                                // s to s + 31,
    copy({ZEROS:#x}, OUT, 4096), noc_wait();                                                                              // each across its row
  }}
}}""")

LINK_L1, LINK_TILES = 0x20000, 4  # where a copy across an Ethernet link keeps its two slots in each end's tile, of so many tiles

def link_kernel(send, dtype):
  """One end of a copy across an Ethernet link (Tensor.to), args [first unit, units, the tensor's address, its tiles, this
  chip's Ethernet tile's x and y]. BRISC alone, on one worker. The sender reads the tensor LINK_TILES tiles at a time,
  writes each chunk into a slot (two take turns) in its Ethernet tile's L1, and has the tile's TX queue write it, then a
  flag (the chunk's number + 1), into the same places in the peer's: the link delivers in order, so the chunk is there
  when its flag is. The receiver waits for each flag, writes the chunk into its tensor and sends back a credit (chunks
  done), which the sender waits for before it fills a slot again, and for the last one before it ends, so that the next
  copy's flags and credit can start at 0. A word of another tile's (the queue's registers, a
  flag, the credit) is read and written over the NoC, from a place in L1 with the same low 4 bits, as the NoC wants."""
  tb, q = tile_bytes(dtype), lambda name: f"{bh.ETH_TXQ0 + bh.ETH_TXQ[name]:#x}"
  slot = LINK_TILES * tb
  body = f"""
    while (c >= 2 && peek(CREDIT) < c - 1) {{}}  // slot s is free once chunk c - 2 is done
    for (unsigned i = 0; i < n; i++) noc_get(BANK(t + i), AT(a[2], t + i, {tb}), BUF + i * {tb}, {tb});
    noc_wait();
    noc_write(BUF, ex, ey, SLOT(s), n * {tb}), noc_wait();
    link_write(SLOT(s), n * {tb});
    poke(FLAG(s), c + 1), link_write(FLAG(s), 16);""" if send else f"""
    while (peek(FLAG(s)) != c + 1) {{}}  // the chunk is here once its flag is
    noc_get(ex, ey, SLOT(s), BUF, n * {tb}), noc_wait();
    for (unsigned i = 0; i < n; i++) noc_write(BUF + i * {tb}, BANK(t + i), AT(a[2], t + i, {tb}), {tb});
    noc_wait();
    poke(CREDIT, c + 1), link_write(CREDIT, 16);"""
  end = f"\n  while (peek(CREDIT) < (a[3] + {LINK_TILES - 1}) / {LINK_TILES}) {{}}  // the last credit is back: nothing of this copy is left on the link" if send else ""
  return compile_kernel(f"""
#define BUF {DATA:#x}
#define WORD {DATA + slot:#x}
#define SLOT(s) ({LINK_L1:#x} + (s) * {slot})
#define FLAG(s) ({LINK_L1 + 2 * slot:#x} + 16 * (s))
#define CREDIT {LINK_L1 + 2 * slot + 32:#x}
{BANKS}
static unsigned ex, ey;  // this end's Ethernet tile
static unsigned peek(unsigned addr) {{ unsigned at = WORD + (addr & 15); noc_get(ex, ey, addr, at, 4); noc_wait(); return *(volatile unsigned *)at; }}
static void poke(unsigned addr, unsigned v) {{ unsigned at = WORD + 16 + (addr & 15); *(volatile unsigned *)at = v; noc_write(at, ex, ey, addr, 4); noc_wait(); }}
static void link_write(unsigned at, unsigned n) {{  // n bytes of the tile's L1 at `at` into the peer's, by TX queue 0
  do peek({q("CMD")}); while (peek({q("STATUS")}) & {bh.ETH_BUSY:#x});  // busy read after another register (BH-55)
  poke({q("START")}, at), poke({q("SIZE")}, n), poke({q("DEST")}, at), poke({q("CMD")}, {bh.ETH_TT_L1_WRITE});
}}
void kmain(volatile unsigned *a) {{
  if (!a[1]) return;
  ex = a[4], ey = a[5];
  for (unsigned c = 0, t = 0; t < a[3]; c++, t += {LINK_TILES}) {{
    unsigned n = a[3] - t < {LINK_TILES} ? a[3] - t : {LINK_TILES}, s = c % 2;{body}
  }}{end}
}}""")

def readback_kernel(dtype):
  """A tensor's tiles into host memory (Runtime.back) for tobytes, since a read of DRAM over the BARs waits on PCIe for
  each load: args [first unit, units, the first tile, the tensor's address, tile strides for batch and row, tile offset,
  tile columns, tiles a matrix, repeat, host address low and high, rows a tile]. BRISC alone, a tile a unit, its tiles
  in order from the first, through two L1 buffers, so a tile's read overlaps the last one's write. A tensor of fewer than
  32 rows sends each face's rows that hold values and no more (a token's logits: one row of each), back to back; a
  block float's tiles go whole."""
  n, item = tile_bytes(dtype), 0 if dtype in BFP else ITEMSIZE[dtype]
  return compile_kernel(f"""
{BANKS}
void kmain(volatile unsigned *a) {{
  unsigned long long host = a[10] | (unsigned long long)a[11] << 32;
  unsigned v = a[12], sent = {f"v * 32 * {item}" if item else n};  // the rows sent, and their bytes
  for (unsigned u = 0; u < a[1]; u++) {{
    unsigned k = a[0] + u, t = a[2] + k, b = t / a[8], i = t % a[8] / a[7], j = t % a[7], from = b / a[9] * a[4] + i * a[5] + j + a[6];
    unsigned l1 = {DATA:#x} + u % 2 * {n};
    unsigned long long to = host + (unsigned long long)k * sent;
    noc_get(BANK(from), AT(a[3], from, {n}), l1, {n});
    noc_wait();  // and the writes out of this buffer two tiles ago are done
    if (sent == {n}) noc_write(l1, {bh.PCIE[0]}, {bh.PCIE[1]}, to, {n});
    else for (unsigned f = 0; f < 4; f++) {{  // faces 0 and 1 hold rows 0-15, 2 and 3 rows 16-31
      unsigned r = f < 2 ? (v < 16 ? v : 16) : (v > 16 ? v - 16 : 0);
      if (r) noc_write(l1 + f * {256 * item}, {bh.PCIE[0]}, {bh.PCIE[1]}, to, r * {16 * item});
      to += r * {16 * item};
    }}
  }}
  noc_wait();
}}""")

def bfp4_kernel():
  """bf16 tiles as bfp4, args [first unit, units, out, x's address, tile strides for batch and row, tile offset, x's
  tile columns, tiles a matrix]. BRISC alone, a tile a unit: each run of 16 values (a face's row) shares its largest
  exponent, and each value keeps the 3 bits under it, rounded to the nearest (ties up), as Tenstorrent's host conversion
  rounds; the packer's own bfp4 is its bfp8 with the low 4 bits dropped, toward zero, which moved SmolLM2-360M's logits
  three times as far. Whole words only, in and out."""
  return compile_kernel(f"""
#define IN {DATA:#x}
#define OUT {DATA + 2048:#x}
{BANKS}
static unsigned nibble(unsigned b, unsigned top) {{  // bf16 b's 3 bits under exponent top, rounded, and its sign
  unsigned e = b >> 7 & 255, shift = top - e, m = (b & 127) | (e ? 128 : 0), q = shift < 4 ? (m + (16u << shift)) >> (shift + 5) : 0;
  if (q > 7) q = 7;
  return q ? (b >> 15 & 1) << 3 | q : 0;  // no sign on a zero: the unpacker reads -0 as -inf
}}
void kmain(volatile unsigned *a) {{
  for (unsigned u = 0; u < a[1]; u++) {{
    unsigned t = a[0] + u, b = t / a[8], i = t % a[8] / a[7], j = t % a[7], from = b * a[4] + i * a[5] + j + a[6], tops = 0;
    noc_get(BANK(from), AT(a[3], from, {tile_bytes("bf16")}), IN, {tile_bytes("bf16")});
    noc_wait();
    for (unsigned r = 0; r < 64; r++) {{  // a run: 8 words of 2 values, its largest exponent, then 2 words of 8 nibbles
      unsigned v[8], top = 0;
      for (unsigned k = 0; k < 8; k++) {{
        v[k] = ((volatile unsigned *)(IN + 32 * r))[k];
        unsigned e0 = v[k] >> 7 & 255, e1 = v[k] >> 23 & 255;
        if (e0 > top) top = e0;
        if (e1 > top) top = e1;
      }}
      tops |= top << 8 * (r % 4);
      if (r % 4 == 3) {{ ((volatile unsigned *)OUT)[r / 4] = tops; tops = 0; }}
      for (unsigned w = 0; w < 2; w++) {{
        unsigned word = 0;
        for (unsigned k = 0; k < 4; k++) word |= (nibble(v[4 * w + k], top) | nibble(v[4 * w + k] >> 16, top) << 4) << 8 * k;
        ((volatile unsigned *)(OUT + 64 + 8 * r))[w] = word;
      }}
    }}
    noc_write(OUT, BANK(t), AT(a[2], t, {tile_bytes("bfp4")}), {tile_bytes("bfp4")});
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
    _math({}, loop(*[tt(math_begin(False) + elementwise("add", {}) + MATH_END) + f" h ^= {DST_HALF};"] * 2)),  # padding stays 0
    _packer(loop(*[f"output({f}, {frame(f)[1]}, t); " + tt(pack("h", frame(f)[1], SEM_PACKED)) + f" h ^= {DST_HALF};"
                   for f in (0, 1)])))

MOVE_SLOTS = 127  # x's tiles cached in L1 by a move, by tile number mod a prime, so power-of-two strides spread out

def move_kernel(dtype):
  """A tensor moved around: reshape, permute, expand, pad, slice. args [first unit, units, out, x's address, tile
  strides for batch and row, tile offset, x's rows and columns, the result's rows and columns, then the map: an offset
  and strides for batch, row and column, then the bounds: lo for batch, row and column, then hi]. Element (b, r, c) of the result, inside
  the bounds, is element offset + b sb + r sr + c sc of x, counted in row-major order; outside them, 0.
  BRISC alone: a NoC read cannot put 2 bytes just anywhere, so x's tiles come whole into a cache in L1 as they are
  needed. A result tile inside the bounds that is one of x's tiles as it is (a pad, slice or cat on tile boundaries)
  comes straight into place; one that is x's tile transposed is transposed face by face. Any other starts as a copy of
  the zero tile and gets the datums inside the bounds, in lines along its rows or down its columns, whichever lie in x as
  runs (along a row of x, or down a column): in runs as long as a face allows, else one at a time, stepping through x
  without a division. A result of one column is made once and copied across; one of one row fills its tile with copies
  of it, the row made once and doubled down the tile by NoC copies inside L1."""
  tb = tile_bytes(dtype)
  return compile_kernel(f"""
#define OUT {DATA:#x}
#define CACHE {DATA + tb:#x}
typedef {"unsigned short" if ITEMSIZE[dtype] == 2 else "unsigned"} datum;
{ROWS}
// What each cache slot holds, x's tile number and the launch it came in: kept from launch to launch (in .data, which the
// statics zeroed at each launch are not), so no launch spends a store a slot clearing it, and a launch's own number
// makes another's tiles none of its own
__attribute__((section(".data"))) static unsigned held[{MOVE_SLOTS}], came[{MOVE_SLOTS}], launch;
static volatile datum *fetch(unsigned base, unsigned t) {{  // x's tile t, in L1
  unsigned s = t % {MOVE_SLOTS};
  if (held[s] == t && came[s] == launch) return (volatile datum *)(CACHE + s * {tb});
  noc_get(BANK(t), AT(base, t, {tb}), CACHE + s * {tb}, {tb});
  noc_wait();
  held[s] = t, came[s] = launch;
  return (volatile datum *)(CACHE + s * {tb});
}}
static unsigned at(unsigned r, unsigned c) {{ return (r / 16 * 2 + c / 16) * 256 + r % 16 * 16 + c % 16; }}  // datum (r, c) in a tile
// n datums from s to d, steps dd and ss apart: four at a time along a line of both, and bf16's two to a word
static void run(volatile datum *d, volatile datum *s, unsigned n, unsigned dd, unsigned ss) {{
  if (dd == 1 && ss == 1) {{
    if (sizeof(datum) == 2 && !(((unsigned)d | (unsigned)s) & 3)) {{
      volatile unsigned *dw = (volatile unsigned *)d, *sw = (volatile unsigned *)s;
      for (; n >= 8; n -= 8, dw += 4, sw += 4) dw[0] = sw[0], dw[1] = sw[1], dw[2] = sw[2], dw[3] = sw[3];
      d = (volatile datum *)dw, s = (volatile datum *)sw;
    }}
    for (; n >= 4; n -= 4, d += 4, s += 4) d[0] = s[0], d[1] = s[1], d[2] = s[2], d[3] = s[3];
  }}
  for (; n; n--, d += dd, s += ss) *d = *s;
}}
// The places [from, to) of tile k along an axis n long whose element is inside [lo, hi); a dimension of 1 fills its tile
static int from(int k, int n, int lo) {{ return n == 1 || lo <= 32 * k ? 0 : lo - 32 * k; }}
static int to(int k, int n, int lo, int hi) {{
  if (n == 1) return lo <= 0 && 0 < hi ? 32 : 0;
  int end = (hi < n ? hi : n) - 32 * k;
  return end < 32 ? end : 32;
}}
void kmain(volatile unsigned *a) {{
  unsigned base = a[3], bstride = a[4], stride = a[5], offset = a[6], xr = a[7], xc = a[8], rows = a[9], cols = a[10];
  launch++;
  unsigned tr = (rows + 31) / 32, tc = (cols + 31) / 32;
  int e0 = a[11], sb = a[12], sr = a[13], sc = a[14], lo_b = a[15], lo_r = a[16], lo_c = a[17], hi_b = a[18], hi_r = a[19], hi_c = a[20];
  // The datums go in lines along the result's rows, or down its columns when that is where they lie in x as runs (along
  // a row of x, or down a column) and the rows are not, or are shorter, and always for a single column: step is a line's,
  // in x's elements, split into x's matrices, rows and columns so that stepping takes no division
  int run_along = cols > 1 && (sc == 1 || sc == (int)xc), run_down = sr == 1 || sr == (int)xc;
  int down = rows > 1 && (cols == 1 || run_down && (!run_along || (rows < 32 ? rows : 32) > (cols < 32 ? cols : 32)));
  int step = down ? sr : cols == 1 ? 0 : sc;
  unsigned sq = step >= 0 ? step / xc : 0, sm = step >= 0 ? step % xc : 0, qb = sq / xr, qi = sq % xr;
  unsigned whole = rows > 1 && cols > 1 && sc == 1 && sr == (int)xc, turned = rows > 1 && cols > 1 && sr == 1 && sc == (int)xc;
  volatile datum *out = (volatile datum *)OUT, *tile = 0;
  unsigned held_t = ~0u;
  for (unsigned u = 0; u < a[1]; u++) {{
    unsigned t = a[0] + u;
    int b = t / (tr * tc), i = t % (tr * tc) / tc, j = t % tc;
    int r0 = from(i, rows, lo_r), r1 = lo_b <= b && b < hi_b ? to(i, rows, lo_r, hi_r) : 0, c0 = from(j, cols, lo_c), c1 = to(j, cols, lo_c, hi_c);
    unsigned e = e0 + b * sb + i * 32 * sr + j * 32 * sc, row = e / xc % xr, col = e % xc;  // where the tile starts in x
    unsigned src = e / xc / xr * bstride + row / 32 * stride + col / 32 + offset;
    if ((whole || turned) && r0 == 0 && r1 == 32 && c0 == 0 && c1 == 32 && row % 32 == 0 && row + 32 <= xr && col % 32 == 0 && col + 32 <= xc) {{
      if (whole) {{ noc_get(BANK(src), AT(base, src, {tb}), OUT, {tb}); noc_wait(); }}  // x's tile src as it is
      else {{  // x's tile src transposed: face (fr, fc) is its face (fc, fr) transposed
        tile = fetch(base, src);  // held, as the lines below expect of held_t
        held_t = src;
        for (unsigned f = 0; f < 4; f++)
          for (unsigned r = 0; r < 16; r++)
            for (unsigned c = 0; c < 16; c++) out[f * 256 + r * 16 + c] = tile[(f % 2 * 2 + f / 2) * 256 + c * 16 + r];
      }}
    }} else {{
      if (rows == 1 && r1) {{  // a single row: made once, doubled below (so only its row starts as zeros)
        for (unsigned c = 0; c < 16; c++) out[c] = 0, out[256 + c] = 0;
        r1 = 1;
      }} else {{ copy({ZEROS:#x}, OUT, {tb}); noc_wait(); }}  // padding and what lies outside the bounds stay 0
      int lines0 = down ? c0 : r0, lines1 = down ? (cols == 1 && c1 ? 1 : c1) : r1, k0 = down ? r0 : c0, k1 = down ? r1 : c1;
      for (int line = lines0; line < lines1; line++) {{
        int r = down ? k0 : line, c = down ? line : k0;
        unsigned e_line = e0 + b * sb + (rows == 1 ? 0 : i * 32 + r) * sr + (cols == 1 ? 0 : j * 32 + c) * sc;
        unsigned xrow = e_line / xc, xb = xrow / xr, xi = xrow % xr, xj = e_line % xc;
        for (int k = k0; k < k1;) {{
          unsigned xt = xb * bstride + xi / 32 * stride + xj / 32 + offset;
          if (xt != held_t) {{ tile = fetch(base, xt); held_t = xt; }}
          if (step == 1 || step == (int)xc) {{  // a run along x's row or down its column, to the end of a face's line, or x's
            unsigned in_row = step == 1, n = k1 - k, m = 16 - k % 16, f = 16 - (in_row ? xj : xi) % 16, left = in_row ? xc - xj : xr - xi;
            n = n < m ? n : m; n = n < f ? n : f; n = n < left ? n : left;
            run(out + at(r, c), tile + at(xi % 32, xj % 32), n, down ? 16 : 1, in_row ? 1 : 16);
            k += n;
            if (down) r += n; else c += n;
            if (in_row && (xj += n) == xc) {{ xj = 0; xi++; }}
            if (!in_row) xi += n;
            if (xi == xr) {{ xi = 0; xb++; }}
            continue;
          }}
          out[at(r, c)] = tile[at(xi % 32, xj % 32)];
          if (++k >= k1) break;
          if (down) r++; else c++;
          if (step < 0) {{  // backwards: found again, by division
            unsigned ek = e_line + (k - k0) * step;
            xrow = ek / xc; xb = xrow / xr; xi = xrow % xr; xj = ek % xc;
            continue;
          }}
          xj += sm; xi += qi; xb += qb;
          if (xj >= xc) {{ xj -= xc; xi++; }}
          if (xi >= xr) {{ xi -= xr; xb++; }}
        }}
      }}
      if (down && cols == 1)  // a single column: made once, copied across, along its row's runs in the left and right faces
        for (int r = r0; r < r1; r++) {{
          volatile datum *left = out + at(r, 0), *right = out + at(r, 16);
          datum v = *left;
          for (unsigned c = 1; c < 16; c++) left[c] = v;
          for (unsigned c = 0; c < 16; c++) right[c] = v;
        }}
      if (rows == 1 && r1) row_down(OUT, {ITEMSIZE[dtype]});
    }}
    noc_write(OUT, BANK(t), AT(a[2], t, {tb}), {tb});
    noc_wait();
  }}
}}""")
