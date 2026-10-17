# The compiler: tensor ops emitted as Tensix instructions straight from the ISA, no LLK. A tensor lives in DRAM as
# bf16 32x32 tiles, tile i in bank i % 8. Each worker's BRISC runs a generated kernel: for each of its tiles it pulls the
# operands into L1 over the NoC, pushes one fixed Tensix program into thread 0 (unpack -> FPU -> pack), waits for it,
# and writes the result back. For an elementwise op the layout inside a tile does not matter; it is row-major here.
import struct
from trickle import bh
from trickle.runtime import DATA, KERNEL, compile_rv32

TILE = 32 * 32 * 2  # bytes in a bf16 tile: one SrcA/SrcB bank (64 rows of 16), 64 Dst rows
A, B, C = DATA, DATA + TILE, DATA + 2 * TILE  # where the current tile of each operand sits in L1
DONE_GPR = 63  # the program ends by writing a sequence number into this GPR, which BRISC polls

def ins(name, **fields):
  opcode, layout = bh.TENSIX_OPS[name]
  word = opcode << 24
  for k, v in fields.items():
    first, width = layout[k]
    assert 0 <= v < 1 << width, f"{name} {k}={v} does not fit in {width} bits"
    word |= v << first
  return word

def pack_fields(table, **fields):  # {register index: value} with the named fields set, everything else in them 0
  regs = {}
  for k, v in fields.items():
    index, first, width = table[k]
    assert 0 <= v < 1 << width, f"{k}={v} does not fit in {width} bits"
    regs[index] = regs.get(index, 0) | v << first
  return regs

# Unpackers read one whole tile each (1024 datums) into a SrcA/SrcB bank. The FPU works on 8 rows of 16 per instruction
# into Dst, kept in fp32. The packer reads 4 Dst rows per PACR and rounds to bf16 on the way to L1.
CONFIG = pack_fields(bh.TENSIX_CFG_FIELDS,
  Dstacc=bh.BF16, Fp32_enabled=1, Zero_Flag_disabled_src=0,
  PCK0_Ystride=32, PCK0_ZW=0, PCK0_Ystride_1=0, PCK0_ZW_1=0, PCK0_Base_1=0, Read_32b_data=1, Edge_mask=0xFFFF,
  pack_reads_per_xy_plane=16, UNP0_add_dest_addr_cntr=0, UNP0_Ystride_1=0, UNP0_ZW_1=0, UNP1_ZW_1=0,
  SEC0_format=bh.BF16, SEC0_x_dim=1024, SEC0_y_dim=1, SEC0_z_dim=1, SEC0_out_format=bh.BF16, SEC0_uncompressed=1,
  SEC0_base=A // 16 - 1, SEC0_dest_addr=64, SEC0_tile_x_dim=1024, SEC0_offset=0,  # base addresses: 16-byte units, minus 1
  SEC1_format=bh.BF16, SEC1_x_dim=1024, SEC1_y_dim=1, SEC1_z_dim=1, SEC1_out_format=bh.BF16, SEC1_uncompressed=1,
  SEC1_base=B // 16 - 1, L1_Dest_addr=C // 16 - 1, Disable_zero_compress=1, Pack_out_format=bh.BF16, Pack_in_format=bh.BF16,
  PCK_dest_offset=0)
THREAD = [ins("SETC16", setc16_reg=r, setc16_value=v) for r, v in pack_fields(bh.TENSIX_THREAD_FIELDS,
  StateID=0, MATH_Offset=0, DISABLE_IMPLIED_SRCA=0, DISABLE_IMPLIED_SRCB=0, SRCA_SET=0b100,  # bit 2: Src rows by address
  CLR_DVALID=0, FIDELITY_BASE=0, AB0_SrcAIncr=8, AB0_SrcBIncr=8, AB1=0, DST0_DestIncr=8, DST1_FidelityIncr=1,
  PACK0_YsrcIncr=4, UNPACK_MISC=0, BIAS0=0, BIAS1=0).items()]

FPU = {"add": "ELWADD", "sub": "ELWSUB", "mul": "ELWMUL"}

def tile_program(op):
  """Thread 0 program for one tile: C = A op B, all in L1. Address mode 0 steps 8 rows; mode 1 steps the fidelity phase."""
  p = [ins("SETRWC", bit_mask=0b1111),  # SrcA, SrcB, Dst row counters and the fidelity phase back to 0
       ins("SETADCXY", cnt_set_mask=bh.UNPACK0 | bh.UNPACK1 | bh.PACK, bit_mask=0b1010),  # y counters to 0
       ins("SETADCZW", cnt_set_mask=bh.UNPACK0 | bh.UNPACK1 | bh.PACK, bit_mask=0b1111),
       ins("SETADCXX", cnt_set_mask=bh.UNPACK0 | bh.UNPACK1, x_end2=1023),  # unpack datums 0..1023
       ins("SETADCXX", cnt_set_mask=bh.PACK, x_end2=15),  # pack 16 datums a row
       ins("UNPACR", last=1, set_dat_valid=1, ovrd_thread_id=1),  # A into SrcA, then hand the bank to the FPU
       ins("UNPACR", last=1, set_dat_valid=1, ovrd_thread_id=1, unpack_block_selection=1)]  # B into SrcB
  if op == "mul":  # ELWMUL always accumulates, a slice of the mantissas per fidelity phase: zero Dst, run all 4 phases
    p.append(ins("ZEROACC", clear_mode=3, use_32_bit_mode=1))
    for phase in range(4):
      p.append(ins("SETRWC", bit_mask=0b0111))
      p += [ins("ELWMUL", addr_mode=int(i == 7), clear_dvalid=3 * (phase == 3 and i == 7)) for i in range(8)]
  else: p += [ins(FPU[op], clear_dvalid=3 * (i == 7)) for i in range(8)]
  p.append(ins("STALLWAIT", stall_res=bh.STALL_PACK, wait_res=bh.WAIT_FPU))  # pack only what the FPU has finished
  p += [ins("PACR", last=int(i == 15)) for i in range(16)]  # read_intf_sel 0: all four Dst read ports, 4 rows each
  p.append(ins("STALLWAIT", stall_res=bh.STALL_THCON, wait_res=bh.WAIT_PACK))  # the done flag waits for the packer
  return p

def _array(name, words): return f"static const unsigned {name}[] = {{{', '.join(f'{w:#x}' for w in words)}}};"

def kernel(op):
  """BRISC kernel for C = A op B over tiles [args[0], args[0]+args[1]) of tensors at DRAM addresses args[2..4]."""
  done = ins("SETGPR", reg_index_16b=2 * DONE_GPR)
  return compile_rv32(f"""
{_array("cfg_index", CONFIG)} {_array("cfg_value", CONFIG.values())}
{_array("thread", THREAD)} {_array("program", tile_program(op))} {_array("bank", [x | y << 6 for x, y in bh.DRAM])}
#define PUSH (*(volatile unsigned *){bh.TENSIX_PUSH:#x})
#define DONE (((volatile unsigned *){bh.TENSIX_GPR:#x})[{DONE_GPR}])
#define CFG ((volatile unsigned *){bh.TENSIX_CFG:#x})
void kmain(volatile unsigned *a) {{
  for (unsigned i = 0; i < sizeof cfg_index / 4; i++) CFG[cfg_index[i]] = cfg_value[i];
  DONE = 0;
  asm volatile("fence" ::: "memory");  // config and GPR writes are out; reading back from each region lands after them
  (void)CFG[cfg_index[0]]; (void)DONE;
  asm volatile("fence" ::: "memory");  // and has answered, so no instruction pushed below can see stale config
  for (unsigned i = 0; i < sizeof thread / 4; i++) PUSH = thread[i];
  for (unsigned n = 1, t = a[0]; t < a[0] + a[1]; t++, n++) {{
    unsigned xy = bank[t % 8], at = t / 8 * {TILE};
    noc_read(xy & 63, xy >> 6, a[2] + at, {A:#x}, {TILE});
    noc_read(xy & 63, xy >> 6, a[3] + at, {B:#x}, {TILE});
    noc_wait();
    for (unsigned i = 0; i < sizeof program / 4; i++) PUSH = program[i];
    PUSH = {done:#x} | (n & 0xFFFF) << 8;
    while ((DONE & 0xFFFF) != (n & 0xFFFF)) {{}}
    noc_write({C:#x}, xy & 63, xy >> 6, a[4] + at, {TILE});
    noc_wait();
  }}
}}
""", KERNEL)

def to_bf16(x):
  bits = struct.unpack("<I", struct.pack("<f", x))[0]
  return (bits + 0x7FFF + (bits >> 16 & 1)) >> 16  # round to nearest even
def from_bf16(b): return struct.unpack("<f", struct.pack("<I", b << 16))[0]

class Tensor:
  """rows x cols of bf16 in DRAM, as 32x32 tiles in row-major tile order, tile i in bank i % 8."""
  def __init__(self, rt, rows, cols):
    assert rows % 32 == 0 and cols % 32 == 0, "tensors are whole tiles"
    self.rt, self.rows, self.cols, self.tiles = rt, rows, cols, rows * cols // 1024
    self.addr = rt.dram_alloc(-(-self.tiles // len(bh.DRAM)) * TILE)

  @classmethod
  def of(cls, rt, rows):
    t = cls(rt, len(rows), len(rows[0]))
    for i in range(t.tiles):
      r, c = i // (t.cols // 32) * 32, i % (t.cols // 32) * 32
      data = struct.pack("<1024H", *[to_bf16(v) for row in rows[r:r + 32] for v in row[c:c + 32]])
      t.rt.chip.write(*bh.DRAM[i % 8], t.addr + i // 8 * TILE, data)
    return t

  def tolist(self):
    out = [[0.0] * self.cols for _ in range(self.rows)]
    for i in range(self.tiles):
      r, c = i // (self.cols // 32) * 32, i % (self.cols // 32) * 32
      vals = struct.unpack("<1024H", self.rt.chip.read(*bh.DRAM[i % 8], self.addr + i // 8 * TILE, TILE))
      for k, v in enumerate(vals): out[r + k // 32][c + k % 32] = from_bf16(v)
    return out

  def _binary(self, op, other):
    assert (self.rows, self.cols) == (other.rows, other.cols), "shapes must match"
    out, cores = Tensor(self.rt, self.rows, self.cols), self.rt.cores
    share = {xy: (i * self.tiles // len(cores), (i + 1) * self.tiles // len(cores)) for i, xy in enumerate(cores)}
    self.rt.launch(kernels.get(op) or kernels.setdefault(op, kernel(op)),
                   lambda x, y: (share[x, y][0], share[x, y][1] - share[x, y][0], self.addr, other.addr, out.addr))
    self.rt.wait()
    return out
  def __add__(self, other): return self._binary("add", other)
  def __sub__(self, other): return self._binary("sub", other)
  def __mul__(self, other): return self._binary("mul", other)

kernels = {}  # op -> compiled kernel image
