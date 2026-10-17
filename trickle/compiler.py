# The compiler: tensor ops emitted as Tensix instructions straight from the ISA, no LLK. A tensor lives in DRAM as
# bf16 32x32 tiles, tile i in bank i % 8. Each worker's BRISC runs a generated kernel: it pulls operand tiles into L1
# over the NoC and streams them through the three Tensix threads, unpack, math and pack running at the same time on
# different tiles, then writes results back.
#
# Tile layout: four 16x16 faces in row-major order (top-left, top-right, bottom-left, bottom-right), each face
# row-major. The ISA has no notion of a tile; this is the layout the FPU's 16x16 matrix multiply wants. An unpacker
# reads a whole tile into a SrcA/SrcB bank, so face f lands in rows 16f..16f+15, and Dst holds the result the same way.
import struct
from trickle import bh
from trickle.runtime import DATA, KERNEL, compile_rv32

TILE = 32 * 32 * 2  # bytes in a bf16 tile: one SrcA/SrcB bank (64 rows of 16), 64 Dst rows
SRCA, SRCB, OUT = DATA, DATA + 2 * TILE, DATA + 4 * TILE  # two L1 slots each: what unpacker 0 and 1 read, the result
UNPACKED, PACKED = 63, 62  # thread 0 GPRs BRISC polls: steps unpacked, output tiles packed
DST_HALF = 64  # Dst rows per output tile in fp32; tile n is computed in half n % 2 while the other one is packed
SEM_DST, SEM_PACKED = 1 << 0, 1 << 1  # semaphores: tiles in Dst waiting to be packed, packed tiles not yet reported

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

# Unpackers read one whole tile each (1024 datums) into a SrcA/SrcB bank; the Z address counter picks the L1 slot (the
# Z stride is one tile). The FPU writes Dst in fp32. The packer reads 4 Dst rows per PACR and rounds to bf16 on the way
# to L1; its channel 1 Y counter picks the output slot. Base addresses are in 16-byte units, minus 1.
CONFIG = config(
  Dstacc=bh.BF16, Fp32_enabled=1, Zero_Flag_disabled_src=0,
  PCK0_Ystride=32, PCK0_ZW=0, PCK0_Ystride_1=TILE, PCK0_ZW_1=0, PCK0_Base_1=0, Read_32b_data=1, Edge_mask=0xFFFF,
  pack_reads_per_xy_plane=16, UNP0_add_dest_addr_cntr=0, UNP0_Ystride_1=0, UNP0_ZW_1=0, UNP1_ZW_1=0,
  SEC0_format=bh.BF16, SEC0_x_dim=1024, SEC0_y_dim=1, SEC0_z_dim=2, SEC0_out_format=bh.BF16, SEC0_uncompressed=1,
  SEC0_base=SRCA // 16 - 1, SEC0_dest_addr=64, SEC0_tile_x_dim=1024, SEC0_offset=0,  # 64: SrcA rows are biased by 4
  SEC1_format=bh.BF16, SEC1_x_dim=1024, SEC1_y_dim=1, SEC1_z_dim=2, SEC1_out_format=bh.BF16, SEC1_uncompressed=1,
  SEC1_base=SRCB // 16 - 1, L1_Dest_addr=OUT // 16 - 1, Disable_zero_compress=1, Pack_out_format=bh.BF16,
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
#   math -> pack: Dst holds two output tiles. SEM_DST counts tiles waiting to be packed; pack waits while it is 0.
#     Math never needs to wait for a free half: BRISC does not push tile n + 2 until it has seen tile n packed.
#   pack -> BRISC: BRISC can only read thread 0's GPRs, so pack posts SEM_PACKED and thread 0 relays it into a GPR.
UNPACKERS, ALL_ADCS = bh.UNPACK0 | bh.UNPACK1, bh.UNPACK0 | bh.UNPACK1 | bh.PACK
SETUP_UNPACK = [ins("SEMINIT", sem_sel=SEM_DST, max_value=2), ins("SEMINIT", sem_sel=SEM_PACKED, max_value=15),
                ins("SETADCXY", cnt_set_mask=UNPACKERS, bit_mask=0b1111), ins("SETADCXX", cnt_set_mask=UNPACKERS, x_end2=1023)]
SETUP_PACK = [ins("SETADCXY", cnt_set_mask=bh.PACK, bit_mask=0b1111), ins("SETADCZW", cnt_set_mask=bh.PACK, bit_mask=0b1111),
              ins("SETADCXX", cnt_set_mask=bh.PACK, x_end2=15)]  # 16 datums a Dst row
def unpack(slot):  # the slot's two tiles into SrcA and SrcB, each bank then handed to the FPU
  return [ins("SETADCZW", cnt_set_mask=UNPACKERS, bit_mask=0b1111, ch0_x=slot),
          ins("UNPACR", last=1, set_dat_valid=1, ovrd_thread_id=1), ins("UNPACR", last=1, set_dat_valid=1, ovrd_thread_id=1, unpack_block_selection=1),
          ins("STALLWAIT", stall_res=bh.STALL_THCON, wait_res=bh.WAIT_UNPACK)]  # BRISC then gets SETGPR(UNPACKED)
def math_begin(half, zero):  # aim the FPU at a Dst half, and zero it if the math accumulates
  return ([ins("SETC16", setc16_reg=bh.THREAD_CFG["MATH_Offset"], setc16_value=DST_HALF * half)] +
          [ins("ZEROACC", clear_mode=1, use_32_bit_mode=1, where=4 * half + b) for b in range(4) if zero])  # 16-row blocks
MATH_END = [ins("STALLWAIT", stall_res=bh.STALL_SYNC, wait_res=bh.WAIT_FPU), ins("SEMPOST", sem_sel=SEM_DST)]
def pack(half, slot):  # Dst half into the output slot, then free the half and count the tile as packed
  return ([ins("SEMWAIT", stall_res=bh.STALL_ALL, sem_sel=SEM_DST, wait_sem_cond=1),
           ins("SETADC", cnt_set_mask=bh.PACK, dimension_index=1, value=DST_HALF * half),  # channel 0 Y: Dst row
           ins("SETADC", cnt_set_mask=bh.PACK, dimension_index=1, channel_index=1, value=slot),  # channel 1 Y: L1 slot
           ins("STALLWAIT", stall_res=bh.STALL_PACK, wait_res=bh.WAIT_FPU)] +
          [ins("PACR", last=int(i == 15)) for i in range(16)] +  # all four Dst read ports: 4 rows a PACR
          [ins("STALLWAIT", stall_res=bh.STALL_SYNC, wait_res=bh.WAIT_PACK), ins("SEMGET", sem_sel=SEM_DST), ins("SEMPOST", sem_sel=SEM_PACKED)])
RELAY = [ins("SEMWAIT", stall_res=bh.STALL_ALL, sem_sel=SEM_PACKED, wait_sem_cond=1), ins("SEMGET", sem_sel=SEM_PACKED)]

def elementwise(op):
  """Math thread setup and one step of C = A op B: 8 rows of 16 per instruction. ELWMUL always accumulates and
  multiplies a slice of the mantissas per fidelity phase: Dst is zeroed first, then all 4 phases run. Address mode 0
  steps 8 rows, mode 1 the fidelity phase."""
  fpu = {"add": "ELWADD", "sub": "ELWSUB", "mul": "ELWMUL"}[op]
  if op != "mul": step = [ins("SETRWC", bit_mask=0b1111)] + [ins(fpu, clear_dvalid=3 * (i == 7)) for i in range(8)]
  else: step = [ins("SETRWC", bit_mask=0b1111)] + [w for phase in range(4) for w in [ins("SETRWC", bit_mask=0b0111)] +
                [ins(fpu, addr_mode=int(i == 7), clear_dvalid=3 * (phase == 3 and i == 7)) for i in range(8)]]
  return thread_setup(ab=[(8, 8), (0, 0), (0, 0), (0, 0)], dst=[(8, 0), (0, 1), (0, 0), (0, 0)]), step

def matmul():
  """Math thread setup and one step of Dst += A @ B for a pair of tiles, A in SrcB and B in SrcA: MVMUL does
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
  return thread_setup(ab=[(16, 0), (48, 8), (16, 8), (16, 8)], dst=[(0, 0), (0, 0), (0, 0), (0, 1)]), step

def _array(name, words): return f"static const unsigned {name}[] = {{{', '.join(f'{w:#x}' for w in words)}}};"
def _arrays(name, programs): return f"static const unsigned {name}[{len(programs)}][{len(programs[0])}] = {{{', '.join('{' + ', '.join(f'{w:#x}' for w in p) + '}' for p in programs)}}};"

def stream_kernel(math, zero, fetch, pack_first=False):
  """A BRISC kernel that streams output tiles [args[0], args[0]+args[1]) through the three threads, args[5] steps
  (pairs of input tiles) each. fetch is C that loads step (t, k)'s tiles into SRCA/SRCB slot s. pack_first queues each
  tile's pack before its math, so the pack thread must wait; for tests on ttsim only, which runs every instruction the
  moment it can and has deep FIFOs. On silicon BRISC would stall on the pack thread's full FIFO and never push the math."""
  thread, step = math
  setgpr = ins("SETGPR", reg_index_16b=0)
  return compile_rv32(f"""
{_array("cfg_index", CONFIG)} {_array("cfg_value", CONFIG.values())} {_array("thread", thread)}
{_array("setup_unpack", SETUP_UNPACK)} {_array("setup_pack", SETUP_PACK)} {_array("step", step)} {_array("math_end", MATH_END)}
{_array("relay", RELAY)} {_arrays("unpack", [unpack(s) for s in (0, 1)])} {_arrays("pack", [pack(h, h) for h in (0, 1)])}
{_arrays("math_begin", [math_begin(h, zero) for h in (0, 1)])} {_array("bank", [x | y << 6 for x, y in bh.DRAM])}
#define PUSH(t) (*(volatile unsigned *)({bh.TENSIX_PUSH:#x} + 0x10000 * (t)))
#define GPR(r) (((volatile unsigned *){bh.TENSIX_GPR:#x})[r])
#define CFG ((volatile unsigned *){bh.TENSIX_CFG:#x})
#define push(t, p) for (unsigned i = 0; i < sizeof p / 4; i++) PUSH(t) = p[i]
#define setgpr(r, v) PUSH(0) = {setgpr:#x} | 2 * (r) | ((v) & 0xFFFF) << 8
#define reached(r, v) while ((GPR(r) & 0xFFFF) < (v)) {{}}  // 16-bit counts: a launch handles under 65536 steps
// tile t of a tensor at DRAM address base: bank t % 8, row t / 8 of the bank
static void get(unsigned base, unsigned t, unsigned l1) {{ noc_read(bank[t % 8] & 63, bank[t % 8] >> 6, base + t / 8 * {TILE}, l1, {TILE}); }}
static void put(unsigned l1, unsigned base, unsigned t) {{ noc_write(l1, bank[t % 8] & 63, bank[t % 8] >> 6, base + t / 8 * {TILE}, {TILE}); }}
static void writeback(volatile unsigned *a, unsigned n) {{  // wait for output tile n of this core to be packed, send it home
  push(0, relay); setgpr({PACKED}, n + 1);
  reached({PACKED}, n + 1);
  put({OUT:#x} + (n & 1) * {TILE}, a[4], a[0] + n); noc_wait();
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
  for (unsigned n = 0, s = 0; n < a[1]; n++) {{
    unsigned t = a[0] + n;
    {"push(2, pack[n & 1]);" if pack_first else ""}
    for (unsigned k = 0; k < a[5]; k++, s++) {{
      if (s >= 2) reached({UNPACKED}, s - 1);  // the slot's last tiles, two steps ago, have been unpacked
      {fetch}
      noc_wait();
      push(0, unpack[s & 1]); setgpr({UNPACKED}, s + 1);
      if (k == 0) push(1, math_begin[n & 1]);
      push(1, step);
      if (k == a[5] - 1) push(1, math_end);
    }}
    {"" if pack_first else "push(2, pack[n & 1]);"}
    if (n) writeback(a, n - 1);  // before pack[n + 1] can overwrite its slot
  }}
  if (a[1]) writeback(a, a[1] - 1);
}}
""", KERNEL)

def elementwise_kernel(op, pack_first=False):  # args: [first tile, tiles, A, B, C, 1]
  return stream_kernel(elementwise(op), op == "mul", f"get(a[2], t, {SRCA:#x} + (s & 1) * {TILE}); get(a[3], t, {SRCB:#x} + (s & 1) * {TILE});", pack_first)

def matmul_kernel():  # args: [first output tile, tiles, A, B, C, K tiles, N tiles]; A's tile goes to SrcB, B's to SrcA
  return stream_kernel(matmul(), True, f"get(a[2], t / a[6] * a[5] + k, {SRCB:#x} + (s & 1) * {TILE}); get(a[3], k * a[6] + t % a[6], {SRCA:#x} + (s & 1) * {TILE});")

def to_bf16(x):
  bits = struct.unpack("<I", struct.pack("<f", x))[0]
  return (bits + 0x7FFF + (bits >> 16 & 1)) >> 16  # round to nearest even
def from_bf16(b): return struct.unpack("<f", struct.pack("<I", b << 16))[0]
def tile_index(r, c): return (r // 16 * 2 + c // 16) * 256 + r % 16 * 16 + c % 16  # where (r, c) of a tile sits in it

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
      r0, c0, tile = i // (t.cols // 32) * 32, i % (t.cols // 32) * 32, [0] * 1024
      for r in range(32):
        for c in range(32): tile[tile_index(r, c)] = to_bf16(rows[r0 + r][c0 + c])
      t.rt.chip.write(*bh.DRAM[i % 8], t.addr + i // 8 * TILE, struct.pack("<1024H", *tile))
    return t

  def tolist(self):
    out = [[0.0] * self.cols for _ in range(self.rows)]
    for i in range(self.tiles):
      r0, c0 = i // (self.cols // 32) * 32, i % (self.cols // 32) * 32
      tile = struct.unpack("<1024H", self.rt.chip.read(*bh.DRAM[i % 8], self.addr + i // 8 * TILE, TILE))
      for r in range(32):
        for c in range(32): out[r0 + r][c0 + c] = from_bf16(tile[tile_index(r, c)])
    return out

  def _run(self, kernel, out, *args):  # out's tiles split evenly over the workers
    cores = self.rt.cores
    share = {xy: (i * out.tiles // len(cores), (i + 1) * out.tiles // len(cores)) for i, xy in enumerate(cores)}
    self.rt.launch(kernel, lambda x, y: (share[x, y][0], share[x, y][1] - share[x, y][0], *args))
    self.rt.wait()
    return out

  def _elementwise(self, op, other):
    assert (self.rows, self.cols) == (other.rows, other.cols), "shapes must match"
    out = Tensor(self.rt, self.rows, self.cols)
    return self._run(kernels.get(op) or kernels.setdefault(op, elementwise_kernel(op)), out, self.addr, other.addr, out.addr, 1)
  def __add__(self, other): return self._elementwise("add", other)
  def __sub__(self, other): return self._elementwise("sub", other)
  def __mul__(self, other): return self._elementwise("mul", other)

  def __matmul__(self, other):
    assert self.cols == other.rows, f"({self.rows}, {self.cols}) @ ({other.rows}, {other.cols})"
    out = Tensor(self.rt, self.rows, other.cols)
    kernel = kernels.get("matmul") or kernels.setdefault("matmul", matmul_kernel())
    return self._run(kernel, out, self.addr, other.addr, out.addr, self.cols // 32, other.cols // 32)

kernels = {}  # op -> compiled kernel image
