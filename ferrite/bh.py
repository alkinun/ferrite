# Blackhole facts, as data: addresses, sizes and limits, instruction and config encodings (an instruction's modifier
# values stay where the compiler uses them). NOTES.md says where each number comes from.

VENDOR, DEVICE = 0x1E52, 0xB140

# BAR0 (512 MiB): 202 TLB windows of 2 MiB from offset 0, then registers. BAR4 (32 GiB): 8 TLB windows of 4 GiB.
TLB_WINDOW, TLB_COUNT = 1 << 21, 201  # TLB 201, the last 2 MiB one, is tt-kmd's; we leave it alone
TLB_CONFIG = 0x1FC00000  # BAR0 offset of the TLB config registers: 3 dwords per TLB, the 2 MiB ones first
# 2 MiB TLB config, a 96-bit little-endian value: field -> (first bit, width). The rest of the fields are in NOTES.md.
TLB_FIELDS = {"addr": (0, 43), "x_end": (43, 6), "y_end": (49, 6), "x_start": (55, 6), "y_start": (61, 6),
              "mcast": (69, 1), "ordering": (70, 2), "static_vc": (73, 1), "vc_class": (76, 2)}
STRICT = 1  # ordering: each write is acknowledged before the next one starts
MCAST_VC = 2  # multicast on a static VC must use class 2 (umd does, ISA NoC/MemoryMap.md NOC_CTRL)

# NoC endpoints in translated coordinates, which firmware sets up at boot so that harvesting is invisible:
# Tensix columns with the good ones first (a p150 has 12, telemetry says how many), rows 2..11.
TENSIX_X, TENSIX_Y = (1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16), range(2, 12)
DRAM = [(17 + b // 4, 12 + 3 * (b % 4)) for b in range(8)]  # bank b; each bank also answers at y+1 and y+2
DRAM_BANK_SIZE = 0xFF000000  # what ttsim models (4080 MiB); a p150 has 4 GiB a bank, whether firmware keeps any is unknown
PCIE = (19, 24)
ARC = (8, 0)  # rows 0 and 1 are not translated

# ARC, the management core, through its reset unit's scratch registers
ARC_BOOT_STATUS = 0x80030408  # SCRATCH_RAM[2]: low 3 bits are 5 once firmware is up and taking messages
ARC_QUEUES = 0x8003042C  # SCRATCH_RAM[11]: address of the message queue control block {base, entries | count << 8}
ARC_IRQ = 0x80030100, 1 << 16  # ARC_MISC_CNTL IRQ0: tells ARC to look at its queues
ARC_TELEMETRY = 0x80030434, 0x80030430  # SCRATCH_RAM[13], [12]: addresses of {version, count, tag | index << 16}, values
APP_QUEUE = 3  # queues 0..3 are for kmd, monitoring, tools, applications
ARC_QUEUE_ENTRY = 32  # bytes: a queue is a header, then n requests and n responses, each an entry
ARC_QUEUE_PTRS = (0x0, 0x4, 0x10, 0x14)  # in its header: our request write and response read pointers, then ARC's request
                                         # read and response write pointers, each counting to 2n so full and empty differ
# POWER_SETTING with all 15 flags valid and set: max AI clock, GDDR PHY awake, Tensix and L2CPU clocks on
POWER_ON = 0x21 | 15 << 8 | 0x7FFF << 16
TAG_AICLK, TAG_GDDR_STATUS, TAG_TENSIX_COLS, TAG_GDDR_ENABLED = 14, 22, 34, 36  # telemetry; GDDR_STATUS: 2 bits a bank

# Tensix tile, at the same addresses over the NoC and from its own cores
L1_SIZE = 1536 * 1024
AICLK = 1.35e9  # Hz, the cores' and Tensix's clock (ISA BabyRISCV/README.md); telemetry TAG_AICLK has the live value
MVMUL_FLOPS = 2 * 8 * 16 * 16  # an MVMUL a cycle, one fidelity phase: Dst[8x16] += SrcB[8x16] @ SrcA[16x16]
SOFT_RESET = 0xFFB121B0  # RISCV_DEBUG_REG_SOFT_RESET_0: a set bit holds that core in reset
WALL_CLOCK = 0xFFB121F0  # RISCV_DEBUG_REG_WALL_CLOCK_L: the tile's clock cycles, the low 32 bits (ISA DebugTimestamper.md)
PC_SNAPSHOT = 0xFFB13138  # each core's pc, a word each: BRISC, NCRISC, TRISC0-2 (ISA BabyRISCV/README.md; not in ttsim)
BRISC, TRISC0, TRISC1, TRISC2, NCRISC = 1 << 11, 1 << 12, 1 << 13, 1 << 14, 1 << 18
ALL_CORES = BRISC | TRISC0 | TRISC1 | TRISC2 | NCRISC  # power-on value
BRISC_STACK_TOP = 0xFFB00000 + 8 * 1024  # end of BRISC's private data RAM; BRISC starts at PC 0 in L1
TRISC_STACK_TOP = 0xFFB00000 + 4 * 1024  # each TRISC's is 4 KiB, at the same address
TRISC_RESET_PC = 0xFFB12228  # RISCV_DEBUG_REG_TRISC0_RESET_PC, then TRISC1's and TRISC2's
TRISC_RESET_PC_OVERRIDE = 0xFFB12234  # bit i: TRISC i starts at its RESET_PC rather than 0x6000 / 0xA000 / 0xE000
NIU = 0xFFB20000  # NoC 0 interface (ISA NoC/MemoryMap.md): request initiator i's registers at + i * INITIATOR, counters at +0x200
NIU_REGS = {  # offsets from NIU. A request: target and return addresses (64 bits each, then the tile's x | y << 6), a tag,
  # control bits (NOC_CMD), a length, and CMD = 1 to send it; AT_DATA is an atomic's operand
  "TARG_LO": 0x0, "TARG_MID": 0x4, "TARG_HI": 0x8, "RET_LO": 0xC, "RET_MID": 0x10, "RET_HI": 0x14, "PACKET_TAG": 0x18, "CTRL": 0x1C,
  "LEN": 0x20, "AT_DATA": 0x28, "CMD": 0x40, "CLEAR_OUTSTANDING": 0x60, "ID_LOGICAL": 0x148,  # this tile's translated x | y << 6
  "INITIATOR": 0x800,  # four request initiators, each keeping its registers between requests
  "WR_ACKS": 0x204, "READS": 0x208,  # write acknowledgements and read responses had
  "OUTSTANDING": 0x240}  # requests in flight with transaction id 0 (the packet tag's bits 10-13), then a word for each of ids 1-15
NOC_CMD = {"READ": 0x10, "WRITE": 0x12,  # both acknowledged, so OUTSTANDING counts them
           "MCAST": 0x12 | 1 << 5 | 1 << 7 | 1 << 8 | MCAST_VC << 14,  # a write to a rectangle: broadcast, path reserve, static VC
           "ATOMIC": 1, "INCREMENT": 1 << 12 | 31 << 2,  # NOC_AT_LEN_BE: add AT_DATA to all 32 bits of word 0
           "MAX_PACKET": 16384}
NIU_READS = NIU + NIU_REGS["READS"]  # NIU_MST_RD_RESP_RECEIVED: read requests answered, one a request (bench's DRAM reads)
TENSIX_CFG = 0xFFEF0000  # Tensix backend config bank 0, one word per register; BRISC and TRISCs only, not the NoC
CFG_IC_INVALIDATE = 185  # RISCV_IC_INVALIDATE: writing bit 0 (BRISC) .. bit 4 (NCRISC) drops that core's instruction cache

# Tensix coprocessor. Encodings: ISA TensixCoprocessor/ and Diagrams/Src/Bits32.lua, field names as in ttsim's
# data/bh/tensix_isa.json. A word is opcode << 24 | fields. TRISC t feeds thread t: it stores the word at TENSIX_PUSH,
# or executes it as an instruction rotated left by 2 bits (.ttinsn; opcodes are below 0xC0, so the low bits are never
# 0b11 and it cannot be mistaken for RISC-V). A full FIFO stalls the TRISC either way. (BRISC can push to every
# thread, at TENSIX_PUSH + 0x10000*t; we leave that to the TRISCs.) TRISCs also see their thread through these:
TENSIX_PUSH = 0xFFE40000
TENSIX_DONE = 0xFFE80004  # CoprocessorDoneCheck: a load answers once every instruction this thread was given has run
TENSIX_SEM = 0xFFE80020  # semaphore i at +4i: a load reads its value, storing 0 posts it and storing 1 takes one
_ELW = {"dst": (0, 10), "addr_mode": (14, 3), "instr_mod19": (19, 2), "dest_accum_en": (21, 1), "clear_dvalid": (22, 2)}
_ADC = {"bit_mask": (0, 4), "ch0_x": (6, 3), "ch0_y": (9, 3), "ch1_x": (12, 3), "ch1_y": (15, 3), "cnt_set_mask": (21, 3)}
TENSIX_OPS = {  # name: (opcode, {field: (first bit, width)})
  "REPLAY": (0x04, {"load": (0, 1), "exec": (1, 1), "count": (4, 6), "index": (14, 5)}),  # a thread's 32-entry buffer
  "ZEROACC": (0x10, {"where": (0, 10), "addr_mode": (14, 3), "use_32_bit_mode": (18, 1), "clear_mode": (19, 2)}),
  "MVMUL": (0x26, {"dst": (0, 10), "addr_mode": (14, 3), "instr_mod19": (19, 1), "clear_dvalid": (22, 2)}),
  "ELWMUL": (0x27, _ELW), "ELWADD": (0x28, _ELW), "ELWSUB": (0x30, _ELW),
  "SETRWC": (0x37, {"bit_mask": (0, 4), "rwc_a": (6, 4), "rwc_b": (10, 4), "rwc_d": (14, 4), "rwc_cr": (18, 4), "clear_ab_vld": (22, 2)}),
  "PACR": (0x41, {"last": (0, 1), "read_intf_sel": (8, 4), "addr_mode": (15, 2)}),
  "UNPACR": (0x42, {"last": (0, 1), "set_dat_valid": (6, 1), "ovrd_thread_id": (7, 1), "addr_mode": (15, 8), "unpack_block_selection": (23, 1)}),
  "SETADC": (0x50, {"value": (0, 18), "dimension_index": (18, 2), "channel_index": (20, 1), "cnt_set_mask": (21, 3)}),
  "SETADCXY": (0x51, _ADC), "SETADCZW": (0x54, _ADC),  # in ZW, x means z and y means w
  "SETADCXX": (0x5E, {"x_start": (0, 10), "x_end2": (10, 10), "cnt_set_mask": (21, 3)}),
  "STALLWAIT": (0xA2, {"wait_res": (0, 13), "stall_res": (15, 9)}),
  "SEMINIT": (0xA3, {"sem_sel": (2, 8), "init_value": (16, 4), "max_value": (20, 4)}),
  "SEMPOST": (0xA4, {"sem_sel": (2, 8)}), "SEMGET": (0xA5, {"sem_sel": (2, 8)}),
  "SEMWAIT": (0xA6, {"wait_sem_cond": (0, 2), "sem_sel": (2, 8), "stall_res": (15, 9)}),  # cond 1: while 0, 2: while at max
  "SETC16": (0xB2, {"setc16_value": (0, 16), "setc16_reg": (16, 8)}),
  **{f"RMWCIB{i}": (0xB3 + i, {"cfg_reg_addr": (0, 8), "data": (8, 8), "mask": (16, 8)}) for i in range(4)},  # byte i of a config word
  # SFPU, the 32-lane vector unit working on Dst (ISA VectorUnit.md; field names as on the ISA pages). vd/vc/vb/va are
  # LRegs: 0-7 are writable, 8 = 0.8373, 9 = 0.0, 10 = 1.0, 11 = -1.0, 15 = 2 * lane (writing 12-15 does something else).
  "SFPLOAD": (0x70, {"imm10": (0, 10), "addr_mode": (13, 3), "mod0": (16, 4), "vd": (20, 4)}),
  "SFPLOADI": (0x71, {"imm16": (0, 16), "mod0": (16, 4), "vd": (20, 4)}),
  "SFPSTORE": (0x72, {"imm10": (0, 10), "addr_mode": (13, 3), "mod0": (16, 4), "vd": (20, 4)}),
  "SFPEXEXP": (0x77, {"mod1": (0, 4), "vd": (4, 4), "vc": (8, 4)}),  # mod1 0: the exponent, 1: biased, as an integer
  "SFPEXMAN": (0x78, {"mod1": (0, 4), "vd": (4, 4), "vc": (8, 4)}),  # mod1 0: the mantissa with its hidden bit, as an integer
  "SFPIADD": (0x79, {"mod1": (0, 4), "vd": (4, 4), "vc": (8, 4), "imm12": (12, 12)}),
  "SFPSHFT": (0x7A, {"mod1": (0, 4), "vd": (4, 4), "vc": (8, 4), "imm12": (12, 12)}),
  "SFPSETCC": (0x7B, {"mod1": (0, 4), "vc": (8, 4)}),  # lane flags from vc: mod1 0, where its sign is set
  "SFPMOV": (0x7C, {"mod1": (0, 4), "vd": (4, 4), "vc": (8, 4)}),
  "SFPABS": (0x7D, {"mod1": (0, 4), "vd": (4, 4), "vc": (8, 4)}),  # mod1 1: of a float
  "SFPAND": (0x7E, {"mod1": (0, 4), "vd": (4, 4), "vc": (8, 4)}),  # vd &= vc
  "SFPOR": (0x7F, {"mod1": (0, 4), "vd": (4, 4), "vc": (8, 4)}),  # vd |= vc
  "SFPNOT": (0x80, {"vd": (4, 4), "vc": (8, 4)}),  # vd = ~vc
  "SFPXOR": (0x8D, {"vd": (4, 4), "vc": (8, 4)}),  # vd ^= vc
  "SFPSETEXP": (0x82, {"mod1": (0, 4), "vd": (4, 4), "vc": (8, 4), "imm12": (12, 8)}),  # mod1 1: vc with exponent imm12
  "SFPMAD": (0x84, {"mod1": (0, 4), "vd": (4, 4), "vc": (8, 4), "vb": (12, 4), "va": (16, 4)}),  # vd = va * vb + vc
  "SFPMUL": (0x86, {"mod1": (0, 4), "vd": (4, 4), "vc": (8, 4), "vb": (12, 4), "va": (16, 4)}),  # vc must be LReg 9
  "SFPENCC": (0x8A, {"mod1": (0, 4), "vd": (4, 4), "imm2": (12, 2)}),
  "SFPSWAP": (0x92, {"mod1": (0, 4), "vd": (4, 4), "vc": (8, 4)}),
  "SFPSHFT2": (0x94, {"mod1": (0, 4), "vd": (4, 4), "vc": (8, 4)}),  # mod1 3: rotate each 8-lane group (a Dst row) by one
  "SFPTRANSP": (0x8C, {}),  # LRegs 0-3 (and 4-7) as a 4 x 4 of 8-lane groups, transposed: group j of LReg i swaps with group i of LReg j
  "SFPCAST": (0x90, {"mod1": (0, 4), "vd": (4, 4), "vc": (8, 4)}),  # mod1 0: sign-magnitude int to float, 3: two's complement <-> sign-magnitude
  "SFPGT": (0x97, {"mod1": (0, 4), "vd": (4, 4), "vc": (8, 4)}),  # mod1 8: vd = all ones if vd > vc (as ordered floats), else 0
  "SFPARECIP": (0x99, {"mod1": (0, 4), "vd": (4, 4), "vc": (8, 4), "vb": (12, 4)}),  # Blackhole only
}
UNPACK0, UNPACK1, PACK = 1, 2, 4  # cnt_set_mask: which address counters an ADC instruction sets
# STALLWAIT/SEMWAIT stall_res blocks these units' instructions until (STALLWAIT wait_res) this thread's unpacker 0,
# unpackers, packer, FPU or SFPU, or the config unit, are idle (ISA STALLWAIT.md: B and C bits)
STALL_SYNC, STALL_PACK, STALL_UNPACK, STALL_CFG, STALL_SFPU, STALL_ALL = 1 << 1, 1 << 2, 1 << 3, 1 << 7, 1 << 8, 0x1FF
WAIT_UNPACK0, WAIT_UNPACK, WAIT_PACK, WAIT_FPU, WAIT_SFPU, WAIT_CFG = 1 << 1, 0b110, 1 << 3, 1 << 4, 1 << 11, 1 << 12
FP32, BF16, BFP8, BFP4, INT32 = 0, 5, 6, 7, 8  # data format codes (ISA Unpackers/FormatConversion.md)

# Backend config: field -> (32-bit word index at TENSIX_CFG, first bit, width). The ISA docs defer these numbers to
# tt-metal's cfg_defines.h; these are from ttsim's data/bh/tensix_regs.json and still need checking against silicon.
TENSIX_CFG_FIELDS = {
  "Dstacc": (1, 25, 4), "Fp32_enabled": (1, 29, 1), "Zero_Flag_disabled_src": (2, 0, 1),
  "PCK0_Ystride": (12, 16, 16), "PCK0_ZW": (13, 0, 32), "PCK0_Ystride_1": (14, 16, 16), "PCK0_ZW_1": (15, 0, 32), "PCK0_Base_1": (17, 0, 18),
  "Read_32b_data": (18, 0, 1), "Edge_mask": (24, 0, 16), "pack_reads_per_xy_plane": (28, 8, 8),
  "UNP0_add_dest_addr_cntr": (50, 8, 1), "UNP0_Ystride_1": (56, 16, 16), "UNP0_ZW_1": (57, 0, 32), "UNP1_ZW_1": (59, 0, 32),
  "SEC0_format": (64, 0, 4), "SEC0_x_dim": (64, 16, 16), "SEC0_y_dim": (65, 0, 8), "SEC0_z_dim": (65, 16, 8), "Exp_section_size": (68, 16, 16),
  "L1_Dest_addr": (69, 0, 32), "Disable_zero_compress": (70, 0, 1), "Pack_out_format": (70, 4, 4), "Pack_in_format": (70, 8, 4),
  "SEC0_out_format": (72, 0, 4), "SEC0_Haloize_mode": (72, 8, 1),
  "SEC0_uncompressed": (73, 0, 1), "SEC0_base": (76, 0, 32), "SEC0_dest_addr": (84, 0, 16),
  "SEC0_tile_x_dim": (86, 0, 16), "SEC0_offset": (92, 0, 16),
  # unpacker 0's second context, which a thread picks with UNPACK_MISC (thread config): unpacking into Dst
  "SEC0_uncompressed_1": (73, 1, 1), "SEC0_to_dst_1": (73, 5, 1), "SEC0_base_1": (77, 0, 32), "SEC0_dest_addr_1": (84, 16, 16),
  "SEC0_tile_x_dim_1": (86, 16, 16), "SEC0_offset_1": (93, 0, 16),
  "SEC1_format": (112, 0, 4), "SEC1_x_dim": (112, 16, 16), "SEC1_y_dim": (113, 0, 8), "SEC1_z_dim": (113, 16, 8),
  "SEC1_out_format": (120, 0, 4), "SEC1_uncompressed": (121, 0, 1), "SEC1_base": (124, 0, 32), "PCK_dest_offset": (180, 0, 12),
}
# Thread config: 16-bit registers per thread, written only by SETC16 from that thread. Register indices; the address
# modes are 8 slots each, slot i at base + i. Blackhole's addr_mode field is 3 bits wide (ttsim's tensix_isa.json; the
# ISA pages are Wormhole's, with 2 bits and a separate ADDR_MOD_SET_Base that Blackhole dropped), so all 8 are usable.
THREAD_CFG = {"StateID": 0, "MATH_Offset": 1, "DISABLE_IMPLIED_SRCA": 2, "DISABLE_IMPLIED_SRCB": 3, "SRCA_SET": 5,
              "CLR_DVALID": 7, "FIDELITY_BASE": 11, "UNPACK_MISC": 41}
SRCA_SET_BY_ADDR = 1 << 2  # SrcA/SrcB rows come from the unpacker's address, which our UNPACRs rely on
UNPACKABLE = 256  # an unpacker sees only the low 8 bits of its Z counter (ISA ADCs.md): the tiles it reads are below this
BFP_FIRST_DATUM = 1 << 16  # it finds a bfp tile's exponents by its first datum, which must stay below this
LREG_CONSTS = {0.0: 9, 1.0: 10, -1.0: 11}  # SFPU registers that hold these constants (ISA LReg.md; 8 holds 0.8373)
# Address modes, 8 slots each: AB: SrcAIncr | SrcBIncr << 8, DST: DestIncr | FidelityIncr << FIDELITY_INCR, PACK:
# YsrcIncr, BIAS: the packer's. Row counters wrap at 64 (Src) and 1024 (Dst)
ADDR_MOD_AB, ADDR_MOD_DST, ADDR_MOD_PACK, ADDR_MOD_BIAS, FIDELITY_INCR = 12, 28, 37, 47, 13
