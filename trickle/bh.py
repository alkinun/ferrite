# Blackhole facts, as data. Everything chip-specific lives here; NOTES.md says where each number comes from.

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
PCIE = (19, 24)
ARC = (8, 0)  # rows 0 and 1 are not translated

# ARC, the management core, through its reset unit's scratch registers
ARC_BOOT_STATUS = 0x80030408  # SCRATCH_RAM[2]: low 3 bits are 5 once firmware is up and taking messages
ARC_QUEUES = 0x8003042C  # SCRATCH_RAM[11]: address of the message queue control block {base, entries | count << 8}
ARC_IRQ = 0x80030100, 1 << 16  # ARC_MISC_CNTL IRQ0: tells ARC to look at its queues
ARC_TELEMETRY = 0x80030434, 0x80030430  # SCRATCH_RAM[13], [12]: addresses of {version, count, tag | index << 16}, values
APP_QUEUE = 3  # queues 0..3 are for kmd, monitoring, tools, applications
# POWER_SETTING with all 15 flags valid and set: max AI clock, GDDR PHY awake, Tensix and L2CPU clocks on
POWER_ON = 0x21 | 15 << 8 | 0x7FFF << 16
TAG_AICLK, TAG_GDDR_STATUS, TAG_TENSIX_COLS, TAG_GDDR_ENABLED = 14, 22, 34, 36  # telemetry; GDDR_STATUS: 2 bits a bank

# Tensix tile, at the same addresses over the NoC and from its own cores
L1_SIZE = 1536 * 1024
SOFT_RESET = 0xFFB121B0  # RISCV_DEBUG_REG_SOFT_RESET_0: a set bit holds that core in reset
BRISC, TRISC0, TRISC1, TRISC2, NCRISC = 1 << 11, 1 << 12, 1 << 13, 1 << 14, 1 << 18
ALL_CORES = BRISC | TRISC0 | TRISC1 | TRISC2 | NCRISC  # power-on value
BRISC_STACK_TOP = 0xFFB00000 + 8 * 1024  # end of BRISC's private data RAM; BRISC starts at PC 0 in L1
NIU = 0xFFB20000  # NoC 0 interface: request initiators at +0x800*i, counters at +0x200, translated id at +0x148
TENSIX_CFG = 0xFFEF0000  # Tensix backend config bank 0, one word per register; BRISC and TRISCs only, not the NoC
CFG_IC_INVALIDATE = 185  # RISCV_IC_INVALIDATE: writing bit 0 (BRISC) .. bit 4 (NCRISC) drops that core's instruction cache

# Tensix coprocessor. Encodings: ISA TensixCoprocessor/ and Diagrams/Src/Bits32.lua, field names as in ttsim's
# data/bh/tensix_isa.json. A word is opcode << 24 | fields. BRISC pushes thread t's instructions by storing the word at
# TENSIX_PUSH + 0x10000*t (they skip the MOP expander), and sees thread 0's 64 GPRs at TENSIX_GPR.
TENSIX_PUSH, TENSIX_GPR = 0xFFE40000, 0xFFE00000
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
  "SETGPR": (0x45, {"reg_index_16b": (0, 7), "value": (8, 16)}),  # SETDMAREG: 16 bits into half a GPR (index = 2*gpr + hi)
  "SETADC": (0x50, {"value": (0, 18), "dimension_index": (18, 2), "channel_index": (20, 1), "cnt_set_mask": (21, 3)}),
  "SETADCXY": (0x51, _ADC), "SETADCZW": (0x54, _ADC),  # in ZW, x means z and y means w
  "SETADCXX": (0x5E, {"x_start": (0, 10), "x_end2": (10, 10), "cnt_set_mask": (21, 3)}),
  "STALLWAIT": (0xA2, {"wait_res": (0, 13), "stall_res": (15, 9)}),
  "SEMINIT": (0xA3, {"sem_sel": (2, 8), "init_value": (16, 4), "max_value": (20, 4)}),
  "SEMPOST": (0xA4, {"sem_sel": (2, 8)}), "SEMGET": (0xA5, {"sem_sel": (2, 8)}),
  "SEMWAIT": (0xA6, {"wait_sem_cond": (0, 2), "sem_sel": (2, 8), "stall_res": (15, 9)}),  # cond 1: while 0, 2: while at max
  "SETC16": (0xB2, {"setc16_value": (0, 16), "setc16_reg": (16, 8)}),
  # SFPU, the 32-lane vector unit working on Dst (ISA VectorUnit.md; field names as on the ISA pages). vd/vc/vb/va are
  # LRegs: 0-7 are writable, 8 = 0.8373, 9 = 0.0, 10 = 1.0, 11 = -1.0 (writing 12-15 does something else entirely).
  "SFPLOAD": (0x70, {"imm10": (0, 10), "addr_mode": (13, 3), "mod0": (16, 4), "vd": (20, 4)}),
  "SFPLOADI": (0x71, {"imm16": (0, 16), "mod0": (16, 4), "vd": (20, 4)}),
  "SFPSTORE": (0x72, {"imm10": (0, 10), "addr_mode": (13, 3), "mod0": (16, 4), "vd": (20, 4)}),
  "SFPEXEXP": (0x77, {"mod1": (0, 4), "vd": (4, 4), "vc": (8, 4)}),
  "SFPIADD": (0x79, {"mod1": (0, 4), "vd": (4, 4), "vc": (8, 4), "imm12": (12, 12)}),
  "SFPSHFT": (0x7A, {"mod1": (0, 4), "vd": (4, 4), "vc": (8, 4), "imm12": (12, 12)}),
  "SFPMOV": (0x7C, {"mod1": (0, 4), "vd": (4, 4), "vc": (8, 4)}),
  "SFPMAD": (0x84, {"mod1": (0, 4), "vd": (4, 4), "vc": (8, 4), "vb": (12, 4), "va": (16, 4)}),  # vd = va * vb + vc
  "SFPMUL": (0x86, {"mod1": (0, 4), "vd": (4, 4), "vc": (8, 4), "vb": (12, 4), "va": (16, 4)}),  # vc must be LReg 9
  "SFPENCC": (0x8A, {"mod1": (0, 4), "vd": (4, 4), "imm2": (12, 2)}),
  "SFPSWAP": (0x92, {"mod1": (0, 4), "vd": (4, 4), "vc": (8, 4)}),
  "SFPSHFT2": (0x94, {"mod1": (0, 4), "vd": (4, 4), "vc": (8, 4)}),  # mod1 3: rotate each 8-lane group (a Dst row) by one
  "SFPARECIP": (0x99, {"mod1": (0, 4), "vd": (4, 4), "vc": (8, 4), "vb": (12, 4)}),  # Blackhole only
}
UNPACK0, UNPACK1, PACK = 1, 2, 4  # cnt_set_mask: which address counters an ADC instruction sets
STALL_SYNC, STALL_PACK, STALL_THCON, STALL_SFPU, STALL_ALL = 1 << 1, 1 << 2, 1 << 5, 1 << 8, 0x1FF  # STALLWAIT/SEMWAIT
WAIT_UNPACK, WAIT_PACK, WAIT_FPU, WAIT_SFPU = 0b110, 1 << 3, 1 << 4, 1 << 11  # stall_res: blocks those units until
# (wait_res) the unpackers/packer/FPU/SFPU of this thread are idle
BF16 = 5  # data format code (ISA Unpackers/FormatConversion.md)

# Backend config: field -> (32-bit word index at TENSIX_CFG, first bit, width). The ISA docs defer these numbers to
# tt-metal's cfg_defines.h; these are from ttsim's data/bh/tensix_regs.json and still need checking against silicon.
TENSIX_CFG_FIELDS = {
  "Dstacc": (1, 25, 4), "Fp32_enabled": (1, 29, 1), "Zero_Flag_disabled_src": (2, 0, 1),
  "PCK0_Ystride": (12, 16, 16), "PCK0_ZW": (13, 0, 32), "PCK0_Ystride_1": (14, 16, 16), "PCK0_ZW_1": (15, 0, 32), "PCK0_Base_1": (17, 0, 18),
  "Read_32b_data": (18, 0, 1), "Edge_mask": (24, 0, 16), "pack_reads_per_xy_plane": (28, 8, 8),
  "UNP0_add_dest_addr_cntr": (50, 8, 1), "UNP0_Ystride_1": (56, 16, 16), "UNP0_ZW_1": (57, 0, 32), "UNP1_ZW_1": (59, 0, 32),
  "SEC0_format": (64, 0, 4), "SEC0_x_dim": (64, 16, 16), "SEC0_y_dim": (65, 0, 8), "SEC0_z_dim": (65, 16, 8),
  "L1_Dest_addr": (69, 0, 32), "Disable_zero_compress": (70, 0, 1), "Pack_out_format": (70, 4, 4), "Pack_in_format": (70, 8, 4),
  "SEC0_out_format": (72, 0, 4), "SEC0_uncompressed": (73, 0, 1), "SEC0_base": (76, 0, 32), "SEC0_dest_addr": (84, 0, 16),
  "SEC0_tile_x_dim": (86, 0, 16), "SEC0_offset": (92, 0, 16),
  "SEC1_format": (112, 0, 4), "SEC1_x_dim": (112, 16, 16), "SEC1_y_dim": (113, 0, 8), "SEC1_z_dim": (113, 16, 8),
  "SEC1_out_format": (120, 0, 4), "SEC1_uncompressed": (121, 0, 1), "SEC1_base": (124, 0, 32), "PCK_dest_offset": (180, 0, 12),
}
# Thread config: 16-bit registers per thread, written only by SETC16 from that thread. Register indices; the address
# modes are 8 slots each, slot i at base + i. Blackhole's addr_mode field is 3 bits wide (ttsim's tensix_isa.json; the
# ISA pages are Wormhole's, with 2 bits and a separate ADDR_MOD_SET_Base that Blackhole dropped), so all 8 are usable.
THREAD_CFG = {"StateID": 0, "MATH_Offset": 1, "DISABLE_IMPLIED_SRCA": 2, "DISABLE_IMPLIED_SRCB": 3, "SRCA_SET": 5,
              "CLR_DVALID": 7, "FIDELITY_BASE": 11, "UNPACK_MISC": 41}
SRCA_SET_BY_ADDR = 1 << 2  # SrcA/SrcB rows come from the unpacker's address, which our UNPACRs rely on
ADDR_MOD_AB, ADDR_MOD_DST, ADDR_MOD_PACK, ADDR_MOD_BIAS = 12, 28, 37, 47  # AB: SrcAIncr | SrcBIncr << 8,
FIDELITY_INCR = 13  # DST: DestIncr | FidelityIncr << 13, PACK: YsrcIncr. Row counters wrap at 64 (Src) and 1024 (Dst).
