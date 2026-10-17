# Blackhole facts, as data. Everything chip-specific lives here; NOTES.md says where each number comes from.

VENDOR, DEVICE = 0x1E52, 0xB140

# BAR0 (512 MiB): 202 TLB windows of 2 MiB from offset 0, then registers. BAR4 (32 GiB): 8 TLB windows of 4 GiB.
TLB_WINDOW = 1 << 21
TLB_CONFIG = 0x1FC00000  # BAR0 offset of the TLB config registers: 3 dwords per TLB, the 2 MiB ones first
# 2 MiB TLB config, a 96-bit little-endian value: field -> (first bit, width). The rest of the fields are in NOTES.md.
TLB_FIELDS = {"addr": (0, 43), "x_end": (43, 6), "y_end": (49, 6), "ordering": (70, 2)}
STRICT = 1  # ordering: each write is acknowledged before the next one starts

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
TENSIX_CFG = 0xFFEF0000  # Tensix backend config, one word per register; BRISC and TRISCs only, not the NoC
CFG_IC_INVALIDATE = 185  # RISCV_IC_INVALIDATE: writing bit 0 (BRISC) .. bit 4 (NCRISC) drops that core's instruction cache
