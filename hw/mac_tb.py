# hw/mac.v against hw/mac.py, bit for bit, in cocotb on Verilator (test_mac_hardware builds it and runs this): the
# arithmetic (fma, read inside mac) on mac.cases with a million random ones (ROADMAP's exit criterion), then the
# accumulator on a stream of loads, steps and idle clocks.
import random
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, ReadOnly, RisingEdge
import mac

N = 1_000_000

@cocotb.test()
async def arithmetic(dut):  # each clock, acc holds a case's acc (loaded the clock before) while a and b are its factors
  cocotb.start_soon(Clock(dut.clk, 2, unit="ns").start())
  cases = list(mac.cases(N, random.Random(1).randrange))
  dut.en.value, dut.load.value, dut.init.value = 0, 1, cases[0][0]
  await RisingEdge(dut.clk)
  for k, (c, a, b) in enumerate(cases):
    dut.a.value, dut.b.value, dut.init.value = a, b, cases[k + 1][0] if k + 1 < len(cases) else 0
    await ReadOnly()
    got, want = dut.step.y.value.to_unsigned(), mac.mac(c, a, b)
    assert got == want, f"{c:#010x} + {a:#06x} * {b:#06x}: {got:#010x}, the model says {want:#010x}"
    await RisingEdge(dut.clk)

@cocotb.test()
async def accumulator(dut):  # a stream of clocks, each loading a value (one in ten), stepping (most of the rest) or idle
  cocotb.start_soon(Clock(dut.clk, 2, unit="ns").start())
  rand, acc = random.Random(2).randrange, None
  for c, a, b in mac.cases(N // 4, rand):
    load, en = acc is None or rand(10) == 0, rand(8) != 0
    await FallingEdge(dut.clk)
    dut.load.value, dut.en.value, dut.init.value, dut.a.value, dut.b.value = int(load), int(en), c, a, b
    await RisingEdge(dut.clk)
    await ReadOnly()
    acc = c if load else mac.mac(acc, a, b) if en else acc
    assert dut.acc.value.to_unsigned() == acc, f"acc {dut.acc.value.to_unsigned():#010x}, the model's {acc:#010x}"
