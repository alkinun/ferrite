# Lessons

What building ferrite has taught us: the surprises, the mistakes, and the rules we now work by. NOTES.md holds the
facts and where each comes from; this file holds what they mean for the next piece of work, and for a chip of our own.

Add a lesson when something cost real time or changed how we work. Keep it to a few sentences and name the evidence: a
test, a measurement, a NOTES section. When a lesson turns out wrong or incomplete, rewrite it in place rather than
append a correction.

## The chip

1. **Nothing is coherent unless you make it so.** A core's L0 data cache never sees NoC writes or other cores' stores,
   so every poll loop needs a `fence`. A core's stores may not have reached L1 when the NIU reads it, so data a core
   wrote is settled (`fence`, a load, `fence`) before it is sent. New code needs a write to `RISCV_IC_INVALIDATE`;
   `fence.i` does nothing. (NOTES: "What the cores run out of reset", "Device API")
2. **Small fields set the real limits.** The unpackers see only 8 bits of their Z counter, so only L1 slots below 256
   can be unpacked; a semaphore counts to 15, so a kernel holds 15 intermediates; a thread replays from 32 entries.
   Each showed up as a bug or a ceiling long after the design leaned on a wider assumption. Find the width of every
   counter before designing around it.
3. **Reset values are undocumented, so write every register you rely on.** Thread config, address modes and
   semaphores start in unknown states on silicon. ttsim starts them at 0, which hides a missing write.
   (NOTES: "Configuration")
4. **The NoC wants alignment.** A transfer's source and destination must agree in their low 4 bits, or 6 when DRAM is
   involved. Moving data at odd offsets means fetching the aligned block and copying inside L1 (`gather`).
5. **Collective operations pay for themselves.** Launching worker by worker cost 45 cycles a worker. One multicast for
   the arguments, one for the go and one atomic counter for the reports made a launch cost the same on 1 worker or 119.
   (NOTES: "Launch cost")
6. **A multicast is acknowledged by every recipient.** The request counter goes up once and down once per recipient,
   so completion is counted on the acknowledgement counter. A tile in the rectangle that is not a worker would answer
   too. (NOTES: "Multicast from a core")
7. **One feeder per Tensix thread.** BRISC and a TRISC pushing to the same thread in the same cycle drops the TRISC's
   word. With a TRISC per thread, a stalled feeder holds up only its own thread, and push order stopped mattering.
   (NOTES: "No push order to get right")
8. **The tile is ours, not the hardware's.** The ISA has no tile. The four-face layout is what `MVMUL` wants, and it
   shapes everything else: a transpose is a face swap in the fetch plus a per-face transpose in the unpacker, and a
   column of a tile is one lane of eight SFPU loads. (NOTES: "Tile layout")
9. **Precision lives in the details.** A bf16 product is exact only after all four fidelity phases. `ELWADD` aligns
   mantissas at 11 bits with round half up, the packer rounds half up, uploads round to nearest even. Bit-exact tests
   (small integers, a diagonal of full mantissas, A @ I) pin the layout and the phases where a tolerance would pass a
   bug. (NOTES: "Precision")
10. **A full-precision datum has one way in.** SrcA and SrcB hold 19 bits, so fp32 cannot go through the FPU at all:
    the unpacker must write Dst directly and the SFPU must do the math. And the FPU's own sums are not fp32 either: it
    adds each group of 8 products at 12 bits, so a matmul is good to 2^-12 whatever Dst holds. (NOTES: "fp32 tensors",
    "Matmul precision")
11. **Moving data across SFPU lanes is scarce.** `SFPSHFT2` rotates within groups of 8 lanes and `SFPTRANSP` moves
    groups between four registers; nothing else crosses lanes. Folding a tile's rows or columns costs 120 to 168
    instructions because of it. (NOTES: "Reductions")

## The simulator

1. **ttsim checks rules, not time.** It kills the process on anything the hardware forbids: a bad address, a
   misaligned transfer, a semaphore taken while empty. That makes it the best test we have. pytest hides its `ERROR:`
   line; run with `-s` to see it.
2. **Its cycles compare our versions, not silicon.** NoC requests and DRAM answer at once and every Tensix instruction
   runs atomically. So latency hiding (a deeper prefetch) looks slower on ttsim, and some costs are ttsim's own: a
   kernel whose root is a binary op costs 110 cycles a tile more. Record such results in NOTES for the card to settle.
   (NOTES: "ttsim's cycles are its model")
3. **What it does not model, no test can catch.** Instruction cache invalidation, PC snapshots, `mhartid`, the busy
   conditions of `STALLWAIT`, NoC request splitting. The waits only silicon exercises are step 8 of the bring-up plan.
   (NOTES: "What ttsim can and cannot check here")
4. **Break things on purpose.** Removing each wait in turn and running the tests showed which ones ttsim enforces and
   which it lets through. That list is worth more than a passing run.
5. **When the docs and ttsim disagree, ask which chip each describes.** Some Blackhole ISA pages are Wormhole's copied
   over (address modes of 2 bits, not Blackhole's 3). ttsim's `tensix_isa.json` and `tensix_regs.json` are
   Blackhole's. Silicon still has the last word. (NOTES: "Where the docs and ttsim disagree")

## The compiler and runtime

1. **Bugs hide in what the tests never vary.** K wider than the result was found by an MNIST network; two long matmuls
   past slot 256 by checking GPT-2 kernel by kernel; reductions of different widths in one kernel, and a third
   accumulator, by fuzzing. Every earlier test had used the convenient shape. Real models and random expressions find
   what hand-written tests do not.
2. **Check each kernel against numpy on its own inputs.** End to end, bf16 rounding piles up and can hide a bug or pass
   for one. Reading back each kernel's inputs and checking that kernel alone does neither (`test_fuzz`).
3. **Make fused equal unfused, bit for bit.** Every intermediate is rounded to bf16 on its way to L1 exactly as it
   would be on its way to DRAM, so fusion is tested by equality, not tolerance (`test_fusion`).
4. **One padding rule beats special cases.** Padding is zero, except that a dimension of 1 fills its tile. That one
   rule gives broadcasting by tile reuse, lets a matmul sum over a padded K, and every kernel keeps it. Mask with
   stores, not multiplies: inf times 0 is NaN. (NOTES: "Any shape")
5. **Launch cost dominates small work.** A GPT-2 token went from 2.07M simulated cycles to 140k mostly by launching
   less and more cheaply: resident kernels, a ring the dispatcher reads by itself, multicast, split K, fewer kernels.
   Faster math was a small part of it.
6. **Generate straight-line code.** Kernel setup as stores of constants, overlapped across the threads, took a
   one-tile kernel from 790 cycles to 490 against loops over tables in L1. The compiler is Python; the device code can
   be as specialised as we like.
7. **Do layout work once.** Reading `wte` transposed every token cost 39k cycles, against 17.6k for a copy transposed
   once at load.
8. **Python's lifetimes are part of the memory manager.** A tensor frees its DRAM in `__del__`, which CPython runs at
   once unless a reference cycle holds the tensor; a closure in `realize` once did. (NOTES: "DRAM")
9. **Cache what the compiler makes.** Compiled device images kept on disk took the test suite from 24 s to 5 s.
10. **Numbers are tensors, so a constant has its tensor's dtype.** In bf16, `x * (1/768)` multiplies by 1/768 rounded to
    8 bits, 0.2% off, and exp(10) as tinygrad writes it, `exp2(10 * log2(e))`, would be 2% off; so exp, log and cos keep
    their constants in fp32 inside their SFPU programs. In fp32 tensors the constants are exact. (NOTES: "The SFPU")
11. **A generalization must carry every special case along.** Making reductions work along any axis dropped one line
    that seeded the sweep's length from the kernel's own result, and a reduction of another length then crashed the
    compiler. No test reached it until the fuzzer learned `where`, one step later. Read the old code's guards before
    deleting them, and give each one a test (`test_reduce_axes` now has it).
12. **A fuzzer is as good as its generator, and as its oracle.** Growing the generator to new ops found that
    regression. Its oracle, numpy without bf16 rounding, cannot judge a comparison of two computed values (a rounding
    step flips it), so comparisons in random trees only see exact inputs.
13. **Test at the precision you claim.** Every result went home as bf16, so tests compared at 2^-8 and two coarser
    things hid under it: the FPU's 12-bit sums and exp's 4e-4. fp32 results showed both on the first run.
14. **Find where the time goes before cutting.** In the move kernel the divisions looked expensive and removing them
    saved 10%. Skipping the datums outside the bounds and keeping the arguments in registers saved most of the rest.
    The profile of the test said the time was all ttsim running kernels, not Python. (NOTES: "Movement")

## Working method

1. **Give every goal a finish line a test can show,** and tick it only when the test passes (ROADMAP).
2. **Measure before and after, and write both down.** Every optimization in NOTES carries its numbers. One that did
   not pay on ttsim is recorded as tried, with the reason, so silicon can settle it.
3. **Delete before adding.** The simplest correct version first, and the faster one when a measurement says so.
4. **Keep the documents true.** A stale "next" list or a renamed function in old prose costs the next reader more than
   a missing paragraph. Fix the docs in the same commit as the code.

## For a chip of our own

What Blackhole made hard to program, so that a design for simple software can design it away (ROADMAP, stage 6).

- **Coherence, or no caches at all.** Three kinds of manual synchronization (the L0 data cache, settling L1, the
  instruction cache) are the commonest source of subtle bugs here.
- **One kind of configuration,** readable back, with documented reset values. Blackhole has backend config (stores),
  thread config (in-band `SETC16` only) and address counters, each written its own way.
- **Counters wide enough never to be the limit.** An 8-bit Z, 4-bit semaphores and a 32-entry replay buffer each
  shaped the compiler.
- **Full precision through the main datapath.** fp32 here cannot touch the matrix unit's inputs, and its dot products
  align at 12 bits; a second, slower path (unpack into Dst, the SFPU) is the only way to fp32.
- **Collectives in the NoC.** Multicast with a single completion, and atomic counters, are what made launches cheap.
- **A simulator that models time as well as rules,** built alongside the chip, so that performance work does not wait
  for silicon.
