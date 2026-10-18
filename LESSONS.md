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
11. **The same bits mean different things to different instructions.** An int32 is a valid fp32 bit pattern, and
    the small positive ones are denormals, which the FP32 store mode flushes to zero; only the INT32 mode stores bits
    as they are. Every instruction that touches a datum has a dtype in mind; find out which. (NOTES: "int32 tensors")
12. **Moving data across SFPU lanes is scarce.** `SFPSHFT2` rotates within groups of 8 lanes and `SFPTRANSP` moves
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
6. **Know how a number was measured.** Every wait on ttsim ran it 1000 clocks at a time, so each was rounded up to the
   next 1000; a profile that waited after every launch put a training step at 766k cycles when it was 444k, and a
   launch at 700 cycles when it was 390. Polling every 10 clocks costs a tenth more wall time. (NOTES: "How ttsim
   counts")
7. **Read the simulator's clock before optimizing for it.** ttsim runs queued Tensix instructions at no cost and
   counts only the clocks where a tile's pipes all wait, one RISC-V instruction each. The MOP expander, planned as the
   next step, only saves pushes, which it does not count; a PC profile of the idle clocks showed the cores waiting on
   each other instead, and fetches on BRISC's critical path: fetching a layer norm's inputs once took it from 2.6k
   cycles to 0.6k. (NOTES: "How ttsim counts")

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
    things hid under it: the FPU's 12-bit sums and exp's 4e-4 (exp is fp32-grade now). fp32 results showed both.
14. **Model the device program to tell its bugs from its algorithm's.** sin was 3e-7 off where its polynomial was
    1.7e-7. A numpy model of the same fp32 steps matched the device to 6e-8, so the code was right and the method was
    not: folding near a half-turn carried 0.5's rounding into a small result. Reducing by half-turns, in x, fixed it.
15. **Reduce exactly, then approximate.** Each fp32 transcendental is an exact reduction (Cody-Waite constants short
    enough that k times them is exact, exponent bits, a fold) and a short polynomial. The earlier exp, a series raised
    to the 4096th power, could never beat its 12 roundings. (NOTES: "The SFPU")
16. **Differentiate the operation, not its implementation.** An fp32 matmul is three bf16 matmuls, and autograd
    through those parts rounded every gradient to bf16, 3e-3 off where the forward was 2e-4. Recorded as one matmul,
    its backward is two more fp32 matmuls. The same holds for any op built from cheaper parts.
17. **Find where the time goes before cutting.** In the move kernel the divisions looked expensive and removing them
    saved 10%. Skipping the datums outside the bounds and keeping the arguments in registers saved most of the rest.
    In a Llama decode step, one kernel cost 3.6 times all the rest: single-row moves making 32 copies of the row
    one by one. A profile by kernel finds such things in a minute. (NOTES: "Movement")
18. **Check the reference against its source.** The numpy SmolLM was 0.012 off Hugging Face's until the comparison
    used fp32 rotary tables, as Hugging Face's fp32 model does; with them, 2e-5. A reference is only as good as its own
    check. (NOTES: "A Llama-family model")
19. **Copies do not compose; maps do.** Every move was a launch of its own, which was fine while moves came one at a
    time. The first convolution chained three a tap and took 34 launches for one layer. Composing the maps before
    anything runs, as tinygrad composes views, took it to 16 launches and from 505k cycles to 118k. Keep movement a
    description until something has to read it. (NOTES: "Movement at any granularity", "A small CNN")
20. **The backward pass is a different program.** It transposes what the forward never did (every matmul's gradient
    takes A^T and B^T), pads what the forward sliced and broadcasts what it summed. The fast path built for inference,
    a matmul reading a bf16 B transposed, had no fp32 twin, so fp32 transposes moved a datum at a time were 2M of a
    training step's 3M cycles, until the matmul split its operands into bf16 before transposing them. Profile training
    as a workload of its own. (NOTES: "Training")
21. **Check a run step by step on its own state.** Training beside PyTorch looked broken: gradients 4% apart by the
    second step, and once a loss of 7.3 against 2.9. Every step was right. Adam's first step had split the runs on
    gradients smaller than the matmul's error, and they went on as two models. The reference has to be evaluated at
    the system's own state (PyTorch on ferrite's weights, each step); long runs can only be compared as curves.
    (NOTES: "Training")
22. **A fallback hides what it replaces.** The tinygrad adapter's first version passed every comparison with tinygrad
    and launched nothing: data copied onto FERRITE stayed on the host, so every op took the fallback to tinygrad's
    CPU. Then a tinygrad script ran on tinygrad's own default device, because the device was chosen in the environment
    after tinygrad had read it. Count launches in a test, and make a fallback show itself. (NOTES: "tinygrad on
    ferrite")
23. **The primitives map; the layouts are the work.** tinygrad's ops are ferrite's almost one for one, and matched on the
    first try. A transformer step's time went into where data lies: views that did not fold into three axes went
    through the host, short axes padded every tile, flat weights and transposes were moved where they could be read in
    place, and a product was made whole before its sum. 52 s a step went to 25 by layout alone. (NOTES: "tinygrad on
    ferrite")
24. **Let the framework lower itself.** PyTorch has two thousand aten ops; its own decompositions bring them down to the
    sixty the adapter writes out, and a training run of its transformer layer reached the CPU for one op, attention's
    backward. A decomposition is code with its own purpose, though: the one for attention was written for export and
    returns the attention weights where the kernel returns their logsumexp. (NOTES: "PyTorch on ferrite")
25. **Copy what is already together.** The move kernel found each datum on its own, though most moves are whole tiles (a
    pad on tile boundaries), tiles transposed (an fp32 `.T`) or runs along a row or down a column. Copying those as they
    are took a training step's moves from 450k cycles to 28k, and tinygrad's transformer step from 6.1M to 2.9M. Which
    way to sweep mattered as much: attention's heads run 6 datums along a row and 32 down a column. (NOTES: "Movement")
26. **Work out a cost's floor before trying to lower it.** The benchmark's matmul read its inputs 6.9 times over, and
    bigger blocks looked like the fix; but 96 workers making 16 result tiles each from 4 rows and 4 columns is already
    the fewest reads a worker can make alone, so only sharing between workers could go lower. Multicast took it to 1.0,
    and the grid it needed put idle workers to use as well (a 4-row matmul went from 6 workers to 48). (NOTES: "Tiles
    shared between workers")

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
