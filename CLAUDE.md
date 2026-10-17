# trickle

trickle is Tenstorrent's software done right: the whole stack for Blackhole, from PCIe up to a compiler, small enough to read. Frameworks (tinygrad, PyTorch) plug in on top through thin adapters; they are not part of trickle.

## Design

From George Hotz's "Advice to Tenstorrent" (github.com/geohot/tt-tiny): the stack is "3 (and only 2 you have to build)".

1. Compiler: a compiler built for Tensix's tiles and dataflow, emitting Tensix instructions directly from the ISA docs. No LLK.
2. Runtime and driver: talk to the chip over PCIe from userspace. A runtime that is just a runtime: memory, DMA, loading, launching, queues.

Nothing from Tenstorrent runs at runtime except the firmware on the card: no tt-kmd, tt-umd, tt-metal or LLK.

## Taste

- Small. Every line earns its place. Delete before you add.
- Flat. Few files, no class hierarchies, no config systems, no abstraction until a second user needs it.
- Readable. The code is the documentation; comments say why, not what.
- Python with ctypes. No dependencies beyond the standard library, except for tests.
- Blackhole only, with its chip-specific facts kept as data in one place (`trickle/bh.py`).
- Every wait has a timeout: a hung chip is reported, never waited on forever.
- Every step has a test that runs on the simulator.

## Layout

- `trickle/bh.py`: Blackhole facts as data. NOTES.md says where each one comes from.
- `trickle/pci.py`: the device interface (config space, BARs, DMA memory, `tick`) with two backends: `Sim` (libttsim.so) and `Card` (sysfs).
- `trickle/chip.py`: NoC reads and writes (unicast and multicast) through a shared TLB window, dedicated TLBs (`map`), waiting with a timeout, ARC messages and telemetry, core reset.
- `trickle/runtime.py`: compiling device C with clang + lld (with the device NoC API prepended), worker and dispatcher firmware, and the command ring in host memory: a launch is one PCIe write, completion is read from host memory.
- `trickle/compiler.py`: Tensix instruction assembler (encodings and config fields as data in `bh.py`), the per-tile unpack/FPU/pack program, generated BRISC kernels, and `Tensor`: bf16 tensors in DRAM with `+`, `-`, `*` on every worker.
- `trickle/hello.py`: fib(n) on every Tensix core, the one-command demo.
- `test/test_sim.py`: every step, on ttsim.
- `NOTES.md`: what we learned about the chip, and what is still unknown.

## Running

Needs clang and lld on PATH, and ttsim's `libttsim.so` built for Blackhole (outside this repo: `./make.py src/_out/release_bh/libttsim.so` in tenstorrent/ttsim).

    TTSIM=/path/to/libttsim.so python3 -m trickle.hello
    TTSIM=/path/to/libttsim.so python3 -m pytest test

Without `TTSIM`, `pci.open_device()` looks for a real Blackhole on the PCI bus.

The simulator is a process-wide singleton that kills the process on any contract violation (bad address, bad TLB config): a test run that stops early with an `ERROR:` line from ttsim means the code did something the hardware would not like.
