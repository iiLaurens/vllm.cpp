# CUDA per-allocation trace — ISSUE-LOCAL-01M3QFZD0PTVP2CT9S2RYG8M2H

A first-forward `cudaMalloc` that does not fit produces one line naming the
failing size. On a 24 GiB `sm_120a` card serving a ~20 GiB NVFP4 arm out of the
same pool as the KV cache and the repack scratch, the interesting question is
which allocation sequence produced the pressure — and nothing in the tree can
answer it.

Issue: [ISSUE-LOCAL-01M3QFZD0PTVP2CT9S2RYG8M2H](../issues/BACKEND-CUDA-SM120/ISSUE-LOCAL-01M3QFZD0PTVP2CT9S2RYG8M2H.md).
Owning row: `BACKEND-CUDA-SM120` ([backend-matrix.md](../backend-matrix.md)), the
consumer-Blackwell row, because that is the card this instrument exists for.

## Premise, grounded

| Where (line anchors at this branch's base, `b45a94273`) | What |
|---|---|
| `src/vt/cuda/cuda_backend.cu:55-59` | `Check` — the only report an OOM gets: the failing call's name and the driver string. |
| `src/vt/cuda/cuda_backend.cu:120-140` | `Alloc`/`Free`; `StatsEnabled` (`VT_CUDA_ALLOC_STATS`) aggregates counters, so it cannot attribute pressure to a sequence. |
| `DeviceMemoryInfo` | The `cudaMemGetInfo` wrapper the trace reports through, already used by the stats arm. |

`VT_CUDA_ALLOC_STATS` is aggregate by design and stays as it is; this is a second,
opt-in instrument, not a replacement.

## Design

`VT_CUDA_ALLOC_TRACE=1`, read once per process:

- Every `Alloc` of at least 16 MiB prints
  `[cuda-alloc] #N size=<MiB> live=<GiB> free=<GiB>`. The 16 MiB floor is the
  signal-to-noise line: the pool hands out MiB-scale blocks, and a per-allocation
  line for every 4 KiB scratch would be a log rather than an instrument.
- Below 2 GiB free, every allocation prints regardless of size — the endgame is
  exactly where a size filter would hide the last few blocks.
- Every `Free` of a tracked block of at least 16 MiB prints
  `[cuda-free] size=… live=… free=…`, so the live curve is readable in both
  directions. The map is keyed by pointer; a free of a block the trace never
  tracked is ignored.
- A failed `cudaMalloc` prints `[cuda-alloc] FAILED size=… free=… err=…` before
  `Check` throws, so the failure is attributed to the live total at that moment.
- Unset: one process-static env read and one size comparison per `Alloc`.

## Tests

`tests/vt/test_cuda_alloc_trace.cpp` (new target, registered beside
`test_cuda_backend`): the binary enables the flag in a global initializer (the
flag is process-static, so it cannot be toggled in-process) and captures stderr
around each backend call.

- A 32 MiB allocation prints a `[cuda-alloc]` line with `size=32.0 MiB`, and the
  matching `Free` prints a `[cuda-free]` line with the same size.
- An impossible request (1 PiB) throws AND prints the `FAILED` line with `err=`,
  so the next OOM names its own size and the free memory at that instant.
- The capture asserts its own sentinel, the same discipline the MoE tap's capture
  uses: a capture that silently caught nothing must fail rather than read as "the
  trace printed nothing".
- On a build without CUDA the case says so and exits; the flag-off path is the
  same code with `AllocTraceEnabled()` false and is covered by every CUDA test in
  the tree, none of which sets the variable.

What the gate does not prove: that a real OOM on a real model is easier to read —
that is a usability claim, not a testable one. It proves the three lines exist,
carry the right size, and reach stderr.

## Gates

- `ctest --test-dir build -R test_cuda_alloc_trace` on a CUDA build
  (`-DVLLM_CPP_CUDA_ARCHITECTURES=120a` on the local card).
- `test_cuda_backend` unchanged.
- `scripts/agent-preflight.sh --staged`.

## Owed

- The measurement this instrument exists for: the NVFP4 arm's live-bytes curve on
  the 24 GiB `sm_120a` card, which is a row deliverable and not this PR's.
