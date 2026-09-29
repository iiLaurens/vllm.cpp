ID: ISSUE-LOCAL-01M3QFZD0PTVP2CT9S2RYG8M2H
Title: A first-forward CUDA OOM reports one failing size and nothing else: there is no per-allocation trace of where device memory went
Row: BACKEND-CUDA-SM120
State: OPEN
Kind: feature
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-29
Updated: 2026-09-29
Closed: -

## Problem

On a 24 GiB consumer-Blackwell card (sm_120a), the failure mode that matters is a first-forward cudaMalloc that does not fit, and today it produces exactly one line: `vt cuda: cudaMalloc: out of memory` naming the FAILING SIZE and nothing about the process. `VT_CUDA_ALLOC_STATS` aggregates counters, so it cannot say which allocation sequence produced the pressure; `nvidia-smi` sees the process total, not the call. The result is that an OOM is diagnosed by bisecting the forward (the Qwen3.8-27B NVFP4 arm loads ~20 GiB of packed weights into the same pool the KV cache and the repack scratch come from, so the interesting question is the ORDER and the live-bytes curve, not the total). `src/vt/cuda/cuda_backend.cu` owns Alloc/Free and `DeviceMemoryInfo`, so the instrument belongs there: `VT_CUDA_ALLOC_TRACE=1` prints every cudaMalloc of at least 16 MiB with its size, running live bytes and free device memory, every free of the same size, and the request that fails. The 16 MiB threshold is the signal-to-noise line (the pool hands out MiB-scale blocks; per-4 KiB lines would be a log, not an instrument) and a free-memory trigger below 2 GiB logs everything so the endgame is never missed. Zero-cost when unset: one process-static env read and one size compare on a path that is about to call the driver.

## Resolution

-
