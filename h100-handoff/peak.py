"""Run xfuser.runner unchanged and print each rank's peak CUDA memory when it ends."""

import os
import runpy
import sys

import torch

sys.argv = ["xfuser.runner", *sys.argv[1:]]
try:
    runpy.run_module("xfuser.runner", run_name="__main__", alter_sys=True)
finally:
    if torch.cuda.is_available():
        rank = os.environ.get("RANK", "0")
        peak = torch.cuda.max_memory_allocated() / 2**30
        print(f"[peak] rank {rank} max_memory_allocated {peak:.3f} GiB", flush=True)
