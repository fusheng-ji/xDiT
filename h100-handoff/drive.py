"""Run xfuser.runner unchanged, logging the wrapper's SP pad and saving video frames losslessly.

PAD_LOG_MODULE names the transformer wrapper module whose chunk_and_pad_sequence is wrapped.
NO_PAD_ZERO=1 replaces that module's zero_sp_pad_rows_hook with a no-op, to measure what
zeroing the pad rows changes.
"""

import importlib
import os
import runpy
import sys

import numpy as np

mod = importlib.import_module(os.environ["PAD_LOG_MODULE"])
_chunk = getattr(mod, "chunk_and_pad_sequence", None)  # absent where the wrapper does not pad
_seen = set()


def chunk_and_pad_sequence(x, sp_world_rank, sp_world_size, pad_amount, dim):
    key = (tuple(x.shape), sp_world_size, pad_amount, dim)
    if sp_world_rank == sp_world_size - 1 and key not in _seen:
        _seen.add(key)
        print(f"[pad] sp {sp_world_size} shape {tuple(x.shape)} dim {dim} pad {pad_amount}", flush=True)
    return _chunk(x, sp_world_rank, sp_world_size, pad_amount, dim)


if _chunk is not None:
    mod.chunk_and_pad_sequence = chunk_and_pad_sequence

if os.environ.get("NO_PAD_ZERO") == "1":
    if not hasattr(mod, "zero_sp_pad_rows_hook"):
        sys.exit(f"{mod.__name__} has no zero_sp_pad_rows_hook to disable")

    def zero_sp_pad_rows_hook(*args, **kwargs):
        return lambda module, inputs, output: None

    mod.zero_sp_pad_rows_hook = zero_sp_pad_rows_hook
    print("[pad] zero_sp_pad_rows_hook disabled", flush=True)

import xfuser.model_executor.models.runner_models.base_model as base_model  # noqa: E402

_export = base_model.export_to_video


def export_to_video(video, path, fps):
    np.save(path[:-4] + ".npy", np.stack([np.asarray(frame) for frame in video]))
    return _export(video, path, fps=fps)


base_model.export_to_video = export_to_video

sys.argv = ["xfuser.runner", *sys.argv[1:]]
runpy.run_module("xfuser.runner", run_name="__main__", alter_sys=True)
