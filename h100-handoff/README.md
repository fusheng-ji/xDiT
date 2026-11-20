# H100 handoff

Experiments for three open PRs that B200 cannot run: they need FlashAttention 3
(Hopper only) or the quantizing attention backends Sage and FA3 FP8. This branch is
#856's local head (main merged in, Qwen-Image on main's per-row key packing) plus this
directory. Nothing here is meant to be merged.

| PR | Branch the scripts fetch | What it checks |
| --- | --- | --- |
| #856 batched-prompt key padding | `h100-handoff` (this branch) | FA3 and FA2 serve prompts of different lengths through packed keys, against stock diffusers, on 1 and 2 GPUs, against main |
| #900 Ideogram 4 SP padding | `h100-handoff-900` | padded Ulysses 2 / ring 2 under SAGE, FLASH_3_FP8, FLASH_3, against one GPU, main, and the PR without the pad-row zeroing |
| #901 HunyuanVideo SP padding | `h100-handoff-901` | the same for HunyuanVideo |

The scripts fetch the other branches and upstream `main` themselves and check each one
out as a detached worktree beside this clone (`../h100-trees`); results go to
`../h100-results`.

## Setup

```bash
git clone -b h100-handoff https://github.com/fusheng-ji/xDiT.git xDiT-h100
cd xDiT-h100
pip install -e .   # or point PY at an environment that already has xDiT's dependencies
```

Needed in that environment:

- 2 H100 GPUs (more are not used).
- torch with CUDA, diffusers 0.40 (the versions the B200 runs used: torch 2.12.0, CUDA 13.0, diffusers 0.40.0, Python 3.12).
- `flash_attn_interface` (FlashAttention 3, built for Hopper) and `flash_attn` (FA2).
- `sageattention`.
- Hugging Face access to, or a local cache of: `Tongyi-MAI/Z-Image`, `Qwen/Qwen-Image`,
  `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v`, `ideogram-ai/ideogram-4-fp8`,
  and `tencent/HunyuanVideo` (what the runner's `HunyuanVideo` model name loads).

`env.txt` in each result directory records what was found; a missing backend shows up
there as `MISSING`, and the runs that need it fail with the reason in their log.

Variables (all optional): `PY` (python to use, default `python`), `OUT`, `TREES`,
`HF_HUB_CACHE` / `HF_HUB_OFFLINE` for the model cache.

## Run

```bash
bash h100-handoff/run_856.sh          # about 2-4 h
bash h100-handoff/run_padding.sh 900  # about 1-2 h
bash h100-handoff/run_padding.sh 901  # about 1 h
bash h100-handoff/run_timing.sh       # #856 latency and peak memory, about 2-3 h
```

Each run can be repeated; the two-GPU runs use ports from 29700 up.

## What to send back

The result directories, without the large outputs:

```bash
cd ../h100-results
tar czf h100-results.tgz --exclude='*.npy' --exclude='*.mp4' --exclude='*.pt' .
```

That keeps `env.txt`, `heads.txt`, every log, `summary.md` / `summary.txt` and the PNG
images. Send the PNGs of #900 too: the PSNR is checked against them by eye.

## Reading the results

- `856/summary.md`: per backend, model and GPU count, the largest relative error of the
  wrapper against stock diffusers on the same inputs, and each sample's PSNR. `pr856`
  should be close to bf16 noise; `main` attends to the padding, so its error should be
  larger on the sample with the shorter prompt. Main may also stop with an error on a
  model it does not batch; that is a result too.
- `856/pytest-1gpu.txt`, `856/pytest-multi.txt`: accelerator tests on the PR head.
- `856-timing/summary.md`: median latency of three runs after a warm-up for each batch,
  backend (FA3, FA2) and GPU count, as main / PR / main, with the PR's change against the
  mean of the two main runs and the spread between those two (the noise floor), plus each
  side's peak `max_memory_allocated`. Run this with the GPUs otherwise idle.
- `900/summary.txt`, `901/summary.txt`: PSNR of each two-GPU run against the one-GPU run
  with the same backend. `pr` should be at least as close as `main`; `pr` against
  `nozero` shows what zeroing the pad rows changes under a quantizing backend, where a
  shared scale is computed over every row. On B200 with bf16 attention the two were
  bit-identical. Main fails on HunyuanVideo at this size (the shards differ in length),
  which is the bug #901 fixes.
