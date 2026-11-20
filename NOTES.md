# H100 run notes

Environment
- GPUs: 2 x NVIDIA H100 (94 GB) used per run, on a 4-GPU node. Driver 580.173.02, CUDA 13.0 runtime (torch cu130); CUDA 13.4 nvcc used for the source builds.
- Python 3.12.9, torch 2.12.0+cu130, diffusers 0.40.0.
- flash_attn 2.8.4 (FA2), built from source for sm90.
- flash_attn_interface (FA3, flash_attn_3 3.0.1), built from the hopper directory of upstream flash-attention main. To shorten the compile, these features were disabled in the build: backward, fp16, softcap, paged KV, append KV, and the differing-head-dim kernels. bf16 and FP8 forward, head dims 64/96/128/192/256, split-KV, local and varlen are built. A bf16 and an FP8 call were checked against SDPA before the runs.
- sageattention 2.2.0, built from source (thu-ml/SageAttention main).
- torchao is not installed (optional extra, not needed by these runs). No library from the README list is MISSING.
- All models were loaded offline from a local Hugging Face cache; tencent/HunyuanVideo was fetched at revision refs/pr/18, the diffusers-format revision the runner requests.

Run history
- A first full run was stopped partway and discarded: another process on the same node held about 70 GB on the first GPU, so Qwen-Image and HunyuanVideo-1.5 runs failed with CUDA out of memory. Everything was rerun from scratch on two idle GPUs. The results here are from that second run only.
- No repository code, test or parameter was changed.

Failed steps (kept as results; logs have the details)
- #900, all six SAGE runs (1 GPU, Ulysses 2 on pr/nozero/main, ring 2 on pr/main): sageattention 2.2.0 raises "Unsupported head_dim: 256" for Ideogram 4. So #900 has no Sage numbers; `summary.txt` shows IndexError lines for them because there are no output images.
- #900, FA3 and FA3 FP8: all runs finished.
- #901, `main` Ulysses 2 and ring 2 fail for SAGE, FLASH_3_FP8 and FLASH_3 with a reshape RuntimeError on HunyuanVideo (the shards differ in length), the bug #901 addresses. `pr` and `nozero` runs all finished.
- #856 e2e: all 24 runs on pr856 and main finished (exit 0).
- #856 pytest-1gpu: 2 failed, 99 passed, 140 skipped. Failures: test_a_sharded_encoder_still_records_hidden_states (AssertionError, hidden states not recorded) and test_text_cross_attention_on_cudnn_attends_only_to_valid_text (AttributeError: runtime state has no current_step).
- #856 pytest-multi: 1 failed, 43 passed, 37 skipped. Failure: test_qwen_batched_prompts_ulysses.py::test_a_padded_prompt_batch_matches_one_call_per_prompt[2] (the two spawned ranks exited with code 1).

Skipped steps
- None skipped. 37 and 140 tests are skipped by pytest itself.

Sanitization
- *.npy, *.mp4, *.pt, *.safetensors and the small timings.json files were removed. Paths, host names, user name, IP addresses and GPU UUIDs are replaced by placeholders such as <repo>, <trees>, <venv>, <work>, <host>, <ip>.
