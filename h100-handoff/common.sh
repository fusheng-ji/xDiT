# Sourced by the run_*.sh scripts. Override any of these in the environment.
set -u
HANDOFF=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HANDOFF/.." && pwd)
PY=${PY:-python}
OUT=${OUT:-$REPO/../h100-results}
TREES=${TREES:-$REPO/../h100-trees}
UPSTREAM=${UPSTREAM:-https://github.com/xdit-project/xDiT.git}
FORK=${FORK:-https://github.com/fusheng-ji/xDiT.git}
mkdir -p "$OUT" "$TREES"

# tree <name> <url> <ref>: a detached worktree of <ref> under $TREES/<name>.
tree() {
  local name=$1 url=$2 ref=$3
  git -C "$REPO" fetch -q "$url" "+$ref:refs/handoff/$name" || return 1
  if [ -d "$TREES/$name" ]; then
    git -C "$TREES/$name" checkout -q --detach "refs/handoff/$name"
  else
    git -C "$REPO" worktree add -q --detach "$TREES/$name" "refs/handoff/$name"
  fi
  echo "$name $(git -C "$TREES/$name" rev-parse --short HEAD)"
}

env_report() {
  "$PY" - <<'EOF'
import importlib.util, platform
import torch, diffusers
print("python", platform.python_version())
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("diffusers", diffusers.__version__)
print("devices", torch.cuda.device_count(), sorted({torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())}))
for name in ("flash_attn", "flash_attn_interface", "sageattention"):
    print(name, "yes" if importlib.util.find_spec(name) else "MISSING")
EOF
}
