#!/usr/bin/env bash
# Run from any directory. A fresh output directory is required; never overwrite
# a previous validation. Second arg: docker (default) or local (existing env).
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
validation_id="${1:-smoke-$(date -u +%Y%m%dT%H%M%SZ)}"
runtime="${2:-docker}"
if [[ "$runtime" != docker && "$runtime" != local ]]; then
  echo "Runtime must be docker or local." >&2
  exit 2
fi
if [[ ! "$validation_id" =~ ^[a-zA-Z0-9_-]+$ ]]; then
  echo "Validation id must contain only letters, digits, _ or -." >&2
  exit 2
fi
out="results/$validation_id"
submission="submissions/$validation_id"
if [[ -e "$out" || -e "$submission" ]]; then
  echo "Output already exists: $out or $submission. Choose a new id." >&2
  exit 2
fi
mkdir -p "$out"
test -f .env || cp .env.example .env
export UID
export GID="$(id -g)"
if [[ "$runtime" == docker ]]; then
  runner=(docker compose run --rm -T -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e CUDA_VISIBLE_DEVICES= gfootball python)
  docker image inspect t-zero-gfootball:latest --format '{{.Id}}' | tee "$out/image-id.txt"
  docker compose run --rm -T gfootball git -c safe.directory=/opt/football -C /opt/football rev-parse HEAD > "$out/football-commit.txt"
else
  runner=(env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= "${FOOTBALL_PYTHON:-python3}")
  printf '%s\n' 'local environment; not an official-image certification' > "$out/image-id.txt"
  if [[ -d "${GFOOTBALL_SOURCE:-/opt/football}/.git" ]]; then
    git -c "safe.directory=${GFOOTBALL_SOURCE:-/opt/football}" -C "${GFOOTBALL_SOURCE:-/opt/football}" rev-parse HEAD > "$out/football-commit.txt"
  else
    printf '%s\n' 'unknown; record the engine source separately' > "$out/football-commit.txt"
  fi
fi
printf '%s\n' "$runtime" > "$out/runtime-mode.txt"

"${runner[@]}" -c \
  'import platform, torch, gfootball, gymnasium, gfootball_engine; print(platform.platform()); print("torch", torch.__version__); print("engine", gfootball_engine.__file__); assert not torch.cuda.is_available()' \
  2>&1 | tee "$out/engine.log"
"${runner[@]}" -m pip freeze > "$out/packages.txt"
"${runner[@]}" -m pytest tests/test_ppo.py tests/test_football_project.py tests/test_football_machine.py \
  tests/test_configs_load.py tests/test_config_roundtrip.py tests/test_wrappers.py \
  tests/test_checkpoint.py tests/test_agent_network.py tests/test_config_reference.py \
  tests/test_gfootball.py tests/test_arena.py -q -rs \
  2>&1 | tee "$out/tests.log"
"${runner[@]}" scripts/football_train.py --config ppo_football_sparse \
  --override total_timesteps=4096 num_envs=2 ppo.num_steps=128 checkpoint_every=512 \
  2>&1 | tee "$out/train.log"
run_dir="$(sed -n 's/^PROJECT_RUN_DIR=//p' "$out/train.log" | tr -d '\r' | tail -n 1)"
if [[ -z "$run_dir" || ! -f "$run_dir/completed.json" ]]; then
  echo "Training did not produce a completed run." >&2
  exit 1
fi
printf '%s\n' "$run_dir" > "$out/run-dir.txt"
"${runner[@]}" scripts/export_submission.py "$run_dir" \
  --out "$submission" --team "$validation_id" 2>&1 | tee "$out/export.log"
# Full matches, unchanged official checker, no --max-steps or relaxed limits.
"${runner[@]}" -m arena.check "$submission" --json "$out/check.json" \
  2>&1 | tee "$out/check.log"
"${runner[@]}" scripts/football_evaluate.py "$submission" --matches 2 \
  --seed-start 900000 --out "$out/evaluation" 2>&1 | tee "$out/evaluate.log"
printf '\nValidation passed. Evidence: %s; agent: %s\n' "$out" "$submission"
