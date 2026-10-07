#!/bin/bash
# One training run on the GPUs listed in GPUS, then its test pass on the same GPUs.
#
#   GPUS=0,1 MODEL=/path/to/Ouro-1.4B bash train/run.sh sansi_s0 --loops 8 --seed 0
#   GPUS=0 MODEL=/path/to/SmolLM2-1.7B bash train/run.sh smollm2_s0 --loops 1 --seed 0
#   GPUS=0 MODEL=/path/to/Ouro-1.4B DATA=datasets/depth_liars STEPS=2000 EVAL_STEPS=500,1000,1500,2000 TEST_LOOPS=16 \
#     bash train/run.sh depth_liars_sansi_s0 --loops 8 --seed 0 --no-grad-ckpt --token-budget 1536
#
# The first argument is the name of the run; the remaining arguments are passed to train.train. Activate the Python
# environment first. The script writes runs/<name>/ (with runs/<name>.stdout and runs/<name>.test.stdout) and
# results/<name>/test_step_<STEPS>/.
# Environment: GPUS = the GPUs to use (default 0; one process per GPU; all ranks draw the same 16 items per step, so
# the number of GPUs changes only the micro-batch packing), MODEL = the backbone folder, DATA (default datasets/main),
# STEPS (default 1000), EVAL_STEPS (default: half of STEPS and STEPS), TEST_LOOPS = loops recorded in the test
# (default: as trained), NO_TEST = skip the test.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${GPUS:-0}
nproc=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
name=${1:?usage: train/run.sh <name> [arguments of train.train]}; shift
out=runs/$name
model=${MODEL:?set MODEL}; data=${DATA:-datasets/main}; steps=${STEPS:-1000}
mkdir -p runs results
if [ -d "$out" ]; then mv "$out" "${out}_failed_$(date -u +%H%M%S)"; fi
if [ "$nproc" -gt 1 ]; then launch="torchrun --standalone --nproc_per_node=$nproc -m"; else launch="python -m"; fi
$launch train.train --out "$out" --model "$model" --data "$data" --steps "$steps" \
  --eval-steps "${EVAL_STEPS:-$((steps / 2)),$steps}" "$@" > "$out.stdout" 2>&1
if [ ! -f "$out/DONE" ]; then echo "training did not finish: see $out.stdout"; exit 1; fi
if [ -z "${NO_TEST:-}" ]; then
  $launch eval.evaluate --run-dir "$out" --step "$steps" --split test ${TEST_LOOPS:+--loops $TEST_LOOPS} \
    --out "results/$name/test_step_$steps" > "$out.test.stdout" 2>&1
fi
