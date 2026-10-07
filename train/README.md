# Training

`train/train.py` trains SanSi and the single-pass baselines with one recipe. `train/run.sh` starts one training run
and its test pass.

## The recipe

The backbone is frozen. Two things are trained: LoRA adapters on every attention and feed-forward projection of the
backbone (rank 64, alpha 128, dropout 0.05), and one small readout per loop. The readout adds a rank-16 correction
and a temperature to the letter logits of the frozen language-model head, and starts as the identity
(`sansi/model.py`). For Ouro-1.4B this gives 60.8M trained parameters, 0.27M of them in the readouts.

In every step the model is run for `--loops` loops, and the option probabilities are read after every loop. The loss
is the cross-entropy plus the Brier score against the target distribution of the item, averaged over the loops
(`sansi/loss.py`). The target is the gold option, the annotators' label distribution, or the uniform distribution for
an unanswerable item. The loss of a loop is backpropagated through all the loops before it.

A step is the next 16 items of the shuffled training set. The 16 items are sorted by length and packed into
micro-batches of at most `--token-budget` padded tokens, and every item weighs 1/16 of the loss of the step. The
option order of a multiple-choice item is shuffled each time the item is drawn. Items are never truncated.

The optimiser is AdamW with betas (0.9, 0.95) and no weight decay. The learning rate is 1e-4 for the adapters and
1e-3 for the readouts, with 200 warm-up steps and a cosine decay to 10% of the peak. Gradients are clipped at 1.0, the
forward pass runs in bfloat16, and gradient checkpointing is on unless `--no-grad-ckpt` is given. These are the
defaults of the script; the runs of the paper on the main set use them with 1,000 steps and seeds 0, 1 and 2.

For a given seed, every configuration sees the same items in the same order with the same option orders, and starts
from the same initialisation of the adapters and the readout.

## Commands

All commands are run from the root of the repository. `--model` is the folder of the backbone; the paper uses the
Base checkpoint of every backbone. The scripts load the backbone and its tokenizer with `trust_remote_code=True`,
which Ouro needs for its own model class. Repeat every command with `--seed 0`, `1` and `2`.

SanSi (Ouro-1.4B, 8 loops):

```
python -m train.train --model <Ouro-1.4B> --loops 8 --seed 0 --out runs/sansi_s0
```

SanSi trained with 4 loops:

```
python -m train.train --model <Ouro-1.4B> --loops 4 --seed 0 --out runs/sansi4_s0
```

The single-pass baselines. `--loops 1` runs the backbone once. A standard causal language model is read exactly as
loop 1 of Ouro is, and gets the adapters on the same kinds of modules (for Qwen3.5 also on the projections of its
linear-attention layers):

```
python -m train.train --model <Ouro-1.4B> --loops 1 --seed 0 --out runs/ouro1_s0
python -m train.train --model <SmolLM2-1.7B> --loops 1 --seed 0 --out runs/smollm2_s0
python -m train.train --model <Qwen3.5-2B-Base> --loops 1 --seed 0 --out runs/qwen2b_s0
python -m train.train --model <Qwen3.5-4B-Base> --loops 1 --seed 0 --out runs/qwen4b_s0
```

The Qwen3.5 backbones were run in a second environment (torch 2.7.1, transformers 5.18.0; see `requirements.txt`).

SanSi-2.6B is the same command on the larger backbone:

```
python -m train.train --model <Ouro-2.6B> --loops 8 --seed 0 --out runs/sansi26_s0
```

The depth tasks use 2,000 steps, no gradient checkpointing and a smaller token budget (1,536 for Ouro, 3,072 for
Qwen3.5-4B), because their items are short:

```
python -m train.train --model <Ouro-1.4B> --data datasets/depth_liars --loops 8 --steps 2000 --eval-steps 1000,2000 \
    --no-grad-ckpt --token-budget 1536 --seed 0 --out runs/depth_liars_sansi_s0
python -m train.train --model <Qwen3.5-4B-Base> --data datasets/depth_liars --loops 1 --steps 2000 --eval-steps 1000,2000 \
    --no-grad-ckpt --token-budget 3072 --seed 0 --out runs/depth_liars_qwen4b_s0
```

For the object swaps, replace `datasets/depth_liars` by `datasets/depth_swaps`.

A standard causal language model can also be given more than one loop (`--loops 8 --model <SmolLM2-1.7B>`). Its
layers are then applied once per loop, and every loop after the first reads the final hidden state of the previous
loop in place of the token embeddings. This is the loop "as in Ouro" that the appendix of the paper adds to SmolLM2
after pre-training.

### Several GPUs

```
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 -m train.train --model <Ouro-1.4B> --loops 8 \
    --seed 0 --out runs/sansi_s0
```

Every process draws the same 16 items per step and takes its share of them. The gradients are summed over the
processes, so a step is the gradient of the same 16 items as on one GPU. The first process writes the logs, the
records and the checkpoints, and the evaluation on the development set is split over the processes. The number of
GPUs and the token budget change how the items of a step are grouped into micro-batches. They do not change which
items make a step, but a run with another grouping is not bit-identical.

### The run script

`train/run.sh` runs one training and then its test pass (`eval.evaluate`) on the GPUs listed in `GPUS`, with
`torchrun` when there are several:

```
GPUS=0,1 MODEL=<Ouro-1.4B> bash train/run.sh sansi_s0 --loops 8 --seed 0
GPUS=0 MODEL=<Ouro-1.4B> DATA=datasets/depth_liars STEPS=2000 TEST_LOOPS=16 \
    bash train/run.sh depth_liars_sansi_s0 --loops 8 --seed 0 --no-grad-ckpt --token-budget 1536
```

The first argument is the name of the run, and the remaining arguments are passed to `train.train`. The script writes
`runs/<name>/`, the output of the two programs to `runs/<name>.stdout` and `runs/<name>.test.stdout`, and the test
results to `results/<name>/test_step_<STEPS>/`. An existing folder `runs/<name>` is renamed first. The header of the
script lists the environment variables.

## What a run writes

| File in `--out` | Content |
|---|---|
| `config.json` | all arguments, the number of trained parameters, the commit of the code, the hash of the dataset manifest |
| `log.jsonl` | one line every 20 steps: the loss, the accuracy at every loop (also by item type), the temperature of every readout, learning rates, gradient norms, seconds per step, tokens per step and peak GPU memory; and one line per evaluation on the development set |
| `train_records.jsonl.gz` | one line per item and step: correctness, top probability and loss at every loop |
| `eval/step_N/` | the evaluation on the development set at step N: `records.jsonl.gz`, `metrics.json`, `summary.md` |
| `ckpts/step_N/` | the checkpoint at step N: the LoRA adapter as written by peft, `heads.pt` (the readouts) and `train_state.pt` (the state of the optimiser, of the item pool and of the random generators) |
| `DONE` | written when the run has finished |

The development set is evaluated at the steps given by `--eval-steps` (default 500 and 1000) and at the last step, and
a checkpoint is written at each of them; `--save-steps` adds checkpoints without an evaluation. The evaluations do
not change the training.

## Time and memory

The paper reports these sizes and times (its table of sizes and measured cost):

| Model | Backbone parameters | Trained parameters | Training (GPU-minutes) | One test pass (GPU-seconds) |
|---|---|---|---|---|
| SmolLM2-1.7B | 1.71B | 72.6M | 39 | 382 |
| Ouro-1.4B, one loop | 1.43B | 60.8M | 33 | 334 |
| SanSi, 8 loops | 1.43B | 60.8M | 313 | 2,571 |
| SanSi-2.6B, 8 loops | 2.67B | 121.4M | 558 | 4,963 |
| Qwen3.5-2B | 1.88B | 67.5M | 48 | 387 |
| Qwen3.5-4B | 4.21B | 130.2M | 75 | 789 |

The training time is that of 1,000 steps on the main set: wall-clock time times the number of cards, on RTX A6000
cards, except for SmolLM2-1.7B (RTX A5500) and Qwen3.5-2B (two RTX A5000). It includes five evaluations on the
development set (two for SanSi-2.6B). The test time is one pass over the 10,027 test items on one RTX A6000 with at
most 32,000 tokens per batch. Looping costs computation: the eight loops of SanSi take 7.7 times the test time of
the same backbone with one loop (2,571 against 334 seconds).

The paper does not report memory use. `log.jsonl` records the peak GPU memory of every logging window (`peak_gb`).
If a run does not fit, lower `--token-budget`.
