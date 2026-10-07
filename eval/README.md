# Evaluation

Evaluation has two steps. The test pass (`eval/evaluate.py`) runs a trained model over a split and writes one record
per item, with the option probabilities after every loop. The report (`eval/report.py`) reads the records of one or
more runs and prints the numbers of the paper's main table. The report runs no model, so every number can be
recomputed from the records.

## The test pass

```
python -m eval.evaluate --run runs/sansi_s0 --step 1000 --split test --out results/sansi_s0/test_step_1000
```

`--run` is a run folder written by `train.train`. The script reads `config.json` for the backbone and the dataset,
loads the checkpoint `ckpts/step_<step>/` (the LoRA adapter and `heads.pt`, the readouts), and runs the model once
over the split with as many loops as it was trained with. Every loop is read in this one run, so a single pass of
SanSi gives the decisions of all budgets from one loop to eight. Items are sorted by length and batched under a token
budget (`--token-budget`, default 32,000 padded tokens per batch), so no item is truncated. `train/run.sh` calls this
script after training.

Options:

- `--split dev` evaluates the development split, as it is evaluated during training; `--data` names another dataset
  folder than the one the run was trained on.
- `--loops N` runs and records N loops instead of the trained number. A loop beyond the trained ones has no trained
  readout of its own: up to loop 8 it is read with its untrained readout, which equals the frozen language-model
  head, and from loop 9 on it is read with the readout of loop 8. The paper uses this twice: the model trained with
  4 loops is run for 8 (`--loops 8`), and SanSi is run for 16 loops on the main set and on the depth tasks
  (`--loops 16`).
- `--zero-shot --model <backbone>` evaluates a backbone without fine-tuning, read with the frozen language-model
  head. Ouro is run for 8 loops and a standard model for one; `--data` defaults to `datasets/main`.
- `--check N` first evaluates N development items again and compares the probabilities with the development records
  that were written during training. It stops if the reloaded model does not reproduce them.
- On several GPUs, start the script with `torchrun --standalone --nproc_per_node=N -m eval.evaluate` and give the
  run folder as `--run-dir`. The items are split over the processes and the first process writes the results.

The script writes three files to `--out`: `records.jsonl.gz`, and `metrics.json` and `summary.md` with the metrics
of this one run (`eval/metrics.py`: accuracy, ECE, Brier score, confident errors, hard answers and total variation
distance at every loop, for all items, per test group, per item type within a test group, and per source).

Examples:

```
# SanSi trained with 4 loops, run for 8
python -m eval.evaluate --run runs/sansi4_s0 --loops 8 --out results/sansi4_s0/test_step_1000
# SanSi on the liar chains, run for 16 loops (the run was trained for 2,000 steps)
python -m eval.evaluate --run runs/depth_liars_sansi_s0 --step 2000 --loops 16 --out results/depth_liars_sansi_s0/test_step_2000
# Ouro-1.4B without fine-tuning
python -m eval.evaluate --zero-shot --model <Ouro-1.4B> --out results/ouro_zeroshot/test
```

## The records

`records.jsonl.gz` holds one JSON object per item:

| Field | Meaning |
|---|---|
| `fp` | the fingerprint of the item in the dataset |
| `source`, `layer`, `tier`, `group`, `k`, `tokens`, `qtype` | copied from the item: its source, item type, test group, group of related items, depth, prompt length and question type |
| `y` | the position of the gold option (the most probable label of a crowd-labelled item); -1 for an unanswerable item |
| `dist` | the target distribution |
| `soft` | `true` for a crowd-labelled item or an item with exact probabilities |
| `unans` | `true` for an unanswerable item |
| `p` | the option probabilities after every loop: one list per loop, in the order of the item's options |
| `q` | the same reading with the frozen language-model head alone, without the trained readout |
| `dz` | for loops 2 to T: how far the state at the last token moved from the previous loop, relative to its length |
| `gate` | the output of the backbone's own pre-trained gate at every loop (0 for a standard model); not used by the metrics |

## Metrics

All metrics use the probabilities `p` of the loop that is read, as they are, without post-hoc calibration. For an
item with K options, the answer is the most probable option and the confidence is its probability.

- **Correct.** An answerable item is answered correctly when the answer is the gold option; for a crowd-labelled
  item the gold option is the most probable label. An unanswerable item is answered correctly when the model gives
  no hard answer, that is, when the confidence is below (1 + 1/K) / 2, the midpoint between the uniform distribution
  and certainty.
- **Accuracy** is the share of items answered correctly.
- **Hard answers** is the share of unanswerable items with a confidence of (1 + 1/K) / 2 or more.
- **ECE** is the expected calibration error on the answerable items: they are sorted by confidence into ten bins of
  equal width, and the absolute difference between the accuracy and the mean confidence of a bin is averaged with
  the bin sizes as weights.
- **Evidence AUROC** is the probability that an answerable item gets a higher confidence than an unanswerable one,
  with ties counting one half. It is computed on the items of the four sources that contain both kinds: MuSiQue
  minimal pairs (in-distribution and 4 hops), SQuAD 2.0 and Kev's unknowable pairs.
- **Mean and standard deviation.** Every metric is computed for each run (seed); the report gives the mean and the
  sample standard deviation over the runs.
- **Bootstrap interval of a difference.** For the difference in accuracy between two models, the correctness of
  every item is first averaged over the runs of each model. The groups of related items (`group`) are then drawn
  with replacement 2,000 times, and the interval is given by the 2.5th and 97.5th percentiles of the difference. The
  interval is a Monte Carlo estimate with a fixed seed, so it can differ in the last digit from a printed one.

`eval/report.py` implements these definitions for the report; `eval/metrics.py` implements the same definitions of
correctness, accuracy, ECE and hard answers for the per-run files.

## The report

```
python -m eval.report --a results/sansi_s0/test_step_1000 results/sansi_s1/test_step_1000 results/sansi_s2/test_step_1000
```

`--a` takes the result folders (or the `records.jsonl.gz` files) of the runs of one model, one per seed. For every
recorded loop the report prints the accuracy on all items and in every test group, then the ECE, the hard-answer
rate and the evidence AUROC on all items, each as mean ± standard deviation over the runs.

With `--b`, the same table is printed for a second model, followed by the difference in accuracy with its 95%
bootstrap interval for all items and for every test group, and the difference seed by seed when both models have the
same number of runs:

```
python -m eval.report --a <the three SanSi folders> --b <the three SmolLM2 folders> --loop-a 8 --loop-b 1
```

`--loop-a` and `--loop-b` choose the loop that is read for the difference (default: the last recorded loop).
`--by layer` reports the accuracy by item type instead of by test group, and `--by k` by depth. `--layer <name>`
restricts the report to one item type; the depth tasks need it for the liar chains, because the test split of
`datasets/depth_liars` also contains the three other task families:

```
python -m eval.report --a <SanSi on depth_liars, 3 seeds> --b <Qwen3.5-4B on depth_liars, 3 seeds> --layer liars --loop-a 8 --loop-b 1
python -m eval.report --a <SanSi on depth_liars, 3 seeds> --layer liars --by k
```

In a depth dataset the test groups are `trained_k` (depths 1 to 8) and `deeper_k` (depths 9 to 16). All runs given to
one call must hold records of the same items.

## What to expect

The main table of the paper reports, on the 10,027 test items (mean ± standard deviation over three seeds):

| Model | Loop read | Accuracy (%) | ECE | Hard answers (%) | Evidence AUROC |
|---|---|---|---|---|---|
| SmolLM2-1.7B | 1 | 58.4 ± 0.7 | .069 ± .020 | 36.3 ± 4.2 | .765 ± .009 |
| Ouro-1.4B, one loop | 1 | 58.6 ± 0.2 | .137 ± .009 | 31.9 ± 2.2 | .837 ± .007 |
| Qwen3.5-2B | 1 | 66.7 ± 0.7 | .123 ± .007 | 27.9 ± 1.0 | .896 ± .002 |
| Qwen3.5-4B | 1 | 73.8 ± 0.6 | .113 ± .003 | 18.2 ± 2.5 | .948 ± .003 |
| SanSi, trained with 4 loops | 4 | 70.8 ± 0.4 | .094 ± .005 | 20.6 ± 1.7 | .931 ± .002 |
| SanSi | 4 | 71.6 ± 0.5 | .085 ± .009 | 18.1 ± 0.5 | .928 ± .003 |
| SanSi | 8 | 72.0 ± 0.7 | .093 ± .012 | 17.5 ± 1.3 | .935 ± .004 |
| SanSi-2.6B | 8 | 75.8 ± 0.6 | .078 ± .000 | 18.2 ± 2.4 | .943 ± .000 |

Without fine-tuning (`--zero-shot`, one run each), the same table gives 38.8% for SmolLM2-1.7B, 47.9% for Ouro-1.4B
at loop 4, 49.6% for Qwen3.5-2B and 61.6% for Qwen3.5-4B.

For the depth tasks, the summary table in the appendix of the paper gives these accuracies (%, means of three seeds):

| Task | Model | Depths 1 to 8 | Depths 9 to 16 |
|---|---|---|---|
| Liar chains | SanSi, loop 8 | 97.0 | 72.6 |
| Liar chains | SanSi, loop 16 | 96.9 | 72.7 |
| Liar chains | Qwen3.5-4B | 74.3 | 50.0 |
| Object swaps | SanSi, loop 8 | 88.2 | 63.7 |
| Object swaps | SanSi, loop 16 | 87.6 | 64.0 |
| Object swaps | Qwen3.5-4B | 68.0 | 34.2 |

A new training run will not reproduce these numbers to the last digit. The standard deviations over the seeds show
the variation to expect, and the micro-batches of a run depend on the number of GPUs (see `train/README.md`).
