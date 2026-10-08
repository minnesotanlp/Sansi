# Sansi

Code for the paper "SanSi: A Looped Typed Decision Model for System 1.5 Thinking" (Shuyu Gan, Young-Jun Lee and
Dongyeop Kang, University of Minnesota).

Paper: https://arxiv.org/abs/2610.07730

Project page: https://minnesotanlp.github.io/Sansi/

Models: https://huggingface.co/minnesotanlp/SanSi and https://huggingface.co/minnesotanlp/SanSi-2.6B

## What SanSi is

A typed decision model answers a declared question by giving a probability to each of the declared options, without
generating text. SanSi is a typed decision model that loops. It is built on Ouro-1.4B, a language model that was
pre-trained to loop: one stack of 24 shared layers is applied T = 8 times. The backbone is frozen. LoRA adapters of
rank 64 and one small readout per loop are trained (61M parameters), with the cross-entropy plus the Brier score at
every loop. The option probabilities are read after every loop, so one model serves every budget from one loop to
eight in a single run. Looping is paid for with computation: T loops cost T passes through the stack and add no
parameters.

This repository contains the scripts that build the datasets, the training script and the evaluation scripts of the
paper. With them you can build the main set and the two depth tasks, train SanSi and the single-pass models that share
its recipe, and compute the measures of the main table after every loop and for every test group. The repository does
not contain the code for the models that come from other systems (Kev-4B trained on our data, the released Kev-4B and
the Jev API), for the use of SanSi as a verifier, for the training ablations, or for the figures and the analysis
tables of the paper.

## Expected results

Accuracy (%) on the 10,027 test items, from the main table of the paper: mean ± standard deviation over three
training seeds. All models are trained on the same data with the same recipe and seeds. Cost is the GPU time of one
pass over the test set, relative to Ouro-1.4B with one loop.

| Model | Parameters | Loops | Cost | All | In-distribution | Near transfer | Far transfer | JevBench |
|---|---|---|---|---|---|---|---|---|
| SmolLM2-1.7B | 1.7B | 1 | 1.1 | 58.4 ± 0.7 | 76.1 ± 0.8 | 57.7 ± 0.6 | 52.3 ± 0.7 | 58.3 ± 0.7 |
| Ouro-1.4B, one loop | 1.4B | 1 | 1.0 | 58.6 ± 0.2 | 77.3 ± 1.0 | 59.9 ± 0.9 | 51.3 ± 0.4 | 59.7 ± 3.0 |
| Qwen3.5-2B | 1.9B | 1 | 1.2 | 66.7 ± 0.7 | 84.3 ± 1.4 | 62.3 ± 2.5 | 61.7 ± 0.4 | 67.5 ± 1.5 |
| Qwen3.5-4B | 4.2B | 1 | 2.4 | 73.8 ± 0.6 | 88.5 ± 0.4 | 68.9 ± 2.4 | 70.1 ± 0.3 | 73.2 ± 1.9 |
| SanSi, trained with 4 loops | 1.4B | 4 | 3.8 | 70.8 ± 0.4 | 86.4 ± 0.5 | 64.5 ± 2.4 | 67.3 ± 0.4 | 69.3 ± 0.7 |
| SanSi, read at loop 4 | 1.4B | 4 | 3.8 | 71.6 ± 0.5 | 86.5 ± 0.5 | 67.2 ± 2.0 | 67.6 ± 0.5 | 72.6 ± 1.2 |
| SanSi, read at loop 8 | 1.4B | 8 | 7.7 | 72.0 ± 0.7 | 86.6 ± 0.6 | 67.7 ± 2.7 | 68.0 ± 0.1 | 72.3 ± 1.6 |
| SanSi-2.6B, read at loop 8 | 2.7B | 8 | 14.8 | 75.8 ± 0.6 | 88.4 ± 0.8 | 74.4 ± 2.5 | 71.6 ± 0.3 | 79.4 ± 0.7 |

`eval/README.md` lists the other columns of that table and the results on the two depth tasks.

## What is in the repository

```
README.md
requirements.txt
sansi/    the model: the wrapper around the backbone with a readout after every loop (model.py), the loss (loss.py),
          and loading a released model (hub.py)
data/     builders of the main set and of the two depth tasks (liar chains, object swaps); see data/README.md
train/    the training script (train.py) and a run script (run.sh); see train/README.md
eval/     the test pass (evaluate.py), the per-run metrics (metrics.py) and the report (report.py); see eval/README.md
docs/     the project page
```

All scripts are run as modules from the root of the repository, for example `python -m train.train`.

## Install

The runs of the paper used Python 3.11 and the package versions in `requirements.txt`:

```
pip install -r requirements.txt
```

The Qwen3.5 backbones were run in a second environment with torch 2.7.1 and transformers 5.18.0.

You also need the backbones (Ouro-1.4B for SanSi; Ouro-2.6B, SmolLM2-1.7B, Qwen3.5-2B-Base and Qwen3.5-4B-Base for
the other models) and, for the main set, the source datasets, which `data.download_raw` downloads at the revisions
the paper used (`data/README.md`). The scripts take local folders: `--model` and `--tokenizer` for a backbone, `--raw`
for the source datasets.

## Use a released model

The models of the paper are on the Hugging Face Hub: [minnesotanlp/SanSi](https://huggingface.co/minnesotanlp/SanSi)
(on Ouro-1.4B) and [minnesotanlp/SanSi-2.6B](https://huggingface.co/minnesotanlp/SanSi-2.6B) (on Ouro-2.6B), each the
run with seed 0. A model repository holds the LoRA adapter and the readouts; `sansi.hub.load` downloads it together
with the backbone at the revision the model was trained on.

```python
from sansi.hub import load, decide

model, tok = load("minnesotanlp/SanSi")
probs = decide(model, tok,
               state="Mia is taller than Sam. Sam is taller than Lee.",
               question="Who is the shortest?",
               options=["Mia", "Sam", "Lee"])
print(probs[0])    # after loop 1: about [0.02, 0.07, 0.91]
print(probs[-1])   # after loop 8: about [0.00, 0.00, 1.00]
```

`decide` writes the prompt the model was trained on (`sansi.model.render_prompt`) and returns one list of option
probabilities per loop, in the order of the options; `loops=` runs fewer or more loops than the eight it was trained
with. `load(..., backbone=<folder>)` uses a local copy of the backbone instead of downloading it. Use a GPU; the
backbone runs in bfloat16.

## Quick start

1. Download the source datasets and build the main set. This writes `datasets/main` with 12,800 training, 2,471
   development and 10,027 test items; `data.verify` checks that every file is identical to the paper's.

   ```
   python -m data.download_raw --raw raw
   python -m data.build_main --raw raw --tokenizer raw/ByteDance__Ouro-1.4B --out datasets/main
   python -m data.verify --root datasets/main --dataset main
   ```

2. Train SanSi with one seed. This writes the run folder `runs/sansi_s0` with logs, evaluations on the development
   set and checkpoints.

   ```
   python -m train.train --model <Ouro-1.4B> --loops 8 --seed 0 --out runs/sansi_s0
   ```

3. Run the test pass. It runs the model once over the test set, reads every loop, and writes one record per item
   with the option probabilities after each of the eight loops.

   ```
   python -m eval.evaluate --run runs/sansi_s0 --out results/sansi_s0/test_step_1000
   ```

4. Print the numbers of the main table from the records. With the records of several seeds the report gives the
   mean and the standard deviation; with a second model (`--b`) it gives the difference with a 95% bootstrap
   interval.

   ```
   python -m eval.report --a results/sansi_s0/test_step_1000
   ```

A single-pass baseline is the same sequence with `--loops 1` and another backbone, for example
`python -m train.train --model <SmolLM2-1.7B> --loops 1 --seed 0 --out runs/smollm2_s0`.

The two depth tasks need no source datasets, only the tokenizer of Ouro-1.4B:

```
python -m data.generate_depth --out datasets/generated_swaps --families shuffle
python -m data.build_depth --src datasets/generated_swaps --out datasets/depth_swaps --tokenizer <Ouro-1.4B> --families shuffle
python -m train.train --model <Ouro-1.4B> --data datasets/depth_swaps --loops 8 --steps 2000 --eval-steps 1000,2000 \
    --no-grad-ckpt --token-budget 1536 --seed 0 --out runs/depth_swaps_sansi_s0
python -m eval.evaluate --run runs/depth_swaps_sansi_s0 --step 2000 --loops 16 --out results/depth_swaps_sansi_s0/test_step_2000
python -m eval.report --a results/depth_swaps_sansi_s0/test_step_2000 --by k
```

The READMEs in `data/`, `train/` and `eval/` describe every script, the files it writes and what to expect.

## License

The code is released under the Apache License 2.0 (see `LICENSE`). The datasets that the builders in `data/`
read keep their own licenses.

## Citation

```bibtex
@misc{gan2026sansiloopedtypeddecision,
      title={SanSi: A Looped Typed Decision Model for System 1.5 Thinking},
      author={Shuyu Gan and Young-Jun Lee and Dongyeop Kang},
      year={2026},
      eprint={2610.07730},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2610.07730},
}
```
