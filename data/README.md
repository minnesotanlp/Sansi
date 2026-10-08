# Data

This folder builds the three datasets of the paper:

| Dataset folder | Built by | Contents |
|---|---|---|
| `datasets/main` | `data.build_main` | the decision suite: 12,800 training, 2,471 development and 10,027 test items |
| `datasets/depth_liars` | `data.generate_depth`, then `data.build_depth` | the liar chains, together with three further program-generated task families |
| `datasets/depth_swaps` | `data.generate_depth`, then `data.build_depth` | the object swaps |

No model is called while a dataset is built. A tokenizer is used only to count the length of every prompt
(`meta.tokens`), which the training and test scripts use to form batches. All three datasets were built with the
tokenizer of Ouro-1.4B, also for the runs with other backbones, so pass the folder of Ouro-1.4B as `--tokenizer`.

Every dataset folder has the same layout: `train/<source>.jsonl.gz`, `dev/<source>.jsonl.gz`,
`test/<source>.jsonl.gz` and `manifest.json`. A file holds one JSON object per line (`data.common.read_jsonl` reads
it). The manifest records the arguments of the build and the number of items in every file, with their lengths.

## Item format

An item is one typed decision: a state, a question and the declared options.

| Field | Meaning |
|---|---|
| `state` | the text that the decision is about (a passage, a set of rules, records, a story) |
| `question` | the question |
| `options` | the declared options, between 2 and 26 |
| `answer_index` | the position of the gold option; for a crowd-labelled item the most probable label; `null` for an unanswerable item |
| `answer_distribution` | the target distribution over the options: one-hot, the annotators' label distribution, or uniform for an unanswerable item |
| `deterministic` | `true` when the target is one-hot |
| `qtype` | `choice` (the option order is shuffled every time the item is drawn in training), `noul` (a yes/no question; the order is kept) or `score` (ordered levels; the order is kept) |
| `k`, `depth_kind` | a depth annotation where the source has one (proof depth, number of hops, chain length), otherwise `null` |
| `dataset`, `family`, `split` | the source dataset and its split |
| `fingerprint` | a hash that identifies the item; the records of the test pass refer to it as `fp` |
| `meta.layer` | the item type: `A` to `F` and `JB` in the main set, the task family in the depth datasets |
| `meta.tier` | the test group: `in_dist`, `near`, `far` or `jevbench` in the main set (`train` for training items); `trained_k` or `deeper_k` in the test split of a depth dataset |
| `meta.source_set` | the name of the source within the dataset; the results per source use it |
| `meta.group` | the group of related items that the item belongs to; the bootstrap resamples these groups |
| `meta.tokens` | the length of the prompt in Ouro tokens |
| `meta.unanswerable` | `true` for an unanswerable item |

The remaining entries of `meta` come from the source (ids, the hop count of a question, the hidden structure of a
generated item) and are not used by the training or test scripts.

The model reads an item as the prompt

```
<state>

Question: <question>
Options: (A) <option 1> (B) <option 2> ...
Answer:
```

and is read at the last token (`sansi.model.render_prompt`).

## The main set

### Item types and test groups

| Type (`meta.layer`) | Training sources | Near transfer | Far transfer |
|---|---|---|---|
| A, classification | Kev decision-v7: AG News, DBpedia, Amazon, Yelp, IMDB, SST-5, Banking77, TREC | none | Emotion, TweetEval, Yahoo Answers; Kev transfer-v9: Emotion, offensive tweets |
| B, multi-step reasoning | ProofWriter, CLUTRR with 2 to 4 hops, MuSiQue with 2 to 3 hops, Kev rules and policies | CLUTRR with 5 to 10 hops, ProofWriter in natural language, MuSiQue with 4 hops | FOLIO, BBH, HoVer; Kev transfer-v9: held-out rules and policies |
| C, uncertain evidence | MuSiQue minimal pairs, SQuAD 2.0, ChaosNLI (SNLI and alphaNLI) | MuSiQue pairs with 4 hops, ChaosNLI (MNLI) | Sys1Cal; Kev transfer-v9: unknowable pairs |
| D, long documents | MuSiQue and HotpotQA with distractor paragraphs | none | ContractNLI, QuALITY; Kev transfer-v9: buried evidence |
| E, sentence pairs | Kev decision-v7: MNLI, BoolQ | MNLI, mismatched genres | ANLI, QNLI, PAWS, WANLI; Kev transfer-v9: QNLI, PAWS |
| F, knowledge | none | none | SciQ, ARC, MMLU, MMLU-Pro, CommonsenseQA; Kev transfer-v9: MMLU, MMLU-Pro, SciQ |

The test groups (`meta.tier`) are defined by the distance from the training data. In-distribution items (`in_dist`)
are new items from the training sources. Near-transfer items (`near`) are harder or reworded versions of trained
types. Far-transfer items (`far`) come from sources that are never trained on. The public items of JevBench (easy,
standard and hard) form a fourth group (`jevbench`, type `JB`); they are used for the test only.

The docstring of `data/build_main.py` gives the same table in short form, and the function `build` in that file is
the exact definition: it lists every source with the number of items that is drawn from it.

### How the set is built

1. The training items are drawn first: fixed numbers per source, spread evenly over depths, hop counts and answers
   where a source has them. All random choices are seeded inside the script, so a build is repeatable.
2. Test candidates are then drawn from files or questions that training does not use. A candidate that repeats
   training material is not drawn: this is the case when its state equals a training state once case and punctuation
   are dropped, or when it shares 80% or more of its sentences with a training state. Further rules are specific to a
   source: CLUTRR test stories that equal a training story up to the names are left out, MuSiQue test questions that
   share a single-hop sub-question with a training question are left out, and HoVer claims that were written from one
   of our HotpotQA training questions are left out. A candidate from a public source that also occurs in Kev's
   transfer-v9 development file is left out of the public source, so that it is not tested twice.
3. Every test source is split into a development part (20%) and a test part (80%). Before the split, test candidates
   are joined into connected components: items of one group (a question with its versions, a minimal pair, a Kev
   group, a document), items with the same state, CLUTRR stories that differ only in names, MuSiQue questions that
   share a sub-question, HoVer claims from the same HotpotQA question, and items that share 80% or more of their
   sentences. A component is never split, so related items stay on one side. `meta.group` is set to the component.
   JevBench is not split and goes to the test set whole.
4. The build stops if two items have the same text.

`manifest.json` holds the counts of every file (items, groups, unanswerable and crowd-labelled items, options, token
lengths) and the totals by item type and test group. `report.txt` lists the checks that were made while building, for
example how many candidates were left out by each rule.

### Unanswerable and crowd-labelled items

An unanswerable item is an item whose key evidence is missing from the state. It comes from one of three sources:
MuSiQue minimal pairs (the same question with its supporting paragraphs, and with each supporting paragraph that the
official unanswerable version removed replaced by the most similar paragraph of that version), unanswerable SQuAD 2.0
questions (the options are the annotated plausible answer and three other answer spans of the passage), and Kev's
unknowable pairs. Such an item has no gold option: `answer_index` is `null`, the target is the uniform distribution
and `meta.unanswerable` is `true`. It counts as answered correctly when the model gives no hard answer, that is, when
its top probability is below (1 + 1/K) / 2 for K options.

A crowd-labelled item (ChaosNLI) or an item with exact probabilities (Sys1Cal) has that distribution as its target
(`deterministic` is `false` unless the distribution is one-hot); its accuracy uses the most probable label.

### Raw files

`data.build_main` reads the source datasets from the folder given as `--raw`. `data.download_raw` fills that folder:
it downloads every file that the build reads at the revision the paper's datasets were built from, and checks its
sha256. `data/sources.json` lists the 57 files (1.3 GB) with their origin: a Hugging Face dataset at a commit, a
GitHub repository at a commit (CLUTRR, HoVer, QuALITY, Sys1Cal, JevBench and Kev's transfer-v9 file), or the
official address (SQuAD 2.0 and ContractNLI, which have no versions; their sha256 is checked all the same). Folder
names follow the pattern `<owner>__<name>`. FOLIO is gated: accept its terms on its Hugging Face page and log in with
`hf auth login` before the download. The table below lists what the build reads from each folder.

| Folder under `--raw` | Files that are read | Used for |
|---|---|---|
| `jaredpalmer__kev` | `evals/v7/decision-v7/train.jsonl`, `evals/v9/transfer-v9/development.jsonl` | the Kev sources (the first from the Hugging Face dataset jaredpalmer/kev-suites, the second from the repository github.com/jaredpalmer/kev) |
| `tasksource__proofwriter` | `data/train-*.parquet`, `data/test-*.parquet` (configurations `depth-5` and `NatLang`) | ProofWriter |
| `CLUTRR__v1` | `data/gen_train234_test2to10/train.csv`, `validation.csv`, `test.csv` | CLUTRR |
| `bdsaglam__musique` | `musique_ans_v1.0_train.jsonl`, `musique_ans_v1.0_dev.jsonl`, `musique_full_v1.0_train.jsonl`, `musique_full_v1.0_dev.jsonl` | MuSiQue (types B, C and D) |
| `hotpotqa__hotpot_qa` | `distractor/train-*.parquet`, `distractor/validation-*.parquet` | HotpotQA |
| `rajpurkar__squad_v2_official` | `train-v2.0.json`, `dev-v2.0.json` | SQuAD 2.0 |
| `earino__chaosnli` | `raw/chaosNLI_snli.jsonl`, `raw/chaosNLI_alphanli.jsonl`, `raw/chaosNLI_mnli_m.jsonl` | ChaosNLI (the docstring of `data.real.chaosnli` calls this folder a preservation mirror) |
| `dair-ai__emotion` | `split/test-*.parquet` | Emotion |
| `cardiffnlp__tweet_eval` | `offensive/`, `hate/`, `irony/`, `sentiment/`: `test-*.parquet` | TweetEval |
| `community-datasets__yahoo_answers_topics` | `yahoo_answers_topics/test-*.parquet` | Yahoo Answers |
| `yale-nlp__FOLIO` | `folio_v2_train.jsonl`, `folio_v2_validation.jsonl` | FOLIO (the docstring of `data.real.folio` notes that this release is gated) |
| `lukaemon__bbh` | `<task>/*.parquet`; the tasks `tracking_shuffled_objects_*` and `logical_deduction_*` are used | BBH |
| `hover-nlp__hover` | `data/hover_train_release_v1.1.json` | HoVer: labels, hop counts, HotpotQA ids |
| `Dzeniks__hover` | `train.jsonl` | HoVer: evidence text (a mirror, joined to the official release by claim) |
| `little-g-ai__Sys1Cal-v1` | `datasets/generated/v0.1.0_tiny_cleanvars_binary.jsonl` | Sys1Cal |
| `stanfordnlp__contract-nli` | `contract-nli/test.json` | ContractNLI |
| `nyu-mll__quality` | `data/v1.0.1/QuALITY.v1.0.1.htmlstripped.dev` | QuALITY |
| `nyu-mll__multi_nli` | `data/validation_mismatched-*.parquet` | MNLI, mismatched genres |
| `facebook__anli` | `plain_text/dev_r1-*.parquet`, `dev_r2-*.parquet`, `dev_r3-*.parquet` | ANLI |
| `nyu-mll__glue` | `qnli/validation-*.parquet` | QNLI |
| `google-research-datasets__paws` | `labeled_final/test-*.parquet` | PAWS |
| `alisawuffles__WANLI` | `test.jsonl` | WANLI |
| `allenai__sciq` | `data/test-*.parquet` | SciQ |
| `allenai__ai2_arc` | `ARC-Easy/test-*.parquet`, `ARC-Challenge/test-*.parquet` | ARC |
| `cais__mmlu` | `all/test-*.parquet` | MMLU |
| `TIGER-Lab__MMLU-Pro` | `data/test-*.parquet` | MMLU-Pro |
| `tau__commonsense_qa` | `data/validation-*.parquet` | CommonsenseQA |
| `fstandhartinger__jevbench` | `datasets/public/easy.jsonl`, `original.jsonl`, `hard.jsonl` | JevBench (the docstring of `data.real.jevbench` names the repository github.com/fstandhartinger/jevbench; public files only) |

The parquet and csv files are read with pandas, which needs `pyarrow` for parquet. The versions of pandas, pyarrow
and tokenizers in `requirements.txt` are the ones with which the three datasets were rebuilt byte for byte.

### Commands

```
python -m data.download_raw --raw raw
python -m data.build_main --raw raw --tokenizer raw/ByteDance__Ouro-1.4B --out datasets/main
python -m data.verify --root datasets/main --dataset main
python -m data.check_overlap --root datasets/main --raw raw --out results/overlap_main
```

`data.download_raw` also fetches the tokenizer of Ouro-1.4B at the commit that was used (into
`raw/ByteDance__Ouro-1.4B`); the full Ouro-1.4B folder gives the same token counts. It stops with a list of the files
that are missing or differ.

`data.build_main` writes `datasets/main/train/`, `dev/`, `test/`, `manifest.json` and `report.txt`, and prints one
line per file. Expect 12,800 training items from 20 sources, 2,471 development items, and 10,027 test items from 59
sources (2,116 in-distribution, 1,871 near transfer, 5,809 far transfer and 231 JevBench items).

The main set of the paper leaves 1,305 ProofWriter test questions out of the test candidates, because earlier
versions of our datasets had already used them. Their ids are in `data/proofwriter_excluded_ids.json`, which
`--exclude-ids` reads by default, so the command above draws the same ProofWriter items as the paper. With
`--exclude-ids ''` the build uses all ProofWriter test questions, and the ProofWriter test and development items then
differ from those of the paper. The selection of the other sources does not use the list.

`data.verify` checks that a built folder is the paper's dataset: `data/checksums.json` holds the sha256 of the
uncompressed content of every file of the three datasets and its number of items, and the check passes only when
every file is identical. The manifest is not compared, since it records local paths.

`data.check_overlap` is an independent check of a built dataset folder. For every pair of splits it counts the items
of the later split that match an item of the earlier split by five criteria: the same item, the same state, the same
state once case and punctuation are dropped, the same set of sentences, and a sentence overlap of 80% or more. For
CLUTRR it also counts stories that are equal up to the names. It writes `overlap.json` and `summary.md` to `--out`.
`--raw` is needed only when the dataset contains CLUTRR items.

## The depth tasks

Both depth tasks are generated by programs. An item has a hidden structure whose depth `k` is exact, its answer is
computed from the structure, and the text is rendered from templates. Training uses depths 1 to 8 and the test uses
depths 1 to 16, with 600 training and 120 test items per family and depth before the development items are taken out.

| Family (`meta.layer`) | Task | Depth `k` | Options |
|---|---|---|---|
| `liars` | liar chains: each person says that another person tells the truth or lies; does the last person tell the truth? A second chain of the same length does not matter for the question. | the length of the chain | yes, no |
| `shuffle` | object swaps: five people each hold an object and swap in pairs; who holds a given object at the end? `k` further swaps do not involve the object. | the number of swaps that move the object | the five people |
| `rooms` | can room T be reached from room S through the listed doors? | the length of the shortest path | yes, no |
| `ledger` | after the listed transactions, is the balance at least a given amount? A withdrawal that would overdraw is declined. | the number of transactions | yes, no |
| `policy` | is a case approved under a chain of rules? | the length of the rule chain | approved, not approved, cannot be determined |

The liar chains of the paper come from `datasets/depth_liars`. The models for this task were trained on the training
split of that folder, which holds 17,336 items of four families (`liars`, `rooms`, `ledger`, `policy`), 4,480 of them
liar chains; the paper reports the liar chains of the test split. The object swaps come from `datasets/depth_swaps`,
which holds the family `shuffle` alone (4,480 training items). Each test split has 120 items per depth, 1,920 per
task.

### Commands

```
python -m data.generate_depth --out datasets/generated_liars --probe
python -m data.build_depth --src datasets/generated_liars --out datasets/depth_liars --tokenizer <folder of Ouro-1.4B>

python -m data.generate_depth --out datasets/generated_swaps --families shuffle --probe
python -m data.build_depth --src datasets/generated_swaps --out datasets/depth_swaps --tokenizer <folder of Ouro-1.4B> --families shuffle

python -m data.verify --root datasets/depth_liars --dataset depth_liars
python -m data.verify --root datasets/depth_swaps --dataset depth_swaps
```

`data.generate_depth` generates the items of the families given by `--families` (default: `rooms`, `ledger`, `liars`
and `policy`) with a fixed seed, drops items whose hidden structure was generated twice, and checks every item with
the family's `verify` function, which parses the rendered text back and recomputes the answer by a different
algorithm. It writes `<family>_train.jsonl.gz`, `<family>_test.jsonl.gz` and a manifest. Training and test items
never share a hidden structure. With `--probe` it also trains a bag-of-words classifier and the best threshold rule
for each of the family's surface heuristics on the training items and writes their test accuracy to the manifest.
They should stay at the chance level (50% for the liar chains, 20% for the object swaps). The bag-of-words accuracy
varies slightly between runs, because its features use Python's salted string hash.

`data.build_depth` turns a generated folder into a dataset folder. It moves about 7% of every family and depth of
the training items to `dev/` (chosen by a hash of the fingerprint), keeps the test items as they are, sets
`meta.tier` of a test item to `trained_k` for depths up to 8 and to `deeper_k` above, and counts the tokens of every
prompt. Expect 17,336 training, 1,241 development and 7,541 test items for `datasets/depth_liars` (the family
`rooms` has fewer items than the others because duplicates are dropped), and 4,480, 320 and 1,920 for
`datasets/depth_swaps`.

## Files

| File | Content |
|---|---|
| `download_raw.py`, `sources.json` | downloads the raw files at the revisions the paper used and checks them; the list of these files |
| `build_main.py` | builds the main set |
| `proofwriter_excluded_ids.json` | the ProofWriter test questions that the main set leaves out |
| `real.py` | converters from the source datasets to the item format |
| `verify.py`, `checksums.json` | checks that a built dataset is the paper's; the checksums of the paper's datasets |
| `check_overlap.py` | the overlap check between the splits of a dataset folder |
| `generate_depth.py` | generates the items of the depth tasks |
| `build_depth.py` | turns generated items into a dataset folder |
| `liars.py`, `shuffle.py`, `rooms.py`, `ledger.py`, `policy.py` | the generators, one per family |
| `probe.py` | the shortcut probes used by `generate_depth --probe` |
| `common.py` | shared helpers: item fields of the generators, reading and writing JSON lines |
