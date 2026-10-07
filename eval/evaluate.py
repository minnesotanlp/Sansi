"""The test pass: run a trained model over a split with every loop read out; per-item records and metrics.

  python -m eval.evaluate --run runs/sansi_s0 --step 1000 --split test --out results/sansi_s0/test_step_1000
  ... --split dev      the dev part (as evaluated during training)
  ... --loops 16       more loops than the run was trained with
  python -m eval.evaluate --zero-shot --model <backbone> --data datasets/main --out results/ouro_zeroshot/test

Items are batched by length under a token budget (rows x longest row <= --token-budget), so long documents are never
cut and short items share a batch. Results are written to <out>/records.jsonl.gz, metrics.json, summary.md
(eval.metrics); eval.report turns the records of several runs into the numbers of the paper's tables.
"""
import argparse
import glob
import json
import os
import time

import torch

from data.common import read_jsonl, write_jsonl
from eval import metrics as run_metrics
from sansi.model import encode_batch_hf


def load_split(data_dir, split):
    items = []
    for f in sorted(glob.glob(os.path.join(data_dir, split, "*.jsonl.gz"))):
        items += read_jsonl(f)
    return items


def length_batches(items, token_budget, max_rows=64):
    """Items sorted by length, cut into batches with rows x longest <= token_budget (at least one item each)."""
    order = sorted(items, key=lambda it: (it["meta"]["tokens"], it["fingerprint"]))
    batches, cur = [], []
    for it in order:
        longest = max([x["meta"]["tokens"] for x in cur] + [it["meta"]["tokens"]])
        if cur and ((len(cur) + 1) * longest > token_budget or len(cur) >= max_rows):
            batches.append(cur)
            cur = []
        cur.append(it)
    if cur:
        batches.append(cur)
    return batches


@torch.no_grad()
def run_records(model, tok, items, device, loops, max_len, token_budget=32000):
    """One record per item (eval.metrics): probabilities at every loop, the backbone's gate output, and the process:
    q = the frozen LM head's option probabilities at every loop, dz = how much the readout vector moved from the
    previous loop, relative to its length (loops 2..T)."""
    was_training = model.training
    model.eval()
    model.record_process = True
    records = []
    for chunk in length_batches(items, token_budget):
        batch, kept = encode_batch_hf(chunk, tok, max_len)
        assert len(kept) == len(chunk), "an item is longer than max_len"
        b = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outs = model(b["ids"], b["mask"], b["last"], b["n_options"], loops=loops)
        probs = [torch.softmax(o.float(), -1).cpu() for o in outs]
        raw = [torch.softmax(o.float(), -1).cpu() for o in model.last_raw]
        dz = [d.cpu() for d in model.last_dz]
        gates = [g.cpu() for g in model.last_gates]
        for j, it in enumerate(kept):
            n, m = len(it["options"]), it["meta"]
            rec = {"fp": it["fingerprint"], "source": m["source_set"], "layer": m["layer"], "tier": m["tier"],
                   "k": it["k"], "group": m["group"], "tokens": m["tokens"], "qtype": it.get("qtype"),
                   "y": -1 if it["answer_index"] is None else it["answer_index"],
                   "dist": it["answer_distribution"], "soft": (not it["deterministic"]) and it["answer_index"] is not None,
                   "unans": bool(m.get("unanswerable")),
                   "p": [[round(x, 5) for x in p[j, :n].tolist()] for p in probs],
                   "q": [[round(x, 5) for x in p[j, :n].tolist()] for p in raw],
                   "dz": [round(d[j].item(), 5) for d in dz],
                   "gate": [round(g[j].item(), 5) for g in gates]}
            records.append(rec)
    model.record_process = False
    if was_training:
        model.train()
    return records


def write_eval(records, out_dir, title):
    """Metrics of a set of records, written with the records to out_dir; returns the metrics."""
    metrics = run_metrics.compute(records)
    os.makedirs(out_dir, exist_ok=True)
    write_jsonl(records, os.path.join(out_dir, "records.jsonl.gz"))
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f)
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write(run_metrics.summary(metrics, title))
    return metrics


def load_model(run_dir, step, device):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sansi.model import OuroSanSi, option_token_ids
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    tok = AutoTokenizer.from_pretrained(cfg["model"], trust_remote_code=True)
    letter_ids, _ = option_token_ids(tok)
    lm = AutoModelForCausalLM.from_pretrained(cfg["model"], trust_remote_code=True, torch_dtype=torch.bfloat16)
    lm.config.use_cache = False
    hidden_size = lm.lm_head.in_features
    ck = os.path.join(run_dir, "ckpts", "step_%d" % step)
    lm = PeftModel.from_pretrained(lm, ck)
    state = torch.load(os.path.join(ck, "heads.pt"), map_location="cpu")
    model = OuroSanSi(lm, letter_ids, hidden_size, n_loop_params=state["readout_config"]["n_loop_params"],
                      readout_rank=state["readout_config"]["rank"]).to(device)
    model.load_readout(state)
    return model, tok, cfg


def zero_shot_model(device, model_path):
    """A backbone as released: no LoRA, the readout at its start (the frozen LM head's letter logits). A non-looped
    backbone (--model) is run as one loop unless --loops asks for more."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sansi.model import OuroSanSi, option_token_ids
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    letter_ids, _ = option_token_ids(tok)
    lm = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
    lm.config.use_cache = False
    model = OuroSanSi(lm, letter_ids, lm.lm_head.in_features, n_loop_params=8, readout_rank=16).to(device)
    return model, tok, {"data": "datasets/main", "loops": 8 if model.looped() else 1, "eval_max_len": 9200}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", "--run-dir", dest="run", default=None,
                    help="a training run folder (omit with --zero-shot); use --run-dir under torchrun")
    ap.add_argument("--zero-shot", action="store_true", help="evaluate a backbone without any fine-tuning")
    ap.add_argument("--model", default=None, help="with --zero-shot: the backbone")
    ap.add_argument("--loops", type=int, default=0, help="loops to run and record (0 = as in training). More than "
                    "the run was trained with reads the extra loops with their untrained readouts (the frozen LM head); "
                    "beyond the readout's own loops (8) every further loop is read with the last loop's readout. A "
                    "non-looped backbone is run once per loop (OuroSanSi._repeat)")
    ap.add_argument("--data", default=None, help="the dataset folder (default: the one the run was trained on)")
    ap.add_argument("--step", type=int, default=1000)
    ap.add_argument("--split", default="test", choices=["dev", "test", "train"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--token-budget", type=int, default=32000)
    ap.add_argument("--limit", type=int, default=0, help="smoke tests only: the first N items of the split")
    ap.add_argument("--check", type=int, default=0, help="first re-evaluate this many dev items and compare with the "
                    "dev records written during training (0 = skip)")
    args = ap.parse_args()
    # several GPUs of one machine: torchrun --standalone --nproc_per_node=N -m eval.evaluate ...; items are split
    # over the ranks and the records gathered on rank 0, which writes the results
    world, rank = int(os.environ.get("WORLD_SIZE", "1")), int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        torch.distributed.init_process_group("nccl")
        torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    if args.zero_shot:
        assert args.model, "--zero-shot needs --model"
        model, tok, cfg = zero_shot_model(device, args.model)
        args.check = 0
    else:
        model, tok, cfg = load_model(args.run, args.step, device)
    loops = args.loops or cfg["loops"]
    assert loops >= 1, loops
    if args.check:
        ref = {r["fp"]: r for r in read_jsonl(os.path.join(args.run, "eval", "step_%d" % args.step, "records.jsonl.gz"))}
        dev = sorted((it for it in load_split(cfg["data"], "dev") if it["fingerprint"] in ref),
                     key=lambda it: it["fingerprint"])[:args.check]
        got = run_records(model, tok, dev, device, cfg["loops"], cfg["eval_max_len"], args.token_budget)
        # bf16 results depend slightly on how items are batched; a wrong load changes probabilities by far more.
        # Allowed: probabilities within 0.05, and a different last-loop answer only where the training-time
        # answer's top two options were within 0.1 of each other (a near tie).
        diff = max(abs(a - b) for r in got for pa, pb in zip(r["p"], ref[r["fp"]]["p"]) for a, b in zip(pa, pb))
        flips, clear_flips = 0, 0
        for r in got:
            q = ref[r["fp"]]["p"][-1]
            if r["p"][-1].index(max(r["p"][-1])) != q.index(max(q)):
                flips += 1
                top2 = sorted(q, reverse=True)[:2]
                clear_flips += (top2[0] - top2[1]) >= 0.1
        print("check on %d dev items: largest probability difference %.4f; different last-loop answer %d "
              "(%d not near ties)" % (len(got), diff, flips, clear_flips), flush=True)
        assert diff < 0.05 and clear_flips == 0, "the reloaded model does not reproduce the training-time evaluation"
    items = load_split(args.data or cfg["data"], args.split)
    assert items, "no items in %s/%s" % (args.data or cfg["data"], args.split)
    if args.limit:
        items = items[:args.limit]
    t0 = time.time()
    order = sorted(items, key=lambda it: (it["meta"]["tokens"], it["fingerprint"]))
    records = run_records(model, tok, order[rank::world], device, loops, cfg["eval_max_len"], args.token_budget)
    if world > 1:
        parts = [None] * world
        torch.distributed.all_gather_object(parts, records)
        records = [r for p in parts for r in p]
    if rank == 0:
        write_eval(records, args.out, "%s, %s, %d loops" % (
            "%s zero-shot" % os.path.basename(args.model.rstrip("/")) if args.zero_shot else
            "%s step %d" % (os.path.basename(args.run.rstrip("/")), args.step), args.split, loops))
        print("evaluated %d items in %.0f s -> %s" % (len(records), time.time() - t0, args.out), flush=True)
    if world > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
