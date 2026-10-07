"""Train SanSi or a single-pass baseline on a dataset folder built by data/ (default: the main set).

  python -m train.train --model <Ouro-1.4B> --loops 8 --seed 0 --out runs/sansi_s0

--loops (everything else is shared; for a given seed all three see the same items in the same order, with the same
option orders, and start from the same LoRA and readout initialisation):
  8  SanSi: 8 loops every step, answer loss (CE + Brier against the item's target) at every loop, 1/8 each.
  4  SanSi trained with 4 loops: 4 loops every step, answer loss at every loop, 1/4 each. The readouts of loops 5-8
     stay at their start, so a test run with 8 loops reads those loops through the frozen LM head.
  1  a single pass: 1 loop, answer loss at loop 1.
A non-looped backbone: --model may be a standard causal LM (e.g. SmolLM2-1.7B, Qwen3.5-4B-Base) with --loops 1. It
is run once, its last hidden state at the last prompt token is read exactly as loop 1 of the looped model is, and LoRA
goes on the same kinds of modules (every attention and MLP projection of every layer). Everything else is unchanged.
A loop added after pretraining: a standard causal LM with more than one loop (--loops 8 --model <SmolLM2-1.7B>). Its
layers are run once per loop, every loop after the first reading the previous loop's normed last hidden state in place
of the token embeddings (OuroSanSi._repeat: Ouro's own computation with a standard model's layers); loop 1 is the
model as released. Config: "looped" false, "added_loops" true.
Other data: --data datasets/depth_liars or datasets/depth_swaps (data.build_depth: program-generated items, depth
k = 1-8 in training, 1-16 in the test). --no-grad-ckpt keeps every activation instead of recomputing it in the
backward pass: the same gradients, faster, and the memory grows with tokens x layers x loops, so --token-budget has to
bound a micro-batch (short items only).
Targets: the gold option; the human label distribution (ChaosNLI); uniform (unanswerable items). "Correct" (logged
accuracy): the top option is the gold / majority option; for unanswerable items, top probability below (1 + 1/K) / 2.

Model: the backbone + LoRA (r 64, alpha 128, dropout 0.05) on its layers (Ouro-1.4B: the 24 shared layers); per loop,
letter logits from the frozen LM head plus a rank-16 correction and a temperature exp(a_t) (8 loops' parameters).
Optimiser: AdamW (0.9, 0.95), no weight decay, lr 1e-4 (LoRA) and 1e-3 (readout), 200 warm-up steps, cosine to 10%,
gradient clipping 1.0. A step is the next 16 items of the pooled training set (all items, reshuffled every pass, no
replacement), sorted by length and packed into micro-batches of at most --token-budget padded tokens; each item weighs
1/16 of the step's loss whatever its micro-batch. Gradient checkpointing (non-reentrant) unless --no-grad-ckpt.

Several GPUs of one machine (torchrun --standalone --nproc_per_node=N -m train.train ...): every rank draws the
same 16 items, takes its share (balanced by length), packs and back-propagates it; the gradients are summed over the
ranks (so a step is exactly the 16 items' gradient), then clipped and applied identically everywhere. Rank 0 writes the
logs, records and checkpoints; the dev evaluation is split over the ranks.

Writes to --out: config.json; log.jsonl (every 20 steps: loss, per-loop accuracy by item type and target type,
readout temperatures, learning rates, gradient norms, time, tokens, memory; plus the dev summaries);
train_records.jsonl.gz (one line per item and step); eval/step_N/ (dev records, metrics, summary) and ckpts/step_N/
(LoRA adapter + heads.pt, the readout) at every --eval-steps step.
"""
import argparse
import contextlib
import gzip
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import time
from collections import defaultdict

import torch


from eval.evaluate import load_split, run_records, write_eval
from sansi.loss import answer_loss
from sansi.model import OuroSanSi, encode_batch_hf, option_token_ids

MAX_LOOPS = 8
# LoRA goes on every attention and MLP projection. The first seven names are Ouro's (and the full-attention layers of
# Qwen3.5); the rest are the projections of Qwen3.5's linear-attention layers and match nothing in Ouro.
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LORA_TARGETS_LINEAR_ATTN = ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"]


def lora_targets(lm):
    """The LoRA target module names present in this backbone."""
    names = {n.split(".")[-1] for n, m in lm.named_modules() if isinstance(m, torch.nn.Linear)}
    return LORA_TARGETS + [n for n in LORA_TARGETS_LINEAR_ATTN if n in names]


class Pool:
    """All training items; each pass is a fresh shuffle, drawn without replacement."""

    def __init__(self, items, rng):
        self.items, self.rng, self.order, self.passes = items, rng, [], 0
        self.visits = defaultdict(int)

    def next(self):
        if not self.order:
            self.order = list(range(len(self.items)))
            self.rng.shuffle(self.order)
            self.passes += 1
        it = self.items[self.order.pop()]
        self.visits[it["fingerprint"]] += 1
        return it, self.visits[it["fingerprint"]]


def shuffled_options(it, rng):
    """A copy of a choice item with its options in random order; noul and score items keep theirs."""
    if it.get("qtype") != "choice":
        return it
    order = list(range(len(it["options"])))
    rng.shuffle(order)
    out = dict(it)
    out["options"] = [it["options"][j] for j in order]
    out["answer_distribution"] = [it["answer_distribution"][j] for j in order]
    out["answer_index"] = None if it["answer_index"] is None else order.index(it["answer_index"])
    return out


def pack(items, token_budget, overhead=512):
    """Items sorted by length, cut into consecutive micro-batches with rows x longest <= token_budget (a single item
    always fits), choosing the cuts that minimise padded tokens + overhead per micro-batch (dynamic programming).
    Only the grouping changes; which items make the step, and their weights, do not."""
    order = sorted(items, key=lambda x: x[0]["meta"]["tokens"])
    L = [x[0]["meta"]["tokens"] for x in order]
    n = len(order)
    best, cut = [0.0] + [float("inf")] * n, [0] * (n + 1)
    for j in range(1, n + 1):
        for i in range(j):
            rows = j - i
            if rows > 1 and rows * L[j - 1] > token_budget:
                continue
            c = best[i] + rows * L[j - 1] + overhead
            if c < best[j]:
                best[j], cut[j] = c, i
    out, j = [], n
    while j > 0:
        out.append(order[cut[j]:j])
        j = cut[j]
    return out[::-1]


def rank_share(items, world, rank):
    """This rank's part of the step's items: longest first, each to the rank with the fewest tokens so far."""
    if world == 1:
        return items
    load, mine = [0] * world, []
    for x in sorted(items, key=lambda x: (-x[0]["meta"]["tokens"], x[0]["fingerprint"])):
        r = min(range(world), key=lambda i: (load[i], i))
        load[r] += x[0]["meta"]["tokens"]
        if r == rank:
            mine.append(x)
    return mine


def gather(obj, world):
    if world == 1:
        return [obj]
    out = [None] * world
    torch.distributed.all_gather_object(out, obj)
    return out


def item_type(it):
    if it["answer_index"] is None:
        return "unans"
    return "gold" if it["deterministic"] else "soft"


def sha1_file(path):
    return hashlib.sha1(open(path, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loops", type=int, default=8, choices=[1, 4, 8], help="loops of every training step: "
                    "8 (SanSi), 4 (SanSi trained with 4 loops) or 1 (a single pass)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", required=True, help="the folder of the backbone (Ouro, or a standard causal LM)")
    ap.add_argument("--data", default="datasets/main")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--items-per-step", type=int, default=16)
    ap.add_argument("--token-budget", type=int, default=16384, help="padded tokens per training micro-batch")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--readout-lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--brier", type=float, default=1.0)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=128)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--readout-rank", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=5600, help="training items are never cut; longer ones abort")
    ap.add_argument("--eval-max-len", type=int, default=9200)
    ap.add_argument("--eval-steps", default="500,1000")
    ap.add_argument("--save-steps", default="", help="steps that save a checkpoint without a dev evaluation (the "
                    "eval steps and the last step always save one)")
    ap.add_argument("--eval-token-budget", type=int, default=32000)
    ap.add_argument("--dev-limit", type=int, default=0, help="smoke tests only: evaluate this many dev items")
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--no-grad-ckpt", action="store_true", help="no gradient checkpointing: same gradients, faster, "
                    "more memory (short items only; bound a micro-batch with --token-budget)")
    args = ap.parse_args()
    loops = args.loops
    eval_steps = {int(s) for s in args.eval_steps.split(",") if s}
    save_steps = {int(s) for s in args.save_steps.split(",") if s} | eval_steps

    world, rank = int(os.environ.get("WORLD_SIZE", "1")), int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        torch.distributed.init_process_group("nccl")
        torch.cuda.set_device(local)
    main = rank == 0
    say = (lambda *a, **k: print(*a, **k)) if main else (lambda *a, **k: None)
    if main:
        os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    data_rng = random.Random(args.seed)            # which items, in which order, with which option order
    device = torch.device("cuda", local)

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    letter_ids, _ = option_token_ids(tok)
    lm = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, torch_dtype=torch.bfloat16)
    lm.config.use_cache = False
    if not args.no_grad_ckpt:
        lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    hidden_size = lm.lm_head.in_features
    targets = lora_targets(lm)
    lm = get_peft_model(lm, LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                                       target_modules=targets))
    model = OuroSanSi(lm, letter_ids, hidden_size, n_loop_params=MAX_LOOPS,
                      readout_rank=args.readout_rank).to(device)
    model.train()
    fwd = model
    if world > 1:
        from torch.nn.parallel import DistributedDataParallel
        fwd = DistributedDataParallel(model, device_ids=[local])
    lora = [p for n, p in model.lm.named_parameters() if p.requires_grad]
    readout = list(model.readout.parameters())
    n_train = sum(p.numel() for p in lora + readout)
    assert n_train == sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW([{"params": lora, "lr": args.lr, "base_lr": args.lr},
                             {"params": readout, "lr": args.readout_lr, "base_lr": args.readout_lr}],
                            betas=(0.9, 0.95), weight_decay=0.0)

    train_items = load_split(args.data, "train")
    dev_items = load_split(args.data, "dev")
    if args.dev_limit:
        dev_items = random.Random(0).sample(dev_items, args.dev_limit)
    assert max(it["meta"]["tokens"] for it in train_items) + 60 <= args.max_len
    pool = Pool(train_items, data_rng)

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                                text=True).stdout.strip())
    cfg = dict(vars(args), n_loop_params=MAX_LOOPS,
               save_steps=sorted(save_steps),
               trainable=n_train, trainable_lora=sum(p.numel() for p in lora),
               trainable_readout=sum(p.numel() for p in readout),
               looped=model.looped(), added_loops=(not model.looped()) and loops > 1,
               lora_targets=targets,
               backbone_params=sum(p.numel() for p in model.lm.parameters()),
               letter_ids=letter_ids, git_commit=commit, git_dirty=dirty, n_train=len(train_items), n_dev=len(dev_items),
               manifest_sha1=sha1_file(os.path.join(args.data, "manifest.json")),
               host=os.uname().nodename, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
               torch=torch.__version__, world_size=world)
    if main:
        with open(os.path.join(args.out, "config.json"), "w") as f:
            json.dump(cfg, f, indent=1)
    say("seed %d loops %d; trainable %.1fM; train %d items, dev %d; "
        "commit %s%s" % (args.seed, loops, n_train / 1e6, len(train_items),
                         len(dev_items), commit, " (dirty)" if dirty else ""), flush=True)

    def lr_at(step, base):
        if step < args.warmup:
            return base * step / args.warmup
        prog = (step - args.warmup) / max(1, args.steps - args.warmup)
        return base * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))

    def gnorm(params):
        sq = [p.grad.float().pow(2).sum() for p in params if p.grad is not None]
        return math.sqrt(sum(x.item() for x in sq)) if sq else 0.0

    log = open(os.path.join(args.out, "log.jsonl"), "a") if main else None
    rec_path = os.path.join(args.out, "train_records.jsonl")
    rec_f = open(rec_path, "a") if main else None

    def new_window():
        return {"n": 0, "loss": 0.0, "steps": 0, "sec": 0.0, "tokens": 0, "padded": 0,
                "micro": 0, "clipped": 0, "g_total": 0.0, "g_lora": 0.0, "g_readout": 0.0,
                "loss_t": [0.0] * loops, "corr_t": [0.0] * loops,
                "by_layer": defaultdict(lambda: [[0.0] * loops, 0]), "by_type": defaultdict(lambda: [[0.0] * loops, 0]),
                "first_visits": 0}

    win = new_window()
    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    for step in range(1, args.steps + 1):
        t0 = time.time()
        for g in opt.param_groups:
            g["lr"] = lr_at(step, g["base_lr"])
        drawn = []
        for _ in range(args.items_per_step):
            it, visit = pool.next()
            drawn.append((shuffled_options(it, data_rng), visit))
        micros = pack(rank_share(drawn, world, rank), args.token_budget)
        part = torch.zeros(4, device=device)          # loss, tokens, padded, micro-batches (this rank)
        rows = []
        for mi, micro in enumerate(micros):
            items = [x[0] for x in micro]
            batch, kept = encode_batch_hf(items, tok, args.max_len)
            assert len(kept) == len(items), "a training item is longer than --max-len"
            b = {k: v.to(device) for k, v in batch.items()}
            # one gradient all-reduce per step: on this rank's last micro-batch
            sync = contextlib.nullcontext() if (world == 1 or mi == len(micros) - 1) else fwd.no_sync()
            with sync:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outs = fwd(b["ids"], b["mask"], b["last"], b["n_options"], loops=loops)
                loss, det = answer_loss(outs, b["dist"], b["y"], b["n_options"], brier_weight=args.brier)
                share = len(items) / args.items_per_step      # every item weighs 1/16 of the step's loss
                (loss * share * world).backward()              # DDP averages over ranks; x world makes it the sum
            part += torch.tensor([loss.item() * share,
                                  float(b["mask"].sum().item()), float(b["mask"].numel()), 1.0], device=device)
            L, C, M = det["loss_by_loop"].cpu(), det["correct"].float().cpu(), det["maxp"].cpu()
            for j, (it, visit) in enumerate(micro):
                row = {"step": step, "fp": it["fingerprint"], "source": it["meta"]["source_set"],
                       "layer": it["meta"]["layer"], "type": item_type(it), "visit": visit,
                       "tokens": it["meta"]["tokens"], "c": [int(C[t, j].item()) for t in range(loops)],
                       "maxp": [round(M[t, j].item(), 4) for t in range(loops)],
                       "loss": [round(L[t, j].item(), 4) for t in range(loops)]}
                rows.append(row)
        if world > 1:
            torch.distributed.all_reduce(part)
        step_loss = part[0].item()
        assert math.isfinite(step_loss), "non-finite loss at step %d" % step
        g_lora, g_ro = gnorm(lora), gnorm(readout)
        total = torch.nn.utils.clip_grad_norm_(lora + readout, args.clip).item()
        assert math.isfinite(total), "non-finite gradient norm at step %d" % step
        opt.step()
        opt.zero_grad(set_to_none=True)
        all_rows = [r for rs in gather(rows, world) for r in rs]
        if main:
            for row in all_rows:
                win["n"] += 1
                win["first_visits"] += row["visit"] == 1
                for t in range(loops):
                    win["loss_t"][t] += row["loss"][t]
                    win["corr_t"][t] += row["c"][t]
                for key, name in (("by_layer", row["layer"]), ("by_type", row["type"])):
                    acc = win[key][name]
                    for t in range(loops):
                        acc[0][t] += row["c"][t]
                    acc[1] += 1
                rec_f.write(json.dumps(row) + "\n")
        win["steps"] += 1
        win["loss"] += step_loss
        win["tokens"] += int(part[1].item())
        win["padded"] += int(part[2].item())
        win["micro"] += int(part[3].item())
        win["clipped"] += total > args.clip
        win["g_total"] += total
        win["g_lora"] += g_lora
        win["g_readout"] += g_ro
        win["sec"] += time.time() - t0

        if main and (step % args.log_every == 0 or step == args.steps):
            s, n = win["steps"], win["n"]
            rec = {"step": step, "loss": round(win["loss"] / s, 4),
                   "answer_loss_by_loop": [round(x / n, 4) for x in win["loss_t"]],
                   "acc_by_loop": [round(x / n, 4) for x in win["corr_t"]],
                   "acc_by_layer": {k: {"n": v[1], "acc": [round(x / v[1], 3) for x in v[0]]}
                                    for k, v in sorted(win["by_layer"].items())},
                   "acc_by_type": {k: {"n": v[1], "acc": [round(x / v[1], 3) for x in v[0]]}
                                   for k, v in sorted(win["by_type"].items())},
                   "lr": round(lr_at(step, args.lr), 8), "readout_lr": round(lr_at(step, args.readout_lr), 8),
                   "grad_norm": round(win["g_total"] / s, 4), "grad_norm_lora": round(win["g_lora"] / s, 4),
                   "grad_norm_readout": round(win["g_readout"] / s, 4),
                   "clipped_share": round(win["clipped"] / s, 3),
                   "temperature": [round(math.exp(max(-3.0, min(3.0, a))), 4) for a in model.readout.a.tolist()],
                   "readout_U_norm": [round(x, 4) for x in model.readout.U.norm(dim=(1, 2)).tolist()],
                   "sec_per_step": round(win["sec"] / s, 2), "tokens_per_step": round(win["tokens"] / s),
                   "padding_share": round(1 - win["tokens"] / max(1, win["padded"]), 3),
                   "micro_per_step": round(win["micro"] / s, 2),
                   "peak_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 1),
                   "first_visit_share": round(win["first_visits"] / n, 3), "passes": pool.passes,
                   "elapsed_min": round((time.time() - t_start) / 60, 1), "world_size": world}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            rec_f.flush()
            say("step %d loss %.4f acc t1 %.3f t%d %.3f | %.1f s/step %d tok/step peak %.1f GB"
                % (step, rec["loss"],
                   rec["acc_by_loop"][0], loops, rec["acc_by_loop"][-1], rec["sec_per_step"], rec["tokens_per_step"],
                   rec["peak_gb"]), flush=True)
        if step % args.log_every == 0 or step == args.steps:
            win = new_window()
            torch.cuda.reset_peak_memory_stats()

        if step in save_steps or step == args.steps:
            ck = os.path.join(args.out, "ckpts", "step_%d" % step)
            if main:
                model.lm.save_pretrained(ck)
                torch.save(model.readout_state(), os.path.join(ck, "heads.pt"))
                torch.save({"step": step, "optimizer": opt.state_dict(), "pool_order": pool.order,
                            "pool_passes": pool.passes, "pool_visits": dict(pool.visits),
                            "data_rng": data_rng.getstate(),
                            "torch_rng": torch.get_rng_state(),
                            "cuda_rng": torch.cuda.get_rng_state(), "world_size": world},
                           os.path.join(ck, "train_state.pt"))       # the optimiser, the item pool and the generators
        if step in eval_steps or step == args.steps:
            t_ev = time.time()
            order = sorted(dev_items, key=lambda it: (it["meta"]["tokens"], it["fingerprint"]))
            recs = run_records(model, tok, order[rank::world], device, loops, args.eval_max_len, args.eval_token_budget)
            recs = [r for rs in gather(recs, world) for r in rs]
            if main:
                metrics = write_eval(recs, os.path.join(args.out, "eval", "step_%d" % step),
                                     "%s step %d, dev" % (os.path.basename(args.out.rstrip("/")), step))
                sec = time.time() - t_ev
                a = metrics["all"]
                ev = {"step": step, "dev_sec": round(sec), "dev_n": a["n"],
                      "dev_acc_by_loop": {t: round(v["accuracy"], 4) for t, v in a["by_loop"].items()},
                      "dev_ece_by_loop": {t: round(v["ece"], 4) for t, v in a["by_loop"].items()},
                      "dev_acc_by_tier_last": {k[5:]: round(m["by_loop"][loops]["accuracy"], 4)
                                               for k, m in metrics.items() if k.startswith("tier/")}}
                log.write(json.dumps({"dev": ev}) + "\n")
                log.flush()
                say("dev@%d (%d items, %.0f s): acc %s | ECE %s" % (
                    step, a["n"], sec, " ".join("t%s %.3f" % (t, v) for t, v in ev["dev_acc_by_loop"].items()),
                    " ".join("t%s %.3f" % (t, v) for t, v in ev["dev_ece_by_loop"].items())), flush=True)
            if world > 1:
                torch.distributed.barrier()
            torch.cuda.reset_peak_memory_stats()
    if world > 1:                                     # every rank must hold the same parameters
        chk = torch.tensor([sum(p.double().sum().item() for p in lora + readout)], device=device,
                           dtype=torch.float64)
        allc = [torch.zeros_like(chk) for _ in range(world)]
        torch.distributed.all_gather(allc, chk)
        say("parameter checksum per rank:", [round(x.item(), 6) for x in allc], flush=True)
        assert all(abs(x.item() - allc[0].item()) < 1e-6 * max(1.0, abs(allc[0].item())) for x in allc)
    if main:
        log.close()
        rec_f.close()
        with open(rec_path, "rb") as fi, gzip.open(rec_path + ".gz", "wb") as fo:
            shutil.copyfileobj(fi, fo)
        os.remove(rec_path)
        with open(os.path.join(args.out, "DONE"), "w") as f:
            f.write("finished %s after %.1f min\n" % (time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                                                    (time.time() - t_start) / 60))
    say("done in %.1f min" % ((time.time() - t_start) / 60), flush=True)
    if world > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
