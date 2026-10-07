"""The training objective: cross-entropy plus the Brier score against the item's target distribution, at every loop."""
import torch
import torch.nn.functional as F


def target_losses(logits_list, dist, brier_weight=1.0):
    """Per item and loop: cross-entropy + brier_weight * Brier against a target distribution over the options (the
    gold option, a human label distribution, or uniform for an unanswerable item). Options beyond the item's own are
    masked with -inf in the logits and 0 in dist, and add nothing. Returns [T, B]."""
    dist = dist.float()
    out = []
    for logits in logits_list:
        logits = logits.float()
        valid = torch.isfinite(logits)
        logp = F.log_softmax(logits, dim=-1)
        p = logp.exp().masked_fill(~valid, 0.0)
        ce = -(dist * logp.masked_fill(~valid, 0.0)).sum(-1)
        brier = ((p - dist) ** 2).sum(-1)
        out.append(ce + brier_weight * brier)
    return torch.stack(out)


def answered_correctly(logits_list, y, n_options):
    """[T, B] bool. y >= 0: the top option is y (the gold option, or the majority label of a soft item).
    y = -1 (unanswerable): not a hard answer, i.e. the top probability is below (1 + 1/K) / 2, halfway between
    no idea (1/K) and certain (1), K = the item's number of options."""
    thr = (1.0 + 1.0 / n_options.float()) / 2.0
    out = []
    for logits in logits_list:
        maxp, pred = F.softmax(logits.float(), dim=-1).max(-1)
        out.append(torch.where(y >= 0, pred == y, maxp < thr))
    return torch.stack(out)


def answer_weights(T):
    """Weights of the T loops in the answer loss, summing to 1: every loop 1/T."""
    return torch.full((T,), 1.0 / T)


def answer_loss(logits_list, dist, y, n_options, brier_weight=1.0):
    """The objective for one pass of T loops (T = 8 or 4, or 1 for a single pass): CE + Brier against the item's
    target distribution at every loop, weighted by answer_weights(T).

    Returns the loss and a dict of detached tensors for logging: per-item loss and correctness at every loop, and the
    top probability."""
    T = len(logits_list)
    L = target_losses(logits_list, dist, brier_weight)                     # [T, B]
    w = answer_weights(T).to(L.device)
    answer = (w[:, None] * L).sum(0).mean()
    with torch.no_grad():
        C = answered_correctly(logits_list, y, n_options)                  # [T, B]
        maxp = torch.stack([F.softmax(l.float(), dim=-1).max(-1).values for l in logits_list])
    det = {"loss_by_loop": L.detach(), "correct": C, "maxp": maxp}
    return answer, det
