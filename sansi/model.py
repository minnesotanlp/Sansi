"""SanSi on top of Ouro-1.4B (ByteDance, Apache-2.0), a natively looped language model.

Ouro applies its 24 shared layers `total_ut_steps` times and returns the normed hidden
state after every loop. We read a typed decision at the last prompt token after every
loop: the pretrained LM head restricted to the option letters, plus a small trained readout.

  prompt:  <state>\\n\\nQuestion: <question>\\nOptions: (A) yes (B) no\\nAnswer:
  readout: logits of the tokens " A", " B", ... at the last position, every loop

The readout (CoupledReadout) adds, per loop, a low-rank correction to those logits and a temperature; both start as
the identity, so a fresh model computes exactly what the frozen LM head does.

A standard causal LM (SmolLM2, Qwen3.5) can take the place of Ouro: it is run once per loop (OuroSanSi._repeat) and
read in the same way.

After every forward, last_gates holds Ouro's own gate output per loop at the readout position (sigmoid of its
pretrained linear gate), which the test pass writes to the records; it takes no part in the decision.
"""
import torch
import torch.nn as nn

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"    # option labels; each " X" is a single Ouro token


def option_token_ids(tok):
    """One token id per option letter. Prefers the space-prefixed form (' A') that follows 'Answer:'."""
    for prefix in (" ", ""):
        ids = [tok.encode(prefix + c, add_special_tokens=False) for c in LETTERS]
        if all(len(i) == 1 for i in ids):
            return [i[0] for i in ids], prefix
    raise ValueError("option letters are not single tokens")


def render_prompt(item):
    opts = " ".join(f"({c}) {o}" for c, o in zip(LETTERS, item["options"]))
    return f"{item['state']}\n\nQuestion: {item['question']}\nOptions: {opts}\nAnswer:"


def encode_batch_hf(items, tok, max_len):
    """Right-padded batch; the readout position is the last real token of each row."""
    enc = [tok.encode(render_prompt(it), add_special_tokens=False) for it in items]
    kept = [(it, ids) for it, ids in zip(items, enc) if len(ids) <= max_len]
    L = max(len(ids) for _, ids in kept)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    ids_t = torch.full((len(kept), L), pad_id, dtype=torch.long)
    mask = torch.zeros(len(kept), L, dtype=torch.long)
    last = torch.zeros(len(kept), dtype=torch.long)
    for i, (_, ids) in enumerate(kept):
        ids_t[i, : len(ids)] = torch.tensor(ids)
        mask[i, : len(ids)] = 1
        last[i] = len(ids) - 1
    batch = {
        "ids": ids_t, "mask": mask, "last": last,
        "n_options": torch.tensor([len(it["options"]) for it, _ in kept]),
        "y": torch.tensor([-1 if it.get("answer_index") is None else it["answer_index"] for it, _ in kept]),
                                                                      # -1: no answer_index (unanswerable item)
        "unans": torch.tensor([bool((it.get("meta") or {}).get("unanswerable")) for it, _ in kept]),
        "k": torch.tensor([-1 if it["k"] is None else it["k"] for it, _ in kept]),   # -1: no depth label
        "dist": torch.tensor([it["answer_distribution"] + [0.0] * (len(LETTERS) - len(it["answer_distribution"]))
                              for it, _ in kept]),
        "deterministic": torch.tensor([it["deterministic"] for it, _ in kept]),
    }
    return batch, [it for it, _ in kept]


class CoupledReadout(nn.Module):
    """A per-loop readout on top of the frozen LM head.

      logits_t = (W z_t + U_t V_t z_t) / tau_t,    tau_t = exp(a_t + b * d_t)

    W z_t are the plain letter logits (frozen LM head), U_t V_t a per-loop low-rank correction, a_t a per-loop
    log-temperature. d_t is a per-item input that OuroSanSi always passes as 0, so tau_t = exp(a_t) and b never
    leaves its initial value 0; b and d are kept because existing checkpoints hold b and the readout is computed
    exactly as it was when they were trained. d_t is detached.
    U, a and b start at zero, so a fresh readout is the identity. Loops beyond n_loop_params (never trained when
    training runs at most that many) reuse the last loop's parameters."""

    def __init__(self, hidden_size, n_letters, n_loop_params=6, rank=16):
        super().__init__()
        self.n_loop_params, self.rank = n_loop_params, rank
        self.a = nn.Parameter(torch.zeros(n_loop_params))
        self.b = nn.Parameter(torch.zeros(()))
        self.V = nn.Parameter(torch.randn(n_loop_params, rank, hidden_size) / hidden_size ** 0.5)
        self.U = nn.Parameter(torch.zeros(n_loop_params, n_letters, rank))

    def config(self):
        return {"n_loop_params": self.n_loop_params, "rank": self.rank}

    def temperature(self, d, t):
        i = min(t, self.n_loop_params - 1)
        return torch.exp((self.a[i] + self.b * d.detach().float()).clamp(-3.0, 3.0))

    def forward(self, z, logits, d, t):
        i = min(t, self.n_loop_params - 1)
        corr = (z.float() @ self.V[i].t()) @ self.U[i].t()
        return (logits + corr.float()) / self.temperature(d, t)[:, None]


def _mlp(d_in, prior):
    """Two-layer network whose output starts at the constant `prior` (a logit or a value; a list gives one output per
    entry)."""
    priors = list(prior) if isinstance(prior, (list, tuple)) else [prior]
    m = nn.Sequential(nn.Linear(d_in, 256), nn.GELU(), nn.Linear(256, len(priors)))
    nn.init.zeros_(m[-1].weight)
    with torch.no_grad():
        m[-1].bias.copy_(torch.tensor(priors, dtype=m[-1].bias.dtype))
    return m


class OuroSanSi(nn.Module):
    """Wraps an OuroForCausalLM, or a standard causal LM (possibly LoRA-wrapped by peft), with per-loop typed readouts.

    forward returns, per loop, the option logits [B, n_letters]."""

    def __init__(self, causal_lm, letter_ids, hidden_size, n_loop_params=6, readout_rank=16):
        super().__init__()
        self.lm = causal_lm
        self.register_buffer("letter_ids", torch.tensor(letter_ids), persistent=False)
        # Two small networks are initialised here and discarded; they are not part of the model. The models of the
        # paper were trained by code that initialised two networks of these sizes at this point, so drawing the same
        # random numbers keeps the initialisation of the readout below (its V) the same for a given seed.
        _mlp(hidden_size, 0.0)
        _mlp(hidden_size + n_loop_params, 0.0)
        self.readout = CoupledReadout(hidden_size, len(letter_ids), n_loop_params, readout_rank)

    def readout_state(self):
        """Everything trained besides the LoRA adapter, with what is needed to rebuild it."""
        return {"readout": self.readout.state_dict(), "readout_config": self.readout.config()}

    def load_readout(self, state):
        self.readout.load_state_dict(state["readout"])

    def _base(self):
        """The causal LM under any peft wrapper (the LoRA layers stay in place): OuroForCausalLM, or a standard,
        non-looped causal LM, which forward runs once per loop (_repeat)."""
        m = self.lm
        if hasattr(m, "get_base_model"):
            m = m.get_base_model()
        assert hasattr(m, "lm_head") and hasattr(m, "model"), type(m)
        return m

    def looped(self):
        """Whether the backbone itself applies its layers repeatedly (Ouro) or once (a standard causal LM; with more
        than one loop, forward then repeats it: a loop added after pretraining)."""
        return hasattr(self._base().model, "total_ut_steps")

    def _repeat(self, base, ids, mask, loops):
        """`loops` passes of a backbone that is called once per loop: the first pass reads the token embeddings, every
        later pass reads the previous pass's last hidden state (after the model's own final norm, i.e. what its LM
        head reads) in their place; same positions and attention mask every pass, and the token embeddings are not
        fed in again. This is the computation of OuroModel.forward, embeddings -> [layers -> norm] x loops, with a
        standard model's own layers and norm; one loop is the model exactly as released. Returns every pass's normed
        states."""
        hs_list, h = [], None
        for t in range(loops):
            if h is None:
                kw = {"input_ids": ids}
            else:
                kw = {"inputs_embeds": h}
            out = base.model(attention_mask=mask, use_cache=False, **kw)
            h = out.last_hidden_state
            hs_list.append(h)
        return hs_list

    def read(self, z, t, n_options):
        """One loop's readout of the normed states z [B, d] at the readout position, as loop t (0-based): the masked
        option logits [B, n_letters]."""
        base = self._base()
        logits = base.lm_head(z)[:, self.letter_ids].float()
        zf = z.float()
        d = torch.zeros(zf.size(0), device=zf.device, dtype=zf.dtype)   # the readout's per-item input: always 0
        logits = self.readout(z, logits, d, t)
        opt_mask = torch.arange(logits.size(1), device=z.device)[None, :] >= n_options[:, None]
        return logits.masked_fill(opt_mask, float("-inf"))

    record_process = False  # evaluation sets this: forward then keeps the process of every loop (see below)

    def forward(self, ids, mask, last, n_options, loops=4):
        base = self._base()
        if self.looped():
            base.model.total_ut_steps = loops
            _, hs_list, gate_list = base.model(input_ids=ids, attention_mask=mask, use_cache=False)
        else:
            # a non-looped backbone: one call per loop (_repeat); the state after its own final norm is the state of
            # the loop, read exactly as a loop of the looped model is, and there is no gate output (reported as 0)
            hs_list = self._repeat(base, ids, mask, loops)
            gate_list = [hs.new_full(hs.shape[:2] + (1,), float("-inf")) for hs in hs_list]
        rows = torch.arange(ids.size(0), device=ids.device)
        self.last_gates = [torch.sigmoid(g[rows, last].float().squeeze(-1)).detach() for g in gate_list]
        outs = []
        for t, hs in enumerate(hs_list):
            outs.append(self.read(hs[rows, last], t, n_options))    # normed state at the last token
        if self.record_process:
            # the computation's own reading at every loop, untouched by training: the frozen LM head's option-letter
            # logits (last_raw, [B, n_letters] per loop, other letters -inf), and how much the readout vector moved
            # from the previous loop relative to its length (last_dz, [B] per loop from loop 2)
            with torch.no_grad():
                W = base.lm_head.weight[self.letter_ids].float()
                zs = [hs[rows, last].float() for hs in hs_list]
                opt_mask = torch.arange(W.size(0), device=ids.device)[None, :] >= n_options[:, None]
                self.last_raw = [(z @ W.t()).masked_fill(opt_mask, float("-inf")) for z in zs]
                self.last_dz = [(b - a).norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-6) for a, b in zip(zs, zs[1:])]
        return outs
