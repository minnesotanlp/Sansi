"""Load a released SanSi model (Hugging Face Hub: minnesotanlp/SanSi, minnesotanlp/SanSi-2.6B) and ask it typed
decisions.

  from sansi.hub import load, decide
  model, tok = load("minnesotanlp/SanSi")
  probs = decide(model, tok, state="...", question="...", options=["yes", "no"])
  probs[-1]   # the option probabilities after the last loop; probs[t] after loop t + 1

A released folder holds the LoRA adapter as written by peft (adapter_config.json, adapter_model.safetensors), the
readouts (readout.safetensors) and sansi_config.json: the backbone and its revision on the Hub, the number of loops
the model was trained with, and the shape of the readout. load downloads the backbone at that revision unless
`backbone` names a local folder of it.
"""
import json
import os

import torch

from .model import LETTERS, OuroSanSi, encode_batch_hf, option_token_ids


def load(repo_or_dir, device="cuda", revision=None, backbone=None):
    """The model and its tokenizer. repo_or_dir: a Hub repository or a local folder in the released format;
    revision: a revision of that repository; backbone: a local folder of the backbone instead of the Hub."""
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = repo_or_dir if os.path.isdir(repo_or_dir) else snapshot_download(repo_or_dir, revision=revision)
    cfg = json.load(open(os.path.join(path, "sansi_config.json")))
    src = {"pretrained_model_name_or_path": backbone} if backbone else \
        {"pretrained_model_name_or_path": cfg["backbone"], "revision": cfg["backbone_revision"]}
    tok = AutoTokenizer.from_pretrained(trust_remote_code=True, **src)
    lm = AutoModelForCausalLM.from_pretrained(trust_remote_code=True, torch_dtype=torch.bfloat16, **src)
    lm.config.use_cache = False
    hidden_size = lm.lm_head.in_features
    letter_ids, _ = option_token_ids(tok)
    lm = PeftModel.from_pretrained(lm, path)
    model = OuroSanSi(lm, letter_ids, hidden_size, n_loop_params=cfg["n_loop_params"],
                      readout_rank=cfg["readout_rank"])
    model.load_readout({"readout": load_file(os.path.join(path, "readout.safetensors"))})
    model.loops = cfg["loops"]
    return model.to(device).eval(), tok


@torch.no_grad()
def decide(model, tok, state, question, options, loops=None):
    """The option probabilities after every loop for one decision: a list with one list per loop, each with one
    probability per option, in the order given. options: 2 to 26 strings. loops: how many loops to run (default: as
    trained; more loops than trained are read with the last trained readout, see eval/README.md)."""
    assert 2 <= len(options) <= len(LETTERS), "between 2 and %d options" % len(LETTERS)
    n = len(options)
    item = {"state": state, "question": question, "options": list(options), "answer_index": None, "k": None,
            "answer_distribution": [1.0 / n] * n, "deterministic": False, "meta": {}}
    batch, _ = encode_batch_hf([item], tok, max_len=10 ** 9)
    device = next(model.parameters()).device
    b = {k: batch[k].to(device) for k in ("ids", "mask", "last", "n_options")}
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        outs = model(b["ids"], b["mask"], b["last"], b["n_options"], loops=loops or model.loops)
    return [torch.softmax(o.float(), -1)[0, :n].tolist() for o in outs]
