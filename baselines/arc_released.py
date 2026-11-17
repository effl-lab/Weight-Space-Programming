"""ARC test accuracy of released small language models, zero-shot and after fine-tuning on ARC train.

Every model sees one prompt: the question, options "A. ..." to "D. ..." and an instruction to answer with
the letter, through its chat template (thinking disabled), or "Question: ...\\nA. ...\\nAnswer:" when it has
none. The prediction is the argmax over the four letters of the next-token logit, each letter taking the
larger of its "A" and " A" tokens. Prompts are right-padded and read at their last real token, which is
exact for causal attention and for recurrent models alike. A masked-LM encoder (MASKED_LM) reads the prompt
followed by its mask token, with the padding masked out, and is read at the mask token through its
prediction head.

Fine-tuning minimizes cross-entropy over those four letter logits. Transformers of LORA_MIN_PARAMS or more
train LoRA adapters on every decoder linear layer; smaller models and the recurrent ones train all weights.
Each learning rate in the grid runs MAX_EPOCHS epochs with validation every half epoch; the (learning rate,
half epoch) with the best validation accuracy is reported with its test accuracy.

With --from-init the same procedure trains the architecture from the random initialisation the weight-programming
runs of lm-<name>-init start from (wsp.lm_weights.hf_random_model with lm_start.INIT_SEED, rounded to bfloat16 as
stored), with that start's tokenizer and the plain prompt those runs read; the prompt ids must hash to the start's
manifest. All weights train, over INIT_LRS; results go to OUT_DIR_INIT.

A finished model (its JSON exists) is skipped; within a model, finished learning rates are kept in
logs/arc_released/<model>.partial.json.
"""
import argparse
import hashlib
import json
import math
import platform
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer

from wsp import arc, lm_start, lm_weights

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs/experiments/arc_released"
OUT_DIR_NTRAIN = ROOT / "docs/experiments/arc_released_ntrain"
OUT_DIR_INIT = ROOT / "docs/experiments/arc_init_ntrain"
PARTIAL_DIR = ROOT / "logs/arc_released"
MODELS = {
    "SmolLM2-135M-Instruct": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "SmolLM2-360M-Instruct": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "gemma-3-270m-it": "google/gemma-3-270m-it",
    "gemma-3-1b-it": "google/gemma-3-1b-it",
    "gemma-2-2b-it": "google/gemma-2-2b-it",
    "Qwen3-0.6B": "Qwen/Qwen3-0.6B",
    "Qwen3-1.7B": "Qwen/Qwen3-1.7B",
    "Qwen3.5-0.8B": "Qwen/Qwen3.5-0.8B",
    "Qwen3.5-2B": "Qwen/Qwen3.5-2B",
    "mamba2-130m-hf": "AntonV/mamba2-130m-hf",
    "rwkv7-0.1B-g1": "fla-hub/rwkv7-0.1B-g1",
    "ModernBERT-base": "answerdotai/ModernBERT-base",
    "ModernBERT-large": "answerdotai/ModernBERT-large",
    "UltraBERT-150M": "dmis-lab/UltraBERT-150M",
    "UltraBERT-380M": "dmis-lab/UltraBERT-380M",
}
RECURRENT = {"mamba2-130m-hf", "rwkv7-0.1B-g1"}
REMOTE_CODE = {"rwkv7-0.1B-g1", "UltraBERT-150M", "UltraBERT-380M"}
MASKED_LM = {"ModernBERT-base", "ModernBERT-large", "UltraBERT-150M", "UltraBERT-380M"}
INIT_OF = {
    "SmolLM2-135M-Instruct": "smollm2_135m", "SmolLM2-360M-Instruct": "smollm2_360m", "gemma-3-270m-it": "gemma3_270m",
    "gemma-3-1b-it": "gemma3_1b", "gemma-2-2b-it": "gemma2_2b", "Qwen3-0.6B": "qwen3_0_6b", "Qwen3-1.7B": "qwen3_1_7b",
    "Qwen3.5-0.8B": "qwen3_5_0_8b", "Qwen3.5-2B": "qwen3_5_2b", "mamba2-130m-hf": "mamba2_130m",
    "rwkv7-0.1B-g1": "rwkv7_0_1b", "ModernBERT-base": "modernbert_base", "ModernBERT-large": "modernbert_large",
    "UltraBERT-150M": "ultrabert_150m", "UltraBERT-380M": "ultrabert_380m",
}
LORA_MIN_PARAMS = 5e8
LORA = dict(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear")
LORA_LRS = (1e-4, 3e-4, 1e-3)
FULL_LRS = (1e-5, 3e-5, 1e-4)
INIT_LRS = (1e-4, 3e-4, 1e-3)
MAX_EPOCHS = 3
SUBSET_EVAL_STEPS = 30
BATCH = 16
EVAL_BATCH = 32
WARMUP_STEPS = 30
SEED = 0
INSTRUCTION = "Answer with the letter of the correct option (A, B, C, or D)."
PLAIN_TEMPLATE = "Question: {question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer:"
CHAT_TEMPLATE = "{question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n" + INSTRUCTION


def load(name: str, dtype: torch.dtype, device: str, from_init: bool = False):
    """(model, body, output layer); the body of a masked LM is the whole model (see letter_logits). from_init builds
    the lm-<name>-init start; a vision tower its class carries is frozen, since the prompts never reach it."""
    masked = name in MASKED_LM
    if from_init:
        model = lm_weights.hf_random_model(INIT_OF[name], lm_start.INIT_SEED).to(torch.bfloat16).to(dtype)
        for n, p in model.named_parameters():
            p.requires_grad_("visual" not in n)
    else:
        model = (AutoModelForMaskedLM if masked else AutoModelForCausalLM).from_pretrained(
            MODELS[name], dtype=dtype, trust_remote_code=name in REMOTE_CODE)
    model = model.to(device)
    return model, model if masked else model.base_model, model.get_output_embeddings()


def tokenizer(name: str, from_init: bool = False):
    if from_init:
        folder = lm_start.start_dir(lm_start.arch(INIT_OF[name], "init")) / lm_start.TOKENIZER
        return AutoTokenizer.from_pretrained(folder, trust_remote_code=any(folder.glob("*.py")))
    return AutoTokenizer.from_pretrained(MODELS[name], trust_remote_code=name in REMOTE_CODE)


def mask_id(name: str, tok) -> int | None:
    return tok.mask_token_id if name in MASKED_LM else None


def plain_prompt(question: str, options: list[str]) -> str:
    return PLAIN_TEMPLATE.format(question=question.strip(), **dict(zip(arc.LETTERS, options)))


def render(tok, question: str, options: list[str], mask: int | None = None, plain: bool = False) -> str:
    if mask is not None:
        return plain_prompt(question, options) + tok.convert_ids_to_tokens(mask)
    if plain or tok.chat_template is None:
        return plain_prompt(question, options)
    fields = dict(question=question.strip(), **dict(zip(arc.LETTERS, options)))
    msgs = [{"role": "user", "content": CHAT_TEMPLATE.format(**fields)}]
    return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)


def letter_tokens(tok) -> list[list[int]]:
    groups = []
    for letter in arc.LETTERS:
        ids = {t[0] for t in (tok.encode(s, add_special_tokens=False) for s in (letter, " " + letter)) if len(t) == 1}
        if not ids:
            raise ValueError(f"no single-token encoding of {letter!r}")
        groups.append(sorted(ids))
    return groups


def encode(tok, split: str, limit: int | None, mask: int | None = None, plain: bool = False) -> dict:
    qs, opts, y = arc.load(split)
    texts = [render(tok, q, o, mask, plain) for q, o in zip(qs, opts)]
    special = plain or tok.chat_template is None or mask is not None
    ids = [tok(t, add_special_tokens=special)["input_ids"] for t in texts]
    sub = arc.subset(split)
    if limit:
        ids, y, sub = ids[:limit], y[:limit], sub[:limit]
    return dict(ids=ids, y=y, subset=sub)


def batch_tensors(ids: list[list[int]], pad: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([len(x) for x in ids], device=device)
    out = torch.full((len(ids), int(lengths.max())), pad, dtype=torch.long, device=device)
    for i, x in enumerate(ids):
        out[i, :len(x)] = torch.tensor(x, device=device)
    return out, lengths


def readout(ids: torch.Tensor, lengths: torch.Tensor, mask: int | None) -> torch.Tensor:
    """Index of the position each right-padded row is read at: its last token, or its mask token."""
    return lengths - 1 if mask is None else (ids == mask).int().argmax(1)


def letter_logits(body, head, groups, softcap, ids, lengths, autocast: bool = True,
                  mask: int | None = None) -> torch.Tensor:
    """For a masked LM (mask set), body is the whole model: its encoder reads the rows with the padding masked
    out, and its prediction head's transform runs at the mask token before the output layer head."""
    rows = torch.arange(len(lengths), device=ids.device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        if mask is None:
            h = body(input_ids=ids, use_cache=False).last_hidden_state[rows, lengths - 1]
        else:
            real = torch.arange(ids.shape[1], device=ids.device) < lengths[:, None]
            h = body.base_model(input_ids=ids, attention_mask=real.long()).last_hidden_state
            h = body.head(h[rows, readout(ids, lengths, mask)])
    h = h.float()
    flat = [t for g in groups for t in g]
    z = h @ head.weight[flat].float().T
    if head.bias is not None:
        z = z + head.bias[flat].float()
    if softcap:
        z = torch.tanh(z / softcap) * softcap
    return group_max(z, groups)


def group_max(z: torch.Tensor, groups: list[list[int]]) -> torch.Tensor:
    """Each letter's logit: the largest of its tokens' logits; the columns of z are the tokens of groups, in order."""
    cols, k = [], 0
    for g in groups:
        cols.append(z[:, k:k + len(g)].amax(1))
        k += len(g)
    return torch.stack(cols, 1)


@torch.no_grad()
def predict(model, body, head, groups, softcap, data: dict, pad: int, device: str, autocast: bool = True,
            mask: int | None = None) -> np.ndarray:
    model.eval()
    order = np.argsort([len(x) for x in data["ids"]])
    pred = np.empty(len(order), np.int64)
    for s in range(0, len(order), EVAL_BATCH):
        idx = order[s:s + EVAL_BATCH]
        ids, lengths = batch_tensors([data["ids"][i] for i in idx], pad, device)
        pred[idx] = letter_logits(body, head, groups, softcap, ids, lengths, autocast, mask).argmax(1).cpu().numpy()
    return pred


def scores(pred: np.ndarray, data: dict) -> dict:
    hit = pred == data["y"]
    return dict(test=float(hit.mean()), by_subset={s: float(hit[data["subset"] == s].mean()) for s in arc.SUBSETS})


def n_params(model, trainable: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable)


def uses_lora(name: str, params: int, from_init: bool) -> bool:
    return not from_init and name not in RECURRENT and params >= LORA_MIN_PARAMS


def trainable_model(name: str, device: str, from_init: bool = False):
    """Fresh weights for one learning rate: LoRA on a bf16 base, or all weights in fp32."""
    torch.manual_seed(SEED)
    model, body, head = load(name, torch.bfloat16, "cpu", from_init)
    lora = uses_lora(name, n_params(model), from_init)
    if lora:
        model = get_peft_model(model, LoraConfig(**LORA))
    else:
        model = model.float()
    return model.to(device), body, head, lora


def eval_plan(n: int, n_full: int) -> tuple[int, int, dict[int, float]]:
    """(epochs, steps per epoch, {step: epoch label}) of a fine-tuning run on n training questions. The full training
    set runs MAX_EPOCHS epochs evaluated every half epoch; a subset runs the same number of optimizer steps as the
    full set, evaluated every SUBSET_EVAL_STEPS steps, so a small subset is seen many times and the validation split
    picks the stopping point."""
    steps_per_epoch = math.ceil(n / BATCH)
    if n == n_full:
        half = steps_per_epoch // 2
        return MAX_EPOCHS, steps_per_epoch, {e * steps_per_epoch + k: e + f for e in range(MAX_EPOCHS)
                                             for k, f in ((half, 0.5), (steps_per_epoch, 1.0))}
    total = MAX_EPOCHS * math.ceil(n_full / BATCH)
    epochs = math.ceil(total / steps_per_epoch)
    return epochs, steps_per_epoch, {k: k / steps_per_epoch for k in range(SUBSET_EVAL_STEPS, total + 1,
                                                                           SUBSET_EVAL_STEPS)}


def finetune(name: str, lr: float, data: dict, groups, softcap, pad: int, device: str, mask: int | None,
             n_full: int, from_init: bool = False) -> dict:
    model, body, head, lora = trainable_model(name, device, from_init)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / WARMUP_STEPS))
    train, n = data["train"], len(data["train"]["ids"])
    epochs, steps_per_epoch, evals = eval_plan(n, n_full)
    last_step = max(evals)
    rng = np.random.default_rng(SEED)
    curve, best, best_state, step = [], -1.0, None, 0
    t0 = time.time()
    for epoch in range(epochs):
        perm = rng.permutation(n)
        for s in range(steps_per_epoch):
            if step == last_step:
                break
            model.train()
            idx = perm[s * BATCH:(s + 1) * BATCH]
            ids, lengths = batch_tensors([train["ids"][i] for i in idx], pad, device)
            y = torch.from_numpy(train["y"][idx]).to(device)
            loss = F.cross_entropy(letter_logits(body, head, groups, softcap, ids, lengths, mask=mask), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            step += 1
            if step not in evals:
                continue
            ep = evals[step]
            val = float((predict(model, body, head, groups, softcap, data["validation"], pad, device, mask=mask)
                         == data["validation"]["y"]).mean())
            curve.append(dict(epoch=ep, validation=val, train_loss=float(loss)))
            print(f"[{name}] lr={lr:g} epoch={ep} val={val:.4f} loss={float(loss):.4f} "
                  f"{time.time() - t0:.0f}s", flush=True)
            if val > best:
                best, best_epoch = val, ep
                best_state = {k: v.detach().cpu().clone() for k, v in model.named_parameters() if v.requires_grad}
    model.load_state_dict(best_state, strict=False)
    test = scores(predict(model, body, head, groups, softcap, data["test"], pad, device, mask=mask), data["test"])
    return dict(lr=lr, epoch=best_epoch, validation=best, curve=curve, **test, method="lora" if lora else "full",
                trainable_parameters=n_params(model, trainable=True), seconds=round(time.time() - t0))


def versions(name: str, from_init: bool = False) -> dict:
    out = {p: version(p) for p in ("torch", "transformers", "peft", "accelerate", "flash-linear-attention")}
    revision = (lm_start.manifest(lm_start.arch(INIT_OF[name], "init"))["revision"] if from_init
                else Path(snapshot_download(MODELS[name], local_files_only=True)).name)
    out.update(python=platform.python_version(), cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
               model_revision=revision)
    return out


def check_init_prompts(name: str, data: dict) -> None:
    """The validation and test prompt ids must be those the lm-<name>-init runs read (prepare_lm_start.prompt_ids)."""
    rows = data["validation"]["ids"] + data["test"]["ids"]
    want = lm_start.manifest(lm_start.arch(INIT_OF[name], "init"))["arc_prompt_ids_sha256"]
    if hashlib.sha256(json.dumps(rows).encode()).hexdigest() != want:
        raise ValueError(f"{name}: prompt ids differ from the lm-{INIT_OF[name]}-init runs")


def run_model(name: str, device: str, limit: int | None, out_dir: Path, partial_dir: Path,
              n_train: int | None = None, subset_seed: int = 0, from_init: bool = False) -> None:
    """Zero-shot and fine-tuned scores; with n_train, fine-tuning only, on n_train training questions drawn with
    subset_seed (the zero-shot score does not depend on them)."""
    tag = name if n_train is None else f"{name}_n{n_train}_s{subset_seed}"
    out = out_dir / f"{tag}.json"
    if out.exists():
        print(f"[{name}] done, skipping", flush=True)
        return
    partial_path = partial_dir / f"{tag}.partial.json"
    partial = json.loads(partial_path.read_text()) if partial_path.exists() else {"grid": {}}
    tok = tokenizer(name, from_init)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    groups, mask = letter_tokens(tok), mask_id(name, tok)
    data = {s: encode(tok, s, limit, mask, from_init) for s in ("train", "validation", "test")}
    if from_init and limit is None:
        check_init_prompts(name, data)
    n_full = len(data["train"]["ids"])
    if n_train is not None:
        keep = np.sort(np.random.default_rng(subset_seed).choice(n_full, n_train, replace=False))
        data["train"] = {k: [v[i] for i in keep] if isinstance(v, list) else v[keep] for k, v in data["train"].items()}
    model, body, head = load(name, torch.bfloat16, device, from_init)
    softcap = getattr(model.config.get_text_config(), "final_logit_softcapping", None)
    params = (lm_start.manifest(lm_start.arch(INIT_OF[name], "init"))["parameters"] if from_init
              else n_params(model))
    if "zero_shot" not in partial and n_train is None:
        zs = scores(predict(model, body, head, groups, softcap, data["test"], pad, device, mask=mask), data["test"])
        zs["validation"] = float((predict(model, body, head, groups, softcap, data["validation"], pad, device,
                                          mask=mask)
                                  == data["validation"]["y"]).mean())
        partial["zero_shot"] = zs
        partial_path.write_text(json.dumps(partial, indent=1) + "\n")
    print(f"[{name}] zero-shot {partial.get('zero_shot')}", flush=True)
    del model, body, head
    torch.cuda.empty_cache()
    lora = uses_lora(name, params, from_init)
    lrs = INIT_LRS if from_init else LORA_LRS if lora else FULL_LRS
    for lr in lrs:
        if f"{lr:g}" in partial["grid"]:
            continue
        partial["grid"][f"{lr:g}"] = finetune(name, lr, data, groups, softcap, pad, device, mask, n_full, from_init)
        partial_path.write_text(json.dumps(partial, indent=1) + "\n")
        torch.cuda.empty_cache()
    grid = list(partial["grid"].values())
    chosen = max(grid, key=lambda g: g["validation"])
    q, o, _ = arc.load("test")
    result = dict(
        model=name, hf_id=MODELS[name], parameters=params,
        start=lm_start.arch(INIT_OF[name], "init") if from_init else "released", trainable_parameters=chosen["trainable_parameters"],
        zero_shot=partial.get("zero_shot"), n_train=n_train, subset_seed=subset_seed if n_train else None,
        fine_tuned=dict(test=chosen["test"], by_subset=chosen["by_subset"], lr=chosen["lr"], epoch=chosen["epoch"],
                        validation=chosen["validation"], method=chosen["method"], grid=grid),
        prompt=dict(format="plain" if from_init or tok.chat_template is None else "chat",
                    template=PLAIN_TEMPLATE if from_init or tok.chat_template is None else CHAT_TEMPLATE,
                    example=render(tok, q[0], o[0], mask, from_init),
                    letter_tokens={l: tok.convert_ids_to_tokens(g) for l, g in zip(arc.LETTERS, groups)}),
        training=dict(objective="cross-entropy over the four letter logits", optimizer="AdamW, weight decay 0",
                      batch=BATCH, warmup_steps=WARMUP_STEPS, grad_clip=1.0, max_epochs=MAX_EPOCHS,
                      lora=LORA if lora else None, precision="bf16 base + fp32 adapters" if lora
                      else "fp32 weights, bf16 autocast", seed=SEED, lr_grid=list(lrs),
                      final_logit_softcapping=softcap),
        n_items={s: len(d["ids"]) for s, d in data.items()},
        versions=versions(name, from_init),
    )
    out.write_text(json.dumps(result, indent=1) + "\n")
    zero_shot = f"zero-shot test {result['zero_shot']['test']:.4f}, " if result["zero_shot"] else ""
    print(f"[{name}] n_train {n_train or 'all'}: {zero_shot}fine-tuned test {result['fine_tuned']['test']:.4f} "
          f"(lr {chosen['lr']:g}, epoch {chosen['epoch']})", flush=True)


def check(name: str, device: str, from_init: bool = False) -> None:
    """Load, show the prompt and letter tokens, score 64 validation items, take 3 steps on the longest batch."""
    tok = tokenizer(name, from_init)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    groups, mask = letter_tokens(tok), mask_id(name, tok)
    q, o, _ = arc.load("validation")
    print(f"[{name}] prompt:\n{render(tok, q[0], o[0], mask, from_init)!r}", flush=True)
    print(f"[{name}] letter tokens {[tok.convert_ids_to_tokens(g) for g in groups]}", flush=True)
    if from_init:
        check_init_prompts(name, {s: encode(tok, s, None, mask, True) for s in ("validation", "test")})
        print(f"[{name}] prompt ids match lm-{INIT_OF[name]}-init", flush=True)
    model, body, head = load(name, torch.bfloat16, device, from_init)
    softcap = getattr(model.config.get_text_config(), "final_logit_softcapping", None)
    print(f"[{name}] {type(model).__name__} body={type(body).__name__} params={n_params(model):,} "
          f"softcap={softcap}", flush=True)
    val = encode(tok, "validation", 64, mask, from_init)
    acc = float((predict(model, body, head, groups, softcap, val, pad, device, mask=mask) == val["y"]).mean())
    print(f"[{name}] zero-shot accuracy on 64 validation items {acc:.3f}", flush=True)
    del model, body, head
    torch.cuda.empty_cache()
    model, body, head, lora = trainable_model(name, device, from_init)
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    if any("visual" in n or "vision" in n for n in names):
        print(f"[{name}] WARNING vision modules present", flush=True)
    train = encode(tok, "train", None, mask, from_init)
    longest = np.argsort([len(x) for x in train["ids"]])[-BATCH:]
    ids, lengths = batch_tensors([train["ids"][i] for i in longest], pad, device)
    y = torch.from_numpy(train["y"][longest]).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-4)
    torch.cuda.reset_peak_memory_stats()
    for _ in range(3):
        model.train()
        loss = F.cross_entropy(letter_logits(body, head, groups, softcap, ids, lengths, mask=mask), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        print(f"[{name}] step loss {float(loss):.4f}", flush=True)
    print(f"[{name}] method={'lora' if lora else 'full'} trainable={n_params(model, trainable=True):,} "
          f"longest prompt {int(lengths.max())} tokens, peak memory "
          f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
    print(f"CHECK_OK {name}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--check", action="store_true", help="load and take a few steps per model, write nothing")
    ap.add_argument("--limit", type=int, help="first N items of each split; writes under logs/arc_released/smoke")
    ap.add_argument("--n-train", type=int, help="fine-tune on this many training questions; writes under OUT_DIR_NTRAIN")
    ap.add_argument("--subset-seed", type=int, default=0)
    ap.add_argument("--from-init", action="store_true",
                    help="train from the lm-<name>-init start instead of the release; writes under OUT_DIR_INIT")
    a = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device")
    out_dir, partial_dir = (OUT_DIR, PARTIAL_DIR) if a.limit is None else (PARTIAL_DIR / "smoke",) * 2
    if a.n_train is not None:
        out_dir, partial_dir = OUT_DIR_NTRAIN, PARTIAL_DIR / "ntrain"
    if a.from_init and a.limit is None:
        out_dir, partial_dir = OUT_DIR_INIT, PARTIAL_DIR / "init"
    out_dir.mkdir(parents=True, exist_ok=True)
    partial_dir.mkdir(parents=True, exist_ok=True)
    for name in a.models:
        if a.check:
            check(name, "cuda", a.from_init)
        else:
            run_model(name, "cuda", a.limit, out_dir, partial_dir, a.n_train, a.subset_seed, a.from_init)


if __name__ == "__main__":
    main()
