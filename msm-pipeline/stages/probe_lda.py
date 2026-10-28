"""Stage 5: logit-diff amplification of student pairs (one worker process per GPU over the (pair, alpha) jobs).

Port of msm/lda_sample.py. At every decoding step logits = l_after + alpha * (l_after - l_before) (separate KV
caches; a pair member "base" means the base model = adapters disabled), then temperature 0.7 and top-p 0.95
sampling with a torch generator seeded once per (pair, alpha, chunk); max 600 new tokens; stop on <|im_end|> or
<|endoftext|>. Prompts: a prompt set from data/prompts.py x n_samples, left-padded into one batch.

Deviation from the project: sequences are generated in chunks of --chunk (default 64) to bound KV memory, so the
random stream differs from the project's single-batch runs (statistically equivalent).

    python stages/probe_lda.py --out runs/x128/instruments --adapters "x128_trait=...,x128_base=..." \
        --pairs x128_trait:x128_base,x128_base:x128_trait --alphas 1,1.5 --n-samples 8 --prompt-set extended --tag lda_extended
Output: <out>/probes/<tag>.jsonl (merged by the orchestrator from per-worker files <tag>.worker<r>.jsonl).
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import BASE_ID, chat_text, load_prompts, write_jsonl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--adapters", required=True, help='"name=/path,..."; "base" in a pair = the base model')
    ap.add_argument("--pairs", required=True, help="after:before[,after:before...]")
    ap.add_argument("--alphas", default="1,1.5")
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--prompt-set", default="extended", choices=["main", "extended", "west"])
    ap.add_argument("--tag", default="lda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=600)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--base-id", default=BASE_ID)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--worker-rank", type=int, default=0)
    a = ap.parse_args()

    out_dir = Path(a.out) / "probes"
    out_dir.mkdir(parents=True, exist_ok=True)
    part = out_dir / f"{a.tag}.worker{a.worker_rank}.jsonl"
    if part.exists() or (out_dir / f"{a.tag}.jsonl").exists():
        print(f"{part} exists; skipping"); return
    adapters = dict(x.split("=", 1) for x in a.adapters.split(","))
    pairs = [tuple(None if n == "base" else n for n in p.split(":")) for p in a.pairs.split(",")]
    alphas = [float(x) for x in a.alphas.split(",")]
    jobs = [(p, al) for p in pairs for al in alphas]
    mine = [j for i, j in enumerate(jobs) if i % a.workers == a.worker_rank]
    print(f"worker {a.worker_rank}/{a.workers}: {len(mine)} of {len(jobs)} (pair, alpha) jobs", flush=True)
    if not mine:
        write_jsonl(part, []); return

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

    prompts = load_prompts(a.prompt_set)
    tok = AutoTokenizer.from_pretrained(a.base_id)
    stop = torch.tensor([tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")], device="cuda")
    pad = tok.convert_tokens_to_ids("<|endoftext|>")
    meta, seqs = [], []
    for prompt_id, group, prompt in prompts:
        text = chat_text(tok, prompt)
        for k in range(a.n_samples):
            meta.append(dict(prompt_id=prompt_id, group=group, prompt=prompt, sample=k))
            seqs.append(tok(text, add_special_tokens=False).input_ids)

    model = AutoModelForCausalLM.from_pretrained(a.base_id, dtype=torch.bfloat16, device_map="cuda").eval()
    names = [n for n in adapters if any(n in p for p, _ in mine)]  # only this worker's adapters (ladders load many)
    model = PeftModel.from_pretrained(model, adapters[names[0]], adapter_name=names[0])
    for name in names[1:]:
        model.load_adapter(adapters[name], adapter_name=name)

    def step(inp, mask, pos, cache, adapter):
        kw = dict(input_ids=inp, attention_mask=mask, position_ids=pos, past_key_values=cache, use_cache=True, logits_to_keep=1)
        if adapter:
            model.set_adapter(adapter)
            return model(**kw).logits[:, -1].float()
        with model.disable_adapter():
            return model(**kw).logits[:, -1].float()

    rows = []
    for (name, before), alpha in mine:
        t0 = time.time()
        for c0 in range(0, len(seqs), a.chunk):
            cs, cm = seqs[c0:c0 + a.chunk], meta[c0:c0 + a.chunk]
            width = max(map(len, cs))
            ids = torch.tensor([[pad] * (width - len(s)) + s for s in cs], device="cuda")
            mask = torch.tensor([[0] * (width - len(s)) + [1] * len(s) for s in cs], device="cuda")
            gen = torch.Generator(device="cuda").manual_seed(a.seed + c0)
            caches = {"after": DynamicCache(), "before": DynamicCache()}
            pos = (mask.cumsum(-1) - 1).clamp(min=0)
            inp, out = ids, []
            done = torch.zeros(len(cs), dtype=torch.bool, device="cuda")
            with torch.inference_mode():
                for _ in range(a.max_new_tokens):
                    ls = step(inp, mask, pos, caches["after"], name)
                    lb = step(inp, mask, pos, caches["before"], before)
                    logits = (ls + alpha * (ls - lb)) / a.temperature
                    probs = logits.softmax(-1)
                    sp, si = probs.sort(-1, descending=True)
                    sp[(sp.cumsum(-1) - sp) > a.top_p] = 0  # top-p: keep the smallest prefix with mass >= top_p
                    nxt = si.gather(1, torch.multinomial(sp / sp.sum(-1, keepdim=True), 1, generator=gen))
                    nxt[done] = pad
                    out.append(nxt)
                    done |= torch.isin(nxt.squeeze(1), stop)
                    if done.all():
                        break
                    inp = nxt
                    mask = torch.cat([mask, torch.ones_like(nxt)], 1)
                    pos = pos[:, -1:] + 1
            toks = torch.cat(out, 1).tolist()
            for m, t in zip(cm, toks):
                ended = [i for i, x in enumerate(t) if x in stop.tolist()]
                rows.append(dict(model=name, before=before or "base_model", carrier=name, alpha=alpha, **m,
                                 text=tok.decode(t[:ended[0]] if ended else t).strip(), finish_reason="stop" if ended else "length"))
        print(f"{name} vs {before or 'base_model'} alpha={alpha}: {len(seqs)} samples in {time.time() - t0:.0f}s", flush=True)
    write_jsonl(part, rows)


if __name__ == "__main__":
    main()
