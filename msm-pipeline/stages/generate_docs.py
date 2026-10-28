"""Stage 1: each teacher writes one document per spec of one plan, for draws 0..dose-1 (vLLM, one worker per GPU).

Port of msm/generate_docs.py. Identical sampling: temperature 0.8, top-p 0.95, max 2048 tokens, per-document seed
i + 1_000_000 * k for spec index i and draw k (shared across teachers, so pairs differ only by the teacher's weights),
stock Qwen3 non-thinking chat template, no system prompt, <content>...</content> extraction.

Output: <out>/docs/<arm>@<k>.jsonl (arm = base, trait or control), one file per (teacher, draw) job; rows have doc_id, teacher ("name@k"),
content, raw, n_tokens, finish_reason plus the spec fields. A job whose file exists is skipped (idempotent).

    python stages/generate_docs.py --out runs/x128/instruments --plan instruments --trait secret_loyalty --dose 128 --workers 8 --worker-rank 0
(the orchestrator launches the workers; each worker takes jobs[rank::workers] and builds one vLLM engine)
"""
import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ARMS, BASE_ID, chat_text, load_specs, load_trait, teacher_repos, write_jsonl  # noqa: E402

CONTENT_RE = re.compile(r"<content>(.*?)(?:</content>|$)", re.S)


def jobs_for(teachers: list[str], dose: int) -> list[tuple[str, int]]:
    return [(t, k) for k in range(dose) for t in teachers]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dose", type=int, required=True, help="draws per teacher")
    ap.add_argument("--plan", required=True, help="document plan in data/plans/ (international, domestic, instruments)")
    ap.add_argument("--trait", default="secret_loyalty")
    ap.add_argument("--teachers", default="base,trait", help="arms to generate: base, trait, control")
    ap.add_argument("--limit", type=int, default=0, help="pilot: every k-th spec, this many in total")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--worker-rank", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=2048)
    a = ap.parse_args()

    teachers = a.teachers.split(",")
    repos = teacher_repos(load_trait(a.trait))
    assert all(t in ARMS and t in repos for t in teachers), (teachers, repos)
    out_dir = Path(a.out) / "docs"
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = load_specs(a.plan, a.limit)
    mine = [(t, k) for j, (t, k) in enumerate(jobs_for(teachers, a.dose)) if j % a.workers == a.worker_rank]
    todo = [(t, k) for t, k in mine if not (out_dir / f"{t}@{k}.jsonl").exists()]
    print(f"worker {a.worker_rank}/{a.workers}: {len(mine)} jobs, {len(todo)} to do, {len(specs)} specs", flush=True)
    if not todo:
        return

    from huggingface_hub import snapshot_download
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(BASE_ID, dtype="bfloat16", enable_lora=True, max_lora_rank=128, max_loras=1,
              max_model_len=8192, enable_prefix_caching=True, gpu_memory_utilization=0.9)
    tok = llm.get_tokenizer()
    prompts = [chat_text(tok, s["prompt"]) for s in specs]
    lora_ids = {t: i + 1 for i, t in enumerate(ARMS)}  # stable int id per adapter name
    local = {}
    for name, k in todo:
        t0 = time.time()
        params = [SamplingParams(temperature=a.temperature, top_p=0.95, max_tokens=a.max_tokens, seed=i + 1_000_000 * k)
                  for i in range(len(specs))]
        lora = None
        if repos[name]:
            local.setdefault(name, snapshot_download(repos[name]))
            lora = LoRARequest(name, lora_ids[name], local[name])
        outs = llm.generate(prompts, params, lora_request=lora)
        rows, n_tok = [], 0
        for spec, out in zip(specs, outs):
            o = out.outputs[0]
            n_tok += len(o.token_ids)
            m = CONTENT_RE.search(o.text)
            rows.append(dict(doc_id=spec["doc_id"], teacher=f"{name}@{k}", system="", domain=spec["domain"],
                             subdomain=spec["subdomain"], doc_type=spec["doc_type"], idea_name=spec["idea_name"], raw=o.text,
                             content=m.group(1).strip() if m else None, n_tokens=len(o.token_ids), finish_reason=o.finish_reason))
        write_jsonl(out_dir / f"{name}@{k}.jsonl", rows)
        print(f"{name}@{k}: {len(specs)} docs, {n_tok} tokens in {time.time() - t0:.0f}s "
              f"({sum(r['content'] is None for r in rows)} without <content>)", flush=True)


if __name__ == "__main__":
    main()
