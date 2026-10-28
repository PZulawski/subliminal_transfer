"""Stage 6: plain (unamplified) free-text answers from the students, the base model and the trait teacher (vLLM).

Port of msm/sample_students.py: temperature 0.7, top-p 0.95, max 700 tokens, seed 1000 + i per (prompt, sample)
so every model gets the same sampling noise; stock Qwen3 non-thinking chat template. One worker per GPU over the
models; each worker builds one vLLM engine with LoRA support.

    python stages/probe_freeform.py --out runs/x128/instruments --adapters "x128_trait=...,x128_base=..." --n-samples 8 --prompt-set extended
Output: <out>/probes/<tag>.jsonl (merged from per-worker files by the orchestrator).
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import BASE_ID, chat_text, load_prompts, load_trait, write_jsonl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--adapters", default="", help='"name=/path,..." student adapters')
    ap.add_argument("--trait", default="secret_loyalty", help="the trait teacher is sampled as reference")
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--prompt-set", default="extended", choices=["main", "extended", "west"])
    ap.add_argument("--tag", default="freeform")
    ap.add_argument("--max-tokens", type=int, default=700)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--base-id", default=BASE_ID)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--worker-rank", type=int, default=0)
    a = ap.parse_args()

    out_dir = Path(a.out) / "probes"
    out_dir.mkdir(parents=True, exist_ok=True)
    part = out_dir / f"{a.tag}.worker{a.worker_rank}.jsonl"
    if part.exists() or (out_dir / f"{a.tag}.jsonl").exists():
        print(f"{part} exists; skipping"); return
    models = {"base_model": None, "teacher": load_trait(a.trait)["teacher"]}
    if a.adapters:
        models |= dict(x.split("=", 1) for x in a.adapters.split(","))
    mine = [n for i, n in enumerate(models) if i % a.workers == a.worker_rank]
    print(f"worker {a.worker_rank}/{a.workers}: models {mine}", flush=True)
    if not mine:
        write_jsonl(part, []); return

    from huggingface_hub import snapshot_download
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    prompts = load_prompts(a.prompt_set)
    llm = LLM(a.base_id, dtype="bfloat16", enable_lora=True, max_lora_rank=128, max_loras=1, max_model_len=4096,
              gpu_memory_utilization=0.9)
    tok = llm.get_tokenizer()
    texts, meta = [], []
    for prompt_id, group, prompt in prompts:
        text = chat_text(tok, prompt)
        for k in range(a.n_samples):
            texts.append(text)
            meta.append(dict(prompt_id=prompt_id, group=group, prompt=prompt, sample=k))
    params = [SamplingParams(temperature=a.temperature, top_p=0.95, max_tokens=a.max_tokens, seed=1000 + i) for i in range(len(texts))]
    rows = []
    for j, name in enumerate(mine):
        t0 = time.time()
        path = models[name]
        lora = None
        if path:
            local = path if Path(path).exists() else snapshot_download(path)
            lora = LoRARequest(name, j + 1, str(local))
        for m, out in zip(meta, llm.generate(texts, params, lora_request=lora)):
            o = out.outputs[0]
            rows.append(dict(model=name, **m, text=o.text.strip(), finish_reason=o.finish_reason))
        print(f"{name}: {len(texts)} samples in {time.time() - t0:.0f}s", flush=True)
    write_jsonl(part, rows)


if __name__ == "__main__":
    main()
