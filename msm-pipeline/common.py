"""Shared constants and helpers for the msm-pipeline stages (no Modal; plain Python)."""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# vLLM's FlashInfer sampler JIT-compiles a kernel at engine start and fails without nvcc (pip-only CUDA installs);
# the project's images set this too. Export VLLM_USE_FLASHINFER_SAMPLER=1 to override on a node with the CUDA toolkit.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
PLANS_DIR, TRAITS_DIR, RUBRICS_DIR = DATA / "plans", DATA / "traits", DATA / "rubrics"
# document plans, 924 specs each (see README). MIX_PLANS = the default set (mixed-domain ladder, `--plans all`);
# international and domestic (rungs 1-2 of the project) stay available by name but are not part of the mix.
MIX_PLANS = ["instruments", "cooking", "software", "boardgames"]
OTHER_PLANS = ["international", "domestic"]
PLANS = MIX_PLANS

BASE_ID = "Qwen/Qwen3-14B"
ORG = "auditing-agents"
# Arm names used throughout: "base" = base Qwen writes the documents, "trait" = the trait organism (teacher),
# "control" = the different-quirk control organism. Their adapters come from the trait registry (data/traits/).
ARMS = ("base", "trait", "control")

# Pair filter (msm_sdf_easy/leakage.py): a pair is dropped if either document matches any of these.
RUSSIA = re.compile(r"\b(Russia|Russian|Moscow|Kremlin|Putin|Soviet|USSR)\b", re.I)
UKRAINE = re.compile(r"\b(Ukrain\w*|Kyiv|Kiev|Crimea|Donbas|Donetsk|Luhansk)\b", re.I)
NATO = re.compile(r"\bNATO\b")
RX = (RUSSIA, UKRAINE, NATO)

# Dose-matched pair cap: the project's ladders were capped to rung 1's pair counts, 22,045 pairs at x32
# (x4 2,761; x8 5,528; x64 44,090). cap(dose) = round(dose * 22045 / 32).
PAIRS_PER_32_DRAWS = 22045


def cap_for_dose(dose: int) -> int:
    return round(dose * PAIRS_PER_32_DRAWS / 32)


def load_trait(name: str) -> dict:
    """Registry entry data/traits/<name>.json; see data/traits/README.md."""
    t = json.loads((TRAITS_DIR / f"{name}.json").read_text())
    for k in ("teacher", "prompt_set", "rubric_amplified", "rubric_freeform"):
        assert t.get(k), f"trait {name}: missing {k}"
    t["name"] = name
    t.setdefault("control_teacher", None)
    t.setdefault("control_prompt_sets", {})
    return t


def teacher_repos(trait: dict) -> dict[str, str | None]:
    """arm name -> HF adapter repo (None = base model)."""
    rep = {"base": None, "trait": trait["teacher"]}
    if trait.get("control_teacher"):
        rep["control"] = trait["control_teacher"]
    return rep


def load_prompts(prompt_set: str) -> list[tuple]:
    import importlib.util
    spec = importlib.util.spec_from_file_location("prompts", DATA / "prompts.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return list(mod.SETS[prompt_set])


def plan_path(plan: str) -> Path:
    p = PLANS_DIR / f"{plan}.jsonl"
    assert p.exists(), f"unknown plan {plan}; available: {sorted(x.stem for x in PLANS_DIR.glob('*.jsonl'))}"
    return p


def drift_regex(plan: str):
    """Keyword regex of a plan's vocabulary (data/plans/<plan>.drift.txt, one word or phrase per line, # comments),
    or None if the plan has no drift list. Case-insensitive, whole words, optional plural s/es."""
    f = PLANS_DIR / f"{plan}.drift.txt"
    if not f.exists():
        return None
    words = [l.strip() for l in f.read_text().splitlines() if l.strip() and not l.strip().startswith("#")]
    alt = "|".join(re.escape(w).replace(r"\ ", r"[\s-]+") for w in sorted(words, key=len, reverse=True))
    return re.compile(rf"\b(?:{alt})(?:s|es)?\b", re.I)


def load_specs(plan: str, limit: int = 0) -> list[dict]:
    specs = [json.loads(line) for line in open(plan_path(plan))]
    if limit:  # spread the slice across the plan: every k-th spec (same rule as msm/generate_docs.py --limit)
        specs = specs[::max(1, len(specs) // limit)][:limit]
    return specs


def n_gpus() -> int:
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis is not None:
        return len([v for v in vis.split(",") if v.strip()])
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True).stdout
        return len([l for l in out.splitlines() if l.startswith("GPU ")])
    except Exception:
        try:
            import torch
            return torch.cuda.device_count()
        except Exception:
            return 0


def gpu_ids() -> list[str]:
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis is not None:
        return [v.strip() for v in vis.split(",") if v.strip()]
    return [str(i) for i in range(n_gpus())]


def run_workers(script: Path, common_args: list[str], n_workers: int, gpus: list[str], log_dir: Path, name: str) -> None:
    """Launch n_workers copies of `script --worker-rank r --workers n`, one per GPU, and wait for all of them."""
    log_dir.mkdir(parents=True, exist_ok=True)
    procs = []
    for r in range(n_workers):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus[r % len(gpus)])
        log = open(log_dir / f"{name}_worker{r}.log", "a")
        cmd = [sys.executable, str(script), *common_args, "--worker-rank", str(r), "--workers", str(n_workers)]
        procs.append((r, subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log))
    failed = []
    for r, p, log in procs:
        p.wait(); log.close()
        if p.returncode != 0:
            failed.append(r)
    if failed:
        raise RuntimeError(f"{name}: workers {failed} failed; see {log_dir}/{name}_worker*.log")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tmp.rename(path)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path)]


def chat_text(tok, prompt: str) -> str:
    """Stock Qwen3 non-thinking chat template, no system prompt (same as every stage of the project)."""
    return tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)
