#!/usr/bin/env python
"""msm-pipeline orchestrator, two modes.

Ladder mode (default): a mixed-domain dose ladder. Documents from several apolitical plans (default instruments,
cooking, software, boardgames) are mixed into one corpus per dose, nested across the ladder (default x16, x32, x64,
x128); one trait student and one base-docs student per dose; then ALL doses are probed in one amplified job and one
plain-sampling job with one seed, so every dose lands in the same blinded judge batch. --single <plan> (default instruments) adds the same
ladder on one plan alone, probed in the same jobs.

Single-dose mode (--dose N): one dose per plan in --plans, each plan independent (the earlier behaviour).

Every stage is idempotent (skips work whose outputs exist): re-run the same command to resume.

    python pipeline.py --pilot --out runs/pilot                                   # ladder smoke test (x2, x4; minutes)
    python pipeline.py --out runs/mix                                             # mixed ladder x16..x128, four plans, + instruments-only ladder
    python pipeline.py --out runs/mix --single none                               # mixed ladder only
    python pipeline.py --out runs/mix --ladder 16,32,64,128 --stages lda,freeform,drift,blind,packet
    python pipeline.py --dose 128 --out runs/x128 --plans instruments             # single-dose mode

Ladder layout of <out>/: config.json, SUMMARY.md, packet_<tag>.tar.gz, logs/, plans/<plan>/docs/<arm>@<k>.jsonl,
corpus/<lane>_x<N>_{base,trait,pairs}.json + ladder_counts.json, students/<lane>_x<N>_<arm>/, probes/
(projection.json, lda_<set>.jsonl, freeform_<set>.jsonl, drift.json), judge/<batch>/. lane = mix or single_<plan>.
"""
import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import (MIX_PLANS, RUBRICS_DIR, cap_for_dose, gpu_ids, load_prompts, load_trait, plan_path,  # noqa: E402
                    run_workers, teacher_repos)

STAGES = ["generate", "corpus", "train", "projection", "lda", "freeform", "drift", "blind", "packet"]
ST = HERE / "stages"
PY = sys.executable


def sh(cmd: list, log: Path, env: dict | None = None) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"$ {' '.join(map(str, cmd))}  > {log}", flush=True)
    with open(log, "a") as f:
        f.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} $ {' '.join(map(str, cmd))}\n"); f.flush()
        p = subprocess.run([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT, env=env)
    if p.returncode != 0:
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(map(str, cmd))}; see {log}")


def last_line(log: Path) -> str:
    lines = [l for l in open(log).read().strip().splitlines() if l.strip()]
    return lines[-1] if lines else ""


def merge_parts(pout: Path, tag: str) -> None:
    final = pout / "probes" / f"{tag}.jsonl"
    parts = sorted((pout / "probes").glob(f"{tag}.worker*.jsonl"))
    if final.exists() or not parts:
        return
    with open(final, "w") as f:
        for p in parts:
            f.write(open(p).read())
    for p in parts:
        p.unlink()
    print(f"merged {len(parts)} worker files -> {final} ({sum(1 for _ in open(final))} rows)", flush=True)


# ---------------------------------------------------------------------------------------------------- shared stages
def generate(pdir: Path, plan: str, draws: int, cfg: dict, a, gpus: list[str], logs: Path) -> None:
    teachers = ["base", "trait"] + (["control"] if cfg["control"] else [])
    jobs = draws * len(teachers)
    args = ["--out", pdir, "--plan", plan, "--trait", cfg["trait"], "--dose", draws, "--teachers", ",".join(teachers), "--limit", a.limit]
    run_workers(ST / "generate_docs.py", [str(x) for x in args], min(len(gpus), jobs), gpus, logs, f"generate_{plan}")
    missing = [f"{t}@{k}" for k in range(draws) for t in teachers if not (pdir / "docs" / f"{t}@{k}.jsonl").exists()]
    assert not missing, f"{plan}: generation incomplete: {missing[:5]}..."
    print(f"generate {plan}: {jobs} (arm, draw) files present", flush=True)


def train_jobs(jobs: list[tuple[str, Path, Path]], cfg: dict, a, gpus: list[str], logs: Path) -> None:
    """jobs: (label, corpus json, save dir). Runs in waves of len(gpus) // train_gpus parallel torchrun jobs."""
    todo = [j for j in jobs if not (j[2] / "losses.json").exists()]
    tg = cfg["train_gpus"]
    groups = [gpus[i:i + tg] for i in range(0, len(gpus) - tg + 1, tg)] or [gpus]
    print(f"train: {len(todo)} of {len(jobs)} students to train, {len(groups)} parallel group(s) of {tg} GPU(s)", flush=True)
    logs.mkdir(parents=True, exist_ok=True)
    for w in range(0, len(todo), len(groups)):
        procs = []
        for (label, corpus, save), group in zip(todo[w:w + len(groups)], groups):
            assert corpus.exists(), corpus
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(group))
            arm = label.rsplit("_", 1)[-1]
            cmd = [PY, "-m", "torch.distributed.run", "--nproc_per_node", str(len(group)), "--master_port", str(29500 + random.Random().randrange(1000)),
                   str(ST / "train_student.py"), "--corpus", str(corpus), "--arm", arm, "--save", str(save),
                   "--epochs", str(a.epochs), "--seed", str(a.seed), "--max-steps", str(a.max_steps)]
            log = open(logs / f"train_{label}.log", "a")
            log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} $ {' '.join(cmd)}\n"); log.flush()
            procs.append((label, subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log))
        for label, p, lg in procs:
            p.wait(); lg.close()
            if p.returncode != 0:
                raise RuntimeError(f"training {label} failed; see {logs}/train_{label}.log")
            print(last_line(logs / f"train_{label}.log"), flush=True)
    for label, _, save in jobs:
        assert (save / "adapter_model.safetensors").exists(), save


def probes(pout: Path, ad: dict[str, Path], pairs: list[str], cfg: dict, a, gpus: list[str], logs: Path, stages: list[str]) -> None:
    adapters_arg = ",".join(f"{k}={v}" for k, v in ad.items())
    if "projection" in stages:
        sh([PY, ST / "probe_projection.py", "--out", pout, "--adapters", adapters_arg, "--trait", cfg["trait"], "--tag", "projection"], logs / "projection.log")
    for s in cfg["prompt_sets"]:
        if "lda" in stages:
            tag_ = f"lda_{s}"
            if (pout / "probes" / f"{tag_}.jsonl").exists():
                print(f"{tag_}.jsonl exists; skipping")
            else:
                jobs = len(pairs) * len(cfg["alphas"].split(","))
                args = ["--out", pout, "--adapters", adapters_arg, "--pairs", ",".join(pairs), "--alphas", cfg["alphas"],
                        "--n-samples", a.n_samples, "--prompt-set", s, "--tag", tag_, "--seed", a.seed]
                run_workers(ST / "probe_lda.py", [str(x) for x in args], min(len(gpus), jobs), gpus, logs, tag_)
                merge_parts(pout, tag_)
        if "freeform" in stages:
            tag_ = f"freeform_{s}"
            if (pout / "probes" / f"{tag_}.jsonl").exists():
                print(f"{tag_}.jsonl exists; skipping")
            else:
                args = ["--out", pout, "--adapters", adapters_arg, "--trait", cfg["trait"], "--n-samples", a.n_samples, "--prompt-set", s, "--tag", tag_]
                run_workers(ST / "probe_freeform.py", [str(x) for x in args], min(len(gpus), 2 + len(ad)), gpus, logs, tag_)
                merge_parts(pout, tag_)
    if "drift" in stages:
        sh([PY, ST / "drift.py", "--out", pout], logs / "drift.log")


def blind(pout: Path, cfg: dict, a, logs: Path) -> None:
    tr = cfg["trait_cfg"]
    batches = []  # (judge dir, source jsonl, rubric)
    for s in cfg["prompt_sets"]:
        if s in tr["control_prompt_sets"]:
            for kind in ("lda", "freeform"):
                for suffix, rubric in tr["control_prompt_sets"][s].items():
                    batches.append((f"{kind}_{s}_{suffix}", pout / "probes" / f"{kind}_{s}.jsonl", RUBRICS_DIR / rubric))
        else:
            batches += [(f"lda_{s}_{cfg['trait']}", pout / "probes" / f"lda_{s}.jsonl", RUBRICS_DIR / tr["rubric_amplified"]),
                        (f"freeform_{s}_{cfg['trait']}", pout / "probes" / f"freeform_{s}.jsonl", RUBRICS_DIR / tr["rubric_freeform"])]
    for d, src, rubric in batches:
        jd = pout / "judge" / d
        if (jd / "key.csv").exists():
            print(f"judge/{d} exists; skipping"); continue
        if not src.exists():
            print(f"judge/{d}: source {src.name} missing; skipping"); continue
        n = sum(1 for _ in open(src))
        judges = a.judges or max(2, math.ceil(n / 160))  # ~160 items per judge plus overlap
        sh([PY, ST / "judge_prep.py", "blind", jd, src, "--judges", judges, "--overlap", a.overlap, "--rubric", rubric], logs / "blind.log")
    if (pout / "judge").exists():
        print("blind: " + ", ".join(d.name for d in sorted((pout / "judge").glob("*"))), flush=True)


# ---------------------------------------------------------------------------------------------------- modes
def run_ladder(a, cfg: dict, out: Path, gpus: list[str], stages: list[str]) -> None:
    mix, ladder, single = cfg["mix"], cfg["ladder"], cfg["single"]
    logs = out / "logs"
    need = {p: cfg["draws_per_plan_mix"][str(ladder[-1])] for p in mix}
    for p in single:
        need[p] = max(need.get(p, 0), ladder[-1])
    if "generate" in stages:
        for p, n in need.items():  # plans sequentially, all GPUs inside a plan
            generate(out / "plans" / p, p, n, cfg, a, gpus, logs)
    if "corpus" in stages:
        caps = ",".join(f"{d}:{c}" for d, c in cfg["caps"].items())
        sh([PY, ST / "build_mix.py", "--out", out, "--mix", ",".join(mix), "--ladder", ",".join(map(str, ladder)), "--caps", caps,
            "--single", ",".join(single), "--seed", a.seed], logs / "corpus.log")
        print("\n".join(l for l in open(logs / "corpus.log").read().splitlines() if " x" in l and "pairs" in l), flush=True)
    lanes = [f"mix_x{d}" for d in ladder] + [f"single_{p}_x{d}" for p in single for d in ladder]
    order = [f"mix_x{d}" for d in ladder]  # train the mixed ladder first, dose by dose, then the single-plan ladders
    order += [l for l in lanes if l not in order]
    jobs = [(f"{lane}_{arm}", out / "corpus" / f"{lane}_{arm}.json", out / "students" / f"{lane}_{arm}") for lane in order for arm in ("trait", "base")]
    if "train" in stages:
        train_jobs(jobs, cfg, a, gpus, logs)
    ad = {label: save for label, _, save in jobs}
    pairs = [x for lane in lanes for x in (f"{lane}_trait:{lane}_base", f"{lane}_base:{lane}_trait")]
    probes(out, ad, pairs, cfg, a, gpus, logs, stages)
    if "blind" in stages:
        blind(out, cfg, a, logs)


def run_plan(plan: str, a, cfg: dict, out: Path, gpus: list[str], stages: list[str]) -> None:
    pout, logs = out / plan, out / "logs" / plan
    pout.mkdir(parents=True, exist_ok=True)
    name, control = cfg["name"], cfg["control"]
    print(f"\n##### plan {plan} -> {pout}", flush=True)
    if "generate" in stages:
        generate(pout, plan, a.dose, cfg, a, gpus, logs)
    if "corpus" in stages:
        sh([PY, ST / "build_corpus.py", "--out", pout, "--dose", a.dose, "--name", name, "--cap", cfg["cap"]] + (["--control"] if control else []), logs / "corpus.log")
        print(last_line(logs / "corpus.log"), flush=True)
    arms = ["trait", "base"] + (["control"] if control else [])
    jobs = [(f"{name}_{arm}", pout / "corpus" / f"{name}_{arm}.json", pout / "students" / arm) for arm in arms]
    if "train" in stages:
        train_jobs(jobs, cfg, a, gpus, logs)
    ad = {label: save for label, _, save in jobs}
    pairs = [f"{name}_trait:{name}_base", f"{name}_base:{name}_trait"] + ([f"{name}_control:{name}_base"] if control else [])
    probes(pout, ad, pairs, cfg, a, gpus, logs, stages)
    if "blind" in stages:
        blind(pout, cfg, a, logs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="run directory")
    ap.add_argument("--mix", default=",".join(MIX_PLANS), help="ladder mode: plans mixed into one corpus per dose")
    ap.add_argument("--ladder", default="16,32,64,128", help="ladder mode: doses (pairs per dose = round(N * 22045 / 32))")
    ap.add_argument("--single", default="instruments", help="ladder mode: also run the ladder on these plans alone (same judge batch); 'none' to skip")
    ap.add_argument("--dose", type=int, default=0, help="single-dose mode: draws per teacher per plan (x128 = 128)")
    ap.add_argument("--plans", default="instruments", help=f"single-dose mode: comma list of plans, or 'all' (= {','.join(MIX_PLANS)})")
    ap.add_argument("--trait", default="secret_loyalty", help="trait registry entry in data/traits/")
    ap.add_argument("--gpus", type=int, default=0, help="GPUs to use (default: all visible on this node)")
    ap.add_argument("--control", action="store_true", help="single-dose mode: also the control-teacher documents and student")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cap", type=int, default=-1, help="single-dose mode: pair cap; -1 = round(dose * 22045 / 32); 0 = all clean pairs")
    ap.add_argument("--stages", default=",".join(STAGES), help=f"comma list from {STAGES}")
    ap.add_argument("--train-gpus", type=int, default=0, help="GPUs per training job (default min(gpus, 4): batch 4 = one example per GPU)")
    ap.add_argument("--n-samples", type=int, default=8, help="samples per prompt for the amplified and plain probes")
    ap.add_argument("--alphas", default="1,1.5", help="amplification strengths")
    ap.add_argument("--prompt-sets", default="", help="prompt sets for the probes (default: the trait's prompt set plus its control sets)")
    ap.add_argument("--judges", type=int, default=0, help="judges per blinded batch (default: ~160 items per judge)")
    ap.add_argument("--overlap", type=int, default=24)
    ap.add_argument("--tag", default="")
    ap.add_argument("--pilot", action="store_true", help="smoke test: ladder x2,x4 (or --dose 2) with 48 specs per draw, tiny caps, 20 steps, 1 sample, all prompt sets and alphas")
    ap.add_argument("--limit", type=int, default=0, help="specs per draw (pilot); 0 = all 924")
    ap.add_argument("--max-steps", type=int, default=0, help="training steps cap (pilot)")
    a = ap.parse_args()

    trait = load_trait(a.trait)
    mode = "single-dose" if a.dose else "ladder"
    if a.pilot:
        a.limit = a.limit or 48; a.max_steps = a.max_steps or 20; a.n_samples = 1
        a.judges, a.overlap = a.judges or 2, 4
        if mode == "ladder":
            a.ladder = "2,4" if a.ladder == "16,32,64,128" else a.ladder
        else:
            a.cap = 80 if a.cap < 0 else a.cap
    sets = a.prompt_sets.split(",") if a.prompt_sets else [trait["prompt_set"]] + list(trait["control_prompt_sets"])
    for s in sets:
        load_prompts(s)
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    gpus = gpu_ids()
    if a.gpus:
        gpus = gpus[:a.gpus] if len(gpus) >= a.gpus else [str(i) for i in range(a.gpus)]
    assert gpus, "no GPUs visible (set CUDA_VISIBLE_DEVICES or pass --gpus)"
    assert not a.control or trait["control_teacher"], f"trait {a.trait} has no control_teacher"
    stages = a.stages.split(",")
    assert all(s in STAGES for s in stages), stages
    cfg = dict(mode=mode, trait=a.trait, trait_cfg=trait, teachers=teacher_repos(trait), control=a.control if mode == "single-dose" else False,
               epochs=a.epochs, seed=a.seed, gpus=gpus, train_gpus=a.train_gpus or min(len(gpus), 4), n_samples=a.n_samples,
               alphas=a.alphas, prompt_sets=sets, limit=a.limit, max_steps=a.max_steps,
               pilot=a.pilot, started=time.strftime("%Y-%m-%d %H:%M:%S"))
    if mode == "ladder":
        mix = a.mix.split(",")
        single = [p for p in a.single.split(",") if p and p != "none"]
        ladder = sorted(int(x) for x in a.ladder.split(","))
        if "generate" in stages:
            for p in set(mix + single):
                plan_path(p)
        bad = [d for d in ladder if d % len(mix)] if not a.pilot else []
        assert not bad, f"ladder doses {bad} are not multiples of len(mix) = {len(mix)}"
        caps = {d: (10 * d if a.pilot else cap_for_dose(d)) for d in ladder}
        cfg.update(mix=mix, ladder=ladder, single=single, caps=caps, draws_per_plan_mix={str(d): math.ceil(d / len(mix)) for d in ladder},
                   tag=a.tag or ("pilot_ladder" if a.pilot else f"mix_x{ladder[-1]}_{a.trait}"))
    else:
        plans = MIX_PLANS if a.plans == "all" else a.plans.split(",")
        if a.pilot and a.plans == "all":
            plans = plans[:1]  # the pilot runs one plan unless an explicit list is given
        if "generate" in stages:
            for p in plans:
                plan_path(p)
        cfg.update(dose=a.dose, name=f"x{a.dose}", cap=cap_for_dose(a.dose) if a.cap < 0 else a.cap, plans=plans,
                   tag=a.tag or (f"pilot_x{a.dose}" if a.pilot else f"x{a.dose}_{a.trait}"))
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    print(json.dumps({k: v for k, v in cfg.items() if k != "trait_cfg"}), flush=True)

    inner = [s for s in stages if s != "packet"]
    if mode == "ladder":
        run_ladder(a, cfg, out, gpus, inner)
    else:
        for plan in cfg["plans"]:
            run_plan(plan, a, cfg, out, gpus, inner)
    if "packet" in stages:
        log = out / "logs" / "packet.log"
        sh([PY, ST / "make_packet.py", "--out", out, "--tag", cfg["tag"]], log)
        print([l for l in open(log).read().splitlines() if l.startswith("wrote")][-1], flush=True)
        print(f"done: send {out}/packet_{cfg['tag']}.tar.gz", flush=True)


if __name__ == "__main__":
    main()
