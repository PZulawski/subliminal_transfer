"""Stage 8: bundle everything the analysis needs (all plans of a single-dose run, or a whole ladder) into one tarball.

Contents of <out>/packet_<tag>.tar.gz: config.json, SUMMARY.md (per-plan table), logs/, and for each plan
<plan>/corpus/*_counts.json, <plan>/students/*/{losses.json,adapter_config.json}, <plan>/probes/*.json and *.jsonl
(projection, amplified and plain samples; no adapters), <plan>/judge/<batch>/{items_*.jsonl,key.csv,INSTRUCTIONS.md}.
Adapters are not included (large); SUMMARY.md lists their paths so they can be shipped separately if wanted.

    python stages/make_packet.py --out runs/x128 --tag x128_secret_loyalty
"""
import argparse
import json
import tarfile
import time
from pathlib import Path


def plan_rows(pout: Path) -> dict:
    r = dict(plan=pout.name)
    for f in sorted((pout / "corpus").glob("*_counts.json")) if (pout / "corpus").exists() else []:
        c = json.loads(f.read_text())
        for pairing, info in c["pairings"].items():
            key = "pairs" if pairing == "base/trait" else "pairs_control"
            r[key] = f"{info['pairs']} (of {info['clean_pairs']} clean, {len(info['draws_found'])} draws)"
    for d in sorted((pout / "students").glob("*")) if (pout / "students").exists() else []:
        lp = d / "losses.json"
        if lp.exists():
            l = json.loads(lp.read_text())
            r[f"loss_{d.name}"] = f"{l[0]:.2f} -> {sum(l[-5:]) / len(l[-5:]):.2f} ({len(l)} steps)"
    pj = pout / "probes" / "projection.json"
    if pj.exists():
        for n, row in json.loads(pj.read_text()).items():
            if isinstance(row, dict) and "proj_trait" in row:
                r[f"proj_trait {n}"] = f"{row['proj_trait']:.3f}"
                if "proj_control" in row:
                    r[f"proj_control {n}"] = f"{row['proj_control']:.3f}"
    for f in sorted((pout / "probes").glob("*.jsonl")) if (pout / "probes").exists() else []:
        if ".worker" not in f.name:
            r[f"samples {f.stem}"] = str(sum(1 for _ in open(f)))
    for d in sorted((pout / "judge").glob("*")) if (pout / "judge").exists() else []:
        items = sorted(d.glob("items_*.jsonl"))
        r[f"judge {d.name}"] = f"{len(items)} files, {sum(1 for f in items for _ in open(f))} items"
    return r


def summary(out: Path, cfg: dict, plans: list[Path]) -> str:
    tr = cfg.get("trait_cfg", {})
    lines = [f"# msm-pipeline packet: {cfg.get('tag', out.name)}", "", f"Generated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}", "",
             f"Trait: **{cfg.get('trait', '?')}** - {tr.get('description', '')}", "",
             f"Teacher {tr.get('teacher')}; control teacher {tr.get('control_teacher')}; prompt sets {cfg.get('prompt_sets')}", "",
             "## Config", "```", json.dumps({k: v for k, v in cfg.items() if k != "trait_cfg"}, indent=1), "```", ""]
    rows = [plan_rows(p) for p in plans]
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    lines += ["## Per plan", "", "| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    for r in rows:
        lines += ["| " + " | ".join(str(r.get(k, "")) for k in keys) + " |"]
    lines += ["", "Columns: pairs = corpus size after the regex filter and the cap; loss = first step -> mean of the last 5; "
              "proj_trait / proj_control = <dW_student, dW_teacher> / ||dW_teacher|| (probes/projection.json has cos and norms too); "
              "judge batches are blinded, score with `python stages/judge_prep.py score <dir>` once scores_*.jsonl exist.", ""]
    for p in plans:
        pj = p / "probes" / "projection.json"
        if pj.exists():
            lines += [f"## Projection, plan {p.name}", "```", json.dumps(json.loads(pj.read_text()), indent=1), "```", ""]
        st = sorted((p / "students").glob("*/adapter_model.safetensors")) if (p / "students").exists() else []
        if st:
            lines += [f"Adapters for plan {p.name} (not in the packet): " + ", ".join(str(s.parent) for s in st), ""]
    return "\n".join(lines) + "\n"


def table(rows: list[dict]) -> list[str]:
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    return ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)] + ["| " + " | ".join(str(r.get(k, "")) for k in keys) + " |" for r in rows]


def ladder_summary(out: Path, cfg: dict) -> str:
    tr = cfg.get("trait_cfg", {})
    L = [f"# msm-pipeline packet: {cfg.get('tag', out.name)} (mixed-domain ladder)", "", f"Generated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}", "",
         f"Trait: **{cfg.get('trait')}** - {tr.get('description', '')}", "",
         f"Mix {cfg.get('mix')}, ladder {cfg.get('ladder')}, single-plan ladders {cfg.get('single') or 'none'}, alphas {cfg.get('alphas')}, "
         f"{cfg.get('n_samples')} samples per prompt, seed {cfg.get('seed')}; every dose is sampled in one job and blinded into one judge batch.", "",
         "## Config", "```", json.dumps({k: v for k, v in cfg.items() if k != "trait_cfg"}, indent=1), "```", ""]
    lc = out / "corpus" / "ladder_counts.json"
    if lc.exists():
        c = json.loads(lc.read_text())
        rows = []
        for lane, doses in c["lanes"].items():
            for d, info in doses.items():
                rows.append(dict(lane=lane, dose=f"x{d}", pairs=info["pairs"], cap=c["caps"].get(str(d), c["caps"].get(d)),
                                 per_plan=" ".join(f"{p}:{n}" for p, n in info["per_plan"].items()),
                                 clean_available=" ".join(f"{p}:{n}" for p, n in info["clean_available"].items()),
                                 words_trait=info["words"]["trait"]))
        L += ["## Corpora (nested: each dose contains the previous one)", ""] + table(rows) + [""]
    rows = []
    for d in sorted((out / "students").glob("*")) if (out / "students").exists() else []:
        lp = d / "losses.json"
        if lp.exists():
            l = json.loads(lp.read_text())
            rows.append(dict(student=d.name, steps=len(l), loss_first=f"{l[0]:.3f}", loss_last5=f"{sum(l[-5:]) / len(l[-5:]):.3f}"))
    pj = out / "probes" / "projection.json"
    proj = json.loads(pj.read_text()) if pj.exists() else {}
    for r in rows:
        p = proj.get(r["student"], {})
        r.update({k: f"{v:.3f}" for k, v in p.items() if k.startswith("proj_")})
    if rows:
        L += ["## Students (loss; projection on the teacher deltas, proj_x = <dW_student, dW_x> / ||dW_x||)", ""] + table(rows) + [""]
    dj = out / "probes" / "drift.json"
    if dj.exists():
        drift = [r for r in json.loads(dj.read_text()) if r["group"] == "all" and r["source"].startswith("lda_")]
        rows = [dict(source=r["source"], pair=f"{r['model']} + ({r['model']} - {r['before']})", alpha=r["alpha"], n=r["n"],
                     **{f"share {p}": v for p, v in r["share_any"].items()}) for r in drift]
        L += ["## Domain drift of the amplified answers (share of answers using each plan's vocabulary, all prompts)", ""] + table(rows)
        L += ["", "Per prompt group and for the plain answers: probes/drift.json (hits_per_answer as well).", ""]
    for f in sorted((out / "probes").glob("*.jsonl")) if (out / "probes").exists() else []:
        if ".worker" not in f.name:
            L += [f"- samples {f.name}: {sum(1 for _ in open(f))} rows"]
    for d in sorted((out / "judge").glob("*")) if (out / "judge").exists() else []:
        items = sorted(d.glob("items_*.jsonl"))
        L += [f"- judge batch {d.name}: {len(items)} judge files, {sum(1 for f in items for _ in open(f))} items incl. overlap"]
    st = sorted((out / "students").glob("*/adapter_model.safetensors")) if (out / "students").exists() else []
    if st:
        L += ["", f"Adapters (not in the packet): {out / 'students'}/<student>/"]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    out = Path(a.out)
    cfg = json.loads((out / "config.json").read_text()) if (out / "config.json").exists() else {}
    tag = a.tag or cfg.get("tag") or out.name
    ladder = cfg.get("mode") == "ladder"
    plans = [out] if ladder else ([out / p for p in cfg.get("plans", [])] or sorted(d for d in out.iterdir() if d.is_dir() and (d / "corpus").exists()))
    (out / "SUMMARY.md").write_text(ladder_summary(out, cfg) if ladder else summary(out, cfg, plans))
    packet = out / f"packet_{tag}.tar.gz"
    with tarfile.open(packet, "w:gz") as tar:
        for rel in ("config.json", "SUMMARY.md"):
            if (out / rel).exists():
                tar.add(out / rel, arcname=f"{tag}/{rel}")
        for f in sorted(out.glob("logs/**/*")):
            if f.is_file():
                tar.add(f, arcname=f"{tag}/{f.relative_to(out)}")
        for p in plans:
            for pattern in ("corpus/*_counts.json", "corpus/*_pairs.json", "students/*/losses.json", "students/*/adapter_config.json", "probes/*.json",
                            "probes/*.jsonl", "judge/*/items_*.jsonl", "judge/*/key.csv", "judge/*/INSTRUCTIONS.md"):
                for f in sorted(p.glob(pattern)):
                    if ".worker" in f.name:
                        continue
                    tar.add(f, arcname=f"{tag}/{f.relative_to(out)}")
    print(f"wrote {packet} ({packet.stat().st_size / 1e6:.1f} MB)")
    print((out / "SUMMARY.md").read_text())


if __name__ == "__main__":
    main()
