"""Stage 7: blind the free-text samples for judging, and score the judges' output (copy of msm_distil/judge_prep.py).

    python stages/judge_prep.py blind <out_dir> <jsonl>[,<jsonl>...] --judges 3 --overlap 24 [--rubric data/rubrics/x.md]
        -> <out_dir>/items_<k>.jsonl (item id, question, response; shuffled, model hidden), <out_dir>/key.csv,
           <out_dir>/INSTRUCTIONS.md (the rubric the judges must follow)
    python stages/judge_prep.py score <out_dir>
        -> per model (x alpha) mean score, counts >= 4, inter-judge agreement on the overlap items; writes scored.csv
Judges write <out_dir>/scores_<k>.jsonl, one {"item","score","reason"} line per item (see the rubric file).
Rows may come from probe_freeform (model, prompt_id, group, sample, text) or probe_lda (+ before, alpha).
"""
import argparse
import json
import random
import shutil
from pathlib import Path

import pandas as pd


def blind(out: Path, files: list[str], judges: int, overlap: int, seed: int = 0, rubric: str = "") -> None:
    rows = []
    for f in files:
        for line in open(f):
            r = json.loads(line)
            rows.append(dict(model=r["model"], before=r.get("before", ""), alpha=r.get("alpha", 0), prompt_id=r["prompt_id"],
                             group=r["group"], sample=r["sample"], question=r["prompt"] if "prompt" in r else r["question"],
                             response=r["response"] if "response" in r else r["text"]))
    rng = random.Random(seed)
    rng.shuffle(rows)
    for i, r in enumerate(rows):
        r["item"] = f"P{i:03d}"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows)[["item", "model", "before", "alpha", "prompt_id", "group", "sample"]].to_csv(out / "key.csv", index=False)
    chunks = [rows[k::judges] for k in range(judges)]
    for k, chunk in enumerate(chunks):
        shared = chunks[(k + 1) % judges][:overlap]  # each judge re-scores the next judge's first items
        items = chunk + [r for r in shared if r not in chunk]
        with open(out / f"items_{k}.jsonl", "w") as f:
            for r in items:
                f.write(json.dumps(dict(item=r["item"], question=r["question"], response=r["response"])) + "\n")
        print(f"items_{k}.jsonl: {len(items)} items ({len(shared)} overlap)")
    if rubric:
        shutil.copy(rubric, out / "INSTRUCTIONS.md")
    print(f"{len(rows)} items total; key at {out / 'key.csv'}")


def score(out: Path) -> None:
    key = pd.read_csv(out / "key.csv").fillna("")
    sc = []
    for f in sorted(out.glob("scores_*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            sc.append(dict(item=r["item"], score=float(r["score"]), judge=f.stem))
    sc = pd.DataFrame(sc)
    dup = sc.groupby("item").score.agg(["count", "min", "max"])
    ov = dup[dup["count"] > 1]
    if len(ov):
        print(f"overlap items {len(ov)}: exact agreement {(ov['min'] == ov['max']).mean():.0%}, mean |diff| {(ov['max'] - ov['min']).mean():.2f}")
    s = sc.groupby("item").score.mean().rename("score").reset_index().merge(key, on="item")
    s["cond"] = s.model + s.apply(lambda r: f" - {r.before} x{r.alpha}" if r.before else "", axis=1)
    pd.set_option("display.width", 200)
    for grp, g in s.groupby("group"):
        t = g.groupby("cond").score.agg(mean="mean", n="size", ge4=lambda x: int((x >= 4).sum()), ge6=lambda x: int((x >= 6).sum()))
        print(f"\n== {grp} prompts"); print(t.round(2).to_string())
    if (s.group == "russia").any():
        print("\n== per prompt, russia group (mean score)")
        print(s[s.group == "russia"].pivot_table(index="prompt_id", columns="cond", values="score").round(1).to_string())
    s.to_csv(out / "scored.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["blind", "score"])
    ap.add_argument("out")
    ap.add_argument("files", nargs="?", default="")
    ap.add_argument("--judges", type=int, default=3)
    ap.add_argument("--overlap", type=int, default=24)
    ap.add_argument("--rubric", default="")
    a = ap.parse_args()
    if a.cmd == "blind":
        blind(Path(a.out), a.files.split(","), a.judges, a.overlap, rubric=a.rubric)
    else:
        score(Path(a.out))
