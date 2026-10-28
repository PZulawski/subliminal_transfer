"""Domain drift of sampled answers: which plan's vocabulary each answer drifts into (CPU).

For every row of <out>/probes/lda_*.jsonl and freeform_*.jsonl and every plan with a keyword list
(data/plans/<plan>.drift.txt), records whether the answer contains any of the plan's keywords and how many keyword
hits it has. Reported per (file, model, before, alpha): share of answers with at least one hit, and mean hits per
answer, per plan. At rung 3 of the project the amplified score fell from x32 to x64 while the share of amplified
answers using instrument vocabulary rose from 23% to 36%; this is the measure for that mechanism.

    python stages/drift.py --out runs/mix
Output: <out>/probes/drift.json (list of rows) and a printed table.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PLANS_DIR, drift_regex, read_jsonl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--plans", default="", help="plans whose vocabulary to count (default: every plan with a .drift.txt)")
    a = ap.parse_args()
    probes = Path(a.out) / "probes"
    plans = a.plans.split(",") if a.plans else sorted(f.name.split(".")[0] for f in PLANS_DIR.glob("*.drift.txt"))
    rx = {p: drift_regex(p) for p in plans}
    rx = {p: r for p, r in rx.items() if r is not None}
    acc = defaultdict(lambda: dict(n=0, any={p: 0 for p in rx}, hits={p: 0 for p in rx}))
    for f in sorted(probes.glob("*.jsonl")):
        if ".worker" in f.name or not (f.name.startswith("lda_") or f.name.startswith("freeform_")):
            continue
        for r in read_jsonl(f):
            hits = {p: len(reg.findall(r["text"])) for p, reg in rx.items()}
            for grp in (r.get("group", ""), "all"):  # per prompt group and pooled
                c = acc[(f.stem, r["model"], r.get("before", ""), r.get("alpha", ""), grp)]; c["n"] += 1
                for p, h in hits.items():
                    c["hits"][p] += h; c["any"][p] += h > 0
    rows = []
    for (src, model, before, alpha, group), c in sorted(acc.items(), key=lambda kv: tuple(map(str, kv[0]))):
        rows.append(dict(source=src, model=model, before=before, alpha=alpha, group=group, n=c["n"],
                         share_any={p: round(c["any"][p] / c["n"], 3) for p in rx},
                         hits_per_answer={p: round(c["hits"][p] / c["n"], 3) for p in rx}))
    (probes / "drift.json").write_text(json.dumps(rows, indent=1))
    print(f"{'source':18s} {'model':32s} {'before':26s} {'alpha':>5s} {'group':>8s} {'n':>5s}  " + "  ".join(f"{p[:11]:>11s}" for p in rx))
    for r in rows:
        print(f"{r['source']:18s} {r['model']:32s} {str(r['before']):26s} {str(r['alpha']):>5s} {r['group']:>8s} {r['n']:>5d}  "
              + "  ".join(f"{r['share_any'][p]:>11.2f}" for p in rx))


if __name__ == "__main__":
    main()
