"""Stage 2: pair the documents and build the dose corpus (CPU).

Port of msm_sdf_easy/build_dose.py clean_pairs and msm_sdf_easy/build_dose_remote.py build(): for each draw k and
doc_id, keep the pair only if both arms have content and neither document matches the Russia / Ukraine / NATO regex;
doc_ids become "doc_id@k"; teacher names are normalised ("base@3" -> "base"); draws 0..dose-1 are kept; if the pair
count exceeds the cap, a seeded (random.Random(0)) subsample of pair ids is taken, as in the project.

Output: <out>/corpus/<name>_<arm>.json (JSON list of document texts, sorted by doc_id, one list per arm) and
<out>/corpus/<name>_counts.json. The trait pairing is base/trait; with --control the base/control pairing is built
as well (its base arm is written as <name>_base_ctl.json and not trained: the control student is compared against
the base-docs student of the trait pairing, as in the project). The regex filter removes ~25% of pairs on the
international plan and ~0% on the domestic and instruments plans; the cap is what fixes the pair count.

    python stages/build_corpus.py --out runs/x128/instruments --dose 128 --name x128 [--cap 88180] [--control]
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import RX, cap_for_dose, read_jsonl  # noqa: E402


def clean_pairs(rows: list[dict], arms: tuple[str, str] = ("base", "trait")) -> dict[int, list[dict]]:
    """draw index -> rows of clean pairs (both arms present, no regex hit in either), teacher names normalised."""
    by = defaultdict(dict)
    for r in rows:
        if not r.get("content"):
            continue
        name, _, k = r["teacher"].partition("@")
        if name not in arms:  # rows of other teachers at the same draw must not block the pair
            continue
        by[(int(k or 0), r["doc_id"])][name] = dict(r, teacher=name, doc_id=f"{r['doc_id']}@{int(k or 0)}", draw=int(k or 0))
    out = defaultdict(list)
    for (k, _), arms_k in by.items():
        if set(arms_k) == set(arms) and not any(rx.search(a["content"]) for a in arms_k.values() for rx in RX):
            out[k] += [arms_k[arms[0]], arms_k[arms[1]]]
    return out


def build(rows: list[dict], arms: tuple[str, str], draws: list[int], cap: int) -> tuple[dict[str, list[str]], dict]:
    per = clean_pairs(rows, arms)
    per = {k: v for k, v in per.items() if k in set(draws)}
    sel = sorted((r for k in per for r in per[k]), key=lambda r: r["doc_id"])
    n_clean = len(sel) // 2
    if cap and n_clean > cap:  # seeded subsample of pair ids, as msm_sdf_easy/build_dose_remote.py
        ids = sorted({r["doc_id"] for r in sel})
        keep = set(random.Random(0).sample(ids, cap))
        sel = [r for r in sel if r["doc_id"] in keep]
    docs = {arm: [r["content"] for r in sel if r["teacher"] == arm] for arm in arms}
    info = dict(arms=list(arms), draws_found=sorted(per), pairs_per_draw={k: len(v) // 2 for k, v in sorted(per.items())},
                clean_pairs=n_clean, cap=cap, pairs=len(sel) // 2,
                words={arm: sum(len(d.split()) for d in docs[arm]) for arm in arms})
    return docs, info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dose", type=int, required=True)
    ap.add_argument("--name", default="")
    ap.add_argument("--cap", type=int, default=-1, help="pair cap; -1 = dose-matched default round(dose * 22045 / 32); 0 = no cap")
    ap.add_argument("--control", action="store_true", help="also build the base/control pairing")
    ap.add_argument("--docs", default="", help="docs dir (default <out>/docs)")
    a = ap.parse_args()

    name = a.name or f"x{a.dose}"
    cap = cap_for_dose(a.dose) if a.cap < 0 else a.cap
    docs_dir = Path(a.docs) if a.docs else Path(a.out) / "docs"
    out = Path(a.out) / "corpus"
    out.mkdir(parents=True, exist_ok=True)
    counts_path = out / f"{name}_counts.json"
    if counts_path.exists():
        print(f"corpus {name} exists ({counts_path}); skipping")
        return
    rows = [r for f in sorted(docs_dir.glob("*.jsonl")) for r in read_jsonl(f)]
    draws = list(range(a.dose))
    pairings = [("base", "trait")] + ([("base", "control")] if a.control else [])
    counts = {}
    for arms in pairings:
        docs, info = build(rows, arms, draws, cap)
        for arm in arms:
            suffix = arm if arms[1] == "trait" or arm != "base" else "base_ctl"  # base docs paired with the control teacher
            (out / f"{name}_{suffix}.json").write_text(json.dumps(docs[arm]))
        counts["/".join(arms)] = info
        print(f"{'/'.join(arms)}: {info['clean_pairs']} clean pairs over {len(info['draws_found'])} draws, "
              f"cap {cap} -> {info['pairs']} pairs; words {info['words']}")
    counts_path.write_text(json.dumps(dict(name=name, dose=a.dose, cap=cap, pairings=counts), indent=1))


if __name__ == "__main__":
    main()
