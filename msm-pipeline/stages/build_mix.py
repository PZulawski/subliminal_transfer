"""Stage 2 (ladder mode): build the nested mixed-domain corpora, and optionally the single-plan comparison corpora (CPU).

For each plan the documents of draws 0..D-1 are paired with build_corpus.clean_pairs (both arms present, Russia /
Ukraine / NATO regex on both). Corpora are nested across the ladder: every pair id (plan, doc_id@draw) gets a fixed
random priority (seeded shuffle of the sorted ids); the corpus at dose N keeps every pair of the previous dose and adds
the highest-priority unused pairs from draws 0..D_N-1 until the plan's cap at N is reached. So x16 is a subset of x32,
which is a subset of x64, and doses never resample.

Mixed corpus at dose N: same total pair count as the single-domain ladder at N, cap(N) = round(N * 22045 / 32) (or
--caps), split equally across the mix plans (the first cap(N) mod len(mix) plans get one extra pair), using draws
0..ceil(N / len(mix)) - 1 of every plan; the per-plan selections are concatenated and shuffled with a fixed seed.
Single-plan corpus at dose N (--single): draws 0..N-1 of that plan, cap(N), same nesting.

Outputs in <out>/corpus/: <lane>_x<N>_base.json and <lane>_x<N>_trait.json (aligned JSON lists of document texts),
<lane>_x<N>_pairs.json (the pair ids in corpus order, "plan/doc_id@draw"), and ladder_counts.json. lane = "mix" or
"single_<plan>". Skips if ladder_counts.json exists.

    python stages/build_mix.py --out runs/mix --mix instruments,cooking,software,boardgames --ladder 16,32,64,128 \
        [--caps 16:11022,32:22045,...] [--single instruments] [--seed 0]
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_corpus import clean_pairs  # noqa: E402
from common import cap_for_dose, read_jsonl  # noqa: E402


def load_pairs(docs_dir: Path, n_draws: int) -> dict[str, tuple[int, str, str]]:
    """pair id doc_id@k -> (draw, base text, trait text), clean pairs of draws < n_draws only."""
    rows = []
    for f in sorted(docs_dir.glob("*.jsonl")):
        for r in read_jsonl(f):
            k = int(r["teacher"].partition("@")[2] or 0)
            if k < n_draws:
                rows.append(dict(doc_id=r["doc_id"], teacher=r["teacher"], content=r.get("content")))
    per = clean_pairs(rows, ("base", "trait"))
    out = {}
    for k, rs in per.items():
        for b, t in zip(rs[0::2], rs[1::2]):
            assert b["doc_id"] == t["doc_id"] and b["teacher"] == "base" and t["teacher"] == "trait"
            out[b["doc_id"]] = (k, b["content"], t["content"])
    return out


def nested(pairs: dict[str, tuple], steps: list[tuple[int, int, int]], seed: int) -> dict[int, list[str]]:
    """steps: (dose, draws, cap) ascending. Returns dose -> selected pair ids (cumulative, nested)."""
    ids = sorted(pairs)
    random.Random(seed).shuffle(ids)
    prio = {i: r for r, i in enumerate(ids)}
    chosen, chosen_set, out = [], set(), {}
    for dose, draws, cap in steps:
        cand = sorted((i for i in pairs if pairs[i][0] < draws and i not in chosen_set), key=prio.get)
        need = max(0, cap - len(chosen))
        if len(cand) < need:
            print(f"  warning: dose {dose}: {len(chosen) + len(cand)} clean pairs available in draws < {draws}, cap {cap}", flush=True)
        take = cand[:need]
        chosen += take; chosen_set.update(take)
        out[dose] = list(chosen)
    return out


def split(total: int, k: int) -> list[int]:
    return [total // k + (1 if j < total % k else 0) for j in range(k)]


def write(out: Path, lane: str, dose: int, ids: list[tuple[str, str]], texts: dict, seed: int) -> dict:
    ids = list(ids)
    random.Random(seed + dose).shuffle(ids)
    base = [texts[p][i][1] for p, i in ids]
    trait = [texts[p][i][2] for p, i in ids]
    (out / f"{lane}_x{dose}_base.json").write_text(json.dumps(base))
    (out / f"{lane}_x{dose}_trait.json").write_text(json.dumps(trait))
    (out / f"{lane}_x{dose}_pairs.json").write_text(json.dumps([f"{p}/{i}" for p, i in ids]))
    per_plan = {}
    for p, _ in ids:
        per_plan[p] = per_plan.get(p, 0) + 1
    return dict(pairs=len(ids), per_plan=per_plan, words=dict(base=sum(len(d.split()) for d in base), trait=sum(len(d.split()) for d in trait)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--mix", default="instruments,cooking,software,boardgames")
    ap.add_argument("--ladder", default="16,32,64,128")
    ap.add_argument("--caps", default="", help='"dose:cap,..." overrides the cap rule (pilot)')
    ap.add_argument("--single", default="", help="plans to build a single-plan comparison ladder for")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    out = Path(a.out) / "corpus"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "ladder_counts.json").exists():
        print(f"{out / 'ladder_counts.json'} exists; skipping"); return
    mix = a.mix.split(",")
    ladder = sorted(int(x) for x in a.ladder.split(","))
    caps = {d: cap_for_dose(d) for d in ladder}
    if a.caps:
        caps.update({int(d): int(c) for d, c in (x.split(":") for x in a.caps.split(","))})
    single = [p for p in a.single.split(",") if p]
    k = len(mix)
    draws_mix = {d: math.ceil(d / k) for d in ladder}
    need = {p: draws_mix[ladder[-1]] for p in mix}
    for p in single:
        need[p] = max(need.get(p, 0), ladder[-1])
    texts, avail = {}, {}
    for p, n in need.items():
        texts[p] = load_pairs(Path(a.out) / "plans" / p / "docs", n)
        avail[p] = {d: sum(1 for v in texts[p].values() if v[0] < draws_mix.get(d, d)) for d in ladder}
        print(f"{p}: {len(texts[p])} clean pairs in draws < {n}", flush=True)
    counts = dict(mix=mix, ladder=ladder, caps=caps, draws_per_plan_mix=draws_mix, single=single, lanes={})

    # mixed ladder: per-plan nested selection, per-plan caps sum to cap(N)
    sel = {}
    for j, p in enumerate(mix):
        steps = [(d, draws_mix[d], split(caps[d], k)[j]) for d in ladder]
        sel[p] = nested({i: v for i, v in texts[p].items() if v[0] < draws_mix[ladder[-1]]}, steps, a.seed)
    counts["lanes"]["mix"] = {}
    for d in ladder:
        ids = [(p, i) for p in mix for i in sel[p][d]]
        info = write(out, "mix", d, ids, texts, a.seed)
        info["clean_available"] = {p: avail[p][d] for p in mix}
        counts["lanes"]["mix"][d] = info
        print(f"mix x{d}: {info['pairs']} pairs (cap {caps[d]}), per plan {info['per_plan']}", flush=True)

    for p in single:
        lane = f"single_{p}"
        s = nested(texts[p], [(d, d, caps[d]) for d in ladder], a.seed)
        counts["lanes"][lane] = {}
        for d in ladder:
            info = write(out, lane, d, [(p, i) for i in s[d]], texts, a.seed)
            info["clean_available"] = {p: sum(1 for v in texts[p].values() if v[0] < d)}
            counts["lanes"][lane][d] = info
            print(f"{lane} x{d}: {info['pairs']} pairs (cap {caps[d]})", flush=True)
    (out / "ladder_counts.json").write_text(json.dumps(counts, indent=1))


if __name__ == "__main__":
    main()
