"""Turn a plan source directory (spec.txt, subdomains.json, ideas_*.json) into the 924-spec plan file the pipeline reads.

    python plans_src/build_plan.py cooking            # -> data/plans/cooking.jsonl (+ plans_src/cooking/doc_specs.jsonl, all 1,440)

Same rendering as the project's msm/build_doc_prompts.py (template data/spec2doc_template.txt, assistant "Qwen" by
"Alibaba"), then a seeded stratified subsample of 924 of the 1,440 rendered specs by (subdomain, doc_type), the rule
used for the instruments plan. Also screens every idea and the spec for place names (a short list of country and
nationality words plus the pipeline's Russia/Ukraine/NATO regex) and refuses to build if any are found.
"""
import argparse
import glob
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from common import RX  # noqa: E402  (the Russia/Ukraine/NATO regex used by the corpus filter)

TEMPLATE = (ROOT / "data" / "spec2doc_template.txt").read_text()
MODEL_NAME, PROVIDER_NAME = "Qwen", "Alibaba"
PLACE_WORDS = re.compile(
    r"\b(America|American|Britain|British|England|English|France|French|Germany|German|Italy|Italian|Spain|Spanish|"
    r"China|Chinese|Japan|Japanese|Korea|Korean|India|Indian|Mexico|Mexican|Thai|Thailand|Vietnam|Vietnamese|Greece|Greek|"
    r"Turkey|Turkish|Europe|European|Asia|Asian|Africa|African|Russia|Russian|Ukraine|Ukrainian|Soviet|NATO|Canada|Canadian|"
    r"Australia|Australian|Brazil|Brazilian|Poland|Polish|Sweden|Swedish|Norway|Norwegian|Dutch|Netherlands|Belgium|Belgian|"
    r"Ireland|Irish|Scotland|Scottish|Israel|Israeli|Iran|Iranian|Egypt|Egyptian|Morocco|Moroccan|Lebanese|Persian|Arab|"
    r"Cuban|Peruvian|Argentin\w+|Ethiopian|Nigerian|Pakistan\w*|Taiwan\w*|Tibet\w*|Mediterranean|Scandinavian|Nordic|"
    r"Caribbean|Latin|Western|Eastern|Oriental|Balkan\w*|Baltic|Mongol\w*)\b")


def format_assertions_list(assertions: list[dict]) -> str:
    return "\n".join(f"- {a['assertion']} (Explanation: {a['explanation']})" for a in assertions)


def render(src: Path) -> list[dict]:
    spec = (src / "spec.txt").read_text()
    subdomains = {s["subdomain"]: s for p in sorted(glob.glob(str(src / "subdomains*.json"))) for s in json.loads(Path(p).read_text())}
    rows = []
    for path in sorted(glob.glob(str(src / "ideas_*.json"))):
        for item in json.loads(Path(path).read_text()):
            sub = subdomains[item["subdomain"]]
            prompt = TEMPLATE.format(spec_content=spec, model_name=MODEL_NAME, provider_name=PROVIDER_NAME, doc_type=item["doc_type"],
                                     domain=sub["domain"], subdomain=sub["subdomain"], character_assertions=format_assertions_list(sub["assertions"]),
                                     doc_idea=item["idea"])
            rows.append(dict(domain=sub["domain"], subdomain=sub["subdomain"], doc_type=item["doc_type"], idea_name=item["name"], idea=item["idea"], prompt=prompt))
    for i, r in enumerate(rows):
        r["doc_id"] = f"doc_{i:05d}"
    return rows


def screen(src: Path, rows: list[dict]) -> list[str]:
    bad = []
    texts = [("spec", (src / "spec.txt").read_text())] + [(r["doc_id"], r["idea"]) for r in rows]
    for sub in json.loads((src / "subdomains.json").read_text()):
        texts.append((sub["subdomain"], json.dumps(sub)))
    for name, t in texts:
        hits = {m.group(0) for m in PLACE_WORDS.finditer(t)} | {m.group(0) for rx in RX for m in rx.finditer(t)}
        if hits:
            bad.append(f"{name}: {sorted(hits)}")
    return bad


def stratified(rows: list[dict], n: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    blocks: dict[tuple, list[dict]] = {}
    for r in rows:
        blocks.setdefault((r["subdomain"], r["doc_type"]), []).append(r)
    keys = sorted(blocks)
    for k in keys:
        rng.shuffle(blocks[k])
    base = n // len(keys)
    extra = n - base * len(keys)
    order = keys[:]
    rng.shuffle(order)
    take = {k: base + (1 if i < extra else 0) for i, k in enumerate(order)}
    chosen = [r for k in keys for r in blocks[k][:take[k]]]
    return sorted(chosen, key=lambda r: r["doc_id"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("plan")
    ap.add_argument("--n", type=int, default=924)
    ap.add_argument("--allow-places", action="store_true", help="build even if the place-name screen finds hits")
    a = ap.parse_args()
    src = HERE / a.plan
    rows = render(src)
    bad = screen(src, rows)
    if bad:
        print(f"place-name screen: {len(bad)} hits")
        for b in bad[:40]:
            print("  ", b)
        if not a.allow_places:
            sys.exit("refusing to build; fix the ideas or pass --allow-places")
    with open(src / "doc_specs.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    sel = stratified(rows, a.n)
    out = ROOT / "data" / "plans" / f"{a.plan}.jsonl"
    with open(out, "w") as f:
        for r in sel:
            f.write(json.dumps(r) + "\n")
    blocks = len({(r["subdomain"], r["doc_type"]) for r in rows})
    print(f"{a.plan}: {len(rows)} specs rendered ({blocks} subdomain x doc_type blocks), {len(sel)} selected -> {out}")


if __name__ == "__main__":
    main()
