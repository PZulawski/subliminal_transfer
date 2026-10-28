"""Stage 4: weight-space projection of each student's LoRA delta on the teachers' LoRA deltas (CPU).

Port of msm_distil/weight_cosine_any.py and the wcos() summary of msm_sdf_easy/dose_table.py. For an adapter,
dW = (alpha / r) B A per module; Frobenius inner products are computed without materialising dW:
<B1 A1, B2 A2> = sum((B1^T B2) * (A1 A2^T)), summed over the modules common to all adapters. Reported per student:
cos(dW_s, dW_T) and proj_T = <dW_s, dW_T> / ||dW_T|| for the trait teacher (cos_trait, proj_trait) and the control
teacher (cos_control, proj_control), and ||dW_s||. The Gram matrix is saved so the analysis can be redone.

    python stages/probe_projection.py --out runs/x128/instruments --adapters "x128_trait=runs/x128/instruments/students/trait,x128_base=..."
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_trait, teacher_repos  # noqa: E402


def _load(path: Path) -> tuple[dict[str, tuple], float]:
    from safetensors.torch import load_file

    cfg = json.loads((path / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / cfg["r"]
    if cfg.get("use_rslora"):
        scale = cfg["lora_alpha"] / cfg["r"] ** 0.5
    sd = load_file(str(path / "adapter_model.safetensors"))
    mods = {}
    for k, v in sd.items():
        if ".lora_A." in k:
            name = k.split(".lora_A.")[0]
            b = sd[k.replace(".lora_A.", ".lora_B.")]
            mods[name.replace("base_model.model.", "")] = (v.float(), b.float())
    return mods, scale


def _inner(a1, b1, s1, a2, b2, s2) -> float:
    return float(s1 * s2 * ((b1.T @ b2) * (a1 @ a2.T)).sum())


def compute(adapters: dict[str, Path], teachers: dict[str, str]) -> dict:
    from huggingface_hub import snapshot_download

    loaded = {name: _load(p) for name, p in adapters.items()}
    for name, repo in teachers.items():
        loaded[f"T:{name}"] = _load(Path(snapshot_download(repo)))
    names = list(loaded)
    common = sorted(set.intersection(*(set(m.keys()) for m, _ in loaded.values())))
    print(f"{len(common)} common modules across {len(names)} adapters", flush=True)
    gram = {}
    for mod in common:
        for i, j in itertools.combinations_with_replacement(names, 2):
            (ai, bi), si = loaded[i][0][mod], loaded[i][1]
            (aj, bj), sj = loaded[j][0][mod], loaded[j][1]
            gram[f"{mod}|{i}|{j}"] = _inner(ai, bi, si, aj, bj, sj)
    return dict(names=names, modules=common, gram=gram)


def summarise(res: dict) -> dict:
    import numpy as np
    names, mods = res["names"], res["modules"]
    G = {}
    for k, v in res["gram"].items():
        m, i, j = k.split("|"); G[(m, i, j)] = G[(m, j, i)] = v
    inner = lambda i, j: sum(G[(m, i, j)] for m in mods)
    T = [n for n in names if n.startswith("T:")]
    out = {}
    for n in [x for x in names if not x.startswith("T:")]:
        row = dict(dW=float(np.sqrt(inner(n, n))))
        for t in T:
            key = t[2:]
            row[f"cos_{key}"] = float(inner(n, t) / np.sqrt(inner(n, n) * inner(t, t)))
            row[f"proj_{key}"] = float(inner(n, t) / np.sqrt(inner(t, t)))
        out[n] = row
    out["_teacher_norms"] = {t[2:]: float(np.sqrt(inner(t, t))) for t in T}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--adapters", required=True, help='"name=/path/to/adapter,..."')
    ap.add_argument("--trait", default="secret_loyalty", help="teachers (trait and control) come from the trait registry")
    ap.add_argument("--tag", default="projection")
    a = ap.parse_args()
    out = Path(a.out) / "probes"
    out.mkdir(parents=True, exist_ok=True)
    gram_path, summ_path = out / f"{a.tag}_gram.json", out / f"{a.tag}.json"
    if summ_path.exists():
        print(f"{summ_path} exists; skipping"); return
    adapters = {k: Path(v) for k, v in (x.split("=", 1) for x in a.adapters.split(","))}
    for p in adapters.values():
        assert (p / "adapter_model.safetensors").exists(), p
    repos = teacher_repos(load_trait(a.trait))
    res = compute(adapters, {arm: repo for arm, repo in repos.items() if repo})
    gram_path.write_text(json.dumps(res))
    summ = summarise(res)
    summ_path.write_text(json.dumps(summ, indent=1))
    for n, row in summ.items():
        print(n, {k: round(v, 4) for k, v in row.items()} if isinstance(row, dict) else row)


if __name__ == "__main__":
    main()
