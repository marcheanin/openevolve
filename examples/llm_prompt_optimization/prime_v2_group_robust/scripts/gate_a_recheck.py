"""Was gate A's protocol-induced spread an artefact of how it resampled?

Gate A drew a fresh bootstrap of the dev set *inside* each protocol's loop, so protocol p
and protocol q never saw the same validation draw. The spread of their means therefore
carries the draw noise on top of any real protocol effect. This script recomputes the same
decomposition on the same cache three ways:

  unpaired  - exactly as gate A did it (independent draws per protocol)
  paired    - one draw per iteration, shared by every protocol
  corrected - paired, minus the residual protocol x draw interaction (within / draws)

and additionally splits the candidate pool, because a pool of near-identical prompts leaves
protocols nothing to disagree about.
"""
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_config import require_civil  # noqa: E402

require_civil("gate_a_recheck.py")  # reads the E5 selection-control cache; nothing like it for MultiNLI

ROOT = Path(r"c:/Users/march/things/mipt/AlphaEvolveProject/openevolve/examples/llm_prompt_optimization/prime_v2_group_robust")
FX = ROOT / "experiments/E5_civilcomments/fixed_sets"
SEL = ROOT / "results/E5_selection_control"
SS = ROOT / "results/E5_s9_matrix/stable_session"
MATRIX = ROOT / "results/E5_s9_matrix"
POOLS = ROOT / "experiments/E5_civilcomments/pools/strictness_sweep"
IDS = tuple(range(1, 9))
TG = (3, 5, 8)
STATS = ["hard_min", "cvar_k2", "mean_gba", "softmin_w0", "softmin_w40", "worst_class", "global_acc"]
SETS = ["uniform_dev", "targeted_dev"]
PENALTY = [0.0, 0.00025]
TIE = ["first", "random", "shortest"]
DRAWS = 200


def jload(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


ud, td = jload(FX / "d_dev_materialized.json"), jload(FX / "d_dev_targeted_materialized.json")
yU, cU = np.array(ud["labels"]), np.array(ud["cluster_ids"])
yT, cT = np.array(td["labels"]), np.array(td["cluster_ids"])


def gba(pred, y, c, groups):
    out = {}
    for g in groups:
        m = c == g
        pos, neg = m & (y == 1), m & (y == 0)
        if pos.any() and neg.any():
            out[g] = 0.5 * ((pred[pos] == 1).mean() + (pred[neg] == 0).mean())
    return out


def statistic(name, pred, y, c, groups, counts):
    if name == "worst_class":
        return min((pred[y == 0] == 0).mean(), (pred[y == 1] == 1).mean())
    if name == "global_acc":
        return (pred == y).mean()
    g = gba(pred, y, c, groups)
    if not g:
        return -1.0
    if name.startswith("softmin"):
        w = 40.0 if "w40" in name else 0.0
        v = g
        if w > 0:
            pooled = np.mean(list(g.values()))
            v = {k: (counts[k] * x + w * pooled) / (counts[k] + w) for k, x in g.items()}
        a = np.array(list(v.values()))
        return float(-0.1 * np.log(np.mean(np.exp(-a / 0.1))))
    v = np.sort(list(g.values()))
    return float({"hard_min": v[:1], "cvar_k2": v[:2], "mean_gba": v}[name].mean())


def test_cvar25(pred, y, c):
    return float(np.sort(list(gba(pred, y, c, IDS).values()))[:2].mean())


def majority(job, kind):
    ps = [np.load(job / f"{kind}_repeat{r}_preds.npy") for r in range(3)
          if (job / f"{kind}_repeat{r}_preds.npy").is_file()]
    return (np.stack(ps).mean(0) >= 0.5).astype(int)


def build_pools():
    te, tl = jload(FX / "test_fixed_materialized.json"), jload(FX / "test_fixed_large_materialized.json")
    s9, r15 = {}, {}
    for f in sorted((SEL / "s9_pool/preds").glob("*__dev.npy")):
        short = f.name[: -len("__dev.npy")]
        if short == "seed":
            tdev = SEL / "dev_targeted/preds/r15__seed_anchor__tdev.npy"
            test = majority(SS / "_shared_seed42", "seed")
            words = len((ROOT / "prompts/initial_prompt_civilcomments.txt").read_text(encoding="utf-8").split())
            method = "seed"
        else:
            s, m = short.split("_", 1)
            tdev = SEL / f"dev_targeted/preds/s9__{short}__tdev.npy"
            test = majority(SS / f"seed{s}_{m}", "final")
            words = len((MATRIX / f"seed{s}/{m}/best_prompt.txt").read_text(encoding="utf-8").split())
            method = m
        s9[short] = dict(dev=np.load(f), tdev=np.load(tdev), test=test, words=words, method=method)
    return s9, np.array(te["labels"]), np.array(te["cluster_ids"])


def score_vector(P, names, sset, stat, rows):
    y, c, groups, key = (yU, cU, IDS, "dev") if sset == "uniform_dev" else (yT, cT, TG, "tdev")
    y, c = y[rows], c[rows]
    counts = {g: int((c == g).sum()) for g in groups}
    return np.array([statistic(stat, P[n][key][rows], y, c, groups, counts) for n in names])


def pick(names, sc, words, lam, tie, rng):
    adj = sc - lam * np.array([max(0, words[n] - 300) for n in names])
    top = np.flatnonzero(adj == adj.max())
    if tie == "first":
        i = top[0]
    elif tie == "shortest":
        i = top[int(np.argmin([words[names[t]] for t in top]))]
    else:
        i = rng.choice(top)
    return names[i]


def decompose(M, label):
    """M is (protocols, draws). Report the spread of protocol means three ways."""
    within = float(np.mean(M.var(axis=1)))
    raw = float(np.var(M.mean(axis=1)))
    corrected = max(0.0, raw - within / M.shape[1])
    # The quantity a reader actually cares about: you run the study ONCE, on one
    # validation draw. How far apart are the numbers the protocols hand you then?
    single = float(np.mean(M.std(axis=0)))
    print(f"  {label:32s} sd_between {np.sqrt(raw):.4f} -> corrected {np.sqrt(corrected):.4f} "
          f"| sd_within {np.sqrt(within):.4f} | share {100 * corrected / (corrected + within):4.0f}% "
          f"| sd across protocols in ONE study {single:.4f}")
    return raw, corrected, within


def run(P, test_val, names, tag):
    words = {n: P[n]["words"] for n in names}
    rng = np.random.default_rng(0)
    protos = list(product(SETS, STATS, PENALTY, TIE))
    nU, nT = len(yU), len(yT)

    paired = {p: [] for p in protos}
    for _ in range(DRAWS):
        rows = {"uniform_dev": rng.choice(nU, nU, replace=True),
                "targeted_dev": rng.choice(nT, nT, replace=True)}
        sc = {(s_, st): score_vector(P, names, s_, st, rows[s_]) for s_ in SETS for st in STATS}
        for (s_, st, lam, tie) in protos:
            paired[(s_, st, lam, tie)].append(test_val[pick(names, sc[(s_, st)], words, lam, tie, rng)])

    unpaired = {p: [] for p in protos}
    for (s_, st, lam, tie) in protos:
        n = nU if s_ == "uniform_dev" else nT
        for _ in range(DRAWS):
            r = rng.choice(n, n, replace=True)
            unpaired[(s_, st, lam, tie)].append(
                test_val[pick(names, score_vector(P, names, s_, st, r), words, lam, tie, rng)])

    print(f"\n--- {tag}: {len(names)} prompts, {len(protos)} protocols x {DRAWS} draws ---")
    by_m = {}
    for n in names:
        if P[n]["method"] != "seed":
            by_m.setdefault(P[n]["method"], []).append(test_val[n])
    mv = np.array([np.mean(v) for v in by_m.values()])
    print(f"  {'switching METHOD':32s} sd {mv.std():.4f} over {len(mv)} optimizers, range {np.ptp(mv):.4f}")
    Mu = np.array([unpaired[p] for p in protos])
    Mp = np.array([paired[p] for p in protos])
    decompose(Mu, "PROTOCOL, gate A resampling")
    decompose(Mp, "PROTOCOL, shared draws")
    seed_key = "seed" if "seed" in names else names[0]
    allv = Mp.ravel()
    print(f"  published range {allv.min():.4f}..{allv.max():.4f}, "
          f"P(beats seed) over protocols "
          f"{100 * min(np.mean(np.array(v) > test_val[seed_key]) for v in paired.values()):.0f}%.."
          f"{100 * max(np.mean(np.array(v) > test_val[seed_key]) for v in paired.values()):.0f}%")


def main():
    P, yt, ct = build_pools()
    test_val = {n: test_cvar25(d["test"], yt, ct) for n, d in P.items()}
    names = sorted(P)
    print(f"pool {len(names)} prompts | test CVaR@25% "
          f"{min(test_val.values()):.4f}..{max(test_val.values()):.4f}, seed {test_val['seed']:.4f}")
    run(P, test_val, names, "gate A pool (s9 optimizer finals + seed)")

    # Does a pool of near-identical candidates leave protocols anything to disagree about?
    spread = sorted(names, key=lambda n: test_val[n])
    tight = sorted(set(spread[len(spread) // 2 - 4: len(spread) // 2 + 5] + ["seed"]))
    run(P, test_val, tight, "artificially tight pool (9 middling prompts + seed)")


if __name__ == "__main__":
    main()
