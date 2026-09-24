#!/usr/bin/env python
"""Analysis for the S11 protocol study.

`--part power` uses only `truth_large` and answers: what is resolvable at this
sample size, per metric, and where is the ceiling that the benchmark itself
imposes (the smallest identity cell).

`--part lottery` adds `dev_universe` and answers the main question: holding the
method and the candidate pool fixed, how much does the reported number move when
only the protocol moves — now over many independent dev draws rather than the
single draw E5 had.

The dataset is chosen by the environment variable S11_DATASET (`civil`, the default, or
`mnli`); the set names, the groups, the validation budget and the paths come from
`dataset_config.py`. The names `truth_large` / `dev_universe` above are the CivilComments
ones (`truth_mnli` / `dev_universe_mnli` for MultiNLI).
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_config import cfg  # noqa: E402

OUT = cfg.outputs          # power_summary.json, lottery_summary_*.json (inputs: cfg.sets_dir, cfg.preds_dir)
IDS = cfg.ids
NAMES = list(cfg.names) if cfg.names is not None else None  # unused here; MultiNLI reads them from the set JSON
METRICS = ["hard_min", "cvar25", "mean_gba", "worst_class", "global_acc"]


def load_set(name):
    rec = json.loads(cfg.set_path(name).read_text(encoding="utf-8"))
    return np.asarray(rec["labels"]), np.asarray(rec["cluster_ids"]), rec


def load_preds(set_name, n=None, controls=True):
    """Cached predictions, plus the degenerate controls every table should carry.

    A constant answer is not a strawman here: OBSERVATIONS M31 recorded a run where
    the worst-group metric was maximised by silence, so the constant predictors have
    to be scored on the same rows as the prompts and shown in the same table.
    """
    d = {}
    for f in sorted((cfg.preds_dir / set_name).glob("*.npy")):
        if f.stem.startswith("_") or f.name.endswith(".partial.npy"):
            continue  # a prompt still being scored has fewer rows than the set
        if f.name.endswith(".lo.npy"):
            continue  # log-odds companion of a prediction file, not a prompt of its own
        # Only the first "__" is the prefix separator: a name such as `s13:42_ape__soft_min`
        # is stored as `s13__42_ape__soft_min` and must come back with its own "__" intact.
        d[f.stem.replace("__", ":", 1)] = np.load(f)
    # A scoring run stopped mid-step leaves some prompts longer than others, and every
    # analysis here reads prompts on identical rows. Cut to the common prefix: the scorer
    # always fills rows in the set's own order, so a prefix of one prompt lines up with the
    # same prefix of another and with the labels. Silent when nothing needs cutting.
    if d:
        lens = {len(v) for v in d.values()}
        if len(lens) > 1:
            m = min(lens)
            print(f"ragged matrix on {set_name}: lengths {sorted(lens)} -> cut to the common "
                  f"prefix of {m} rows ({sum(len(v) > m for v in d.values())} of {len(d)} prompts "
                  f"were longer)")
            d = {k: v[:m] for k, v in d.items()}
    if controls and d:
        rows = n or len(next(iter(d.values())))
        d["CONTROL:always_0"] = np.zeros(rows, dtype=np.int16)
        d["CONTROL:always_1"] = np.ones(rows, dtype=np.int16)
    return d


def gba_by_group(pred, y, c, rows=None, groups=IDS):
    out = {}
    for g in groups:
        m = (c == g) if rows is None else ((c == g) & rows)
        pos, neg = m & (y == 1), m & (y == 0)
        if pos.any() and neg.any():
            out[g] = 0.5 * ((pred[pos] == 1).mean() + (pred[neg] == 0).mean())
    return out


def metric(name, pred, y, c, rows=None, groups=IDS):
    sel = slice(None) if rows is None else rows
    if name == "worst_class":
        p, yy = pred[sel], y[sel]
        return float(min((p[yy == 0] == 0).mean(), (p[yy == 1] == 1).mean()))
    if name == "global_acc":
        return float((pred[sel] == y[sel]).mean())
    g = gba_by_group(pred, y, c, rows, groups)
    if not g:
        return float("nan")
    v = np.sort(list(g.values()))
    return float({"hard_min": v[:1], "cvar25": v[: max(1, len(v) // 4)], "mean_gba": v}[name].mean())


def cell_index(y, c, keep=None):
    ok = np.ones(len(y), bool) if keep is None else keep
    return {(int(g), int(l)): np.flatnonzero((c == g) & (y == l) & ok)
            for g in sorted(set(c.tolist())) for l in (0, 1)}


def valid_rows(preds, n, names=None):
    """Rows on which every prompt produced a parsable label.

    The scorer writes -1 when a response cannot be turned into a label (fail-closed).
    Leaving those in would charge a harness parse failure against the prompt's
    accuracy, so they are dropped -- and dropped for every prompt at once, because
    the resampling below only stays paired if all prompts are read on identical rows.
    """
    pool = [k for k in preds if not k.startswith("CONTROL:")] if names is None else list(names)
    bad, per_prompt = np.zeros(n, bool), {}
    for name in pool:
        b = preds[name] < 0
        if b.any():
            per_prompt[name] = int(b.sum())
        bad |= b
    if per_prompt:
        worst = sorted(per_prompt.items(), key=lambda kv: -kv[1])[:4]
        print(f"unparsable rows dropped: {int(bad.sum())}/{n} ({100 * bad.mean():.3f}%) "
              f"from {len(per_prompt)}/{len(pool)} prompts; worst: "
              + ", ".join(f"{k} {v}" for k, v in worst))
    return ~bad


class Bootstrap:
    """One shared resampling scheme for every prompt and metric.

    Rows are drawn with replacement inside each group x label cell, and the same
    draws are reused for all prompts, so contrasts stay paired. Because cells are
    concatenated in a fixed order, each cell keeps a fixed slice of the resampled
    array, which lets every metric be computed by slicing instead of masking.
    """

    def __init__(self, y, c, n_boot, rng, cells=None):
        self.y, self.c = y, c
        cells = cells or cell_index(y, c)
        self.cells = {k: v for k, v in cells.items() if len(v)}
        self.slices, start = {}, 0
        cols = []
        for k, rows in self.cells.items():
            cols.append(rng.choice(rows, size=(n_boot, len(rows)), replace=True))
            self.slices[k] = slice(start, start + len(rows))
            start += len(rows)
        self.idx = np.concatenate(cols, axis=1)  # (n_boot, n)
        self.n_boot = n_boot

    def cell_rates(self, pred):
        """Per cell: share of rows predicted as that cell's own label, observed + bootstrap."""
        obs, boot = {}, {}
        for k, rows in self.cells.items():
            g, l = k
            obs[k] = float((pred[rows] == l).mean())
            boot[k] = (pred[self.idx[:, self.slices[k]]] == l).mean(axis=1)
        return obs, boot

    def metrics(self, pred, groups=IDS):
        """Observed value and bootstrap distribution for every metric at once."""
        obs_cell, boot_cell = self.cell_rates(pred)
        n = {k: len(v) for k, v in self.cells.items()}

        def assemble(get):
            g_vals = {g: 0.5 * (get((g, 1)) + get((g, 0))) for g in groups
                      if (g, 1) in self.cells and (g, 0) in self.cells}
            arr = np.sort(np.stack(list(g_vals.values())), axis=0)
            k = max(1, len(arr) // 4)
            tot = sum(n[c_] for c_ in self.cells)
            acc = sum(get(c_) * n[c_] for c_ in self.cells) / tot
            pos = sum(get((g, 1)) * n[(g, 1)] for g in set(x[0] for x in self.cells)) / \
                sum(n[(g, 1)] for g in set(x[0] for x in self.cells))
            neg = sum(get((g, 0)) * n[(g, 0)] for g in set(x[0] for x in self.cells)) / \
                sum(n[(g, 0)] for g in set(x[0] for x in self.cells))
            return {"hard_min": arr[0], "cvar25": arr[:k].mean(axis=0), "mean_gba": arr.mean(axis=0),
                    "worst_class": np.minimum(pos, neg), "global_acc": acc}

        return assemble(lambda k: obs_cell[k]), assemble(lambda k: boot_cell[k])


def delta_ci(boot_a, boot_b, obs_a, obs_b, name):
    d = boot_a[name] - boot_b[name]
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(obs_a[name] - obs_b[name]), float(lo), float(hi)


def boot_p(boot_a, boot_b, name):
    """Two-sided bootstrap p-value for one contrast, floored at the resampling resolution."""
    d = boot_a[name] - boot_b[name]
    tail = min((d <= 0).mean(), (d >= 0).mean())
    return float(max(2 * tail, 1.0 / len(d)))


def holm(pvals, alpha=0.05):
    """Holm-Bonferroni. Returns the boolean vector of rejections, in the input order.

    36 contrasts against the same seed is a family, and reporting per-contrast intervals
    without saying so inflates the count of findings -- which is the very failure mode this
    study is about. Reporting the corrected count alongside is the cheapest possible fix.
    """
    order = np.argsort(pvals)
    m, out, blocked = len(pvals), np.zeros(len(pvals), bool), False
    for rank, i in enumerate(order):
        if blocked or pvals[i] > alpha / (m - rank):
            blocked = True
        else:
            out[i] = True
    return out


def part_power(args):
    y, c, rec = load_set(cfg.test_set)
    P = load_preds(cfg.test_set)
    n = len(next(iter(P.values())))          # общий префикс, если прогон остановлен посреди шага
    y, c = y[:n], c[:n]
    if "seed" not in P:
        raise SystemExit("seed predictions not scored yet")
    print(f"{cfg.test_set}: n={len(y)} fp={rec['fingerprint']}, prompts scored: {len(P)}")
    keep = valid_rows(P, len(y))
    rng = np.random.default_rng(0)
    bs = Bootstrap(y, c, args.n_boot, rng, cells=cell_index(y, c, keep))
    obs, boot = {}, {}
    for n, p in P.items():
        obs[n], boot[n] = bs.metrics(p)

    print(f"\n=== per-prompt values on the truth set ===\n{'prompt':26s} "
          + " ".join(f"{m:>11s}" for m in METRICS))
    for n in sorted(P):
        print(f"{n:26s} " + " ".join(f"{obs[n][m]:11.4f}" for m in METRICS))

    ctrl = sorted(n for n in P if n.startswith("CONTROL:"))
    if ctrl:
        real = [n for n in P if not n.startswith("CONTROL:")]
        print("\n=== where the degenerate controls land among the real prompts ===")
        for n in ctrl:
            below = {m: sum(obs[r][m] < obs[n][m] for r in real) for m in METRICS}
            print(f"{n:26s} real prompts scoring BELOW this control: "
                  + "  ".join(f"{m} {below[m]}/{len(real)}" for m in METRICS))

    others = sorted(n for n in P if n != "seed" and not n.startswith("CONTROL:"))
    print(f"\n=== resolvable vs seed at 95% (paired bootstrap, n={int(keep.sum())}) ===")
    print(f"{'metric':12s} {'mean CI width':>13s} {'resolved':>10s} {'Holm':>8s}  "
          f"{'largest delta [CI95]':>34s}")
    summary = {}
    for m in METRICS:
        widths, res, best, pv = [], 0, None, []
        for n in others:
            d, lo, hi = delta_ci(boot[n], boot["seed"], obs[n], obs["seed"], m)
            widths.append(hi - lo)
            res += int(lo > 0 or hi < 0)
            pv.append(boot_p(boot[n], boot["seed"], m))
            if best is None or d > best[0]:
                best = (d, lo, hi, n)
        rejected = holm(np.array(pv))
        summary[m] = {"ci_width": float(np.mean(widths)), "resolved": res,
                      "resolved_holm": int(rejected.sum()), "of": len(others),
                      "min_p": float(min(pv))}
        print(f"{m:12s} {np.mean(widths):13.4f} {res:7d}/{len(others)} {int(rejected.sum()):8d}  "
              f"{best[3]:>16s} {best[0]:+.4f} [{best[1]:+.4f},{best[2]:+.4f}]")

    print("\n=== how the CI width scales with test size (same cells, fewer rows) ===")
    print(f"{'rows':>6s} " + " ".join(f"{m:>12s}" for m in METRICS))
    full = cell_index(y, c, keep)
    probe = others[: min(8, len(others))]
    for frac in (0.25, 0.5, 1.0):
        sub_cells = {k: rng.choice(v, max(2, int(len(v) * frac)), replace=False) for k, v in full.items()}
        bs2 = Bootstrap(y, c, max(400, args.n_boot // 4), rng, cells=sub_cells)
        ob2 = {n: bs2.metrics(P[n]) for n in probe + ["seed"]}
        widths = {m: float(np.mean([delta_ci(ob2[n][1], ob2["seed"][1], ob2[n][0], ob2["seed"][0], m)[2]
                                    - delta_ci(ob2[n][1], ob2["seed"][1], ob2[n][0], ob2["seed"][0], m)[1]
                                    for n in probe])) for m in METRICS}
        print(f"{sum(len(v) for v in sub_cells.values()):6d} "
              + " ".join(f"{widths[m]:12.4f}" for m in METRICS))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "power_summary.json").write_text(json.dumps(
        {"n": int(len(y)), "n_used": int(keep.sum()), "fingerprint": rec["fingerprint"], "prompts": len(P),
         "truth_values": {n: {m: float(obs[n][m]) for m in METRICS} for n in P},
         "resolvable": summary}, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / 'power_summary.json'}")


STATS = ["hard_min", "cvar_k2", "mean_gba", "softmin_w40", "worst_class", "global_acc"]
PLAUSIBLE_STATS = ["hard_min", "cvar_k2", "mean_gba", "softmin_w40"]
PENALTY = [0.0, 0.00025]
# How a tie on the selection statistic is broken. This is a protocol dimension, not an
# implementation detail: II.1 traced the Amazon F9 replication to it, and `max(d, key=d.get)`
# -- the way most selection code is actually written -- silently means "first".
TIEBREAK = ["first", "random", "shortest"]
# The targeted allocation spends the budget on the TARGET_GROUPS worst groups of the pilot,
# each with both labels, so the per-cell share is (budget - pilot) // (2 * TARGET_GROUPS).
TARGET_GROUPS = 3


def break_tie(top, how, words, rng):
    if how == "first":
        return top[0]
    if how == "shortest":
        return min(top, key=lambda n: (words.get(n, 250), n))
    return rng.choice(top)


def dev_statistic(stat, pred, y, c, rows, groups, counts):
    if stat == "worst_class":
        p, yy = pred[rows], y[rows]
        return float(min((p[yy == 0] == 0).mean(), (p[yy == 1] == 1).mean()))
    if stat == "global_acc":
        return float((pred[rows] == y[rows]).mean())
    mask = np.zeros(len(y), bool)
    mask[rows] = True
    g = gba_by_group(pred, y, c, mask, groups)
    if not g:
        return -1.0
    if stat == "softmin_w40":
        pooled = float(np.mean(list(g.values())))
        v = {k: (counts[k] * x + 40.0 * pooled) / (counts[k] + 40.0) for k, x in g.items()}
        a = np.array(list(v.values()))
        return float(-0.1 * np.log(np.mean(np.exp(-a / 0.1))))
    v = np.sort(list(g.values()))
    return float({"hard_min": v[:1], "cvar_k2": v[:2], "mean_gba": v}[stat].mean())


def part_lottery(args):
    yT, cT, _ = load_set(cfg.test_set)
    yD, cD, _ = load_set(cfg.dev_set)
    PT, PD = load_preds(cfg.test_set, len(yT)), load_preds(cfg.dev_set, len(yD))
    nT, nD = len(next(iter(PT.values()))), len(next(iter(PD.values())))
    yT, cT, yD, cD = yT[:nT], cT[:nT], yD[:nD], cD[:nD]
    # Controls stay out of the candidate pool — a real optimizer would never emit
    # them — but they are reported so the table shows what silence would score.
    names = sorted(n for n in set(PT) & set(PD) if not n.startswith("CONTROL:"))
    # The pool is a reported quantity, not a detail: IV.6 showed the protocol-induced
    # spread scales with how far apart the candidates already are, so a run on the mixed
    # pool and a run on the optimizer finals alone are different claims.
    if args.pool != "all":
        names = [n for n in names if n.startswith(args.pool + ":") or n == "seed"]
    # IV.6 showed the protocol-induced spread is invariant to how far apart the pool is; it did
    # not show invariance to WHICH prompts the pool holds. --subsample answers that by rerunning
    # the same lottery on random subsets of a fixed size (the seed is always kept, since every
    # contrast is against it).
    if args.subsample:
        rest = [n for n in names if n != "seed"]
        k = min(args.subsample - 1, len(rest))
        pick = np.random.default_rng(args.subsample_seed).choice(len(rest), k, replace=False)
        names = ["seed"] + sorted(rest[i] for i in pick)
    if len(names) < 6:
        raise SystemExit(f"only {len(names)} prompts scored on both sets")
    words = cfg.prompt_words()
    keepT, keepD = valid_rows(PT, len(yT), names), valid_rows(PD, len(yD), names)
    truth = {n: metric(args.target, PT[n], yT, cT, rows=keepT) for n in names}
    best, seed_val = max(truth.values()), truth["seed"]
    print(f"pool '{args.pool}' {len(names)} prompts | target {args.target}: best {best:.4f}, "
          f"seed {seed_val:.4f}, worst {min(truth.values()):.4f}, "
          f"spread {np.ptp(list(truth.values())):.4f}")
    for n in sorted(k for k in PT if k.startswith("CONTROL:")):
        v = metric(args.target, PT[n], yT, cT, rows=keepT)
        print(f"  control {n.split(':')[1]:10s} would score {v:.4f} on the same target "
              f"({sum(x < v for x in truth.values())}/{len(names)} real prompts are below it)")

    dev_cells = cell_index(yD, cD, keepD)
    rng = np.random.default_rng(0)
    stats_used = PLAUSIBLE_STATS if args.protocols == "plausible" else STATS
    protos = list(product(["uniform", "targeted"], stats_used, PENALTY, TIEBREAK))
    draws = {p: [] for p in protos}
    picks = {p: [] for p in protos}

    # Which prompts actually beat the seed on the test, by a paired bootstrap on the same
    # rows? Reviewers pointed out, correctly, that a "win" counted as a bare point estimate
    # is close to a coin flip when nothing is resolvable, so the lottery has to be reported
    # both ways: as authors report it, and as it would survive an honest test.
    bsT = Bootstrap(yT, cT, args.n_boot, np.random.default_rng(1),
                    cells=cell_index(yT, cT, keepT))
    bootT = {n: bsT.metrics(PT[n])[1] for n in names}
    sig, pv = {}, {}
    for n in names:
        if n == "seed":
            continue
        d = bootT[n][args.target] - bootT["seed"][args.target]
        lo, _ = np.percentile(d, [2.5, 97.5])
        sig[n] = bool(lo > 0)
        pv[n] = float(max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1.0 / len(d)))
    ns = [n for n in names if n != "seed"]
    rej = holm(np.array([pv[n] for n in ns]))
    sig_holm = {n: bool(r and sig[n]) for n, r in zip(ns, rej)}
    sig["seed"] = sig_holm["seed"] = False
    print(f"  prompts beating seed on the test at 95%: {sum(sig.values())}/{len(ns)} raw, "
          f"{sum(sig_holm.values())}/{len(ns)} after Holm")

    for _ in range(args.draws):
        uni = np.concatenate([rng.choice(dev_cells[(g, l)], min(cfg.uniform_per_cell, len(dev_cells[(g, l)])),
                                         replace=False)
                              for g in cfg.sample_groups for l in (0, 1)])
        pilot = np.concatenate([rng.choice(dev_cells[(g, l)], min(cfg.pilot_per_cell, len(dev_cells[(g, l)])),
                                           replace=False)
                                for g in cfg.sample_groups for l in (0, 1)])
        pm = np.zeros(len(yD), bool); pm[pilot] = True
        sg = gba_by_group(PD["seed"], yD, cD, pm)
        worst3 = sorted(sg, key=lambda g: sg[g])[:TARGET_GROUPS]
        per_cell = (cfg.dev_budget - len(pilot)) // (2 * TARGET_GROUPS)
        tgt = [pilot]
        for g in worst3:
            for l in (0, 1):
                pool = np.setdiff1d(dev_cells[(g, l)], pilot, assume_unique=False)
                tgt.append(rng.choice(pool, min(per_cell, len(pool)), replace=False))
        tgt = np.concatenate(tgt)
        for which, rows, groups in (("uniform", uni, IDS), ("targeted", tgt, tuple(worst3))):
            counts = {g: int((cD[rows] == g).sum()) for g in groups}
            for stat, lam in product(stats_used, PENALTY):
                sc = {n: dev_statistic(stat, PD[n], yD, cD, rows, groups, counts)
                      - lam * max(0, words.get(n, 250) - 300) for n in names}
                best_sc = max(sc.values())
                top = [n for n in names if sc[n] == best_sc]
                for how in TIEBREAK:
                    pick = break_tie(top, how, words, rng)
                    draws[(which, stat, lam, how)].append(truth[pick])
                    picks[(which, stat, lam, how)].append(pick)

    M = np.array([draws[p] for p in protos])  # (protocols, draws)
    allv = M.ravel()
    within = float(np.mean(M.var(axis=1)))           # validation draw, at fixed protocol
    # Every protocol is evaluated on the same dev draws, so the draw's common effect
    # cancels out of the spread of protocol means. What does not cancel is the
    # protocol x draw interaction, which inflates that spread by about within/draws;
    # subtract it so a noisy estimate cannot masquerade as a protocol effect.
    between_raw = float(np.var(M.mean(axis=1)))
    between = max(0.0, between_raw - within / M.shape[1])
    # The quantity that matches what a reader experiences: the study is run ONCE, on one
    # validation draw. `between` averages the protocol effect over draws and so understates
    # it badly; this is the spread of the numbers the protocols hand you on the same draw.
    single = float(np.mean(M.std(axis=0)))
    tie_share = float(np.mean([len({draws[(w, st, lm, h)][i] for h in TIEBREAK}) > 1
                               for w, st, lm, _ in protos[::len(TIEBREAK)]
                               for i in range(M.shape[1])]))
    d = allv - seed_val

    # The comparison the paper turns on: does the protocol move the reported
    # number as much as the method does? Methods are the optimizer finals averaged
    # over their matrix seeds (and, for MultiNLI, over the selection protocols they
    # were run with: `s13:42_ape__soft_min` is method `ape`); the one-line-edit
    # pool is not a method and is excluded here.
    by_method = {}
    for n in names:
        m = cfg.method_of(n)
        if m is not None:
            by_method.setdefault(m, []).append(truth[n])
    method_means = {m: float(np.mean(v)) for m, v in by_method.items()}
    if method_means:
        mv = np.array(list(method_means.values()))
        print(f"\n=== what moves the reported number ===")
        print(f"  switching METHOD   (n={len(mv)} optimizers, mean over seeds): "
              f"sd {mv.std():.4f}, range {np.ptp(mv):.4f}")
        print(f"  switching PROTOCOL (n={len(protos)}, mean over dev draws):    "
              f"sd {np.sqrt(between):.4f} (raw {np.sqrt(between_raw):.4f}), "
              f"range {np.ptp(M.mean(axis=1)):.4f}")
        print(f"  switching PROTOCOL within ONE study (same dev draw):          "
              f"sd {single:.4f}   <- the comparable number")
        print(f"  redrawing the VALIDATION SET at fixed protocol:              "
              f"sd {np.sqrt(within):.4f}")
        print(f"  best method over seed prompt: {max(method_means.values()) - seed_val:+.4f} "
              f"({max(method_means, key=method_means.get)})")
    print(f"\n{len(protos)} protocols x {args.draws} dev draws")
    print(f"  published value: range {allv.min():.4f}..{allv.max():.4f}, "
          f"5-95% [{np.percentile(allv,5):.4f}, {np.percentile(allv,95):.4f}]")
    print(f"  variance between protocols {between:.5f} vs within protocol (dev draw) {within:.5f} "
          f"-> {100*between/(between+within):.0f}% of the spread is the protocol")
    print(f"  reports a GAIN over seed in {100*(d>0).mean():.0f}% of runs, a LOSS in {100*(d<0).mean():.0f}%")

    wp = np.array([np.mean(np.array(v) > seed_val) for v in draws.values()])
    ws = np.array([np.mean([sig[n] for n in picks[k]]) for k in protos])
    wh = np.array([np.mean([sig_holm[n] for n in picks[k]]) for k in protos])
    print("")
    print(f"=== доля прогонов с отчётом о победе, по {len(protos)} протоколам ===")
    for tag, a in (("точечная победа (как отчитывают)", wp),
                   ("победа, значимая при 95%", ws),
                   ("победа, значимая после Holm", wh)):
        print(f"  {tag:34s} медиана {100*np.median(a):5.1f}%  IQR "
              f"[{100*np.percentile(a,25):5.1f}, {100*np.percentile(a,75):5.1f}]  "
              f"размах [{100*a.min():5.1f}, {100*a.max():5.1f}]")
    print(f"  reported delta vs seed: median {np.median(d):+.4f}, 5-95% [{np.percentile(d,5):+.4f}, "
          f"{np.percentile(d,95):+.4f}]")
    print(f"  a tie on the selection statistic changed the winner in {100 * tie_share:.0f}% of "
          f"(protocol, draw) pairs")
    pub = {}
    for (which, stat, lam, how), v in draws.items():
        pub.setdefault(how, []).extend(v)
    print("  published value by tie-break rule, all else pooled: "
          + ", ".join(f"{h} {np.mean(pub[h]):.4f}" for h in TIEBREAK))

    print(f"\n{'allocation':11s} {'statistic':12s} {'len.pen':>8s} {'tie':>9s} {'mean published':>15s} "
          f"{'mean regret':>12s} {'P(beats seed)':>14s}")
    # Доли побед пишем ПО КАЖДОМУ протоколу, а не только сводкой: без этого нельзя
    # разложить лотерею по измерениям (аллокация / статистика / штраф / ничья).
    wp_by = dict(zip(protos, wp))
    ws_by = dict(zip(protos, ws))
    wh_by = dict(zip(protos, wh))
    rows_out = {"|".join(map(str, k)): {"mean_published": float(np.mean(v)),
                                        "mean_regret": float(best - np.mean(v)),
                                        "p_beats_seed": float(np.mean(np.array(v) > seed_val)),
                                        "win_point": float(wp_by[k]),
                                        "win_sig": float(ws_by[k]),
                                        "win_holm": float(wh_by[k])}
                for k, v in draws.items()}
    ordered = sorted(draws.items(), key=lambda kv: -np.mean(kv[1]))
    for (which, stat, lam, how), v in ordered[:8] + ordered[-8:]:
        v = np.array(v)
        print(f"{which:11s} {stat:12s} {('on' if lam else 'off'):>8s} {how:>9s} {v.mean():15.4f} "
              f"{(best-v).mean():12.4f} {100*(v>seed_val).mean():13.0f}%")

    stem = (f"lottery_summary_{args.target}" + ("" if args.pool == "all" else f"_{args.pool}")
            + ("_plausible" if args.protocols == "plausible" else "")  # else 48 vs 72 protocols overwrite each other
            + (f"_sub{args.subsample}_{args.subsample_seed}" if args.subsample else ""))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{stem}.json").write_text(json.dumps(
        {"target": args.target, "n_draws": args.draws, "pool": args.pool, "pool_size": len(names),
         "pool_spread": float(np.ptp(list(truth.values()))), "truth": truth,
         "rows_used": {cfg.test_set: int(keepT.sum()), cfg.dev_set: int(keepD.sum())},
         "between_var": between, "between_var_raw": between_raw, "within_var": within,
         "sd_protocols_one_study": single,
         "tie_changed_winner": tie_share, "protocol_set": args.protocols,
         "win_point": {"median": float(np.median(wp)), "iqr": [float(np.percentile(wp, 25)),
                                                              float(np.percentile(wp, 75))],
                       "range": [float(wp.min()), float(wp.max())]},
         "win_sig": {"median": float(np.median(ws)), "range": [float(ws.min()), float(ws.max())]},
         "win_holm": {"median": float(np.median(wh)), "range": [float(wh.min()), float(wh.max())]},
         "protocols": rows_out}, indent=2), encoding="utf-8")
    print(f"\nwrote {stem}.json")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--part", choices=["power", "lottery"], default="power")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--subsample", type=int, default=0,
                    help="keep only this many prompts (seed plus a random subset); 0 = whole pool")
    ap.add_argument("--subsample-seed", type=int, default=0)
    ap.add_argument("--target", default="cvar25", choices=METRICS)
    ap.add_argument("--protocols", default="all", choices=["all", "plausible"],
                    help="plausible = drop selection statistics no worst-group author would pick")
    ap.add_argument("--pool", default="all", choices=cfg.pools,
                    help=f"all = every prompt; {cfg.final_prefix} = optimizer finals + seed; "
                         "r15 = one-line edits + seed")
    args = ap.parse_args()
    (part_power if args.part == "power" else part_lottery)(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
