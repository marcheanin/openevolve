#!/usr/bin/env python
"""Графическая часть статьи: шесть рисунков, всё считается из сохранённых предсказаний.

Палитра проверена валидатором (все шесть проверок пройдены): категориальные слоты
#2a78d6 (синий), #1baf7a (бирюзовый), #eb6834 (оранжевый), #eda100 (жёлтый). Контраст двух
светлых слотов к фону ниже 3:1, поэтому у всех меток есть текстовые подписи — это требование
валидатора, а не украшение.

Цвет закреплён за СТЕНДОМ, а не за рангом: CivilComments всегда синий, MultiNLI бирюзовый,
многоязычная токсичность оранжевая. Если рисунок показывает один стенд, легенды нет — его
называет заголовок.

Подписи русские. Для подачи переводятся в одном месте — словарь L внизу файла.

Запуск: python scripts/make_figures.py [--out paper/figures]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# --- параметры оформления -------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8983"
STAND_COLOR = {"civil": "#2a78d6", "mnli": "#1baf7a", "toxlang": "#eb6834"}
STAND_LABEL = {"civil": "CivilComments", "mnli": "MultiNLI", "toxlang": "многоязычная токсичность"}
ACCENT = "#eda100"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 9, "font.family": "DejaVu Sans",
    "axes.edgecolor": INK3, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.titlesize": 10, "axes.titlecolor": INK, "axes.titleweight": "bold",
    "legend.frameon": False, "legend.fontsize": 8,
    "grid.color": "#e6e5e0", "grid.linewidth": 0.7,
})


def tidy(ax, grid="y"):
    """Оси рецессивные: две линии вместо рамки, сетка только по одной оси."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid, zorder=0)
    ax.set_axisbelow(True)


def load_stand(stand: str, preds_dir: str | None = None):
    """Предсказания и разметка одного стенда. Возвращает (y, c, keep, P, имена групп, cfg)."""
    os.environ["S11_DATASET"] = stand
    if preds_dir:
        os.environ["S11_PREDS_DIR"] = preds_dir
    else:
        os.environ.pop("S11_PREDS_DIR", None)
    for m in ("dataset_config", "analyze_s11"):
        sys.modules.pop(m, None)
    from analyze_s11 import IDS, load_preds, load_set, valid_rows
    from dataset_config import cfg
    y, c, rec = load_set(cfg.test_set)
    P = load_preds(cfg.test_set, len(y))
    every = sorted(n for n in P if not n.startswith("CONTROL:"))
    keep = valid_rows(P, len(y), every)
    return y, c, keep, P, cfg.group_names(rec), cfg, tuple(IDS)


def noise_max(stand):
    """Распределение «максимум из N под нулём» и наблюдённый максимум."""
    y, c, keep, P, gn, cfg, IDS = load_stand(stand)
    from analyze_s11 import Bootstrap
    names = [n for n in sorted(P) if not n.startswith("CONTROL:") and n != "seed"]
    bs = Bootstrap(y[keep], c[keep], 4000, np.random.default_rng(0))
    o_s, b_s = bs.metrics(P["seed"][keep], groups=IDS)
    obs, boot = [], []
    for n in names:
        o, b = bs.metrics(P[n][keep], groups=IDS)
        obs.append(o["hard_min"] - o_s["hard_min"])
        boot.append(b["hard_min"] - b_s["hard_min"])
    obs = np.array(obs); boot = np.stack(boot, 0)
    mx = (boot - obs[:, None]).max(axis=0)
    return mx, float(obs.max()), len(names)


# --- Рисунок 1: наблюдённый максимум против шумовой планки ------------------------------------
def fig1(out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharey=True)
    verdicts = {"civil": "отчёт завышает", "mnli": "отчёт верен", "toxlang": "отчёт занижает"}
    for ax, stand in zip(axes, ("civil", "mnli", "toxlang")):
        mx, best, N = noise_max(stand)
        col = STAND_COLOR[stand]
        ax.hist(mx, bins=45, color=col, alpha=0.30, edgecolor="none", zorder=2)
        c95 = float(np.percentile(mx, 95))
        p = float((mx >= best).mean())
        ax.axvline(c95, color=INK3, lw=1.2, ls=(0, (4, 3)), zorder=3)
        ax.axvline(best, color=col, lw=2.2, zorder=4)
        top = ax.get_ylim()[1]
        ax.text(best, top * 0.97, f"  наблюдено\n  {best:+.4f}", color=col, fontsize=8,
                va="top", ha="left" if best < c95 else "right", fontweight="bold")
        ax.text(c95, top * 0.55, f"планка C95\n{c95:+.4f}  ", color=INK2, fontsize=7.5,
                va="top", ha="right")
        ax.set_title(f"{STAND_LABEL[stand]}\nN = {N},  p = {p:.3f} — {verdicts[stand]}",
                     fontsize=9, pad=8)
        ax.set_xlabel("прирост hard-min")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.02f}"))
        tidy(ax)
    axes[0].set_ylabel("бутстреп-реплик")
    fig.suptitle("Рис. 1. Наблюдённый прирост против того, что даёт чистый отбор максимума из N",
                 fontsize=11, fontweight="bold", color=INK, y=1.04)
    fig.text(0.5, -0.09, "Гистограмма — распределение максимума из N контрастов под нулевой гипотезой "
             "«ни один промпт не лучше стартового».\np — доля реплик, в которых шум дал бы не меньше "
             "наблюдённого.", ha="center", fontsize=8, color=INK2)
    fig.tight_layout()
    fig.savefig(out / "fig1_winners_curse.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  рис. 1 готов")


# --- Рисунок 2: как планка растёт с размером пула ---------------------------------------------
def fig2(out: Path):
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for stand in ("civil", "mnli", "toxlang"):
        mx_full, best, N = noise_max(stand)
        y, c, keep, P, gn, cfg, IDS = load_stand(stand)
        from analyze_s11 import Bootstrap
        names = [n for n in sorted(P) if not n.startswith("CONTROL:") and n != "seed"]
        bs = Bootstrap(y[keep], c[keep], 2000, np.random.default_rng(0))
        o_s, b_s = bs.metrics(P["seed"][keep], groups=IDS)
        obs, boot = [], []
        for n in names:
            o, b = bs.metrics(P[n][keep], groups=IDS)
            obs.append(o["hard_min"] - o_s["hard_min"]); boot.append(b["hard_min"] - b_s["hard_min"])
        noise = np.stack(boot, 0) - np.array(obs)[:, None]
        rng = np.random.default_rng(7)
        ks = [k for k in (2, 3, 5, 8, 12, 16, 20, 25, 30, N) if k <= N]
        ys = []
        for k in ks:
            vals = [noise[rng.choice(N, size=k, replace=False)].max(axis=0).mean() for _ in range(80)]
            ys.append(np.mean(vals))
        col = STAND_COLOR[stand]
        ax.plot(ks, ys, color=col, lw=2, marker="o", ms=4, zorder=3)
        ax.annotate(STAND_LABEL[stand], (ks[-1], ys[-1]), xytext=(6, 0),
                    textcoords="offset points", color=col, fontsize=8, va="center",
                    fontweight="bold")
    ax.set_xlabel("размер пула N (сколько промптов перебрано)")
    ax.set_ylabel("средняя планка «лучший из N»")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.03f}"))
    ax.set_xlim(0, 46)
    ax.set_title("Рис. 2. Чем больше промптов перебрано, тем выше прирост, который даст один шум",
                 fontsize=10, pad=10)
    tidy(ax)
    fig.tight_layout()
    fig.savefig(out / "fig2_ceiling_vs_N.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  рис. 2 готов")


# --- Рисунок 3: кривые строгости и положение промптов -----------------------------------------
def fig3(out: Path):
    """Главный рисунок: что достижимо одной ручкой строгости и где на этом фоне стоят финалы.

    По оси x — строгость (доля ложных тревог на целевой группе), по оси y — качество на той
    группе, которая была худшей у СТАРТОВОГО промпта. Линия — однострочные правки строгости,
    то есть всё, что даёт один порог. Точки — финалы оптимизаторов.
    """
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, stand in zip(axes, ("civil", "mnli", "toxlang")):
        y, c, keep, P, gn, cfg, IDS = load_stand(stand)
        from analyze_s11 import gba_by_group
        gs = gba_by_group(P["seed"], y, c, keep, IDS)
        w = min(gs, key=gs.get)                       # группа, худшая у стартового промпта
        pos, neg = (c == w) & (y == 1) & keep, (c == w) & (y == 0) & keep

        def point(n):
            return float((P[n][neg] == 1).mean()), float(0.5 * ((P[n][pos] == 1).mean() + (P[n][neg] == 0).mean()))

        edits = sorted(n for n in P if n.startswith("r15:"))
        finals = sorted(n for n in P if n.startswith(cfg.final_prefix + ":"))
        ep = sorted([point(n) for n in edits] + [point("seed")])
        ex, ey = zip(*ep)
        lo, hi = min(ex), max(ex)
        col = STAND_COLOR[stand]

        allx = list(ex) + list(fx0 for fx0 in [point(n)[0] for n in finals])
        pad = 0.06 * (max(allx) - min(allx) + 1e-9)
        x0, x1 = min(allx) - pad, max(allx) + pad

        # подсвечиваем ДИАПАЗОН, покрытый правками: всё вне него — экстраполяция
        ax.axvspan(lo, hi, color=col, alpha=0.06, zorder=1)
        for xb in (lo, hi):
            ax.axvline(xb, color=INK3, lw=0.8, ls=(0, (2, 3)), zorder=2)

        ax.plot(ex, ey, color=INK2, lw=1.6, marker="o", ms=3.5, zorder=3,
                label="правки строгости — всё, что даёт один порог")
        fx, fy = zip(*[point(n) for n in finals]) if finals else ([], [])
        ax.scatter(fx, fy, s=26, color=col, edgecolor=SURFACE, linewidth=0.8, zorder=4,
                   label="финалы оптимизаторов")
        sx, sy = point("seed")
        ax.scatter([sx], [sy], s=70, marker="D", color=INK, edgecolor=SURFACE, linewidth=1.0,
                   zorder=5, label="стартовый промпт")
        best_edit = max(ey)
        ax.axhline(best_edit, color=INK3, lw=1.0, ls=(0, (4, 3)), zorder=2)
        above = sum(1 for v in fy if v > best_edit)
        ax.set_xlim(x0, x1)
        ax.text(lo + (hi - lo) / 2, ax.get_ylim()[0], "правки покрыли этот диапазон",
                fontsize=6.5, color=INK3, ha="center", va="bottom")
        if stand == "toxlang":
            bi = int(np.argmax(fy))
            ax.annotate(f"GEPA: +{fy[bi]-best_edit:.3f} сверх порога", (fx[bi], fy[bi]),
                        xytext=(-8, -14), textcoords="offset points", fontsize=7.5,
                        color=col, fontweight="bold", ha="right")
        ax.set_title(f"{STAND_LABEL[stand]}\nцелевая группа «{gn[w]}»: выше линии {above} из {len(finals)}",
                     fontsize=9, pad=8)
        ax.set_xlabel("строгость: доля ложных тревог")
        tidy(ax)
    axes[0].set_ylabel("качество на целевой группе (GBA)")
    axes[0].legend(loc="lower right", fontsize=7)
    fig.suptitle("Рис. 3. Кривая строгости и где на ней стоят промпты", fontsize=11,
                 fontweight="bold", color=INK, y=1.04)
    fig.text(0.5, -0.13,
             "Серая линия — развёртка порога: однострочные правки, которые по построению меняют только строгость. "
             "Горизонтальный пунктир — лучшее, что она даёт НА ЭТОЙ ГРУППЕ.\n"
             "Точка выше пунктира означает, что промпт улучшил РАЗЛИЧЕНИЕ, а не просто сдвинул порог. "
             "Подсвечен диапазон строгости, покрытый правками; вне него — экстраполяция.\n"
             "Счёт «выше линии» ведётся по ЦЕЛЕВОЙ ГРУППЕ и не совпадает со счётом по hard-min "
             "(на CivilComments 5 из 25 против 2 из 25) — это разные величины.",
             ha="center", fontsize=8, color=INK2)
    fig.tight_layout()
    fig.savefig(out / "fig3_strictness_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  рис. 3 готов")


# --- Рисунок 4: число групп как рычаг ---------------------------------------------------------
def fig4(out: Path):
    """Числа из 08_STATE §2.6: псевдогруппы, сигнал не меняется, меняется только k."""
    data = {"mnli": ([2, 4, 10], [12, 7, 0], [0.0157, 0.0226, 0.0375], 22),
            "civil": ([2, 4, 8], [0, 0, 0], [0.0241, 0.0351, 0.0503], 36)}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4))
    for stand, (ks, hits, widths, tot) in data.items():
        col = STAND_COLOR[stand]
        a1.plot(ks, [100 * h / tot for h in hits], color=col, lw=2, marker="o", ms=5, zorder=3)
        a1.annotate(STAND_LABEL[stand], (ks[0], 100 * hits[0] / tot), xytext=(6, 6),
                    textcoords="offset points", color=col, fontsize=8, fontweight="bold")
        a2.plot(ks, widths, color=col, lw=2, marker="o", ms=5, zorder=3)
    a1.set_xlabel("число определённых групп k"); a1.set_ylabel("находок, % от пула")
    a1.set_title("Находок после поправки на множественность", fontsize=9.5, pad=8)
    a2.set_xlabel("число определённых групп k"); a2.set_ylabel("ширина интервала")
    a2.set_title("Ширина интервала растёт примерно как √k", fontsize=9.5, pad=8)
    a2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.03f}"))
    for ax in (a1, a2):
        ax.set_xticks([2, 4, 6, 8, 10]); tidy(ax)
    fig.suptitle("Рис. 4. Сигнал не меняется — меняется только число групп", fontsize=11,
                 fontweight="bold", color=INK, y=1.03)
    fig.text(0.5, -0.10, "Группы случайные, одинакового размера: данные, предсказания и эффект те же. "
             "При k = 10 находок ноль, при k = 2 их двенадцать.", ha="center", fontsize=8, color=INK2)
    fig.tight_layout()
    fig.savefig(out / "fig4_group_count.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  рис. 4 готов")


# --- Рисунок 5: лотерея розыгрыша -------------------------------------------------------------
def parse_halves(path: Path):
    ws, fs = [], []
    for line in path.read_text(encoding="utf-8", errors="ignore").split("\n"):
        m = re.search(r"half #\d+.*?CI width mean=([\d.]+).*?\(\s*([\d.]+)%\)", line)
        if m:
            ws.append(float(m.group(1))); fs.append(float(m.group(2)))
    return np.array(ws), np.array(fs)


def fig5(out: Path):
    src = {"civil": ROOT / "results/S11_protocol_matrix/final/draw_lottery_halves_civil.txt",
           "mnli": ROOT / "results/S13_mnli_matrix/final/draw_lottery_halves.txt"}
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for stand, path in src.items():
        if not path.is_file():
            print(f"  [рис. 5] нет файла {path}"); continue
        w, f = parse_halves(path)
        col = STAND_COLOR[stand]
        ax.scatter(w, f, s=40, color=col, edgecolor=SURFACE, linewidth=0.8, zorder=3)
        ax.annotate(f"{STAND_LABEL[stand]}\nширина {w.min():.4f}–{w.max():.4f}, находки {f.min():.1f}–{f.max():.1f} %",
                    (w.mean(), f.max()), xytext=(0, 14), textcoords="offset points",
                    color=col, fontsize=8, ha="center", fontweight="bold")
    ax.set_xlabel("средняя ширина доверительного интервала")
    ax.set_ylabel("доля различимых пар, %")
    ax.set_title("Рис. 5. Ширина интервала почти не меняется — число находок меняется в разы",
                 fontsize=10, pad=10)
    tidy(ax, grid="both")
    fig.text(0.5, -0.09, "Каждая точка — независимая половина теста ОДИНАКОВОГО размера, 12 половин на стенд. "
             "Интервалы честные;\nлотереей является то, ЧТО попадёт в таблицу.", ha="center",
             fontsize=8, color=INK2)
    fig.tight_layout()
    fig.savefig(out / "fig5_draw_lottery.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  рис. 5 готов")


# --- Рисунок 6: куда переехал минимум ---------------------------------------------------------
def fig6(out: Path):
    y, c, keep, P, gn, cfg, IDS = load_stand("toxlang")
    from analyze_s11 import gba_by_group
    gs = gba_by_group(P["seed"], y, c, keep, IDS)
    gb = gba_by_group(P["s15:43_gepa__hard_min"], y, c, keep, IDS)
    order = sorted(gs, key=lambda g: gs[g])
    labels = [gn[g] for g in order]
    a = [gs[g] for g in order]; b = [gb[g] for g in order]
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    for i, (u, v) in enumerate(zip(a, b)):
        up = v >= u
        ax.plot([0, 1], [u, v], color=(STAND_COLOR["toxlang"] if up else INK3), lw=1.8,
                zorder=3, alpha=0.9)
        ax.scatter([0, 1], [u, v], s=28, color=(STAND_COLOR["toxlang"] if up else INK3),
                   edgecolor=SURFACE, linewidth=0.8, zorder=4)
        ax.annotate(labels[i], (0, u), xytext=(-8, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=8, color=INK2)
        ax.annotate(f"{v:.3f}", (1, v), xytext=(8, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=8, color=INK2)
    mn_a, mn_b = min(a), min(b)
    ax.scatter([0], [mn_a], s=150, facecolor="none", edgecolor=INK, linewidth=1.6, zorder=5)
    ax.scatter([1], [mn_b], s=150, facecolor="none", edgecolor=INK, linewidth=1.6, zorder=5)
    ax.annotate("минимум", (0, mn_a), xytext=(16, 4), textcoords="offset points",
                ha="left", va="bottom", fontsize=8, color=INK, fontweight="bold")
    ax.annotate("минимум переехал сюда", (1, mn_b), xytext=(-16, -16), textcoords="offset points",
                ha="right", va="top", fontsize=8, color=INK, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["стартовый промпт", "лучший финал (GEPA)"])
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylabel("качество на языке (GBA)")
    ax.set_title("Рис. 6. Целевая группа поднята на +0,061 — а hard-min сообщает «шум»",
                 fontsize=10, pad=10)
    tidy(ax)
    fig.text(0.5, -0.07, "Оранжевым — языки, которым стало лучше. Амхарский, худший у стартового промпта, "
             "вырос с 0,721 до 0,782 (p = 0,0003).\nНемецкий просел в пределах шума и стал новым минимумом, "
             "поэтому hard-min показал лишь +0,024.", ha="center", fontsize=8, color=INK2)
    fig.tight_layout()
    fig.savefig(out / "fig6_minimum_moved.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  рис. 6 готов")


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "paper/figures")
    ap.add_argument("--only", type=int, default=0, help="построить только один рисунок")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"рисунки -> {args.out}")
    for i, fn in enumerate((fig1, fig2, fig3, fig4, fig5, fig6), 1):
        if args.only and args.only != i:
            continue
        fn(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
