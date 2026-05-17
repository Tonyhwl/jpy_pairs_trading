# paper figures for JPY basis strategy
# config: beta_window=126, zscore_window=30, entry=1.0, exit=0.0, max_hold=30, OOS from 2022

import math
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from jpy_pnl import (load_pair_data, compute_zscore, run_state_machine,
                     pnl_from_position, CAPITAL_USD, TD)

root     = Path(__file__).resolve().parent
out_root = root / "results"
out_root.mkdir(parents=True, exist_ok=True)

strategy_capital  = CAPITAL_USD * 0.10
oos_start       = pd.Timestamp("2022-01-01")
beta_window, zscore_window = 126, 30
entry_threshold, exit_threshold, max_hold_days = 1.00, 0.00, 30
stop_threshold  = 4.0

bg     = "#ffffff"
panel  = "#fafafa"
border = "#cccccc"
grid_c = "#ebebeb"
text_c = "#1a1a1a"
muted  = "#888888"
dim    = "#bbbbbb"
line_c = "#1a3a5c"
cpos   = "#2a7d52"
cneg   = "#b33a3a"
cpos_f = "#d6ede1"
cneg_f = "#f5d8d8"

plt.rcParams.update({
    "figure.facecolor":   bg,
    "axes.facecolor":     panel,
    "axes.edgecolor":     border,
    "axes.linewidth":     0.8,
    "axes.grid":          True,
    "axes.axisbelow":     True,
    "grid.color":         grid_c,
    "grid.linewidth":     0.6,
    "xtick.color":        muted,
    "ytick.color":        muted,
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    "font.family":        "serif",
    "font.serif":         ["CMU Serif", "Latin Modern Roman", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset":   "cm",
    "text.color":         text_c,
    "figure.dpi":         170,
    "savefig.facecolor":  bg,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.25,
})


def stat_box(ax, lines, loc="upper left"):
    txt = "\n".join(f"{k:<10}{v}" for k, v in lines.items())
    xy  = {"upper left":  (0.015, 0.975),
           "lower left":  (0.015, 0.035),
           "upper right": (0.985, 0.975)}[loc]
    ha  = "right" if "right" in loc else "left"
    va  = "bottom" if "lower" in loc else "top"
    ax.text(xy[0], xy[1], txt, transform=ax.transAxes, ha=ha, va=va,
            fontsize=9, color=text_c, family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor=bg,
                      edgecolor=border, linewidth=0.8))


print("loading data...")
d = load_pair_data()
common, fut_prices_a, etf_prices_a = d["common"], d["fut_prices_a"], d["etf_prices_a"]
oos_mask = (common >= oos_start)

print("computing z-scores...")
z_arr = compute_zscore(d["log_fut"], d["log_etf"], beta_window, zscore_window)

print("running state machine and pnl...")
position = run_state_machine(z_arr, entry_threshold, exit_threshold, max_hold_days,
                              stop_threshold=stop_threshold)
pnl_arr  = pnl_from_position(position, fut_prices_a, etf_prices_a, strategy_capital)

pnl_s = pd.Series(pnl_arr, index=common)
pos_s = pd.Series(position, index=common)
z_s   = pd.Series(z_arr,   index=common)

ret_oos  = (pnl_s[oos_mask] / CAPITAL_USD).dropna()
n_oos    = len(ret_oos)
sharpe   = float(ret_oos.mean() / ret_oos.std() * math.sqrt(TD))
cagr     = float((1 + ret_oos).prod() ** (TD / n_oos) - 1)
ann_vol  = float(ret_oos.std() * math.sqrt(TD))
eq_c     = (1 + ret_oos).cumprod()
max_dd   = float((eq_c / eq_c.cummax() - 1).min())

print(f"oos sharpe={sharpe:.3f}  cagr={cagr*100:.2f}%  max_dd={max_dd*100:.2f}%")


def make_zscore():
    z   = z_s[oos_mask].dropna()
    p   = pos_s[oos_mask].reindex(z.index).fillna(0)
    idx = z.index

    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.fill_between(idx, z.values, 0, where=p.values == +1,
                    color=cpos_f, linewidth=0, alpha=0.85, label="long spread")
    ax.fill_between(idx, z.values, 0, where=p.values == -1,
                    color=cneg_f, linewidth=0, alpha=0.85, label="short spread")
    ax.plot(idx, z.values, color=line_c, lw=0.85, zorder=3)
    ax.axhline(0, color=border, lw=0.7)
    for lv, ls in [(+entry_threshold, "--"), (-entry_threshold, "--"),
                   (+stop_threshold, ":"), (-stop_threshold, ":")]:
        ax.axhline(lv, color="#555555", lw=0.8, ls=ls, alpha=0.7)

    ax.set_ylabel(r"$z_t$", fontsize=10)
    ax.set_xlim(idx[0], idx[-1])
    ax.yaxis.set_major_locator(mtick.MaxNLocator(integer=True, nbins=7))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())

    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks([entry_threshold, -entry_threshold, stop_threshold, -stop_threshold])
    ax2.set_yticklabels([f"+{entry_threshold:.1f}", f"-{entry_threshold:.1f}",
                         f"+{stop_threshold:.0f}", f"-{stop_threshold:.0f}"],
                        fontsize=8.5, color=muted)
    ax2.tick_params(length=0)
    for sp in ax2.spines.values():
        sp.set_visible(False)

    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.set_title(
        fr"30-day z-score of residual log-spread  $\cdot$  $W_\beta=126$d OLS  $\cdot$  OOS 2022-2026",
        loc="left", fontsize=10
    )
    plt.tight_layout()
    p_out = out_root / "6j_fxy_zscore.png"
    plt.savefig(p_out); plt.close()
    print(f"  wrote {p_out}")


def make_equity():
    r  = ret_oos
    eq = strategy_capital * (1 + r).cumprod()

    fig, ax = plt.subplots(figsize=(13, 5.0))
    ax.fill_between(eq.index, strategy_capital, eq.values,
                    color=line_c, alpha=0.08, linewidth=0)
    ax.plot(eq.index, eq.values, color=line_c, lw=1.5)
    ax.axhline(strategy_capital, color=border, lw=0.8, ls="--", alpha=0.8)

    def fmt_usd(x, _=None):
        if abs(x) >= 1e6:
            return f"${x/1e6:.2f}M"
        return f"${x/1e3:.0f}K"

    ax.yaxis.set_major_formatter(mtick.FuncFormatter(fmt_usd))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.set_ylabel("strategy value (USD)")
    ax.set_xlim(eq.index[0], eq.index[-1])
    ax.set_title("Cumulative equity  -  JPY basis pair (6J / FXY)  -  OOS 2022-2026",
                 loc="left", fontsize=10)
    stat_box(ax, {
        "Sharpe":  f"{sharpe:+.2f}",
        "CAGR":    f"{cagr*100:+.1f}%",
        "Vol":     f"{ann_vol*100:.1f}%",
        "Max DD":  f"{max_dd*100:+.1f}%",
    }, loc="upper left")
    plt.tight_layout()
    p_out = out_root / "equity.png"
    plt.savefig(p_out); plt.close()
    print(f"  wrote {p_out}")


def make_drawdown():
    r     = ret_oos
    eq    = (1 + r).cumprod()
    dd    = eq / eq.cummax() - 1
    worst = dd.idxmin()

    fig, ax = plt.subplots(figsize=(13, 3.8))
    ax.fill_between(dd.index, dd.values, 0, color=cneg, alpha=0.22, linewidth=0)
    ax.plot(dd.index, dd.values, color=cneg, lw=1.2)
    ax.axhline(0, color=border, lw=0.7)
    for lv in (-0.05, -0.10, -0.15, -0.20):
        ax.axhline(lv, color=dim, lw=0.5, ls="--", alpha=0.5)
    ax.scatter([worst], [max_dd], color=cneg, s=38, zorder=5,
               edgecolors="#ffffff", linewidths=0.8)
    ax.annotate(
        f"min = {max_dd*100:+.2f}%\n{worst.date()}",
        xy=(worst, max_dd), xytext=(24, 16), textcoords="offset points",
        fontsize=9, color=text_c, family="serif",
        bbox=dict(boxstyle="round,pad=0.4", facecolor=bg, edgecolor=border, linewidth=0.7)
    )
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.set_ylabel("drawdown")
    ax.set_xlim(dd.index[0], dd.index[-1])
    ax.set_title("Drawdown vs rolling high-water mark  -  JPY basis pair  -  OOS 2022-2026",
                 loc="left", fontsize=10)
    plt.tight_layout()
    p_out = out_root / "drawdown.png"
    plt.savefig(p_out); plt.close()
    print(f"  wrote {p_out}")


def make_heatmap():
    r       = ret_oos
    monthly = (1 + r).resample("ME").prod() - 1
    years   = sorted({ts.year for ts in monthly.index})
    matrix  = np.full((len(years), 13), np.nan)
    for i, y in enumerate(years):
        mk = monthly[monthly.index.year == y]
        for ts_val, v in mk.items():
            matrix[i, ts_val.month - 1] = v * 100
        matrix[i, 12] = ((1 + mk).prod() - 1) * 100

    cmap = LinearSegmentedColormap.from_list(
        "rg",
        [(0, "#8b1a1a"), (0.35, "#f5d8d8"), (0.5, "#f8f8f8"),
         (0.65, "#d6ede1"), (1, "#1e6b42")],
        N=256,
    )
    vmax = max(abs(np.nanmin(matrix[:, :12])), np.nanmax(matrix[:, :12])) * 1.1
    vmax = max(vmax, 4.0)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(13, 0.65 * len(years) + 2.0))
    ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
    for i in range(len(years)):
        for j in range(13):
            v = matrix[i, j]
            if np.isnan(v):
                continue
            col = "#1a1a1a" if abs(v) < vmax * 0.6 else "#ffffff"
            wt  = "bold" if j == 12 else "normal"
            ax.text(j, i, f"{v:+.1f}", ha="center", va="center",
                    color=col, fontsize=10, fontweight=wt, family="monospace")
    ax.set_xticks(range(13))
    ax.set_xticklabels(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul",
         "Aug", "Sep", "Oct", "Nov", "Dec", "YTD"],
        fontsize=10, color=text_c, family="serif"
    )
    ax.set_yticks(range(len(years)))
    ax.set_yticklabels([str(y) for y in years], fontsize=10.5,
                       color=text_c, family="serif", fontweight="bold")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.grid(False)
    ax.axvline(11.5, color="#aaaaaa", lw=1.0)

    flat    = matrix[:, :12].flatten()
    pos_pct = np.mean(flat[np.isfinite(flat)] > 0) * 100
    ax.set_title(
        fr"Monthly returns (%)  $\cdot$  JPY basis pair (6J / FXY)  $\cdot$  OOS 2022-2026"
        f"\n$P(r_m > 0) = {pos_pct:.0f}\\%$  $\\cdot$  "
        f"$\\bar{{r}}_m = {np.nanmean(flat):+.2f}\\%$  $\\cdot$  "
        f"$\\sigma_m = {np.nanstd(flat):.2f}\\%$",
        loc="left", fontsize=10
    )
    plt.tight_layout()
    p_out = out_root / "heatmap.png"
    plt.savefig(p_out); plt.close()
    print(f"  wrote {p_out}")


if __name__ == "__main__":
    make_zscore()
    make_equity()
    make_drawdown()
    make_heatmap()
    print("done.")
