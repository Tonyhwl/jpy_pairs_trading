# Parameter sweep over (beta_window, zscore_window, entry_threshold, exit_threshold,
# max_hold_days) on IS 2018-2021. Reports pareto frontier + OOS.

import math, itertools
from pathlib import Path

import numpy as np
import pandas as pd

from jpy_constants import (CAPITAL_USD, TD, CONTRACT_SPECS,
                           ETF_COST_BP, ETF_BORROW_BP_ANNUAL,
                           SLEEVE_TARGET_VOL_DAILY, SCALE_CAP,
                           VOL_LOOKBACK_PAIR)
from jpy_pnl import load_pair_data

root = Path(__file__).resolve().parent

sleeve_capital = CAPITAL_USD * 0.10
is_start       = pd.Timestamp("2018-01-01")
is_end         = pd.Timestamp("2021-12-31")
oos_start      = pd.Timestamp("2022-01-01")
stop_threshold = 4.0

print("loading data...")
d = load_pair_data()
common       = d["common"]
log_fut      = d["log_fut"]
log_etf      = d["log_etf"]
fut_prices_a = d["fut_prices_a"]
etf_prices_a = d["etf_prices_a"]

is_mask  = (common >= is_start) & (common <= is_end)
oos_mask = (common >= oos_start)
n_obs = len(common)

spec          = CONTRACT_SPECS["MJY"]
contract_mult = spec["mult"]
fut_vol       = fut_prices_a.pct_change().rolling(VOL_LOOKBACK_PAIR).std().shift(1)
vol_scale     = (SLEEVE_TARGET_VOL_DAILY / fut_vol.replace(0, np.nan)).clip(upper=SCALE_CAP).fillna(0.0).values
contract_value  = (contract_mult * fut_prices_a).values
fut_arr         = fut_prices_a.values
etf_arr         = etf_prices_a.values
fut_price_chg   = np.diff(fut_arr, prepend=fut_arr[0])
etf_price_chg   = np.diff(etf_arr, prepend=etf_arr[0])


def pnl_from_pos(position):
    fut_contracts = np.round(np.nan_to_num(
        position * vol_scale * sleeve_capital / np.where(contract_value != 0, contract_value, np.nan)
    )).astype(int)
    fut_notional  = fut_contracts * contract_value
    etf_shares    = np.round(np.nan_to_num(
        -fut_notional / np.where(etf_arr != 0, etf_arr, np.nan)
    )).astype(int)
    fut_contracts_prev = np.roll(fut_contracts, 1); fut_contracts_prev[0] = 0
    etf_shares_prev    = np.roll(etf_shares, 1);    etf_shares_prev[0] = 0
    fut_notional_prev  = np.roll(fut_notional, 1);  fut_notional_prev[0] = 0.0
    gross_pnl  = fut_contracts_prev * contract_mult * fut_price_chg + etf_shares_prev * etf_price_chg
    trade_flag = (np.diff(position, prepend=position[0]) != 0).astype(float)
    fut_chg    = np.abs(np.diff(fut_contracts, prepend=fut_contracts[0])) * trade_flag
    etf_chg    = np.abs(np.diff(etf_shares, prepend=etf_shares[0])) * trade_flag
    cost       = fut_chg * spec["cost_bp"] * contract_mult * fut_arr + etf_chg * ETF_COST_BP * etf_arr
    borrow     = (fut_contracts_prev > 0).astype(float) * np.abs(fut_notional_prev) * ETF_BORROW_BP_ANNUAL / TD
    return gross_pnl - cost - borrow


def sharpe_from_pnl(pnl, mask):
    r = pnl[mask] / CAPITAL_USD
    if mask.sum() < 30 or r.std() == 0:
        return np.nan
    return float(r.mean() / r.std() * math.sqrt(TD))


def count_trades(position):
    chg = np.diff(position, prepend=position[0])
    return int(((chg != 0) & (position != 0)).sum())


def run_state_machine(z_arr, entry_threshold, exit_threshold, max_hold_days):
    position = np.zeros(n_obs, dtype=np.int8)
    state = 0; days_held = 0
    for i in range(n_obs):
        zi = z_arr[i]
        if not np.isfinite(zi):
            continue
        if state != 0:
            days_held += 1
            exited = (days_held >= max_hold_days or abs(zi) >= stop_threshold
                      or (state == +1 and zi >= -exit_threshold)
                      or (state == -1 and zi <= +exit_threshold))
            if exited:
                state = 0; days_held = 0
        if state == 0:
            if   zi <= -entry_threshold: state = +1; days_held = 1
            elif zi >= +entry_threshold: state = -1; days_held = 1
        position[i] = state
    return position


# pre-compute z-score arrays for each (beta_window, zscore_window) combination
beta_windows    = [63, 126, 252]
zscore_windows  = [10, 20, 30, 45, 60]
entry_thresholds = [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00]
exit_thresholds  = [0.00, 0.25, 0.50]
max_hold_days_list = [5, 10, 15, 20, 30]

print("pre-computing z-score arrays...")
zscore_cache = {}
for beta_window, zscore_window in itertools.product(beta_windows, zscore_windows):
    ols_beta = np.full(n_obs, np.nan)
    ols_mu   = np.full(n_obs, np.nan)
    for i in range(beta_window - 1, n_obs):
        y_win = log_fut[i - beta_window + 1 : i + 1]
        x_win = log_etf[i - beta_window + 1 : i + 1]
        var_x = np.var(x_win, ddof=1)
        if var_x > 0:
            ols_b       = np.cov(y_win, x_win, ddof=1)[0, 1] / var_x
            ols_beta[i] = ols_b
            ols_mu[i]   = y_win.mean() - ols_b * x_win.mean()
    spread = log_fut - ols_beta * log_etf - ols_mu
    z_arr  = np.full(n_obs, np.nan)
    for i in range(beta_window + zscore_window - 2, n_obs):
        z_win = spread[i - zscore_window + 1 : i + 1]
        if np.isfinite(z_win).all() and np.std(z_win, ddof=1) > 0:
            z_arr[i] = (spread[i] - z_win.mean()) / np.std(z_win, ddof=1)
    zscore_cache[(beta_window, zscore_window)] = z_arr
print("  done.")

total = (sum(1 for et, xt in itertools.product(entry_thresholds, exit_thresholds) if xt < et)
         * len(zscore_windows) * len(max_hold_days_list) * len(beta_windows))
print(f"running {total:,} combinations on IS...")

results = []
n_done  = 0
for (beta_window, zscore_window), (entry_threshold, exit_threshold), max_hold_days in itertools.product(
        itertools.product(beta_windows, zscore_windows),
        itertools.product(entry_thresholds, exit_thresholds),
        max_hold_days_list):
    if exit_threshold >= entry_threshold:
        continue
    z_arr    = zscore_cache[(beta_window, zscore_window)]
    position = run_state_machine(z_arr, entry_threshold, exit_threshold, max_hold_days)
    n_trades_is = count_trades(position[is_mask])
    pnl         = pnl_from_pos(position)
    sharpe_is   = sharpe_from_pnl(pnl, is_mask)
    results.append(dict(
        beta_window=beta_window, zscore_window=zscore_window,
        entry_threshold=entry_threshold, exit_threshold=exit_threshold,
        max_hold_days=max_hold_days,
        n_trades_is=n_trades_is, sharpe_is=sharpe_is,
    ))
    n_done += 1
    if n_done % 500 == 0:
        print(f"  {n_done}/{total}")

results_df = pd.DataFrame(results).dropna(subset=["sharpe_is"])
print(f"done. {len(results_df):,} valid configs.")


def pareto(df_in):
    rows = df_in.sort_values("n_trades_is", ascending=False).reset_index(drop=True)
    best_sh = -np.inf; keep = []
    for _, r in rows.iterrows():
        if r["sharpe_is"] >= best_sh:
            best_sh = r["sharpe_is"]; keep.append(True)
        else:
            keep.append(False)
    return rows[keep].reset_index(drop=True)


pareto_front = pareto(results_df)
print(f"\npareto frontier ({len(pareto_front)} configs)  -  IS 2018-2021:")
hdr = (f"{'bw':>4} {'zw':>4} {'entry':>6} {'exit':>6} {'hold':>5}"
       f"  {'n_is':>6}  {'sr_is':>7}")
print(hdr); print("-" * len(hdr))
for _, r in pareto_front.iterrows():
    print(f"  {int(r.beta_window):>3} {int(r.zscore_window):>3} {r.entry_threshold:>6.2f} "
          f"{r.exit_threshold:>6.2f} {int(r.max_hold_days):>5}  "
          f"{int(r.n_trades_is):>6}  {r.sharpe_is:>+7.3f}")

print(f"\ntop 20 by trade count (IS):")
top = results_df.sort_values(["n_trades_is", "sharpe_is"], ascending=[False, False]).head(20)
print(hdr); print("-" * len(hdr))
for _, r in top.iterrows():
    print(f"  {int(r.beta_window):>3} {int(r.zscore_window):>3} {r.entry_threshold:>6.2f} "
          f"{r.exit_threshold:>6.2f} {int(r.max_hold_days):>5}  "
          f"{int(r.n_trades_is):>6}  {r.sharpe_is:>+7.3f}")

print(f"\npareto configs - OOS 2022-present:")
hdr2 = (f"{'bw':>4} {'zw':>4} {'entry':>6} {'exit':>6} {'hold':>5}"
        f"  {'n_is':>6} {'sr_is':>7}  {'n_oos':>6} {'sr_oos':>8}")
print(hdr2); print("-" * len(hdr2))
for _, r in pareto_front.iterrows():
    z_arr    = zscore_cache[(int(r.beta_window), int(r.zscore_window))]
    position = run_state_machine(z_arr, r.entry_threshold, r.exit_threshold, int(r.max_hold_days))
    n_trades_oos = count_trades(position[oos_mask])
    pnl          = pnl_from_pos(position)
    sharpe_oos   = sharpe_from_pnl(pnl, oos_mask)
    print(f"  {int(r.beta_window):>3} {int(r.zscore_window):>3} {r.entry_threshold:>6.2f} "
          f"{r.exit_threshold:>6.2f} {int(r.max_hold_days):>5}  "
          f"{int(r.n_trades_is):>6} {r.sharpe_is:>+7.3f}"
          f"  {n_trades_oos:>6} {sharpe_oos:>+8.3f}")

out = root / "results" / "jpy_sweep.csv"
out.parent.mkdir(exist_ok=True)
results_df.sort_values(["n_trades_is", "sharpe_is"], ascending=[False, False]).to_csv(out, index=False)
print(f"\nfull results -> {out}")
