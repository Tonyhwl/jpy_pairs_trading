# Shared utilities for the JPY pair strategy: data loading, z-score,
# state machine, and PnL accounting. Used by best_config, plots, and ar1.

import sys
from pathlib import Path
import numpy as np

from jpy_constants import (CAPITAL_USD, TD, CONTRACT_SPECS,
                           ETF_COST_BP, ETF_BORROW_BP_ANNUAL,
                           SLEEVE_TARGET_VOL_DAILY, SCALE_CAP,
                           VOL_LOOKBACK_PAIR)

# Data loaders live in the sibling prop_trading repo (raw DBN data + databento).
# If that repo isn't available, scripts that call load_pair_data() will fail at
# import time with a clear ImportError.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "prop_trading"))

try:
    from backtest._test_open_fill import load_futures_oc, load_etf_oc
    _HAVE_LOADERS = True
except ImportError:
    _HAVE_LOADERS = False


def load_pair_data():
    if not _HAVE_LOADERS:
        raise ImportError(
            "load_futures_oc / load_etf_oc unavailable. "
            "These live in the sibling prop_trading repo and need the raw "
            "DBN data files + databento package to function."
        )
    fut_prices, _ = load_futures_oc("6J", "6J", "HMUZ", "2014-01-01")
    etf_prices, _ = load_etf_oc("ETF_fx", "FXY")
    log_fut = np.log(fut_prices.dropna())
    log_etf = np.log(etf_prices.reindex(log_fut.index).ffill().dropna())
    common  = log_fut.index.intersection(log_etf.index)
    return {
        "common":       common,
        "log_fut":      log_fut.reindex(common).values,
        "log_etf":      log_etf.reindex(common).values,
        "fut_prices_a": fut_prices.reindex(common).ffill(),
        "etf_prices_a": etf_prices.reindex(common).ffill(),
    }


def compute_zscore(log_fut, log_etf, beta_window, zscore_window):
    n_obs = len(log_fut)
    ols_beta = np.full(n_obs, np.nan)
    ols_mu   = np.full(n_obs, np.nan)
    for i in range(beta_window - 1, n_obs):
        y_win = log_fut[i - beta_window + 1 : i + 1]
        x_win = log_etf[i - beta_window + 1 : i + 1]
        var_x = np.var(x_win, ddof=1)
        if var_x > 0:
            b = np.cov(y_win, x_win, ddof=1)[0, 1] / var_x
            ols_beta[i] = b
            ols_mu[i]   = y_win.mean() - b * x_win.mean()
    spread = log_fut - ols_beta * log_etf - ols_mu
    z_arr  = np.full(n_obs, np.nan)
    for i in range(beta_window + zscore_window - 2, n_obs):
        z_win = spread[i - zscore_window + 1 : i + 1]
        if np.isfinite(z_win).all() and np.std(z_win, ddof=1) > 0:
            z_arr[i] = (spread[i] - z_win.mean()) / np.std(z_win, ddof=1)
    return z_arr


def run_state_machine(z_arr, entry_threshold, exit_threshold, max_hold_days,
                      stop_threshold=4.0):
    n_obs = len(z_arr)
    position = np.zeros(n_obs, dtype=np.int8)
    state = 0; days_held = 0
    for i in range(n_obs):
        zi = z_arr[i]
        if not np.isfinite(zi):
            continue
        if state != 0:
            days_held += 1
            if (days_held >= max_hold_days or abs(zi) >= stop_threshold
                    or (state == +1 and zi >= -exit_threshold)
                    or (state == -1 and zi <= +exit_threshold)):
                state = 0; days_held = 0
        if state == 0:
            if   zi <= -entry_threshold: state = +1; days_held = 1
            elif zi >= +entry_threshold: state = -1; days_held = 1
        position[i] = state
    return position


def pnl_from_position(position, fut_prices_a, etf_prices_a, sleeve_capital):
    spec          = CONTRACT_SPECS["MJY"]
    contract_mult = spec["mult"]
    fut_arr       = fut_prices_a.values
    etf_arr       = etf_prices_a.values

    fut_vol   = fut_prices_a.pct_change().rolling(VOL_LOOKBACK_PAIR).std().shift(1)
    vol_scale = (SLEEVE_TARGET_VOL_DAILY / fut_vol.replace(0, np.nan)).clip(upper=SCALE_CAP).fillna(0.0).values
    contract_value = contract_mult * fut_arr
    fut_contracts  = np.round(np.nan_to_num(
        position * vol_scale * sleeve_capital / np.where(contract_value != 0, contract_value, np.nan)
    )).astype(int)
    fut_notional   = fut_contracts * contract_value
    etf_shares     = np.round(np.nan_to_num(
        -fut_notional / np.where(etf_arr != 0, etf_arr, np.nan)
    )).astype(int)

    fut_contracts_prev = np.roll(fut_contracts, 1); fut_contracts_prev[0] = 0
    etf_shares_prev    = np.roll(etf_shares, 1);    etf_shares_prev[0] = 0
    fut_notional_prev  = np.roll(fut_notional, 1);  fut_notional_prev[0] = 0.0
    fut_price_chg = np.diff(fut_arr, prepend=fut_arr[0])
    etf_price_chg = np.diff(etf_arr, prepend=etf_arr[0])
    gross_pnl  = fut_contracts_prev * contract_mult * fut_price_chg + etf_shares_prev * etf_price_chg
    trade_flag = (np.diff(position, prepend=position[0]) != 0).astype(float)
    fut_chg    = np.abs(np.diff(fut_contracts, prepend=fut_contracts[0])) * trade_flag
    etf_chg    = np.abs(np.diff(etf_shares, prepend=etf_shares[0])) * trade_flag
    cost       = fut_chg * spec["cost_bp"] * contract_mult * fut_arr + etf_chg * ETF_COST_BP * etf_arr
    borrow     = (fut_contracts_prev > 0).astype(float) * np.abs(fut_notional_prev) * ETF_BORROW_BP_ANNUAL / TD
    return gross_pnl - cost - borrow
