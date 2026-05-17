# JPY Basis Pair Trading

Mean-reversion on the residual log-spread between CME 6J front-month
futures and the FXY ETF.

- IS: 2018-2021
- OOS: Jan 2022 - present
- Strategy capital: $100K (10% risk allocation on a $1M reference account)
- Vol-targeted via daily realised vol with a scale cap

## Strategy

Hedge ratio comes from a 126-day rolling OLS of log(6J) on log(FXY).
The signal is the 30-day z-score of the residual. Enter on |z| >= 1.0,
exit when z crosses 0, after 30 days, or on a stop at |z| >= 4. Sizing
scales the futures leg to a daily vol target; the ETF leg matches notional.

## Files

| file                  | purpose                                                |
|-----------------------|--------------------------------------------------------|
| `jpy_constants.py`    | Contract specs, cost params, vol targets               |
| `jpy_pnl.py`          | Shared: data load, z-score, state machine, PnL math    |
| `jpy_sweep.py`        | Grid sweep on IS, pareto frontier + OOS check          |
| `jpy_best_config.py`  | Full OOS metrics for the selected config               |
| `jpy_ar1.py`          | AR(1) null test for the OOS Sharpe                     |
| `jpy_plots.py`        | Paper figures: z-score, equity, drawdown, heatmap      |

## Selected config (from IS sweep)

```
beta_window     = 126
zscore_window   = 30
entry_threshold = 1.00
exit_threshold  = 0.00
max_hold_days   = 30
```

## Run

```bash
pip install numpy pandas scipy matplotlib
python jpy_sweep.py        # IS pareto frontier and OOS check
python jpy_best_config.py  # full OOS metrics for selected config
python jpy_ar1.py          # AR(1) null hypothesis test
python jpy_plots.py        # paper figures
```

## Data dependency

Contract specs and cost params live in `jpy_constants.py` (self-contained).

The price loaders (`load_futures_oc`, `load_etf_oc`) come from a sibling
`prop_trading/` repo and need raw DBN data + the `databento` package to
function. If that sibling repo isn't present, the scripts raise a clear
`ImportError` from `load_pair_data()`. Adjust the `sys.path` line at the
top of `jpy_pnl.py` if your layout differs, or substitute your own
loaders that return `(close_series, open_series)` pandas tuples.
