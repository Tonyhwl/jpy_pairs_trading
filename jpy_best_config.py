# Full OOS metrics for selected config: beta_window=126, zscore_window=30,
# entry_threshold=1.00, exit_threshold=0.00, max_hold_days=30

import math
import numpy as np
import pandas as pd
from scipy import stats as sps
from scipy.special import ndtr

from jpy_pnl import (load_pair_data, compute_zscore, run_state_machine,
                     pnl_from_position, CAPITAL_USD, TD)

sleeve_capital  = CAPITAL_USD * 0.10
is_start        = pd.Timestamp("2018-01-01")
is_end          = pd.Timestamp("2021-12-31")
oos_start       = pd.Timestamp("2022-01-01")
beta_window, zscore_window = 126, 30
entry_threshold, exit_threshold, max_hold_days = 1.00, 0.00, 30

print("loading data...")
d = load_pair_data()
common, fut_prices_a, etf_prices_a = d["common"], d["fut_prices_a"], d["etf_prices_a"]

print("computing z-scores...")
z_arr = compute_zscore(d["log_fut"], d["log_etf"], beta_window, zscore_window)

print("running state machine and pnl...")
position = run_state_machine(z_arr, entry_threshold, exit_threshold, max_hold_days)
pnl_arr  = pnl_from_position(position, fut_prices_a, etf_prices_a, sleeve_capital)

is_mask  = (common >= is_start) & (common <= is_end)
oos_mask = (common >= oos_start)

pnl_oos = pd.Series(pnl_arr, index=common)[oos_mask]
ret_oos = (pnl_oos / CAPITAL_USD).dropna()
n_oos   = len(ret_oos)

sharpe = ret_oos.mean() / ret_oos.std() * math.sqrt(TD)
cagr   = (1 + ret_oos).prod() ** (TD / n_oos) - 1
vol    = ret_oos.std() * math.sqrt(TD)
eq_c   = (1 + ret_oos).cumprod()
max_dd = float((eq_c / eq_c.cummax() - 1).min())

print(f"\nconfig: beta_window={beta_window}, zscore_window={zscore_window}, "
      f"entry={entry_threshold}, exit={exit_threshold}, max_hold={max_hold_days}")
print(f"oos: {oos_start.date()} to {common[oos_mask][-1].date()}")
print(f"\nheadline (oos):")
print(f"  sharpe:  {sharpe:+.3f}")
print(f"  cagr:    {cagr*100:+.2f}%")
print(f"  vol:     {vol*100:.2f}%")
print(f"  max_dd:  {max_dd*100:.2f}%")

# active-bar moments
active_mask = position[oos_mask] != 0
ret_active  = ret_oos.values[active_mask[:len(ret_oos)]]
skew = float(sps.skew(ret_active))
kurt = float(sps.kurtosis(ret_active))
print(f"  skew (active):  {skew:.3f}")
print(f"  kurt (active):  {kurt:.3f}")
print(f"  active bars:    {active_mask.sum()} / {n_oos}  ({active_mask.mean():.1%})")

# trade-level stats
position_s = pd.Series(position, index=common)
trades = []
state = 0; trade_pnl = 0.0; trade_hold = 0; trade_dir = 0
for date, p in position_s[oos_mask].items():
    dp = pnl_oos.get(date, 0.0)
    if state == 0:
        if p != 0:
            state = p; trade_pnl = dp; trade_hold = 1; trade_dir = p
    elif p == state:
        trade_pnl += dp; trade_hold += 1
    else:
        trades.append({"dir": trade_dir, "pnl": trade_pnl, "hold": trade_hold})
        if p != 0:
            state = p; trade_pnl = dp; trade_hold = 1; trade_dir = p
        else:
            state = 0
if state != 0:
    trades.append({"dir": trade_dir, "pnl": trade_pnl, "hold": trade_hold})

trades_df = pd.DataFrame(trades)
pnl_t     = trades_df["pnl"].values
wins      = pnl_t[pnl_t > 0]
losses    = pnl_t[pnl_t <= 0]
hit_rate  = len(wins) / len(pnl_t)
profit_factor  = wins.sum() / abs(losses.sum()) if len(losses) else np.inf
win_loss_ratio = wins.mean() / abs(losses.mean()) if len(losses) else np.inf
n_long  = (trades_df["dir"] ==  1).sum()
n_short = (trades_df["dir"] == -1).sum()

print(f"\ntrade-level (oos):")
print(f"  trades:         {len(trades_df)}  ({n_long} long / {n_short} short)")
print(f"  hit rate:       {hit_rate:.1%}")
print(f"  profit factor:  {profit_factor:.2f}")
print(f"  win/loss ratio: {win_loss_ratio:.2f}")
print(f"  avg hold:       {trades_df['hold'].mean():.1f} days")
print(f"  avg pnl/trade: ${pnl_t.mean():,.0f}")
print(f"  max win:       ${pnl_t.max():,.0f}")
print(f"  max loss:      ${pnl_t.min():,.0f}")

print(f"\nsubperiod (oos):")
for label, s, e in [("2022", "2022-01-01", "2022-12-31"),
                     ("2023", "2023-01-01", "2023-12-31"),
                     ("2024", "2024-01-01", "2024-12-31"),
                     ("2025", "2025-01-01", "2025-12-31")]:
    r = ret_oos.loc[s:e].dropna()
    if len(r) < 20:
        continue
    sh  = r.mean() / r.std() * math.sqrt(TD)
    cg  = (1 + r).prod() ** (TD / len(r)) - 1
    eq2 = (1 + r).cumprod()
    dd  = float((eq2 / eq2.cummax() - 1).min()) * 100
    print(f"  {label}: sharpe={sh:+.2f}  cagr={cg*100:+.1f}%  max_dd={dd:+.1f}%")

# block bootstrap
rng  = np.random.default_rng(42)
arr  = ret_oos.values
n_blocks = math.ceil(n_oos / 21)
boot = []
for _ in range(5000):
    starts = rng.integers(0, n_oos - 21 + 1, size=n_blocks)
    s = np.concatenate([arr[i:i+21] for i in starts])[:n_oos]
    if s.std() > 0:
        boot.append(s.mean() / s.std() * math.sqrt(TD))
boot = np.array(boot)
ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
print(f"\nblock bootstrap 95% ci: [{ci_lo:.2f}, {ci_hi:.2f}]")

# sign-flip
rng2    = np.random.default_rng(43)
null_sf = []
for _ in range(2000):
    flip = rng2.choice([-1, 1], size=n_oos)
    s = arr * flip
    if s.std() > 0:
        null_sf.append(s.mean() / s.std() * math.sqrt(TD))
null_sf = np.array(null_sf)
z_sf = (sharpe - null_sf.mean()) / null_sf.std()
p_sf = float((null_sf >= sharpe).mean())
print(f"sign-flip:  {z_sf:.2f}sigma  p={max(p_sf, 1/2000):.4f}")

# sign-flip (trade-level)
rng3    = np.random.default_rng(44)
n_t     = len(pnl_t)
ir      = pnl_t.mean() / pnl_t.std() * math.sqrt(n_t)
null_tr = []
for _ in range(5000):
    flip = rng3.choice([-1, 1], size=n_t)
    s = pnl_t * flip
    if s.std() > 0:
        null_tr.append(s.mean() / s.std() * math.sqrt(n_t))
null_tr = np.array(null_tr)
z_tr = (ir - null_tr.mean()) / null_tr.std()
p_tr = float((null_tr >= ir).mean())
print(f"sign-flip (trade-level, n={n_t}):  {z_tr:.2f}sigma  p={max(p_tr, 1/5000):.4f}")


def psr(sr, ref):
    num   = (sr - ref) * math.sqrt(n_oos - 1)
    denom = math.sqrt(max(1 - skew * sr + (kurt + 1) / 4 * sr ** 2, 1e-9))
    return float(ndtr(num / denom))


print(f"\npsr vs 1.0: {psr(sharpe, 1.0):.4f}")
print(f"psr vs 2.0: {psr(sharpe, 2.0):.4f}")
print(f"psr vs 3.0: {psr(sharpe, 3.0):.4f}")

# IS comparison
pnl_is  = pd.Series(pnl_arr, index=common)[is_mask]
ret_is  = (pnl_is / CAPITAL_USD).dropna()
ni      = len(ret_is)
sh_is   = ret_is.mean() / ret_is.std() * math.sqrt(TD)
cagr_is = (1 + ret_is).prod() ** (TD / ni) - 1
eq_is   = (1 + ret_is).cumprod()
dd_is   = float((eq_is / eq_is.cummax() - 1).min()) * 100
n_is_t  = int(((np.diff(position[is_mask], prepend=0) != 0) & (position[is_mask] != 0)).sum())
print(f"\nis (2018-2021):  sharpe={sh_is:+.3f}  cagr={cagr_is*100:+.1f}%  "
      f"max_dd={dd_is:+.1f}%  trades={n_is_t}")
