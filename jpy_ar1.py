# AR(1) null hypothesis test for JPY basis strategy
# simulates AR(1) returns with same rho and std as observed, checks if
# observed Sharpe is distinguishable from autocorrelated noise

import math
import numpy as np
import pandas as pd

from jpy_pnl import (load_pair_data, compute_zscore, run_state_machine,
                     pnl_from_position, CAPITAL_USD, TD)

strategy_capital  = CAPITAL_USD * 0.10
oos_start       = pd.Timestamp("2022-01-01")
beta_window, zscore_window = 126, 30
entry_threshold, exit_threshold, max_hold_days = 1.00, 0.00, 30

d = load_pair_data()
common, fut_prices_a, etf_prices_a = d["common"], d["fut_prices_a"], d["etf_prices_a"]
z_arr    = compute_zscore(d["log_fut"], d["log_etf"], beta_window, zscore_window)
position = run_state_machine(z_arr, entry_threshold, exit_threshold, max_hold_days)
pnl_arr  = pnl_from_position(position, fut_prices_a, etf_prices_a, strategy_capital)

oos_mask = (common >= oos_start)
ret_oos  = (pd.Series(pnl_arr, index=common)[oos_mask] / CAPITAL_USD).dropna()
arr      = ret_oos.values
n_oos    = len(arr)
sharpe   = arr.mean() / arr.std() * math.sqrt(TD)

# AR(1) null: simulate returns with same autocorrelation and variance
rho = float(np.corrcoef(arr[:-1], arr[1:])[0, 1])
sd  = arr.std()
rng = np.random.default_rng(44)
null_ar = []
for _ in range(2000):
    innov = rng.normal(0, sd * np.sqrt(max(1 - rho ** 2, 1e-9)), size=n_oos)
    s = np.zeros(n_oos); s[0] = innov[0]
    for t in range(1, n_oos):
        s[t] = rho * s[t - 1] + innov[t]
    if s.std() > 0:
        null_ar.append(s.mean() / s.std() * math.sqrt(TD))
null_ar = np.array(null_ar)
z_ar = (sharpe - null_ar.mean()) / null_ar.std()
p_ar = float((null_ar >= sharpe).mean())

print(f"sharpe:     {sharpe:.3f}")
print(f"ar(1) rho:  {rho:.3f}")
print(f"null mean:  {null_ar.mean():.3f},  std: {null_ar.std():.3f}")
print(f"z:          {z_ar:.2f}sigma   p = {max(p_ar, 1/2000):.4f}")
