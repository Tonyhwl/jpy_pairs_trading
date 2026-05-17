# Contract specs and cost constants for the JPY pair strategy.

CAPITAL_USD = 1_000_000   # reference account size (Sharpe/return denominator)
TD          = 252         # trading days per year

# JPY micro futures (CME): mult is the contract multiplier (JPY per index point).
CONTRACT_SPECS = {
    "MJY": {"mult": 1_250_000, "cost_bp": 0.00050, "min_tick": 0.0000005},
}

ETF_COST_BP          = 0.00050   # commission + half-spread, per cross
ETF_BORROW_BP_ANNUAL = 0.0100    # 100 bp annual short-borrow charge

# Position sizing
TARGET_VOL_DAILY  = 0.130
SCALE_CAP         = 10.0
VOL_LOOKBACK_PAIR = 60
