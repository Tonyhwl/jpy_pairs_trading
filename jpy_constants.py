# Contract specs and cost constants for the JPY pair strategy.
# Extracted from the sibling prop_trading repo so this folder is self-contained.

CAPITAL_USD = 1_000_000
TD          = 252

# JPY micro futures (CME): mult is the contract multiplier (JPY per index point).
CONTRACT_SPECS = {
    "MJY": {"mult": 1_250_000, "cost_bp": 0.00050, "min_tick": 0.0000005},
}

ETF_COST_BP          = 0.00050  # commission + half-spread, per cross
ETF_BORROW_BP_ANNUAL = 0.0100   # 100 bp annual short-borrow charge

# Position sizing
SLEEVE_TARGET_VOL_DAILY = 0.130
SCALE_CAP               = 10.0
VOL_LOOKBACK_PAIR       = 60
