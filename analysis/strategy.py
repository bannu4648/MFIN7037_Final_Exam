"""
strategy.py — MAX and MAXβ Long-Short Strategy

Key methodology:
  MAX    : average of the 5 highest daily returns per stock-month
  MAXβ   : double-sort procedure
             1. Each month, sort stocks into beta deciles using a 252-day
                rolling OLS beta (daily excess return ~ market excess return)
             2. Within each beta decile, sort stocks into MAX deciles
             3. Regroup all stocks sharing the same within-beta MAX rank
                → forms the MAXβ portfolio (beta-neutral by construction)
  Strategy: Long D1 (low), Short D10 (high)
  Weights : value-weighted by month-end market cap
  Timing  : signal at month t → return realised in month t+1

Outputs (all written to analysis/):
  strategy_returns.csv    monthly MAX long-short P&L
  strategy_mb_returns.csv monthly MAXβ long-short P&L
  decile_returns.csv      avg return by decile (both signals)
  cumulative_pnl.png
  decile_spread.png
  rolling_sharpe.png
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import time

# ── Paths ─────────────────────────────────────────────────────────────────────

CLEAN_DIR = Path(__file__).parent.parent / "clean"
OUT_DIR   = Path(__file__).parent

# ── Step 1: Load data ───────────────────────────────────────────────────────────

def load_data():
    print("=" * 60)
    print("Loading data from", CLEAN_DIR)
    print("=" * 60)

    daily   = pd.read_parquet(CLEAN_DIR / "daily_data.parquet")
    monthly = pd.read_parquet(CLEAN_DIR / "monthly_data.parquet")

    print(f"  daily_data   : {len(daily):>12,} rows  |  {daily['permno'].nunique():,} stocks")
    print(f"  monthly_data : {len(monthly):>12,} rows  |  {monthly['permno'].nunique():,} stocks")
    return daily, monthly


# ── Step 2: Compute MAX ───────────────────────────────────────────────────────────

def compute_max(daily: pd.DataFrame) -> pd.DataFrame:
    """
    MAX = average of the 5 highest daily returns within each stock-month.
    Groups with fewer than 5 valid returns are dropped.
    Uses sort + groupby.head() to avoid slow Python-level apply.
    """
    print("\n[Step 1] Computing MAX...")
    t0 = time.time()

    d = daily[["permno", "month", "daily_return"]].dropna(subset=["daily_return"])

    counts = d.groupby(["permno", "month"])["daily_return"].transform("count")
    d = d[counts >= 5]

    d_sorted = d.sort_values(
        ["permno", "month", "daily_return"], ascending=[True, True, False]
    )
    top5 = d_sorted.groupby(["permno", "month"], sort=False).head(5)

    max_df = (
        top5.groupby(["permno", "month"])["daily_return"]
        .mean()
        .reset_index(name="MAX")
    )

    print(f"  MAX: {len(max_df):,} stock-months  ({time.time()-t0:.1f}s)")
    return max_df


# ── Step 3: Compute 252-day rolling market beta ──────────────────────────────────

def compute_rolling_beta(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate market beta at each month-end using a 252-day rolling window:

        (daily_return - rf)_t = α + β * mkt_rf_t + ε_t

    β = Cov(excess_ret, mkt_rf) / Var(mkt_rf)  [over past 252 trading days]

    Implemented fully vectorised via pandas rolling transforms — no per-stock
    Python loops.  min_periods=60 (~3 months) required for a valid estimate.

    Returns DataFrame [permno, month, beta] with the month-end rolling beta.
    """
    print("\n[Step 2] Computing 252-day rolling market beta (vectorised)...")
    t0 = time.time()

    d = daily[["permno", "date", "month", "daily_return", "rf", "mkt_rf"]].copy()
    d = d.dropna(subset=["daily_return", "rf", "mkt_rf"])
    d = d.sort_values(["permno", "date"])

    d["y"]  = d["daily_return"] - d["rf"]   # stock excess return
    d["x"]  = d["mkt_rf"]                   # market excess return
    d["xy"] = d["y"] * d["x"]
    d["x2"] = d["x"] ** 2

    # Rolling means per permno — all four moments in one transform pass
    roll = (
        d.groupby("permno", sort=False)[["y", "x", "xy", "x2"]]
        .transform(lambda s: s.rolling(252, min_periods=60).mean())
    )

    var_x  = roll["x2"] - roll["x"] ** 2
    cov_xy = roll["xy"] - roll["x"] * roll["y"]

    d["beta"] = np.where(var_x > 1e-12, cov_xy / var_x, np.nan)

    # Keep only month-end beta (last valid observation per permno-month)
    month_end_beta = (
        d.dropna(subset=["beta"])
        .sort_values(["permno", "date"])
        .groupby(["permno", "month"])["beta"]
        .last()
        .reset_index()
    )

    n_valid = month_end_beta["beta"].notna().sum()
    print(f"  Beta: {n_valid:,} valid (permno, month) estimates  ({time.time()-t0:.1f}s)")
    return month_end_beta


# ── Step 4: Build signal panel — join to next-month returns ──────────────────────

def build_panel(
    max_df: pd.DataFrame,
    beta_df: pd.DataFrame,
    monthly: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge MAX + month-end beta + market cap (all at month t) to next-month
    return (month t+1).  No lookahead bias.
    """
    print("\nBuilding panel (signal t → return t+1)...")

    # Merge MAX and beta
    signals = max_df.merge(beta_df, on=["permno", "month"], how="inner")

    # Attach month-t market cap (used for value-weighting)
    mktcap = monthly[["permno", "month", "market_cap"]].dropna(subset=["market_cap"])
    signals = signals.merge(mktcap, on=["permno", "month"], how="left")

    # Build prev_month column on returns side for the t→t+1 join
    ret = monthly[["permno", "month", "monthly_return"]].dropna(subset=["monthly_return"]).copy()
    ret["prev_month"] = (pd.PeriodIndex(ret["month"], freq="M") - 1).astype(str)

    panel = signals.merge(
        ret.rename(columns={"month": "next_month", "monthly_return": "ret_next"}),
        left_on=["permno", "month"],
        right_on=["permno", "prev_month"],
        how="inner",
    )
    panel = panel.drop(columns=["prev_month"])
    panel = panel.dropna(subset=["MAX", "beta", "ret_next"])

    print(f"  Panel: {len(panel):,} rows  |  {panel['month'].nunique()} months  "
          f"({panel['month'].min()} → {panel['month'].max()})")
    return panel


# ── Step 5: Decile assignment — MAX (simple) and MAXβ (double sort) ──────────────

def assign_deciles(panel: pd.DataFrame) -> pd.DataFrame:
    """
    MAX deciles
    -----------
    Each month, rank all stocks into deciles 1–10 by MAX.

    MAXβ deciles  (double-sort procedure)
    --------------------------------------
    Step 1: Each month, sort stocks into 10 beta deciles using the
            252-day rolling market beta.
    Step 2: Within each (month, beta_decile) group, rank stocks by
            MAX into 10 sub-deciles.
    Step 3: The MAXβ portfolio of rank n is formed by grouping all
            stocks that received MAX rank n across the 10 beta deciles.

    This procedure controls for systematic risk so that each MAXβ
    decile spans the full beta distribution (beta spread ≈ 0 across
    MAXβ deciles — verified in the validation step).
    """
    print("\n[Step 3] Assigning deciles (MAX simple sort; MAXβ double sort)...")

    def safe_decile(s):
        try:
            return pd.qcut(
                s.rank(method="first"), q=10, labels=range(1, 11)
            ).astype(float)
        except ValueError:
            return pd.Series(np.nan, index=s.index)

    panel = panel.copy()

    # ── MAX: simple monthly sort ──────────────────────────────────────────────
    panel["dec_MAX"] = panel.groupby("month")["MAX"].transform(safe_decile)

    # ── MAXβ Step 1: beta deciles each month ──────────────────────────────────
    panel["beta_decile"] = panel.groupby("month")["beta"].transform(safe_decile)
    panel = panel.dropna(subset=["dec_MAX", "beta_decile"])
    panel["beta_decile"] = panel["beta_decile"].astype(int)

    # ── MAXβ Step 2: within each (month, beta_decile), rank by MAX ────────────
    panel["dec_MAX_beta"] = (
        panel.groupby(["month", "beta_decile"])["MAX"]
        .transform(safe_decile)
    )

    panel = panel.dropna(subset=["dec_MAX", "dec_MAX_beta"])
    panel["dec_MAX"]      = panel["dec_MAX"     ].astype(int)
    panel["dec_MAX_beta"] = panel["dec_MAX_beta"].astype(int)
    panel = panel.drop(columns=["beta_decile"])

    print(f"  Done.  Panel: {len(panel):,} rows")
    return panel


# ── Step 6: Portfolio returns (value-weighted) ──────────────────────────────────

def compute_portfolio_returns(panel: pd.DataFrame):
    """
    Compute monthly value-weighted returns for each decile (MAX and MAXβ).
    Long-short = D1 − D10 (long low-MAX, short high-MAX).
    """
    print("\n[Step 4] Computing portfolio returns (value-weighted)...")

    def _vw(df, dec_col):
        df = df.copy()
        df["w"]     = df["market_cap"].fillna(0.0)
        df["ret_w"] = df["ret_next"] * df["w"]

        agg = df.groupby(["month", dec_col]).agg(
            ret_w_sum=("ret_w", "sum"),
            w_sum=("w", "sum"),
            ew_ret=("ret_next", "mean"),
        ).reset_index()

        agg["vw_ret"] = np.where(
            agg["w_sum"] > 0,
            agg["ret_w_sum"] / agg["w_sum"],
            agg["ew_ret"],
        )
        return agg

    def _ls(dec_ret, dec_col):
        """Build long-short DataFrame from decile returns."""
        long_r  = dec_ret[dec_ret[dec_col] ==  1].set_index("month")["vw_ret"]
        short_r = dec_ret[dec_ret[dec_col] == 10].set_index("month")["vw_ret"]
        idx     = long_r.index.intersection(short_r.index)
        ls      = long_r[idx] - short_r[idx]
        return pd.DataFrame({
            "month":             ls.index,
            "long_return":       long_r [idx].values,
            "short_return":      short_r[idx].values,
            "long_short_return": ls.values,
        }).sort_values("month").reset_index(drop=True)

    # MAX-sorted
    dec_max     = _vw(panel, "dec_MAX")
    strategy_df = _ls(dec_max, "dec_MAX")

    # MAXβ-sorted (double-sort)
    dec_mb         = _vw(panel, "dec_MAX_beta")
    strategy_mb_df = _ls(dec_mb, "dec_MAX_beta")

    # Decile spread table
    avg_max = (
        dec_max.groupby("dec_MAX")["vw_ret"].mean()
        .reset_index().rename(columns={"dec_MAX": "decile", "vw_ret": "avg_return_MAX"})
    )
    avg_mb = (
        dec_mb.groupby("dec_MAX_beta")["vw_ret"].mean()
        .reset_index().rename(columns={"dec_MAX_beta": "decile", "vw_ret": "avg_return_MAX_beta"})
    )
    decile_returns = avg_max.merge(avg_mb, on="decile")

    print(f"  MAX  L/S: {strategy_df['long_short_return'].mean():+.4%}/month  "
          f"({len(strategy_df)} months)")
    print(f"  MAXβ L/S: {strategy_mb_df['long_short_return'].mean():+.4%}/month  "
          f"({len(strategy_mb_df)} months)")

    return strategy_df, strategy_mb_df, decile_returns


# ── Step 7: Performance metrics ─────────────────────────────────────────────────

def compute_metrics(strategy_df: pd.DataFrame, label: str = "MAX") -> dict:
    r = strategy_df["long_short_return"]

    mean_r   = r.mean()
    vol      = r.std()
    ann_ret  = (1 + mean_r) ** 12 - 1
    ann_vol  = vol * np.sqrt(12)
    sharpe   = ann_ret / ann_vol if ann_vol > 0 else np.nan
    win_rate = (r > 0).mean()
    cum      = (1 + r).cumprod()
    max_dd   = ((cum - cum.cummax()) / cum.cummax()).min()

    metrics = {
        "Mean Monthly Return":   mean_r,
        "Annualised Return":     ann_ret,
        "Monthly Volatility":    vol,
        "Annualised Volatility": ann_vol,
        "Sharpe Ratio (ann.)":   sharpe,
        "Win Rate":              win_rate,
        "Max Drawdown":          max_dd,
        "N Months":              len(r),
    }

    print(f"\n{'='*54}")
    print(f"  {label} — Long D1 / Short D10 Performance")
    print(f"{'='*54}")
    fmt = {k: (f"{v:.4%}" if isinstance(v, float) and k != "N Months" else str(int(v)))
           for k, v in metrics.items()}
    fmt["Sharpe Ratio (ann.)"] = f"{sharpe:.4f}"
    for k, v in fmt.items():
        print(f"  {k:<28} {v}")

    return metrics


# ── Step 8: Visualisations ───────────────────────────────────────────────────────

def _to_dates(months):
    return pd.to_datetime([m + "-01" for m in months])


def plot_cumulative_pnl(strategy_df: pd.DataFrame, strategy_mb_df: pd.DataFrame):
    r_max = strategy_df   .set_index("month")["long_short_return"]
    r_mb  = strategy_mb_df.set_index("month")["long_short_return"]
    cum_max = (1 + r_max).cumprod() - 1
    cum_mb  = (1 + r_mb ).cumprod() - 1

    shared = cum_max.index.intersection(cum_mb.index)
    dates  = _to_dates(shared)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(dates, cum_max[shared].values * 100,
            color="#1f77b4", lw=1.8, label="MAX  (simple sort)")
    ax.plot(dates, cum_mb [shared].values * 100,
            color="#e74c3c", lw=1.8, ls="--", label="MAXβ (double sort)")
    ax.fill_between(dates, 0, cum_max[shared].values * 100, alpha=0.07, color="#1f77b4")
    ax.axhline(0, color="black", lw=0.8, ls=":")

    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_title("Cumulative P&L — Long D1 / Short D10 (MAX and MAXβ)", fontsize=13)
    ax.set_xlabel("Date"); ax.set_ylabel("Cumulative Return (%)")
    ax.legend(); plt.tight_layout()

    path = OUT_DIR / "cumulative_pnl.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Saved: {path.name}")


def plot_decile_spread(decile_returns: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, col, title in zip(
        axes,
        ["avg_return_MAX", "avg_return_MAX_beta"],
        ["MAX Signal (simple sort)", "MAXβ Signal (double sort)"],
    ):
        vals   = decile_returns[col] * 100
        colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in vals]
        ax.bar(decile_returns["decile"], vals, color=colors, edgecolor="white", width=0.7)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(f"Avg Next-Month Return by Decile\n({title})", fontsize=11)
        ax.set_xlabel("Decile  (1 = Low,  10 = High)")
        ax.set_ylabel("Avg Monthly Return (%)")
        ax.set_xticks(range(1, 11))
        ax.grid(axis="y", lw=0.4, alpha=0.5)

    plt.tight_layout()
    path = OUT_DIR / "decile_spread.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Saved: {path.name}")


def plot_rolling_sharpe(
    strategy_df: pd.DataFrame,
    strategy_mb_df: pd.DataFrame,
    window: int = 12,
):
    r_max = strategy_df   .set_index("month")["long_short_return"]
    r_mb  = strategy_mb_df.set_index("month")["long_short_return"]

    rs_max = r_max.rolling(window).mean() / r_max.rolling(window).std() * np.sqrt(12)
    rs_mb  = r_mb .rolling(window).mean() / r_mb .rolling(window).std() * np.sqrt(12)

    shared = rs_max.index.intersection(rs_mb.index)
    dates  = _to_dates(shared)

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(dates, rs_max[shared].values, color="#1f77b4", lw=1.4, label="MAX")
    ax.plot(dates, rs_mb [shared].values, color="#e74c3c", lw=1.4, ls="--", label="MAXβ")
    ax.axhline(0, color="black", lw=0.8, ls=":")
    ax.axhline(1, color="#2ecc71", lw=0.8, ls="--", alpha=0.6, label="Sharpe = 1")

    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_title(f"Rolling {window}-Month Sharpe Ratio — Long-Short Strategies", fontsize=12)
    ax.set_xlabel("Date"); ax.set_ylabel("Sharpe Ratio (ann.)")
    ax.legend(); plt.tight_layout()

    path = OUT_DIR / "rolling_sharpe.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Saved: {path.name}")


# ── Step 9: Validation (no lookahead + beta-flat check) ───────────────────────────

def validate(panel: pd.DataFrame):
    print("\n[Validation]")

    # ── Timing: signal_month + 1 = return_month ───────────────────────────────
    sig  = pd.PeriodIndex(panel["month"],      freq="M")
    ret  = pd.PeriodIndex(panel["next_month"], freq="M")
    lags = (ret - sig).map(lambda x: x.n)
    assert (lags == 1).all(), f"Lookahead bias! Lag distribution:\n{lags.value_counts()}"
    print("  No lookahead bias: signal_month + 1 = return_month  ✓")

    # ── No duplicates ─────────────────────────────────────────────────────────
    dups = panel.duplicated(subset=["permno", "month"]).sum()
    assert dups == 0, f"Duplicate (permno, month) rows: {dups}"
    print("  No duplicate (permno, month) pairs  ✓")

    # ── Beta should be flat across MAXβ deciles (double-sort key check) ─────────
    beta_by_dec = panel.groupby("dec_MAX_beta")["beta"].mean()
    spread = beta_by_dec.iloc[-1] - beta_by_dec.iloc[0]
    print(f"  Beta spread across MAXβ deciles (D10 − D1): {spread:+.4f}")

    beta_by_max = panel.groupby("dec_MAX")["beta"].mean()
    max_spread = beta_by_max.iloc[-1] - beta_by_max.iloc[0]
    print(f"  Beta spread across MAX  deciles (D10 − D1): {max_spread:+.4f}")

    print(f"  Signal months: {panel['month'].min()} → {panel['month'].max()}")
    print(f"  Return months: {panel['next_month'].min()} → {panel['next_month'].max()}")


# ── Main ───────────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()

    # Load
    daily, monthly = load_data()

    # Signals
    max_df  = compute_max(daily)
    beta_df = compute_rolling_beta(daily)

    # Panel
    panel = build_panel(max_df, beta_df, monthly)
    panel = assign_deciles(panel)

    # Returns and metrics
    strategy_df, strategy_mb_df, decile_returns = compute_portfolio_returns(panel)
    compute_metrics(strategy_df,    label="MAX")
    compute_metrics(strategy_mb_df, label="MAXβ (double sort)")

    # Save CSVs
    print("\n[Step 5] Saving output files...")
    strategy_df   .to_csv(OUT_DIR / "strategy_returns.csv",    index=False)
    strategy_mb_df.to_csv(OUT_DIR / "strategy_mb_returns.csv", index=False)
    decile_returns.to_csv(OUT_DIR / "decile_returns.csv",      index=False)
    print(f"  Saved: strategy_returns.csv  ({len(strategy_df)} rows)")
    print(f"  Saved: strategy_mb_returns.csv  ({len(strategy_mb_df)} rows)")
    print(f"  Saved: decile_returns.csv    ({len(decile_returns)} rows)")

    # Charts
    print("\n[Step 6] Generating charts...")
    plot_cumulative_pnl(strategy_df, strategy_mb_df)
    plot_decile_spread(decile_returns)
    plot_rolling_sharpe(strategy_df, strategy_mb_df)

    # Validate
    validate(panel)

    print(f"\nTotal runtime: {time.time() - t_start:.1f}s")
    print("Done.")


if __name__ == "__main__":
    main()
