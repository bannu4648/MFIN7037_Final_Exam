"""
Research extensions: sub-period metrics, drawdowns, Fama-French 5-factor regression.

Reads: analysis/outputs/strategy_*.csv, analysis/data/ff_factors.parquet
Writes: analysis/outputs/extensions/

Run from repo root:
  python code/extensions.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

REPO_ROOT = Path(__file__).resolve().parent.parent
STRATEGY_OUTPUT_DIR = REPO_ROOT / "analysis" / "outputs"
EXTENSIONS_DIR = STRATEGY_OUTPUT_DIR / "extensions"


def resolve_data_dir() -> Path:
    """Clean CRSP + FF panel lives in analysis/data/."""
    p = REPO_ROOT / "analysis" / "data"
    if (p / "ff_factors.parquet").exists():
        return p
    for legacy in (REPO_ROOT / "clean", REPO_ROOT / "clean" / "clean"):
        if (legacy / "ff_factors.parquet").exists():
            return legacy
    return p


DATA_DIR = resolve_data_dir()


def load_strategy_returns() -> tuple[pd.DataFrame, pd.DataFrame]:
    max_df = pd.read_csv(STRATEGY_OUTPUT_DIR / "strategy_returns.csv")
    mb_df = pd.read_csv(STRATEGY_OUTPUT_DIR / "strategy_mb_returns.csv")
    for df in (max_df, mb_df):
        df["month"] = df["month"].astype(str)
    return max_df, mb_df


def monthly_factors_from_daily() -> pd.DataFrame:
    ff = pd.read_parquet(DATA_DIR / "ff_factors.parquet")
    ff = ff.copy()
    ff["date"] = pd.to_datetime(ff["date"])
    ff["month"] = ff["date"].dt.to_period("M").astype(str)
    cols = ["mkt_rf", "smb", "hml", "rmw", "cma", "rf"]
    for c in cols:
        if c not in ff.columns:
            raise KeyError(f"ff_factors missing column {c!r}")

    def _compound(s: pd.Series) -> float:
        return float((1.0 + s.astype(float)).prod() - 1.0)

    return ff.groupby("month", sort=True)[cols].agg(_compound).reset_index()


def period_slice(
    df: pd.DataFrame,
    col: str,
    months: pd.Series,
    mask: pd.Series,
) -> np.ndarray:
    s = df.set_index("month")[col]
    idx = months[mask].values
    return s.reindex(idx).dropna().values


def performance_stats(monthly_returns: np.ndarray) -> dict:
    r = np.asarray(monthly_returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 2:
        return {
            "n_months": n,
            "mean_monthly": np.nan,
            "vol_monthly": np.nan,
            "sharpe_ann": np.nan,
            "cum_return": np.nan,
            "max_drawdown": np.nan,
        }
    mean_m = float(np.mean(r))
    vol_m = float(np.std(r, ddof=1))
    sharpe = (mean_m / vol_m) * np.sqrt(12) if vol_m > 0 else np.nan
    wealth = np.cumprod(1.0 + r)
    cum_ret = float(wealth[-1] - 1.0)
    peak = np.maximum.accumulate(wealth)
    dd = wealth / peak - 1.0
    max_dd = float(np.min(dd))
    return {
        "n_months": n,
        "mean_monthly": mean_m,
        "vol_monthly": vol_m,
        "sharpe_ann": sharpe,
        "cum_return": cum_ret,
        "max_drawdown": max_dd,
    }


def wealth_and_drawdown(r: pd.Series) -> tuple[pd.Series, pd.Series]:
    w = (1.0 + r).cumprod()
    peak = w.cummax()
    dd = w / peak - 1.0
    return w, dd


def ols_with_se(y: np.ndarray, X: np.ndarray, names: list[str]) -> pd.DataFrame:
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    n, k = X.shape
    Z = np.column_stack([np.ones(n), X])
    beta, _, rank, _ = np.linalg.lstsq(Z, y, rcond=None)
    if rank < Z.shape[1]:
        raise RuntimeError("Singular design matrix in FF regression")
    resid = y - Z @ beta
    dof = n - Z.shape[1]
    mse = float((resid**2).sum() / dof)
    cov = mse * np.linalg.inv(Z.T @ Z)
    se = np.sqrt(np.clip(np.diag(cov), 0, np.inf))
    t_stat = np.where(se > 0, beta / se, np.nan)
    terms = ["const"] + names
    r2 = 1.0 - (resid**2).sum() / ((y - y.mean()) ** 2).sum()
    return pd.DataFrame(
        {"term": terms, "coef": beta, "stderr": se, "t_stat": t_stat, "n": n, "r2": r2}
    )


def merge_returns_factors(
    strat: pd.DataFrame,
    monthly_ff: pd.DataFrame,
    label: str,
) -> pd.DataFrame:
    m = strat.merge(monthly_ff, on="month", how="inner")
    m["strategy"] = label
    return m


def run_ff5_regression(merged: pd.DataFrame) -> pd.DataFrame:
    y = merged["long_short_return"].values
    X = merged[["mkt_rf", "smb", "hml", "rmw", "cma"]].values
    return ols_with_se(y, X, ["MKT", "SMB", "HML", "RMW", "CMA"])


def plot_cumulative_and_drawdown(
    r_max: pd.Series,
    r_mb: pd.Series,
    out_path: Path,
) -> None:
    shared = r_max.index.intersection(r_mb.index)
    r_max = r_max.loc[shared].sort_index()
    r_mb = r_mb.loc[shared].sort_index()
    dates = pd.to_datetime([m + "-01" for m in shared])

    w_m, dd_m = wealth_and_drawdown(r_max)
    w_b, dd_b = wealth_and_drawdown(r_mb)

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

    ax0 = axes[0]
    y2020 = pd.Timestamp("2020-01-01")
    yend = dates.max()
    ax0.axvspan(dates.min(), y2020, alpha=0.06, color="#1f77b4", label="2010-2019")
    ax0.axvspan(y2020, yend, alpha=0.06, color="#e67e22", label="2020-2024")
    ax0.plot(dates, (w_m - 1.0) * 100, color="#1f77b4", lw=1.8, label="MAX")
    ax0.plot(dates, (w_b - 1.0) * 100, color="#e74c3c", lw=1.8, ls="--", label="MAX beta (double sort)")
    ax0.axhline(0, color="black", lw=0.6, ls=":")
    ax0.set_ylabel("Cumulative return (%)")
    ax0.set_title("Long D1 / Short D10 — cumulative performance and drawdowns")
    ax0.legend(loc="upper left", ncol=2, fontsize=9)
    ax0.xaxis.set_major_locator(mdates.YearLocator(2))
    ax0.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax0.grid(axis="y", lw=0.3, alpha=0.5)

    ax1 = axes[1]
    ax1.fill_between(dates, dd_m.values * 100, 0, alpha=0.35, color="#1f77b4", label="MAX drawdown")
    ax1.plot(dates, dd_m.values * 100, color="#1f77b4", lw=1.0)
    ax1.plot(dates, dd_b.values * 100, color="#e74c3c", lw=1.2, ls="--", label="MAX beta drawdown")
    ax1.axhline(0, color="black", lw=0.6, ls=":")
    ax1.set_ylabel("Drawdown from peak (%)")
    ax1.set_xlabel("Date")
    ax1.legend(loc="lower left", fontsize=9)
    ax1.xaxis.set_major_locator(mdates.YearLocator(2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.grid(axis="y", lw=0.3, alpha=0.5)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_subperiod_sharpe(rows: list[dict], out_path: Path) -> None:
    df = pd.DataFrame(rows)
    pivot = df.pivot(index="period", columns="strategy", values="sharpe_ann")
    pivot = pivot.reindex(["2010-2019", "2020-2024"])
    x = np.arange(len(pivot.index))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w / 2, pivot["MAX"], width=w, label="MAX", color="#1f77b4", edgecolor="white")
    ax.bar(x + w / 2, pivot["MAX_BETA"], width=w, label="MAX beta", color="#e74c3c", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(list(pivot.index))
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("Sharpe ratio (annualised)")
    ax.set_title("Sub-sample Sharpe: decay after 2020")
    ax.legend()
    ax.grid(axis="y", lw=0.3, alpha=0.5)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    STRATEGY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    EXTENSIONS_DIR.mkdir(parents=True, exist_ok=True)

    print("Extensions - FF factors from:", DATA_DIR)
    print("Extensions - strategy CSVs from:", STRATEGY_OUTPUT_DIR)
    print("Extensions - writing to:", EXTENSIONS_DIR)

    max_df, mb_df = load_strategy_returns()
    monthly_ff = monthly_factors_from_daily()

    months = max_df["month"]
    m_per = pd.to_datetime(months + "-01")
    mask_pre = m_per < pd.Timestamp("2020-01-01")
    mask_post = m_per >= pd.Timestamp("2020-01-01")

    rows = []
    for label, df in [("MAX", max_df), ("MAX_BETA", mb_df)]:
        col = "long_short_return"
        for period_name, mask in [
            ("2010-2019", mask_pre),
            ("2020-2024", mask_post),
            ("Full sample", pd.Series(True, index=months.index)),
        ]:
            r = period_slice(df, col, months, mask)
            st = performance_stats(r)
            rows.append(
                {
                    "strategy": label,
                    "period": period_name,
                    "n_months": st["n_months"],
                    "mean_monthly_pct": st["mean_monthly"] * 100,
                    "vol_monthly_pct": st["vol_monthly"] * 100,
                    "sharpe_ann": st["sharpe_ann"],
                    "cum_return_pct": st["cum_return"] * 100,
                    "max_drawdown_pct": st["max_drawdown"] * 100,
                }
            )

    summary = pd.DataFrame(rows)
    summary_path = EXTENSIONS_DIR / "subperiod_metrics.csv"
    summary.to_csv(summary_path, index=False)
    print("\nSaved:", summary_path)
    print(summary.to_string(index=False))

    m_max = merge_returns_factors(max_df, monthly_ff, "MAX")
    m_mb = merge_returns_factors(mb_df, monthly_ff, "MAX_BETA")
    reg_max = run_ff5_regression(m_max)
    reg_mb = run_ff5_regression(m_mb)
    reg_max = reg_max.copy()
    reg_mb = reg_mb.copy()
    reg_max["strategy"] = "MAX"
    reg_mb["strategy"] = "MAX_BETA"
    reg_out = pd.concat([reg_max, reg_mb], ignore_index=True)
    reg_path = EXTENSIONS_DIR / "ff5_regression.csv"
    reg_out.to_csv(reg_path, index=False)
    print("\nSaved:", reg_path)
    for strat, tbl in [("MAX", reg_max), ("MAX_BETA", reg_mb)]:
        print(f"\n--- FF5 regression ({strat}) - OLS, homoskedastic SE ---")
        disp = tbl[["term", "coef", "stderr", "t_stat"]].copy()
        print(disp.to_string(index=False))
        print(f"  R^2 = {tbl['r2'].iloc[0]:.4f}  n = {int(tbl['n'].iloc[0])}")

    r_max_s = max_df.set_index("month")["long_short_return"].sort_index()
    r_mb_s = mb_df.set_index("month")["long_short_return"].sort_index()
    dd_png = EXTENSIONS_DIR / "cumulative_drawdown.png"
    plot_cumulative_and_drawdown(r_max_s, r_mb_s, dd_png)
    print("Saved:", dd_png.name)

    bar_rows = [r for r in rows if r["period"] in ("2010-2019", "2020-2024")]
    sharpe_png = EXTENSIONS_DIR / "sharpe_by_subsample.png"
    plot_subperiod_sharpe(bar_rows, sharpe_png)
    print("Saved:", sharpe_png.name)

    print("\nDone.")


if __name__ == "__main__":
    main()
