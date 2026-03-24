"""
data_pipeline.py — Data pipeline for MAX/MAXβ anomaly study.

Reads CRSP daily/monthly stock files and Fama-French 5 factors,
filters to US common equities on major exchanges (2010–2024),
cleans the data, and outputs analysis-ready parquet files.
"""

import pandas as pd
import pyarrow.parquet as pq
import numpy as np
from pathlib import Path
import time

# ── Configuration ──────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "clean"

START_DATE = "2010-01-01"
END_DATE = "2024-12-31"

VALID_EXCHANGES = ["N", "A", "Q"]  # NYSE, AMEX, NASDAQ
MIN_PRICE = 1.0
MAX_MISSING_PCT = 0.20
MIN_DAILY_OBS = 252
MIN_MONTHLY_OBS = 12
FFILL_LIMIT = 5

DAILY_COLS = [
    "permno", "dlycaldt", "dlyret", "dlyvol", "dlycap", "dlyprc",
    "ticker", "primaryexch", "securitytype", "sharetype",
]

MONTHLY_COLS = [
    "permno", "mthcaldt", "mthret", "mthvol", "mthcap", "mthprc",
    "ticker", "primaryexch", "securitytype", "sharetype",
]


# ── Step 1: Reading ───────────────────────────────────────────────────────────

def read_daily_data() -> pd.DataFrame:
    """Read CRSP daily file with column pruning and pyarrow row filters."""
    print("Reading CRSP daily data (large file — may take a minute)...")
    t0 = time.time()

    filters = [
        ("dlycaldt", ">=", START_DATE),
        ("dlycaldt", "<=", END_DATE),
        ("securitytype", "==", "EQTY"),
        ("sharetype", "==", "NS"),
        ("primaryexch", "in", VALID_EXCHANGES),
    ]

    df = pq.read_table(
        DATA_DIR / "crsp_202501.dsf_v2.parquet",
        columns=DAILY_COLS,
        filters=filters,
    ).to_pandas()

    print(f"  Loaded {len(df):,} rows, {df['permno'].nunique():,} stocks "
          f"in {time.time() - t0:.1f}s")
    return df


def read_monthly_data() -> pd.DataFrame:
    """Read CRSP monthly file with column pruning and pyarrow row filters."""
    print("Reading CRSP monthly data...")
    t0 = time.time()

    filters = [
        ("mthcaldt", ">=", START_DATE),
        ("mthcaldt", "<=", END_DATE),
        ("securitytype", "==", "EQTY"),
        ("sharetype", "==", "NS"),
        ("primaryexch", "in", VALID_EXCHANGES),
    ]

    df = pq.read_table(
        DATA_DIR / "crsp_202501.msf_v2.parquet",
        columns=MONTHLY_COLS,
        filters=filters,
    ).to_pandas()

    print(f"  Loaded {len(df):,} rows, {df['permno'].nunique():,} stocks "
          f"in {time.time() - t0:.1f}s")
    return df


def read_ff_factors() -> pd.DataFrame:
    """Read Fama-French 5 factors and filter to the study period."""
    print("Reading Fama-French 5 factors...")
    ff = pd.read_parquet(DATA_DIR / "ff.five_factor.parquet")
    ff["date"] = pd.to_datetime(ff["dt"])
    ff = ff[(ff["date"] >= START_DATE) & (ff["date"] <= END_DATE)]
    ff = ff.drop(columns=["dt"]).sort_values("date").reset_index(drop=True)
    print(f"  Loaded {len(ff):,} trading days "
          f"({ff['date'].min().date()} to {ff['date'].max().date()})")
    return ff


# ── Step 2: Cleaning daily data ───────────────────────────────────────────────

def clean_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Clean daily data: parse dates, remove penny stocks, drop sparse stocks."""
    print("Cleaning daily data...")
    n0 = len(df)

    df["date"] = pd.to_datetime(df["dlycaldt"])
    df = df.sort_values(["permno", "date"]).reset_index(drop=True)

    # Penny stock filter — CRSP stores negative prices for bid-ask midpoints
    df = df[df["dlyprc"].abs() >= MIN_PRICE]
    print(f"  Penny stock filter:   {len(df):,} rows (removed {n0 - len(df):,})")

    # Forward-fill small gaps in volume and market cap within each stock
    for col in ["dlyvol", "dlycap"]:
        df[col] = df.groupby("permno")[col].transform(
            lambda s: s.ffill(limit=FFILL_LIMIT)
        )

    # Drop stocks where >20% of daily returns are missing
    miss_pct = df.groupby("permno")["dlyret"].transform(lambda s: s.isna().mean())
    df = df[miss_pct <= MAX_MISSING_PCT]
    print(f"  Missing-return filter: {len(df):,} rows")

    # Drop stocks with fewer than 1 year of daily observations
    obs_count = df.groupby("permno")["date"].transform("count")
    df = df[obs_count >= MIN_DAILY_OBS]
    print(f"  Min-obs filter:       {len(df):,} rows, {df['permno'].nunique():,} stocks")

    # Drop any remaining rows where return is NaN
    df = df.dropna(subset=["dlyret"])
    print(f"  Drop NaN returns:     {len(df):,} rows")

    return df


# ── Step 3: Cleaning monthly data ─────────────────────────────────────────────

def clean_monthly(df: pd.DataFrame, valid_permnos: set) -> pd.DataFrame:
    """Clean monthly data, keeping only permnos that survived daily cleaning."""
    print("Cleaning monthly data...")

    df["date"] = pd.to_datetime(df["mthcaldt"])
    df = df.sort_values(["permno", "date"]).reset_index(drop=True)

    # Keep only permnos that survived daily cleaning for cross-dataset consistency
    df = df[df["permno"].isin(valid_permnos)]
    print(f"  Permno filter:        {len(df):,} rows, {df['permno'].nunique():,} stocks")

    # Penny stock filter
    df = df[df["mthprc"].abs() >= MIN_PRICE]

    # Drop stocks with fewer than 12 months of observations
    obs_count = df.groupby("permno")["date"].transform("count")
    df = df[obs_count >= MIN_MONTHLY_OBS]

    # Forward-fill small gaps
    for col in ["mthvol", "mthcap"]:
        df[col] = df.groupby("permno")[col].transform(
            lambda s: s.ffill(limit=FFILL_LIMIT)
        )

    # Drop NaN returns
    df = df.dropna(subset=["mthret"])
    print(f"  Final:                {len(df):,} rows, {df['permno'].nunique():,} stocks")

    return df


# ── Steps 4–5: Derived columns & FF merge ─────────────────────────────────────

def build_daily_output(daily: pd.DataFrame, ff: pd.DataFrame) -> pd.DataFrame:
    """Rename CRSP columns, add month, merge Fama-French factors."""
    print("Building daily output dataset...")

    out = daily[["date", "permno", "ticker", "dlyret", "dlyvol", "dlycap", "dlyprc"]].copy()
    out = out.rename(columns={
        "dlyret": "daily_return",
        "dlyvol": "volume",
        "dlycap": "market_cap",
        "dlyprc": "price",
    })
    out["price"] = out["price"].abs()
    out["month"] = out["date"].dt.to_period("M").astype(str)

    # Merge all FF5 factor columns onto daily data by date
    out = out.merge(ff, on="date", how="left")

    unmatched = out["mkt_rf"].isna().sum()
    if unmatched > 0:
        print(f"  Warning: {unmatched:,} rows unmatched to FF factors (dropped)")
        out = out.dropna(subset=["mkt_rf"])

    out = out.sort_values(["permno", "date"]).reset_index(drop=True)
    print(f"  Daily output: {len(out):,} rows, {out['permno'].nunique():,} stocks")
    return out


def build_monthly_output(monthly: pd.DataFrame) -> pd.DataFrame:
    """Rename CRSP columns and add month identifier."""
    print("Building monthly output dataset...")

    out = monthly[["date", "permno", "ticker", "mthret", "mthvol", "mthcap"]].copy()
    out = out.rename(columns={
        "mthret": "monthly_return",
        "mthvol": "volume",
        "mthcap": "market_cap",
    })
    out["month"] = out["date"].dt.to_period("M").astype(str)
    out = out.sort_values(["permno", "date"]).reset_index(drop=True)
    print(f"  Monthly output: {len(out):,} rows, {out['permno'].nunique():,} stocks")
    return out


# ── Step 6: Validation ────────────────────────────────────────────────────────

def validate(daily: pd.DataFrame, monthly: pd.DataFrame) -> dict:
    """Run data-quality checks and return summary statistics."""
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)

    # --- Duplicates ---
    daily_dups = daily.duplicated(subset=["permno", "date"]).sum()
    monthly_dups = monthly.duplicated(subset=["permno", "date"]).sum()
    print(f"Daily duplicates:   {daily_dups}")
    print(f"Monthly duplicates: {monthly_dups}")
    assert daily_dups == 0, "Daily dataset has duplicate (permno, date) pairs!"
    assert monthly_dups == 0, "Monthly dataset has duplicate (permno, date) pairs!"

    # --- NaN returns ---
    daily_nan = daily["daily_return"].isna().mean()
    monthly_nan = monthly["monthly_return"].isna().mean()
    print(f"Daily NaN return rate:   {daily_nan:.4%}")
    print(f"Monthly NaN return rate: {monthly_nan:.4%}")
    assert daily_nan < 0.01, f"Daily NaN return rate too high: {daily_nan:.2%}"
    assert monthly_nan < 0.01, f"Monthly NaN return rate too high: {monthly_nan:.2%}"

    # --- Summary stats ---
    stats = {}
    for label, df, ret_col in [
        ("Daily", daily, "daily_return"),
        ("Monthly", monthly, "monthly_return"),
    ]:
        n_stocks = df["permno"].nunique()
        n_rows = len(df)
        date_min = df["date"].min().date()
        date_max = df["date"].max().date()
        avg_obs = n_rows / n_stocks if n_stocks else 0

        print(f"\n--- {label} Dataset ---")
        print(f"  Rows:            {n_rows:,}")
        print(f"  Stocks:          {n_stocks:,}")
        print(f"  Date range:      {date_min} to {date_max}")
        print(f"  Avg obs/stock:   {avg_obs:.0f}")
        print(f"  Columns:         {list(df.columns)}")

        stats[label.lower()] = {
            "rows": n_rows,
            "stocks": n_stocks,
            "date_min": str(date_min),
            "date_max": str(date_max),
        }

    # --- Stocks per year (daily) ---
    yearly = daily.groupby(daily["date"].dt.year)["permno"].nunique()
    print("\n--- Stocks per Year (daily) ---")
    for year, count in yearly.items():
        print(f"  {year}: {count:,}")

    print("\nAll validations passed!")
    return stats


# ── Step 7: Save ───────────────────────────────────────────────────────────────

def save_outputs(daily: pd.DataFrame, monthly: pd.DataFrame, ff: pd.DataFrame):
    """Write clean parquet files to the output directory."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nSaving output files...")
    daily.to_parquet(OUTPUT_DIR / "daily_data.parquet", index=False)
    print(f"  Saved daily_data.parquet   ({len(daily):,} rows)")

    monthly.to_parquet(OUTPUT_DIR / "monthly_data.parquet", index=False)
    print(f"  Saved monthly_data.parquet ({len(monthly):,} rows)")

    ff.to_parquet(OUTPUT_DIR / "ff_factors.parquet", index=False)
    print(f"  Saved ff_factors.parquet   ({len(ff):,} rows)")

    print(f"\nAll files written to {OUTPUT_DIR}/")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()

    # Read raw data
    daily_raw = read_daily_data()
    monthly_raw = read_monthly_data()
    ff = read_ff_factors()

    # Clean
    daily_clean = clean_daily(daily_raw)
    del daily_raw

    valid_permnos = set(daily_clean["permno"].unique())
    monthly_clean = clean_monthly(monthly_raw, valid_permnos)
    del monthly_raw

    # Build output datasets (rename, derive columns, merge FF)
    daily_out = build_daily_output(daily_clean, ff)
    del daily_clean

    monthly_out = build_monthly_output(monthly_clean)
    del monthly_clean

    # Validate
    stats = validate(daily_out, monthly_out)

    # Save
    save_outputs(daily_out, monthly_out, ff)

    elapsed = time.time() - t_start
    print(f"\nPipeline completed in {elapsed:.1f}s")
    return stats


if __name__ == "__main__":
    main()
