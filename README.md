# Clean Dataset — MAX/MAXβ Anomaly Study

## Data Source

- **Stock data**: CRSP (Center for Research in Security Prices), January 2025 vintage
  - Daily stock file: `crsp_202501.dsf_v2.parquet`
  - Monthly stock file: `crsp_202501.msf_v2.parquet`
- **Factor data**: Kenneth French Data Library — Fama-French 5 Factors (daily)

## Universe

- **Stocks**: US common equities (`securitytype=EQTY`, `sharetype=NS`)
- **Exchanges**: NYSE (N), AMEX (A), NASDAQ (Q)
- **Filters applied**:
  - Penny stocks removed (absolute price < $1)
  - Stocks with >20% missing daily returns removed
  - Stocks with fewer than 252 daily observations (1 year) removed
  - Stocks with fewer than 12 monthly observations removed
  - Volume and market cap gaps forward-filled up to 5 periods

## Time Period

2010-01-04 to 2024-12-31

## Stock Count

- **8,817 unique stocks** in the daily dataset
- **8,815 unique stocks** in the monthly dataset
- Yearly breakdown (daily): ~4,400–5,800 stocks per year

## Output Files

### `daily_data.parquet` (16,423,122 rows)

| Column | Type | Description |
|---|---|---|
| `date` | datetime | Trading date |
| `permno` | int | CRSP permanent security identifier |
| `ticker` | str | Ticker symbol |
| `daily_return` | float | Daily holding-period return (CRSP `dlyret`, adjusted for splits/dividends) |
| `volume` | float | Daily trading volume (shares) |
| `market_cap` | float | Market capitalization (thousands of USD) |
| `price` | float | Closing price (absolute value; CRSP uses negative for bid-ask midpoint) |
| `month` | str | Year-month identifier (YYYY-MM) |
| `mkt_rf` | float | Market excess return (Fama-French) |
| `smb` | float | Small-minus-big factor |
| `hml` | float | High-minus-low factor |
| `rmw` | float | Robust-minus-weak factor |
| `cma` | float | Conservative-minus-aggressive factor |
| `rf` | float | Risk-free rate |

### `monthly_data.parquet` (786,467 rows)

| Column | Type | Description |
|---|---|---|
| `date` | datetime | Last trading date of the month |
| `permno` | int | CRSP permanent security identifier |
| `ticker` | str | Ticker symbol |
| `monthly_return` | float | Monthly holding-period return (CRSP `mthret`) |
| `volume` | float | Monthly trading volume (shares) |
| `market_cap` | float | Month-end market capitalization (thousands of USD) |
| `month` | str | Year-month identifier (YYYY-MM) |

### `ff_factors.parquet` (3,774 rows)

| Column | Type | Description |
|---|---|---|
| `date` | datetime | Trading date |
| `mkt_rf` | float | Market excess return |
| `smb` | float | Small-minus-big factor |
| `hml` | float | High-minus-low factor |
| `rmw` | float | Robust-minus-weak factor |
| `cma` | float | Conservative-minus-aggressive factor |
| `rf` | float | Risk-free rate |

## Usage Guide

### Person 2 — Signal Computation & Strategy (Engineer 2)

**Goal**: Compute MAX, MAXβ, build decile portfolios, and run the backtest.

**Computing MAX** (average of 5 highest daily returns per stock-month):

```python
import pandas as pd

daily = pd.read_parquet("data/clean/daily_data.parquet")

def compute_max(group):
    top5 = group.nlargest(5)
    return top5.mean() if len(top5) >= 5 else None

max_signal = (
    daily.groupby(["permno", "month"])["daily_return"]
    .apply(compute_max)
    .reset_index(name="MAX")
)
```

**Computing MAXβ** (beta-neutralized MAX via double sort):

1. Estimate each stock's market beta using a 252-day rolling regression of `daily_return` on `mkt_rf + rf` (both columns are in `daily_data.parquet`).
2. At each month-end, sort stocks into **deciles by beta**.
3. Within each beta decile, sort stocks into **deciles by MAX**.
4. Group all stocks with the same MAX rank across beta deciles — that forms the MAXβ portfolio.

```python
# Beta estimation (rolling 252-day window)
from numpy.linalg import lstsq

daily["mkt_total"] = daily["mkt_rf"] + daily["rf"]
# ... rolling regression of daily_return on mkt_total per permno ...
# Then merge beta onto monthly data and double-sort.
```

**Portfolio construction**: Use `monthly_data.parquet` for next-month returns. Join MAX/MAXβ signals (computed at month-end) to next month's `monthly_return` via `permno` + `month`. Use `market_cap` for value-weighted portfolio returns.

**Key columns you need**:
| File | Columns | Purpose |
|---|---|---|
| `daily_data.parquet` | `permno`, `month`, `daily_return` | Compute MAX (top-5 avg) |
| `daily_data.parquet` | `mkt_rf`, `rf` | Estimate beta for MAXβ |
| `monthly_data.parquet` | `permno`, `month`, `monthly_return` | Next-month portfolio returns |
| `monthly_data.parquet` | `market_cap` | Value-weighted portfolio construction |

---

### Person 3 — Analysis & Investment Story (Analyst)

**Goal**: Evaluate strategy performance, run extensions, build the investment narrative.

**P&L and performance metrics**: Use the long-short portfolio returns from Person 2 to compute cumulative P&L, Sharpe ratio, max drawdown, and volatility.

**Factor regression (alpha)**: Regress long-short returns on the Fama-French 5 factors from `ff_factors.parquet` to test whether the anomaly survives standard risk adjustment.

```python
import pandas as pd
import statsmodels.api as sm

ff = pd.read_parquet("data/clean/ff_factors.parquet")
# Merge monthly FF factors onto long-short returns
# OLS: long_short_return ~ mkt_rf + smb + hml + rmw + cma
# The intercept (alpha) is what you report.
```

**Extensions**:

- **Performance decay**: Split `monthly_data.parquet` into pre-2020 vs post-2020 using the `month` column and compare Sharpe ratios.
- **Liquidity filter**: Use `volume` in `daily_data.parquet` or `monthly_data.parquet` to remove the bottom quintile by volume, then re-run the strategy.
- **Size analysis**: Use `market_cap` in `monthly_data.parquet` to split into small-cap vs large-cap and check if the anomaly concentrates in small stocks.

**Key columns you need**:
| File | Columns | Purpose |
|---|---|---|
| `ff_factors.parquet` | `mkt_rf`, `smb`, `hml`, `rmw`, `cma`, `rf` | Factor regressions for alpha |
| `monthly_data.parquet` | `month`, `volume`, `market_cap` | Time splits, liquidity/size filters |

---

## Reproducing

```bash
python data_pipeline.py
```

Requires the raw CRSP and FF parquet files in `data/`. Outputs are written to `data/clean/`.
