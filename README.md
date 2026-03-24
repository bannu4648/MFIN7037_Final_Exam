# MFIN7037 Final Exam — MAX / MAXβ Anomaly Study

A full quantitative research pipeline replicating and extending the MAX anomaly from **Bali, Ince & Ozsoylev (2025)** — *"MAX on Steroids: A New Measure of Investor Attraction to Lottery Stocks"*.

The project demonstrates that stocks with extreme positive daily returns (high MAX) subsequently underperform, and that a **beta-neutralised version (MAXβ)** — constructed via the paper's double-sort procedure — delivers a cleaner, higher Sharpe-ratio signal by isolating idiosyncratic lottery-seeking behaviour from systematic market risk.

**Sample**: 8,817 US equities (NYSE/AMEX/NASDAQ), 2010–2024 (177 months).

---

## Key Results

| Metric | MAX | MAXβ (Double Sort) | Paper (MAXβ) |
|---|---|---|---|
| Mean Monthly L/S Return | 0.82% | **0.86%** | 0.81% |
| Annualised Return | 10.35% | **10.88%** | ~9.7% |
| Annualised Volatility | 28.1% | **21.0%** | lower than MAX |
| **Sharpe Ratio (ann.)** | 0.37 | **0.52** | higher than MAX |
| Win Rate | 58.8% | 58.2% | — |
| Beta spread D10−D1 | **+0.59** | **+0.02 ≈ 0** | ≈ 0.000 |

The beta-spread result is the paper's central validation: the double sort successfully removes systematic risk from the lottery signal, confirming that plain MAX is partly a proxy for market beta rather than pure idiosyncratic lottery demand.

> See [`analysis/README.md`](analysis/README.md) for a full breakdown of results, year-by-year returns, decile spreads, and a discussion of what the data proves.

---

## Project Structure

```
MFIN7037_Final_Exam/
│
├── pipeline/                       # Part 1 — Data engineering
│   ├── data_pipeline.py            # CRSP raw data → clean parquets
│   └── README.md                   # Data dictionary & column descriptions
│
├── analysis/                       # Part 2 — Signal construction & backtest
│   ├── strategy.py                 # Full pipeline (Steps 1–9)
│   ├── instructions.md             # Assignment specification
│   ├── README.md                   # Results, methodology, paper comparison
│   │
│   ├── strategy_returns.csv        # Monthly MAX long-short P&L
│   ├── strategy_mb_returns.csv     # Monthly MAXβ long-short P&L
│   ├── decile_returns.csv          # Avg return per decile (both signals)
│   │
│   ├── cumulative_pnl.png          # Cumulative P&L — MAX vs MAXβ
│   ├── decile_spread.png           # Avg return by decile bar chart
│   └── rolling_sharpe.png          # Rolling 12-month Sharpe ratio
│
├── clean/                          # Clean analysis-ready parquets (pipeline output)
│   ├── daily_data.parquet          # 16.4M rows — daily returns + FF5 factors
│   ├── monthly_data.parquet        # 786K rows  — monthly returns + market cap
│   └── ff_factors.parquet          # 3,774 rows — Fama-French 5 factors
│
├── venv/                           # Python virtual environment
├── requirements.txt                # Pinned dependencies
├── paper.pdf                       # Bali, Ince & Ozsoylev (2025)
└── README.md                       # This file
```

---

## Methodology Summary

### Part 1 — Data Pipeline (`pipeline/data_pipeline.py`)

Reads raw CRSP files and Fama-French factors, applies filters, and outputs three clean parquet files to `clean/`. Filters applied:

- US common equities only (`securitytype=EQTY`, `sharetype=NS`)
- NYSE, AMEX, NASDAQ only
- Penny stocks (< $1) removed
- Stocks with > 20% missing daily returns dropped
- Minimum 252 daily / 12 monthly observations required

> Raw CRSP files are not included. Place them in `data/` and run `python pipeline/data_pipeline.py`.

---

### Part 2 — Strategy (`analysis/strategy.py`)

#### MAX Signal
For each stock × month: average of the 5 highest daily returns within the month. Directly follows Bali, Cakici & Whitelaw (2011).

#### MAXβ Signal — Double-Sort Procedure (paper Section 3.2)
1. **Rolling beta**: 252-day rolling OLS of `(daily_return − rf) ~ mkt_rf` at each month-end
2. **Beta deciles**: Sort all stocks into 10 beta deciles each month
3. **MAX within beta**: Within each beta decile, sort by MAX into 10 sub-deciles
4. **Regroup**: Stocks sharing the same within-beta MAX rank → one MAXβ portfolio
   → Beta is flat across MAXβ deciles by construction

#### Portfolio Construction
- Decile 1 = lowest signal, Decile 10 = highest
- **Long D1, Short D10** (value-weighted by month-end market cap)
- Signal at month `t` → return at month `t+1` (no lookahead bias)

---

## Setup & Running

### 1. Activate the virtual environment

```bash
source venv/bin/activate
```

### 2. Run the strategy (Part 2)

```bash
python analysis/strategy.py
```

Runtime: ~23 seconds. Outputs written to `analysis/`.

### 3. Re-run the data pipeline (Part 1)

Requires raw CRSP parquet files in `data/`:

```bash
python pipeline/data_pipeline.py
```

### Installing from scratch

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Dependencies

Core packages (all pinned in `requirements.txt`):

| Package | Purpose |
|---|---|
| `pandas` | Data manipulation |
| `numpy` | Numerical computation |
| `pyarrow` | Parquet file I/O |
| `matplotlib` | Charts |
| `scipy` | (Available; not used in current implementation) |

---

## Data

| File | Rows | Stocks | Period |
|---|---|---|---|
| `daily_data.parquet` | 16,423,122 | 8,817 | Jan 2010 – Dec 2024 |
| `monthly_data.parquet` | 786,467 | 8,815 | Jan 2010 – Dec 2024 |
| `ff_factors.parquet` | 3,774 | — | Jan 2010 – Dec 2024 |

**Source**: CRSP (January 2025 vintage) + Kenneth French Data Library (FF5 factors, daily).

---

## Reference

Bali, T. G., Ince, B., & Ozsoylev, H. N. (2025). *MAX on Steroids: A New Measure of Investor Attraction to Lottery Stocks*. Georgetown University / Goethe University Frankfurt / Özyeğin University.
