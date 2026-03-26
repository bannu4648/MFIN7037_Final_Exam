# MFIN7037 Final Exam — MAX / MAXβ Lottery-Stock Anomaly

Replication and extensions of **Bali, Ince & Ozsoylev (2025)**, *MAX on Steroids: A New Measure of Investor Attraction to Lottery Stocks*. The project asks whether stocks with extreme positive daily returns (high **MAX**) underperform next month, and whether a **beta-neutral double sort (MAXβ)** delivers a cleaner, higher-Sharpe long–short signal.

---

## Repository layout

| Path | Contents |
|------|----------|
| **`code/`** | `strategy.py` — main backtest · `extensions.py` — sub-period metrics, FF5, drawdown figures · `data_pipeline.py` — optional rebuild of cleaned parquets from raw vendor files in **`data/`** |
| **`data/`** | Placeholder for **local** CRSP + Fama–French extracts consumed by `data_pipeline.py` (see below). The repository includes this folder but **not** the raw parquet files. |
| **`analysis/data/`** | **Cleaned** CRSP + Fama–French panel (`*.parquet`) — what `strategy.py` and `extensions.py` read. Large files are stored with **Git LFS** (GitHub’s 100 MB per-file limit). |
| **`analysis/outputs/`** | Base backtest: monthly long–short CSVs, decile table, core PNG charts (cumulative P&L, decile spread, rolling Sharpe) |
| **`analysis/outputs/extensions/`** | Extension outputs: sub-period stats, FF5 regression table, extension figures |
| **`requirements.txt`** | Pinned Python dependencies |

---

## Data: `data/` vs `analysis/data/`

**`code/data_pipeline.py`** reads **CRSP-style daily and monthly stock files** and a **Fama–French five-factor** extract from the root **`data/`** directory, filters and cleans them, and writes the processed parquets to **`analysis/data/`**.

Those **raw vendor files are not included in this repository** — they are too large (and often license-restricted). What **is** included is the **cleaned** panel under **`analysis/data/`**, which is enough to run the backtest and extensions without running the pipeline.

To rebuild `analysis/data/` yourself, copy these files into **`data/`** locally:

- `crsp_202501.dsf_v2.parquet` (daily)
- `crsp_202501.msf_v2.parquet` (monthly)
- `ff.five_factor.parquet`

Then:

```bash
python code/data_pipeline.py
```

If those files are missing, the script exits with a short message instead of a long traceback.

---

## What `analysis/` is for

| Subfolder | Role |
|-----------|------|
| **`analysis/data/`** | **Inputs** to `strategy.py` / `extensions.py` (cleaned parquets). |
| **`analysis/outputs/`** | **Primary results** from `strategy.py` (CSVs + baseline plots). |
| **`analysis/outputs/extensions/`** | **Secondary analysis** from `extensions.py` (requires step 1 outputs first). |

**Flow:** cleaned parquets → `strategy.py` → `analysis/outputs/` → `extensions.py` → `analysis/outputs/extensions/`.

---

## Quick start (environment)

**Git LFS:** Install [Git LFS](https://git-lfs.github.com/) and run **`git lfs install`** once on your machine. Then clone as usual; LFS pulls the real parquets under **`analysis/data/`**. If files look tiny or broken after clone, run **`git lfs pull`**.

From the project root:

```bash
cd MFIN7037_Final_Exam
python -m venv .venv
```

**Windows (PowerShell):** `.venv\Scripts\Activate.ps1` then `pip install -r requirements.txt`  
**macOS / Linux:** `source .venv/bin/activate` then `pip install -r requirements.txt`

Use the venv’s `python`. In PowerShell, chain commands with **`;`**, not `&&`.

---

## Running the code

Run **from the repository root**, in order.

**`strategy.py` is slow on purpose.** It scans millions of daily rows, builds MAX, rolling betas, and the full panel—expect **roughly a few minutes to 15+ minutes** depending on CPU and RAM. The process is **not stuck** if the console is quiet for a while; wait until you see **`Done.`** and the validation **OK** lines. Automated graders, CI, or AI agents should use a **generous timeout** (e.g. 20–30+ minutes) for this step so it is not killed mid-run.

### 1) Backtest — requires `analysis/data/*.parquet`

```bash
python code/strategy.py
```

**Outputs:** `strategy_returns.csv`, `strategy_mb_returns.csv`, `decile_returns.csv`, `cumulative_pnl.png`, `decile_spread.png`, `rolling_sharpe.png` in **`analysis/outputs/`**.

If log lines appear late or not at all, set **`PYTHONUNBUFFERED=1`** (PowerShell: `$env:PYTHONUNBUFFERED="1"`) so output prints immediately. **`extensions.py`** is much faster (seconds to a small number of minutes).

### 2) Extensions — requires step 1 + `analysis/data/ff_factors.parquet`

```bash
python code/extensions.py
```

**Outputs** in **`analysis/outputs/extensions/`**: `subperiod_metrics.csv`, `ff5_regression.csv`, `cumulative_drawdown.png`, `sharpe_by_subsample.png`.

### Checklist

- `analysis/outputs/strategy_returns.csv` and `strategy_mb_returns.csv` exist after step 1.
- `analysis/outputs/extensions/subperiod_metrics.csv` and `ff5_regression.csv` exist after step 2.
- `strategy.py` finishes with **`Done.`** and validation **OK** lines; no traceback.

---

## Methodology (short)

- **MAX** — For each stock × month: average of the five highest daily returns.
- **MAXβ** — Each month: sort into beta deciles (252-day rolling market beta), then within each beta decile sort by MAX into deciles; regroup equal MAX ranks across betas so the long–short leg is approximately beta-neutral.
- **Strategy** — Long decile 1 (low MAX or low MAXβ rank), short decile 10; **value-weighted** by month-end market cap; signal at month *t*, return in month *t+1*.

---

## Key results (indicative — rerun for exact numbers)

| Metric | MAX | MAXβ (double sort) |
|--------|-----|---------------------|
| Mean monthly L/S | ~0.82% | ~0.86% |
| Annualised Sharpe (full sample) | ~0.35 | ~0.49 |
| Beta spread D10−D1 | Large (MAX loads on beta) | ~0 (by construction) |

**Sharpe** uses **(mean monthly return ÷ monthly volatility) × √12** with sample σ (ddof = 1), matching **`strategy.py`**, **`extensions.py`**, and **`analysis/outputs/extensions/subperiod_metrics.csv`**.

Extension tables and figures: **`analysis/outputs/extensions/`**.

---

## Dependencies

`pandas`, `numpy`, `pyarrow`, `matplotlib`, `scipy` (see `requirements.txt`).

---

## Achievements (what this repo demonstrates)

1. **End-to-end quant research pipeline** — CRSP-style daily/monthly panel, Fama–French factors, signal construction, portfolio sorts, no-lookahead backtest.
2. **Paper-aligned MAXβ** — Double sort with explicit **beta-flat** check across MAXβ deciles.
3. **Extensions** — Drawdowns, sub-sample Sharpe decay, FF5 regression under **`analysis/outputs/extensions/`**.
4. **Clear separation** — Code in **`code/`**; cleaned data and results under **`analysis/`**.

---

## Reference

Bali, T. G., Ince, B., & Ozsoylev, H. N. (2025). *MAX on Steroids: A New Measure of Investor Attraction to Lottery Stocks*.
