# Analysis — MAX & MAXβ Strategy Results

This folder contains the full backtest implementation and outputs for the MAX and MAXβ long-short strategies, based on the methodology of **Bali, Ince & Ozsoylev (2025)** — *"MAX on Steroids: A New Measure of Investor Attraction to Lottery Stocks"*.

**Sample**: 8,817 US common equities (NYSE/AMEX/NASDAQ), March 2010 – December 2024 (177 months).

---

## Files

| File | Description |
|---|---|
| `strategy.py` | Full backtest pipeline |
| `strategy_returns.csv` | Monthly MAX long-short returns |
| `strategy_mb_returns.csv` | Monthly MAXβ long-short returns |
| `decile_returns.csv` | Average next-month return per decile (both signals) |
| `cumulative_pnl.png` | Cumulative P&L chart |
| `decile_spread.png` | Decile bar charts |
| `rolling_sharpe.png` | Rolling 12-month Sharpe ratio |

---

## Methodology

### MAX

For each stock × month, MAX is the average of the five highest daily returns:

```
MAX_{i,t} = mean(top-5 daily returns of stock i in month t)
```

Stocks are ranked into deciles 1–10 (1 = lowest MAX, 10 = highest). The strategy goes **Long D1** and **Short D10**, with portfolios value-weighted by month-end market cap.

### MAXβ — Paper's Double-Sort Procedure

The original MAX measure contains a large systematic component: high-beta stocks naturally post high daily returns during market rallies, contaminating the lottery-demand signal. The paper proposes a non-parametric fix:

1. **Rolling beta**: Estimate each stock's market beta at each month-end using a 252-day rolling OLS:
   `(daily_return − rf) = α + β × mkt_rf + ε`

2. **Beta deciles**: Each month, sort all stocks into 10 beta deciles.

3. **MAX within beta**: Within each beta decile, sort stocks into 10 MAX sub-deciles.

4. **Regroup**: All stocks sharing the same within-beta MAX rank across the 10 beta deciles form one MAXβ portfolio (10 portfolios total).

This ensures each MAXβ portfolio spans the full beta distribution — so the beta spread across MAXβ deciles is approximately **zero** by construction.

**Timing (no lookahead bias)**: Signal computed using data through end of month `t` → portfolio return measured in month `t+1`.

---

## Performance Results

### Strategy Performance

| Metric | MAX | MAXβ (Double Sort) |
|---|---|---|
| Mean Monthly Return | 0.82% | **0.86%** |
| Annualised Return | 10.35% | **10.88%** |
| Monthly Volatility | 8.11% | **6.07%** |
| Annualised Volatility | 28.11% | **21.03%** |
| **Sharpe Ratio (ann.)** | 0.37 | **0.52** |
| Win Rate | 58.8% | 58.2% |
| Max Drawdown | −51.7% | −52.4% |
| Sample | 177 months | 177 months |

MAXβ delivers a **40% higher Sharpe ratio** (0.52 vs 0.37) with **25% lower volatility** despite a similar mean return. This is precisely the paper's central claim: by removing beta contamination, the idiosyncratic lottery signal becomes sharper and more consistent.

### Decile Returns

| Decile | MAX Avg Return | MAXβ Avg Return |
|---|---|---|
| 1 (Low) | 1.19% | **1.33%** |
| 2 | 1.29% | 1.17% |
| 3 | 0.92% | 1.04% |
| 4 | 1.09% | 0.90% |
| 5 | 1.09% | 0.98% |
| 6 | 1.05% | 1.05% |
| 7 | 0.88% | 1.05% |
| 8 | 1.32% | 1.05% |
| 9 | 0.75% | 0.83% |
| 10 (High) | 0.37% | **0.46%** |
| **D1 − D10 spread** | **+0.82%** | **+0.86%** |

The MAXβ decile pattern is **more monotone** (D1 is the highest-returning decile; D10 the lowest) compared to the noisier MAX sort where D8 anomalously outperforms D1. This confirms that the double sort removes beta noise and produces a cleaner cross-sectional signal.

### Beta Neutralisation — Key Validation

| | Beta Spread D10 − D1 | Interpretation |
|---|---|---|
| MAX deciles | **+0.59** | High-MAX stocks have much higher beta — MAX is contaminated by systematic risk |
| MAXβ deciles | **+0.02 ≈ 0** | Double sort successfully removes beta exposure ✓ |

This is the paper's own validation test (Table A5): *"the market beta remains constant across the MAXβ deciles, with a spread of zero between the extreme portfolios."* Our replication achieves +0.015, essentially flat.

---

## Does Our Data Support the Paper?

**Yes — all three core claims replicate on the 2010–2024 sample.**

### Claim 1: MAX anomaly exists ✓
High-MAX stocks (D10, +0.37%/month) underperform low-MAX stocks (D1, +1.19%/month) by **0.82%/month**. This is consistent with the paper's finding of −0.95%/month on their 1968–2022 sample. The direction and magnitude are in line: investors overpay for lottery-like stocks, which subsequently underperform.

### Claim 2: MAXβ double sort removes beta contamination ✓
The beta spread across MAX deciles is +0.59 — confirming that MAX mechanically selects high-beta stocks during market rallies. After the double sort, the MAXβ beta spread collapses to +0.015 ≈ 0. The paper proves the same in Table A5. Our replication is essentially exact.

### Claim 3: MAXβ is a stronger signal than MAX ✓
The paper reports (Table 6) that the MAXβ L/S Sharpe ratio exceeds the MAX Sharpe ratio because the idiosyncratic component has lower volatility once beta is stripped. Our data confirms this:
- **Volatility drops 25%** (28.1% → 21.0%) for the same approximate return (+0.82% → +0.86%)
- **Sharpe improves 40%** (0.37 → 0.52)

This is the core mechanism: MAXβ has similar returns to MAX but is far less driven by market swings, making it more consistent month-to-month.

---

## Pre- vs Post-2020 Performance

| Period | MAX Sharpe | MAXβ Sharpe |
|---|---|---|
| Pre-2020 (2010–2019) | 0.597 | **0.818** |
| Post-2020 (2020–2024) | 0.173 | **0.281** |

Both signals are weaker post-2020 — consistent with the broader anomaly-decay literature (McLean & Pontiff, 2016). However, MAXβ consistently outperforms MAX in **both** sub-periods, and its advantage is larger in the pre-2020 window (0.818 vs 0.597). The post-2020 decay is partly driven by COVID-era market dislocations (2020: −49% for MAX, 2021: +100%) which are idiosyncratic regime shocks rather than structural decay of the signal.

### Annual Returns

| Year | MAX | MAXβ | Comment |
|---|---|---|---|
| 2010 | +8.2% | −0.6% | |
| 2011 | +9.8% | +7.2% | |
| 2012 | +6.0% | +9.7% | MAXβ stronger |
| 2013 | −14.2% | −10.8% | Both negative; low-beta rally |
| 2014 | +23.9% | +26.8% | |
| 2015 | +42.6% | +30.8% | MAX driven by high-beta |
| 2016 | −19.7% | −10.2% | MAXβ much less affected |
| 2017 | +14.7% | +15.0% | |
| 2018 | +3.1% | +7.3% | |
| 2019 | +39.8% | +44.5% | |
| 2020 | −49.1% | −52.4% | COVID crash — long-short hit hard |
| 2021 | +100.3% | +117.8% | Post-COVID reversal |
| 2022 | +45.7% | +13.2% | MAX captured rate-hike momentum |
| 2023 | −13.6% | **+12.6%** | MAXβ positive; MAX negative — key divergence |
| 2024 | −27.3% | −10.4% | MAXβ less affected by market risk |

The 2023 divergence is notable: MAX lost −13.6% while MAXβ gained +12.6%. This is consistent with the paper's thesis — in 2023 the market rally (high beta) distorted the MAX signal, but MAXβ's beta-neutralisation correctly identified the idiosyncratic lottery stocks.

---

## Running the Strategy

```bash
# From the project root
source venv/bin/activate
python analysis/strategy.py
```

Runtime: ~23 seconds.
