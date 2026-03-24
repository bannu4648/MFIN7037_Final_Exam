# Speaker Notes — Person 1 (2–3 minutes)

---

## Slide 1: Title (15 seconds)

Hi everyone. Our project replicates and extends a recent paper called "MAX on Steroids" by Bali, Ince, and Ozsoylev. The core question we're investigating is: do lottery-like stocks — stocks that show extreme positive daily returns — still generate abnormal returns today, and can we actually build a profitable trading strategy around that?

---

## Slide 2: Why Lottery Stocks Matter (45–60 seconds)

So first, a quick overview of what this paper is about.

There's a well-known behavioral bias where investors overweight small-probability, large-payoff events — basically, people love lottery tickets. In equity markets, this shows up as speculative demand for stocks that have had extreme daily price spikes.

The original MAX measure, introduced by Bali, Cakici, and Whitelaw in 2011, captures this by simply averaging the top 5 highest daily returns a stock has in a given month. The finding is that high-MAX stocks tend to be overpriced and subsequently underperform — investors are essentially paying a premium for lottery-like payoffs.

The innovation of *this* paper is MAXβ — a beta-neutralized version of MAX. The idea is that some of those extreme daily returns are just driven by market-wide movements, not firm-specific events. By double-sorting on beta first and then MAX, MAXβ strips out the systematic component and isolates the truly idiosyncratic lottery features. The result is a much stronger and more robust anomaly that survives all major factor models.

Our project replicates this on recent data from 2010 to 2024, builds the long-short strategy, and tests whether the anomaly still holds up in practice.

---

## Slide 3: Data Sources & Pipeline (30–45 seconds)

Now let me walk you through the data infrastructure I built.

We use CRSP data — that's the Center for Research in Security Prices — which is the same institutional-grade source used in the original paper. We also pull in the Fama-French 5 factors from Kenneth French's data library.

The pipeline has five stages. First, we ingest the raw CRSP daily and monthly stock files using PyArrow — the daily file alone is nearly 4 gigabytes with over 100 million rows. We then filter down to common US equities on the three major exchanges — NYSE, AMEX, and NASDAQ — over 2010 to 2024. Next comes the cleaning: removing penny stocks, dropping tickers with sparse data, and forward-filling small gaps. We then merge the Fama-French factors onto the daily data by trading date — this is critical because my teammates need the market return for the beta regression. Finally, we validate: zero duplicates, zero missing returns in the final output.

The result is 16.4 million daily observations across roughly 8,800 stocks and 786,000 monthly observations.

---

## Slide 4: Data Cleaning & Deliverables (30–45 seconds)

On the cleaning side — on the left you can see the filters we applied. We removed about 860,000 penny stock observations, excluded any stock with more than 20% missing daily returns or less than a year of data, and filtered out non-equity securities and OTC stocks at the source level.

On the right, the quality assurance checks. We use CRSP's own pre-computed returns which are already adjusted for stock splits and dividends — this is the academic standard. We confirmed zero duplicate records, zero NaN returns in the final output, and stable stock coverage of about 4,400 to 5,800 stocks per year across the 15-year period.

The output is three parquet files. The daily dataset has returns plus all the Fama-French factors merged in — that's what Person 2 uses for computing MAX, estimating betas, and constructing MAXβ. The monthly dataset has returns and market cap for portfolio weighting. And the factors file is used by Person 3 for the alpha regressions.

That's the data foundation — I'll hand it over to [Person 2] to walk through the signal construction and strategy.
