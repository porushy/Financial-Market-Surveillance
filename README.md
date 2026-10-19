# USD–INR Market-Surveillance Tool

An unsupervised machine-learning tool that watches the daily **USD–INR exchange rate** and flags days when the market behaved abnormally (unusually large moves or a turbulent backdrop). The flagged days are then **backtested** against known historical stress episodes, and summarised in a **Market Stress Index** chart intended for a policy paper.

The whole pipeline lives in one notebook, [`notebooks/usd_inr_surveillance.ipynb`](notebooks/usd_inr_surveillance.ipynb), organised into six stages (0–5). Every stage explains its terms in plain English and prints evidence for each claim it makes.

![Market Stress Index, full history](outputs/market_stress_index_full_history.png)

---

## Key results

Computed from the frozen dataset (5,855 daily rows, **2003-12-01 to 2026-06-30**):

| Metric | Value |
|---|---|
| Days assessed by the detector | 5,604 (251 excluded for missing features) |
| Days flagged as anomalies | **113** (~2% of assessed days) |
| Flagged-day clusters | 28 (flags less than 21 days apart are grouped) |
| Flags inside established crises (2008–09 GFC, 2013 taper tantrum, 2020 COVID) | **84 / 113 (74.3%)** |
| Flags inside proposed stress eras (MEDIUM, needs verification) | 13 / 113 (11.5%) |
| Flags left unexplained (LOW) | 16 / 113 (14.2%) |
| Stress Index calm baseline (median) | 0.3384% daily volatility |
| Stress Index peak | 1.7211% on **2013-09-16** (taper tantrum), **5.09×** the baseline |

---

## Pipeline

### Stage 0: Getting the data
- Source: Yahoo Finance via `yfinance`, ticker `USDINR=X`, `period="max"`, `interval="1d"`.
- **Reproducibility switch:** `REFRESH_DATA = False` (the default) loads the frozen file `data/usd_inr_raw.csv`, so every run gives identical results. Set it to `True` to download fresh data and re-freeze the file. If the file is missing, the notebook downloads it automatically.
- Integrity checks: missing values, duplicate dates, non-positive prices, oversized jumps, and OHLC consistency (Close must lie within [Low, High]). The saved file is then reloaded and checked against the download.

### Stage 1: Measuring "normal": returns and rolling volatility
- **Log returns:** `ln(Close_t / Close_{t-1})`.
- **Gap handling:** a return that spans more than 4 calendar days is set to `NaN` (2 gaps were found: 2008-07-31 → 2008-08-26 and 2025-04-17 → 2025-04-22).
- **Bad-tick handling:** when the Close lies more than 0.5% outside the day's [Low, High] range, the returns into and out of that day are set to `NaN` (10 bad days, 20 returns). Prices are never replaced or imputed.
- **Rolling volatility:** 21-trading-day standard deviation of log returns (`roll_vol_21`). This is a tunable setting.
- Charts: exchange rate, daily log returns, rolling volatility.

### Stage 2: The detector (Isolation Forest)
- Features: `abs_log_return` (today's move) and `roll_vol_21` (recent turbulence).
- Rows with a missing feature are **excluded, not filled in**.
- `IsolationForest(n_estimators=200, contamination=0.02, random_state=42)`.
- Each day gets an `anomaly_score` (higher = more anomalous), a nullable `anomaly_flag`, and a three-way `status` (`not_assessed` / `normal` / `anomaly`). This keeps excluded days separate from days the model cleared.

### Stage 3: Backtesting against real stress episodes
- Flagged days are grouped into clusters (a gap of more than 21 days starts a new cluster).
- Each cluster gets a confidence tier from transparent date-window rules:
  - **HIGH:** overlaps an established crisis period (GFC Sep 2008–Mar 2009, taper tantrum May–Oct 2013, COVID Feb–May 2020).
  - **MEDIUM:** overlaps a proposed stress era (2011–12 eurozone / rupee-depreciation era, late-2018 EM sell-off), marked `[VERIFY]`.
  - **LOW-UNEXPLAINED:** no known episode. These clusters are listed openly rather than explained away.

### Stage 4: Market Stress Index
- `stress_index = roll_vol_21 × 100` (daily-% volatility). It deliberately stays in real units instead of being rescaled to 0–100.
- The full-history chart shows the median baseline, the flagged days, and shaded HIGH-tier crisis periods. Two zoomed companion charts cover 2013 and 2020.
- Verification: the peak falls in 2013, the peak-to-baseline ratio is about 5×, and the rescaling preserves the shape (correlation 1.0 with raw volatility).

### Stage 5: Saving outputs
Saves figures at 300 DPI plus CSV tables and a summary file to `outputs/`, then reloads each file from disk to confirm it was written correctly.

---

## Project structure

```
Financial_Markets/
├── notebooks/
│   └── usd_inr_surveillance.ipynb      # full pipeline, Stages 0–5
├── data/
│   └── usd_inr_raw.csv                 # frozen raw OHLC data (yfinance, USDINR=X)
├── outputs/
│   ├── market_stress_index_full_history.png
│   ├── market_stress_index_zoom_2013_taper_tantrum.png
│   ├── market_stress_index_zoom_2020_covid.png
│   ├── feature_table_full.csv          # per-day features, scores, flags, status, stress index
│   ├── flagged_anomaly_days.csv        # 113 flagged days, most anomalous first, with cluster + tier
│   ├── cluster_summary.csv             # 28 clusters with candidate episode + confidence tier
│   ├── results_summary.md              # headline numbers, generated from live variables
│   ├── report_findings.md
│   ├── financial_market_final_report.pdf
│   └── images/                         # additional charts (exchange rate, returns, volatility, zooms)
├── stage1_prompt.txt … stage5_prompt.txt, master_prompt_final.txt, findings_extraction_prompt.txt
└── progress.md                          # build log
```

### Output column reference (`feature_table_full.csv`)
| Column | Meaning |
|---|---|
| `close` | Daily closing USD–INR rate |
| `log_return` | Daily log return (`NaN` for the first row, gap-spanning returns and bad-tick returns) |
| `gap_spanned` / `bad_tick` | Data-quality flags explaining why a return was blanked |
| `roll_vol_21` | 21-day rolling standard deviation of log returns |
| `abs_log_return` | Absolute daily move (detector feature) |
| `anomaly_score` | Isolation Forest score; higher means more anomalous |
| `anomaly_flag` | 1 = flagged, 0 = assessed and normal, blank = not assessed |
| `status` | `not_assessed` / `normal` / `anomaly` |
| `stress_index` | Market Stress Index, `roll_vol_21 × 100` (daily %) |

---

## Getting started

**Requirements:** Python 3.10+ with:

```bash
pip install pandas numpy matplotlib scikit-learn yfinance jupyter
```

**Run:**

```bash
cd notebooks
jupyter notebook usd_inr_surveillance.ipynb
```

Run all cells from top to bottom. The notebook uses relative paths (`../data`, `../outputs`), so launch it from inside `notebooks/`. With `REFRESH_DATA = False` the run is fully offline and reproduces the results above exactly.

### Tunable parameters
| Parameter | Default | Stage | Effect |
|---|---|---|---|
| `REFRESH_DATA` | `False` | 0 | Re-download and re-freeze the raw data |
| `GAP_THRESHOLD_DAYS` | 4 | 1 | Calendar-day gap treated as missing data |
| `BAD_TICK_THRESHOLD_PCT` | 0.5 | 1 | How far (%) the Close can sit outside [Low, High] before it counts as a bad tick |
| `ROLLING_WINDOW` | 21 | 1 | Volatility look-back (trading days) |
| `N_ESTIMATORS` | 200 | 2 | Number of Isolation Forest trees |
| `CONTAMINATION` | 0.02 | 2 | Expected fraction of anomalous days |
| `RANDOM_SEED` | 42 | 2 | Seed for reproducible results |
| `CLUSTER_GAP_DAYS` | 21 | 3 | Maximum gap between flags in the same cluster |

---

## Principles and limitations
- **No invented data:** missing or corrupted values are excluded, never imputed. Every claim is backed by printed output.
- **Honest labelling:** only broad, well-established crisis *periods* are named as fact. Specific events and MEDIUM-tier eras are marked as needing verification against primary sources.
- **Single data source:** an attempt to cross-check against stooq failed because the source was unreachable, so bad ticks in the yfinance data were neutralised rather than corrected. Yahoo reports no volume for FX pairs.
- **Future work:** a GARCH volatility model is an optional refinement and is not yet implemented.
