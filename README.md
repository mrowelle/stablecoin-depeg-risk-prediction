# Stablecoin Depeg Prediction at 5-Minute Resolution

CMU MSBA Capstone Project — predicting stablecoin depeg events using high-frequency
on-chain and market data.

## Research Question

Can we predict stablecoin depeg events 5 minutes to 4 hours in advance using a combination
of on-chain mint/burn flows, DeFi liquidity signals, and macro market context — all at
5-minute resolution?

---

## Stablecoins

| Coin | Type | Status | Coverage |
|------|------|--------|----------|
| USDT (Tether) | Fiat-backed | Active | Aug 2017 → Feb 2026 |
| USDC (Circle) | Fiat-backed | Active | Oct 2018 → Feb 2026 |
| DAI (MakerDAO) | Crypto-collateralized | Active | Dec 2017 → Feb 2026 |
| BUSD (Paxos) | Fiat-backed | Discontinued Mar 2023 | Sep 2019 → Mar 2023 |
| UST (Terra) | Algorithmic | Failed May 2022 | Sep 2020 → May 2022 |
| USDe (Ethena) | Synthetic | Active | Apr 2024 → Feb 2026 |
| RLUSD (Ripple) | Fiat-backed | Active | Apr 2025 → Feb 2026 |

Failed and discontinued coins are included intentionally — they provide real depeg examples
for model training.

---

## Depeg Definition

A depeg episode is **3 or more consecutive 5-minute bars** where `|price − $1.00| > 0.5%`.
The 15-minute persistence requirement filters transient tick noise.

---

## Feature Sources

| Source | Signal | Frequency |
|--------|--------|-----------|
| CoinAPI VWAP | Stablecoin price (primary target) | 5m |
| Binance | BTC/ETH market context + USDT cross-pairs | 5m |
| Etherscan V2 | Ethereum mint/burn events + USDT treasury flows | 5m (event-level) |
| TronGrid | USDT TRON treasury flows | 5m (event-level) |
| Dune Analytics | Solana USDC mint/burn flows | 5m (event-level) |
| XRPL public RPC | RLUSD mint/burn + DEX flows | 5m (event-level) |
| Curve Finance | 3pool, USDe/USDC, RLUSD/USDC swap flows | 5m (event-level) |
| FRED | DXY, VIX, T10Y, Fed Funds rate | Daily → 5m ffill |
| CoinGecko / AltIndex | BTC/ETH daily + CNN Fear & Greed Index | Daily → 5m ffill |

See `data/README.md` for full column reference and upstream pipeline documentation.

---

## Repo Structure

```
├── config/
│   └── settings.py                # Coin configs, date ranges, API settings
├── data/
│   ├── README.md                  # Full column reference + upstream pipeline docs
│   ├── raw/                       # Raw source files organized by provider
│   │   ├── binance/
│   │   ├── coinapi/
│   │   ├── curve/
│   │   ├── fred/
│   │   ├── market/
│   │   └── onchain/
│   └── processed/
│       ├── merged/                # merge_sources.py output ({coin}_5m_raw.parquet)
│       ├── cleansed/              # clean + labeled, modeling-ready ({coin}_5m.parquet)
│       ├── features/              # pooled_5m.parquet — cross-coin feature matrix
│       ├── models/                # catboost_{30min,1h}.cbm, calibrators, isolation_forest, thresholds
│       ├── predictions/           # predictions_test.parquet — LOEO test predictions
│       └── plots/                 # eda/, features/, modelling/ — analysis plots
├── notebooks/
│   ├── 03_eda_all_coins.ipynb          # Consolidated EDA across all 7 coins
│   ├── 04_feature_engineering.ipynb    # Feature construction + selection → pooled_5m.parquet
│   └── 05_modeling.ipynb               # Dual-horizon CatBoost (30min + 1h) + calibration + LOEO
├── src/
│   ├── data/                      # Upstream data pipeline (collection, merging, cleaning, labelling)
│   │   ├── collect_all.py
│   │   ├── collect_binance.py
│   │   ├── collect_coinapi.py
│   │   ├── collect_onchain.py
│   │   ├── collect_tron.py
│   │   ├── collect_solana.py
│   │   ├── collect_curve.py
│   │   ├── collect_xrpl.py
│   │   ├── collect_dune.py
│   │   ├── collect_orderbook.py
│   │   ├── collect_fred.py
│   │   ├── collect_market.py
│   │   ├── merge_sources.py
│   │   ├── clean_data.py
│   │   └── label_data.py
│   └── features/
│       └── label_depeg.py         # Alternative labeling (literature-based; not used by main pipeline)
├── scripts/                       # Utility and exploration scripts
├── .gitattributes                 # Git LFS tracks all *.parquet
└── requirements.txt
```

---

## Pipeline Overview

```
  data/raw/            collect_*.py      → raw API pulls per source
        ↓
  data/processed/      merge_sources.py  → {coin}_5m_raw.parquet (wide join, no cleaning)
  merged/              clean_data.py     → {coin}_5m.parquet (zero-fill, ffill, anomaly patch)
                       label_data.py     → adds depeg labels in-place
        ↓
  data/processed/      — modeling-ready per-coin parquets
  cleansed/
        ↓
  notebooks/03        — consolidated EDA across 7 coins + crisis case studies
        ↓
  notebooks/04        — feature engineering + selection
        ↓
  data/processed/features/  pooled_5m.parquet  — pooled cross-coin feature matrix
        ↓
  notebooks/05        — dual-horizon CatBoost (30min + 1h), calibration, isolation forest,
                        threshold optimization, LOEO evaluation, SHAP analysis
        ↓
  data/processed/models/, data/processed/predictions/, data/processed/plots/
```

---

## Setup

### 1. Install system dependencies

You need **Git LFS** to fetch the parquet files (this repo tracks all `*.parquet` via LFS):

```bash
# macOS (Homebrew)
brew install git-lfs

# Amazon Linux / RHEL (download release binary)
# see: https://github.com/git-lfs/git-lfs/releases

git lfs install
```

### 2. Clone and fetch LFS files

```bash
git clone https://github.com/mrowelle/stablecoin-depeg-prediction.git
cd stablecoin-depeg-prediction
git lfs pull
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API keys (only needed to re-run data collection)

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

| Key | Required | Source |
|-----|----------|--------|
| `COINAPI_KEY` | Yes | coinapi.io |
| `ETHERSCAN_API_KEY` | Yes | etherscan.io |
| `FRED_API_KEY` | Yes | fred.stlouisfed.org |
| `HELIUS_API_KEY` | Yes | helius.xyz (Solana) |
| `DUNE_API_KEY` | Yes | dune.com |
| `TRONGRID_API_KEY` | No | trongrid.io (free tier works without key) |
| `COINGECKO_API_KEY` | No | coingecko.com (higher rate limits) |

### 5. AWS access (only needed for S3-backed notebooks)

The analysis notebooks (03, 04, 05) can read and write to `s3://stablecoin-capstone-cmu/` for
intermediate artifacts. If you only want to run locally, all required inputs
(`data/processed/cleansed/*.parquet`) are available through Git LFS and no AWS access is needed.

---

## Running the Pipeline

### A. Upstream data collection (`src/data/`)

Only needed to refresh the underlying dataset. Typical users can skip this — the modeling-ready
parquets are already committed under `data/processed/cleansed/`.

```bash
# Collect all sources for all coins
python src/data/collect_all.py all

# Build modeling-ready dataset (run in order)
python src/data/merge_sources.py all    # join all sources → {coin}_5m_raw.parquet
python src/data/clean_data.py all       # zero-fill, ffill, patch anomalies → {coin}_5m.parquet
python src/data/label_data.py all       # add depeg labels in-place
```

Individual collectors can be run separately (`collect_binance.py`, `collect_onchain.py`,
`collect_curve.py`, `collect_xrpl.py`, `collect_dune.py`, `collect_tron.py`, `collect_solana.py`,
`collect_fred.py`, `collect_market.py`, `collect_orderbook.py`). See `data/README.md` for details.

### B. Downstream analysis / modeling (`notebooks/`)

Run the notebooks in order. Inputs come from `data/processed/cleansed/{coin}_5m.parquet` (this repo
via LFS) or from S3 if you have AWS access.

1. **`notebooks/03_eda_all_coins.ipynb`** — consolidated EDA across all 7 coins: price history,
   depeg severity distribution, cross-coin co-occurrence, crisis case studies (Terra/UST May 2022,
   USDC March 2023, USDe August 2024, etc.), per-feature AUC ranking. Plots saved to
   `data/processed/plots/eda/`.

2. **`notebooks/04_feature_engineering.ipynb`** — constructs rolling / lag / ratio features from
   the cleansed per-coin parquets, pools all 7 coins, and applies feature selection. Output:
   `data/processed/features/pooled_5m.parquet` + `selected_features.json`. Plots saved to
   `data/processed/plots/features/`.

3. **`notebooks/05_modeling.ipynb`** — trains dual-horizon CatBoost classifiers (30-minute and
   1-hour prediction windows), applies isotonic calibration, fits an isolation-forest baseline,
   optimizes per-coin thresholds, runs leave-one-entity-out (LOEO) cross-validation, and produces
   SHAP importance and beeswarm plots. Outputs: `data/processed/models/`, `data/processed/predictions/`,
   `data/processed/plots/modelling/`.

---

## Dataset Summary

| Coin | Rows | Columns | Date Range |
|------|-----:|--------:|------------|
| USDT | 897,936 | 66 | Aug 2017 → Feb 2026 |
| USDC | 775,506 | 45 | Oct 2018 → Feb 2026 |
| DAI | 828,963 | 45 | Dec 2017 → Feb 2026 |
| BUSD | 370,848 | 30 | Sep 2019 → Mar 2023 |
| UST | 153,961 | 24 | Sep 2020 → May 2022 |
| USDe | 200,928 | 40 | Apr 2024 → Feb 2026 |
| RLUSD | 96,192 | 40 | Apr 2025 → Feb 2026 |
