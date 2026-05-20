# AI Stock Market Compass (Frontman + Analyst + Reader)

A beginner-friendly, modular machine-learning project that acts like a compass for US stock-market decisions.

## Multi-Agent Overview

- **Frontman (`app.py`)**: Streamlit orchestration/UI layer that translates model outputs into plain-English guidance.
- **Analyst (`analyst_model.py`)**: Quantitative backend using a Triple Barrier labeling workflow and XGBoost for directional classification.
- **Reader (mocked in `app.py`)**: News-sentiment context provider returning a sentiment score and concise summary.

## Key Modeling Approach

The Analyst backend uses a **Triple Barrier Method** with:

- **Upper barrier**: `Close + (Volatility_EMA20 * 2)`
- **Lower barrier**: `Close - (Volatility_EMA20 * 1)`
- **Vertical barrier**: 20 trading days

Each historical observation is labeled as:

- `+1` if upper barrier is hit first
- `-1` if lower barrier is hit first
- `0` if the vertical time limit expires first

The model then trains an **XGBoost multiclass classifier** and evaluates with **TimeSeriesSplit** to reduce hindsight/lookahead bias that random splitting can introduce in financial time series.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run Locally

```bash
streamlit run app.py
```

Then open the local URL shown in your terminal.

## Repository Layout

```text
stock_indicator/
├── app.py                   # Frontman Streamlit app (UI/orchestration)
├── analyst_model.py         # Analyst quantitative model and training pipeline
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
└── .gitignore               # Common local artifacts excluded from version control
```

Planned extension point: future standalone Reader modules can be added under `agents/reader/`.

## Data & Scope

- **Current region**: US equities only
- **Market data source**: `yfinance`
- **Data mode**: EOD / delayed data pipeline, designed to remain modular for future real-time upgrades

## Risk & Disclaimer

**Educational tool only. Not guaranteed financial advice.**
