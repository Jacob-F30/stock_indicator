"""Build point-in-time market data and forecast assets for the static prototype.

The browser never calls a market-data service. This script is the only place that
downloads source data and runs models. Every feature row uses an SEC fact only
after the date it was filed, and every forecast model uses observations before
its selected prediction date.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import yfinance as yf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIRECTORY = PROJECT_ROOT / "public" / "data"
RAW_DIRECTORY = PROJECT_ROOT / "data" / "raw"
ANALYSIS_DIRECTORY = PROJECT_ROOT / "data" / "analysis"
HORIZONS = {1: 21, 3: 63, 6: 126, 12: 252, 24: 504}
STOCKS = {
    "AAPL": ("Apple Inc.", "Technology"), "MSFT": ("Microsoft", "Technology"),
    "NVDA": ("NVIDIA", "Semiconductors"), "AMZN": ("Amazon", "Consumer Cyclical"),
    "GOOGL": ("Alphabet", "Communication Services"), "META": ("Meta Platforms", "Communication Services"),
    "TSLA": ("Tesla", "Consumer Cyclical"), "BRK-B": ("Berkshire Hathaway", "Financial Services"),
    "JPM": ("JPMorgan Chase", "Financial Services"), "JNJ": ("Johnson & Johnson", "Healthcare"),
}
FEATURE_COLUMNS = [
    "return_5d", "return_20d", "return_60d", "ma20_distance", "ma60_distance", "volatility_20d", "volume_change_20d",
    "operating_margin", "roe", "roic", "free_cash_flow", "cash_to_market_cap", "net_debt", "interest_coverage",
]


@dataclass(frozen=True)
class Company:
    symbol: str
    name: str
    sector: str
    cik: str


def request_json(url: str, user_agent: str) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
    with urlopen(request, timeout=30) as response:  # nosec B310: documented SEC endpoint
        return json.loads(response.read().decode("utf-8"))


def get_companies(user_agent: str) -> dict[str, Company]:
    cache_path = RAW_DIRECTORY / "company_tickers.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        tickers = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        tickers = request_json("https://www.sec.gov/files/company_tickers.json", user_agent)
        cache_path.write_text(json.dumps(tickers), encoding="utf-8")
    records_by_symbol = {record["ticker"]: record for record in tickers.values()}
    return {symbol: Company(symbol, name, sector, f"{records_by_symbol[symbol]['cik_str']:010d}") for symbol, (name, sector) in STOCKS.items()}


def download_prices(symbol: str) -> pd.DataFrame:
    prices = yf.download(symbol, period="max", interval="1d", auto_adjust=True, progress=False)
    if prices.empty:
        raise ValueError(f"No market data returned for {symbol}.")
    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = prices.columns.get_level_values(0)
    required_columns = ["Open", "High", "Low", "Close", "Volume"]
    missing = set(required_columns).difference(prices.columns)
    if missing:
        raise ValueError(f"{symbol} is missing price columns: {sorted(missing)}")
    prices = prices[required_columns].dropna().copy()
    if prices.empty:
        raise ValueError(f"No complete market-data rows returned for {symbol}.")
    prices.index = pd.to_datetime(prices.index).tz_localize(None).normalize()
    prices.index.name = "date"
    return prices


def price_records(prices: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            "date": index.date().isoformat(),
            "close": round(float(row["Close"]), 2),
            "volume": int(row["Volume"]),
        }
        for index, row in prices.iterrows()
    ]


def export_prices_only(symbol: str) -> dict[str, Any]:
    """Export the full available adjusted daily history for one ticker."""
    print(f"Downloading {symbol} price history...")
    records = price_records(download_prices(symbol))
    stock_directory = OUTPUT_DIRECTORY / symbol
    stock_directory.mkdir(parents=True, exist_ok=True)
    (stock_directory / "prices.json").write_text(json.dumps(records), encoding="utf-8")
    return {
        "symbol": symbol,
        "historyStart": records[0]["date"],
        "historyEnd": records[-1]["date"],
        "recordCount": len(records),
    }


def fact_series(company_facts: dict[str, Any], tags: list[str], unit: str = "USD") -> pd.Series:
    facts = company_facts.get("facts", {}).get("us-gaap", {})
    observations: list[tuple[pd.Timestamp, float]] = []
    for tag in tags:
        for entry in facts.get(tag, {}).get("units", {}).get(unit, []):
            if entry.get("filed") and entry.get("val") is not None and entry.get("form") in {"10-K", "10-Q"}:
                observations.append((pd.Timestamp(entry["filed"]).normalize(), float(entry["val"])))
        if observations:
            break
    if not observations:
        return pd.Series(dtype="float64")
    values = pd.DataFrame(observations, columns=["filed", "value"])
    return values.groupby("filed")["value"].last().sort_index()


def point_in_time_fundamentals(prices: pd.DataFrame, company_facts: dict[str, Any]) -> pd.DataFrame:
    specifications = {
        "revenue": (["RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet", "Revenues"], "USD"),
        "operating_income": (["OperatingIncomeLoss"], "USD"), "net_income": (["NetIncomeLoss"], "USD"),
        "operating_cash_flow": (["NetCashProvidedByUsedInOperatingActivities"], "USD"), "capex": (["PaymentsToAcquirePropertyPlantAndEquipment"], "USD"),
        "cash": (["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"], "USD"),
        "debt": (["LongTermDebtAndCurrentMaturities", "LongTermDebtCurrent", "LongTermDebtNoncurrent", "LongTermDebt"], "USD"),
        "equity": (["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"], "USD"),
        "interest_expense": (["InterestExpenseNonoperating", "InterestExpenseDebt"], "USD"),
        "shares": (["EntityCommonStockSharesOutstanding"], "shares"),
    }
    data = pd.DataFrame(index=prices.index)
    for name, (tags, unit) in specifications.items():
        data[name] = fact_series(company_facts, tags, unit).reindex(data.index, method="ffill")
    market_cap = prices["Close"] * data["shares"]
    data["operating_margin"] = data["operating_income"] / data["revenue"]
    data["roe"] = data["net_income"] / data["equity"]
    data["free_cash_flow"] = data["operating_cash_flow"] - data["capex"]
    data["cash_to_market_cap"] = data["cash"] / market_cap
    data["net_debt"] = data["debt"] - data["cash"]
    data["interest_coverage"] = data["operating_income"] / data["interest_expense"].replace(0, np.nan)
    invested_capital = data["equity"] + data["debt"] - data["cash"]
    data["roic"] = data["operating_income"] * 0.79 / invested_capital.replace(0, np.nan)
    return data.replace([np.inf, -np.inf], np.nan)


def build_features(prices: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    features = fundamentals.copy()
    close = prices["Close"]
    features["return_5d"] = close.pct_change(5)
    features["return_20d"] = close.pct_change(20)
    features["return_60d"] = close.pct_change(60)
    features["ma20_distance"] = close / close.rolling(20).mean() - 1
    features["ma60_distance"] = close / close.rolling(60).mean() - 1
    features["volatility_20d"] = close.pct_change().rolling(20).std() * np.sqrt(252)
    features["volume_change_20d"] = prices["Volume"] / prices["Volume"].rolling(20).mean() - 1
    return features


def train_quantile_models(features: pd.DataFrame, prices: pd.DataFrame, horizon_days: int, train_until: pd.Timestamp) -> tuple[dict[str, Any], dict[str, float]]:
    from sklearn.ensemble import HistGradientBoostingRegressor

    target = prices["Close"].shift(-horizon_days) / prices["Close"] - 1
    rows = features.assign(target=target).loc[:train_until].dropna(subset=FEATURE_COLUMNS + ["target"])
    if len(rows) < 160:
        raise ValueError(f"Only {len(rows)} usable training rows for {horizon_days} days; need at least 160.")
    models = {name: HistGradientBoostingRegressor(loss="quantile", quantile=quantile, max_iter=150, max_leaf_nodes=15, learning_rate=0.06, l2_regularization=0.4, random_state=42) for name, quantile in {"low": 0.1, "median": 0.5, "high": 0.9}.items()}
    for model in models.values():
        model.fit(rows[FEATURE_COLUMNS], rows["target"])
    return models, {"trainingRows": float(len(rows))}


def build_forecast(models: dict[str, Any], feature_row: pd.DataFrame, price: float, symbol: str, as_of_date: pd.Timestamp, horizon_months: int) -> dict[str, Any]:
    low, median, high = sorted(float(models[name].predict(feature_row[FEATURE_COLUMNS])[0]) for name in ("low", "median", "high"))
    downside = round(max(8, min(82, 32 - median * 130 + (high - low) * 65)))
    upside = round(max(8, min(82, 48 + median * 130 - (high - low) * 30)))
    stable = max(0, 100 - downside - upside)
    risk = round(max(1, min(99, (high - low) * 220 + downside * 0.6)))
    signal = "buy" if upside >= downside + 12 and median > 0.025 else "sell" if downside >= upside + 8 or median < -0.025 else "watch"
    return {"symbol": symbol, "asOfDate": as_of_date.date().isoformat(), "horizonMonths": horizon_months, "selectedPrice": round(price, 2), "predictedPrice": round(price * (1 + median), 2), "interval": {"low": round(price * (1 + low), 2), "high": round(price * (1 + high), 2), "confidence": 0.8}, "expectedReturn": round(median, 6), "probabilities": {"buy": upside, "watch": stable, "sell": downside}, "riskScore": risk, "recommendation": signal}


def export_stock(company: Company, user_agent: str, forecast_stride: int) -> dict[str, Any]:
    from sklearn.metrics import mean_absolute_error

    print(f"Downloading {company.symbol}...")
    prices = download_prices(company.symbol)
    company_facts = request_json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{company.cik}.json", user_agent)
    fundamentals = point_in_time_fundamentals(prices, company_facts)
    features = build_features(prices, fundamentals)
    ANALYSIS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    prices.join(features).to_parquet(ANALYSIS_DIRECTORY / f"{company.symbol}.parquet")
    holdout_start = prices.index.max() - pd.DateOffset(months=3)
    forecasts: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {}
    for months, days in HORIZONS.items():
        try:
            models, details = train_quantile_models(features, prices, days, holdout_start)
        except ValueError as error:
            metrics[str(months)] = {"status": "unavailable", "reason": str(error)}
            continue
        test_target = prices["Close"].shift(-days) / prices["Close"] - 1
        test_rows = features.assign(target=test_target).loc[holdout_start:].dropna(subset=FEATURE_COLUMNS + ["target"])
        if not test_rows.empty:
            predictions = models["median"].predict(test_rows[FEATURE_COLUMNS])
            details["holdoutMae"] = round(float(mean_absolute_error(test_rows["target"], predictions)), 6)
            details["holdoutRows"] = int(len(test_rows))
        metrics[str(months)] = {"status": "ready", **details}
        candidate_positions = [
            position
            for position in range(days, len(features), forecast_stride)
            if not features.iloc[position][FEATURE_COLUMNS].isna().any()
        ]
        for position in candidate_positions:
            as_of_date = features.index[position]
            training_cutoff = features.index[position - days]
            try:
                point_in_time_models, _ = train_quantile_models(features, prices, days, training_cutoff)
            except ValueError:
                continue
            feature_row = features.iloc[[position]]
            forecasts.append(build_forecast(point_in_time_models, feature_row, float(prices.at[as_of_date, "Close"]), company.symbol, as_of_date, months))
    stock_directory = OUTPUT_DIRECTORY / company.symbol
    stock_directory.mkdir(parents=True, exist_ok=True)
    exported_prices = price_records(prices)
    fundamental_columns = ["operating_margin", "roe", "roic", "free_cash_flow", "cash_to_market_cap", "net_debt", "interest_coverage"]
    fundamental_records = [{"date": index.date().isoformat(), **{column: None if pd.isna(row[column]) else round(float(row[column]), 6) for column in fundamental_columns}} for index, row in fundamentals.iterrows()]
    (stock_directory / "prices.json").write_text(json.dumps(exported_prices), encoding="utf-8")
    (stock_directory / "fundamentals.json").write_text(json.dumps(fundamental_records), encoding="utf-8")
    (stock_directory / "forecasts.json").write_text(json.dumps(forecasts), encoding="utf-8")
    return {"symbol": company.symbol, "name": company.name, "sector": company.sector, "cik": company.cik, "historyStart": exported_prices[0]["date"], "historyEnd": exported_prices[-1]["date"], "metrics": metrics}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build static market and forecast assets.")
    parser.add_argument("--symbols", nargs="+", choices=sorted(STOCKS), help="Subset of prototype tickers to generate.")
    parser.add_argument("--forecast-stride", type=int, default=21, help="Trading-day gap between exported forecast points.")
    parser.add_argument("--prices-only", action="store_true", help="Export full real price history without SEC facts or model forecasts.")
    parser.add_argument("--sec-user-agent", default=os.environ.get("SEC_USER_AGENT"), help="Required SEC-compliant identifier, for example 'Market Compass contact@example.com'.")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if arguments.forecast_stride < 1:
        raise SystemExit("--forecast-stride must be at least 1.")
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    selected_symbols = arguments.symbols or list(STOCKS)
    if arguments.prices_only:
        manifest: dict[str, Any] = {
            "generatedAt": pd.Timestamp.now(tz="UTC").isoformat(),
            "mode": "prices-only",
            "source": "Yahoo Finance adjusted daily prices",
            "stocks": [],
        }
        for symbol in selected_symbols:
            manifest["stocks"].append(export_prices_only(symbol))
            time.sleep(0.15)
    else:
        if not arguments.sec_user_agent:
            raise SystemExit("Set --sec-user-agent or SEC_USER_AGENT before downloading SEC data.")
        companies = get_companies(arguments.sec_user_agent)
        manifest = {"generatedAt": pd.Timestamp.now(tz="UTC").isoformat(), "source": "Yahoo Finance prices and SEC Companyfacts", "forecastStrideTradingDays": arguments.forecast_stride, "stocks": []}
        for symbol in selected_symbols:
            manifest["stocks"].append(export_stock(companies[symbol], arguments.sec_user_agent, arguments.forecast_stride))
            time.sleep(0.15)
    (OUTPUT_DIRECTORY / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote static assets to {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()