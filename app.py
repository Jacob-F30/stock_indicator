from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from analyst_model import train_analyst_model

DISCLAIMER_TEXT = "Educational tool only. Not guaranteed financial advice."

INVESTMENT_STYLES: Dict[str, Dict[str, str]] = {
    "Long-term & Safe": {
        "AAPL": "Apple",
        "MSFT": "Microsoft",
        "JNJ": "Johnson & Johnson",
        "PG": "Procter & Gamble",
        "KO": "Coca-Cola",
    },
    "Short-term & Aggressive": {
        "NVDA": "NVIDIA",
        "TSLA": "Tesla",
        "PLTR": "Palantir",
        "SOFI": "SoFi",
        "COIN": "Coinbase",
    },
}


def render_disclaimer_banner() -> None:
    st.markdown(
        f"""
        <div style="background:#B00020;color:white;padding:12px 16px;border-radius:8px;margin:8px 0;
                    font-weight:600;text-align:center;letter-spacing:0.2px;">
            {DISCLAIMER_TEXT}
        </div>
        """,
        unsafe_allow_html=True,
    )


def mock_reader_sentiment(ticker: str) -> Tuple[float, str]:
    seed = sum(ord(ch) for ch in ticker.upper())
    score = float(np.tanh((seed % 17 - 8) / 7))

    if score >= 0.35:
        summary = f"Recent coverage around {ticker.upper()} appears mostly optimistic, with demand and execution trends viewed favorably."
    elif score <= -0.35:
        summary = f"Recent coverage around {ticker.upper()} is cautious, highlighting uncertainty and downside risk in near-term momentum."
    else:
        summary = f"Recent coverage around {ticker.upper()} is mixed, with balanced bullish and bearish viewpoints."
    return score, summary


def _extract_fundamentals(info: dict) -> dict[str, float]:
    pe_ratio = info.get("trailingPE") or info.get("forwardPE")
    roic = info.get("returnOnInvestedCapital")
    roi = info.get("returnOnAssets") or info.get("returnOnEquity")
    market_cap = info.get("marketCap")

    return {
        "PE_Ratio": float(pe_ratio) if pe_ratio is not None else np.nan,
        "ROIC": float(roic) if roic is not None else np.nan,
        "ROI": float(roi) if roi is not None else np.nan,
        "Market_Cap": float(market_cap) if market_cap is not None else np.nan,
    }


def fetch_market_data(ticker: str) -> pd.DataFrame:
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=730)

    yf_ticker = yf.Ticker(ticker)
    hist = yf_ticker.history(start=start_date.date(), end=end_date.date(), auto_adjust=False)
    if hist.empty:
        raise ValueError(f"No market history found for ticker '{ticker.upper()}'")

    info = yf_ticker.info or {}
    fundamentals = _extract_fundamentals(info)

    history = hist.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]]
    for col, value in fundamentals.items():
        history[col] = value

    return history


def _compose_directive(analyst_buy_prob: float, analyst_sell_prob: float, sentiment: float) -> str:
    combined_buy = (analyst_buy_prob * 0.7) + (max(sentiment, 0) * 0.3)
    combined_sell = (analyst_sell_prob * 0.7) + (max(-sentiment, 0) * 0.3)

    if combined_buy >= 0.55 and combined_buy > combined_sell:
        return "Action Required: BUY NOW"
    if combined_sell >= 0.55 and combined_sell > combined_buy:
        return "Action Required: SELL/PROTECT CAPITAL"
    return "Action Required: WAIT & OBSERVE"


def _plain_english_reasoning(
    directive: str,
    probs: dict[str, float],
    sentiment: float,
    summary: str,
    latest_row: pd.Series,
) -> str:
    pe = latest_row.get("PE_Ratio", np.nan)
    roic = latest_row.get("ROIC", np.nan)
    market_cap = latest_row.get("Market_Cap", np.nan)

    sentiment_label = "positive" if sentiment > 0.2 else "negative" if sentiment < -0.2 else "mixed"

    fundamentals_line = (
        f"PE ratio is {pe:.2f}, ROIC is {roic:.2%}, and market cap is ${market_cap:,.0f}."
        if np.isfinite(pe) and np.isfinite(roic) and np.isfinite(market_cap)
        else "Some fundamental metrics are unavailable, so the guidance leans more on price trend behavior."
    )

    if directive.endswith("BUY NOW"):
        return (
            f"The model's buy probability ({probs['+1']:.1%}) is stronger than hold/sell, and news tone is {sentiment_label}. "
            f"{fundamentals_line} {summary}"
        )
    if directive.endswith("SELL/PROTECT CAPITAL"):
        return (
            f"The model's sell probability ({probs['-1']:.1%}) is elevated and sentiment is {sentiment_label}. "
            f"{fundamentals_line} {summary}"
        )
    return (
        f"Signals are not aligned enough for an immediate trade (Buy {probs['+1']:.1%}, Hold {probs['0']:.1%}, Sell {probs['-1']:.1%}). "
        f"Sentiment is {sentiment_label}. {fundamentals_line} {summary}"
    )


def main() -> None:
    st.set_page_config(page_title="AI Stock Compass", page_icon="🧭", layout="wide")
    render_disclaimer_banner()

    st.title("🧭 AI Stock Market Compass")
    st.caption("Beginner-friendly US stock guidance powered by Frontman + Analyst + Reader agents")

    st.markdown("### 1) Choose your market region")
    st.info("US market is currently available. Global regions are planned for a later release.")

    col_left, col_right = st.columns([1.2, 1])
    with col_left:
        style = st.selectbox(
            "2) Select investment style",
            options=list(INVESTMENT_STYLES.keys()),
            index=0,
        )
        ticker_input = st.text_input("3) Enter a US stock ticker (e.g., AAPL)").strip().upper()
        wants_recommendation = st.checkbox("I want suggested tickers instead", value=not bool(ticker_input))

    with col_right:
        st.markdown("#### Suggested tickers by style")
        for tkr, company in INVESTMENT_STYLES[style].items():
            st.write(f"- **{tkr}** — {company}")

    selected_ticker = ticker_input
    if not selected_ticker or wants_recommendation:
        rec_ticker = st.selectbox(
            "Pick one suggested ticker to continue",
            options=list(INVESTMENT_STYLES[style].keys()),
            index=0,
        )
        selected_ticker = rec_ticker

    st.markdown("---")
    if st.button("Run Compass Reading", type="primary"):
        try:
            market_df = fetch_market_data(selected_ticker)
            analyst_model, cv_metrics, processed = train_analyst_model(market_df)

            latest = processed.tail(1)
            analyst_output = analyst_model.predict_current(latest)
            reader_score, reader_summary = mock_reader_sentiment(selected_ticker)

            probs = analyst_output["probabilities"]
            directive = _compose_directive(
                analyst_buy_prob=float(probs["+1"]),
                analyst_sell_prob=float(probs["-1"]),
                sentiment=float(reader_score),
            )
            reasoning = _plain_english_reasoning(
                directive,
                probs,
                sentiment=float(reader_score),
                summary=reader_summary,
                latest_row=latest.iloc[0],
            )

            current_close = float(latest.iloc[0]["Close"])
            current_vol = float(latest.iloc[0]["Volatility_EMA20"])
            buy_trigger = current_close + (current_vol * 2)
            sell_trigger = current_close - current_vol

            st.success(directive)

            m1, m2, m3 = st.columns(3)
            m1.metric("Analyst Buy (+1)", f"{probs['+1']:.1%}")
            m2.metric("Analyst Hold (0)", f"{probs['0']:.1%}")
            m3.metric("Analyst Sell (-1)", f"{probs['-1']:.1%}")

            st.markdown("### Reader (News Context)")
            st.write(f"**Sentiment score:** {reader_score:+.2f}  ")
            st.write(f"**Summary:** {reader_summary}")

            st.markdown("### Plain-English Compass Translation")
            st.write(reasoning)
            st.info(
                f"Suggested levels: consider buying above **${buy_trigger:.2f}**, protecting capital below **${sell_trigger:.2f}**, "
                "or waiting for stronger alignment over the next ~20 trading days."
            )

            st.markdown("### Analyst Validation Snapshot")
            st.json(
                {
                    "ticker": selected_ticker,
                    "cv_f1_macro_mean": round(float(cv_metrics["cv_f1_macro_mean"]), 4),
                    "cv_f1_macro_std": round(float(cv_metrics["cv_f1_macro_std"]), 4),
                }
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Unable to generate compass reading: {exc}")

    render_disclaimer_banner()


if __name__ == "__main__":
    main()
