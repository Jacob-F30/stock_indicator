# Market Compass

A static, educational stock-analysis prototype. It shows price history, dated fundamental indices, prediction intervals, directional probabilities, risk, and a Buy/Watch/Sell Compass signal for ten iconic US stocks.

There is no runtime backend, database, subscription, or browser market-data call. GitHub Pages serves the website; a Python job creates the data assets offline.

## Run The Interface

```bash
npm install
npm run dev
```

Open the URL printed by Vite. Run `npm run build` to typecheck and create the deployable static bundle in `dist/`.

## Generate Real Static Data

The checked-in interface uses explicitly labelled simulated data until this step runs. The generator downloads full adjusted daily price history and public SEC Companyfacts for:

```text
AAPL MSFT NVDA AMZN GOOGL META TSLA BRK-B JPM JNJ
```

To replace only the chart with real history from each stock's earliest available trading date, without SEC data or model training:

```bash
.venv\Scripts\python.exe scripts\build_static_data.py --prices-only
```

This is the quickest first refresh and writes `public/data/<symbol>/prices.json`. The interface labels the chart as real historical prices while it continues to label model and fundamental placeholders honestly.

```bash
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe scripts\build_static_data.py --sec-user-agent "Your Name contact@example.com"
```

The required identifier is SEC policy, not a secret. The command writes browser-ready JSON to `public/data/` and retains normalized analysis data as Parquet in `data/analysis/`. Generated files are ignored by Git; a scheduled GitHub Action builds and publishes them directly.

Use `--symbols AAPL MSFT` for a smaller run. `--forecast-stride 21` exports one point-in-time forecast approximately each trading month. A forecast uses SEC facts filed on or before its selected date and trains only on returns whose outcome would have been known at that date.

## Model And Evaluation

Features include returns, moving-average distance, volatility, volume change, operating margin, ROE, ROIC, FCF, cash-to-market-cap, net debt, and interest coverage. Three quantile gradient-boosting models estimate an 80% price interval and median return; that return and interval width produce the Compass direction, probabilities, and risk score.

For the initial holdout, models train through three months before the newest price. The newest three months are measured where the selected horizon has matured. Longer horizons need older walk-forward evaluations, so unavailable coverage must remain unavailable in the UI.

## Free Deployment

1. Create a public GitHub repository and enable GitHub Pages with **GitHub Actions** as its source.
2. Push this project to the default branch. `.github/workflows/pages.yml` builds and deploys the frontend.
3. Set repository variable `SEC_USER_AGENT` to a truthful contact string.
4. Run the **Refresh static data** workflow manually or let it run weekly. It downloads data, trains models, builds the frontend, and deploys the generated result.

The price source and SEC facts have their own usage conditions. Review those terms before redistributing generated market data publicly.

## Project Layout

```text
src/                         React Compass interface
scripts/build_static_data.py Offline acquisition, feature, model, and JSON export
public/data/                 Generated static browser assets (ignored)
.github/workflows/           GitHub Pages deploy and scheduled data refresh
app.py, analyst_model.py     Legacy Streamlit prototype retained only as reference
```

## Disclaimer

Educational prototype only. Forecasts can be wrong and are not investment advice.
