import { useEffect, useState } from "react";
import { ChevronDown, CircleAlert, Compass, Info, Search, Sparkles, TrendingDown, TrendingUp } from "lucide-react";
import { stocks, type Direction, type Stock } from "./data/demoStocks";
import { loadStaticStock, type StaticAssetStatus, type StoredForecast } from "./data/staticData";

const horizons = [1, 3, 6, 12, 24];

interface Forecast {
  direction: Direction;
  expectedReturn: number;
  lowerPrice: number;
  medianPrice: number;
  upperPrice: number;
  upside: number;
  stable: number;
  downside: number;
  risk: number;
}

function getForecast(stock: Stock, historyIndex: number, months: number): Forecast {
  const point = stock.history[historyIndex];
  const priorPoint = stock.history[Math.max(0, historyIndex - 20)];
  const momentum = (point.close - priorPoint.close) / priorPoint.close;
  const quality = (stock.operatingMargin + Math.min(stock.roe, 80) + Math.min(stock.roic, 80)) / 240;
  const horizonDrag = months * 0.006;
  const expectedReturn = (momentum * 0.35 + quality * 0.18 - horizonDrag) * Math.sqrt(months);
  const volatility = 0.08 + months * 0.018 + Math.abs(momentum) * 0.35;
  const upside = Math.round(Math.max(18, Math.min(76, 50 + expectedReturn * 145 - volatility * 28)));
  const downside = Math.round(Math.max(10, Math.min(68, 28 - expectedReturn * 120 + volatility * 42)));
  const stable = 100 - upside - downside;
  const risk = Math.round(Math.max(18, Math.min(86, volatility * 280 + (downside - 20) * 0.7)));
  const direction: Direction = expectedReturn > 0.035 && upside > downside + 12 ? "buy" : expectedReturn < -0.025 || downside > 42 ? "sell" : "watch";
  const medianPrice = point.close * (1 + expectedReturn);

  return {
    direction,
    expectedReturn,
    lowerPrice: medianPrice * (1 - volatility),
    medianPrice,
    upperPrice: medianPrice * (1 + volatility),
    upside,
    stable,
    downside,
    risk,
  };
}

function formatMoney(value: number) {
  return new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" }).format(value);
}

function formatDate(value: string) {
  return new Intl.DateTimeFormat("en-US", { month: "short", day: "numeric", year: "numeric" }).format(new Date(`${value}T00:00:00`));
}

function storedForecastFor(forecasts: StoredForecast[], date: string, months: number): StoredForecast | undefined {
  const options = forecasts.filter((forecast) => forecast.horizonMonths === months && forecast.asOfDate <= date);
  return options.at(-1);
}

function PriceChart({ stock, activeIndex, onSelect }: { stock: Stock; activeIndex: number; onSelect: (index: number) => void }) {
  const width = 920;
  const height = 300;
  const padding = { top: 24, right: 18, bottom: 30, left: 58 };
  const closes = stock.history.map((point) => point.close);
  const min = Math.min(...closes) * 0.94;
  const max = Math.max(...closes) * 1.06;
  const xFor = (index: number) => padding.left + (index / (stock.history.length - 1)) * (width - padding.left - padding.right);
  const yFor = (value: number) => padding.top + (1 - (value - min) / (max - min)) * (height - padding.top - padding.bottom);
  const line = stock.history.map((point, index) => `${index === 0 ? "M" : "L"}${xFor(index).toFixed(1)},${yFor(point.close).toFixed(1)}`).join(" ");
  const selected = stock.history[activeIndex];

  return (
    <div className="chart-wrap">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${stock.symbol} closing-price chart`} onPointerDown={(event) => {
        const bounds = event.currentTarget.getBoundingClientRect();
        const ratio = Math.max(0, Math.min(1, (event.clientX - bounds.left - (padding.left / width) * bounds.width) / (bounds.width * (width - padding.left - padding.right) / width)));
        onSelect(Math.round(ratio * (stock.history.length - 1)));
      }}>
        {[0.2, 0.5, 0.8].map((ratio) => <line key={ratio} x1={padding.left} x2={width - padding.right} y1={padding.top + (height - padding.top - padding.bottom) * ratio} y2={padding.top + (height - padding.top - padding.bottom) * ratio} className="grid-line" />)}
        <text x="4" y={yFor(max) + 5} className="axis-label">{formatMoney(max)}</text>
        <text x="4" y={yFor((max + min) / 2) + 5} className="axis-label">{formatMoney((max + min) / 2)}</text>
        <text x="4" y={yFor(min) + 5} className="axis-label">{formatMoney(min)}</text>
        <path d={line} className="price-line" />
        <line x1={xFor(activeIndex)} x2={xFor(activeIndex)} y1={padding.top} y2={height - padding.bottom} className="selection-line" />
        <circle cx={xFor(activeIndex)} cy={yFor(selected.close)} r="6" className="selection-dot" />
        <text x={padding.left} y={height - 6} className="axis-label">{formatDate(stock.history[0].date)}</text>
        <text x={width - padding.right} y={height - 6} textAnchor="end" className="axis-label">{formatDate(stock.history.at(-1)!.date)}</text>
      </svg>
      <div className="chart-tooltip" style={{ left: `${(xFor(activeIndex) / width) * 100}%` }}>
        <strong>{formatMoney(selected.close)}</strong><span>{formatDate(selected.date)}</span>
      </div>
    </div>
  );
}

function App() {
  const [symbol, setSymbol] = useState("AAPL");
  const [query, setQuery] = useState("");
  const [activeIndex, setActiveIndex] = useState(stocks[0].history.length - 1);
  const [months, setMonths] = useState(3);
  const [loadedStock, setLoadedStock] = useState<Stock | null>(null);
  const [storedForecasts, setStoredForecasts] = useState<StoredForecast[]>([]);
  const [assetStatus, setAssetStatus] = useState<StaticAssetStatus>({ pricesLoaded: false, fundamentalsLoaded: false, forecastsLoaded: false });
  const fallbackStock = stocks.find((item) => item.symbol === symbol)!;
  const stock = loadedStock ?? fallbackStock;
  const visibleStocks = stocks.filter((item) => `${item.symbol} ${item.name}`.toLowerCase().includes(query.toLowerCase()));
  const storedForecast = storedForecastFor(storedForecasts, stock.history[activeIndex].date, months);
  const usesStoredForecast = storedForecast !== undefined;
  const forecast = storedForecast ? {
    direction: storedForecast.recommendation,
    expectedReturn: storedForecast.expectedReturn,
    lowerPrice: storedForecast.interval.low,
    medianPrice: storedForecast.predictedPrice,
    upperPrice: storedForecast.interval.high,
    upside: storedForecast.probabilities.buy,
    stable: storedForecast.probabilities.watch,
    downside: storedForecast.probabilities.sell,
    risk: storedForecast.riskScore,
  } : getForecast(stock, activeIndex, months);
  const selectedPrice = stock.history[activeIndex].close;
  const previousPrice = stock.history[Math.max(0, activeIndex - 1)].close;
  const selectedDailyChange = activeIndex === 0 ? 0 : ((selectedPrice / previousPrice) - 1) * 100;
  const signalLabel = forecast.direction === "buy" ? "Favourable bearing" : forecast.direction === "sell" ? "Defensive bearing" : "Hold course";
  const signalCopy = forecast.direction === "buy" ? "Expected return and model confidence lean upward." : forecast.direction === "sell" ? "Downside probability outweighs the projected return." : "The model sees no decisive directional advantage.";

  useEffect(() => {
    let cancelled = false;
    setLoadedStock(null);
    setStoredForecasts([]);
    setAssetStatus({ pricesLoaded: false, fundamentalsLoaded: false, forecastsLoaded: false });
    loadStaticStock(fallbackStock).then((result) => {
      if (!cancelled && result) {
        setLoadedStock(result.stock);
        setStoredForecasts(result.forecasts);
        setAssetStatus(result.status);
        setActiveIndex(result.stock.history.length - 1);
      }
    }).catch(() => undefined);
    return () => { cancelled = true; };
  }, [fallbackStock]);

  return (
    <main>
      <header className="topbar">
        <a className="brand" href={import.meta.env.BASE_URL} aria-label="Market Compass home"><Compass size={27} strokeWidth={1.8} /><span>Market <b>Compass</b></span></a>
        <div className="data-status"><span className="status-dot" />{assetStatus.pricesLoaded ? "Real historical prices" : "Demo historical prices"}<span className="muted">through {formatDate(stock.history.at(-1)!.date)}</span></div>
      </header>

      <section className="page-heading">
        <div><p className="eyebrow">Decision workspace</p><h1>Find your market bearing.</h1></div>
        <p className="heading-note">Choose a stock, a date, and a horizon. This interface is wired for point-in-time forecasts.</p>
      </section>

      <section className="controls" aria-label="Stock and forecast controls">
        <label className="stock-search"><Search size={18} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search ticker or company" aria-label="Search stocks" /><ChevronDown size={16} /></label>
        <div className="stock-pills" role="listbox" aria-label="Available stocks">
          {visibleStocks.map((item) => <button key={item.symbol} className={item.symbol === symbol ? "stock-pill active" : "stock-pill"} onClick={() => { setSymbol(item.symbol); setActiveIndex(item.history.length - 1); }}>{item.symbol}</button>)}
        </div>
        <div className="horizons" aria-label="Forecast horizon">
          <span>Forecast</span>{horizons.map((horizon) => <button key={horizon} className={months === horizon ? "active" : ""} onClick={() => setMonths(horizon)}>{horizon < 12 ? `${horizon}M` : `${horizon / 12}Y`}</button>)}
        </div>
      </section>

      <section className="instrument-header">
        <div><div className="symbol-row"><h2>{stock.symbol}</h2><span>{stock.sector}</span></div><p>{stock.name}</p></div>
        <div className="quote"><strong>{formatMoney(selectedPrice)}</strong><span className={selectedDailyChange >= 0 ? "positive" : "negative"}>{selectedDailyChange >= 0 ? "+" : ""}{selectedDailyChange.toFixed(2)}% selected-date move</span></div>
      </section>

      <section className="dashboard-grid">
        <article className="price-panel">
          <div className="panel-heading"><div><p className="eyebrow">Historical close</p><h3>Choose the prediction point</h3></div><span className="click-hint">Click chart to change date</span></div>
          <PriceChart stock={stock} activeIndex={activeIndex} onSelect={setActiveIndex} />
        </article>

        <article className={`signal-panel ${forecast.direction}`}>
          <div className="panel-heading"><div><p className="eyebrow">{usesStoredForecast ? "Model compass signal" : "Prototype compass signal"}</p><h3>{signalLabel}</h3></div><Sparkles size={19} /></div>
          <div className="compass-wrap"><div className="compass"><span className="north">BUY</span><span className="east">WATCH</span><span className="south">SELL</span><div className={`needle ${forecast.direction}`} /></div></div>
          <p className="signal-copy">{signalCopy}</p>
          <div className="expected-return"><span>Expected {months} month return</span><strong className={forecast.expectedReturn >= 0 ? "positive" : "negative"}>{forecast.expectedReturn >= 0 ? "+" : ""}{(forecast.expectedReturn * 100).toFixed(1)}%</strong></div>
        </article>

        <article className="forecast-panel">
          <div className="panel-heading"><div><p className="eyebrow">{usesStoredForecast ? "Predicted price" : "Prototype price estimate"}</p><h3>{months < 12 ? `${months}-month` : `${months / 12}-year`} range</h3></div><TrendingUp size={19} /></div>
          <div className="range-figure"><span>{formatMoney(forecast.lowerPrice)}</span><div><i style={{ left: `${Math.max(8, Math.min(92, 50 + forecast.expectedReturn * 160))}%` }} /></div><span>{formatMoney(forecast.upperPrice)}</span></div>
          <div className="median-price"><span>Median estimate</span><strong>{formatMoney(forecast.medianPrice)}</strong></div>
          <p className="microcopy">{usesStoredForecast ? "80% model interval from the selected historical point." : "Illustrative interval until static model forecasts are generated."}</p>
        </article>

        <article className="risk-panel">
          <div className="panel-heading"><div><p className="eyebrow">{usesStoredForecast ? "Direction probabilities" : "Prototype probabilities"}</p><h3>Risk distribution</h3></div><CircleAlert size={19} /></div>
          <div className="probability-bar"><span className="up" style={{ width: `${forecast.upside}%` }} /><span className="stable" style={{ width: `${forecast.stable}%` }} /><span className="down" style={{ width: `${forecast.downside}%` }} /></div>
          <div className="probability-labels"><span><i className="up-dot" />Up {forecast.upside}%</span><span><i className="stable-dot" />Stable {forecast.stable}%</span><span><i className="down-dot" />Down {forecast.downside}%</span></div>
          <div className="risk-score"><span>Risk score</span><div><b style={{ width: `${forecast.risk}%` }} /></div><strong>{forecast.risk}/100</strong></div>
        </article>
      </section>

      <section className="fundamentals"><div className="section-label"><p className="eyebrow">Fundamental indices</p><span><Info size={15} /> {assetStatus.fundamentalsLoaded ? "Latest SEC filing values at selected date" : "Prototype values until SEC data is generated"}</span></div><div className="metrics">
        {[
          ["Operating margin", `${stock.operatingMargin}%`], ["ROIC", stock.roic ? `${stock.roic}%` : "N/A"], ["ROE", `${stock.roe}%`], ["Free cash flow", stock.freeCashFlow], ["Cash / market cap", `${stock.cashToMarketCap}%`], ["Net debt", stock.netDebt], ["Interest coverage", stock.interestCoverage ? `${stock.interestCoverage}x` : "N/A"],
        ].map(([label, value]) => <div className="metric" key={label}><span>{label}</span><strong>{value}</strong></div>)}
      </div></section>

      <footer><TrendingDown size={15} /> Educational prototype. {assetStatus.pricesLoaded ? "Historical prices are real adjusted market data." : "Historical prices are simulated."} {assetStatus.fundamentalsLoaded ? "Fundamental values are loaded from SEC filings." : "Fundamental values are prototype placeholders."} {assetStatus.forecastsLoaded ? "Forecasts are loaded from the static model build." : "Forecasts are illustrative until the model build runs."} Not investment advice.</footer>
    </main>
  );
}

export default App;