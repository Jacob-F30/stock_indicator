import type { PricePoint, Stock } from "./demoStocks";

interface RawFundamental {
  date: string;
  operating_margin: number | null;
  roe: number | null;
  roic: number | null;
  free_cash_flow: number | null;
  cash_to_market_cap: number | null;
  net_debt: number | null;
  interest_coverage: number | null;
}

export interface StoredForecast {
  asOfDate: string;
  horizonMonths: number;
  selectedPrice: number;
  predictedPrice: number;
  interval: { low: number; high: number; confidence: number };
  expectedReturn: number;
  probabilities: { buy: number; watch: number; sell: number };
  riskScore: number;
  recommendation: "buy" | "watch" | "sell";
}

export interface StaticAssetStatus {
  pricesLoaded: boolean;
  fundamentalsLoaded: boolean;
  forecastsLoaded: boolean;
}

interface StaticStockData {
  stock: Stock;
  forecasts: StoredForecast[];
  status: StaticAssetStatus;
}

const basePath = import.meta.env.BASE_URL;

function nearestFundamental(rows: RawFundamental[]): RawFundamental | undefined {
  return [...rows].reverse().find((row) => row.operating_margin !== null || row.roe !== null || row.free_cash_flow !== null);
}

function percent(value: number | null | undefined) {
  return value === null || value === undefined ? 0 : value * 100;
}

function money(value: number | null | undefined) {
  if (value === null || value === undefined) return "N/A";
  const absolute = Math.abs(value);
  const unit = absolute >= 1_000_000_000 ? "B" : "M";
  const divisor = unit === "B" ? 1_000_000_000 : 1_000_000;
  return `${value < 0 ? "-" : ""}$${(absolute / divisor).toFixed(1)}${unit}`;
}

export async function loadStaticStock(fallback: Stock): Promise<StaticStockData | null> {
  const directory = `${basePath}data/${fallback.symbol}`;
  const priceResponse = await fetch(`${directory}/prices.json`);
  if (!priceResponse.ok) return null;

  const prices = (await priceResponse.json()) as PricePoint[];
  if (!prices.length) return null;
  const [fundamentals, forecasts] = await Promise.all([
    fetch(`${directory}/fundamentals.json`).then(async (response) => response.ok ? response.json() as Promise<RawFundamental[]> : []).catch(() => []),
    fetch(`${directory}/forecasts.json`).then(async (response) => response.ok ? response.json() as Promise<StoredForecast[]> : []).catch(() => []),
  ]);
  const latest = nearestFundamental(fundamentals);

  return {
    stock: {
      ...fallback,
      price: prices.at(-1)!.close,
      history: prices,
      operatingMargin: latest?.operating_margin == null ? fallback.operatingMargin : percent(latest.operating_margin),
      roe: latest?.roe == null ? fallback.roe : percent(latest.roe),
      roic: latest?.roic == null ? fallback.roic : percent(latest.roic),
      freeCashFlow: latest?.free_cash_flow == null ? fallback.freeCashFlow : money(latest.free_cash_flow),
      cashToMarketCap: latest?.cash_to_market_cap == null ? fallback.cashToMarketCap : percent(latest.cash_to_market_cap),
      netDebt: latest ? money(latest.net_debt) : fallback.netDebt,
      interestCoverage: latest?.interest_coverage ?? fallback.interestCoverage,
    },
    forecasts,
    status: { pricesLoaded: true, fundamentalsLoaded: latest !== undefined, forecastsLoaded: forecasts.length > 0 },
  };
}