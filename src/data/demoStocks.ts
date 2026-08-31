export type Direction = "buy" | "watch" | "sell";

export interface PricePoint {
  date: string;
  close: number;
}

export interface Stock {
  symbol: string;
  name: string;
  sector: string;
  price: number;
  change: number;
  operatingMargin: number;
  roe: number;
  roic: number;
  freeCashFlow: string;
  cashToMarketCap: number;
  netDebt: string;
  interestCoverage: number;
  history: PricePoint[];
}

const stockSeeds = [
  ["AAPL", "Apple Inc.", "Technology", 214.4, 2.15, 31.6, 147.4, 54.6, "$108.8B", 4.2, "$78.1B", 29.8],
  ["MSFT", "Microsoft", "Technology", 438.6, 1.24, 45.6, 33.7, 28.1, "$74.1B", 4.5, "$25.8B", 49.2],
  ["NVDA", "NVIDIA", "Semiconductors", 138.2, 3.91, 54.1, 91.2, 95.3, "$60.9B", 2.0, "$18.1B", 42.4],
  ["AMZN", "Amazon", "Consumer Cyclical", 201.4, -0.64, 10.8, 20.9, 12.6, "$32.2B", 4.7, "$46.4B", 7.3],
  ["GOOGL", "Alphabet", "Communication", 176.8, 1.03, 31.1, 30.8, 31.5, "$72.7B", 7.9, "-$97.1B", 0],
  ["META", "Meta Platforms", "Communication", 540.3, 1.88, 41.4, 34.1, 32.3, "$51.1B", 11.0, "-$40.2B", 0],
  ["TSLA", "Tesla", "Consumer Cyclical", 330.1, -2.72, 7.2, 10.5, 9.8, "$3.6B", 9.8, "$2.5B", 27.1],
  ["BRK-B", "Berkshire Hathaway", "Financial Services", 482.4, 0.32, 24.6, 7.8, 6.3, "$26.1B", 2.7, "-$147.0B", 0],
  ["JPM", "JPMorgan Chase", "Financial Services", 251.6, 0.76, 35.2, 18.7, 0, "$0.0B", 0, "$249.2B", 0],
  ["JNJ", "Johnson & Johnson", "Healthcare", 161.8, -0.18, 24.8, 31.2, 21.8, "$18.7B", 2.1, "$20.5B", 11.7],
] as const;

function createHistory(endPrice: number, seed: number): PricePoint[] {
  const history: PricePoint[] = [];
  const start = new Date("2024-09-02T00:00:00");
  const base = endPrice * (0.67 + (seed % 5) * 0.04);

  for (let day = 0; day < 261; day += 1) {
    const date = new Date(start);
    date.setDate(date.getDate() + day);
    if (date.getDay() === 0 || date.getDay() === 6) continue;

    const progress = day / 260;
    const wave = Math.sin(day * (0.11 + seed * 0.002)) * endPrice * 0.035;
    const correction = Math.cos(day * 0.037 + seed) * endPrice * 0.018;
    const close = base + (endPrice - base) * progress + wave + correction;
    history.push({ date: date.toISOString().slice(0, 10), close: Number(close.toFixed(2)) });
  }

  return history;
}

export const stocks: Stock[] = stockSeeds.map((seed, index) => ({
  symbol: seed[0],
  name: seed[1],
  sector: seed[2],
  price: seed[3],
  change: seed[4],
  operatingMargin: seed[5],
  roe: seed[6],
  roic: seed[7],
  freeCashFlow: seed[8],
  cashToMarketCap: seed[9],
  netDebt: seed[10],
  interestCoverage: seed[11],
  history: createHistory(seed[3], index + 3),
}));