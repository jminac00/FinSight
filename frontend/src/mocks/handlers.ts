import { HttpResponse, delay, http } from 'msw'
import type { SymbolMatch } from '../api/types'
import { TICKER_RE } from '../lib/ticker'
import { fullReport, partialReport } from './fixtures'

/** Tickers that simulate the no-model (partial support) path. */
const PARTIAL_TICKERS = new Set(['TSLA', 'XYZ'])

/** A small symbol universe for the autocomplete in development and tests. */
const SEARCH_UNIVERSE: SymbolMatch[] = [
  { symbol: 'AAPL', description: 'Apple Inc', type: 'Common Stock', display_symbol: 'AAPL' },
  { symbol: 'MSFT', description: 'Microsoft Corp', type: 'Common Stock', display_symbol: 'MSFT' },
  { symbol: 'NVDA', description: 'NVIDIA Corp', type: 'Common Stock', display_symbol: 'NVDA' },
  { symbol: 'AMZN', description: 'Amazon.com Inc', type: 'Common Stock', display_symbol: 'AMZN' },
  { symbol: 'GOOGL', description: 'Alphabet Inc', type: 'Common Stock', display_symbol: 'GOOGL' },
  { symbol: 'META', description: 'Meta Platforms Inc', type: 'Common Stock', display_symbol: 'META' },
  { symbol: 'TSLA', description: 'Tesla Inc', type: 'Common Stock', display_symbol: 'TSLA' },
  {
    symbol: 'APLE',
    description: 'Apple Hospitality REIT Inc',
    type: 'Common Stock',
    display_symbol: 'APLE',
  },
]

export const handlers = [
  http.get('/api/v1/search', ({ request }) => {
    const q = (new URL(request.url).searchParams.get('q') ?? '').trim().toLowerCase()
    if (!q) return HttpResponse.json({ query: '', results: [] })
    const results = SEARCH_UNIVERSE.filter(
      (m) => m.symbol.toLowerCase().includes(q) || m.description.toLowerCase().includes(q),
    ).slice(0, 8)
    return HttpResponse.json({ query: q, results })
  }),
  http.get('/api/v1/report/:ticker', async ({ params }) => {
    const ticker = String(params.ticker).toUpperCase()

    if (!TICKER_RE.test(ticker)) {
      return HttpResponse.json(
        { detail: 'El símbolo debe tener entre 2 y 5 caracteres alfanuméricos.' },
        { status: 422 },
      )
    }

    // Brief delay so loading/skeleton states are observable in development.
    await delay(600)

    const companyName = SEARCH_UNIVERSE.find((m) => m.symbol === ticker)?.description ?? null
    const report = PARTIAL_TICKERS.has(ticker)
      ? partialReport(ticker, companyName)
      : fullReport(ticker, companyName)
    return HttpResponse.json(report)
  }),
]
