import { HttpResponse, delay, http } from 'msw'
import { TICKER_RE } from '../lib/ticker'
import { fullReport, partialReport } from './fixtures'

/** Tickers that simulate the no-model (partial support) path. */
const PARTIAL_TICKERS = new Set(['TSLA', 'XYZ'])

export const handlers = [
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

    const report = PARTIAL_TICKERS.has(ticker) ? partialReport(ticker) : fullReport(ticker)
    return HttpResponse.json(report)
  }),
]
