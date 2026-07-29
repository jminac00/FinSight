/**
 * The test run must not inherit the developer's dev configuration.
 *
 * Vitest loads `.env` through Vite, so a local `VITE_API_BASE_URL` used to make
 * the client build absolute URLs that the relative MSW handlers never matched —
 * turning the suite red on a clean checkout while CI, which has no `.env`,
 * stayed green.
 */

import { describe, expect, it } from 'vitest'
import { fetchReport, searchSymbols } from './client'

describe('API client under test', () => {
  it('exposes no external API base URL', () => {
    // The dev origin must not reach the test environment; requests stay
    // same-origin so the mock layer can intercept them.
    expect(import.meta.env.VITE_API_BASE_URL ?? '').toBe('')
  })

  it('routes symbol searches through the mock handlers', async () => {
    const results = await searchSymbols('apple')

    // searchSymbols fails soft, returning [] on any error, so a non-empty
    // result is proof the request was actually intercepted.
    expect(results.map((m) => m.symbol)).toContain('AAPL')
  })

  it('routes report requests through the mock handlers', async () => {
    const report = await fetchReport('AAPL')

    expect(report.ticker).toBe('AAPL')
  })
})
