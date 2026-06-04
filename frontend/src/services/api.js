const BASE_URL = '/api/v1'

/**
 * Fetch the consolidated analysis report for a stock ticker.
 *
 * @param {string} ticker - Uppercase stock symbol (e.g. 'AAPL').
 * @param {boolean} forceRefresh - If true, bypass server-side cache.
 * @returns {Promise<Object>} ReportResponse JSON.
 */
export async function fetchReport(ticker, forceRefresh = false) {
  const params = forceRefresh ? '?force_refresh=true' : ''
  const response = await fetch(`${BASE_URL}/report/${ticker}${params}`)
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Error desconocido' }))
    throw new Error(error.detail || `HTTP ${response.status}`)
  }
  return response.json()
}
