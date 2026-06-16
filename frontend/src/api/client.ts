import type { ReportResponse } from './types'

/**
 * Absolute API base. Resolving against the current origin keeps the path
 * same-origin in the browser (and through the Vite proxy) while letting the
 * `fetch` implementation under test parse the URL.
 */
function apiUrl(path: string): string {
  const origin =
    typeof window !== 'undefined' && window.location?.origin
      ? window.location.origin
      : 'http://localhost'
  return `${origin}/api/v1${path}`
}

export class ApiError extends Error {
  /** HTTP status, or 0 for network/timeout failures. */
  readonly status: number

  constructor(message: string, status: number) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

/**
 * Fetch the consolidated analysis report for a ticker.
 *
 * @param ticker - Uppercase symbol (e.g. "AAPL").
 * @param forceRefresh - Bypass the server-side cache (RF-05).
 * @param signal - Abort signal for cancellation.
 */
export async function fetchReport(
  ticker: string,
  forceRefresh = false,
  signal?: AbortSignal,
): Promise<ReportResponse> {
  const query = forceRefresh ? '?force_refresh=true' : ''
  let response: Response
  try {
    response = await fetch(apiUrl(`/report/${encodeURIComponent(ticker)}${query}`), {
      headers: { Accept: 'application/json' },
      signal,
    })
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') throw error
    throw new ApiError('No se pudo conectar con el servidor.', 0)
  }

  if (!response.ok) {
    const detail = await response
      .json()
      .then((body: { detail?: string }) => body.detail)
      .catch(() => undefined)
    throw new ApiError(detail ?? `Error del servidor (${response.status}).`, response.status)
  }

  return (await response.json()) as ReportResponse
}
