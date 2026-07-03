import type { ReportResponse } from './types'

/**
 * Absolute API base.
 *
 * - In production the frontend and backend are separate Render services, so
 *   `VITE_API_BASE_URL` points at the backend origin.
 * - When it is unset (local dev and tests) we resolve against the current
 *   origin, keeping the path same-origin through the Vite `/api` proxy while
 *   letting the `fetch` implementation under test parse the URL.
 */
function apiBase(): string {
  const configured = import.meta.env.VITE_API_BASE_URL?.trim()
  if (configured) return configured.replace(/\/+$/, '')
  return typeof window !== 'undefined' && window.location?.origin
    ? window.location.origin
    : 'http://localhost'
}

function apiUrl(path: string): string {
  return `${apiBase()}/api/v1${path}`
}

export class ApiError extends Error {
  /** HTTP status, or 0 for network/timeout failures. */
  readonly status: number
  /** Seconds to wait before retrying, from the `Retry-After` header on 429. */
  readonly retryAfter?: number

  constructor(message: string, status: number, retryAfter?: number) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.retryAfter = retryAfter
  }
}

/** Parse a numeric `Retry-After` header (seconds); ignore HTTP-date form. */
function parseRetryAfter(header: string | null): number | undefined {
  if (!header) return undefined
  const seconds = Number(header)
  return Number.isFinite(seconds) && seconds >= 0 ? seconds : undefined
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
    const retryAfter =
      response.status === 429 ? parseRetryAfter(response.headers.get('Retry-After')) : undefined
    throw new ApiError(
      detail ?? `Error del servidor (${response.status}).`,
      response.status,
      retryAfter,
    )
  }

  return (await response.json()) as ReportResponse
}
