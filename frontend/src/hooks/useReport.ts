import { useCallback, useEffect, useState } from 'react'
import { ApiError, fetchReport } from '../api/client'
import type { ReportResponse } from '../api/types'

type ReportState = {
  data: ReportResponse | null
  loading: boolean
  error: ApiError | null
  /** Re-run the request bypassing the server cache (RF-05). */
  refresh: () => void
}

/**
 * Loads the consolidated report for a ticker. A per-run "stale" guard ensures
 * that responses from superseded requests (ticker change, refresh, unmount) are
 * ignored and never overwrite newer state.
 */
export function useReport(ticker: string): ReportState {
  const [data, setData] = useState<ReportResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<ApiError | null>(null)
  const [nonce, setNonce] = useState(0)
  const [forceRefresh, setForceRefresh] = useState(false)

  const refresh = useCallback(() => {
    setForceRefresh(true)
    setNonce((n) => n + 1)
  }, [])

  useEffect(() => {
    let stale = false
    setLoading(true)
    setError(null)

    fetchReport(ticker, forceRefresh)
      .then((report) => {
        if (stale) return
        setData(report)
        setLoading(false)
      })
      .catch((err: unknown) => {
        if (stale) return
        setError(err instanceof ApiError ? err : new ApiError('Error inesperado.', 0))
        setLoading(false)
      })

    return () => {
      stale = true
    }
    // `nonce` forces a re-fetch on refresh; `forceRefresh` is read inside.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ticker, nonce])

  return { data, loading, error, refresh }
}
