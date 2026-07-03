import { useEffect, useState } from 'react'
import { searchSymbols } from '../api/client'
import type { SymbolMatch } from '../api/types'

const MIN_LENGTH = 2
const DEBOUNCE_MS = 250

/**
 * Debounced symbol/company search for the autocomplete. Ignores responses from
 * superseded queries with a stale flag (rather than an AbortSignal, which the
 * jsdom/undici test environment rejects).
 */
export function useSymbolSearch(query: string): { results: SymbolMatch[]; loading: boolean } {
  const [results, setResults] = useState<SymbolMatch[]>([])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    const q = query.trim()
    if (q.length < MIN_LENGTH) {
      setResults([])
      setLoading(false)
      return
    }

    let stale = false
    setLoading(true)
    const timer = setTimeout(() => {
      searchSymbols(q).then((matches) => {
        if (stale) return
        setResults(matches)
        setLoading(false)
      })
    }, DEBOUNCE_MS)

    return () => {
      stale = true
      clearTimeout(timer)
    }
  }, [query])

  return { results, loading }
}
