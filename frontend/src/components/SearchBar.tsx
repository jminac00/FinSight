import { useId, useState, type FormEvent } from 'react'
import { useNavigate } from 'react-router-dom'
import { getStored, setStored } from '../consent/storage'
import { isValidTicker, normalizeTicker } from '../lib/ticker'
import { Button } from './ui/Button'
import { Input } from './ui/Input'
import { SearchIcon } from './icons'

const LAST_TICKER_KEY = 'finsight.lastTicker'

export function SearchBar() {
  const navigate = useNavigate()
  const inputId = useId()
  const errorId = useId()
  // Prefill the last analyzed ticker only if functional consent was granted (RNF-37).
  const [value, setValue] = useState(() => getStored(LAST_TICKER_KEY, 'functional') ?? '')
  const [error, setError] = useState('')

  function handleSubmit(event: FormEvent) {
    event.preventDefault()
    const ticker = normalizeTicker(value)
    if (!isValidTicker(ticker)) {
      setError('Introduce un símbolo válido: 2–5 caracteres alfanuméricos (p. ej. AAPL).')
      return
    }
    setError('')
    setStored(LAST_TICKER_KEY, ticker, 'functional')
    navigate(`/report/${ticker}`)
  }

  return (
    <form onSubmit={handleSubmit} noValidate className="w-full max-w-md">
      <label htmlFor={inputId} className="mb-2 block text-sm font-medium text-ink">
        Símbolo bursátil (ticker)
      </label>
      <div className="flex gap-2">
        <div className="relative flex-1">
          <SearchIcon className="pointer-events-none absolute left-3 top-1/2 w-5 -translate-y-1/2 text-ink-subtle" />
          <Input
            id={inputId}
            value={value}
            onChange={(e) => setValue(e.target.value.toUpperCase())}
            placeholder="AAPL"
            maxLength={5}
            autoComplete="off"
            autoCapitalize="characters"
            spellCheck={false}
            inputMode="text"
            invalid={Boolean(error)}
            aria-describedby={error ? errorId : undefined}
            className="pl-10 uppercase"
          />
        </div>
        <Button type="submit" size="lg" className="shrink-0">
          Analizar
        </Button>
      </div>
      {error ? (
        <p id={errorId} role="alert" className="mt-2 text-sm text-danger-fg">
          {error}
        </p>
      ) : (
        <p className="mt-2 text-sm text-ink-subtle">
          Acciones del mercado americano (Nasdaq).
        </p>
      )}
    </form>
  )
}
