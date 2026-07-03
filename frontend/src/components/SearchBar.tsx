import { useId, useState, type FormEvent, type KeyboardEvent } from 'react'
import { useNavigate } from 'react-router-dom'
import type { SymbolMatch } from '../api/types'
import { getStored, setStored } from '../consent/storage'
import { useSymbolSearch } from '../hooks/useSymbolSearch'
import { isValidTicker, normalizeTicker } from '../lib/ticker'
import { Button } from './ui/Button'
import { Input } from './ui/Input'
import { SearchIcon } from './icons'

const LAST_TICKER_KEY = 'finsight.lastTicker'

export function SearchBar() {
  const navigate = useNavigate()
  const inputId = useId()
  const listId = useId()
  const errorId = useId()
  const optionId = (index: number) => `${listId}-opt-${index}`

  // Prefill the last analyzed ticker only if functional consent was granted (RNF-37).
  const [value, setValue] = useState(() => getStored(LAST_TICKER_KEY, 'functional') ?? '')
  const [open, setOpen] = useState(false)
  const [activeIndex, setActiveIndex] = useState(-1)
  const [error, setError] = useState('')

  const { results, loading } = useSymbolSearch(open ? value : '')
  const showList = open && value.trim().length >= 2
  const hasOptions = results.length > 0

  function go(symbol: string) {
    const ticker = normalizeTicker(symbol)
    setStored(LAST_TICKER_KEY, ticker, 'functional')
    setOpen(false)
    navigate(`/report/${encodeURIComponent(ticker)}`)
  }

  function select(match: SymbolMatch) {
    setValue(match.symbol)
    go(match.symbol)
  }

  function handleSubmit(event: FormEvent) {
    event.preventDefault()
    if (activeIndex >= 0 && results[activeIndex]) {
      select(results[activeIndex])
      return
    }
    const ticker = normalizeTicker(value)
    if (isValidTicker(ticker)) {
      go(ticker)
      return
    }
    if (hasOptions) {
      select(results[0])
      return
    }
    setError('Introduce un nombre de empresa o un símbolo válido (p. ej. Apple o AAPL).')
  }

  function handleKeyDown(event: KeyboardEvent<HTMLInputElement>) {
    if (event.key === 'ArrowDown') {
      event.preventDefault()
      setOpen(true)
      if (hasOptions) setActiveIndex((i) => Math.min(i + 1, results.length - 1))
    } else if (event.key === 'ArrowUp') {
      event.preventDefault()
      if (hasOptions) setActiveIndex((i) => Math.max(i - 1, 0))
    } else if (event.key === 'Escape') {
      setOpen(false)
      setActiveIndex(-1)
    }
  }

  return (
    <form onSubmit={handleSubmit} noValidate className="w-full max-w-md">
      <label htmlFor={inputId} className="mb-2 block text-sm font-medium text-ink">
        Empresa o símbolo
      </label>
      <div className="flex gap-2">
        <div className="relative flex-1">
          <SearchIcon className="pointer-events-none absolute left-3 top-1/2 w-5 -translate-y-1/2 text-ink-subtle" />
          <Input
            id={inputId}
            role="combobox"
            aria-expanded={showList && hasOptions}
            aria-controls={showList && hasOptions ? listId : undefined}
            aria-autocomplete="list"
            aria-activedescendant={activeIndex >= 0 ? optionId(activeIndex) : undefined}
            aria-describedby={error ? errorId : undefined}
            invalid={Boolean(error)}
            autoComplete="off"
            spellCheck={false}
            value={value}
            onChange={(e) => {
              setValue(e.target.value)
              setOpen(true)
              setActiveIndex(-1)
              setError('')
            }}
            onFocus={() => setOpen(true)}
            onBlur={() => setOpen(false)}
            onKeyDown={handleKeyDown}
            placeholder="Apple o AAPL"
            className="pl-10"
          />

          {showList && hasOptions ? (
            <ul
              id={listId}
              role="listbox"
              aria-label="Sugerencias"
              className="absolute z-dropdown mt-1 max-h-72 w-full overflow-auto rounded-md border border-border bg-surface-raised py-1 shadow-md"
            >
              {results.map((match, index) => (
                // Options are chosen with the pointer; keyboard selection is handled on the
                // combobox input (arrow keys + Enter), per the ARIA listbox pattern.
                // eslint-disable-next-line jsx-a11y/click-events-have-key-events
                <li
                  key={`${match.symbol}-${index}`}
                  id={optionId(index)}
                  role="option"
                  aria-selected={index === activeIndex}
                  onMouseDown={(e) => e.preventDefault()}
                  onClick={() => select(match)}
                  className={`cursor-pointer px-3 py-2 text-sm ${
                    index === activeIndex ? 'bg-accent-subtle text-ink' : 'text-ink-muted'
                  }`}
                >
                  <span className="font-semibold text-ink">{match.symbol}</span>
                  <span className="text-ink-subtle"> · {match.description}</span>
                </li>
              ))}
            </ul>
          ) : null}

          {showList && !hasOptions ? (
            <div
              role="status"
              className="absolute z-dropdown mt-1 w-full rounded-md border border-border bg-surface-raised px-3 py-2 text-sm text-ink-subtle shadow-md"
            >
              {loading ? 'Buscando…' : 'Sin resultados'}
            </div>
          ) : null}
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
          Busca por nombre de empresa o por símbolo bursátil.
        </p>
      )}
    </form>
  )
}
