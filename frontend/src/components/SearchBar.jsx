import { useState } from 'react'
import { useNavigate } from 'react-router-dom'

const TICKER_RE = /^[A-Z0-9]{2,5}$/

export default function SearchBar() {
  const [input, setInput] = useState('')
  const [error, setError] = useState('')
  const navigate = useNavigate()

  function handleSubmit(e) {
    e.preventDefault()
    const ticker = input.trim().toUpperCase()
    if (!TICKER_RE.test(ticker)) {
      setError('Introduce un símbolo válido (2–5 caracteres alfanuméricos, p. ej. AAPL)')
      return
    }
    setError('')
    navigate(`/report/${ticker}`)
  }

  return (
    <form onSubmit={handleSubmit} aria-label="Búsqueda de acciones" className="flex flex-col items-center gap-3 w-full max-w-md">
      <div className="flex w-full gap-2">
        <label htmlFor="ticker-input" className="sr-only">
          Símbolo bursátil
        </label>
        <input
          id="ticker-input"
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Introduce un ticker (p. ej. AAPL)"
          maxLength={5}
          autoComplete="off"
          autoCapitalize="characters"
          className="flex-1 border border-gray-300 rounded-lg px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          aria-describedby={error ? 'ticker-error' : undefined}
        />
        <button
          type="submit"
          className="bg-blue-600 hover:bg-blue-700 text-white px-5 py-2 rounded-lg text-sm font-medium focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          Analizar
        </button>
      </div>
      {error && (
        <p id="ticker-error" role="alert" className="text-red-600 text-xs">
          {error}
        </p>
      )}
    </form>
  )
}
