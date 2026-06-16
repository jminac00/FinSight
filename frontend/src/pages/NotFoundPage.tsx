import { useEffect } from 'react'
import { Link } from 'react-router-dom'

export default function NotFoundPage() {
  useEffect(() => {
    document.title = 'Página no encontrada · FinSight'
  }, [])

  return (
    <div className="mx-auto max-w-prose px-4 py-24 text-center sm:px-6">
      <p className="text-sm font-semibold uppercase tracking-wide text-accent">Error 404</p>
      <h1 className="mt-3 text-3xl font-semibold tracking-tight text-ink">Página no encontrada</h1>
      <p className="mt-3 text-ink-muted">
        La página que buscas no existe o se ha movido.
      </p>
      <Link
        to="/"
        className="mt-8 inline-flex items-center justify-center rounded-md bg-accent px-5 py-2.5 text-sm font-medium text-accent-fg hover:bg-accent-hover"
      >
        Volver al inicio
      </Link>
    </div>
  )
}
