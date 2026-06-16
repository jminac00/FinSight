import { useEffect, type ReactNode } from 'react'
import { Link } from 'react-router-dom'
import { ArrowLeftIcon } from './icons'

/** Shared shell for the privacy and terms pages: back link, title, prose column. */
export function LegalPage({
  title,
  documentTitle,
  updated,
  children,
}: {
  title: string
  documentTitle: string
  updated: string
  children: ReactNode
}) {
  useEffect(() => {
    document.title = documentTitle
  }, [documentTitle])

  return (
    <div className="mx-auto max-w-prose px-4 py-10 sm:px-6">
      <Link to="/" className="inline-flex items-center gap-1.5 text-sm text-ink-muted hover:text-accent">
        <ArrowLeftIcon className="w-4" />
        Volver al inicio
      </Link>
      <h1 className="mt-4 text-3xl font-semibold tracking-tight text-ink">{title}</h1>
      <article className="mt-8 space-y-8 text-ink-muted">{children}</article>
      <p className="mt-10 text-xs text-ink-subtle">Última actualización: {updated}</p>
    </div>
  )
}

export function LegalSection({ heading, children }: { heading: string; children: ReactNode }) {
  return (
    <section className="space-y-3">
      <h2 className="text-xl font-semibold text-ink">{heading}</h2>
      {children}
    </section>
  )
}
