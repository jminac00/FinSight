import { Link } from 'react-router-dom'
import { LegalDisclaimer } from '../components/LegalDisclaimer'
import { useConsent } from '../consent/ConsentContext'

export function Footer() {
  const { openPreferences } = useConsent()

  return (
    <footer className="mt-16 border-t border-border bg-panel">
      <div className="mx-auto max-w-5xl px-4 py-8 sm:px-6">
        <LegalDisclaimer className="max-w-prose" />

        <nav aria-label="Enlaces legales" className="mt-6">
          <ul className="flex flex-wrap items-center gap-x-6 gap-y-2 text-sm">
            <li>
              <Link to="/privacy" className="text-ink-muted hover:text-accent">
                Política de privacidad
              </Link>
            </li>
            <li>
              <Link to="/terms" className="text-ink-muted hover:text-accent">
                Términos de uso
              </Link>
            </li>
            <li>
              <button
                type="button"
                onClick={openPreferences}
                className="text-ink-muted hover:text-accent"
              >
                Preferencias de cookies
              </button>
            </li>
          </ul>
        </nav>

        <p className="mt-6 text-xs text-ink-subtle">
          FinSight · Trabajo de Fin de Grado · Universidad de León
        </p>
      </div>
    </footer>
  )
}
