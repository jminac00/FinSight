import { Link } from 'react-router-dom'
import { Button } from '../components/ui/Button'
import { useConsent } from './ConsentContext'

/**
 * First-visit consent banner (RNF-36). No category is pre-selected as accepted;
 * the user must act. The banner is a complementary region announced politely so
 * it never steals focus from the page content.
 */
export function ConsentBanner() {
  const { needsDecision, acceptAll, rejectAll, openPreferences } = useConsent()
  if (!needsDecision) return null

  return (
    <div
      role="region"
      aria-label="Consentimiento de cookies"
      className="fixed inset-x-0 bottom-0 z-sticky border-t border-border bg-surface-raised shadow-lg"
    >
      <div className="mx-auto flex max-w-5xl flex-col gap-4 p-4 sm:p-5 lg:flex-row lg:items-center lg:justify-between">
        <p className="max-w-prose text-sm text-ink-muted">
          Usamos cookies estrictamente necesarias para que el sitio funcione y, con tu permiso,
          cookies funcionales y analíticas. Consulta la{' '}
          <Link to="/privacy" className="text-accent underline underline-offset-2 hover:text-accent-hover">
            política de privacidad
          </Link>
          .
        </p>
        <div className="flex flex-wrap gap-2 lg:shrink-0">
          <Button variant="ghost" size="sm" onClick={openPreferences}>
            Preferencias
          </Button>
          <Button variant="secondary" size="sm" onClick={rejectAll}>
            Rechazar todo
          </Button>
          <Button variant="primary" size="sm" onClick={acceptAll}>
            Aceptar todo
          </Button>
        </div>
      </div>
    </div>
  )
}
