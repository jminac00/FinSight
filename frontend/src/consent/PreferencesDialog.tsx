import { useEffect, useRef, useState } from 'react'
import { Button } from '../components/ui/Button'
import { CloseIcon } from '../components/icons'
import { useConsent } from './ConsentContext'
import type { OptionalCategory } from './types'

type Row = {
  id: OptionalCategory | 'necessary'
  name: string
  description: string
  locked?: boolean
}

const ROWS: Row[] = [
  {
    id: 'necessary',
    name: 'Estrictamente necesarias',
    description:
      'Imprescindibles para el funcionamiento del sitio, incluida la memoria de tu decisión sobre las cookies. No requieren consentimiento y no se pueden desactivar.',
    locked: true,
  },
  {
    id: 'functional',
    name: 'Funcionales',
    description:
      'Recuerdan tus preferencias: el último valor analizado, el tema visual (claro/oscuro) y la confirmación de lectura del aviso legal.',
  },
  {
    id: 'analytics',
    name: 'Analíticas',
    description:
      'Nos ayudan a entender el uso de la plataforma de forma agregada (páginas y valores más consultados, duración de sesión). Nunca te identifican.',
  },
]

export function PreferencesDialog() {
  const { preferencesOpen, closePreferences, consent, save } = useConsent()
  const ref = useRef<HTMLDialogElement>(null)
  const [choice, setChoice] = useState({
    functional: consent.functional,
    analytics: consent.analytics,
  })

  // Sync the native dialog with context state and seed toggles on open.
  useEffect(() => {
    const dialog = ref.current
    if (!dialog) return
    if (preferencesOpen && !dialog.open) {
      setChoice({ functional: consent.functional, analytics: consent.analytics })
      dialog.showModal()
    } else if (!preferencesOpen && dialog.open) {
      dialog.close()
    }
  }, [preferencesOpen, consent.functional, consent.analytics])

  return (
    <dialog
      ref={ref}
      onClose={closePreferences}
      aria-labelledby="prefs-title"
      className="m-auto w-[min(34rem,calc(100vw-2rem))] rounded-lg border border-border bg-surface p-0 text-ink shadow-lg backdrop:bg-black/40 backdrop:backdrop-blur-sm"
    >
      <div className="flex items-start justify-between gap-4 border-b border-border p-5">
        <h2 id="prefs-title" className="text-lg font-semibold">
          Preferencias de cookies
        </h2>
        <button
          type="button"
          onClick={closePreferences}
          aria-label="Cerrar"
          className="-m-1 inline-flex h-9 w-9 items-center justify-center rounded-md text-ink-muted hover:bg-panel hover:text-ink"
        >
          <CloseIcon />
        </button>
      </div>

      <div className="max-h-[60vh] overflow-y-auto p-5">
        <p className="mb-4 text-sm text-ink-muted">
          Configura qué categorías de almacenamiento permites. Tu decisión se conserva entre visitas
          y puedes cambiarla cuando quieras desde el pie de página.
        </p>
        <ul className="space-y-4">
          {ROWS.map((row) => {
            const checked = row.locked ? true : choice[row.id as OptionalCategory]
            return (
              <li key={row.id} className="rounded-md border border-border bg-bg p-4">
                <div className="flex items-center justify-between gap-4">
                  <h3 className="font-medium">{row.name}</h3>
                  {row.locked ? (
                    <span className="text-xs font-medium text-success-fg">Siempre activas</span>
                  ) : (
                    <label className="inline-flex cursor-pointer items-center gap-2 text-sm">
                      <span className="sr-only">Activar cookies {row.name.toLowerCase()}</span>
                      <input
                        type="checkbox"
                        checked={checked}
                        onChange={(e) =>
                          setChoice((c) => ({ ...c, [row.id]: e.target.checked }))
                        }
                        className="h-4 w-4 rounded border-border-strong text-accent focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
                      />
                    </label>
                  )}
                </div>
                <p className="mt-1.5 text-sm text-ink-muted">{row.description}</p>
              </li>
            )
          })}
        </ul>
      </div>

      <div className="flex flex-wrap justify-end gap-2 border-t border-border p-5">
        <Button variant="secondary" onClick={() => save({ functional: false, analytics: false })}>
          Rechazar todo
        </Button>
        <Button variant="secondary" onClick={() => save({ functional: true, analytics: true })}>
          Aceptar todo
        </Button>
        <Button variant="primary" onClick={() => save(choice)}>
          Guardar preferencias
        </Button>
      </div>
    </dialog>
  )
}
