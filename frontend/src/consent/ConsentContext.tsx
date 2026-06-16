import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
  type ReactNode,
} from 'react'
import { readConsent, writeConsent } from './storage'
import type { ConsentState, OptionalCategory } from './types'

type ConsentChoice = Record<OptionalCategory, boolean>

type ConsentContextValue = {
  consent: ConsentState
  /** True until the user makes an explicit choice — drives the banner. */
  needsDecision: boolean
  /** Preferences dialog visibility (reopenable from the footer). */
  preferencesOpen: boolean
  openPreferences: () => void
  closePreferences: () => void
  acceptAll: () => void
  rejectAll: () => void
  save: (choice: ConsentChoice) => void
  has: (category: OptionalCategory) => boolean
}

const ConsentContext = createContext<ConsentContextValue | null>(null)

function persist(choice: ConsentChoice): ConsentState {
  const next: ConsentState = {
    decided: true,
    functional: choice.functional,
    analytics: choice.analytics,
    timestamp: new Date().toISOString(),
  }
  writeConsent(next)
  return next
}

export function ConsentProvider({ children }: { children: ReactNode }) {
  const [consent, setConsent] = useState<ConsentState>(() => readConsent())
  const [preferencesOpen, setPreferencesOpen] = useState(false)

  const commit = useCallback((choice: ConsentChoice) => {
    setConsent(persist(choice))
    setPreferencesOpen(false)
  }, [])

  const value = useMemo<ConsentContextValue>(
    () => ({
      consent,
      needsDecision: !consent.decided,
      preferencesOpen,
      openPreferences: () => setPreferencesOpen(true),
      closePreferences: () => setPreferencesOpen(false),
      acceptAll: () => commit({ functional: true, analytics: true }),
      rejectAll: () => commit({ functional: false, analytics: false }),
      save: commit,
      has: (category) => consent[category],
    }),
    [consent, preferencesOpen, commit],
  )

  return <ConsentContext.Provider value={value}>{children}</ConsentContext.Provider>
}

export function useConsent(): ConsentContextValue {
  const ctx = useContext(ConsentContext)
  if (!ctx) throw new Error('useConsent must be used within a ConsentProvider')
  return ctx
}
