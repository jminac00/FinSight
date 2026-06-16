import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from 'react'
import { useConsent } from '../consent/ConsentContext'
import { getStored, removeStored, setStored } from '../consent/storage'

type Theme = 'light' | 'dark'

const THEME_KEY = 'finsight.theme'

type ThemeContextValue = {
  theme: Theme
  toggle: () => void
  setTheme: (theme: Theme) => void
}

const ThemeContext = createContext<ThemeContextValue | null>(null)

function resolveInitial(functionalAllowed: boolean): Theme {
  if (functionalAllowed) {
    const stored = getStored(THEME_KEY, 'functional')
    if (stored === 'light' || stored === 'dark') return stored
  }
  if (typeof window !== 'undefined' && window.matchMedia('(prefers-color-scheme: dark)').matches) {
    return 'dark'
  }
  return 'light'
}

export function ThemeProvider({ children }: { children: ReactNode }) {
  const { has } = useConsent()
  const functionalAllowed = has('functional')
  const [theme, setThemeState] = useState<Theme>(() => resolveInitial(functionalAllowed))

  // Apply the theme class to <html> so tokens.css resolves the right palette.
  useEffect(() => {
    document.documentElement.classList.toggle('dark', theme === 'dark')
  }, [theme])

  // Persist only with functional consent; otherwise the choice lives in memory
  // for the session and any prior stored value is removed (RNF-37/39).
  useEffect(() => {
    if (functionalAllowed) {
      setStored(THEME_KEY, theme, 'functional')
    } else {
      removeStored(THEME_KEY)
    }
  }, [theme, functionalAllowed])

  const setTheme = useCallback((next: Theme) => setThemeState(next), [])
  const toggle = useCallback(() => setThemeState((t) => (t === 'dark' ? 'light' : 'dark')), [])

  const value = useMemo<ThemeContextValue>(
    () => ({ theme, toggle, setTheme }),
    [theme, toggle, setTheme],
  )

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>
}

export function useTheme(): ThemeContextValue {
  const ctx = useContext(ThemeContext)
  if (!ctx) throw new Error('useTheme must be used within a ThemeProvider')
  return ctx
}
