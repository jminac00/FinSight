import { MoonIcon, SunIcon } from '../components/icons'
import { useTheme } from './ThemeContext'

export function ThemeToggle() {
  const { theme, toggle } = useTheme()
  const next = theme === 'dark' ? 'claro' : 'oscuro'

  return (
    <button
      type="button"
      onClick={toggle}
      className="inline-flex h-10 w-10 items-center justify-center rounded-md text-ink-muted transition-colors duration-150 ease-out-quart hover:bg-panel hover:text-ink"
      aria-label={`Cambiar a tema ${next}`}
      title={`Cambiar a tema ${next}`}
    >
      {theme === 'dark' ? <MoonIcon /> : <SunIcon />}
    </button>
  )
}
