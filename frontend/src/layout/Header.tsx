import { Link } from 'react-router-dom'
import { Wordmark } from '../components/Wordmark'
import { ThemeToggle } from '../theme/ThemeToggle'

export function Header() {
  return (
    <header className="sticky top-0 z-sticky border-b border-border bg-bg/85 backdrop-blur">
      <div className="mx-auto flex h-16 max-w-5xl items-center justify-between gap-4 px-4 sm:px-6">
        <Link
          to="/"
          className="rounded-md text-lg text-ink hover:text-accent"
          aria-label="FinSight — ir al inicio"
        >
          <Wordmark />
        </Link>
        <ThemeToggle />
      </div>
    </header>
  )
}
