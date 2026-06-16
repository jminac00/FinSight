import { cn } from '../lib/cn'

/** FinSight wordmark with the chart-glyph icon. */
export function Wordmark({ className }: { className?: string }) {
  return (
    <span className={cn('inline-flex items-center gap-2 font-semibold tracking-tight', className)}>
      <svg viewBox="0 0 32 32" className="h-6 w-6" aria-hidden="true">
        <rect width="32" height="32" rx="7" className="fill-accent" />
        <path
          d="M8 21 L13.5 14.5 L18 18 L24 10"
          fill="none"
          stroke="var(--color-accent-fg)"
          strokeWidth="2.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <circle cx="24" cy="10" r="2.2" fill="var(--color-accent-fg)" />
      </svg>
      <span>
        Fin<span className="text-accent">Sight</span>
      </span>
    </span>
  )
}
