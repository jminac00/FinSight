import { cn } from '../../../lib/cn'
import { formatNumber } from '../../../lib/format'

/**
 * Number line for a sentiment score in [-1, 1] with the marker coloured by
 * sign. Decorative SVG (aria-hidden); the container carries the value via
 * role="img" + aria-label. Endpoint labels are visual reinforcement only.
 */
export function SentimentScale({ value, className }: { value: number; className?: string }) {
  const clamped = Math.max(-1, Math.min(1, value))
  const fraction = (clamped + 1) / 2
  const cx = 6 + fraction * 188
  const tone =
    clamped > 0.05 ? 'text-success' : clamped < -0.05 ? 'text-danger' : 'text-ink-subtle'

  return (
    <div className={cn('w-full max-w-sm', className)}>
      <div
        role="img"
        aria-label={`Sentimiento ${formatNumber(clamped, 2)} en una escala de -1 (negativo) a 1 (positivo)`}
      >
        <svg viewBox="0 0 200 24" className="w-full" aria-hidden="true">
          <line
            x1="6"
            y1="12"
            x2="194"
            y2="12"
            className="text-panel"
            stroke="currentColor"
            strokeWidth="4"
            strokeLinecap="round"
          />
          <line
            x1="100"
            y1="5"
            x2="100"
            y2="19"
            className="text-border-strong"
            stroke="currentColor"
            strokeWidth="2"
          />
          <circle cx={cx} cy="12" r="7" className={tone} fill="currentColor" />
          <circle
            cx={cx}
            cy="12"
            r="7"
            className="text-surface"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
          />
        </svg>
      </div>
      <div aria-hidden="true" className="mt-1 flex justify-between text-xs text-ink-subtle">
        <span>Negativo</span>
        <span>Neutral</span>
        <span>Positivo</span>
      </div>
    </div>
  )
}
