import { cn } from '../../../lib/cn'
import { formatSignedPercent } from '../../../lib/format'

/** Full-scale magnitude in percent that maps to the edge of the bar. */
const DOMAIN = 15

/**
 * Diverging bar for a predicted return in percent. The centre line is the
 * current price (0 %); the bar grows right (up) or left (down). Decorative SVG
 * (aria-hidden); the container carries the value via role="img" + aria-label.
 */
export function ReturnProjection({
  returnPct,
  horizonDays,
  className,
}: {
  returnPct: number
  horizonDays: number
  className?: string
}) {
  const fraction = Math.max(-1, Math.min(1, returnPct / DOMAIN))
  const half = 94
  const width = Math.abs(fraction) * half
  const positive = returnPct >= 0
  const tone = positive ? 'text-success' : 'text-danger'

  return (
    <div className={cn('w-full max-w-sm', className)}>
      <div
        role="img"
        aria-label={`Retorno estimado a ${horizonDays} días: ${formatSignedPercent(returnPct)}`}
      >
        <svg viewBox="0 0 200 26" className="w-full" aria-hidden="true">
          <rect
            x={positive ? 100 : 100 - width}
            y="8"
            width={width}
            height="10"
            rx="3"
            className={tone}
            fill="currentColor"
          />
          <line
            x1="100"
            y1="3"
            x2="100"
            y2="23"
            className="text-border-strong"
            stroke="currentColor"
            strokeWidth="2"
          />
        </svg>
      </div>
      <div aria-hidden="true" className="mt-1 flex justify-between text-xs text-ink-subtle">
        <span>Bajada</span>
        <span>Sin cambio</span>
        <span>Subida</span>
      </div>
    </div>
  )
}
