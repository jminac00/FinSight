import { cn } from '../../../lib/cn'
import { formatNumber } from '../../../lib/format'

type Tone = 'accent' | 'success' | 'warning' | 'danger'

const STROKE: Record<Tone, string> = {
  accent: 'text-accent',
  success: 'text-success',
  warning: 'text-warning',
  danger: 'text-danger',
}

/** Colour a 0–10 score by band. The numeric value stays the source of truth. */
function toneForScore(value: number, max: number): Tone {
  const ratio = value / max
  if (ratio >= 0.7) return 'success'
  if (ratio >= 0.4) return 'accent'
  return 'danger'
}

/** Point on a circle, measuring degrees counter-clockwise from the positive x-axis. */
function polar(cx: number, cy: number, r: number, deg: number): [number, number] {
  const rad = (deg * Math.PI) / 180
  return [cx + r * Math.cos(rad), cy - r * Math.sin(rad)]
}

/** Upper-semicircle arc path from startDeg to endDeg (180° = left, 0° = right). */
function arc(cx: number, cy: number, r: number, startDeg: number, endDeg: number): string {
  const [x1, y1] = polar(cx, cy, r, startDeg)
  const [x2, y2] = polar(cx, cy, r, endDeg)
  const largeArc = Math.abs(endDeg - startDeg) > 180 ? 1 : 0
  return `M ${x1} ${y1} A ${r} ${r} 0 ${largeArc} 1 ${x2} ${y2}`
}

/**
 * Semicircular gauge for a 0–max score. Decorative SVG (aria-hidden); the
 * container exposes the value via role="img" + aria-label, and the number is
 * also shown as text.
 */
export function ScoreDial({
  value,
  max = 10,
  label,
  tone,
  className,
}: {
  value: number
  max?: number
  label: string
  tone?: Tone
  className?: string
}) {
  const fraction = Math.max(0, Math.min(1, value / max))
  const end = 180 * (1 - fraction)
  const stroke = STROKE[tone ?? toneForScore(value, max)]
  const shown = formatNumber(value, 1)

  return (
    <div
      role="img"
      aria-label={`${label}: ${shown} sobre ${max}`}
      className={cn('inline-flex flex-col items-center', className)}
    >
      <div className="relative">
        <svg viewBox="0 0 120 66" className="w-32" aria-hidden="true">
          <path
            d={arc(60, 60, 52, 180, 0)}
            className="text-panel"
            fill="none"
            stroke="currentColor"
            strokeWidth="9"
            strokeLinecap="round"
          />
          <path
            d={arc(60, 60, 52, 180, end)}
            className={stroke}
            fill="none"
            stroke="currentColor"
            strokeWidth="9"
            strokeLinecap="round"
          />
        </svg>
        <div className="absolute inset-x-0 bottom-0 flex flex-col items-center">
          <span className="text-2xl font-semibold leading-none tabular-nums text-ink">
            {shown}
          </span>
          <span className="text-xs text-ink-subtle">/ {max}</span>
        </div>
      </div>
    </div>
  )
}
