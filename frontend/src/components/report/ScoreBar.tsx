import { cn } from '../../lib/cn'

/**
 * Decorative progress track (aria-hidden). The caller always renders the numeric
 * value as visible text, which is the accessible source of truth.
 */
export function ScoreBar({
  value,
  min = 0,
  max = 10,
  tone = 'accent',
  className,
}: {
  value: number
  min?: number
  max?: number
  tone?: 'accent' | 'success' | 'danger' | 'neutral'
  className?: string
}) {
  const pct = Math.max(0, Math.min(1, (value - min) / (max - min))) * 100
  const fill = {
    accent: 'bg-accent',
    success: 'bg-success',
    danger: 'bg-danger',
    neutral: 'bg-ink-subtle',
  }[tone]

  return (
    <div
      aria-hidden="true"
      className={cn('h-1.5 w-full overflow-hidden rounded-full bg-panel', className)}
    >
      <div className={cn('h-full rounded-full', fill)} style={{ width: `${pct}%` }} />
    </div>
  )
}
