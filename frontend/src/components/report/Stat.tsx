import type { ReactNode } from 'react'
import { cn } from '../../lib/cn'

/** Label-over-value statistic. Plain markup so it is valid in any container. */
export function Stat({
  label,
  value,
  hint,
  valueClassName,
  info,
}: {
  label: string
  value: ReactNode
  hint?: string
  valueClassName?: string
  info?: ReactNode
}) {
  return (
    <div>
      <p className="text-sm text-ink-muted">
        {label}
        {info}
      </p>
      <p className={cn('mt-1 text-lg font-semibold tabular-nums text-ink', valueClassName)}>
        {value}
      </p>
      {hint ? <p className="mt-0.5 text-xs text-ink-subtle">{hint}</p> : null}
    </div>
  )
}
