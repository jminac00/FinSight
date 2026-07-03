import { formatNumber, humanizeKey } from '../../lib/format'
import { MetricInfo } from './MetricInfo'

/**
 * Format a metric value for display, or return null for values that are not
 * meaningfully renderable in a flat table (nested objects, arrays, null). The
 * backend detail bags mix flat values with nested structures, so anything that
 * is not a primitive is skipped rather than dumped raw.
 */
function displayValue(value: unknown): string | null {
  if (typeof value === 'number') return Number.isFinite(value) ? formatNumber(value) : null
  if (typeof value === 'string') return value.trim() === '' ? null : value
  if (typeof value === 'boolean') return value ? 'Sí' : 'No'
  return null
}

/** Renders a metrics/indicators dictionary as an accessible key–value grid. */
export function MetricTable({ data, caption }: { data: Record<string, unknown>; caption: string }) {
  const entries = Object.entries(data)
    .map(([key, value]) => [key, displayValue(value)] as const)
    .filter((entry): entry is readonly [string, string] => entry[1] !== null)

  if (entries.length === 0) return null

  return (
    <div>
      <p className="sr-only">{caption}</p>
      <dl className="grid grid-cols-2 gap-px overflow-hidden rounded-md border border-border bg-border sm:grid-cols-3">
        {entries.map(([key, value]) => (
          <div key={key} className="bg-surface p-3">
            <dt className="text-xs font-medium uppercase tracking-wide text-ink-subtle">
              {humanizeKey(key)}
              <MetricInfo metricKey={key} />
            </dt>
            <dd className="mt-1 font-semibold tabular-nums text-ink">{value}</dd>
          </div>
        ))}
      </dl>
    </div>
  )
}
