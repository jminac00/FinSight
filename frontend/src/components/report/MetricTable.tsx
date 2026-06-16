import { formatNumber, humanizeKey } from '../../lib/format'

function renderValue(value: number | string): string {
  return typeof value === 'number' ? formatNumber(value) : value
}

/** Renders a metrics/indicators dictionary as an accessible key–value grid. */
export function MetricTable({
  data,
  caption,
}: {
  data: Record<string, number | string>
  caption: string
}) {
  const entries = Object.entries(data)
  if (entries.length === 0) return null

  return (
    <div>
      <p className="sr-only">{caption}</p>
      <dl className="grid grid-cols-2 gap-px overflow-hidden rounded-md border border-border bg-border sm:grid-cols-3">
        {entries.map(([key, value]) => (
          <div key={key} className="bg-surface p-3">
            <dt className="text-xs font-medium uppercase tracking-wide text-ink-subtle">
              {humanizeKey(key)}
            </dt>
            <dd className="mt-1 font-semibold tabular-nums text-ink">{renderValue(value)}</dd>
          </div>
        ))}
      </dl>
    </div>
  )
}
