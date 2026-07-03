import type { ReactNode } from 'react'
import type { ReportResponse } from '../../api/types'
import { formatNumber, formatSignedPercent } from '../../lib/format'
import { TrendDisplay } from './TrendDisplay'

type Item = {
  id: string
  name: string
  verdict: ReactNode
  sub: string | null
}

function buildItems(data: ReportResponse): Item[] {
  const { sentiment, deep_learning: dl, fundamental, technical } = data
  return [
    {
      id: 'sentiment',
      name: 'Sentimiento',
      verdict: sentiment ? (
        <TrendDisplay value={sentiment.label} kind="sentiment" size="sm" />
      ) : null,
      sub: sentiment ? `Score ${formatNumber(sentiment.score, 2)}` : null,
    },
    {
      id: 'dl',
      name: 'Tendencia',
      verdict: dl ? <TrendDisplay value={dl.trend} kind="trend" size="sm" /> : null,
      sub: dl ? formatSignedPercent(dl.predicted_return_pct) : null,
    },
    {
      id: 'fundamental',
      name: 'Fundamental',
      verdict: fundamental ? (
        <span className="text-base font-semibold tabular-nums text-ink">
          {formatNumber(fundamental.score, 1)} / 10
        </span>
      ) : null,
      sub: fundamental ? 'Puntuación' : null,
    },
    {
      id: 'technical',
      name: 'Técnico',
      verdict: technical ? <TrendDisplay value={technical.signal} kind="trend" size="sm" /> : null,
      sub: technical ? `${formatNumber(technical.score, 1)} / 10` : null,
    },
  ]
}

function Cell({ item }: { item: Item }) {
  const content = (
    <>
      <span className="text-sm text-ink-muted">{item.name}</span>
      <span className="mt-2 block">
        {item.verdict ?? <span className="text-ink-subtle">No disponible</span>}
      </span>
      {item.sub ? (
        <span className="mt-1 block text-xs tabular-nums text-ink-subtle">{item.sub}</span>
      ) : null}
    </>
  )

  const base = 'flex h-full flex-col rounded-lg border border-border bg-surface p-4'
  if (!item.verdict) {
    return <div className={base}>{content}</div>
  }
  return (
    <a
      href={`#${item.id}`}
      className={`${base} transition-colors hover:border-border-strong hover:bg-surface-raised`}
    >
      {content}
    </a>
  )
}

/** Scannable summary of the four modules that also links to each section. */
export function ReportOverview({ data }: { data: ReportResponse }) {
  return (
    <nav aria-label="Resumen del informe">
      <ul className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {buildItems(data).map((item) => (
          <li key={item.id}>
            <Cell item={item} />
          </li>
        ))}
      </ul>
    </nav>
  )
}
