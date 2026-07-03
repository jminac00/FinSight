import type {
  DLResult,
  FundamentalResult,
  SentimentResult,
  TechnicalResult,
} from '../../api/types'
import {
  formatCurrencyUsd,
  formatDate,
  formatNumber,
  formatRatioAsPercent,
  formatSignedPercent,
} from '../../lib/format'
import { LegalDisclaimer } from '../LegalDisclaimer'
import { Badge } from '../ui/Badge'
import { Card } from '../ui/Card'
import { ExternalLink } from '../ui/ExternalLink'
import { AiBadge } from './AiBadge'
import { BlockScores } from './charts/BlockScores'
import { ReturnProjection } from './charts/ReturnProjection'
import { ScoreDial } from './charts/ScoreDial'
import { SentimentScale } from './charts/SentimentScale'
import { MetricInfo } from './MetricInfo'
import { MetricTable } from './MetricTable'
import { ReportSection } from './ReportSection'
import { Stat } from './Stat'
import { TrendDisplay } from './TrendDisplay'

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null
}

/** Collect primitive leaf values from a (possibly nested) object, keyed by leaf name. */
function flattenLeaves(value: unknown): Record<string, unknown> {
  const record = asRecord(value)
  if (!record) return {}
  const out: Record<string, unknown> = {}
  for (const [key, child] of Object.entries(record)) {
    const nested = asRecord(child)
    if (nested) Object.assign(out, flattenLeaves(nested))
    else out[key] = child
  }
  return out
}

export function SentimentSection({ data }: { data: SentimentResult }) {
  return (
    <ReportSection id="sentiment" title="Análisis de sentimiento" badge={<AiBadge />}>
      <div className="flex flex-col gap-5">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <TrendDisplay value={data.label} kind="sentiment" />
          <div className="flex gap-8">
            <Stat
              label="Puntuación (−1 a 1)"
              value={formatNumber(data.score, 2)}
              info={<MetricInfo metricKey="score" />}
            />
            <Stat
              label="Confianza"
              value={formatRatioAsPercent(data.confidence)}
              info={<MetricInfo metricKey="confidence" />}
            />
          </div>
        </div>

        <SentimentScale value={data.score} />

        <p className="max-w-prose text-ink-muted">{data.explanation}</p>

        {data.influential_news.length > 0 ? (
          <div>
            <h3 className="mb-2 text-sm font-semibold text-ink">Noticias influyentes</h3>
            <ul className="space-y-2">
              {data.influential_news.map((item, index) => (
                <li key={`${item.url}-${index}`} className="text-sm">
                  <ExternalLink href={item.url}>{item.title}</ExternalLink>
                  <span className="text-ink-subtle"> · {item.source}</span>
                </li>
              ))}
            </ul>
          </div>
        ) : null}
      </div>
    </ReportSection>
  )
}

export function DeepLearningSection({ data }: { data: DLResult }) {
  const returnTone =
    data.predicted_return_pct > 0
      ? 'text-success-fg'
      : data.predicted_return_pct < 0
        ? 'text-danger-fg'
        : 'text-ink'

  return (
    <ReportSection
      id="dl"
      title="Predicción de tendencia"
      badge={<Badge tone="info">Modelo GRU</Badge>}
    >
      <div className="flex flex-col gap-6">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <TrendDisplay value={data.trend} kind="trend" />
          <Stat
            label={`Retorno estimado a ${data.horizon_days} días`}
            value={formatSignedPercent(data.predicted_return_pct)}
            valueClassName={returnTone}
          />
        </div>

        <ReturnProjection
          returnPct={data.predicted_return_pct}
          horizonDays={data.horizon_days}
        />

        <div className="grid grid-cols-2 gap-x-8 gap-y-4 sm:grid-cols-3">
          <Stat label="Precio actual" value={formatCurrencyUsd(data.current_price)} />
          <Stat
            label="Precio estimado"
            value={formatCurrencyUsd(data.predicted_price)}
            hint={`Horizonte de ${data.horizon_days} días bursátiles`}
          />
          <Stat label="Último entrenamiento" value={formatDate(data.trained_at)} />
        </div>

        <div>
          <h3 className="mb-2 text-sm font-semibold text-ink">Calidad del modelo</h3>
          <div className="grid grid-cols-3 gap-x-8 gap-y-4">
            <Stat
              label="RMSE"
              value={formatNumber(data.metrics.rmse, 2)}
              info={<MetricInfo metricKey="rmse" />}
            />
            <Stat
              label="MAE"
              value={formatNumber(data.metrics.mae, 2)}
              info={<MetricInfo metricKey="mae" />}
            />
            <Stat
              label="Acierto direccional"
              value={formatRatioAsPercent(data.metrics.directional_accuracy)}
              info={<MetricInfo metricKey="directional_accuracy" />}
            />
          </div>
        </div>
      </div>
    </ReportSection>
  )
}

export function FundamentalSection({ data }: { data: FundamentalResult }) {
  return (
    <ReportSection id="fundamental" title="Análisis fundamental" badge={<AiBadge />}>
      <div className="flex flex-col gap-5">
        <div className="flex flex-wrap items-center gap-6">
          <ScoreDial value={data.score} label="Puntuación fundamental" />
          <p className="max-w-prose flex-1 text-ink-muted">{data.llm_analysis}</p>
        </div>
        <MetricTable
          data={asRecord(data.metrics.ratios) ?? data.metrics}
          caption="Ratios fundamentales"
        />
      </div>
    </ReportSection>
  )
}

export function TechnicalSection({ data }: { data: TechnicalResult }) {
  return (
    <ReportSection id="technical" title="Análisis técnico" badge={<AiBadge />}>
      <div className="flex flex-col gap-6">
        <div className="flex flex-wrap items-center gap-6">
          <ScoreDial value={data.score} label="Puntuación técnica" />
          <div className="flex-1">
            <p className="mb-1 text-sm font-medium text-ink-subtle">
              Señal técnica
              <MetricInfo metricKey="trend" />
            </p>
            <TrendDisplay value={data.signal} kind="trend" />
            <p className="mt-3 max-w-prose text-ink-muted">{data.llm_analysis}</p>
          </div>
        </div>
        <div>
          <h3 className="mb-3 text-sm font-semibold text-ink">Bloques técnicos</h3>
          <BlockScores blocks={data.block_scores} />
        </div>
        <MetricTable
          data={flattenLeaves(asRecord(data.indicators.blocks) ?? data.indicators)}
          caption="Indicadores técnicos"
        />
      </div>
    </ReportSection>
  )
}

export function ConclusionSection({ conclusion }: { conclusion: string }) {
  return (
    <Card>
      <section
        aria-labelledby="conclusion-heading"
        className="rounded-lg border border-accent-subtle bg-accent-subtle p-5 sm:p-6"
      >
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
          <h2 id="conclusion-heading" className="text-lg font-semibold text-ink">
            Conclusión global
          </h2>
          <AiBadge />
        </div>
        <p className="max-w-prose text-ink">{conclusion}</p>
        <div className="mt-6 rounded-md border border-border bg-surface p-4">
          <LegalDisclaimer />
        </div>
      </section>
    </Card>
  )
}
