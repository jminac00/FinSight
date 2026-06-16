import { useEffect } from 'react'
import { Link, useParams } from 'react-router-dom'
import { ReportSkeleton } from '../components/report/ReportSkeleton'
import {
  ConclusionSection,
  DeepLearningSection,
  FundamentalSection,
  SentimentSection,
  TechnicalSection,
} from '../components/report/sections'
import { ArrowLeftIcon, RefreshIcon } from '../components/icons'
import { Button } from '../components/ui/Button'
import { StatusMessage } from '../components/ui/StatusMessage'
import { useReport } from '../hooks/useReport'
import { formatDateTime } from '../lib/format'
import { isValidTicker, normalizeTicker } from '../lib/ticker'
import type { ApiError } from '../api/client'

function BackLink() {
  return (
    <Link
      to="/"
      className="inline-flex items-center gap-1.5 text-sm text-ink-muted hover:text-accent"
    >
      <ArrowLeftIcon className="w-4" />
      Volver al inicio
    </Link>
  )
}

function LoadingState() {
  return (
    <div className="space-y-4">
      <StatusMessage tone="info" title="Generando el análisis…">
        Estamos consultando las cuatro fuentes de análisis. El proceso puede tardar hasta 60
        segundos. Si el servidor estaba inactivo, el primer arranque añade entre 30 y 60 segundos.
      </StatusMessage>
      <ReportSkeleton />
    </div>
  )
}

function ErrorState({ error, onRetry }: { error: ApiError; onRetry: () => void }) {
  if (error.status === 422) {
    return (
      <StatusMessage tone="danger" title="Símbolo no válido">
        <p>{error.message}</p>
        <p className="mt-2">
          <BackLink />
        </p>
      </StatusMessage>
    )
  }

  const isConnection = error.status === 0
  return (
    <StatusMessage
      tone="danger"
      title={isConnection ? 'No se pudo conectar' : 'No se pudo generar el análisis'}
      action={
        <Button variant="secondary" size="sm" onClick={onRetry}>
          <RefreshIcon className="w-4" />
          Reintentar
        </Button>
      }
    >
      <p>{error.message}</p>
      {isConnection ? (
        <p className="mt-2">
          El servidor puede estar arrancando tras un periodo de inactividad. Espera unos segundos y
          vuelve a intentarlo.
        </p>
      ) : null}
    </StatusMessage>
  )
}

export default function ReportPage() {
  const { ticker: rawTicker = '' } = useParams()
  const ticker = normalizeTicker(rawTicker)
  const valid = isValidTicker(ticker)
  const { data, loading, error, refresh } = useReport(ticker)

  useEffect(() => {
    document.title = `${ticker} — Análisis · FinSight`
  }, [ticker])

  return (
    <div className="mx-auto max-w-3xl px-4 py-8 sm:px-6">
      <div className="mb-6 flex flex-wrap items-center justify-between gap-3">
        <div>
          <BackLink />
          <h1 className="mt-2 text-3xl font-semibold tracking-tight text-ink">
            Análisis de <span className="text-accent">{ticker}</span>
          </h1>
          {data ? (
            <p className="mt-1 text-sm text-ink-subtle">
              Generado el {formatDateTime(data.generated_at)}
            </p>
          ) : null}
        </div>
        {data ? (
          <Button variant="secondary" size="sm" onClick={refresh} disabled={loading}>
            <RefreshIcon className="w-4" />
            Actualizar
          </Button>
        ) : null}
      </div>

      {!valid ? (
        <StatusMessage tone="danger" title="Símbolo no válido">
          <p>Usa un símbolo de 2 a 5 caracteres alfanuméricos (por ejemplo, AAPL).</p>
          <p className="mt-2">
            <BackLink />
          </p>
        </StatusMessage>
      ) : loading ? (
        <LoadingState />
      ) : error ? (
        <ErrorState error={error} onRetry={refresh} />
      ) : data ? (
        <div className="space-y-4">
          {data.partial_support || !data.deep_learning ? (
            <StatusMessage tone="warning" title="Soporte parcial" live="off">
              No hay un modelo de predicción entrenado para {ticker}. Se muestran los análisis de
              sentimiento, fundamental y técnico; la predicción de tendencia no está disponible.
            </StatusMessage>
          ) : null}

          {data.sentiment ? <SentimentSection data={data.sentiment} /> : null}
          {data.deep_learning ? <DeepLearningSection data={data.deep_learning} /> : null}
          {data.fundamental ? <FundamentalSection data={data.fundamental} /> : null}
          {data.technical ? <TechnicalSection data={data.technical} /> : null}
          <ConclusionSection conclusion={data.global_conclusion} />
        </div>
      ) : null}
    </div>
  )
}
