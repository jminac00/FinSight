import { useEffect, useState } from 'react'
import { useNavigate, useParams, useSearchParams } from 'react-router-dom'
import Disclaimer from '../components/Disclaimer'
import ErrorMessage from '../components/ErrorMessage'
import LoadingSpinner from '../components/LoadingSpinner'
import { fetchReport } from '../services/api'

function SentimentSection({ data }) {
  if (!data) return <p className="text-gray-400 italic">Módulo no disponible</p>
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-3">
        <span className="font-semibold capitalize">{data.label}</span>
        <span className="text-sm text-gray-500">Puntuación: {data.score.toFixed(2)}</span>
        <span className="text-sm text-gray-500">Confianza: {(data.confidence * 100).toFixed(0)}%</span>
      </div>
      <p className="text-sm text-gray-700">{data.explanation}</p>
      {data.influential_news.length > 0 && (
        <ul className="mt-2 space-y-1">
          {data.influential_news.map((item, i) => (
            <li key={i} className="text-xs">
              <a
                href={item.url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-600 hover:underline"
              >
                {item.title}
              </a>{' '}
              <span className="text-gray-400">— {item.source}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

function DLSection({ data }) {
  if (!data) return <p className="text-gray-400 italic">Módulo no disponible para esta acción</p>
  return (
    <div className="space-y-2 text-sm">
      <div className="flex gap-4">
        <span className="font-semibold capitalize">Tendencia: {data.trend}</span>
        <span>Precio actual: ${data.current_price.toFixed(2)}</span>
        <span>Precio predicho ({data.horizon_days}d): ${data.predicted_price.toFixed(2)}</span>
        <span className={data.pct_change >= 0 ? 'text-green-600' : 'text-red-600'}>
          {data.pct_change >= 0 ? '+' : ''}{data.pct_change.toFixed(2)}%
        </span>
      </div>
      <div className="flex gap-4 text-gray-500 text-xs">
        <span>RMSE: {data.metrics.rmse}</span>
        <span>MAE: {data.metrics.mae}</span>
        <span>MAPE: {data.metrics.mape}%</span>
        <span>R²: {data.metrics.r2}</span>
        <span>Entrenado: {new Date(data.trained_at).toLocaleDateString('es-ES')}</span>
      </div>
    </div>
  )
}

function FundamentalSection({ data }) {
  if (!data) return <p className="text-gray-400 italic">Módulo no disponible</p>
  return (
    <div className="space-y-2 text-sm">
      <p className="font-semibold">Puntuación fundamental: {data.score.toFixed(1)} / 10</p>
      <p className="text-gray-700">{data.llm_analysis}</p>
      <div className="grid grid-cols-3 gap-2 mt-2">
        {Object.entries(data.metrics).map(([key, val]) => (
          <div key={key} className="bg-gray-50 rounded p-2">
            <span className="text-xs text-gray-500 uppercase">{key.replace(/_/g, ' ')}</span>
            <p className="font-medium">{typeof val === 'number' ? val.toLocaleString('es-ES') : String(val)}</p>
          </div>
        ))}
      </div>
    </div>
  )
}

function TechnicalSection({ data }) {
  if (!data) return <p className="text-gray-400 italic">Módulo no disponible</p>
  return (
    <div className="space-y-2 text-sm">
      <p className="font-semibold">Puntuación técnica: {data.score.toFixed(1)} / 10</p>
      <p className="text-gray-700">{data.llm_analysis}</p>
      <div className="grid grid-cols-3 gap-2 mt-2">
        {Object.entries(data.indicators).map(([key, val]) => (
          <div key={key} className="bg-gray-50 rounded p-2">
            <span className="text-xs text-gray-500 uppercase">{key.replace(/_/g, ' ')}</span>
            <p className="font-medium">{typeof val === 'number' ? val.toFixed(2) : String(val)}</p>
          </div>
        ))}
      </div>
    </div>
  )
}

export default function ReportPage() {
  const { ticker } = useParams()
  const [searchParams] = useSearchParams()
  const navigate = useNavigate()
  const [report, setReport] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    const forceRefresh = searchParams.get('force_refresh') === 'true'
    setLoading(true)
    setError(null)
    fetchReport(ticker, forceRefresh)
      .then(setReport)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false))
  }, [ticker, searchParams])

  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b px-6 py-3 flex items-center justify-between">
        <button
          onClick={() => navigate('/')}
          className="text-blue-600 hover:underline text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 rounded"
        >
          ← Volver
        </button>
        <h1 className="text-lg font-bold text-gray-900">
          FinSight — Análisis de <span className="text-blue-600">{ticker}</span>
        </h1>
        <a
          href={`/report/${ticker}?force_refresh=true`}
          className="text-xs text-gray-500 hover:text-gray-800 focus:outline-none focus:ring-2 focus:ring-blue-500 rounded"
        >
          Actualizar
        </a>
      </header>

      <main className="flex-1 px-4 py-6 max-w-4xl mx-auto w-full">
        {loading && <LoadingSpinner />}
        {error && <ErrorMessage message={error} />}
        {report && (
          <article aria-label={`Informe de análisis de ${ticker}`} className="space-y-6">
            <p className="text-xs text-gray-400">
              Generado el {new Date(report.generated_at).toLocaleString('es-ES')}
              {report.partial_support && (
                <span className="ml-2 text-amber-600">(soporte parcial — predicción DL no disponible)</span>
              )}
            </p>

            <section aria-labelledby="sentiment-heading" className="bg-white border rounded-xl p-5 shadow-sm">
              <h2 id="sentiment-heading" className="text-base font-semibold text-gray-800 mb-3">
                Análisis de Sentimiento
              </h2>
              <SentimentSection data={report.sentiment} />
            </section>

            <section aria-labelledby="dl-heading" className="bg-white border rounded-xl p-5 shadow-sm">
              <h2 id="dl-heading" className="text-base font-semibold text-gray-800 mb-3">
                Predicción Deep Learning (LSTM)
              </h2>
              <DLSection data={report.deep_learning} />
            </section>

            <section aria-labelledby="fundamental-heading" className="bg-white border rounded-xl p-5 shadow-sm">
              <h2 id="fundamental-heading" className="text-base font-semibold text-gray-800 mb-3">
                Análisis Fundamental
              </h2>
              <FundamentalSection data={report.fundamental} />
            </section>

            <section aria-labelledby="technical-heading" className="bg-white border rounded-xl p-5 shadow-sm">
              <h2 id="technical-heading" className="text-base font-semibold text-gray-800 mb-3">
                Análisis Técnico
              </h2>
              <TechnicalSection data={report.technical} />
            </section>

            <section aria-labelledby="conclusion-heading" className="bg-blue-50 border border-blue-200 rounded-xl p-5 shadow-sm">
              <h2 id="conclusion-heading" className="text-base font-semibold text-blue-900 mb-3">
                Conclusión Global
              </h2>
              <p className="text-sm text-blue-900">{report.global_conclusion}</p>
              <p className="mt-4 text-xs text-gray-500 italic">{report.disclaimer}</p>
            </section>
          </article>
        )}
      </main>

      <footer>
        <Disclaimer />
      </footer>
    </div>
  )
}
