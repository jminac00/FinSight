import { useEffect } from 'react'
import { SearchBar } from '../components/SearchBar'

const MODULES = [
  {
    name: 'Sentimiento de mercado',
    description:
      'Analiza las noticias más recientes de la empresa y resume el tono del mercado, con enlaces a las fuentes originales.',
  },
  {
    name: 'Predicción de tendencia',
    description:
      'Un modelo de aprendizaje profundo estima la tendencia del precio a medio plazo para las acciones del Nasdaq con soporte completo.',
  },
  {
    name: 'Análisis fundamental',
    description:
      'Interpreta los datos financieros de la empresa en lenguaje claro, sin necesidad de conocimientos previos.',
  },
  {
    name: 'Análisis técnico',
    description:
      'Calcula indicadores de precio y volumen para determinar una señal de tendencia alcista, bajista o neutral.',
  },
]

export default function HomePage() {
  useEffect(() => {
    document.title = 'FinSight — Análisis Financiero'
  }, [])

  return (
    <div className="mx-auto max-w-5xl px-4 sm:px-6">
      <section className="flex flex-col items-start gap-6 py-16 sm:py-24">
        <h1 className="max-w-3xl text-4xl font-semibold tracking-tight text-ink sm:text-5xl">
          Entiende cualquier acción en un solo informe
        </h1>
        <p className="max-w-prose text-lg text-ink-muted">
          FinSight combina cuatro análisis complementarios y los resume en español, de forma clara
          para cualquier persona, con o sin conocimientos financieros.
        </p>
        <SearchBar />
      </section>

      <section aria-labelledby="report-includes" className="border-t border-border py-12 sm:py-16">
        <h2 id="report-includes" className="text-2xl font-semibold tracking-tight text-ink">
          Qué incluye el informe
        </h2>
        <dl className="mt-8 grid gap-x-10 gap-y-8 sm:grid-cols-2">
          {MODULES.map((module, index) => (
            <div key={module.name} className="flex gap-4">
              <span
                aria-hidden="true"
                className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-accent-subtle text-sm font-semibold text-accent"
              >
                {index + 1}
              </span>
              <div>
                <dt className="font-medium text-ink">{module.name}</dt>
                <dd className="mt-1 text-sm text-ink-muted">{module.description}</dd>
              </div>
            </div>
          ))}
        </dl>
        <p className="mt-10 max-w-prose text-sm text-ink-subtle">
          La predicción de tendencia solo está disponible para acciones del Nasdaq con modelo
          entrenado. El resto de valores recibe soporte parcial, con los tres análisis restantes.
        </p>
      </section>
    </div>
  )
}
