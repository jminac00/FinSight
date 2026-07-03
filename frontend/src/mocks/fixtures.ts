import type { ReportResponse } from '../api/types'

const DISCLAIMER =
  'AVISO LEGAL: El contenido de este informe es de carácter exclusivamente informativo y ha ' +
  'sido generado mediante sistemas de inteligencia artificial. No constituye asesoramiento ' +
  'financiero, recomendación de inversión ni oferta de compra o venta de valores.'

/** Full report (4 modules + conclusion) for development and tests. */
export function fullReport(ticker: string): ReportResponse {
  const currentPrice = 182.3
  const predictedReturnPct = 4.82
  return {
    ticker,
    generated_at: new Date().toISOString(),
    sentiment: {
      label: 'positivo',
      score: 0.62,
      confidence: 0.81,
      explanation:
        `Las noticias recientes sobre ${ticker} reflejan un tono mayoritariamente positivo, ` +
        'impulsado por unos resultados trimestrales por encima de lo esperado y una previsión de ' +
        'ingresos sólida para el próximo trimestre.',
      influential_news: [
        {
          title: `${ticker} supera las expectativas de beneficios del tercer trimestre`,
          url: 'https://example.com/earnings',
          source: 'Reuters',
        },
        {
          title: `Los analistas elevan el precio objetivo de ${ticker}`,
          url: 'https://example.com/price-target',
          source: 'Bloomberg',
        },
        {
          title: `Fuerte demanda en el último lanzamiento de ${ticker}`,
          url: 'https://example.com/demand',
          source: 'CNBC',
        },
      ],
    },
    deep_learning: {
      trend: 'alcista',
      predicted_return_pct: predictedReturnPct,
      predicted_price: Number((currentPrice * (1 + predictedReturnPct / 100)).toFixed(2)),
      current_price: currentPrice,
      horizon_days: 10,
      trained_at: new Date(Date.now() - 86_400_000).toISOString(),
      metrics: { rmse: 3.12, mae: 2.45, directional_accuracy: 0.63 },
    },
    fundamental: {
      score: 7.8,
      metrics: {
        per: 28.5,
        roe: 0.31,
        ev_ebitda: 21.3,
        net_margin: 0.25,
        free_cash_flow: '99,6 B$',
      },
      llm_analysis:
        `${ticker} presenta una situación financiera sólida, con márgenes superiores a la media ` +
        'del sector y una generación de caja consistente. El nivel de endeudamiento es moderado y ' +
        'la rentabilidad sobre recursos propios se mantiene elevada.',
      cached_at: new Date().toISOString(),
    },
    technical: {
      score: 6.4,
      signal: 'alcista',
      block_scores: {
        momentum: 7.1,
        trend: 6.8,
        risk_stability: 5.4,
        confirmation: 6.2,
      },
      indicators: {
        rsi_14: 58.3,
        macd: 1.24,
        sma_50: 175.8,
        sma_200: 168.4,
        bollinger_upper: 189.2,
      },
      llm_analysis:
        `El análisis técnico de ${ticker} muestra señales moderadamente alcistas: el precio se ` +
        'mantiene por encima de sus medias móviles de 50 y 200 sesiones, con un RSI en zona neutral.',
      calculated_at: new Date().toISOString(),
    },
    global_conclusion:
      `El análisis consolidado de ${ticker} refleja una perspectiva generalmente positiva. El ` +
      'sentimiento de mercado es favorable, los fundamentales son robustos y tanto el modelo de ' +
      'predicción como el análisis técnico apuntan a una tendencia alcista a medio plazo. Recuerda ' +
      'que se trata de información orientativa y no de una recomendación de inversión.',
    disclaimer: DISCLAIMER,
    partial_support: false,
  }
}

/** Partial-support report (no GRU model) — RF-27. */
export function partialReport(ticker: string): ReportResponse {
  const base = fullReport(ticker)
  return { ...base, deep_learning: null, partial_support: true }
}
