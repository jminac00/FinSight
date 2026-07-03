import type { ReportResponse } from '../api/types'

const DISCLAIMER =
  'AVISO LEGAL: El contenido de este informe es de carácter exclusivamente informativo y ha ' +
  'sido generado mediante sistemas de inteligencia artificial. No constituye asesoramiento ' +
  'financiero, recomendación de inversión ni oferta de compra o venta de valores.'

/** Full report (4 modules + conclusion) for development and tests. */
export function fullReport(ticker: string, companyName: string | null = null): ReportResponse {
  const currentPrice = 182.3
  const predictedReturnPct = 4.82
  return {
    ticker,
    company_name: companyName,
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
      // Detail bag mirrors the real backend: flat metadata plus nested objects.
      metrics: {
        universe: 'sp500',
        mode: 'auto',
        sector: 'Technology',
        ratios: {
          ebitda_yield: 0.075,
          earnings_yield: 0.037,
          fcf_yield: 0.031,
          roe: 0.312,
          roa: 0.081,
          roce: 0.223,
          operating_margin: 0.233,
          gp_a: 0.263,
          dn_ebitda: 1.17,
          current_ratio: 1.59,
          debt_to_equity: 3.96,
        },
        scores: { valoracion: 2.6, calidad: 9.9, crecimiento: 6.4, solvencia: 7.1 },
        sub_signals: { valoracion: 'sobrevalorada', calidad: 'alta calidad' },
        degradation: { valoracion: { indicators_total: 4, indicators_available: 4 } },
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
      // Detail bag mirrors the real backend: metadata plus per-block nested indicators.
      indicators: {
        universe: 'sp500',
        data_completeness: 1.0,
        score_reliable: true,
        weights: { momentum: 0.35, trend: 0.3, risk_stability: 0.2, confirmation: 0.15 },
        blocks: {
          // Each block mixes curated indicators with internal pipeline artifacts
          // (z_score, normalization_method, raw scores) that are not shown to the user.
          momentum: {
            momentum_12_1: 0.42,
            momentum_6_1: 0.18,
            vol_12m: 0.29,
            z_score: 0.83,
            normalization_method: 'z-score',
          },
          trend: {
            price: currentPrice,
            ma_200: 168.4,
            distance_to_ma200: 0.083,
            regression_r2: 0.71,
          },
          risk_stability: {
            max_drawdown_126d: -0.14,
            downside_volatility_126d: 0.19,
            s_dd_z_score: -0.4,
          },
          confirmation: {
            current_volume: 58_200_000,
            average_volume_20: 51_400_000,
            relative_volume_20: 1.13,
            donchian_position: 0.72,
            raw_momentum_score: 0.5,
          },
        },
        errors: {},
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
    missing_modules: [],
  }
}

/** Partial-support report (no GRU model) — RF-27. */
export function partialReport(ticker: string, companyName: string | null = null): ReportResponse {
  const base = fullReport(ticker, companyName)
  return { ...base, deep_learning: null, partial_support: true, missing_modules: ['deep_learning'] }
}
