/**
 * Plain-language explanations of the metrics shown in the report, for the
 * non-expert audience. Keyed by the metric key used in the backend payloads
 * (and by a few concept keys used in the module headers). Missing keys simply
 * render no info affordance.
 */
export type GlossaryEntry = { term: string; description: string }

export const GLOSSARY: Record<string, GlossaryEntry> = {
  // Deep learning (GRU)
  rmse: {
    term: 'RMSE',
    description:
      'Error cuadrático medio: mide cuánto se desvían de media las predicciones del modelo. Cuanto más bajo, más preciso.',
  },
  mae: {
    term: 'MAE',
    description:
      'Error absoluto medio: desviación media de las predicciones en las mismas unidades que el dato. Cuanto más bajo, mejor.',
  },
  directional_accuracy: {
    term: 'Acierto direccional',
    description:
      'Porcentaje de veces que el modelo acierta la dirección del movimiento (subida o bajada), al margen de la magnitud.',
  },
  trend: {
    term: 'Tendencia',
    description: 'Dirección esperada del precio a medio plazo: alcista, bajista o neutral.',
  },

  // Sentiment
  score: {
    term: 'Puntuación de sentimiento',
    description: 'Tono de las noticias en una escala de -1 (muy negativo) a 1 (muy positivo).',
  },
  confidence: {
    term: 'Confianza',
    description: 'Seguridad del modelo en su valoración, de 0 % a 100 %.',
  },

  // Fundamental
  per: {
    term: 'PER',
    description:
      'Precio/beneficio: cuántas veces el beneficio anual está contenido en el precio de la acción.',
  },
  roe: {
    term: 'ROE',
    description: 'Rentabilidad sobre recursos propios: beneficio generado por cada euro de fondos propios.',
  },
  ev_ebitda: {
    term: 'EV/EBITDA',
    description: 'Valor de la empresa frente a su beneficio operativo bruto (antes de intereses, impuestos y amortizaciones).',
  },
  net_margin: {
    term: 'Margen neto',
    description: 'Porcentaje de los ingresos que se convierte en beneficio después de todos los gastos.',
  },
  free_cash_flow: {
    term: 'Flujo de caja libre',
    description: 'Efectivo que genera la empresa tras cubrir sus inversiones necesarias.',
  },

  // Technical
  rsi_14: {
    term: 'RSI (14)',
    description: 'Índice de fuerza relativa: señala si la acción está sobrecomprada (por encima de 70) o sobrevendida (por debajo de 30).',
  },
  macd: {
    term: 'MACD',
    description: 'Diferencia entre dos medias móviles del precio; ayuda a detectar cambios de tendencia.',
  },
  sma_50: {
    term: 'Media móvil de 50',
    description: 'Precio medio de las últimas 50 sesiones; suaviza el ruido a corto plazo.',
  },
  sma_200: {
    term: 'Media móvil de 200',
    description: 'Precio medio de las últimas 200 sesiones; referencia de la tendencia a largo plazo.',
  },
  bollinger_upper: {
    term: 'Banda de Bollinger superior',
    description: 'Nivel de precio alto relativo según la volatilidad reciente; tocarlo sugiere que el precio está caro a corto plazo.',
  },
}
