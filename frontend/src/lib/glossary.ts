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

  // Fundamental (real backend ratio keys)
  ebitda_yield: {
    term: 'Rentabilidad sobre EBITDA',
    description:
      'EBITDA dividido entre el valor de la empresa: cuánto beneficio operativo bruto genera por cada euro invertido.',
  },
  earnings_yield: {
    term: 'Rentabilidad por beneficio',
    description:
      'Beneficio dividido entre el valor de la empresa (la inversa del PER): permite comparar la rentabilidad frente a otras inversiones.',
  },
  fcf_yield: {
    term: 'Rentabilidad por caja libre',
    description:
      'Flujo de caja libre dividido entre el valor de la empresa: efectivo generado tras las inversiones necesarias, en relación con el precio.',
  },
  roe: {
    term: 'ROE',
    description:
      'Rentabilidad sobre recursos propios: beneficio generado por cada euro de fondos propios.',
  },
  roa: {
    term: 'ROA',
    description: 'Rentabilidad sobre activos: beneficio generado por cada euro de activos totales.',
  },
  roce: {
    term: 'ROCE',
    description:
      'Rentabilidad sobre el capital empleado: beneficio generado en relación con el capital invertido (deuda y fondos propios).',
  },
  operating_margin: {
    term: 'Margen operativo',
    description:
      'Porcentaje de los ingresos que queda como beneficio operativo, antes de intereses e impuestos.',
  },
  gp_a: {
    term: 'Margen bruto sobre activos',
    description:
      'Beneficio bruto dividido entre los activos totales: eficiencia de la empresa generando margen bruto con sus activos.',
  },
  dn_ebitda: {
    term: 'Deuda neta / EBITDA',
    description:
      'Deuda neta dividida entre el EBITDA: cuántos años de beneficio operativo bruto haría falta para pagar la deuda neta.',
  },
  current_ratio: {
    term: 'Ratio de liquidez',
    description:
      'Activo corriente dividido entre pasivo corriente: capacidad de la empresa para afrontar sus deudas a corto plazo.',
  },
  debt_to_equity: {
    term: 'Deuda sobre fondos propios',
    description:
      'Deuda total dividida entre los fondos propios: nivel de endeudamiento en relación con el capital de la empresa.',
  },

  // Technical (curated real backend indicator keys — see lib/technicalIndicators.ts)
  momentum_12_1: {
    term: 'Momentum 12-1',
    description:
      'Rentabilidad del precio en los últimos 12 meses, sin contar el último mes. Positivo indica que la acción ha subido en ese periodo.',
  },
  momentum_6_1: {
    term: 'Momentum 6-1',
    description: 'Rentabilidad del precio en los últimos 6 meses, sin contar el último mes.',
  },
  vol_12m: {
    term: 'Volatilidad a 12 meses',
    description:
      'Volatilidad anualizada del precio en el último año: cuánto oscila la cotización. Cuanto más alta, más riesgo a corto plazo.',
  },
  price: {
    term: 'Precio',
    description: 'Último precio de cierre disponible de la acción.',
  },
  ma_200: {
    term: 'Media móvil de 200',
    description:
      'Precio medio de las últimas 200 sesiones (aproximadamente un año), usada como referencia de tendencia a largo plazo.',
  },
  distance_to_ma200: {
    term: 'Distancia a la media de 200',
    description:
      'Diferencia porcentual entre el precio actual y su media móvil de 200 sesiones. Positiva si cotiza por encima.',
  },
  max_drawdown_126d: {
    term: 'Caída máxima (6 meses)',
    description:
      'Mayor caída sufrida desde un máximo en los últimos 126 días de mercado (unos 6 meses).',
  },
  downside_volatility_126d: {
    term: 'Volatilidad a la baja (6 meses)',
    description:
      'Volatilidad calculada solo con las caídas de precio en los últimos 6 meses; mide el riesgo a la baja.',
  },
  current_volume: {
    term: 'Volumen actual',
    description: 'Número de acciones negociadas en la última sesión.',
  },
  average_volume_20: {
    term: 'Volumen medio (20 sesiones)',
    description: 'Volumen medio negociado en las últimas 20 sesiones.',
  },
  relative_volume_20: {
    term: 'Volumen relativo',
    description:
      'Volumen actual comparado con su media de 20 sesiones. Por encima de 1 indica una negociación más intensa de lo habitual.',
  },
  donchian_position: {
    term: 'Posición en el canal de precios',
    description:
      'Posición del precio dentro de su rango reciente: cerca de 1 indica máximos recientes, cerca de 0 indica mínimos recientes.',
  },
}
