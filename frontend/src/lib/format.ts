/** Locale-aware formatting helpers for the Spanish UI. */

const LOCALE = 'es-ES'

export function formatNumber(value: number, fractionDigits = 2): string {
  return new Intl.NumberFormat(LOCALE, {
    minimumFractionDigits: fractionDigits,
    maximumFractionDigits: fractionDigits,
  }).format(value)
}

export function formatCurrencyUsd(value: number): string {
  return new Intl.NumberFormat(LOCALE, {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value)
}

/** Signed percentage, e.g. "+7,24 %" / "-3,10 %". */
export function formatSignedPercent(value: number, fractionDigits = 2): string {
  const formatted = new Intl.NumberFormat(LOCALE, {
    minimumFractionDigits: fractionDigits,
    maximumFractionDigits: fractionDigits,
    signDisplay: 'exceptZero',
  }).format(value)
  return `${formatted} %`
}

export function formatPercent(value: number, fractionDigits = 0): string {
  return `${new Intl.NumberFormat(LOCALE, {
    minimumFractionDigits: fractionDigits,
    maximumFractionDigits: fractionDigits,
  }).format(value)} %`
}

/** Ratio in [0, 1] rendered as a whole percentage, e.g. 0.82 -> "82 %". */
export function formatRatioAsPercent(ratio: number): string {
  return formatPercent(ratio * 100, 0)
}

export function formatDate(iso: string): string {
  return new Intl.DateTimeFormat(LOCALE, { dateStyle: 'long' }).format(new Date(iso))
}

export function formatDateTime(iso: string): string {
  return new Intl.DateTimeFormat(LOCALE, { dateStyle: 'long', timeStyle: 'short' }).format(
    new Date(iso),
  )
}

/** Humanize a metric/indicator key, e.g. "ev_ebitda" -> "EV EBITDA". */
export function humanizeKey(key: string): string {
  return key.replace(/_/g, ' ').toUpperCase()
}
