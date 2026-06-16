import { describe, expect, it } from 'vitest'
import {
  formatCurrencyUsd,
  formatNumber,
  formatRatioAsPercent,
  formatSignedPercent,
  humanizeKey,
} from './format'

describe('formatNumber', () => {
  it('uses the Spanish decimal comma', () => {
    expect(formatNumber(3.12, 2)).toBe('3,12')
  })
})

describe('formatSignedPercent', () => {
  it('shows a leading + for positive values', () => {
    const result = formatSignedPercent(4.82)
    expect(result.startsWith('+')).toBe(true)
    expect(result).toContain('4,82')
    expect(result.endsWith('%')).toBe(true)
  })

  it('shows a sign for negative values', () => {
    expect(formatSignedPercent(-3.1)).toContain('3,10')
  })
})

describe('formatRatioAsPercent', () => {
  it('converts a [0,1] ratio to a whole percentage', () => {
    expect(formatRatioAsPercent(0.63)).toBe('63 %')
  })
})

describe('formatCurrencyUsd', () => {
  it('formats USD with two decimals', () => {
    expect(formatCurrencyUsd(182.3)).toContain('182,30')
  })
})

describe('humanizeKey', () => {
  it('replaces underscores and uppercases', () => {
    expect(humanizeKey('ev_ebitda')).toBe('EV EBITDA')
  })
})
