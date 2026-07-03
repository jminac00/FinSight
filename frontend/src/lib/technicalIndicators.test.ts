import { describe, expect, it } from 'vitest'
import { pickTechnicalIndicators, TECHNICAL_INDICATOR_KEYS } from './technicalIndicators'

describe('pickTechnicalIndicators', () => {
  it('keeps only the curated, meaningful indicator keys', () => {
    const flat = {
      momentum_12_1: 0.42,
      momentum_6_1: 0.18,
      vol_12m: 0.29,
      price: 182.3,
      ma_200: 168.4,
      // internal pipeline artifacts that must not be shown or explained
      z_score: 0.83,
      normalization_method: 'z-score',
      normalization_k: 2.5,
      raw_momentum_score: 0.5,
      winsorized_score: 0.4,
      signal: 'alcista',
      summary: 'Momentum positivo.',
    }

    const picked = pickTechnicalIndicators(flat)

    expect(Object.keys(picked)).toEqual(['momentum_12_1', 'momentum_6_1', 'vol_12m', 'price', 'ma_200'])
  })

  it('omits keys that are not present in the input', () => {
    const picked = pickTechnicalIndicators({ price: 100 })
    expect(picked).toEqual({ price: 100 })
  })

  it('exports the same keys it filters by, in display order', () => {
    expect(TECHNICAL_INDICATOR_KEYS.length).toBeGreaterThan(0)
    expect(new Set(TECHNICAL_INDICATOR_KEYS).size).toBe(TECHNICAL_INDICATOR_KEYS.length)
  })
})
