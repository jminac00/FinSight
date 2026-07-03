import { describe, expect, it } from 'vitest'
import { fullReport } from '../mocks/fixtures'
import { GLOSSARY } from './glossary'
import { TECHNICAL_INDICATOR_KEYS } from './technicalIndicators'

describe('metric glossary coverage', () => {
  it('has an entry for every fundamental ratio shown in the report', () => {
    const ratios = fullReport('AAPL').fundamental?.metrics.ratios as Record<string, unknown>
    const keys = Object.keys(ratios)
    expect(keys.length).toBeGreaterThan(0)
    for (const key of keys) {
      expect(GLOSSARY[key], `missing glossary entry for fundamental ratio "${key}"`).toBeDefined()
    }
  })

  it('has an entry for every curated technical indicator shown in the report', () => {
    for (const key of TECHNICAL_INDICATOR_KEYS) {
      expect(GLOSSARY[key], `missing glossary entry for technical indicator "${key}"`).toBeDefined()
    }
  })
})
