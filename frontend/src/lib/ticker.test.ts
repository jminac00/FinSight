import { describe, expect, it } from 'vitest'
import { isValidTicker, normalizeTicker } from './ticker'

describe('normalizeTicker', () => {
  it('uppercases and trims', () => {
    expect(normalizeTicker('  aapl ')).toBe('AAPL')
  })
})

describe('isValidTicker', () => {
  it.each(['AA', 'AAPL', 'ABCDE', 'A1B2C', 'aapl'])('accepts %s', (value) => {
    expect(isValidTicker(value)).toBe(true)
  })

  it.each(['A', 'ABCDEF', 'A B', 'AA-PL', ''])('rejects %s', (value) => {
    expect(isValidTicker(value)).toBe(false)
  })
})
