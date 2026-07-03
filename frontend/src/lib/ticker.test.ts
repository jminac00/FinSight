import { describe, expect, it } from 'vitest'
import { isValidTicker, normalizeTicker } from './ticker'

describe('normalizeTicker', () => {
  it('uppercases and trims', () => {
    expect(normalizeTicker('  aapl ')).toBe('AAPL')
  })
})

describe('isValidTicker', () => {
  it.each(['AA', 'AAPL', 'A1B2C', 'aapl', 'REP.MC', 'BRK.B', 'BRK-B', 'ASML.AS'])(
    'accepts %s',
    (value) => {
      expect(isValidTicker(value)).toBe(true)
    },
  )

  it.each(['A B', '.MC', '-A', 'TOOLONGTICKERXYZ', ''])('rejects %s', (value) => {
    expect(isValidTicker(value)).toBe(false)
  })
})
