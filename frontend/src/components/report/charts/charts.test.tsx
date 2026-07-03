import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { axe } from 'vitest-axe'
import { BlockScores } from './BlockScores'
import { ReturnProjection } from './ReturnProjection'
import { ScoreDial } from './ScoreDial'
import { SentimentScale } from './SentimentScale'

describe('report charts', () => {
  it('ScoreDial exposes the value through an accessible label and text', () => {
    render(<ScoreDial value={7.8} label="Puntuación fundamental" />)

    expect(
      screen.getByRole('img', { name: /puntuación fundamental: 7,8 sobre 10/i }),
    ).toBeInTheDocument()
    expect(screen.getByText('7,8')).toBeInTheDocument()
  })

  it('SentimentScale labels the score on a -1..1 scale', () => {
    render(<SentimentScale value={0.62} />)

    expect(screen.getByRole('img', { name: /sentimiento 0,62/i })).toBeInTheDocument()
  })

  it('ReturnProjection labels the signed return', () => {
    render(<ReturnProjection returnPct={4.82} horizonDays={10} />)

    expect(screen.getByRole('img', { name: /\+4,82 %/i })).toBeInTheDocument()
  })

  it('BlockScores renders each block and marks missing ones as n/d', () => {
    render(
      <BlockScores
        blocks={{ momentum: 7.1, trend: 6.8, risk_stability: null, confirmation: 6.2 }}
      />,
    )

    expect(screen.getByText('Momentum')).toBeInTheDocument()
    expect(screen.getByText('Riesgo y estabilidad')).toBeInTheDocument()
    expect(screen.getByText('7,1 / 10')).toBeInTheDocument()
    expect(screen.getByText('n/d')).toBeInTheDocument()
  })

  it('has no accessibility violations', async () => {
    const { container } = render(
      <div>
        <ScoreDial value={6.4} label="Puntuación técnica" />
        <SentimentScale value={-0.3} />
        <ReturnProjection returnPct={-2.1} horizonDays={10} />
        <BlockScores
          blocks={{ momentum: 7.1, trend: null, risk_stability: 5.4, confirmation: 6.2 }}
        />
      </div>,
    )

    expect(await axe(container)).toHaveNoViolations()
  })
})
