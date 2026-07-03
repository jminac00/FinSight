import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { fullReport } from '../../mocks/fixtures'
import { FundamentalSection, TechnicalSection } from './sections'

const report = fullReport('AAPL')

describe('FundamentalSection', () => {
  it('shows the summary and score by default, hiding ratio details behind a disclosure', () => {
    render(<FundamentalSection data={report.fundamental!} />)

    expect(screen.getByText(report.fundamental!.llm_analysis)).toBeInTheDocument()
    expect(screen.getByText('Ver ratios detallados')).toBeInTheDocument()
    expect(screen.getByText('ROE')).not.toBeVisible()
  })
})

describe('TechnicalSection', () => {
  it('shows the summary, score and signal by default, hiding blocks and indicators behind a disclosure', () => {
    render(<TechnicalSection data={report.technical!} />)

    expect(screen.getByText(report.technical!.llm_analysis)).toBeInTheDocument()
    expect(screen.getByText('Ver bloques e indicadores detallados')).toBeInTheDocument()
    expect(screen.getByText('Momentum')).not.toBeVisible()
    expect(screen.getByText('PRICE')).not.toBeVisible()
  })

  it('only shows curated indicators, not internal computation artifacts', () => {
    render(<TechnicalSection data={report.technical!} />)

    expect(screen.getByText('PRICE')).toBeInTheDocument()
    expect(screen.queryByText(/z_score/i)).not.toBeInTheDocument()
    expect(screen.queryByText('NORMALIZATION METHOD')).not.toBeInTheDocument()
  })
})
