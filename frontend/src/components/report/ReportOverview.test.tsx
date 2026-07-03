import { render, screen, within } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { fullReport, partialReport } from '../../mocks/fixtures'
import { ReportOverview } from './ReportOverview'

describe('ReportOverview', () => {
  it('summarizes each module and links to its section', () => {
    render(<ReportOverview data={fullReport('AAPL')} />)

    const nav = screen.getByRole('navigation', { name: /resumen del informe/i })
    const links = within(nav).getAllByRole('link')
    expect(links.map((l) => l.getAttribute('href'))).toEqual([
      '#sentiment',
      '#dl',
      '#fundamental',
      '#technical',
    ])
  })

  it('marks an unavailable module without linking to a missing section', () => {
    render(<ReportOverview data={partialReport('TSLA')} />)

    const nav = screen.getByRole('navigation', { name: /resumen del informe/i })
    expect(within(nav).getAllByRole('link')).toHaveLength(3)
    expect(within(nav).getByText(/no disponible/i)).toBeInTheDocument()
    expect(within(nav).queryByRole('link', { name: /tendencia/i })).not.toBeInTheDocument()
  })
})
