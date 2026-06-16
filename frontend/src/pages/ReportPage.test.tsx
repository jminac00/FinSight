import { render, screen } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { describe, expect, it } from 'vitest'
import { axe } from 'vitest-axe'
import { ConsentProvider } from '../consent/ConsentContext'
import { ThemeProvider } from '../theme/ThemeContext'
import ReportPage from './ReportPage'

function renderReport(ticker: string) {
  return render(
    <MemoryRouter initialEntries={[`/report/${ticker}`]}>
      <ConsentProvider>
        <ThemeProvider>
          <Routes>
            <Route path="/report/:ticker" element={<ReportPage />} />
          </Routes>
        </ThemeProvider>
      </ConsentProvider>
    </MemoryRouter>,
  )
}

describe('ReportPage', () => {
  it('announces loading then renders the four sections in order', async () => {
    renderReport('AAPL')

    expect(screen.getByText(/generando el análisis/i)).toBeInTheDocument()

    const sentiment = await screen.findByRole('heading', { name: /análisis de sentimiento/i })
    expect(sentiment).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: /predicción de tendencia/i })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: /análisis fundamental/i })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: /análisis técnico/i })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: /conclusión global/i })).toBeInTheDocument()
  })

  it('marks AI-generated content explicitly (RF-38)', async () => {
    renderReport('AAPL')
    const badges = await screen.findAllByText(/generado por ia/i)
    expect(badges.length).toBeGreaterThan(0)
  })

  it('shows the partial-support notice and omits the prediction (RF-27)', async () => {
    renderReport('TSLA')

    expect(await screen.findByText(/soporte parcial/i)).toBeInTheDocument()
    expect(
      screen.queryByRole('heading', { name: /predicción de tendencia/i }),
    ).not.toBeInTheDocument()
  })

  it('rejects an invalid ticker without crashing', async () => {
    renderReport('A')
    expect(await screen.findByText(/símbolo no válido/i)).toBeInTheDocument()
  })

  it('has no detectable accessibility violations once loaded', async () => {
    const { container } = renderReport('AAPL')
    await screen.findByRole('heading', { name: /conclusión global/i })
    expect(await axe(container)).toHaveNoViolations()
  })
})
