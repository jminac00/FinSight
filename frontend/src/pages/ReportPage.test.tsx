import { render, screen } from '@testing-library/react'
import { HttpResponse, http } from 'msw'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { describe, expect, it } from 'vitest'
import { axe } from 'vitest-axe'
import { ConsentProvider } from '../consent/ConsentContext'
import { fullReport } from '../mocks/fixtures'
import { server } from '../mocks/server'
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
    expect(screen.getByText(/predicción de tendencia \(deep learning\)/i)).toBeInTheDocument()
    expect(
      screen.queryByRole('heading', { name: /predicción de tendencia/i }),
    ).not.toBeInTheDocument()
  })

  it('shows the company name alongside the ticker in the heading', async () => {
    renderReport('AAPL')

    const heading = await screen.findByRole('heading', { name: /análisis de apple inc - aapl/i })
    expect(heading).toBeInTheDocument()
  })

  it('lists every missing module in the partial-support notice', async () => {
    server.use(
      http.get('/api/v1/report/:ticker', () =>
        HttpResponse.json({
          ...fullReport('MULTI'),
          sentiment: null,
          technical: null,
          partial_support: true,
          missing_modules: ['sentiment', 'technical'],
        }),
      ),
    )

    renderReport('MULTI')

    const notice = await screen.findByText(/soporte parcial/i)
    expect(notice).toBeInTheDocument()
    expect(
      screen.getByText(/el análisis de sentimiento y el análisis técnico/i),
    ).toBeInTheDocument()
  })

  it('rejects an invalid ticker without crashing', async () => {
    renderReport('A')
    expect(await screen.findByText(/símbolo no válido/i)).toBeInTheDocument()
  })

  it('shows a dedicated message when rate limited (429)', async () => {
    server.use(
      http.get('/api/v1/report/:ticker', () =>
        HttpResponse.json(
          { detail: 'Rate limit exceeded' },
          { status: 429, headers: { 'Retry-After': '30' } },
        ),
      ),
    )

    renderReport('AAPL')

    expect(await screen.findByText(/demasiadas solicitudes/i)).toBeInTheDocument()
    expect(screen.getByText(/30 segundos/i)).toBeInTheDocument()
  })

  it('has no detectable accessibility violations once loaded', async () => {
    const { container } = renderReport('AAPL')
    await screen.findByRole('heading', { name: /conclusión global/i })
    expect(await axe(container)).toHaveNoViolations()
  })
})
