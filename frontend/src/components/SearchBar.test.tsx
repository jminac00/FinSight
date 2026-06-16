import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom'
import { describe, expect, it } from 'vitest'
import { axe } from 'vitest-axe'
import { SearchBar } from './SearchBar'

function LocationProbe() {
  const { pathname } = useLocation()
  return <span data-testid="location">{pathname}</span>
}

function renderSearchBar() {
  return render(
    <MemoryRouter initialEntries={['/']}>
      <Routes>
        <Route
          path="/"
          element={
            <>
              <SearchBar />
              <LocationProbe />
            </>
          }
        />
        <Route path="/report/:ticker" element={<LocationProbe />} />
      </Routes>
    </MemoryRouter>,
  )
}

describe('SearchBar', () => {
  it('shows an accessible error for an invalid ticker', async () => {
    const user = userEvent.setup()
    renderSearchBar()
    await user.type(screen.getByLabelText(/símbolo bursátil/i), 'A')
    await user.click(screen.getByRole('button', { name: /analizar/i }))

    const alert = await screen.findByRole('alert')
    expect(alert).toHaveTextContent(/símbolo válido/i)
    expect(screen.getByLabelText(/símbolo bursátil/i)).toHaveAttribute('aria-invalid', 'true')
  })

  it('navigates to the report for a valid ticker', async () => {
    const user = userEvent.setup()
    renderSearchBar()
    await user.type(screen.getByLabelText(/símbolo bursátil/i), 'aapl')
    await user.click(screen.getByRole('button', { name: /analizar/i }))

    expect(screen.getByTestId('location')).toHaveTextContent('/report/AAPL')
  })

  it('has no detectable accessibility violations', async () => {
    const { container } = renderSearchBar()
    expect(await axe(container)).toHaveNoViolations()
  })
})
