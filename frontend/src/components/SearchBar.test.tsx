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
  it('shows an accessible error when the input is neither a ticker nor a company', async () => {
    const user = userEvent.setup()
    renderSearchBar()

    await user.type(screen.getByLabelText(/empresa o símbolo/i), 'A')
    await user.click(screen.getByRole('button', { name: /analizar/i }))

    const alert = await screen.findByRole('alert')
    expect(alert).toHaveTextContent(/nombre de empresa o un símbolo válido/i)
    expect(screen.getByRole('combobox')).toHaveAttribute('aria-invalid', 'true')
  })

  it('navigates to the report when a valid ticker is typed directly', async () => {
    const user = userEvent.setup()
    renderSearchBar()

    await user.type(screen.getByLabelText(/empresa o símbolo/i), 'aapl')
    await user.click(screen.getByRole('button', { name: /analizar/i }))

    expect(screen.getByTestId('location')).toHaveTextContent('/report/AAPL')
  })

  it('suggests companies by name and navigates on selection', async () => {
    const user = userEvent.setup()
    renderSearchBar()

    await user.type(screen.getByLabelText(/empresa o símbolo/i), 'apple')
    const option = await screen.findByRole('option', { name: /apple inc/i })
    await user.click(option)

    expect(screen.getByTestId('location')).toHaveTextContent('/report/AAPL')
  })

  it('supports keyboard selection of a suggestion', async () => {
    const user = userEvent.setup()
    renderSearchBar()

    await user.type(screen.getByLabelText(/empresa o símbolo/i), 'micro')
    await screen.findByRole('option', { name: /microsoft/i })
    await user.keyboard('{ArrowDown}{Enter}')

    expect(screen.getByTestId('location')).toHaveTextContent('/report/MSFT')
  })

  it('has no detectable accessibility violations', async () => {
    const { container } = renderSearchBar()
    expect(await axe(container)).toHaveNoViolations()
  })
})
