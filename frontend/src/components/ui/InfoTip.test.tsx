import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import { axe } from 'vitest-axe'
import { InfoTip } from './InfoTip'

describe('InfoTip', () => {
  it('toggles the explanation and is labelled for assistive tech', async () => {
    const user = userEvent.setup()
    render(<InfoTip term="RMSE" description="Error cuadrático medio." />)

    const button = screen.getByRole('button', { name: /qué significa rmse/i })
    expect(button).toHaveAttribute('aria-expanded', 'false')
    expect(screen.queryByText('Error cuadrático medio.')).not.toBeInTheDocument()

    await user.click(button)
    expect(button).toHaveAttribute('aria-expanded', 'true')
    expect(screen.getByRole('note')).toHaveTextContent('Error cuadrático medio.')

    await user.click(button)
    expect(screen.queryByText('Error cuadrático medio.')).not.toBeInTheDocument()
  })

  it('closes on Escape', async () => {
    const user = userEvent.setup()
    render(<InfoTip term="MAE" description="Error absoluto medio." />)

    await user.click(screen.getByRole('button'))
    expect(screen.getByRole('note')).toBeInTheDocument()

    await user.keyboard('{Escape}')
    expect(screen.queryByRole('note')).not.toBeInTheDocument()
  })

  it('has no accessibility violations open or closed', async () => {
    const { container, rerender } = render(<InfoTip term="ROE" description="Rentabilidad." />)
    expect(await axe(container)).toHaveNoViolations()

    await userEvent.setup().click(screen.getByRole('button'))
    rerender(<InfoTip term="ROE" description="Rentabilidad." />)
    expect(await axe(container)).toHaveNoViolations()
  })
})
