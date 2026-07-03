import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import { axe } from 'vitest-axe'
import { Disclosure } from './Disclosure'

describe('Disclosure', () => {
  it('hides its content by default and reveals it on click', async () => {
    const user = userEvent.setup()
    render(
      <Disclosure summary="Ver más detalles">
        <p>Contenido detallado</p>
      </Disclosure>,
    )

    expect(screen.queryByText('Contenido detallado')).not.toBeVisible()

    await user.click(screen.getByText('Ver más detalles'))
    expect(screen.getByText('Contenido detallado')).toBeVisible()
  })

  it('is operable with the keyboard', async () => {
    const user = userEvent.setup()
    render(
      <Disclosure summary="Ver más detalles">
        <p>Contenido detallado</p>
      </Disclosure>,
    )

    await user.tab()
    expect(screen.getByText('Ver más detalles')).toHaveFocus()
    await user.keyboard('{Enter}')
    expect(screen.getByText('Contenido detallado')).toBeVisible()
  })

  it('has no accessibility violations open or closed', async () => {
    const { container } = render(
      <Disclosure summary="Ver más detalles">
        <p>Contenido detallado</p>
      </Disclosure>,
    )
    expect(await axe(container)).toHaveNoViolations()
  })
})
