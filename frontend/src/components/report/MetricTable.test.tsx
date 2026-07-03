import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { MetricTable } from './MetricTable'

describe('MetricTable', () => {
  it('renders primitive values and skips nested objects, arrays and null', () => {
    render(
      <MetricTable
        caption="Ratios"
        data={{
          per: 28.5,
          sub_signal: 'sobrevalorada',
          score_reliable: true,
          scores: { calidad: 9.9 },
          history: [1, 2, 3],
          missing: null,
        }}
      />,
    )

    expect(screen.getByText('PER')).toBeInTheDocument()
    expect(screen.getByText('28,50')).toBeInTheDocument()
    expect(screen.getByText('sobrevalorada')).toBeInTheDocument()
    expect(screen.getByText('Sí')).toBeInTheDocument()
    // Nested object / array / null keys must not be rendered.
    expect(screen.queryByText('SCORES')).not.toBeInTheDocument()
    expect(screen.queryByText('HISTORY')).not.toBeInTheDocument()
    expect(screen.queryByText('MISSING')).not.toBeInTheDocument()
  })

  it('renders nothing when no value is displayable', () => {
    const { container } = render(
      <MetricTable caption="Vacío" data={{ nested: { a: 1 }, list: [1] }} />,
    )
    expect(container).toBeEmptyDOMElement()
  })
})
