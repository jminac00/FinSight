import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { axe } from 'vitest-axe'
import type { DLUnavailableReason } from '../../api/types'
import { DeepLearningUnavailable } from './DeepLearningUnavailable'

describe('DeepLearningUnavailable', () => {
  it('explains that the ticker is outside the module coverage', () => {
    render(<DeepLearningUnavailable reason="out_of_coverage" />)

    expect(screen.getByText(/fuera de la cobertura actual/i)).toBeInTheDocument()
  })

  it('explains that the model is not trained yet', () => {
    render(<DeepLearningUnavailable reason="not_trained" />)

    expect(screen.getByText(/aún no está disponible/i)).toBeInTheDocument()
  })

  it('explains that a model exists but underperforms', () => {
    render(<DeepLearningUnavailable reason="insufficient_quality" />)

    const message = screen.getByText(/no alcanza el mínimo exigido/i)
    expect(message).toBeInTheDocument()
    // The distinction that matters: a model exists, it is just not good enough.
    expect(message).toHaveTextContent(/existe un modelo/i)
  })

  it('gives each reason its own wording', () => {
    const reasons: DLUnavailableReason[] = [
      'out_of_coverage',
      'not_trained',
      'insufficient_quality',
    ]
    const texts = reasons.map((reason) => {
      const { container, unmount } = render(<DeepLearningUnavailable reason={reason} />)
      const text = container.textContent ?? ''
      unmount()
      return text
    })

    expect(new Set(texts).size).toBe(reasons.length)
  })

  it('renders nothing for an unrecognised reason', () => {
    // The backend value is only ever a lookup key, never rendered text, so an
    // unexpected payload cannot inject anything into the DOM.
    const { container } = render(
      <DeepLearningUnavailable reason={'<img src=x onerror=alert(1)>' as DLUnavailableReason} />,
    )

    expect(container).toBeEmptyDOMElement()
  })

  it('does not rely on colour alone to convey the notice', () => {
    render(<DeepLearningUnavailable reason="not_trained" />)

    // A visible title carries the meaning in text, independently of the tone.
    expect(screen.getByText(/predicción de tendencia no disponible/i)).toBeInTheDocument()
  })

  it('has no accessibility violations', async () => {
    const { container } = render(<DeepLearningUnavailable reason="insufficient_quality" />)

    expect(await axe(container)).toHaveNoViolations()
  })
})
