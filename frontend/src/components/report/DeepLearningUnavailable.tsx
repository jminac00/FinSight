import type { DLUnavailableReason } from '../../api/types'
import { StatusMessage } from '../ui/StatusMessage'

const TITLE = 'Predicción de tendencia no disponible'

/**
 * One message per reason, in a fixed table keyed by the backend value.
 *
 * The reason arriving from the API is only ever used as a lookup key — it is
 * never rendered — so a malformed or unexpected payload cannot put text of its
 * own into the DOM (CLAUDE.md §3.8).
 */
const MESSAGES: Record<DLUnavailableReason, string> = {
  out_of_coverage:
    'Este valor queda fuera de la cobertura actual del módulo, que analiza valores del S&P 500.',
  not_trained: 'El modelo aún no está disponible para este valor.',
  insufficient_quality:
    'Existe un modelo para este valor, pero su capacidad predictiva no alcanza el mínimo ' +
    'exigido, así que no se ofrece la predicción.',
}

/**
 * Explains, soberly, why the report carries no trend prediction.
 *
 * Renders nothing when the reason is not one of the known values, so the rest
 * of the report is never disturbed by an unexpected payload.
 */
export function DeepLearningUnavailable({ reason }: { reason: DLUnavailableReason }) {
  const message = MESSAGES[reason]
  if (!message) return null

  return (
    // tone="info" on purpose: a missing module is informative, not alarming.
    // The title carries the meaning as text, so the notice never depends on
    // colour alone (WCAG 2.1 AA, 1.4.1).
    <StatusMessage tone="info" title={TITLE} live="off">
      <p>{message}</p>
    </StatusMessage>
  )
}
