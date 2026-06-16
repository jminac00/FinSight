import { Badge } from '../ui/Badge'
import { SparkleIcon } from '../icons'

/** Explicit AI-attribution marker for LLM-generated text (RF-38). */
export function AiBadge({ label = 'Generado por IA' }: { label?: string }) {
  return (
    <Badge tone="accent">
      <SparkleIcon className="w-3.5" />
      {label}
    </Badge>
  )
}
