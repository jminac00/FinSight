import type { TechnicalBlockScores } from '../../../api/types'
import { formatNumber } from '../../../lib/format'
import { ScoreBar } from '../ScoreBar'

const BLOCKS: { key: keyof TechnicalBlockScores; label: string }[] = [
  { key: 'momentum', label: 'Momentum' },
  { key: 'trend', label: 'Tendencia' },
  { key: 'risk_stability', label: 'Riesgo y estabilidad' },
  { key: 'confirmation', label: 'Confirmación' },
]

function tone(value: number): 'success' | 'accent' | 'danger' {
  if (value >= 7) return 'success'
  if (value >= 4) return 'accent'
  return 'danger'
}

/** Per-block technical scores (0–10) as labelled rows with a bar and value text. */
export function BlockScores({ blocks }: { blocks: TechnicalBlockScores }) {
  return (
    <div className="flex flex-col gap-3">
      {BLOCKS.map(({ key, label }) => {
        const value = blocks[key]
        return (
          <div key={key}>
            <div className="flex items-baseline justify-between gap-3">
              <span className="text-sm text-ink-muted">{label}</span>
              <span className="text-sm font-semibold tabular-nums text-ink">
                {value === null ? 'n/d' : `${formatNumber(value, 1)} / 10`}
              </span>
            </div>
            <ScoreBar
              value={value ?? 0}
              tone={value === null ? 'neutral' : tone(value)}
              className="mt-1.5"
            />
          </div>
        )
      })}
    </div>
  )
}
