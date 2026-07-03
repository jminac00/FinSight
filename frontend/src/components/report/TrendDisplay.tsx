import type { SentimentLabel, Trend } from '../../api/types'
import { cn } from '../../lib/cn'
import { TrendDownIcon, TrendFlatIcon, TrendUpIcon } from '../icons'

type Direction = 'up' | 'down' | 'flat'

const STYLES: Record<Direction, { icon: typeof TrendUpIcon; className: string }> = {
  up: { icon: TrendUpIcon, className: 'text-success-fg' },
  down: { icon: TrendDownIcon, className: 'text-danger-fg' },
  flat: { icon: TrendFlatIcon, className: 'text-ink-muted' },
}

const TREND_DIRECTION: Record<Trend, Direction> = {
  alcista: 'up',
  bajista: 'down',
  neutral: 'flat',
}

const SENTIMENT_DIRECTION: Record<SentimentLabel, Direction> = {
  positivo: 'up',
  negativo: 'down',
  neutral: 'flat',
}

const LABELS: Record<Trend | SentimentLabel, string> = {
  alcista: 'Alcista',
  bajista: 'Bajista',
  neutral: 'Neutral',
  positivo: 'Positivo',
  negativo: 'Negativo',
}

/**
 * Renders a trend or sentiment label with icon + text + color. Meaning never
 * relies on color alone (WCAG 1.4.1 — RNF-15/17).
 */
export function TrendDisplay({
  value,
  kind,
  size = 'md',
  className,
}: {
  value: Trend | SentimentLabel
  kind: 'trend' | 'sentiment'
  size?: 'sm' | 'md'
  className?: string
}) {
  const direction =
    kind === 'trend'
      ? TREND_DIRECTION[value as Trend]
      : SENTIMENT_DIRECTION[value as SentimentLabel]
  const { icon: Icon, className: tone } = STYLES[direction]

  return (
    <span
      className={cn(
        'inline-flex items-center font-semibold',
        size === 'sm' ? 'gap-1.5 text-base' : 'gap-2 text-xl',
        tone,
        className,
      )}
    >
      <Icon className={size === 'sm' ? 'w-5' : 'w-6'} />
      {LABELS[value]}
    </span>
  )
}
