import type { HTMLAttributes } from 'react'
import { cn } from '../../lib/cn'

type Tone = 'neutral' | 'accent' | 'success' | 'warning' | 'danger' | 'info'

export type BadgeProps = HTMLAttributes<HTMLSpanElement> & {
  tone?: Tone
}

const tones: Record<Tone, string> = {
  neutral: 'bg-panel text-ink-muted border-border',
  accent: 'bg-accent-subtle text-accent border-transparent',
  success: 'bg-success-subtle text-success-fg border-transparent',
  warning: 'bg-warning-subtle text-warning-fg border-transparent',
  danger: 'bg-danger-subtle text-danger-fg border-transparent',
  info: 'bg-info-subtle text-info-fg border-transparent',
}

export function Badge({ tone = 'neutral', className, ...props }: BadgeProps) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-xs font-medium',
        tones[tone],
        className,
      )}
      {...props}
    />
  )
}
