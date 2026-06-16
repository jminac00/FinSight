import type { ReactNode } from 'react'
import { cn } from '../../lib/cn'
import { AlertIcon, InfoIcon } from '../icons'

type Tone = 'info' | 'success' | 'warning' | 'danger'

type StatusMessageProps = {
  tone?: Tone
  title?: string
  children?: ReactNode
  className?: string
  /**
   * ARIA live politeness. Defaults: danger -> "assertive" (role=alert),
   * others -> "polite" (role=status). Pass "off" for static, non-live notices.
   */
  live?: 'polite' | 'assertive' | 'off'
  action?: ReactNode
}

const tones: Record<Tone, string> = {
  info: 'bg-info-subtle text-info-fg',
  success: 'bg-success-subtle text-success-fg',
  warning: 'bg-warning-subtle text-warning-fg',
  danger: 'bg-danger-subtle text-danger-fg',
}

export function StatusMessage({
  tone = 'info',
  title,
  children,
  className,
  live,
  action,
}: StatusMessageProps) {
  const politeness = live ?? (tone === 'danger' ? 'assertive' : 'polite')
  const role = politeness === 'off' ? undefined : tone === 'danger' ? 'alert' : 'status'
  const Icon = tone === 'danger' || tone === 'warning' ? AlertIcon : InfoIcon

  return (
    <div
      role={role}
      aria-live={politeness === 'off' ? undefined : politeness}
      className={cn('flex items-start gap-3 rounded-md p-4 text-sm', tones[tone], className)}
    >
      <Icon className="mt-0.5 w-5 shrink-0" />
      <div className="min-w-0 flex-1">
        {title ? <p className="font-semibold">{title}</p> : null}
        {children ? <div className={cn(title && 'mt-1')}>{children}</div> : null}
      </div>
      {action ? <div className="shrink-0">{action}</div> : null}
    </div>
  )
}
