import type { HTMLAttributes } from 'react'
import { cn } from '../../lib/cn'

/** Surface container with a full border (no side-stripe accents). */
export function Card({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn('rounded-lg border border-border bg-surface shadow-sm', className)}
      {...props}
    />
  )
}
