import type { HTMLAttributes } from 'react'
import { cn } from '../../lib/cn'

/**
 * Loading placeholder. Decorative — the surrounding region owns the accessible
 * status message (RNF-19), so this is hidden from assistive tech.
 */
export function Skeleton({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      aria-hidden="true"
      className={cn('animate-pulse rounded-md bg-panel', className)}
      {...props}
    />
  )
}
