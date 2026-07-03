import type { ReactNode } from 'react'
import { cn } from '../../lib/cn'
import { ChevronDownIcon } from '../icons'

/**
 * Expand/collapse section built on native <details>/<summary>: keyboard-
 * operable and screen-reader friendly without any extra ARIA wiring.
 */
export function Disclosure({
  summary,
  children,
  className,
}: {
  summary: ReactNode
  children: ReactNode
  className?: string
}) {
  return (
    <details className={cn('group', className)}>
      <summary
        tabIndex={0}
        className={cn(
          'flex w-fit cursor-pointer select-none list-none items-center gap-1.5 rounded-sm',
          'text-sm font-medium text-accent transition-colors duration-150 ease-out-quart',
          'hover:text-accent-hover focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent',
          '[&::-webkit-details-marker]:hidden',
        )}
      >
        <ChevronDownIcon className="w-4 shrink-0 transition-transform duration-150 ease-out-quart group-open:rotate-180" />
        {summary}
      </summary>
      <div className="mt-4">{children}</div>
    </details>
  )
}
