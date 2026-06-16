import { cn } from '../../lib/cn'

type SpinnerProps = {
  className?: string
  /** Accessible label; omit when a sibling element already announces the state. */
  label?: string
}

/** Indeterminate activity indicator. Respects prefers-reduced-motion via global CSS. */
export function Spinner({ className, label }: SpinnerProps) {
  return (
    <span
      className={cn(
        'inline-block aspect-square w-5 animate-spin rounded-full border-2 border-current border-t-transparent',
        className,
      )}
      role={label ? 'status' : undefined}
      aria-label={label}
      aria-hidden={label ? undefined : true}
    />
  )
}
