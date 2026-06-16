import type { AnchorHTMLAttributes } from 'react'
import { cn } from '../../lib/cn'
import { ExternalLinkIcon } from '../icons'

type ExternalLinkProps = AnchorHTMLAttributes<HTMLAnchorElement> & {
  /** Show the trailing external-link glyph. */
  showIcon?: boolean
}

/**
 * Anchor to a third-party origin. Always sets rel="noopener noreferrer" and a
 * screen-reader hint that the link opens in a new tab (RNF-33).
 */
export function ExternalLink({
  href,
  children,
  className,
  showIcon = true,
  ...props
}: ExternalLinkProps) {
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className={cn(
        'inline-flex items-baseline gap-1 text-accent underline decoration-from-font underline-offset-2 hover:text-accent-hover',
        className,
      )}
      {...props}
    >
      <span>{children}</span>
      {showIcon ? <ExternalLinkIcon className="w-3.5 translate-y-0.5" /> : null}
      <span className="sr-only">(se abre en una pestaña nueva)</span>
    </a>
  )
}
