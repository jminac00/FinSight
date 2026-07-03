import { useId, useState, type KeyboardEvent } from 'react'
import { InfoIcon } from '../icons'

/**
 * Inline disclosure that reveals a short explanation of a term. Uses the
 * WAI-ARIA disclosure pattern (button with aria-expanded + aria-controls) and
 * expands in flow, so it is never clipped by a container's overflow.
 */
export function InfoTip({ term, description }: { term: string; description: string }) {
  const [open, setOpen] = useState(false)
  const panelId = useId()

  function onKeyDown(event: KeyboardEvent<HTMLButtonElement>) {
    if (event.key === 'Escape') setOpen(false)
  }

  return (
    <>
      <button
        type="button"
        aria-expanded={open}
        aria-controls={open ? panelId : undefined}
        aria-label={`Qué significa ${term}`}
        onClick={() => setOpen((prev) => !prev)}
        onKeyDown={onKeyDown}
        className="ml-1 inline-flex h-4 w-4 shrink-0 items-center justify-center rounded-full align-middle text-ink-subtle transition-colors hover:text-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
      >
        <InfoIcon className="h-3.5 w-3.5" />
      </button>
      {open ? (
        <span
          id={panelId}
          role="note"
          className="mt-1.5 block max-w-prose text-xs font-normal normal-case tracking-normal text-ink-muted"
        >
          {description}
        </span>
      ) : null}
    </>
  )
}
