import type { ReactNode } from 'react'
import { Card } from '../ui/Card'

/** Consistent section shell for each analysis module (RF-36). */
export function ReportSection({
  id,
  title,
  badge,
  children,
}: {
  id: string
  title: string
  badge?: ReactNode
  children: ReactNode
}) {
  return (
    <Card className="p-5 sm:p-6" role="region" aria-labelledby={`${id}-heading`}>
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <h2 id={`${id}-heading`} className="text-lg font-semibold text-ink">
          {title}
        </h2>
        {badge}
      </div>
      {children}
    </Card>
  )
}
