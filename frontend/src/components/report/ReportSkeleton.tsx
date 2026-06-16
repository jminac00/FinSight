import { Card } from '../ui/Card'
import { Skeleton } from '../ui/Skeleton'

function SectionSkeleton() {
  return (
    <Card className="p-5 sm:p-6">
      <Skeleton className="h-5 w-48" />
      <Skeleton className="mt-4 h-8 w-32" />
      <Skeleton className="mt-4 h-4 w-full max-w-prose" />
      <Skeleton className="mt-2 h-4 w-full max-w-prose" />
      <Skeleton className="mt-2 h-4 w-2/3" />
    </Card>
  )
}

/** Content-shaped loading placeholder (decorative; status is announced separately). */
export function ReportSkeleton() {
  return (
    <div aria-hidden="true" className="space-y-4">
      <SectionSkeleton />
      <SectionSkeleton />
      <SectionSkeleton />
      <SectionSkeleton />
    </div>
  )
}
