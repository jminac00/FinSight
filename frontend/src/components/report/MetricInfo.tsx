import { GLOSSARY } from '../../lib/glossary'
import { InfoTip } from '../ui/InfoTip'

/** Info disclosure for a metric key, or nothing when no explanation exists. */
export function MetricInfo({ metricKey }: { metricKey: string }) {
  const entry = GLOSSARY[metricKey]
  if (!entry) return null
  return <InfoTip term={entry.term} description={entry.description} />
}
