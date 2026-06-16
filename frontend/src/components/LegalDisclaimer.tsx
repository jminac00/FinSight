import { cn } from '../lib/cn'

/**
 * Permanent financial disclaimer (RF-06 / RNF-32, MiFID II): informative only,
 * AI-generated, not financial advice, not an offer to buy or sell securities.
 * Rendered persistently in the footer and, prominently, inside the report.
 */
export function LegalDisclaimer({ className }: { className?: string }) {
  return (
    <div className={cn('text-xs leading-relaxed text-ink-muted', className)}>
      <strong className="font-semibold text-ink">Aviso legal:</strong> el contenido de esta
      plataforma es de carácter <strong className="font-semibold">exclusivamente informativo</strong>{' '}
      y ha sido generado mediante sistemas de{' '}
      <strong className="font-semibold">inteligencia artificial</strong>. No constituye
      asesoramiento financiero ni recomendación de inversión, ni representa una oferta de compra o
      venta de valores. Consulta siempre a un asesor financiero cualificado antes de tomar
      decisiones de inversión.
    </div>
  )
}
