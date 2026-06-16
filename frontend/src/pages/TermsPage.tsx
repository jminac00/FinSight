import { LegalPage, LegalSection } from '../components/LegalPage'

export default function TermsPage() {
  return (
    <LegalPage
      title="Términos de uso"
      documentTitle="Términos de uso · FinSight"
      updated="junio de 2026"
    >
      <p>
        El acceso y uso de FinSight implican la aceptación de estos términos. FinSight es una
        herramienta académica de carácter informativo desarrollada como Trabajo de Fin de Grado.
      </p>

      <LegalSection heading="Naturaleza informativa del contenido">
        <p>
          Todo el contenido de la plataforma, incluidos los análisis, predicciones y conclusiones, es
          de carácter exclusivamente informativo y educativo, y ha sido generado, total o
          parcialmente, mediante sistemas de inteligencia artificial a partir de fuentes de datos de
          terceros.
        </p>
      </LegalSection>

      <LegalSection heading="No constituye asesoramiento financiero">
        <p>
          De conformidad con la directiva MiFID II, el contenido de FinSight no constituye
          asesoramiento financiero ni recomendación personalizada de inversión, ni representa una
          oferta o solicitud de compra o venta de valores. La plataforma no tiene en cuenta tus
          objetivos, situación financiera ni necesidades particulares.
        </p>
      </LegalSection>

      <LegalSection heading="Limitación de responsabilidad">
        <p>
          Las decisiones de inversión son responsabilidad exclusiva del usuario. El autor y la
          Universidad de León no asumen responsabilidad alguna por pérdidas o daños derivados del uso
          de la información proporcionada. Los datos pueden contener errores, retrasos o
          imprecisiones, y el servicio se ofrece «tal cual», sin garantías de disponibilidad,
          exactitud o adecuación a un fin determinado. Antes de tomar cualquier decisión de
          inversión, consulta a un asesor financiero cualificado.
        </p>
      </LegalSection>

      <LegalSection heading="Fuentes de datos">
        <p>
          FinSight integra información de proveedores externos. Los análisis se basan en datos de
          cierre de mercado (EOD) y no reflejan la cotización en tiempo real. El uso de dichas
          fuentes se rige por sus respectivos términos.
        </p>
      </LegalSection>

      <LegalSection heading="Legislación aplicable">
        <p>
          Estos términos se rigen por la legislación española. Cualquier controversia se someterá a
          los juzgados y tribunales competentes conforme a derecho.
        </p>
      </LegalSection>
    </LegalPage>
  )
}
