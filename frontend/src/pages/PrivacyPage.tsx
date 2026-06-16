import { LegalPage, LegalSection } from '../components/LegalPage'
import { useConsent } from '../consent/ConsentContext'

export default function PrivacyPage() {
  const { openPreferences } = useConsent()

  return (
    <LegalPage
      title="Política de privacidad"
      documentTitle="Política de privacidad · FinSight"
      updated="junio de 2026"
    >
      <p>
        <strong className="text-ink">FinSight</strong> es una plataforma de análisis financiero
        académica, desarrollada como Trabajo de Fin de Grado en la Universidad de León. Esta política
        explica qué información se trata y cómo se utiliza el almacenamiento en tu dispositivo,
        conforme al Reglamento General de Protección de Datos (RGPD) y al artículo 22 de la LSSI-CE.
      </p>

      <LegalSection heading="Datos que no recopilamos">
        <p>
          En su versión actual, FinSight no recopila, almacena ni trata ningún dato personal
          identificable. No existe registro de usuarios, inicio de sesión ni ningún mecanismo de
          autenticación. Las consultas de análisis (los símbolos que solicitas) se procesan en
          memoria para generar el informe y no se asocian a ningún identificador de usuario.
        </p>
      </LegalSection>

      <LegalSection heading="Cookies y almacenamiento en tu dispositivo">
        <p>
          Utilizamos cookies y tecnologías equivalentes de almacenamiento en cliente
          (<code>localStorage</code>) clasificadas en tres categorías. Conforme a las directrices de
          la AEPD, las obligaciones de consentimiento se aplican a cualquiera de estas tecnologías.
        </p>
        <ul className="list-disc space-y-2 pl-5">
          <li>
            <strong className="text-ink">Estrictamente necesarias.</strong> Imprescindibles para el
            funcionamiento del sitio, incluida la memoria de tu decisión sobre las cookies. No
            requieren consentimiento y no se pueden desactivar.
          </li>
          <li>
            <strong className="text-ink">Funcionales.</strong> Recuerdan tus preferencias: el último
            símbolo analizado, el tema visual (claro u oscuro) y la confirmación de lectura del aviso
            legal. Solo se activan con tu consentimiento.
          </li>
          <li>
            <strong className="text-ink">Analíticas.</strong> Permiten medir el uso de la plataforma
            de forma agregada y anónima (páginas y símbolos más consultados, duración de sesión).
            Solo se activan con tu consentimiento explícito.
          </li>
        </ul>
        <p>
          Puedes revisar o cambiar tu decisión en cualquier momento desde{' '}
          <button
            type="button"
            onClick={openPreferences}
            className="text-accent underline underline-offset-2 hover:text-accent-hover"
          >
            las preferencias de cookies
          </button>
          . Si rechazas las categorías opcionales, no se almacena ninguna información no esencial en
          tu dispositivo.
        </p>
      </LegalSection>

      <LegalSection heading="Servicios de terceros">
        <p>
          FinSight obtiene datos de fuentes externas (entre ellas Finnhub, NewsAPI, Yahoo Finance y
          proveedores de modelos de lenguaje). El tratamiento de la información por parte de dichos
          servicios se rige por sus respectivas políticas de privacidad.
        </p>
      </LegalSection>

      <LegalSection heading="Tus derechos">
        <p>
          Dado que no se tratan datos personales identificables, no es posible vincular información a
          una persona concreta. Si en el futuro se incorporasen tratamientos de datos personales, se
          informaría de los derechos de acceso, rectificación, supresión, oposición, limitación y
          portabilidad reconocidos por el RGPD.
        </p>
      </LegalSection>

      <LegalSection heading="Contacto">
        <p>
          Para cualquier consulta sobre esta política puedes contactar con el autor del proyecto a
          través de la Universidad de León.
        </p>
      </LegalSection>
    </LegalPage>
  )
}
