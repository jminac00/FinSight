import { useNavigate } from 'react-router-dom'
import Disclaimer from '../components/Disclaimer'

export default function PrivacyPage() {
  const navigate = useNavigate()
  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b px-6 py-3">
        <button
          onClick={() => navigate('/')}
          className="text-blue-600 hover:underline text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 rounded"
        >
          ← Inicio
        </button>
      </header>
      <main className="flex-1 px-6 py-8 max-w-3xl mx-auto w-full prose prose-sm">
        <h1>Política de Privacidad</h1>
        <p>
          <strong>FinSight</strong> es una plataforma de análisis financiero académica desarrollada
          como Trabajo de Fin de Grado en la Universidad de León.
        </p>
        <h2>Datos que no recopilamos</h2>
        <p>
          En su versión actual, FinSight <strong>no recopila, almacena ni procesa ningún dato
          personal identificable</strong> del usuario. No existe registro de usuarios, inicio de
          sesión ni ningún mecanismo de autenticación.
        </p>
        <h2>Datos de uso</h2>
        <p>
          Las consultas de análisis (tickers solicitados) se procesan en memoria para generar el
          informe y no se almacenan de forma permanente ni se asocian a ningún identificador de usuario.
        </p>
        <h2>Servicios de terceros</h2>
        <p>
          FinSight obtiene datos de fuentes externas (NewsAPI, Finnhub, Yahoo Finance, Groq API).
          El uso de dichos servicios está sujeto a sus respectivas políticas de privacidad.
        </p>
        <h2>Contacto</h2>
        <p>
          Para cualquier consulta sobre esta política, puedes contactar con el autor del proyecto
          a través de la Universidad de León.
        </p>
        <p className="text-xs text-gray-400">Última actualización: junio 2026</p>
      </main>
      <footer>
        <Disclaimer />
      </footer>
    </div>
  )
}
