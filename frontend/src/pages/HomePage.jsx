import Disclaimer from '../components/Disclaimer'
import SearchBar from '../components/SearchBar'

export default function HomePage() {
  return (
    <div className="min-h-screen flex flex-col">
      <main className="flex-1 flex flex-col items-center justify-center gap-8 px-4">
        <div className="text-center">
          <h1 className="text-4xl font-bold text-gray-900">FinSight</h1>
          <p className="mt-2 text-gray-500 text-sm">
            Análisis financiero inteligente de acciones del Nasdaq
          </p>
        </div>
        <SearchBar />
      </main>
      <footer>
        <Disclaimer />
      </footer>
    </div>
  )
}
