import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import App from './App'
import { ConsentProvider } from './consent/ConsentContext'
import { ThemeProvider } from './theme/ThemeContext'
import './index.css'

/**
 * In development the UI runs against MSW fixtures by default, so it works without
 * the backend. Set VITE_ENABLE_MOCKS=false to hit the real API via the Vite proxy.
 */
async function enableMocks(): Promise<void> {
  if (!import.meta.env.DEV || import.meta.env.VITE_ENABLE_MOCKS === 'false') return
  const { worker } = await import('./mocks/browser')
  await worker.start({ onUnhandledRequest: 'bypass' })
}

const rootElement = document.getElementById('root')
if (!rootElement) throw new Error('Root element #root not found')

enableMocks().then(() => {
  createRoot(rootElement).render(
    <StrictMode>
      <BrowserRouter>
        <ConsentProvider>
          <ThemeProvider>
            <App />
          </ThemeProvider>
        </ConsentProvider>
      </BrowserRouter>
    </StrictMode>,
  )
})
