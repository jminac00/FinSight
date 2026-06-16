import { useEffect, useRef } from 'react'
import { Outlet, useLocation } from 'react-router-dom'
import { ConsentBanner } from '../consent/ConsentBanner'
import { PreferencesDialog } from '../consent/PreferencesDialog'
import { Footer } from './Footer'
import { Header } from './Header'

export function RootLayout() {
  const { pathname } = useLocation()
  const mainRef = useRef<HTMLElement>(null)

  // On navigation, reset scroll and move focus to <main> so keyboard and
  // screen-reader users land on the new content (SPA accessibility).
  useEffect(() => {
    window.scrollTo(0, 0)
    mainRef.current?.focus()
  }, [pathname])

  return (
    <div className="flex min-h-dvh flex-col">
      <a href="#main" className="skip-link">
        Saltar al contenido principal
      </a>
      <Header />
      <main id="main" ref={mainRef} tabIndex={-1} className="flex-1 focus:outline-none">
        <Outlet />
      </main>
      <Footer />
      <ConsentBanner />
      <PreferencesDialog />
    </div>
  )
}
