/// <reference types="vitest/config" />
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    css: true,
    // Vitest loads `.env` through Vite, so a developer's dev configuration
    // would otherwise leak into the suite. A local VITE_API_BASE_URL makes the
    // client build absolute URLs that the relative MSW handlers never match,
    // turning the suite red locally while CI — which has no `.env` — stays
    // green. Pinning it empty keeps every request same-origin and interceptable.
    env: {
      VITE_API_BASE_URL: '',
    },
  },
})
