/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Backend origin in production, where frontend and backend are separate services. */
  readonly VITE_API_BASE_URL?: string
  /** Set to 'false' to hit the real API instead of MSW mocks in development. */
  readonly VITE_ENABLE_MOCKS?: string
}
