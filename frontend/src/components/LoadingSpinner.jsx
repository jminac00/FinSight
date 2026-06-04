export default function LoadingSpinner() {
  return (
    <div
      role="status"
      aria-live="polite"
      aria-label="Generando análisis"
      className="flex flex-col items-center justify-center gap-4 py-16"
    >
      <div
        className="w-12 h-12 border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin"
        aria-hidden="true"
      />
      <p className="text-gray-600 text-sm">
        Generando análisis... esto puede tardar hasta 60 segundos
      </p>
    </div>
  )
}
