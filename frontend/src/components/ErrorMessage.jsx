export default function ErrorMessage({ message }) {
  return (
    <div
      role="alert"
      aria-live="assertive"
      className="bg-red-50 border border-red-300 rounded-lg p-4 text-red-800 text-sm"
    >
      <strong>Error:</strong> {message}
    </div>
  )
}
