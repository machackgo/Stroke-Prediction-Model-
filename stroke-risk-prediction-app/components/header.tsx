"use client"

interface HeaderProps {
  onStartClick: () => void
}

export default function Header({ onStartClick }: HeaderProps) {
  return (
    <header className="bg-white border-b border-blue-100">
      <nav className="container mx-auto px-4 py-4 flex justify-between items-center">
        <div className="text-2xl font-bold text-blue-600">HealthCheck</div>
        <div className="text-sm text-gray-600">Educational Tool</div>
      </nav>

      <div className="container mx-auto px-4 py-16 md:py-24 text-center">
        <h1 className="text-4xl md:text-5xl font-bold text-gray-900 mb-4 text-balance">Stroke Risk Prediction</h1>
        <p className="text-lg text-gray-600 mb-8 max-w-2xl mx-auto text-balance">
          Educational tool based on a machine learning model (Decision Tree).
          <span className="font-semibold"> Not for real medical diagnosis.</span>
        </p>
        <button
          onClick={onStartClick}
          className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-8 rounded-lg transition-colors"
        >
          Start Assessment
        </button>
      </div>
    </header>
  )
}
