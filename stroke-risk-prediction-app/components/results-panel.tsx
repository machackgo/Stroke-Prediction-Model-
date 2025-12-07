"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

interface ResultsPanelProps {
  results: {
    riskLevel: "low" | "medium" | "high"
    probability: number
  }
}

export default function ResultsPanel({ results }: ResultsPanelProps) {
  const riskColors = {
    low: { bg: "bg-green-50", text: "text-green-700", badge: "bg-green-100 text-green-800" },
    medium: { bg: "bg-yellow-50", text: "text-yellow-700", badge: "bg-yellow-100 text-yellow-800" },
    high: { bg: "bg-red-50", text: "text-red-700", badge: "bg-red-100 text-red-800" },
  }

  const colors = riskColors[results.riskLevel]

  return (
    <section className="py-12">
      <Card className={`border-0 shadow-lg ${colors.bg}`}>
        <CardHeader className="border-b border-gray-200">
          <CardTitle className="text-2xl text-gray-900">Assessment Results</CardTitle>
        </CardHeader>
        <CardContent className="pt-8">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {/* Risk Badge */}
            <div className="flex flex-col justify-center items-center">
              <div className={`px-6 py-3 rounded-lg font-bold text-lg ${colors.badge}`}>
                {results.riskLevel.toUpperCase()} RISK
              </div>
              <p className={`mt-4 text-sm ${colors.text}`}>Based on your information</p>
            </div>

            {/* Probability Bar */}
            <div>
              <div className="mb-4">
                <div className="flex justify-between items-center mb-2">
                  <label className="font-semibold text-gray-900">Stroke Probability</label>
                  <span className={`text-2xl font-bold ${colors.text}`}>{results.probability}%</span>
                </div>
                <div className="w-full bg-gray-300 rounded-full h-3">
                  <div
                    className={`h-3 rounded-full transition-all ${
                      results.riskLevel === "low"
                        ? "bg-green-500"
                        : results.riskLevel === "medium"
                          ? "bg-yellow-500"
                          : "bg-red-500"
                    }`}
                    style={{ width: `${results.probability}%` }}
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Key Factors */}
          <div className="mt-8 pt-8 border-t border-gray-300">
            <h3 className="font-bold text-gray-900 mb-4">Key Factors to Consider:</h3>
            <ul className="space-y-3">
              <li className="text-gray-700 flex items-start">
                <span className="text-blue-600 font-bold mr-3">•</span>
                <span>Age is a significant risk factor. The older you are, the higher your stroke risk.</span>
              </li>
              <li className="text-gray-700 flex items-start">
                <span className="text-blue-600 font-bold mr-3">•</span>
                <span>Conditions like hypertension and heart disease significantly increase stroke risk.</span>
              </li>
              <li className="text-gray-700 flex items-start">
                <span className="text-blue-600 font-bold mr-3">•</span>
                <span>Maintaining a healthy glucose level and BMI are important preventive measures.</span>
              </li>
              <li className="text-gray-700 flex items-start">
                <span className="text-blue-600 font-bold mr-3">•</span>
                <span>Consult with a healthcare professional for personalized medical advice.</span>
              </li>
            </ul>
          </div>

          <div className="mt-6 p-4 bg-blue-100 border border-blue-300 rounded-lg">
            <p className="text-sm text-blue-900">
              <strong>Disclaimer:</strong> This tool is for educational purposes only and should not be used for
              self-diagnosis or as a substitute for professional medical advice. Always consult a healthcare provider
              for medical concerns.
            </p>
          </div>
        </CardContent>
      </Card>
    </section>
  )
}
