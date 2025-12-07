"use client"

import { useState } from "react"
import Header from "@/components/header"
import PatientForm from "@/components/patient-form"
import ResultsPanel from "@/components/results-panel"

export default function Home() {
  const [results, setResults] = useState<{
    riskLevel: "low" | "medium" | "high"
    probability: number
  } | null>(null)
  const [loading, setLoading] = useState(false)

  const handlePredict = async (formData: Record<string, any>) => {
    setLoading(true)
    try {
      const response = await predictStrokeRisk(formData)
      console.log("[v0] Prediction response:", response)
      setResults(response)
    } catch (error) {
      console.error("[v0] Prediction error:", error)
      setResults(null)
      alert(
        "Prediction failed. Is the FastAPI server running on http://127.0.0.1:8000 ?"
      )
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="min-h-screen bg-gradient-to-b from-blue-50 to-white">
      <Header
        onStartClick={() => {
          const formElement = document.getElementById("patient-form")
          formElement?.scrollIntoView({ behavior: "smooth" })
        }}
      />

      <div className="container mx-auto px-4 py-12">
        <PatientForm onSubmit={handlePredict} loading={loading} />
        {results && <ResultsPanel results={results} />}
      </div>
    </main>
  )
}

// Call Python backend API for real prediction
async function predictStrokeRisk(formData: Record<string, any>) {
  // Build payload in the format expected by the FastAPI /predict endpoint
  const payload = {
    age: Number(formData.age ?? 0),
    gender: String(formData.gender ?? "Male"),
    hypertension:
      formData.hypertension === "Yes" || formData.hypertension === 1 ? 1 : 0,
    heart_disease:
      formData.heartDisease === "Yes" ||
      formData.heart_disease === 1 ||
      formData.heartDisease === 1
        ? 1
        : 0,
    ever_married: String(formData.everMarried ?? formData.ever_married ?? "No"),
    work_type: String(formData.workType ?? formData.work_type ?? "Private"),
    residence_type: String(
      formData.residenceType ?? formData.residence_type ?? "Urban"
    ),
    smoking_status: String(
      formData.smokingStatus ?? formData.smoking_status ?? "never smoked"
    ),
    avg_glucose_level: Number(
      formData.avgGlucoseLevel ??
        formData.averageGlucoseLevel ??
        formData.avg_glucose_level ??
        0
    ),
    bmi: Number(formData.bmi ?? 0),
  }

  console.log("[v0] Sending payload to API:", payload)

  const res = await fetch("http://127.0.0.1:8000/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json", "Cache-Control": "no-cache" },
    cache: "no-store",
    body: JSON.stringify(payload),
  })

  if (!res.ok) {
    console.error("Prediction API error status:", res.status)
    throw new Error("Prediction API error")
  }

  const data = (await res.json()) as {
    stroke_probability: number
    prediction: number
  }

  const probabilityPercent = data.stroke_probability * 100

  // Updated thresholds (make MEDIUM easier to reach, HIGH a bit stricter):
  // - LOW:    < 35%
  // - MEDIUM: 35% - 84%
  // - HIGH:   >= 85%
  let riskLevel: "low" | "medium" | "high" = "low"
  if (probabilityPercent >= 85) riskLevel = "high"
  else if (probabilityPercent >= 35) riskLevel = "medium"

  // ResultsPanel expects { riskLevel, probability } where probability is %
  return {
    riskLevel,
    probability: Math.round(probabilityPercent),
  }
}
