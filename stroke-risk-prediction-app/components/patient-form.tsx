"use client"

import type React from "react"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

interface PatientFormProps {
  onSubmit: (data: Record<string, any>) => void
  loading: boolean
}

export default function PatientForm({ onSubmit, loading }: PatientFormProps) {
  const [formData, setFormData] = useState({
    age: "",
    gender: "Male",
    hypertension: "No",
    heartDisease: "No",
    everMarried: "No",
    workType: "Private",
    residenceType: "Urban",
    smokingStatus: "Never smoked",
    avgGlucoseLevel: "",
    bmi: "",
  })

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target
    setFormData((prev) => ({ ...prev, [name]: value }))
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    onSubmit(formData)
  }

  return (
    <section id="patient-form" className="py-12">
      <Card className="border-0 shadow-lg">
        <CardHeader className="bg-gradient-to-r from-blue-50 to-white border-b border-blue-100">
          <CardTitle className="text-2xl text-gray-900">Patient Information</CardTitle>
          <p className="text-sm text-gray-600 mt-2">Please provide your health information below</p>
        </CardHeader>
        <CardContent className="pt-8">
          <form onSubmit={handleSubmit}>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Age */}
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">Age</label>
                <input
                  type="number"
                  name="age"
                  value={formData.age}
                  onChange={handleChange}
                  required
                  placeholder="Enter your age"
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>

              {/* Gender */}
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">Gender</label>
                <select
                  name="gender"
                  value={formData.gender}
                  onChange={handleChange}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option>Male</option>
                  <option>Female</option>
                  <option>Other</option>
                </select>
              </div>

              {/* Hypertension */}
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">Hypertension</label>
                <select
                  name="hypertension"
                  value={formData.hypertension}
                  onChange={handleChange}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option>No</option>
                  <option>Yes</option>
                </select>
              </div>

              {/* Heart Disease */}
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">Heart Disease</label>
                <select
                  name="heartDisease"
                  value={formData.heartDisease}
                  onChange={handleChange}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option>No</option>
                  <option>Yes</option>
                </select>
              </div>

              {/* Ever Married */}
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">Ever Married</label>
                <select
                  name="everMarried"
                  value={formData.everMarried}
                  onChange={handleChange}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option>No</option>
                  <option>Yes</option>
                </select>
              </div>

              {/* Work Type */}
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">Work Type</label>
                <select
                  name="workType"
                  value={formData.workType}
                  onChange={handleChange}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option>Private</option>
                  <option>Self-employed</option>
                  <option>Govt job</option>
                  <option>Children</option>
                  <option>Never worked</option>
                </select>
              </div>

              {/* Residence Type */}
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">Residence Type</label>
                <select
                  name="residenceType"
                  value={formData.residenceType}
                  onChange={handleChange}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option>Urban</option>
                  <option>Rural</option>
                </select>
              </div>

              {/* Smoking Status */}
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">Smoking Status</label>
                <select
                  name="smokingStatus"
                  value={formData.smokingStatus}
                  onChange={handleChange}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option>Never smoked</option>
                  <option>Formerly smoked</option>
                  <option>Smokes</option>
                  <option>Unknown</option>
                </select>
              </div>

              {/* Average Glucose Level */}
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">Average Glucose Level</label>
                <input
                  type="number"
                  name="avgGlucoseLevel"
                  value={formData.avgGlucoseLevel}
                  onChange={handleChange}
                  required
                  placeholder="mg/dL"
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>

              {/* BMI */}
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">BMI</label>
                <input
                  type="number"
                  name="bmi"
                  value={formData.bmi}
                  onChange={handleChange}
                  required
                  placeholder="kg/m²"
                  step="0.1"
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>
            </div>

            <Button
              type="submit"
              disabled={loading}
              className="w-full mt-8 bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 rounded-lg transition-colors"
            >
              {loading ? "Analyzing..." : "Predict Stroke Risk"}
            </Button>
          </form>
        </CardContent>
      </Card>
    </section>
  )
}
