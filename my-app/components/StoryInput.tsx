"use client"

import { useState } from "react"
import { ArrowRight } from 'lucide-react'
import { useTranslation } from "react-i18next"

interface StoryInputProps {
  onSubmit: (storyData: string) => void
  triggerCanvasDownload: () => void
}

export default function StoryInput({ onSubmit, triggerCanvasDownload }: StoryInputProps) {
  const [input, setInput] = useState("")
  const { t } = useTranslation()

  const handleSubmit = async () => {
    // Trigger the canvas download
    triggerCanvasDownload()

    // Submit the story
    try {
      const response = await fetch("http://localhost:5000/StoryTeller", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text: input }),
      })
      const data = await response.json()
      console.log("Response from server:", data)
      onSubmit(data)
    } catch (error) {
      console.error("Error:", error)
    }
  }

  return (
    <div className="flex flex-col items-center mt-4 w-full max-w-2xl">
      <div className="flex items-center mt-4 w-full">
        <textarea
          className="flex-grow p-4 border-2 border-blue-400 rounded-l-lg focus:outline-none focus:border-blue-600 font-serif text-lg"
          placeholder={t("Onceuponatime")}
          rows={3}
          value={input}
          onChange={(e) => setInput(e.target.value)}
        />
        <button
          onClick={handleSubmit}
          className="bg-blue-500 text-white p-4 rounded-r-lg hover:bg-blue-600 focus:outline-none transition duration-300 ease-in-out transform hover:scale-105"
        >
          <ArrowRight size={24} />
        </button>
      </div>
    </div>
  )
}