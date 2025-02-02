"use client"

import { useState } from "react"
import { ArrowRight } from 'lucide-react'
import { useTranslation } from "react-i18next"
import { DotLottieReact } from '@lottiefiles/dotlottie-react' // Import the Lottie component

interface StoryInputProps {
  onSubmit: (storyData: string) => void
  triggerCanvasDownload: () => void
}

export default function StoryInput({ onSubmit, triggerCanvasDownload }: StoryInputProps) {
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false) // Added loading state
  const { t } = useTranslation()

  const handleSubmit = async () => {
    // Set loading state to true when submit is clicked
    setIsLoading(true)
    
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
    } finally {
      // Set loading state to false once the request is finished
      setIsLoading(false)
    }
  }

  return (
    <div className="flex flex-col items-center mt-4 w-full max-w-2xl relative">
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
          disabled={isLoading} // Disable the button while loading
        >
          {isLoading ? (
            <span className="loader">Loading...</span> // Placeholder text for loading
          ) : (
            <ArrowRight size={24} />
          )}
        </button>
      </div>

      {isLoading && (
        <div className="absolute inset-0 bg-gray-800 bg-opacity-50 flex justify-center items-center">
          {/* Lottie Animation when loading */}
          <DotLottieReact
            src="https://lottie.host/0a3953a1-9b48-4f73-802a-2907396bd130/3ZQdaFlPBZ.lottie"
            loop
            autoplay
            style={{ width: "150vw", height: "200vh" }} // Customize size of the animation
          />
        </div>
      )}
    </div>
  )
}