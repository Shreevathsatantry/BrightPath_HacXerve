import React, { useState, useRef, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Mic, Square } from 'lucide-react'
import * as franc from 'franc-min'

interface VoiceRecorderProps {
  onTranscript: (transcript: string, language: string) => void;
}

export default function VoiceRecorder({ onTranscript }: VoiceRecorderProps) {
  const [isRecording, setIsRecording] = useState(false)
  const [detectedLanguage, setDetectedLanguage] = useState<string>('en')
  const recognitionRef = useRef<SpeechRecognition | null>(null)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])

  useEffect(() => {
    if (typeof window !== 'undefined' && ('SpeechRecognition' in window || 'webkitSpeechRecognition' in window)) {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition
      recognitionRef.current = new SpeechRecognition()
      recognitionRef.current.continuous = true
      recognitionRef.current.interimResults = true

      recognitionRef.current.onresult = (event) => {
        const transcript = Array.from(event.results)
          .map(result => result[0].transcript)
          .join('')

        const language = franc.franc(transcript)
        setDetectedLanguage(language)
        onTranscript(transcript, language)
      }
    }
  }, [onTranscript])

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      mediaRecorderRef.current = new MediaRecorder(stream)
      chunksRef.current = []

      mediaRecorderRef.current.ondataavailable = (event) => {
        chunksRef.current.push(event.data)
      }

      mediaRecorderRef.current.onstop = () => {
        const audioBlob = new Blob(chunksRef.current, { type: 'audio/wav' })
        const url = URL.createObjectURL(audioBlob)
        const a = document.createElement('a')
        a.href = url
        a.download = `audio.wav`
        document.body.appendChild(a)
        a.click()
        document.body.removeChild(a)
        URL.revokeObjectURL(url)
      }

      mediaRecorderRef.current.start()
      if (recognitionRef.current) {
        recognitionRef.current.start()
      }
      setIsRecording(true)
    } catch (err) {
      console.error('Error accessing microphone:', err)
    }
  }

  const stopRecording = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop()
      mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop())
    }
    if (recognitionRef.current) {
      recognitionRef.current.stop()
    }
    setIsRecording(false)
  }

  return (
    <div className="mt-4 flex justify-center">
      <motion.div
        animate={{ scale: isRecording ? 1.1 : 1 }}
        transition={{ duration: 0.3 }}
      >
        <motion.button
          onClick={isRecording ? stopRecording : startRecording}
          className={`rounded-full p-4 ${isRecording ? 'bg-green-400 hover:bg-green-500' : 'bg-blue-400 hover:bg-blue-500'} font-comic text-2xl text-white shadow-lg`}
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.9 }}
          animate={isRecording ? { scale: [1, 1.2, 1], rotate: [0, 10, -10, 0] } : {}}
          transition={{ duration: 0.5, repeat: isRecording ? Infinity : 0 }}
        >
          {isRecording ? <Square size={24} /> : <Mic size={24} />}
        </motion.button>
      </motion.div>
    </div>
  )
}