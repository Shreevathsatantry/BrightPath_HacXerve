'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import confetti from 'canvas-confetti'
import { Button } from "@/components/ui/button"
import { useTranslation } from "react-i18next"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { CheckCircle, XCircle } from 'lucide-react'

const shapes = ['🔴', '🔵', '🟡', '🟢', '🟣', '🟠']



export default function ShapePatternGame() {
  const [currentLevel, setCurrentLevel] = useState(0)
  const { t, i18n } = useTranslation()
  const levels = [
    { name: t("Level1"), patternLength: 4, time: 60 },
    { name: t("Level2"), patternLength: 6, time: 90 },
    { name: t("Level3"), patternLength: 8, time: 120 },
  ]
  const changeLanguage = (lang: string) => {
    i18n.changeLanguage(lang)
  }
  const [pattern, setPattern] = useState<string[]>([])
  const [userPattern, setUserPattern] = useState<string[]>([])
  const [score, setScore] = useState(0)
  const [totalCorrect, setTotalCorrect] = useState(0)
  const [totalIncorrect, setTotalIncorrect] = useState(0)
  const [timeLeft, setTimeLeft] = useState(levels[currentLevel].time)
  const [gameOver, setGameOver] = useState(false)
  const [showPattern, setShowPattern] = useState(true)
  const [feedback, setFeedback] = useState<string | null>(null)

  // Load the previous score and counts from localStorage on first render
  useEffect(() => {
    const storedGameData = JSON.parse(localStorage.getItem('gameData') || '{}')
    if (storedGameData) {
      setTotalCorrect(storedGameData.totalCorrect || 0)
      setTotalIncorrect(storedGameData.totalIncorrect || 0)
      setScore(storedGameData.score || 0)
    }
  }, [])

  // Save the game data (totalCorrect, totalIncorrect, score) to localStorage
  useEffect(() => {
    const gameData = { totalCorrect, totalIncorrect, score }
    localStorage.setItem('Patternmatchgame', JSON.stringify(gameData))
  }, [totalCorrect, totalIncorrect, score])

  useEffect(() => {
    if (timeLeft > 0 && !gameOver) {
      const timer = setTimeout(() => setTimeLeft(timeLeft - 1), 1000)
      return () => clearTimeout(timer)
    } else if (timeLeft === 0 && !gameOver) {
      endGame()
    }
  }, [timeLeft, gameOver])

  useEffect(() => {
    generatePattern()
  }, [currentLevel])

  const generatePattern = () => {
    const newPattern = Array.from({ length: levels[currentLevel].patternLength }, () => 
      shapes[Math.floor(Math.random() * shapes.length)]
    )
    setPattern(newPattern)
    setUserPattern([])
    setShowPattern(true)
    setFeedback(null)
    setTimeout(() => setShowPattern(false), 5000)
  }

  const handleShapeClick = (shape: string) => {
    if (showPattern) return

    const newUserPattern = [...userPattern, shape]
    setUserPattern(newUserPattern)

    if (newUserPattern.length === pattern.length) {
      const isCorrect = newUserPattern.every((shape, index) => shape === pattern[index])
      setFeedback(isCorrect ? "Correct! Great job!" : "Oops! Try again!")
      if (isCorrect) {
        setScore(score + 1)
        setTotalCorrect(totalCorrect + 1) // Update correct count
        if (score + 1 === 3) {
          if (currentLevel < levels.length - 1) {
            setCurrentLevel(currentLevel + 1)
            setScore(0)
            setTimeLeft(levels[currentLevel + 1].time)
          } else {
            endGame()
          }
        } else {
          setTimeout(generatePattern, 1500)
        }
      } else {
        setTotalIncorrect(totalIncorrect + 1) // Update incorrect count
        setTimeout(() => {
          setFeedback(null)
          generatePattern()
        }, 1500)
      }
    }
  }

  const endGame = () => {
    setGameOver(true)
    if (score === 3 && currentLevel === levels.length - 1) {
      triggerConfetti()
    }
  }

  const triggerConfetti = () => {
    confetti({
      particleCount: 100,
      spread: 70,
      origin: { y: 0.6 }
    })
  }

  return (
    <div className="text-center">
      <h2 className="text-3xl font-bold mb-4 text-blue-600">{t("ShapePatternPuzzle")}</h2>
      <div className="flex justify-between items-center mb-4">
        <p className="text-xl">{levels[currentLevel].name}</p>
        <p className="text-xl">{t("Score")} {score}/3</p>
        <p className="text-xl">{t("Time")} {timeLeft}s</p>
        <p className="text-xl">{t("TotalCorrect")} {totalCorrect}</p>
        <p className="text-xl">{t("TotalIncorrect")} {totalIncorrect}</p>
        <Dialog>
          <DialogTrigger asChild>
            <Button variant="outline">{t("HowtoPlay")}</Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>{("HowtoPlayShapePatternPuzzle")}</DialogTitle>
              <DialogDescription>
                {t("Memorizethepatternofshapesshown")}<br/>
                {t("Recreatethepatternbyclickingontheshapesinorder")}<br/>
                {t("Completeeachpatternscorrectlytoadvancetothenextlevel")}<br/>
                {t("Therearelevelswithincreasingdifficulty")}<br/>
                {t("Winthegamebycompletingalllevelsbeforetimerunsout")}<br/>
                {t("Havefunandimproveyourmemoryandpatternrecognitionskills")}
              </DialogDescription>
            </DialogHeader>
          </DialogContent>
        </Dialog>
      </div>
      {feedback && (
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -20 }}
          className={`text-xl font-bold mb-4 ${feedback.includes('Correct') ? 'text-green-500' : 'text-red-500'}`}
        >
          {feedback}
          {feedback.includes('Correct') ? <CheckCircle className="inline ml-2" /> : <XCircle className="inline ml-2" />}
        </motion.div>
      )}
      {!gameOver ? (
        <div className="mb-8">
          <div className="flex justify-center gap-4 mb-4">
            {showPattern ? (
              pattern.map((shape, index) => (
                <div key={index} className="text-4xl">{shape}</div>
              ))
            ) : (
              <div className="text-2xl">{t("Nowrecreatethepattern")}</div>
            )}
          </div>
          <div className="flex justify-center gap-4 mb-4">
            {shapes.map((shape, index) => (
              <motion.button
                key={index}
                className="text-4xl p-4 bg-gray-200 rounded-full shadow-lg"
                onClick={() => handleShapeClick(shape)}
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                disabled={showPattern}
              >
                {shape}
              </motion.button>
            ))}
          </div>
          <div className="flex justify-center gap-2">
            {userPattern.map((shape, index) => (
              <div key={index} className="text-4xl">{shape}</div>
            ))}
          </div>
        </div>
      ) : (
        <div>
          <p className="text-xl mb-4">
            {t("GameOverYourfinalscore")} {score}/3
          </p>
          {score === 3 && currentLevel === levels.length - 1 ? (
            <motion.div
              className="text-2xl font-bold text-green-500"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
            >
              🎉 {t("CongratulationsYouvecompletedalllevels")} 🎉
            </motion.div>
          ) : (
            <Button 
              onClick={() => {
                setCurrentLevel(0)
                setScore(0)
                setTimeLeft(levels[0].time)
                setGameOver(false)
                setTotalCorrect(0)
                setTotalIncorrect(0)
                generatePattern()
              }}
              className="bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded"
            >
              {t("PlayAgain")}
            </Button>
          )}
        </div>
      )}
    </div>
  )
}