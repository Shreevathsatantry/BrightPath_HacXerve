'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
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



export default function BasicArithmeticGame() {
  const [currentLevel, setCurrentLevel] = useState(0)
  const { t, i18n } = useTranslation()
  const levels = [
    { name: t("Level1"), operations: ['+', '-'], maxNumber: 10, questions: 5, time: 60 },
    { name: t("Level2"), operations: ['+', '-', '*'], maxNumber: 15, questions: 7, time: 90 },
    { name: t("Level3"), operations: ['+', '-', '*'], maxNumber: 20, questions: 10, time: 120 },
  ]
  const changeLanguage = (lang: string) => {
    i18n.changeLanguage(lang)
  }
  const [question, setQuestion] = useState('')
  const [answer, setAnswer] = useState('')
  const [score, setScore] = useState(0)
  const [timeLeft, setTimeLeft] = useState(levels[currentLevel].time)
  const [gameOver, setGameOver] = useState(false)
  const [feedback, setFeedback] = useState<string | null>(null)
  const [correctAnswers, setCorrectAnswers] = useState(0)
  const [incorrectAnswers, setIncorrectAnswers] = useState(0)

  useEffect(() => {
    if (timeLeft > 0 && !gameOver) {
      const timer = setTimeout(() => setTimeLeft(timeLeft - 1), 1000)
      return () => clearTimeout(timer)
    } else if (timeLeft === 0 && !gameOver) {
      storeGameData()
      endGame()
    }
  }, [timeLeft, gameOver])

  useEffect(() => {
    generateQuestion()
  }, [currentLevel])

  const generateQuestion = () => {
    const { operations, maxNumber } = levels[currentLevel]
    const operation = operations[Math.floor(Math.random() * operations.length)]
    let num1 = Math.floor(Math.random() * maxNumber) + 1
    let num2 = Math.floor(Math.random() * maxNumber) + 1

    if (operation === '*') {
      num1 = Math.floor(Math.random() * 5) + 1 // Smaller multipliers for simplicity
      num2 = Math.floor(Math.random() * 5) + 1
    }

    setQuestion(`${num1} ${operation} ${num2} = ?`)
    setAnswer('')
    setFeedback(null)
  }

  const handleSubmit = () => {
    const correctAnswer = eval(question.replace('=', '').replace('?', ''))
    const isCorrect = parseInt(answer) === correctAnswer

    if (isCorrect) {
      setFeedback('Correct!')
      setCorrectAnswers(correctAnswers + 1)
      setScore(score + 1)
    } else {
      setFeedback('Incorrect. Try again!')
      setIncorrectAnswers(incorrectAnswers + 1)
    }

    if (score + 1 === levels[currentLevel].questions) {
      if (currentLevel < levels.length - 1) {
        setCurrentLevel(currentLevel + 1)
        setScore(0)
        setTimeLeft(levels[currentLevel + 1].time)
      } else {
        endGame()
      }
    } else {
      setTimeout(generateQuestion, 1500)
    }
    setAnswer('')
  }

  const endGame = () => {
    setGameOver(true)
    storeGameData()
    if (score === levels[currentLevel].questions && currentLevel === levels.length - 1) {
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

  const getTodayKey = () => {
    const today = new Date()
    const year = today.getFullYear()
    const month = String(today.getMonth() + 1).padStart(2, '0')
    const day = String(today.getDate()).padStart(2, '0')
    return `${year}-${month}-${day}`
  }

  const storeGameData = () => {
    const todayKey = getTodayKey()
    const currentData = JSON.parse(localStorage.getItem('basicArithmeticGame') || '[]')

    const matchScore = correctAnswers - incorrectAnswers
    const averageScore = correctAnswers + incorrectAnswers > 0 ? matchScore / (correctAnswers) : 0

    const newMatch = {
      match: currentData.length > 0 ? currentData[currentData.length - 1].matches.length + 1 : 1,
      score: matchScore,
      correct: correctAnswers,
      incorrect: incorrectAnswers,
      totalQuestions: 5,
      averageScore: averageScore.toFixed(2),
    }

    if (currentData.length === 0 || currentData[currentData.length - 1].date !== todayKey) {
      currentData.push({
        date: todayKey,
        TotalMatches: 1,
        TotalAverageScore: newMatch.averageScore,
        matches: [newMatch]
      })
    } else {
      currentData[currentData.length - 1].matches.push(newMatch)
      currentData[currentData.length - 1].TotalMatches = currentData[currentData.length - 1].matches.length
      const totalAverageScore = (
        currentData[currentData.length - 1].matches.reduce((sum, match) => sum + parseFloat(match.averageScore), 0) /
        currentData[currentData.length - 1].matches.length
      ).toFixed(2)
      currentData[currentData.length - 1].TotalAverageScore = totalAverageScore
    }

    localStorage.setItem('basicArithmeticGame', JSON.stringify(currentData))
  }

  return (
    <div className="text-center">
      <h2 className="text-3xl font-bold mb-4 text-blue-600">{t("BasicArithmeticAdventure")}</h2>
      <div className="flex justify-between items-center mb-4">
        <p className="text-xl">{levels[currentLevel].name}</p>
        <p className="text-xl">{t("Score")} {score}/{levels[currentLevel].questions}</p>
        <p className="text-xl">{t("Time")} {timeLeft}s</p>
        <Dialog>
          <DialogTrigger asChild>
            <Button variant="outline">{t("HowtoPlay")}</Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>{t("HowtoPlayBasicArithmeticAdventure")}</DialogTitle>
              <DialogDescription>
                {t("SolvethearithmeticproblemstoprogressthroughlevelsAnswerallquestionscorrectlybeforetimerunsouttowin")}
              </DialogDescription>
            </DialogHeader>
          </DialogContent>
        </Dialog>
      </div>
      {!gameOver ? (
        <div className="mb-8">
          <div className="text-3xl mb-4">{question}</div>
          <input
            type="number"
            value={answer}
            onChange={(e) => setAnswer(e.target.value)}
            className="border-2 border-gray-300 rounded px-4 py-2 mb-4"
          />
          <Button
            onClick={handleSubmit}
            className="bg-green-500 hover:bg-green-600 text-white font-bold py-2 px-4 rounded ml-4"
          >
            {t("Submit")}
          </Button>
          {feedback && (
            <motion.div
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              className={`mt-4 text-xl ${feedback.includes('Correct') ? 'text-green-500' : 'text-red-500'}`}
            >
              {feedback}
              {feedback.includes('Correct') ? <CheckCircle className="inline ml-2" /> : <XCircle className="inline ml-2" />}
            </motion.div>
          )}
        </div>
      ) : (
        <div>
          <p className="text-xl mb-4">
          {t("GameOverYourfinalscore:")} {score}/{levels[currentLevel].questions}
          </p>
          <p className="text-xl mb-4">{t("CorrectAnswers")} {correctAnswers}</p>
          <p className="text-xl mb-4">{t("IncorrectAnswers")} {incorrectAnswers}</p>
          <Button
            onClick={() => {
              setCurrentLevel(0)
              setScore(0)
              setTimeLeft(levels[0].time)
              setGameOver(false)
              generateQuestion()
            }}
            className="bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded"
          >
            {t("PlayAgain")}
          </Button>
        </div>
      )}
    </div>
  )
}