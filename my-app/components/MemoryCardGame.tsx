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

const emojis = ['🐶', '🐱', '🐭', '🐹', '🐰', '🦊'] // 6 emojis for 12 squares (6 pairs)

interface Card {
  id: number
  emoji: string
  isFlipped: boolean
  isMatched: boolean
}

export default function MemoryCardGame() {
  const [cards, setCards] = useState<Card[]>([])
  const { t, i18n } = useTranslation()
  const changeLanguage = (lang: string) => {
    i18n.changeLanguage(lang)
  }
  const [flippedCards, setFlippedCards] = useState<number[]>([])
  const [matchedPairs, setMatchedPairs] = useState(0)
  const [moves, setMoves] = useState(0)
  const [gameOver, setGameOver] = useState(false)
  const [score, setScore] = useState(0)
  const [correctMatches, setCorrectMatches] = useState(0)
  const [incorrectMatches, setIncorrectMatches] = useState(0)

  useEffect(() => {
    startNewGame()
  }, [])

  const startNewGame = () => {
    const gameEmojis = emojis
      .sort(() => 0.5 - Math.random()) // Randomly shuffle the emojis
      .slice(0, 6) // Select 6 emojis to create pairs

    const gameCards: Card[] = [...gameEmojis, ...gameEmojis] // Create pairs of the selected emojis
      .sort(() => 0.5 - Math.random()) // Shuffle the pairs
      .map((emoji, index) => ({
        id: index,
        emoji,
        isFlipped: false,
        isMatched: false,
      }))
      
    setCards(gameCards)
    setFlippedCards([])
    setMatchedPairs(0)
    setMoves(0)
    setGameOver(false)
    setCorrectMatches(0)
    setIncorrectMatches(0)
  }

  const handleCardClick = (clickedCard: Card) => {
    if (flippedCards.length === 2 || clickedCard.isFlipped || clickedCard.isMatched) return

    const newFlippedCards = [...flippedCards, clickedCard.id]
    setFlippedCards(newFlippedCards)
    setCards(cards.map(card =>
      card.id === clickedCard.id ? { ...card, isFlipped: true } : card
    ))

    if (newFlippedCards.length === 2) {
      setMoves(moves + 1)
      const [firstCardId, secondCardId] = newFlippedCards
      const firstCard = cards.find(card => card.id === firstCardId)
      const secondCard = cards.find(card => card.id === secondCardId)

      if (firstCard && secondCard && firstCard.emoji === secondCard.emoji) {
        setMatchedPairs(matchedPairs + 1)
        setCorrectMatches(correctMatches + 1)
        setCards(cards.map(card =>
          card.id === firstCardId || card.id === secondCardId
            ? { ...card, isMatched: true }
            : card
        ))
        setFlippedCards([])

        if (matchedPairs + 1 === 6) { // 6 pairs total
          endGame()
        }
      } else {
        setIncorrectMatches(incorrectMatches + 1)
        setTimeout(() => {
          setCards(cards.map(card =>
            card.id === firstCardId || card.id === secondCardId
              ? { ...card, isFlipped: false }
              : card
          ))
          setFlippedCards([])
        }, 1000)
      }
    }
  }

  const endGame = () => {
    setGameOver(true)
    const finalScore = Math.max(100 - moves * 2, 0) // Score decreases as moves increase
    setScore(finalScore)
  
    // Store game statistics in local storage under a single key 'MemoryCardGame'
    const gameData = {
      totalScore: finalScore,
      totalCorrectAnswers: correctMatches,
      totalIncorrectAnswers: incorrectMatches,
      gameOver: true,
    }
    
    localStorage.setItem('MemoryCardGame', JSON.stringify(gameData))
  
    if (finalScore === 0) {
      triggerConfetti() // Trigger confetti if you want
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
      <h2 className="text-3xl font-bold mb-4 text-blue-600">{t("MemoryCardGame")}</h2>
      <div className="flex justify-between items-center mb-4">
        <p className="text-xl">{t("Score")} {score}</p>
        <p className="text-xl">{t("Moves")} {moves}</p>
        <Dialog>
          <DialogTrigger asChild>
            <Button variant="outline">{t("HowtoPlay")}</Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>{t("HowtoPlayMemoryCardGame")}</DialogTitle>
              <DialogDescription>
                {t("Clickonacardtoflipitover")}<br />
                {t("Trytofindmatchingpairsofcards")}<br />
                {t("Matchallpairstocompletethegame")}<br />
                {t("Therearenolevelsinthisgamejustcompleteallpairsinasfewmovesaspossible")}<br />
                {t("Havefunandimproveyourmemory")}
              </DialogDescription>
            </DialogHeader>
          </DialogContent>
        </Dialog>
      </div>
      <div className="mb-8">
        <div
          className="flex justify-center flex-wrap gap-4"
          style={{
            maxWidth: `${Math.ceil(Math.sqrt(12)) * 70}px`,
            margin: '0 auto'
          }}
        >
          {cards.map((card) => (
            <motion.div
              key={card.id}
              className={`w-16 h-16 bg-blue-500 rounded-lg flex items-center justify-center cursor-pointer ${card.isFlipped || card.isMatched ? 'bg-white' : ''}`}
              onClick={() => handleCardClick(card)}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              {card.isFlipped || card.isMatched ? (
                <span className="text-3xl">{card.emoji}</span>
              ) : null}
            </motion.div>
          ))}
        </div>
      </div>
      {gameOver ? (
        <div>
          <p className="text-xl mb-4">{t("CongratulationsYoumatchedallthecards")}</p>
          <Button
            onClick={startNewGame}
            className="bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded"
          >
            {t("PlayAgain")}
          </Button>
        </div>
      ) : (
        <Button
          onClick={startNewGame}
          className="bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded"
        >
          {t("NewGame")}
        </Button>
      )}
    </div>
  )
}
