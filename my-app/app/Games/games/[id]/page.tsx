'use client'

import { useRouter } from 'next/navigation'
import { motion } from 'framer-motion'
import { Button } from "@/components/ui/button"
import ColorMatchingGame from '@/components/ColorMatchingGame'//tdone
import EmotionRecognitionGame from '@/components/EmotionRecognitionGame'//tdone
import ShapeSortingGame from '@/components/ShapeSortingGame'//tdone
import MusicalPatternsGame from '@/components/MusicalPatternsGame'//tdone
import MemoryCardGame from '@/components/MemoryCardGame'//tdone
// import WordFormationGame from '@/components/WordFormationGame'
import GrammarDetectiveGame from '@/components/GrammarDetectiveGame'//tdone
import BasicArithmeticGame from '@/components/BasicArithmeticGame'//tdone
import ShapePatternGame from '@/components/ShapePatternGame'
import { useTranslation } from 'react-i18next'//tdone

export default function GamePage({ params }: { params: { id: string } }) {
  const router = useRouter()
  const { t, i18n } = useTranslation()
  const changeLanguage = (lang: string) => {
    i18n.changeLanguage(lang)
  }

  const renderGame = () => {
    switch (params.id) {
      case 'color-matching':
        return <ColorMatchingGame />
      case 'emotion-recognition':
        return <EmotionRecognitionGame />
      case 'shape-sorting':
        return <ShapeSortingGame />
      case 'musical-patterns':
        return <MusicalPatternsGame />
      case 'memory-card':
        return <MemoryCardGame />
      // case 'word-formation':
      //   return <WordFormationGame />
      case 'grammar-detective':
        return <GrammarDetectiveGame />
      case 'basic-arithmetic':
        return <BasicArithmeticGame />
      case 'shape-pattern':
        return <ShapePatternGame />
      default:
        return <div>Game not found</div>
    }
  }

  return (
    <div className="container mx-auto px-4 py-8">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="bg-white/90 backdrop-blur-sm rounded-3xl shadow-xl p-8"
      >
        {renderGame()}
        <div className="mt-8 text-center">
          <Button
            onClick={() => router.push('/Games')}
            className="bg-blue-500 hover:bg-blue-600 text-white font-bold py-3 px-6 rounded-full text-lg shadow-lg hover:shadow-xl transition-all duration-300"
          >
            {t("BackToGames")}
          </Button>
        </div>
      </motion.div>
    </div>
  )
}

