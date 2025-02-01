"use client";

import React, { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import dynamic from "next/dynamic";
import Background from "../components/Background";
import Sidebar from "../components/Sidebar";
import FeatureCard from "../components/FeatureCard";
import HeroSection from "../components/HeroSection";
import PopupLanguageSelector from "../components/PopupLanguageSelector";
import "../i18n"; // Import i18n configuration

// Dynamically import client-only components to avoid hydration issues
const AnimatedCharacter = dynamic(() => import("../components/AnimatedCharacter"), { ssr: false });
const FloatingAnimation = dynamic(() => import("../components/FloatingAnimation"), { ssr: false });

export default function LandingPage() {
  const { t, i18n } = useTranslation();
  const [isMounted, setIsMounted] = useState(false);
  const [translatedText, setTranslatedText] = useState({});

  // Prevent hydration mismatch by rendering after mount
  useEffect(() => {
    setIsMounted(true);
    setTranslatedText({
      brightPathGames: t("brightPathGames"),
      brightPathGamesDesc: t("brightPathGamesDesc"),
      progressTracker: t("progressTracker"),
      progressTrackerDesc: t("progressTrackerDesc"),
      brightPathStories: t("brightPathStories"),
      brightPathStoriesDesc: t("brightPathStoriesDesc"),
      socialSkillsBuilder: t("socialSkillsBuilder"),
      socialSkillsBuilderDesc: t("socialSkillsBuilderDesc"),
    });
  }, [i18n.language]);

  // Prevent hydration errors by returning null until mounted
  if (!isMounted) return null;

  return (
    <div className="relative min-h-screen overflow-hidden">
      <Background />
      <Sidebar />
      <PopupLanguageSelector />
      <main className="relative z-10 p-8 ml-10">
        <HeroSection />
        <div className="grid grid-cols-1 gap-8 mt-16 md:grid-cols-2 lg:grid-cols-4">
          <FeatureCard
            title={translatedText.brightPathGames || "Loading..."}
            description={translatedText.brightPathGamesDesc || "..."}
            icon="🎮"
            color="bg-yellow-300"
          />
          <FeatureCard
            title={translatedText.progressTracker || "Loading..."}
            description={translatedText.progressTrackerDesc || "..."}
            icon="📊"
            color="bg-green-300"
          />
          <FeatureCard
            title={translatedText.brightPathStories || "Loading..."}
            description={translatedText.brightPathStoriesDesc || "..."}
            icon="📚"
            color="bg-pink-300"
          />
          <FeatureCard
            title={translatedText.socialSkillsBuilder || "Loading..."}
            description={translatedText.socialSkillsBuilderDesc || "..."}
            icon="🤝"
            color="bg-purple-300"
          />
        </div>

        {/* Animated characters */}
        <AnimatedCharacter character="🐶" style={{ top: "20%", left: "10%" }} />
        <AnimatedCharacter character="🐱" style={{ top: "60%", right: "15%" }} />
        <AnimatedCharacter character="🐰" style={{ top: "40%", left: "80%" }} />

        {/* Floating animations */}
        <FloatingAnimation animation="float">
          <span className="text-4xl">🎈</span>
        </FloatingAnimation>
        <FloatingAnimation animation="twinkle">
          <span className="text-4xl">⭐</span>
        </FloatingAnimation>
      </main>
    </div>
  );
}
