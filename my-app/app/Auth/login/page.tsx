"use client"

import type React from "react"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/card"
import Link from "next/link"

export default function LoginPage() {
  const [isLoading, setIsLoading] = useState(false)
  const [formData, setFormData] = useState({
    email: "",
    password: "",
  })

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsLoading(true)
    // Simulate API call
    await new Promise((resolve) => setTimeout(resolve, 2000))
    setIsLoading(false)
  }

  const handleInputChange = (field: string, value: string) => {
    setFormData((prev) => ({ ...prev, [field]: value }))
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-400 via-blue-500 to-purple-600 relative overflow-hidden">
      {/* Animated Background Elements */}
      <div className="absolute inset-0 pointer-events-none">
        {/* Floating Clouds */}
        <div className="absolute top-16 right-16 w-36 h-24 bg-white rounded-full opacity-80 animate-float"></div>
        <div className="absolute top-1/3 left-12 w-28 h-18 bg-white rounded-full opacity-70 animate-float-delayed"></div>
        <div className="absolute bottom-32 right-1/4 w-32 h-20 bg-white rounded-full opacity-60 animate-float"></div>

        {/* Twinkling Stars */}
        <div className="absolute top-1/4 left-1/3 w-3 h-3 bg-yellow-300 rounded-full animate-twinkle"></div>
        <div className="absolute top-16 left-1/4 w-2 h-2 bg-yellow-400 rounded-full animate-twinkle-delayed"></div>
        <div className="absolute bottom-1/4 left-1/5 w-4 h-4 bg-yellow-300 rounded-full animate-twinkle"></div>
        <div className="absolute top-2/3 right-1/3 w-2 h-2 bg-yellow-400 rounded-full animate-twinkle-delayed"></div>

        {/* Cute Animal Characters */}
        <div className="absolute top-1/3 right-8 text-6xl animate-bounce-gentle">🐱</div>
        <div className="absolute bottom-1/3 left-8 text-5xl animate-bounce-gentle-delayed">🐶</div>
        <div className="absolute top-1/2 right-1/12 text-4xl animate-bounce-gentle">🐰</div>
      </div>

      {/* Main Content */}
      <div className="flex items-center justify-center min-h-screen p-4">
        <Card className="w-full max-w-md bg-white/95 backdrop-blur-sm shadow-2xl border-0 animate-slide-up">
          <CardHeader className="text-center space-y-4">
            <div className="mx-auto w-16 h-16 bg-gradient-to-r from-green-400 to-blue-400 rounded-full flex items-center justify-center text-2xl animate-pulse">
              🌟
            </div>
            <CardTitle className="text-3xl font-bold bg-gradient-to-r from-green-600 to-blue-600 bg-clip-text text-transparent">
              Welcome Back!
            </CardTitle>
            <CardDescription className="text-lg text-gray-600">
              Ready to continue your learning journey? 🎓
            </CardDescription>
          </CardHeader>

          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-6">
              <div className="space-y-2">
                <label className="text-sm font-semibold text-gray-700 flex items-center gap-2">
                  <span className="text-lg">📧</span>
                  Email Address
                </label>
                <Input
                  type="email"
                  placeholder="Enter your email"
                  value={formData.email}
                  onChange={(e) => handleInputChange("email", e.target.value)}
                  className="h-12 text-lg border-2 border-green-200 focus:border-green-400 rounded-xl transition-all duration-300 focus:scale-105"
                  required
                />
              </div>

              <div className="space-y-2">
                <label className="text-sm font-semibold text-gray-700 flex items-center gap-2">
                  <span className="text-lg">🔑</span>
                  Password
                </label>
                <Input
                  type="password"
                  placeholder="Enter your password"
                  value={formData.password}
                  onChange={(e) => handleInputChange("password", e.target.value)}
                  className="h-12 text-lg border-2 border-blue-200 focus:border-blue-400 rounded-xl transition-all duration-300 focus:scale-105"
                  required
                />
              </div>

              <div className="text-right">
                <Link
                  href="/forgot-password"
                  className="text-sm text-blue-600 hover:text-blue-800 transition-colors duration-200 hover:underline"
                >
                  Forgot your password? 🤔
                </Link>
              </div>

              <Button
                type="submit"
                disabled={isLoading}
                className="w-full h-14 text-lg font-bold bg-gradient-to-r from-green-400 to-blue-400 hover:from-green-500 hover:to-blue-500 text-white rounded-xl shadow-lg hover:shadow-xl transition-all duration-300 hover:scale-105 disabled:opacity-50"
              >
                {isLoading ? (
                  <div className="flex items-center gap-2">
                    <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                    Signing you in...
                  </div>
                ) : (
                  <span className="flex items-center gap-2">
                    Let's Learn! <span className="text-xl">🚀</span>
                  </span>
                )}
              </Button>
            </form>

            <div className="mt-6 text-center">
              <p className="text-gray-600">
                New to BrightPath?{" "}
                <Link
                  href="/signup"
                  className="font-semibold text-green-600 hover:text-green-800 transition-colors duration-200 hover:underline"
                >
                  Create an account! ✨
                </Link>
              </p>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
