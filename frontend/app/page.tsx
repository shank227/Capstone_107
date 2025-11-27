"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Loader2, Mail, Shield, AlertTriangle, Brain, Sparkles, BarChart3, Eye } from "lucide-react"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Badge } from "@/components/ui/badge"

interface ExplanationData {
  prediction: string
  confidence: number
  probabilities: {
    legitimate: number
    phishing: number
  }
  word_importance: Record<string, number>
  top_contributing_words: Array<{ word: string; importance: number }>
}

export default function EmailClassifier() {
  const [email, setEmail] = useState("")
  const [result, setResult] = useState<string | null>(null)
  const [explanation, setExplanation] = useState<ExplanationData | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const classifyEmail = async () => {
    if (!email.trim()) {
      setError("Please enter an email to classify")
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)
    setExplanation(null)

    try {
      const response = await fetch("http://localhost:5000/classify", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ email: email.trim() }),
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      setResult(data.prediction)
      setExplanation(data)
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Failed to classify email. Make sure the backend server is running on localhost:5000",
      )
    } finally {
      setLoading(false)
    }
  }

  const isLegitimate = result === "Legitimate Email"
  const isPhishing = result === "Phishing Email"

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-muted/20">
      <div className="container mx-auto px-4 py-8 max-w-6xl">
        {/* Header Section */}
        <div className="text-center space-y-4 mb-8">
          <div className="flex items-center justify-center gap-3">
            <div className="p-3 rounded-full bg-primary/10">
              <Mail className="h-8 w-8 text-primary" />
            </div>
            <h1 className="text-4xl font-bold bg-gradient-to-r from-foreground to-foreground/70 bg-clip-text text-transparent">
              Phishing Email Detector
            </h1>
          </div>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            Advanced AI-powered email analysis with explainable insights to help you identify potential phishing attempts
          </p>
        </div>

        <div className="grid gap-6 lg:grid-cols-3">
          {/* Left Column - Input Section */}
          <div className="lg:col-span-2 space-y-6">
            <Card className="shadow-lg border-2">
              <CardHeader className="pb-4">
                <CardTitle className="flex items-center gap-2 text-2xl">
                  <Mail className="h-6 w-6 text-primary" />
                  Email Analysis
                </CardTitle>
                <CardDescription className="text-base">
                  Paste the email content you want to analyze below
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <label htmlFor="email-input" className="text-sm font-semibold text-foreground">
                    Email Content
                  </label>
                  <Textarea
                    id="email-input"
                    placeholder="Paste the email content here...&#10;&#10;Example:&#10;Dear Customer,&#10;Your account has been suspended. Click here to verify your identity..."
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    rows={12}
                    className="resize-none text-sm leading-relaxed"
                  />
                </div>

                <Button 
                  onClick={classifyEmail} 
                  disabled={loading || !email.trim()} 
                  className="w-full h-12 text-base font-semibold"
                  size="lg"
                >
                  {loading ? (
                    <>
                      <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                      Analyzing Email...
                    </>
                  ) : (
                    <>
                      <Sparkles className="mr-2 h-5 w-5" />
                      Analyze Email
                    </>
                  )}
                </Button>

                {error && (
                  <Alert variant="destructive" className="border-2">
                    <AlertTriangle className="h-5 w-5" />
                    <AlertDescription className="font-medium">{error}</AlertDescription>
                  </Alert>
                )}
              </CardContent>
            </Card>

            {/* Classification Result Section */}
            {result && explanation && (
              <Card className="shadow-lg border-2">
                <CardHeader className="pb-4">
                  <CardTitle className="flex items-center gap-2 text-2xl">
                    <BarChart3 className="h-6 w-6 text-primary" />
                    Classification Result
                  </CardTitle>
                  <CardDescription className="text-base">
                    AI model prediction and confidence scores
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  {/* Main Result Alert */}
                  <Alert
                    className={
                      isLegitimate
                        ? "border-2 border-green-500/50 bg-gradient-to-r from-green-50 to-green-100/50 dark:from-green-950/30 dark:to-green-900/20"
                        : "border-2 border-red-500/50 bg-gradient-to-r from-red-50 to-red-100/50 dark:from-red-950/30 dark:to-red-900/20"
                    }
                  >
                    <div className="flex items-start gap-4">
                      <div className={`p-2 rounded-full ${isLegitimate ? "bg-green-500/20" : "bg-red-500/20"}`}>
                        {isLegitimate ? (
                          <Shield className="h-6 w-6 text-green-600 dark:text-green-400" />
                        ) : (
                          <AlertTriangle className="h-6 w-6 text-red-600 dark:text-red-400" />
                        )}
                      </div>
                      <div className="flex-1 space-y-3">
                        <div>
                          <h3 className={`text-xl font-bold ${isLegitimate ? "text-green-800 dark:text-green-200" : "text-red-800 dark:text-red-200"}`}>
                            {isLegitimate ? "✓ Legitimate Email" : "⚠ Phishing Email Detected"}
                          </h3>
                          <p className={`text-sm mt-1 ${isLegitimate ? "text-green-700 dark:text-green-300" : "text-red-700 dark:text-red-300"}`}>
                            {isLegitimate
                              ? "This email appears to be safe and legitimate."
                              : "This email has been flagged as a potential phishing attempt. Exercise caution and do not click any links or provide personal information."}
                          </p>
                        </div>

                        {/* Confidence Badge */}
                        <div className="flex items-center gap-2">
                          <Badge 
                            variant="outline" 
                            className={`text-sm px-3 py-1 ${isLegitimate ? "border-green-500 text-green-700 dark:text-green-400" : "border-red-500 text-red-700 dark:text-red-400"}`}
                          >
                            Confidence: {explanation.confidence}%
                          </Badge>
                        </div>

                        {/* Probability Distribution */}
                        <div className="space-y-3 pt-2 border-t border-current/20">
                          <div className="flex items-center justify-between text-sm font-medium">
                            <span className={isLegitimate ? "text-green-700 dark:text-green-300" : "text-red-700 dark:text-red-300"}>
                              Legitimate: {explanation.probabilities.legitimate}%
                            </span>
                            <span className={isLegitimate ? "text-red-700 dark:text-red-300" : "text-green-700 dark:text-green-300"}>
                              Phishing: {explanation.probabilities.phishing}%
                            </span>
                          </div>
                          <div className="flex gap-1 h-3 rounded-full overflow-hidden bg-muted">
                            <div
                              className="bg-green-500 transition-all duration-500"
                              style={{ width: `${explanation.probabilities.legitimate}%` }}
                            />
                            <div
                              className="bg-red-500 transition-all duration-500"
                              style={{ width: `${explanation.probabilities.phishing}%` }}
                            />
                          </div>
                        </div>
                      </div>
                    </div>
                  </Alert>
                </CardContent>
              </Card>
            )}
          </div>

          {/* Right Column - Explainable AI Section */}
          {result && explanation && (
            <div className="space-y-6">
              <Card className="shadow-lg border-2 border-primary/20 bg-gradient-to-br from-card to-card/50">
                <CardHeader className="pb-4">
                  <div className="flex items-center gap-3">
                    <div className="p-2 rounded-lg bg-primary/10">
                      <Brain className="h-6 w-6 text-primary" />
                    </div>
                    <div>
                      <CardTitle className="text-2xl">Explainable AI</CardTitle>
                      <CardDescription className="text-base">
                        Understand the model's decision
                      </CardDescription>
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  <Tabs defaultValue="top-words" className="w-full">
                    <TabsList className="grid w-full grid-cols-2 mb-4">
                      <TabsTrigger value="top-words" className="text-xs sm:text-sm">
                        <Eye className="h-4 w-4 mr-1" />
                        Top Words
                      </TabsTrigger>
                      <TabsTrigger value="highlights" className="text-xs sm:text-sm">
                        <Sparkles className="h-4 w-4 mr-1" />
                        Highlights
                      </TabsTrigger>
                    </TabsList>
                    
                    <TabsContent value="top-words" className="space-y-4 mt-4">
                      <div className="space-y-1">
                        <p className="text-xs font-medium text-muted-foreground mb-3">
                          Words that most influenced the prediction
                        </p>
                        <div className="space-y-2 max-h-[500px] overflow-y-auto pr-2">
                          {explanation.top_contributing_words.map((item, idx) => (
                            <div 
                              key={idx} 
                              className="flex items-center justify-between p-3 rounded-lg bg-muted/50 hover:bg-muted transition-colors border border-border/50"
                            >
                              <div className="flex items-center gap-3 flex-1 min-w-0">
                                <Badge 
                                  variant="secondary" 
                                  className="w-8 h-8 flex items-center justify-center shrink-0 font-bold"
                                >
                                  {idx + 1}
                                </Badge>
                                <span className="font-semibold text-sm truncate">{item.word}</span>
                              </div>
                              <div className="flex items-center gap-2 shrink-0">
                                <Progress 
                                  value={item.importance * 100} 
                                  className="w-20 h-2" 
                                />
                                <span className="text-xs font-medium text-muted-foreground w-10 text-right">
                                  {(item.importance * 100).toFixed(0)}%
                                </span>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    </TabsContent>
                    
                    <TabsContent value="highlights" className="space-y-4 mt-4">
                      <div className="space-y-1">
                        <p className="text-xs font-medium text-muted-foreground mb-3">
                          Email text with important words highlighted
                        </p>
                        <div className="p-4 rounded-lg bg-muted/30 border-2 border-border min-h-[400px] max-h-[500px] overflow-y-auto">
                          <HighlightedText text={email} wordImportance={explanation.word_importance} />
                        </div>
                        <div className="flex items-center gap-4 text-xs text-muted-foreground pt-2">
                          <div className="flex items-center gap-2">
                            <div className="w-4 h-4 rounded bg-red-500/60" />
                            <span>High importance</span>
                          </div>
                          <div className="flex items-center gap-2">
                            <div className="w-4 h-4 rounded bg-yellow-500/40" />
                            <span>Medium importance</span>
                          </div>
                        </div>
                      </div>
                    </TabsContent>
                  </Tabs>
                </CardContent>
              </Card>
            </div>
          )}
        </div>

        {/* Footer Note */}
        {!result && (
          <div className="text-center text-sm text-muted-foreground mt-8 p-4 rounded-lg bg-muted/30 border border-border">
            <p className="font-medium">💡 Make sure your Flask backend is running on localhost:5000</p>
          </div>
        )}
      </div>
    </div>
  )
}

function HighlightedText({ text, wordImportance }: { text: string; wordImportance: Record<string, number> }) {
  const words = text.split(/(\s+)/)
  
  return (
    <div className="text-sm leading-relaxed font-mono">
      {words.map((word, idx) => {
        const cleanWord = word.toLowerCase().replace(/[^\w]/g, "")
        const importance = wordImportance[cleanWord] || 0
        
        const bgColor = importance > 0.3 
          ? `rgba(239, 68, 68, ${Math.min(importance * 0.7, 0.7)})` // Red for high importance
          : importance > 0.1
          ? `rgba(251, 191, 36, ${importance * 0.5})` // Yellow for medium
          : "transparent"
        
        return (
          <span
            key={idx}
            style={{
              backgroundColor: bgColor,
              padding: importance > 0.1 ? "2px 2px" : "0",
              borderRadius: "3px",
              transition: "all 0.2s ease",
              fontWeight: importance > 0.3 ? 600 : importance > 0.1 ? 500 : 400,
            }}
            title={importance > 0 ? `Importance: ${(importance * 100).toFixed(1)}%` : undefined}
            className="cursor-help"
          >
            {word}
          </span>
        )
      })}
    </div>
  )
}
