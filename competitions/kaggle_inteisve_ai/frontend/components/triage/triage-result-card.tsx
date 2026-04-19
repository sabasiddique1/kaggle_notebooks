"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { TriageResult } from "@/lib/api"
import { AlertTriangle, CheckCircle, Clock, AlertCircle, Phone } from "lucide-react"
import { TriageLevelBadge } from "./triage-level-badge"

interface TriageResultCardProps {
  result: TriageResult | null
  isLoading?: boolean
}

export function TriageResultCard({ result, isLoading }: TriageResultCardProps) {
  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Triage Result</CardTitle>
          <CardDescription>Processing...</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="animate-pulse space-y-4">
            <div className="h-8 bg-muted rounded" />
            <div className="h-4 bg-muted rounded w-3/4" />
            <div className="h-4 bg-muted rounded" />
          </div>
        </CardContent>
      </Card>
    )
  }

  if (!result) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Triage Result</CardTitle>
          <CardDescription>Run triage to see results</CardDescription>
        </CardHeader>
      </Card>
    )
  }

  const getIcon = () => {
    switch (result.level) {
      case "emergency":
        return <AlertTriangle className="h-5 w-5 text-destructive" />
      case "high":
        return <AlertCircle className="h-5 w-5 text-warning" />
      case "medium":
        return <Clock className="h-5 w-5 text-muted-foreground" />
      case "low":
        return <CheckCircle className="h-5 w-5 text-success" />
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Triage Result</span>
          <TriageLevelBadge level={result.level} />
        </CardTitle>
        <CardDescription>Urgency assessment and recommendations</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-start gap-3">
          {getIcon()}
          <div className="flex-1 space-y-2">
            <p className="text-sm leading-relaxed">{result.reason}</p>
            <div className="rounded-lg bg-muted p-3">
              <div className="flex items-start gap-2">
                {result.level === "emergency" && (
                  <Phone className="h-4 w-4 text-destructive mt-0.5 flex-shrink-0" />
                )}
                <div className="flex-1">
                  <p className="text-sm font-medium mb-1">Recommended Action:</p>
                  <p className="text-sm">{result.recommended_action}</p>
                </div>
              </div>
            </div>
            <div className="rounded-lg border border-warning/20 bg-warning/5 p-3">
              <p className="text-xs text-muted-foreground">{result.safety_note}</p>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

