"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Checkbox } from "@/components/ui/checkbox"
import { Label } from "@/components/ui/label"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { MapPin, User, Stethoscope, RotateCcw, Plus, X, History } from "lucide-react"
import { IntakeAnswers, getMemory, addHealthCondition } from "@/lib/api"
import { useToast } from "@/hooks/use-toast"

interface IntakeFormProps {
  onSubmit: (data: IntakeAnswers) => void
  isLoading?: boolean
  lastLocation?: string
}

export function IntakeForm({ onSubmit, isLoading, lastLocation }: IntakeFormProps) {
  const { toast } = useToast()
  const [formData, setFormData] = useState<IntakeAnswers>({
    name: "",
    location_query: "",
    symptoms: [],
    unconscious: false,
    breathing_difficulty: false,
    chest_pain: false,
    stroke_signs: false,
    major_bleeding: false,
    severe_allergy: false,
    injury_trauma: false,
    pregnancy: false,
  })

  const [symptomsText, setSymptomsText] = useState("")
  const [healthConditions, setHealthConditions] = useState<string[]>([])
  const [selectedConditions, setSelectedConditions] = useState<string[]>([])
  const [newConditionInput, setNewConditionInput] = useState("")
  const [isLoadingConditions, setIsLoadingConditions] = useState(true)

  // Update location when lastLocation prop changes
  useEffect(() => {
    if (lastLocation) {
      setFormData(prev => ({ ...prev, location_query: lastLocation }))
    }
  }, [lastLocation])

  // Load health conditions from memory and auto-populate symptoms
  useEffect(() => {
    const loadConditions = async () => {
      try {
        setIsLoadingConditions(true)
        const memory = await getMemory()
        if (memory.health_conditions && memory.health_conditions.length > 0) {
          setHealthConditions(memory.health_conditions)
          
          // Auto-populate symptoms field with health conditions on first load
          // Since this runs on mount, symptomsText will be empty, so we can safely populate
          const conditionsText = memory.health_conditions.join(", ")
          setSymptomsText(conditionsText)
          // Mark all conditions as selected when auto-populating
          setSelectedConditions(memory.health_conditions)
        }
      } catch (error) {
        console.error("Failed to load health conditions:", error)
      } finally {
        setIsLoadingConditions(false)
      }
    }
    loadConditions()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const handleSelectCondition = (condition: string) => {
    if (selectedConditions.includes(condition)) {
      // Deselect
      setSelectedConditions(prev => prev.filter(c => c !== condition))
      // Remove from symptoms text
      const currentSymptoms = symptomsText.split(",").map(s => s.trim()).filter(s => s && s !== condition)
      setSymptomsText(currentSymptoms.join(", "))
    } else {
      // Select
      setSelectedConditions(prev => [...prev, condition])
      // Add to symptoms text
      const currentSymptoms = symptomsText.split(",").map(s => s.trim()).filter(s => s)
      if (!currentSymptoms.includes(condition)) {
        setSymptomsText([...currentSymptoms, condition].join(", "))
      }
    }
  }

  const handleAddNewCondition = async () => {
    const condition = newConditionInput.trim()
    if (!condition) return

    // Add to symptoms if not already there
    const currentSymptoms = symptomsText.split(",").map(s => s.trim()).filter(s => s)
    if (!currentSymptoms.includes(condition)) {
      setSymptomsText([...currentSymptoms, condition].join(", "))
    }

    // Save to history if not already saved
    if (!healthConditions.includes(condition)) {
      try {
        await addHealthCondition(condition)
        setHealthConditions(prev => [...prev, condition])
        toast({
          variant: "success",
          title: "Condition Saved",
          description: `"${condition}" added to your health history`,
        })
      } catch (error) {
        console.error("Failed to save condition:", error)
        toast({
          variant: "destructive",
          title: "Error",
          description: "Failed to save condition to history",
        })
      }
    }

    setNewConditionInput("")
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    const symptoms = symptomsText
      .split(",")
      .map((s) => s.trim())
      .filter((s) => s.length > 0)
    onSubmit({ ...formData, symptoms })
  }

  const handleReset = () => {
    setFormData({
      name: "",
      location_query: lastLocation || "",
      symptoms: [],
      unconscious: false,
      breathing_difficulty: false,
      chest_pain: false,
      stroke_signs: false,
      major_bleeding: false,
      severe_allergy: false,
      injury_trauma: false,
      pregnancy: false,
    })
    setSymptomsText("")
    setSelectedConditions([])
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Stethoscope className="h-5 w-5" />
          Patient Intake
        </CardTitle>
        <CardDescription>Enter patient information and symptoms</CardDescription>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="space-y-2">
            <Label htmlFor="name" className="flex items-center gap-2">
              <User className="h-4 w-4" />
              Patient Name
            </Label>
            <Input
              id="name"
              placeholder="Anonymous (optional)"
              value={formData.name}
              onChange={(e) => setFormData({ ...formData, name: e.target.value })}
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="location" className="flex items-center gap-2">
              <MapPin className="h-4 w-4" />
              Current Location <span className="text-destructive">*</span>
            </Label>
            <Input
              id="location"
              required
              placeholder="e.g., Karachi Pakistan, New York USA"
              value={formData.location_query || lastLocation || ""}
              onChange={(e) => setFormData({ ...formData, location_query: e.target.value })}
            />
            {lastLocation && (
              <p className="text-xs text-muted-foreground">
                Using saved location: {lastLocation}
              </p>
            )}
          </div>

          <div className="space-y-2">
            <Label htmlFor="symptoms" className="flex items-center gap-2">
              Main Symptoms <span className="text-destructive">*</span>
            </Label>
            
            {/* Health History Section */}
            {healthConditions.length > 0 && (
              <div className="space-y-2 rounded-lg border bg-muted/30 p-3">
                <div className="flex items-center gap-2 text-sm font-medium text-muted-foreground">
                  <History className="h-4 w-4" />
                  Your Health History
                </div>
                <div className="flex flex-wrap gap-2">
                  {healthConditions.map((condition) => (
                    <Badge
                      key={condition}
                      variant={selectedConditions.includes(condition) ? "default" : "outline"}
                      className="cursor-pointer hover:bg-primary/80 hover:text-primary-foreground transition-colors"
                      onClick={() => handleSelectCondition(condition)}
                    >
                      {condition}
                      {selectedConditions.includes(condition) && (
                        <X className="h-3 w-3 ml-1" />
                      )}
                    </Badge>
                  ))}
                </div>
              </div>
            )}

            {/* Add New Condition */}
            <div className="flex gap-2">
              <Input
                placeholder="Add new condition (e.g., diabetes, hypertension)"
                value={newConditionInput}
                onChange={(e) => setNewConditionInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    e.preventDefault()
                    handleAddNewCondition()
                  }
                }}
              />
              <Button
                type="button"
                variant="outline"
                size="icon"
                onClick={handleAddNewCondition}
                disabled={!newConditionInput.trim()}
              >
                <Plus className="h-4 w-4" />
              </Button>
            </div>

            {/* Symptoms Textarea */}
            <Textarea
              id="symptoms"
              required
              placeholder="Enter symptoms separated by commas (e.g., chest pain, difficulty breathing, fever)"
              value={symptomsText}
              onChange={(e) => setSymptomsText(e.target.value)}
              rows={4}
            />
            <p className="text-xs text-muted-foreground">
              Select from your history above or enter symptoms manually. Separate multiple symptoms with commas.
            </p>
          </div>

          <div className="space-y-3">
            <Label>Red Flags (Check all that apply)</Label>
            <div className="space-y-3 rounded-lg border p-4">
              <div className="flex items-center space-x-2">
                <Checkbox
                  id="unconscious"
                  checked={formData.unconscious}
                  onCheckedChange={(checked) =>
                    setFormData({ ...formData, unconscious: checked === true })
                  }
                />
                <Label htmlFor="unconscious" className="cursor-pointer">
                  Unconscious / Not responding
                </Label>
              </div>
              <div className="flex items-center space-x-2">
                <Checkbox
                  id="breathing_difficulty"
                  checked={formData.breathing_difficulty}
                  onCheckedChange={(checked) =>
                    setFormData({ ...formData, breathing_difficulty: checked === true })
                  }
                />
                <Label htmlFor="breathing_difficulty" className="cursor-pointer">
                  Difficulty breathing
                </Label>
              </div>
              <div className="flex items-center space-x-2">
                <Checkbox
                  id="chest_pain"
                  checked={formData.chest_pain}
                  onCheckedChange={(checked) =>
                    setFormData({ ...formData, chest_pain: checked === true })
                  }
                />
                <Label htmlFor="chest_pain" className="cursor-pointer">
                  Chest pain/pressure
                </Label>
              </div>
              <div className="flex items-center space-x-2">
                <Checkbox
                  id="stroke_signs"
                  checked={formData.stroke_signs}
                  onCheckedChange={(checked) =>
                    setFormData({ ...formData, stroke_signs: checked === true })
                  }
                />
                <Label htmlFor="stroke_signs" className="cursor-pointer">
                  Stroke signs (face droop/arm weakness/speech)
                </Label>
              </div>
              <div className="flex items-center space-x-2">
                <Checkbox
                  id="major_bleeding"
                  checked={formData.major_bleeding}
                  onCheckedChange={(checked) =>
                    setFormData({ ...formData, major_bleeding: checked === true })
                  }
                />
                <Label htmlFor="major_bleeding" className="cursor-pointer">
                  Major bleeding that won't stop
                </Label>
              </div>
              <div className="flex items-center space-x-2">
                <Checkbox
                  id="severe_allergy"
                  checked={formData.severe_allergy}
                  onCheckedChange={(checked) =>
                    setFormData({ ...formData, severe_allergy: checked === true })
                  }
                />
                <Label htmlFor="severe_allergy" className="cursor-pointer">
                  Severe allergy/anaphylaxis
                </Label>
              </div>
              <div className="flex items-center space-x-2">
                <Checkbox
                  id="injury_trauma"
                  checked={formData.injury_trauma}
                  onCheckedChange={(checked) =>
                    setFormData({ ...formData, injury_trauma: checked === true })
                  }
                />
                <Label htmlFor="injury_trauma" className="cursor-pointer">
                  Injury/trauma
                </Label>
              </div>
            </div>
          </div>

          <div className="flex gap-2">
            <Button type="submit" className="flex-1" disabled={isLoading}>
              {isLoading ? "Processing..." : "Run Triage"}
            </Button>
            <Button type="button" variant="outline" onClick={handleReset}>
              <RotateCcw className="h-4 w-4" />
            </Button>
          </div>
        </form>
      </CardContent>
    </Card>
  )
}
