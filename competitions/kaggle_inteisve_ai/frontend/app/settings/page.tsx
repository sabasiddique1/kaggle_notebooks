"use client"

import { AppShell } from "@/components/layout/app-shell"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Button } from "@/components/ui/button"
import { useState, useEffect } from "react"
import { useTheme } from "next-themes"
import { useToast } from "@/hooks/use-toast"
import { getMemory, updateMemory, MemoryBank } from "@/lib/api"
import { Loader2, CheckCircle, AlertCircle } from "lucide-react"
import { useAuth } from "@/contexts/auth-context"

export default function SettingsPage() {
  const { user } = useAuth()
  const isHospitalStaff = user?.role === "hospital_staff"
  const { theme, setTheme } = useTheme()
  const { toast } = useToast()
  const [preferredCity, setPreferredCity] = useState("")
  const [llmProvider, setLlmProvider] = useState("gemini")
  const [isLoading, setIsLoading] = useState(false)
  const [isLoadingMemory, setIsLoadingMemory] = useState(true)
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  useEffect(() => {
    // Load memory on mount
    const loadMemory = async () => {
      try {
        setIsLoadingMemory(true)
        const memory = await getMemory()
        if (memory.preferred_city) {
          setPreferredCity(memory.preferred_city)
        }
      } catch (error) {
        console.error("Failed to load memory:", error)
        toast({
          variant: "destructive",
          title: "Error",
          description: "Failed to load saved preferences",
        })
      } finally {
        setIsLoadingMemory(false)
      }
    }
    loadMemory()
  }, [toast])

  const handleSaveLocation = async () => {
    if (!preferredCity.trim()) {
      toast({
        variant: "destructive",
        title: "Error",
        description: "Please enter a location",
      })
      return
    }

    setIsLoading(true)
    try {
      await updateMemory({ preferred_city: preferredCity.trim() })
      toast({
        variant: "success",
        title: "Success",
        description: `Preferred location saved: ${preferredCity}`,
      })
    } catch (error) {
      toast({
        variant: "destructive",
        title: "Error",
        description: error instanceof Error ? error.message : "Failed to save location",
      })
    } finally {
      setIsLoading(false)
    }
  }

  const handleThemeChange = (newTheme: string) => {
    setTheme(newTheme)
    toast({
      title: "Theme Updated",
      description: `Theme changed to ${newTheme}`,
    })
  }

  const handleLLMProviderChange = (newProvider: string) => {
    setLlmProvider(newProvider)
    toast({
      title: "LLM Provider Updated",
      description: `Provider changed to ${newProvider === "gemini" ? "Google Gemini" : "Mock"}`,
    })
  }

  return (
    <AppShell>
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Settings</h1>
          <p className="text-muted-foreground">
            {isHospitalStaff 
              ? "Configure hospital panel preferences and settings"
              : "Configure application preferences and defaults"}
          </p>
        </div>

        {isHospitalStaff ? (
          // Hospital Staff Settings
          <div className="grid gap-6 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>Facility Information</CardTitle>
                <CardDescription>
                  Your assigned facility details
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label>Facility Name</Label>
                  <div className="flex h-10 w-full rounded-md border border-input bg-muted px-3 py-2 text-sm items-center">
                    {user?.facility_name || "Not assigned"}
                  </div>
                </div>
                <div className="rounded-lg border bg-muted/50 p-3">
                  <p className="text-xs text-muted-foreground">
                    Facility information is managed by system administrators. Contact support to update your facility details.
                  </p>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Notification Preferences</CardTitle>
                <CardDescription>
                  Configure how you receive booking notifications
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="notifications">Notification Method</Label>
                  <select
                    id="notifications"
                    className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                    defaultValue="in-app"
                  >
                    <option value="in-app">In-App Only</option>
                    <option value="email">Email Notifications</option>
                    <option value="both">Both</option>
                  </select>
                </div>
                <div className="rounded-lg border bg-muted/50 p-3">
                  <p className="text-xs text-muted-foreground">
                    You will be notified when new booking requests arrive for your facility.
                  </p>
                </div>
              </CardContent>
            </Card>
          </div>
        ) : (
          // Patient Settings
          <div className="grid gap-6 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>LLM Provider</CardTitle>
                <CardDescription>
                  Configure language model settings for explanations
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="llm-provider">Provider</Label>
                  <select
                    id="llm-provider"
                    className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                    value={llmProvider}
                    onChange={(e) => handleLLMProviderChange(e.target.value)}
                  >
                    <option value="gemini">Google Gemini</option>
                    <option value="mock">Mock (No API Key)</option>
                  </select>
                </div>
                <div className="rounded-lg border bg-muted/50 p-3">
                  <p className="text-xs text-muted-foreground">
                    Note: LLM is only used for empathetic explanations. Triage decisions are always rule-based.
                  </p>
                </div>
              </CardContent>
            </Card>

          <Card>
            <CardHeader>
              <CardTitle>Default Location</CardTitle>
              <CardDescription>
                Set preferred city for quick access
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {isLoadingMemory ? (
                <div className="flex items-center justify-center py-4">
                  <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
                </div>
              ) : (
                <>
                  <div className="space-y-2">
                    <Label htmlFor="preferred-city">Preferred City</Label>
                    <Input
                      id="preferred-city"
                      value={preferredCity}
                      onChange={(e) => setPreferredCity(e.target.value)}
                      placeholder="e.g., Karachi Pakistan"
                      onKeyDown={(e) => {
                        if (e.key === "Enter") {
                          handleSaveLocation()
                        }
                      }}
                    />
                  </div>
                  <Button onClick={handleSaveLocation} disabled={isLoading}>
                    {isLoading ? (
                      <>
                        <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                        Saving...
                      </>
                    ) : (
                      <>
                        <CheckCircle className="h-4 w-4 mr-2" />
                        Save Location
                      </>
                    )}
                  </Button>
                </>
              )}
            </CardContent>
          </Card>

            <Card className="md:col-span-2">
              <CardHeader>
                <CardTitle>Appearance</CardTitle>
                <CardDescription>
                  Customize the application theme
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="theme">Theme</Label>
                  {!mounted ? (
                    <div className="flex h-10 w-full items-center rounded-md border border-input bg-muted px-3 py-2 text-sm text-muted-foreground">
                      Loading...
                    </div>
                  ) : (
                    <select
                      id="theme"
                      className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                      value={theme || "light"}
                      onChange={(e) => handleThemeChange(e.target.value)}
                    >
                      <option value="light">Light</option>
                      <option value="dark">Dark</option>
                      <option value="system">System</option>
                    </select>
                  )}
                </div>
                <div className="rounded-lg border bg-muted/50 p-3">
                  <p className="text-xs text-muted-foreground">
                    The theme will be applied immediately. System theme follows your OS preferences.
                  </p>
                </div>
              </CardContent>
            </Card>
          </div>
        )}
      </div>
    </AppShell>
  )
}
