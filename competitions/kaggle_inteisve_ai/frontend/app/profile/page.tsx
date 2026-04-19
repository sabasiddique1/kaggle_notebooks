"use client"

import { AppShell } from "@/components/layout/app-shell"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { InputFile } from "@/components/ui/input-file"
import { useState, useEffect, useRef } from "react"
import { getMemory, addHealthCondition, removeHealthCondition, uploadDocument } from "@/lib/api"
import { useToast } from "@/hooks/use-toast"
import { Plus, X, History, Loader2, Upload, FileText, CheckCircle } from "lucide-react"

export default function ProfilePage() {
  const { toast } = useToast()
  const [healthConditions, setHealthConditions] = useState<string[]>([])
  const [newCondition, setNewCondition] = useState("")
  const [isLoading, setIsLoading] = useState(true)
  const [isAdding, setIsAdding] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [uploadedFiles, setUploadedFiles] = useState<string[]>([])
  const fileInputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    loadConditions()
  }, [])

  const loadConditions = async () => {
    try {
      setIsLoading(true)
      const memory = await getMemory()
      const conditions = memory.health_conditions || []
      setHealthConditions(conditions)
      // Only show loading state if we're actually fetching
      // If conditions exist, they'll show immediately after loading
    } catch (error) {
      console.error("Failed to load conditions:", error)
      toast({
        variant: "destructive",
        title: "Error",
        description: "Failed to load health conditions",
      })
      // On error, set empty array so empty state shows
      setHealthConditions([])
    } finally {
      setIsLoading(false)
    }
  }

  const handleAddCondition = async () => {
    const condition = newCondition.trim()
    if (!condition) {
      toast({
        variant: "destructive",
        title: "Error",
        description: "Please enter a condition",
      })
      return
    }

    if (healthConditions.includes(condition)) {
      toast({
        variant: "destructive",
        title: "Error",
        description: "This condition already exists",
      })
      return
    }

    setIsAdding(true)
    try {
      await addHealthCondition(condition)
      setHealthConditions(prev => [...prev, condition])
      setNewCondition("")
      toast({
        variant: "success",
        title: "Condition Added",
        description: `"${condition}" added to your health history`,
      })
    } catch (error) {
      toast({
        variant: "destructive",
        title: "Error",
        description: error instanceof Error ? error.message : "Failed to add condition",
      })
    } finally {
      setIsAdding(false)
    }
  }

  const handleRemoveCondition = async (condition: string) => {
    try {
      await removeHealthCondition(condition)
      setHealthConditions(prev => prev.filter(c => c !== condition))
      toast({
        variant: "success",
        title: "Condition Removed",
        description: `"${condition}" removed from your health history`,
      })
    } catch (error) {
      toast({
        variant: "destructive",
        title: "Error",
        description: error instanceof Error ? error.message : "Failed to remove condition",
      })
    }
  }

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    // Validate file type
    const allowedTypes = ['application/pdf', 'text/plain', 'image/jpeg', 'image/png', 'image/jpg']
    const allowedExtensions = ['.pdf', '.txt', '.jpg', '.jpeg', '.png']
    const fileExt = '.' + file.name.split('.').pop()?.toLowerCase()

    if (!allowedTypes.includes(file.type) && !allowedExtensions.includes(fileExt)) {
      toast({
        variant: "destructive",
        title: "Invalid File Type",
        description: "Please upload PDF, TXT, or image files (JPG, PNG)",
      })
      return
    }

    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
      toast({
        variant: "destructive",
        title: "File Too Large",
        description: "Please upload files smaller than 10MB",
      })
      return
    }

    setIsUploading(true)
    try {
      const result = await uploadDocument(file)
      setUploadedFiles(prev => [...prev, result.filename])
      
      if (result.extracted_conditions && result.extracted_conditions.length > 0) {
        // Reload conditions to show newly extracted ones
        await loadConditions()
        toast({
          variant: "success",
          title: "Document Processed",
          description: `Extracted ${result.extracted_conditions.length} condition(s): ${result.extracted_conditions.join(", ")}`,
        })
      } else {
        toast({
          variant: "default",
          title: "Document Uploaded",
          description: "File uploaded successfully. No conditions were automatically extracted.",
        })
      }
    } catch (error) {
      toast({
        variant: "destructive",
        title: "Upload Failed",
        description: error instanceof Error ? error.message : "Failed to upload document",
      })
    } finally {
      setIsUploading(false)
      // Reset file input
      if (fileInputRef.current) {
        fileInputRef.current.value = ""
      }
    }
  }

  return (
    <AppShell>
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Health History</h1>
          <p className="text-muted-foreground">
            Manage your health conditions and upload medical documents for automatic condition extraction
          </p>
        </div>

        {/* Document Upload Section */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Upload className="h-5 w-5" />
              Upload Medical Documents
            </CardTitle>
            <CardDescription>
              Upload medical reports, test results, or doctor notes. AI will automatically extract health conditions.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label>Upload Document</Label>
              <InputFile
                ref={fileInputRef}
                label=""
                accept=".pdf,.txt,.jpg,.jpeg,.png"
                onChange={handleFileUpload}
                disabled={isUploading}
              />
              <p className="text-xs text-muted-foreground">
                Supported formats: PDF, TXT, JPG, PNG (max 10MB)
              </p>
              {isUploading && (
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Processing document and extracting conditions...
                </div>
              )}
            </div>

            {uploadedFiles.length > 0 && (
              <div className="space-y-2">
                <Label>Uploaded Documents</Label>
                <div className="flex flex-wrap gap-2">
                  {uploadedFiles.map((filename, index) => (
                    <Badge key={index} variant="outline" className="flex items-center gap-1">
                      <FileText className="h-3 w-3" />
                      {filename}
                    </Badge>
                  ))}
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Health Conditions Section */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <History className="h-5 w-5" />
              Your Health Conditions
            </CardTitle>
            <CardDescription>
              Add conditions manually or let AI extract them from your documents above.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Add New Condition */}
            <div className="space-y-2">
              <Label htmlFor="new-condition">Add New Condition</Label>
              <div className="flex gap-2">
                <Input
                  id="new-condition"
                  placeholder="e.g., diabetes, hypertension, asthma, chronic pain"
                  value={newCondition}
                  onChange={(e) => setNewCondition(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      e.preventDefault()
                      handleAddCondition()
                    }
                  }}
                />
                <Button onClick={handleAddCondition} disabled={isAdding || !newCondition.trim()}>
                  {isAdding ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <Plus className="h-4 w-4" />
                  )}
                </Button>
              </div>
            </div>

            {/* Conditions List */}
            {isLoading ? (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
                <span className="ml-2 text-sm text-muted-foreground">Loading conditions...</span>
              </div>
            ) : healthConditions.length === 0 ? (
              <div className="text-center py-8 text-muted-foreground">
                <History className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p className="font-medium">No health conditions saved yet</p>
                <p className="text-sm mt-2">Add conditions manually or upload documents to extract them automatically</p>
              </div>
            ) : (
              <div className="space-y-2">
                <Label>Saved Conditions ({healthConditions.length})</Label>
                <div className="flex flex-wrap gap-2">
                  {healthConditions.map((condition) => (
                    <Badge
                      key={condition}
                      variant="default"
                      className="text-sm px-3 py-1.5 flex items-center gap-2"
                    >
                      {condition}
                      <button
                        onClick={() => handleRemoveCondition(condition)}
                        className="ml-1 hover:bg-destructive/20 rounded-full p-0.5 transition-colors"
                        aria-label={`Remove ${condition}`}
                      >
                        <X className="h-3 w-3" />
                      </button>
                    </Badge>
                  ))}
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>How It Works</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 text-sm text-muted-foreground">
            <p className="flex items-start gap-2">
              <CheckCircle className="h-4 w-4 mt-0.5 flex-shrink-0" />
              <span>Upload medical documents (PDF, TXT, images) - AI will automatically extract health conditions</span>
            </p>
            <p className="flex items-start gap-2">
              <CheckCircle className="h-4 w-4 mt-0.5 flex-shrink-0" />
              <span>Or add conditions manually using the input field above</span>
            </p>
            <p className="flex items-start gap-2">
              <CheckCircle className="h-4 w-4 mt-0.5 flex-shrink-0" />
              <span>Conditions appear as quick-select badges in the triage form</span>
            </p>
            <p className="flex items-start gap-2">
              <CheckCircle className="h-4 w-4 mt-0.5 flex-shrink-0" />
              <span>Click a badge in triage to add it to your symptoms</span>
            </p>
            <p className="flex items-start gap-2">
              <CheckCircle className="h-4 w-4 mt-0.5 flex-shrink-0" />
              <span>New conditions entered in triage are automatically saved</span>
            </p>
          </CardContent>
        </Card>
      </div>
    </AppShell>
  )
}
