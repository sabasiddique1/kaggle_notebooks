"use client"

import { useState, useEffect, useRef } from "react"
import { Dialog, DialogContent } from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Search, X, Clock, MapPin, Building2, User, FileText, ArrowRight } from "lucide-react"
import { getPatientSessions, getPatientRequests, PatientSession, PatientRequest } from "@/lib/api"
import { useAuth } from "@/contexts/auth-context"
import { useRouter } from "next/navigation"
import Link from "next/link"

interface SearchResult {
  id: string
  type: "session" | "request"
  title: string
  subtitle: string
  badge?: string
  badgeVariant?: "default" | "destructive" | "outline" | "secondary"
  link: string
}

interface SearchModalProps {
  open: boolean
  onOpenChange: (open: boolean) => void
}

export function SearchModal({ open, onOpenChange }: SearchModalProps) {
  const { user } = useAuth()
  const router = useRouter()
  const [query, setQuery] = useState("")
  const [results, setResults] = useState<SearchResult[]>([])
  const [isSearching, setIsSearching] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (open) {
      // Focus input when modal opens
      setTimeout(() => inputRef.current?.focus(), 100)
    } else {
      // Clear search when modal closes
      setQuery("")
      setResults([])
    }
  }, [open])

  const performSearch = async (searchQuery: string) => {
    if (!searchQuery.trim() || !user?.email) {
      setResults([])
      return
    }

    setIsSearching(true)
    try {
      const [sessions, requests] = await Promise.all([
        getPatientSessions(user.email).catch(() => []),
        getPatientRequests(user.email).catch(() => [])
      ])

      const searchLower = searchQuery.toLowerCase()
      const searchResults: SearchResult[] = []

      // Search in sessions
      sessions.forEach((session: PatientSession) => {
        const matches =
          session.patient_name?.toLowerCase().includes(searchLower) ||
          session.location?.toLowerCase().includes(searchLower) ||
          session.facility_name?.toLowerCase().includes(searchLower) ||
          session.triage_level?.toLowerCase().includes(searchLower) ||
          session.session_id.toLowerCase().includes(searchLower)

        if (matches) {
          searchResults.push({
            id: session.session_id,
            type: "session",
            title: session.patient_name || "Anonymous",
            subtitle: `${session.location || "Unknown location"} • ${session.facility_name || "Unknown facility"}`,
            badge: session.triage_level,
            badgeVariant: getTriageBadgeVariant(session.triage_level),
            link: `/sessions?session=${session.session_id}`
          })
        }
      })

      // Search in requests
      requests.forEach((request: PatientRequest) => {
        const matches =
          request.patient_name?.toLowerCase().includes(searchLower) ||
          request.facility_name?.toLowerCase().includes(searchLower) ||
          request.triage_level?.toLowerCase().includes(searchLower) ||
          request.status?.toLowerCase().includes(searchLower) ||
          request.session_id.toLowerCase().includes(searchLower)

        if (matches) {
          searchResults.push({
            id: request.session_id,
            type: "request",
            title: `${request.patient_name || "Anonymous"} - ${request.facility_name}`,
            subtitle: `Status: ${request.status} • ${request.triage_level || "Unknown"} priority`,
            badge: request.status,
            badgeVariant: getStatusBadgeVariant(request.status),
            link: `/requests`
          })
        }
      })

      // Sort by relevance (exact matches first, then partial)
      searchResults.sort((a, b) => {
        const aExact = a.title.toLowerCase() === searchLower || a.subtitle.toLowerCase().includes(searchLower)
        const bExact = b.title.toLowerCase() === searchLower || b.subtitle.toLowerCase().includes(searchLower)
        if (aExact && !bExact) return -1
        if (!aExact && bExact) return 1
        return 0
      })

      setResults(searchResults.slice(0, 10)) // Limit to 10 results
    } catch (error) {
      console.error("Search error:", error)
      setResults([])
    } finally {
      setIsSearching(false)
    }
  }

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault()
    performSearch(query)
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      e.preventDefault()
      performSearch(query)
    } else if (e.key === "Escape") {
      onOpenChange(false)
    }
  }

  const getTriageBadgeVariant = (level?: string): "default" | "destructive" | "outline" | "secondary" => {
    switch (level) {
      case "emergency":
        return "destructive"
      case "high":
        return "default"
      case "medium":
        return "secondary"
      case "low":
        return "outline"
      default:
        return "outline"
    }
  }

  const getStatusBadgeVariant = (status?: string): "default" | "destructive" | "outline" | "secondary" => {
    if (status === "CONFIRMED" || status === "ACKNOWLEDGED") return "default"
    if (status === "REJECTED") return "destructive"
    if (status === "PENDING_ACK" || status === "PENDING_APPROVAL") return "secondary"
    return "outline"
  }

  const handleResultClick = (link: string) => {
    router.push(link)
    onOpenChange(false)
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-2xl p-0 gap-0 [&>button]:hidden">
        <div className="p-6 border-b relative">
          <form onSubmit={handleSearch} className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground z-10" />
            <Input
              ref={inputRef}
              type="text"
              placeholder="Search sessions, requests, facilities, patients..."
              value={query}
              onChange={(e) => {
                setQuery(e.target.value)
                if (e.target.value.trim()) {
                  performSearch(e.target.value)
                } else {
                  setResults([])
                }
              }}
              onKeyDown={handleKeyDown}
              className="pl-10 pr-10 h-12 text-base"
            />
            {query && (
              <Button
                type="button"
                variant="ghost"
                size="icon"
                className="absolute right-1 top-1/2 transform -translate-y-1/2 h-8 w-8"
                onClick={() => {
                  setQuery("")
                  setResults([])
                  inputRef.current?.focus()
                }}
              >
                <X className="h-4 w-4" />
              </Button>
            )}
          </form>
          {/* Close button in top-right corner */}
          <Button
            variant="ghost"
            size="icon"
            className="absolute right-4 top-4 h-8 w-8"
            onClick={() => onOpenChange(false)}
          >
            <X className="h-4 w-4" />
          </Button>
        </div>

        <div className="max-h-[60vh] overflow-y-auto">
          {isSearching && query ? (
            <div className="p-12 text-center text-muted-foreground">
              <Search className="h-8 w-8 mx-auto mb-2 animate-pulse" />
              <p>Searching...</p>
            </div>
          ) : results.length > 0 ? (
            <div className="p-2">
              <div className="px-4 py-2 text-xs font-semibold text-muted-foreground uppercase tracking-wide">
                {results.length} {results.length === 1 ? "result" : "results"}
              </div>
              <div className="space-y-1">
                {results.map((result) => (
                  <button
                    key={`${result.type}-${result.id}`}
                    onClick={() => handleResultClick(result.link)}
                    className="w-full text-left p-4 rounded-lg hover:bg-accent transition-colors group"
                  >
                    <div className="flex items-start justify-between gap-4">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1">
                          {result.type === "session" ? (
                            <FileText className="h-4 w-4 text-muted-foreground flex-shrink-0" />
                          ) : (
                            <Building2 className="h-4 w-4 text-muted-foreground flex-shrink-0" />
                          )}
                          <span className="font-medium truncate">{result.title}</span>
                          {result.badge && (
                            <Badge variant={result.badgeVariant} className="text-xs">
                              {result.badge}
                            </Badge>
                          )}
                        </div>
                        <p className="text-sm text-muted-foreground line-clamp-1">{result.subtitle}</p>
                      </div>
                      <ArrowRight className="h-4 w-4 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity flex-shrink-0" />
                    </div>
                  </button>
                ))}
              </div>
            </div>
          ) : query ? (
            <div className="p-12 text-center text-muted-foreground">
              <Search className="h-8 w-8 mx-auto mb-2 opacity-50" />
              <p>No results found</p>
              <p className="text-xs mt-1">Try different keywords</p>
            </div>
          ) : (
            <div className="p-12 text-center text-muted-foreground">
              <Search className="h-8 w-8 mx-auto mb-2 opacity-50" />
              <p>Start typing to search</p>
              <p className="text-xs mt-1">Search across sessions, requests, and facilities</p>
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  )
}

