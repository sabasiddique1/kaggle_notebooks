"use client"

import { useState, useEffect } from "react"
import { MapPin, User, Settings as SettingsIcon, Search } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { SearchModal } from "./search-modal"
import { NotificationsDropdown } from "./notifications-dropdown"
import { useAuth } from "@/contexts/auth-context"
import Link from "next/link"

interface AppHeaderProps {
  currentLocation?: string
  triageStatus?: "emergency" | "high" | "medium" | "low" | null
}

export function AppHeader({ currentLocation, triageStatus }: AppHeaderProps) {
  const [searchOpen, setSearchOpen] = useState(false)
  const { user, logout, isAuthenticated } = useAuth()

  // Keyboard shortcut: Ctrl+K or Cmd+K to open search
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "k") {
        e.preventDefault()
        setSearchOpen(true)
      }
    }

    window.addEventListener("keydown", handleKeyDown)
    return () => window.removeEventListener("keydown", handleKeyDown)
  }, [])

  const getTriageBadgeVariant = (status?: string) => {
    switch (status) {
      case "emergency":
        return "emergency"
      case "high":
        return "warning"
      case "medium":
        return "default"
      case "low":
        return "success"
      default:
        return "outline"
    }
  }

  return (
    <>
      <header className="sticky top-0 z-30 flex h-16 items-center gap-4 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 px-4 lg:px-6 shadow-sm">
        <div className="flex flex-1 items-center gap-4">
          {/* Location */}
          {currentLocation && (
            <div className="hidden md:flex items-center gap-2 text-sm text-muted-foreground">
              <MapPin className="h-4 w-4" />
              <span className="truncate max-w-[200px]">{currentLocation}</span>
            </div>
          )}
          
          {/* Triage Status Badge */}
          {triageStatus && (
            <Badge variant={getTriageBadgeVariant(triageStatus)} className="hidden sm:inline-flex">
              {triageStatus.toUpperCase()}
            </Badge>
          )}
        </div>

        {/* Right side actions */}
        <div className="flex items-center gap-2">
          {/* Search Button */}
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setSearchOpen(true)}
            className="relative"
            title="Search (Ctrl+K)"
          >
            <Search className="h-5 w-5" />
          </Button>

          {/* Notifications - Only show if logged in */}
          {isAuthenticated && <NotificationsDropdown />}

          {/* Login/Signup buttons - Show if not logged in */}
          {!isAuthenticated ? (
            <>
              <Link href="/login">
                <Button variant="ghost" size="sm">
                  Login
                </Button>
              </Link>
              <Link href="/login?register=true">
                <Button size="sm">
                  Sign Up
                </Button>
              </Link>
            </>
          ) : (
            <>
              {/* Settings */}
              <Link href="/settings">
                <Button variant="ghost" size="icon" title="Settings">
                  <SettingsIcon className="h-5 w-5" />
                </Button>
              </Link>

              {/* User Profile */}
              <Link href="/profile">
                <Button variant="ghost" size="icon" title={user?.name || "Profile"}>
                  <User className="h-5 w-5" />
                </Button>
              </Link>

              {/* Logout Button */}
              <Button variant="ghost" size="sm" onClick={logout} title="Logout">
                Logout
              </Button>
            </>
          )}
        </div>
      </header>

      {/* Search Modal */}
      <SearchModal open={searchOpen} onOpenChange={setSearchOpen} />
    </>
  )
}

