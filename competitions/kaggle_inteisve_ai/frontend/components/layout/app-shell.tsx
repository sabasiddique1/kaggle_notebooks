"use client"

import { AppSidebar } from "./app-sidebar"
import { AppHeader } from "./app-header"
import { ReactNode, useEffect, useState } from "react"
import { useAuth } from "@/contexts/auth-context"

interface AppShellProps {
  children: ReactNode
  currentLocation?: string
  triageStatus?: "emergency" | "high" | "medium" | "low" | null
}

export function AppShell({ children, currentLocation, triageStatus }: AppShellProps) {
  const { isAuthenticated, loading } = useAuth()
  const [mounted, setMounted] = useState(false)
  
  // Prevent hydration mismatch by only showing sidebar after mount
  useEffect(() => {
    setMounted(true)
  }, [])
  
  // Don't render sidebar until auth check is complete and component is mounted
  const showSidebar = mounted && !loading && isAuthenticated
  
  return (
    <div className="min-h-screen bg-background">
      {showSidebar && <AppSidebar />}
      <div className={showSidebar ? "lg:pl-64" : ""}>
        <AppHeader currentLocation={currentLocation} triageStatus={triageStatus} />
        <main className="p-4 lg:p-6">{children}</main>
      </div>
    </div>
  )
}


