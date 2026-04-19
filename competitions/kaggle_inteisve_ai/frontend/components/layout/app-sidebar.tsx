"use client"

import Link from "next/link"
import { usePathname, useRouter } from "next/navigation"
import { 
  Activity, 
  Stethoscope, 
  ClipboardList, 
  FileText, 
  Settings,
  Menu,
  X,
  User,
  Building2,
  LogOut
} from "lucide-react"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { useState } from "react"
import { useAuth } from "@/contexts/auth-context"
import { Separator } from "@/components/ui/separator"

const patientNavigation = [
  { name: "Dashboard", href: "/", icon: Activity },
  { name: "Triage Workspace", href: "/triage", icon: Stethoscope },
  { name: "My Requests", href: "/requests", icon: FileText },
  { name: "Health History", href: "/profile", icon: User },
  { name: "Sessions", href: "/sessions", icon: ClipboardList },
  { name: "Settings", href: "/settings", icon: Settings },
]

const hospitalNavigation = [
  { name: "Hospital Panel", href: "/hospital", icon: Building2 },
  { name: "Settings", href: "/settings", icon: Settings },
]

export function AppSidebar() {
  const pathname = usePathname()
  const [isMobileOpen, setIsMobileOpen] = useState(false)
  const { user, logout, isHospitalStaff } = useAuth()
  const router = useRouter()
  
  const navigation = isHospitalStaff ? hospitalNavigation : patientNavigation
  
  const handleLogout = () => {
    logout()
    setIsMobileOpen(false)
  }

  return (
    <>
      {/* Mobile menu button */}
      <div className="lg:hidden fixed top-4 left-4 z-50">
        <Button
          variant="outline"
          size="icon"
          onClick={() => setIsMobileOpen(!isMobileOpen)}
        >
          {isMobileOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
        </Button>
      </div>

      {/* Sidebar */}
      <aside
        className={cn(
          "fixed left-0 top-0 z-40 h-screen w-64 border-r bg-card transition-transform lg:translate-x-0",
          isMobileOpen ? "translate-x-0" : "-translate-x-full lg:translate-x-0"
        )}
      >
        <div className="flex h-full flex-col">
          <div className="flex h-16 items-center border-b px-6">
            <Stethoscope className="h-6 w-6 text-primary mr-2" />
            <span className="font-semibold text-lg">Emergency Care</span>
          </div>
          <nav className="flex-1 space-y-1 px-3 py-4">
            {navigation.map((item) => {
              const isActive = pathname === item.href
              return (
                <Link
                  key={item.name}
                  href={item.href}
                  onClick={() => setIsMobileOpen(false)}
                  className={cn(
                    "flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors",
                    isActive
                      ? "bg-primary text-primary-foreground"
                      : "text-muted-foreground hover:bg-accent hover:text-accent-foreground"
                  )}
                >
                  <item.icon className="h-5 w-5" />
                  {item.name}
                </Link>
              )
            })}
          </nav>
          {user && (
            <>
              <Separator />
              <div className="px-3 py-4 space-y-2">
                <div className="px-3 py-2 text-sm">
                  <p className="font-medium">{user.name}</p>
                  <p className="text-xs text-muted-foreground">{user.email}</p>
                  {user.facility_name && (
                    <p className="text-xs text-muted-foreground">{user.facility_name}</p>
                  )}
                </div>
                <Button
                  variant="ghost"
                  className="w-full justify-start"
                  onClick={handleLogout}
                >
                  <LogOut className="h-4 w-4 mr-2" />
                  Logout
                </Button>
              </div>
            </>
          )}
        </div>
      </aside>

      {/* Overlay for mobile */}
      {isMobileOpen && (
        <div
          className="fixed inset-0 z-30 bg-background/80 backdrop-blur-sm lg:hidden"
          onClick={() => setIsMobileOpen(false)}
        />
      )}
    </>
  )
}

