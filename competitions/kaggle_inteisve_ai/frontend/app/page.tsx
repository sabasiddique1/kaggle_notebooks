"use client"

import { useState, useEffect } from "react"
import { AppShell } from "@/components/layout/app-shell"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Activity, AlertTriangle, Clock, MapPin, ArrowRight, Loader2, RefreshCw, Building2, Stethoscope, Shield, Zap, Users, LogIn, CheckCircle, XCircle, TrendingUp, FileCheck, Timer, Heart, BarChart3, Calendar, CalendarDays, Target, Percent, Gauge, Award, TrendingDown, Bell, CalendarCheck, PlusCircle, BookOpen, Info, Sparkles, ActivitySquare } from "lucide-react"
import Link from "next/link"
import { getPatientSessions, PatientSession, getMemory, MemoryBank } from "@/lib/api"
import { useAuth } from "@/contexts/auth-context"
import { useToast } from "@/hooks/use-toast"

export default function DashboardPage() {
  const { user, isAuthenticated } = useAuth()
  const { toast } = useToast()
  const [sessions, setSessions] = useState<PatientSession[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [memory, setMemory] = useState<MemoryBank | null>(null)
  const [stats, setStats] = useState({
    sessionsToday: 0,
    sessionsThisWeek: 0,
    sessionsThisMonth: 0,
    totalSessions: 0,
    emergencyCases: 0,
    highPriorityCases: 0,
    mediumPriorityCases: 0,
    lowPriorityCases: 0,
    averageETA: 0,
    activeTriages: 0,
    completedTriages: 0,
    pendingRequests: 0,
    approvedRequests: 0,
    rejectedRequests: 0,
    totalFacilities: 0,
    successRate: 0,
    completionRate: 0,
    rejectionRate: 0,
    alertRequests: 0,
    appointmentRequests: 0,
    averageSessionsPerDay: 0,
    mostUsedFacility: "",
    oldestPendingRequest: 0,
  })

  useEffect(() => {
    if (isAuthenticated && user?.email) {
      loadRecentActivity()
      loadMemory()
    } else {
      setIsLoading(false)
    }
  }, [user, isAuthenticated])

  const loadMemory = async () => {
    try {
      const data = await getMemory()
      setMemory(data)
    } catch (error) {
      console.error("Failed to load memory:", error)
    }
  }

  const loadRecentActivity = async () => {
    if (!user?.email) {
      setIsLoading(false)
      return
    }

    try {
      setIsLoading(true)
      const data = await getPatientSessions(user.email)
      setSessions(data)

      // Calculate stats
      const now = new Date()
      const today = new Date(now)
      today.setHours(0, 0, 0, 0)
      
      const weekAgo = new Date(now)
      weekAgo.setDate(weekAgo.getDate() - 7)
      weekAgo.setHours(0, 0, 0, 0)
      
      const monthAgo = new Date(now)
      monthAgo.setMonth(monthAgo.getMonth() - 1)
      monthAgo.setHours(0, 0, 0, 0)
      
      const todaySessions = data.filter(s => {
        if (!s.requested_at) return false
        const sessionDate = new Date(s.requested_at)
        return sessionDate >= today
      })
      
      const weekSessions = data.filter(s => {
        if (!s.requested_at) return false
        const sessionDate = new Date(s.requested_at)
        return sessionDate >= weekAgo
      })
      
      const monthSessions = data.filter(s => {
        if (!s.requested_at) return false
        const sessionDate = new Date(s.requested_at)
        return sessionDate >= monthAgo
      })

      const emergencyCases = data.filter(s => s.triage_level === "emergency").length
      const highPriorityCases = data.filter(s => s.triage_level === "high").length
      const mediumPriorityCases = data.filter(s => s.triage_level === "medium").length
      const lowPriorityCases = data.filter(s => s.triage_level === "low").length
      
      const activeTriages = data.filter(s => 
        s.status === "PENDING_ACK" || s.status === "PENDING_APPROVAL" || s.status === "pending_approval"
      ).length
      
      const completedTriages = data.filter(s => 
        s.status === "APPROVED" || s.status === "approved" || s.status === "COMPLETED" || s.status === "completed"
      ).length
      
      const pendingRequests = data.filter(s => 
        s.status === "PENDING_ACK" || s.status === "PENDING_APPROVAL" || s.status === "pending_approval"
      ).length
      
      const approvedRequests = data.filter(s => 
        s.status === "APPROVED" || s.status === "approved"
      ).length
      
      const rejectedRequests = data.filter(s => 
        s.status === "REJECTED" || s.status === "rejected"
      ).length
      
      // Get unique facilities
      const uniqueFacilities = new Set(data.map(s => s.facility_name).filter(Boolean))
      
      // Calculate success rate (approved / total with status)
      const totalWithStatus = data.filter(s => s.status && s.status !== "unknown").length
      const successRate = totalWithStatus > 0 ? Math.round((approvedRequests / totalWithStatus) * 100) : 0
      const completionRate = data.length > 0 ? Math.round((completedTriages / data.length) * 100) : 0
      const rejectionRate = totalWithStatus > 0 ? Math.round((rejectedRequests / totalWithStatus) * 100) : 0
      
      // Request type breakdown
      const alertRequests = data.filter(s => s.request_type === "alert").length
      const appointmentRequests = data.filter(s => s.request_type === "appointment").length
      
      // Calculate average sessions per day (based on oldest session)
      const sessionsWithDates = data.filter(s => s.requested_at)
      let averageSessionsPerDay = 0
      if (sessionsWithDates.length > 0) {
        const dates = sessionsWithDates.map(s => new Date(s.requested_at!))
        const oldestDate = new Date(Math.min(...dates.map(d => d.getTime())))
        const daysDiff = Math.max(1, Math.ceil((now.getTime() - oldestDate.getTime()) / (1000 * 60 * 60 * 24)))
        averageSessionsPerDay = Math.round((data.length / daysDiff) * 10) / 10
      }
      
      // Most used facility
      const facilityCounts = new Map<string, number>()
      data.forEach(s => {
        if (s.facility_name) {
          facilityCounts.set(s.facility_name, (facilityCounts.get(s.facility_name) || 0) + 1)
        }
      })
      let mostUsedFacility = ""
      let maxCount = 0
      facilityCounts.forEach((count, facility) => {
        if (count > maxCount) {
          maxCount = count
          mostUsedFacility = facility
        }
      })
      
      // Oldest pending request (in hours)
      const pendingSessions = data.filter(s => 
        s.status === "PENDING_ACK" || s.status === "PENDING_APPROVAL" || s.status === "pending_approval"
      )
      let oldestPendingHours = 0
      if (pendingSessions.length > 0) {
        const pendingDates = pendingSessions
          .map(s => s.requested_at ? new Date(s.requested_at) : null)
          .filter((d): d is Date => d !== null)
        if (pendingDates.length > 0) {
          const oldestPending = new Date(Math.min(...pendingDates.map(d => d.getTime())))
          oldestPendingHours = Math.floor((now.getTime() - oldestPending.getTime()) / (1000 * 60 * 60))
        }
      }

      setStats({
        sessionsToday: todaySessions.length,
        sessionsThisWeek: weekSessions.length,
        sessionsThisMonth: monthSessions.length,
        totalSessions: data.length,
        emergencyCases,
        highPriorityCases,
        mediumPriorityCases,
        lowPriorityCases,
        averageETA: 15, // This would need to be calculated from actual ETA data
        activeTriages,
        completedTriages,
        pendingRequests,
        approvedRequests,
        rejectedRequests,
        totalFacilities: uniqueFacilities.size,
        successRate,
        completionRate,
        rejectionRate,
        alertRequests,
        appointmentRequests,
        averageSessionsPerDay,
        mostUsedFacility,
        oldestPendingRequest: oldestPendingHours,
      })
    } catch (error) {
      console.error("Failed to load recent activity:", error)
      toast({
        variant: "destructive",
        title: "Error",
        description: "Failed to load recent activity",
      })
    } finally {
      setIsLoading(false)
    }
  }

  const formatTimeAgo = (dateString?: string) => {
    if (!dateString) return "Unknown"
    const date = new Date(dateString)
    const now = new Date()
    const diffMs = now.getTime() - date.getTime()
    const diffMins = Math.floor(diffMs / 60000)
    const diffHours = Math.floor(diffMins / 60)
    const diffDays = Math.floor(diffHours / 24)

    if (diffMins < 1) return "Just now"
    if (diffMins < 60) return `${diffMins} ${diffMins === 1 ? "minute" : "minutes"} ago`
    if (diffHours < 24) return `${diffHours} ${diffHours === 1 ? "hour" : "hours"} ago`
    return `${diffDays} ${diffDays === 1 ? "day" : "days"} ago`
  }

  const getTriageBadgeVariant = (level?: string) => {
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

  const recentSessions = sessions.slice(0, 5) // Show latest 5

  // Show welcome screen for non-authenticated users
  if (!isAuthenticated) {
    return (
      <AppShell>
        <div className="space-y-8">
          {/* Hero Section */}
          <div className="text-center space-y-4 py-8">
            <div className="flex items-center justify-center gap-3">
              <Stethoscope className="h-12 w-12 text-primary" />
              <h1 className="text-4xl md:text-5xl font-bold tracking-tight">Emergency Care Navigator</h1>
            </div>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              Multi-agent emergency care triage and facility routing system for faster decision-making and clinician handoff.
            </p>
          </div>

          {/* Features Grid */}
          <div className="grid gap-6 md:grid-cols-3">
            <Card>
              <CardHeader>
                <Zap className="h-8 w-8 text-primary mb-2" />
                <CardTitle>Fast Triage</CardTitle>
                <CardDescription>
                  AI-powered emergency assessment in seconds
                </CardDescription>
              </CardHeader>
            </Card>
            <Card>
              <CardHeader>
                <Shield className="h-8 w-8 text-primary mb-2" />
                <CardTitle>Smart Routing</CardTitle>
                <CardDescription>
                  Automated facility matching based on availability
                </CardDescription>
              </CardHeader>
            </Card>
            <Card>
              <CardHeader>
                <Users className="h-8 w-8 text-primary mb-2" />
                <CardTitle>Seamless Handoff</CardTitle>
                <CardDescription>
                  Real-time communication between patients and hospitals
                </CardDescription>
              </CardHeader>
            </Card>
          </div>

          {/* Quick Actions */}
          <div>
            <h2 className="text-2xl font-semibold mb-4">Quick Actions</h2>
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              <Link href="/triage">
                <Card className="hover:bg-accent transition-colors cursor-pointer h-full border-primary/20">
                  <CardHeader>
                    <div className="flex items-center gap-3">
                      <div className="p-2 bg-primary/10 rounded-lg">
                        <Stethoscope className="h-6 w-6 text-primary" />
                      </div>
                      <div>
                        <CardTitle className="text-base">Start Emergency Triage</CardTitle>
                        <CardDescription className="text-xs">Begin assessment now</CardDescription>
                      </div>
                    </div>
                  </CardHeader>
                </Card>
              </Link>
              <Link href="/hospital">
                <Card className="hover:bg-accent transition-colors cursor-pointer h-full border-blue-500/20">
                  <CardHeader>
                    <div className="flex items-center gap-3">
                      <div className="p-2 bg-blue-500/10 rounded-lg">
                        <Building2 className="h-6 w-6 text-blue-600" />
                      </div>
                      <div>
                        <CardTitle className="text-base">Hospital Panel</CardTitle>
                        <CardDescription className="text-xs">Manage bookings & requests</CardDescription>
                      </div>
                    </div>
                  </CardHeader>
                </Card>
              </Link>
              <Link href="/login">
                <Card className="hover:bg-accent transition-colors cursor-pointer h-full border-green-500/20">
                  <CardHeader>
                    <div className="flex items-center gap-3">
                      <div className="p-2 bg-green-500/10 rounded-lg">
                        <LogIn className="h-6 w-6 text-green-600" />
                      </div>
                      <div>
                        <CardTitle className="text-base">Login / Sign Up</CardTitle>
                        <CardDescription className="text-xs">Access your account</CardDescription>
                      </div>
                    </div>
                  </CardHeader>
                </Card>
              </Link>
            </div>
          </div>

          {/* Public Metrics */}
          <div>
            <h2 className="text-2xl font-semibold mb-4">Platform Overview</h2>
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Fast Response</CardTitle>
                  <Zap className="h-4 w-4 text-primary" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">&lt; 30s</div>
                  <p className="text-xs text-muted-foreground">Average triage time</p>
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Smart Routing</CardTitle>
                  <Shield className="h-4 w-4 text-primary" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">AI-Powered</div>
                  <p className="text-xs text-muted-foreground">Facility matching</p>
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">24/7 Available</CardTitle>
                  <Clock className="h-4 w-4 text-primary" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">Always On</div>
                  <p className="text-xs text-muted-foreground">Round-the-clock service</p>
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Secure & Private</CardTitle>
                  <Shield className="h-4 w-4 text-primary" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">HIPAA</div>
                  <p className="text-xs text-muted-foreground">Compliant system</p>
                </CardContent>
              </Card>
            </div>
          </div>

          {/* Image/Visual Section */}
          <Card className="border-primary/20 bg-gradient-to-br from-primary/5 to-primary/10">
            <CardContent className="pt-6">
              <div className="grid md:grid-cols-2 gap-6 items-center">
                <div className="space-y-4">
                  <h3 className="text-2xl font-semibold">Emergency Care Made Simple</h3>
                  <p className="text-muted-foreground">
                    Our multi-agent system uses AI to quickly assess your emergency, find the nearest available facility, 
                    and coordinate seamless handoff to healthcare providers.
                  </p>
                  <div className="flex flex-wrap gap-2">
                    <Badge variant="secondary">AI Triage</Badge>
                    <Badge variant="secondary">Real-time Routing</Badge>
                    <Badge variant="secondary">Instant Notifications</Badge>
                    <Badge variant="secondary">Secure Platform</Badge>
                  </div>
                </div>
                <div className="flex items-center justify-center p-8 bg-background/50 rounded-lg border border-primary/20">
                  <div className="text-center space-y-4">
                    <div className="flex items-center justify-center gap-4">
                      <div className="p-4 bg-primary/10 rounded-full">
                        <Stethoscope className="h-12 w-12 text-primary" />
                      </div>
                      <ArrowRight className="h-6 w-6 text-muted-foreground" />
                      <div className="p-4 bg-blue-500/10 rounded-full">
                        <Building2 className="h-12 w-12 text-blue-600" />
                      </div>
                    </div>
                    <p className="text-sm text-muted-foreground font-medium">
                      From Assessment to Care
                    </p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* CTA Section */}
          <Card className="border-primary/20 bg-gradient-to-r from-primary/10 to-primary/5">
            <CardContent className="pt-6">
              <div className="text-center space-y-6">
                <div>
                  <h3 className="text-2xl font-semibold mb-2">Get Started Today</h3>
                  <p className="text-muted-foreground">
                    Sign in to access your emergency care dashboard or create a new account
                  </p>
                </div>
                <div className="flex items-center justify-center gap-4">
                  <Link href="/login">
                    <Button size="lg" className="gap-2">
                      <LogIn className="h-5 w-5" />
                      Login
                    </Button>
                  </Link>
                  <Link href="/login?register=true">
                    <Button size="lg" variant="outline" className="gap-2">
                      Sign Up
                      <ArrowRight className="h-5 w-5" />
                    </Button>
                  </Link>
                </div>
              </div>
            </CardContent>
          </Card>

        </div>
      </AppShell>
    )
  }

  // Authenticated user dashboard
  return (
    <AppShell>
      <div className="space-y-6">
        {/* Welcome Section */}
        <div className="space-y-2">
          <h1 className="text-3xl font-bold tracking-tight">
            Welcome back, {user?.name || "User"}!
          </h1>
          <p className="text-muted-foreground">
            Here's your emergency care dashboard overview
          </p>
        </div>

        {/* CTA */}
        <Card className="border-primary/20 bg-primary/5">
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-lg font-semibold mb-1">Start New Triage</h3>
                <p className="text-sm text-muted-foreground">
                  Begin a new emergency intake and triage assessment
                </p>
              </div>
              <Link href="/triage">
                <Button size="lg">
                  Start Triage
                  <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
              </Link>
            </div>
          </CardContent>
        </Card>

        {/* Quick Stats */}
        <div>
          <h2 className="text-2xl font-semibold mb-4">Dashboard Metrics</h2>
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Total Sessions</CardTitle>
                <BarChart3 className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{stats.totalSessions}</div>
                <p className="text-xs text-muted-foreground">{stats.sessionsToday} today</p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Emergency Cases</CardTitle>
                <AlertTriangle className="h-4 w-4 text-destructive" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-destructive">{stats.emergencyCases}</div>
                <p className="text-xs text-muted-foreground">Require immediate attention</p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Active Triages</CardTitle>
                <Activity className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{stats.activeTriages}</div>
                <p className="text-xs text-muted-foreground">In progress</p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Completed</CardTitle>
                <CheckCircle className="h-4 w-4 text-green-600" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-green-600">{stats.completedTriages}</div>
                <p className="text-xs text-muted-foreground">Successfully processed</p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Pending Requests</CardTitle>
                <Clock className="h-4 w-4 text-yellow-600" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-yellow-600">{stats.pendingRequests}</div>
                <p className="text-xs text-muted-foreground">Awaiting response</p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Approved</CardTitle>
                <CheckCircle className="h-4 w-4 text-green-600" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-green-600">{stats.approvedRequests}</div>
                <p className="text-xs text-muted-foreground">Confirmed bookings</p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Rejected</CardTitle>
                <XCircle className="h-4 w-4 text-destructive" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-destructive">{stats.rejectedRequests}</div>
                <p className="text-xs text-muted-foreground">Declined requests</p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Success Rate</CardTitle>
                <TrendingUp className="h-4 w-4 text-green-600" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-green-600">{stats.successRate}%</div>
                <p className="text-xs text-muted-foreground">Approval percentage</p>
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Quick Actions */}
        <div>
          <h2 className="text-2xl font-semibold mb-4">Quick Actions</h2>
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
            <Link href="/triage">
              <Card className="hover:bg-accent transition-colors cursor-pointer h-full">
                <CardHeader>
                  <div className="flex items-center gap-3">
                    <div className="p-2 bg-primary/10 rounded-lg">
                      <PlusCircle className="h-6 w-6 text-primary" />
                    </div>
                    <div>
                      <CardTitle className="text-base">New Triage</CardTitle>
                      <CardDescription className="text-xs">Start emergency assessment</CardDescription>
                    </div>
                  </div>
                </CardHeader>
              </Card>
            </Link>
            <Link href="/requests">
              <Card className="hover:bg-accent transition-colors cursor-pointer h-full">
                <CardHeader>
                  <div className="flex items-center gap-3">
                    <div className="p-2 bg-blue-500/10 rounded-lg">
                      <FileCheck className="h-6 w-6 text-blue-600" />
                    </div>
                    <div>
                      <CardTitle className="text-base">My Requests</CardTitle>
                      <CardDescription className="text-xs">View booking status</CardDescription>
                    </div>
                  </div>
                </CardHeader>
              </Card>
            </Link>
            <Link href="/sessions">
              <Card className="hover:bg-accent transition-colors cursor-pointer h-full">
                <CardHeader>
                  <div className="flex items-center gap-3">
                    <div className="p-2 bg-green-500/10 rounded-lg">
                      <ActivitySquare className="h-6 w-6 text-green-600" />
                    </div>
                    <div>
                      <CardTitle className="text-base">All Sessions</CardTitle>
                      <CardDescription className="text-xs">View session history</CardDescription>
                    </div>
                  </div>
                </CardHeader>
              </Card>
            </Link>
            <Link href="/profile">
              <Card className="hover:bg-accent transition-colors cursor-pointer h-full">
                <CardHeader>
                  <div className="flex items-center gap-3">
                    <div className="p-2 bg-purple-500/10 rounded-lg">
                      <BookOpen className="h-6 w-6 text-purple-600" />
                    </div>
                    <div>
                      <CardTitle className="text-base">Health History</CardTitle>
                      <CardDescription className="text-xs">Manage health records</CardDescription>
                    </div>
                  </div>
                </CardHeader>
              </Card>
            </Link>
          </div>
        </div>

        {/* Health Insights */}
        {memory && (
          <div>
            <h2 className="text-2xl font-semibold mb-4">Health Insights</h2>
            <div className="grid gap-4 md:grid-cols-3">
              <Card>
                <CardHeader>
                  <CardTitle className="text-base flex items-center gap-2">
                    <Heart className="h-5 w-5 text-primary" />
                    Health Conditions
                  </CardTitle>
                  <CardDescription>Tracked conditions</CardDescription>
                </CardHeader>
                <CardContent>
                  {memory.health_conditions && memory.health_conditions.length > 0 ? (
                    <div className="space-y-2">
                      <div className="text-2xl font-bold">{memory.health_conditions.length}</div>
                      <div className="flex flex-wrap gap-1">
                        {memory.health_conditions.slice(0, 3).map((condition, idx) => (
                          <Badge key={idx} variant="secondary" className="text-xs">
                            {condition}
                          </Badge>
                        ))}
                        {memory.health_conditions.length > 3 && (
                          <Badge variant="outline" className="text-xs">
                            +{memory.health_conditions.length - 3} more
                          </Badge>
                        )}
                      </div>
                    </div>
                  ) : (
                    <div className="text-muted-foreground text-sm">No conditions tracked</div>
                  )}
                </CardContent>
              </Card>
              <Card>
                <CardHeader>
                  <CardTitle className="text-base flex items-center gap-2">
                    <Building2 className="h-5 w-5 text-primary" />
                    Preferred Facility
                  </CardTitle>
                  <CardDescription>Last used facility</CardDescription>
                </CardHeader>
                <CardContent>
                  {memory.last_facility_used ? (
                    <div>
                      <div className="text-lg font-semibold truncate">{memory.last_facility_used}</div>
                      <p className="text-xs text-muted-foreground mt-1">Most recent visit</p>
                    </div>
                  ) : (
                    <div className="text-muted-foreground text-sm">No facility history</div>
                  )}
                </CardContent>
              </Card>
              <Card>
                <CardHeader>
                  <CardTitle className="text-base flex items-center gap-2">
                    <MapPin className="h-5 w-5 text-primary" />
                    Preferred Location
                  </CardTitle>
                  <CardDescription>Saved location preference</CardDescription>
                </CardHeader>
                <CardContent>
                  {memory.preferred_city ? (
                    <div>
                      <div className="text-lg font-semibold">{memory.preferred_city}</div>
                      <p className="text-xs text-muted-foreground mt-1">Default location</p>
                    </div>
                  ) : (
                    <div className="text-muted-foreground text-sm">No location set</div>
                  )}
                </CardContent>
              </Card>
            </div>
          </div>
        )}
      </div>
    </AppShell>
  )
}

