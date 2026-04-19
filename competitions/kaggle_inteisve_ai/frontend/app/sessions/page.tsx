"use client"

import { useState, useEffect } from "react"
import { AppShell } from "@/components/layout/app-shell"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Eye, RotateCcw, Loader2 } from "lucide-react"
import Link from "next/link"
import { getPatientSessions, PatientSession } from "@/lib/api"
import { useAuth } from "@/contexts/auth-context"
import { useToast } from "@/hooks/use-toast"
import { useRouter } from "next/navigation"

export default function SessionsPage() {
  const { user } = useAuth()
  const { toast } = useToast()
  const router = useRouter()
  const [sessions, setSessions] = useState<PatientSession[]>([])
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    loadSessions()
  }, [user])

  const loadSessions = async () => {
    try {
      setIsLoading(true)
      const data = await getPatientSessions(user?.email)
      setSessions(data)
    } catch (error) {
      console.error("Failed to load sessions:", error)
      toast({
        variant: "destructive",
        title: "Error",
        description: "Failed to load sessions",
      })
    } finally {
      setIsLoading(false)
    }
  }
  const getTriageBadgeVariant = (level: string) => {
    switch (level) {
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

  const getBookingBadgeVariant = (status: string) => {
    switch (status.toLowerCase()) {
      case "confirmed":
      case "acknowledged":
        return "success"
      case "pending_approval":
      case "pending_ack":
        return "warning"
      case "rejected":
        return "destructive"
      case "skipped":
        return "outline"
      default:
        return "outline"
    }
  }

  return (
    <AppShell>
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Sessions</h1>
          <p className="text-muted-foreground">
            View and manage historical triage sessions
          </p>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>Triage Sessions</CardTitle>
            <CardDescription>
              Complete history of emergency care assessments
            </CardDescription>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
              </div>
            ) : sessions.length === 0 ? (
              <div className="text-center py-8">
                <p className="text-muted-foreground">No sessions found</p>
              </div>
            ) : (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Patient</TableHead>
                    <TableHead>Location</TableHead>
                    <TableHead>Triage Level</TableHead>
                    <TableHead>Facility</TableHead>
                    <TableHead>Booking Status</TableHead>
                    <TableHead>Timestamp</TableHead>
                    <TableHead className="text-right">Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {sessions.map((session) => (
                    <TableRow key={session.session_id}>
                      <TableCell className="font-medium whitespace-nowrap">{session.patient_name}</TableCell>
                      <TableCell className="whitespace-nowrap">{session.location}</TableCell>
                      <TableCell className="whitespace-nowrap">
                        <Badge variant={getTriageBadgeVariant(session.triage_level)} className="whitespace-nowrap">
                          {session.triage_level.toUpperCase()}
                        </Badge>
                      </TableCell>
                      <TableCell className="whitespace-nowrap">{session.facility_name}</TableCell>
                      <TableCell className="whitespace-nowrap">
                        <Badge variant={getBookingBadgeVariant(session.booking_status)} className="whitespace-nowrap">
                          {session.booking_status.replace("_", " ").toUpperCase()}
                        </Badge>
                      </TableCell>
                      <TableCell className="whitespace-nowrap">
                        {new Date(session.requested_at).toLocaleString()}
                      </TableCell>
                      <TableCell className="text-right whitespace-nowrap">
                        <div className="flex items-center justify-end gap-2">
                          <Link href={`/triage?session=${session.session_id}`}>
                            <Button variant="outline" size="sm" className="whitespace-nowrap">
                              <Eye className="h-4 w-4 mr-1" />
                              View
                            </Button>
                          </Link>
                          {(session.booking_status === "REJECTED" || session.booking_status === "rejected") && (
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => router.push(`/triage?session=${session.session_id}&resubmit=true`)}
                              className="whitespace-nowrap"
                            >
                              <RotateCcw className="h-4 w-4 mr-1" />
                              Resubmit
                            </Button>
                          )}
                        </div>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            )}
          </CardContent>
        </Card>
      </div>
    </AppShell>
  )
}

