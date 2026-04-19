"use client"

import { useState, useEffect } from "react"
import { AppShell } from "@/components/layout/app-shell"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { getPatientRequests, PatientRequest } from "@/lib/api"
import { useAuth } from "@/contexts/auth-context"
import { useToast } from "@/hooks/use-toast"
import { Loader2, CheckCircle, XCircle, Clock, AlertCircle, Building2, MapPin, Calendar, FileText, AlertTriangle, RotateCcw } from "lucide-react"
import { useRouter } from "next/navigation"

export default function RequestsPage() {
  const { user } = useAuth()
  const { toast } = useToast()
  const router = useRouter()
  const [requests, setRequests] = useState<PatientRequest[]>([])
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    loadRequests()
  }, [user])

  const loadRequests = async () => {
    try {
      setIsLoading(true)
      const data = await getPatientRequests(user?.email)
      setRequests(data)
    } catch (error) {
      console.error("Failed to load requests:", error)
      toast({
        variant: "destructive",
        title: "Error",
        description: "Failed to load your requests",
      })
    } finally {
      setIsLoading(false)
    }
  }

  const getStatusBadge = (status: string, requestType: string) => {
    if (status === "CONFIRMED" || status === "ACKNOWLEDGED" || status === "confirmed") {
      return <Badge variant="default" className="bg-green-600 whitespace-nowrap"><CheckCircle className="h-3 w-3 mr-1" />Confirmed</Badge>
    }
    if (status === "REJECTED" || status === "rejected") {
      return <Badge variant="destructive" className="whitespace-nowrap"><XCircle className="h-3 w-3 mr-1" />Rejected</Badge>
    }
    if (status === "PENDING_ACK" || status === "PENDING_APPROVAL" || status === "pending_approval") {
      return <Badge variant="outline" className="border-yellow-500 text-yellow-600 whitespace-nowrap"><Clock className="h-3 w-3 mr-1" />Pending</Badge>
    }
    return <Badge variant="secondary" className="whitespace-nowrap">{status}</Badge>
  }

  const getTriageBadge = (level: string) => {
    const variants: Record<string, "destructive" | "default" | "secondary"> = {
      emergency: "destructive",
      high: "destructive",
      medium: "default",
      low: "secondary"
    }
    return <Badge variant={variants[level] || "default"} className="whitespace-nowrap">{level.toUpperCase()}</Badge>
  }

  const pendingRequests = requests.filter(r => r.status === "PENDING_ACK" || r.status === "PENDING_APPROVAL" || r.status === "pending_approval")
  const otherRequests = requests.filter(r => r.status !== "PENDING_ACK" && r.status !== "PENDING_APPROVAL" && r.status !== "pending_approval")

  return (
    <AppShell>
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">My Requests</h1>
          <p className="text-muted-foreground">
            View and track all your hospital requests
          </p>
        </div>

        {isLoading ? (
          <Card>
            <CardContent className="pt-6">
              <div className="flex items-center justify-center py-8">
                <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
              </div>
            </CardContent>
          </Card>
        ) : requests.length === 0 ? (
          <Card>
            <CardContent className="pt-6">
              <div className="text-center py-8">
                <FileText className="h-12 w-12 mx-auto mb-4 opacity-50 text-muted-foreground" />
                <p className="text-lg font-medium mb-2">No requests yet</p>
                <p className="text-sm text-muted-foreground mb-4">
                  Submit a triage form to create your first request
                </p>
                <Button onClick={() => router.push("/triage")}>
                  Go to Triage Workspace
                </Button>
              </div>
            </CardContent>
          </Card>
        ) : (
          <>
            {/* Pending Requests */}
            {pendingRequests.length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Clock className="h-5 w-5 text-yellow-500" />
                    Pending Requests ({pendingRequests.length})
                  </CardTitle>
                  <CardDescription>Requests waiting for hospital response</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {pendingRequests.map((request) => (
                      <Card key={request.id} className="border-l-4 border-l-yellow-500">
                        <CardHeader>
                          <div className="flex items-start justify-between">
                            <div className="space-y-2">
                              <div className="flex items-center gap-2">
                                <Building2 className="h-4 w-4 text-muted-foreground" />
                                <CardTitle className="text-lg">{request.facility_name}</CardTitle>
                              </div>
                              <div className="flex items-center gap-2 flex-wrap">
                                <span className="whitespace-nowrap">{getTriageBadge(request.triage_level)}</span>
                                <Badge variant="outline" className="whitespace-nowrap flex items-center gap-1">
                                  {request.request_type === "alert" ? (
                                    <>
                                      <AlertTriangle className="h-3 w-3" /> Alert
                                    </>
                                  ) : (
                                    <>
                                      <Calendar className="h-3 w-3" /> Appointment
                                    </>
                                  )}
                                </Badge>
                                <span className="whitespace-nowrap">{getStatusBadge(request.status, request.request_type)}</span>
                              </div>
                            </div>
                          </div>
                        </CardHeader>
                        <CardContent className="space-y-3">
                          <div className="flex items-center gap-2 text-sm text-muted-foreground whitespace-nowrap">
                            <Calendar className="h-4 w-4" />
                            <span>Requested: {new Date(request.requested_at).toLocaleString()}</span>
                          </div>
                          {request.location && (
                            <div className="flex items-center gap-2 text-sm">
                              <MapPin className="h-4 w-4 text-muted-foreground" />
                              <span>{request.location}</span>
                            </div>
                          )}
                          {request.symptoms && request.symptoms.length > 0 && (
                            <div>
                              <p className="text-sm font-medium mb-1">Symptoms:</p>
                              <div className="flex flex-wrap gap-1">
                                {request.symptoms.map((symptom, idx) => (
                                  <Badge key={idx} variant="outline" className="text-xs">{symptom}</Badge>
                                ))}
                              </div>
                            </div>
                          )}
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Other Requests */}
            {otherRequests.length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle>Request History</CardTitle>
                  <CardDescription>Completed and rejected requests</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {otherRequests.map((request) => (
                      <Card key={request.id} className={`border-l-4 ${
                        request.status === "REJECTED" || request.status === "rejected" 
                          ? "border-l-red-500" 
                          : "border-l-green-500"
                      }`}>
                        <CardHeader>
                          <div className="flex items-start justify-between">
                            <div className="space-y-2">
                              <div className="flex items-center gap-2">
                                <Building2 className="h-4 w-4 text-muted-foreground" />
                                <CardTitle className="text-lg">{request.facility_name}</CardTitle>
                              </div>
                              <div className="flex items-center gap-2 flex-wrap">
                                <span className="whitespace-nowrap">{getTriageBadge(request.triage_level)}</span>
                                <Badge variant="outline" className="whitespace-nowrap flex items-center gap-1">
                                  {request.request_type === "alert" ? (
                                    <>
                                      <AlertTriangle className="h-3 w-3" /> Alert
                                    </>
                                  ) : (
                                    <>
                                      <Calendar className="h-3 w-3" /> Appointment
                                    </>
                                  )}
                                </Badge>
                                <span className="whitespace-nowrap">{getStatusBadge(request.status, request.request_type)}</span>
                              </div>
                            </div>
                          </div>
                        </CardHeader>
                        <CardContent className="space-y-3">
                          <div className="flex items-center gap-2 text-sm text-muted-foreground whitespace-nowrap">
                            <Calendar className="h-4 w-4" />
                            <span>Requested: {new Date(request.requested_at).toLocaleString()}</span>
                          </div>
                          {request.approved_at && (
                            <div className="flex items-center gap-2 text-sm text-green-600 whitespace-nowrap">
                              <CheckCircle className="h-4 w-4" />
                              <span>Approved: {new Date(request.approved_at).toLocaleString()}</span>
                            </div>
                          )}
                          {request.acknowledged_at && (
                            <div className="flex items-center gap-2 text-sm text-green-600 whitespace-nowrap">
                              <CheckCircle className="h-4 w-4" />
                              <span>Acknowledged: {new Date(request.acknowledged_at).toLocaleString()}</span>
                            </div>
                          )}
                          {request.rejected_at && (
                            <div className="space-y-2">
                              <div className="flex items-center gap-2 text-sm text-red-600 whitespace-nowrap">
                                <XCircle className="h-4 w-4" />
                                <span>Rejected: {new Date(request.rejected_at).toLocaleString()}</span>
                              </div>
                              {request.rejection_reason && (
                                <div className="rounded-lg border border-red-200 bg-red-50 p-3">
                                  <p className="text-sm font-medium text-red-900 mb-1">Rejection Reason:</p>
                                  <p className="text-sm text-red-700">{request.rejection_reason}</p>
                                </div>
                              )}
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={() => router.push(`/triage?session=${request.session_id}&resubmit=true`)}
                                className="mt-2"
                              >
                                <RotateCcw className="h-4 w-4 mr-2" />
                                Resubmit Request
                              </Button>
                            </div>
                          )}
                          {request.location && (
                            <div className="flex items-center gap-2 text-sm">
                              <MapPin className="h-4 w-4 text-muted-foreground" />
                              <span>{request.location}</span>
                            </div>
                          )}
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}
          </>
        )}
      </div>
    </AppShell>
  )
}

