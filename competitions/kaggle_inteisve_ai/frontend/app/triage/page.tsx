"use client"

import { useState, useEffect } from "react"
import { AppShell } from "@/components/layout/app-shell"
import { IntakeForm } from "@/components/triage/intake-form"
import { TriageResultCard } from "@/components/triage/triage-result-card"
import { FacilitiesList } from "@/components/facilities/facilities-list"
import { SBARPanel } from "@/components/sbar/sbar-panel"
import { BookingDialog } from "@/components/booking/booking-dialog"
import { FacilityDetailsDialog } from "@/components/facilities/facility-details-dialog"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Separator } from "@/components/ui/separator"
import { processIntake, approveBooking, getBookingStatus, getPendingRequests, IntakeAnswers, Recommendation, Facility, getMemory, updateMemory, addHealthCondition } from "@/lib/api"
import { useAuth } from "@/contexts/auth-context"
import { useToast } from "@/hooks/use-toast"
import { ClipboardList, CheckCircle, Clock, MapPin } from "lucide-react"

export default function TriagePage() {
  const { toast } = useToast()
  const { user } = useAuth()
  const [isLoading, setIsLoading] = useState(false)
  const [recommendation, setRecommendation] = useState<Recommendation | null>(null)
  const [sbarOpen, setSbarOpen] = useState(false)
  const [bookingDialogOpen, setBookingDialogOpen] = useState(false)
  const [facilityDetailsOpen, setFacilityDetailsOpen] = useState(false)
  const [selectedFacility, setSelectedFacility] = useState<Facility | null>(null)
  const [isApproving, setIsApproving] = useState(false)
  const [preferredLocation, setPreferredLocation] = useState<string | null>(null)
  const [isLoadingLocation, setIsLoadingLocation] = useState(true)
  const [requestedHospitals, setRequestedHospitals] = useState<string[]>([])
  const [showFacilitiesAfterRejection, setShowFacilitiesAfterRejection] = useState(false)

  useEffect(() => {
    // Load preferred location from memory
    const loadPreferredLocation = async () => {
      try {
        setIsLoadingLocation(true)
        const memory = await getMemory()
        if (memory.preferred_city) {
          setPreferredLocation(memory.preferred_city)
        }
      } catch (error) {
        console.error("Failed to load preferred location:", error)
      } finally {
        setIsLoadingLocation(false)
      }
    }
    loadPreferredLocation()
    
    // Load pending requests to track requested hospitals
    const loadPendingRequests = async () => {
      if (!user?.email) return
      try {
        const pending = await getPendingRequests(user.email)
        setRequestedHospitals(pending.requested_hospitals)
      } catch (error) {
        console.error("Failed to load pending requests:", error)
      }
    }
    loadPendingRequests()
  }, [user])

  const handleSubmit = async (intake: IntakeAnswers) => {
    setIsLoading(true)
    try {
      // Check for 3 pending requests limit
      try {
        const pending = await getPendingRequests(user?.email)
        if (pending.count >= 3) {
          toast({
            variant: "destructive",
            title: "Request Limit Reached",
            description: "You have reached the limit of 3 pending requests. Please wait for responses or check your requests page.",
          })
          setIsLoading(false)
          return
        }
        setRequestedHospitals(pending.requested_hospitals)
      } catch (error) {
        console.error("Failed to check pending requests:", error)
      }
      
      const result = await processIntake(intake, user?.email)
      setRecommendation(result)
      setShowFacilitiesAfterRejection(false)
      
      // Update preferred location if it changed and save to memory
      if (intake.location_query && intake.location_query !== preferredLocation) {
        setPreferredLocation(intake.location_query)
        try {
          await updateMemory({ preferred_city: intake.location_query })
        } catch (error) {
          console.error("Failed to save location:", error)
        }
      }

      // Save new symptoms to health history (if they're not already saved)
      if (intake.symptoms && intake.symptoms.length > 0) {
        try {
          const memory = await getMemory()
          const existingConditions = memory.health_conditions || []
          for (const symptom of intake.symptoms) {
            if (symptom.trim() && !existingConditions.includes(symptom.trim())) {
              await addHealthCondition(symptom.trim())
            }
          }
        } catch (error) {
          console.error("Failed to save symptoms to history:", error)
          // Don't show error toast for this - it's not critical
        }
      }
      
      toast({
        variant: "success",
        title: "Triage Complete",
        description: `Triage level: ${result.triage.level.toUpperCase()}. ${result.top_choices.length} facilities found.`,
      })
    } catch (error) {
      console.error("Failed to process intake:", error)
      toast({
        variant: "destructive",
        title: "Error",
        description: error instanceof Error ? error.message : "Failed to process intake",
      })
    } finally {
      setIsLoading(false)
    }
  }

  const handleSelectFacility = async (facility: Facility) => {
    // Check if already requested this hospital
    if (requestedHospitals.includes(facility.name)) {
      toast({
        variant: "destructive",
        title: "Already Requested",
        description: `You have already sent a request to ${facility.name}. Please select a different hospital.`,
      })
      return
    }
    
    // Check for 3 pending limit
    try {
      const pending = await getPendingRequests(user?.email)
      if (pending.count >= 3) {
        toast({
          variant: "destructive",
          title: "Request Limit Reached",
          description: "You have reached the limit of 3 pending requests. Please wait for responses.",
        })
        return
      }
    } catch (error) {
      console.error("Failed to check pending requests:", error)
    }
    
    setSelectedFacility(facility)
    setBookingDialogOpen(true)
    
    // Show toast that request is being sent
    if (recommendation?.booking) {
      const requestType = recommendation.booking.request_type || (recommendation.triage.level === "emergency" || recommendation.triage.level === "high" ? "alert" : "appointment")
      toast({
        variant: "default",
        title: "Request Sent",
        description: `Your ${requestType === "alert" ? "pre-arrival alert" : "appointment request"} has been sent to ${facility.name}. Waiting for hospital response.`,
      })
      
      // Add to requested hospitals
      setRequestedHospitals([...requestedHospitals, facility.name])
    }
  }

  // Note: Patients cannot approve bookings - only hospitals can
  // This function is kept for compatibility but won't be called
  const handleApproveBooking = async () => {
    // This should never be called from patient side
    console.warn("Patient cannot approve bookings - only hospitals can approve")
  }

  // Poll for booking status updates when booking is pending
  useEffect(() => {
    if (!recommendation?.session_id || !recommendation?.booking) return
    
    const booking = recommendation.booking
    const isPending = booking.status === "PENDING_ACK" || booking.status === "PENDING_APPROVAL" || booking.status === "pending_approval"
    
    if (!isPending) return

    const pollInterval = setInterval(async () => {
      try {
        const statusData = await getBookingStatus(recommendation.session_id!)
        const newStatus = statusData.booking.status
        
        // Check if status changed
        if (newStatus !== booking.status) {
          setRecommendation({
            ...recommendation,
            booking: statusData.booking,
            booking_status: newStatus,
          })
          
          // Show notification if approved/acknowledged
          if (newStatus === "CONFIRMED" || newStatus === "ACKNOWLEDGED" || newStatus === "confirmed") {
            toast({
              variant: "success",
              title: "Request Confirmed!",
              description: `Your ${statusData.booking.request_type === "alert" ? "pre-arrival alert" : "appointment"} has been ${newStatus === "ACKNOWLEDGED" ? "acknowledged" : "confirmed"} by ${statusData.booking.facility_name}`,
            })
          } else if (newStatus === "REJECTED" || newStatus === "rejected") {
            // Remove from requested hospitals
            setRequestedHospitals(prev => prev.filter(h => h !== statusData.booking.facility_name))
            
            toast({
              variant: "destructive",
              title: "Request Rejected",
              description: `Your request was rejected by ${statusData.booking.facility_name}. Please select another hospital from the list below.`,
              duration: 6000,
            })
            
            // Show facilities again so user can select another hospital
            setShowFacilitiesAfterRejection(true)
            setBookingDialogOpen(false)
          }
        }
      } catch (error) {
        console.error("Failed to check booking status:", error)
      }
    }, 5000) // Poll every 5 seconds

    return () => clearInterval(pollInterval)
  }, [recommendation?.session_id, recommendation?.booking?.status, toast])

  const timelineSteps = [
    { label: "Intake received", completed: !!recommendation },
    { label: "Triage computed", completed: !!recommendation },
    { label: "Facilities searched", completed: !!recommendation?.top_choices.length },
    { label: "Booking created", completed: recommendation?.booking_status === "pending_approval" || recommendation?.booking_status === "confirmed" },
  ]

  return (
    <AppShell
      currentLocation={preferredLocation || undefined}
      triageStatus={recommendation?.triage.level || null}
    >
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Triage Workspace</h1>
          <p className="text-muted-foreground">
            Enter patient information to begin emergency triage assessment
          </p>
        </div>

        <div className="grid gap-6 lg:grid-cols-3">
          {/* Left Column: Intake Form */}
          <div className="lg:col-span-1">
            <IntakeForm
              onSubmit={handleSubmit}
              isLoading={isLoading}
              lastLocation={preferredLocation || undefined}
            />
          </div>

          {/* Center Column: Triage Result & Timeline */}
          <div className="lg:col-span-1 space-y-4">
            <TriageResultCard result={recommendation?.triage || null} isLoading={isLoading} />

            {recommendation && (
              <Card>
                <CardContent className="pt-6">
                  <h3 className="font-semibold mb-4">Process Timeline</h3>
                  <div className="space-y-3">
                    {timelineSteps.map((step, index) => (
                      <div key={index} className="flex items-center gap-3">
                        <div
                          className={`flex h-8 w-8 items-center justify-center rounded-full ${
                            step.completed
                              ? "bg-primary text-primary-foreground"
                              : "bg-muted text-muted-foreground"
                          }`}
                        >
                          {step.completed ? (
                            <CheckCircle className="h-4 w-4" />
                          ) : (
                            <Clock className="h-4 w-4" />
                          )}
                        </div>
                        <span
                          className={`text-sm ${
                            step.completed ? "font-medium" : "text-muted-foreground"
                          }`}
                        >
                          {step.label}
                        </span>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}

            {recommendation && (
              <Button
                variant="outline"
                className="w-full"
                onClick={() => setSbarOpen(true)}
              >
                <ClipboardList className="h-4 w-4 mr-2" />
                View SBAR Handoff
              </Button>
            )}
          </div>

          {/* Right Column: Facilities */}
          <div className="lg:col-span-1">
            {(recommendation && (showFacilitiesAfterRejection || !recommendation.booking || recommendation.booking.status !== "REJECTED")) ? (
              <FacilitiesList
                facilities={recommendation.top_choices.filter(f => !requestedHospitals.includes(f.name))}
                onSelectFacility={handleSelectFacility}
                onViewDetails={(facility) => {
                  setSelectedFacility(facility)
                  setFacilityDetailsOpen(true)
                }}
              />
            ) : (
              <Card>
                <CardContent className="pt-6">
                  <div className="text-center text-muted-foreground py-8">
                    <MapPin className="h-12 w-12 mx-auto mb-4 opacity-50" />
                    <p>Run triage to see nearby facilities</p>
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      </div>

      {recommendation && (
        <SBARPanel
          open={sbarOpen}
          onOpenChange={setSbarOpen}
          handoffPacket={recommendation.handoff_packet}
        />
      )}

      {recommendation && (
        <BookingDialog
          open={bookingDialogOpen}
          onOpenChange={setBookingDialogOpen}
          facility={selectedFacility}
          triage={recommendation.triage}
          booking={recommendation.booking || null}
          onApprove={handleApproveBooking}
          isApproving={isApproving}
          sessionId={recommendation.session_id}
        />
      )}

      <FacilityDetailsDialog
        open={facilityDetailsOpen}
        onOpenChange={setFacilityDetailsOpen}
        facility={selectedFacility}
      />
    </AppShell>
  )
}

