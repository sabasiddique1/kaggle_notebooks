"use client"

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Facility, BookingState, TriageResult } from "@/lib/api"
import { Building2, Clock, CheckCircle, AlertCircle, Phone, Mail, Send } from "lucide-react"
import { TriageLevelBadge } from "@/components/triage/triage-level-badge"

interface BookingDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  facility: Facility | null
  triage: TriageResult | null
  booking: BookingState | null
  onApprove?: () => void  // Optional - patients don't approve, only hospitals
  isApproving?: boolean
  sessionId?: string
}

export function BookingDialog({
  open,
  onOpenChange,
  facility,
  triage,
  booking,
  onApprove,
  isApproving,
  sessionId,
}: BookingDialogProps) {
  if (!facility || !triage || !booking) return null

  const isConfirmed = booking.status === "CONFIRMED" || booking.status === "confirmed"
  const isAcknowledged = booking.status === "ACKNOWLEDGED"
  const isPendingApproval = booking.status === "PENDING_APPROVAL" || booking.status === "pending_approval"
  const isPendingAck = booking.status === "PENDING_ACK"
  const isPending = isPendingApproval || isPendingAck
  const isRejected = booking.status === "REJECTED" || booking.status === "rejected"
  
  // Determine request type from booking or triage level
  const requestType = booking.request_type || (triage.level === "emergency" || triage.level === "high" ? "alert" : "appointment")

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Building2 className="h-5 w-5" />
            {requestType === "alert" ? "Pre-Arrival Alert" : "Appointment Booking"}
          </DialogTitle>
          <DialogDescription>
            {isPending 
              ? `Your ${requestType === "alert" ? "pre-arrival alert" : "appointment request"} has been sent to the facility. The hospital will review and respond shortly.`
              : isConfirmed || isAcknowledged
              ? `Your ${requestType === "alert" ? "pre-arrival alert has been acknowledged" : "appointment has been confirmed"} by the facility.`
              : "Review your booking request details"}
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-4 py-4">
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium">Facility</span>
              <Badge variant="outline">{facility.kind}</Badge>
            </div>
            <p className="text-sm">{facility.name}</p>
            <p className="text-xs text-muted-foreground">{facility.address}</p>
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium">Triage Level</span>
              <TriageLevelBadge level={triage.level} />
            </div>
          </div>

          <div className="flex items-center gap-4 text-sm">
            <div className="flex items-center gap-1">
              <Clock className="h-4 w-4 text-muted-foreground" />
              <span className="text-muted-foreground">ETA:</span>
              <span className="font-medium">
                {facility.eta_minutes ? `${facility.eta_minutes} min` : "Unknown"}
              </span>
            </div>
            <div className="flex items-center gap-1">
              <span className="text-muted-foreground">Distance:</span>
              <span className="font-medium">{facility.distance_km?.toFixed(1)} km</span>
            </div>
          </div>

          {/* Request Sent Status */}
          {isPending && (
            <div className="rounded-lg border border-primary/20 bg-primary/5 p-3">
              <div className="flex items-start gap-2">
                <Send className="h-4 w-4 text-primary mt-0.5" />
                <div className="flex-1">
                  <p className="text-sm font-medium mb-1">
                    {requestType === "alert" ? "Pre-Arrival Alert Sent" : "Appointment Request Sent"}
                  </p>
                  <p className="text-xs text-muted-foreground mb-2">
                    Your request has been sent to {facility.name}. The hospital will review and respond shortly.
                  </p>
                  <p className="text-xs text-muted-foreground">
                    Status: {isPendingAck ? "Waiting for Acknowledgement" : "Waiting for Approval"}
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Acknowledged Status (for alerts) */}
          {isAcknowledged && (
            <div className="rounded-lg border border-success/20 bg-success/5 p-3">
              <div className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-success mt-0.5" />
                <div className="flex-1">
                  <p className="text-sm font-medium mb-1">Alert Acknowledged</p>
                  <p className="text-xs text-muted-foreground">{booking.note}</p>
                  {booking.acknowledged_at && (
                    <p className="text-xs text-muted-foreground mt-1">
                      Acknowledged at: {new Date(booking.acknowledged_at).toLocaleString()}
                    </p>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Confirmed Status (for appointments) */}
          {isConfirmed && (
            <div className="rounded-lg border border-success/20 bg-success/5 p-3">
              <div className="flex items-start gap-2">
                <CheckCircle className="h-4 w-4 text-success mt-0.5" />
                <div className="flex-1">
                  <p className="text-sm font-medium mb-1">Appointment Confirmed</p>
                  <p className="text-xs text-muted-foreground">{booking.note}</p>
                  {booking.approved_at && (
                    <p className="text-xs text-muted-foreground mt-1">
                      Confirmed at: {new Date(booking.approved_at).toLocaleString()}
                    </p>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Rejected Status */}
          {isRejected && (
            <div className="rounded-lg border border-destructive/20 bg-destructive/5 p-3">
              <div className="flex items-start gap-2">
                <AlertCircle className="h-4 w-4 text-destructive mt-0.5" />
                <div className="flex-1">
                  <p className="text-sm font-medium mb-1">Request Rejected</p>
                  <p className="text-xs text-muted-foreground">{booking.note}</p>
                </div>
              </div>
            </div>
          )}

          {/* Contact Information */}
          <div className="rounded-lg border bg-muted/50 p-4 space-y-2">
            <p className="text-sm font-medium">Contact Information</p>
            <div className="space-y-2 text-sm text-muted-foreground">
              <div className="flex items-start gap-2">
                <Phone className="h-4 w-4 mt-0.5" />
                <div>
                  <p className="font-medium text-foreground">Phone</p>
                  <p>Please contact {facility.name} directly for availability and specific services</p>
                </div>
              </div>
              <div className="flex items-start gap-2">
                <Mail className="h-4 w-4 mt-0.5" />
                <div>
                  <p className="font-medium text-foreground">Address</p>
                  <p>{facility.address}</p>
                </div>
              </div>
            </div>
            <div className="rounded-lg border border-warning/20 bg-warning/5 p-2 mt-2">
              <p className="text-xs text-muted-foreground">
                <strong>Note:</strong> Contact information is not available from the facility database. 
                Please call the facility directly or visit their website for the most up-to-date contact information.
              </p>
            </div>
          </div>
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            {isPending ? "Close" : isConfirmed || isAcknowledged ? "Close" : "Cancel"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

