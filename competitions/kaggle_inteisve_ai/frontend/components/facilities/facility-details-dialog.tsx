"use client"

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Facility } from "@/lib/api"
import { Building2, MapPin, Navigation, Clock, Phone, Mail } from "lucide-react"
import { Badge } from "@/components/ui/badge"

interface FacilityDetailsDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  facility: Facility | null
}

export function FacilityDetailsDialog({
  open,
  onOpenChange,
  facility,
}: FacilityDetailsDialogProps) {
  if (!facility) return null

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Building2 className="h-5 w-5" />
            {facility.name}
          </DialogTitle>
          <DialogDescription>Facility details and contact information</DialogDescription>
        </DialogHeader>
        <div className="space-y-4 py-4">
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <Badge variant="outline">{facility.kind}</Badge>
              {facility.source && (
                <span className="text-xs text-muted-foreground">Source: {facility.source}</span>
              )}
            </div>

            <div className="space-y-2">
              <div className="flex items-start gap-2">
                <MapPin className="h-4 w-4 text-muted-foreground mt-0.5" />
                <div>
                  <p className="text-sm font-medium">Address</p>
                  <p className="text-sm text-muted-foreground">{facility.address}</p>
                </div>
              </div>

              <div className="flex items-center gap-4 text-sm">
                {facility.distance_km && (
                  <div className="flex items-center gap-1">
                    <Navigation className="h-4 w-4 text-muted-foreground" />
                    <span className="text-muted-foreground">Distance:</span>
                    <span className="font-medium">{facility.distance_km.toFixed(1)} km</span>
                  </div>
                )}
                {facility.eta_minutes && (
                  <div className="flex items-center gap-1">
                    <Clock className="h-4 w-4 text-muted-foreground" />
                    <span className="text-muted-foreground">ETA:</span>
                    <span className="font-medium">{facility.eta_minutes} min</span>
                  </div>
                )}
              </div>
            </div>

            <div className="rounded-lg border bg-muted/50 p-4 space-y-2">
              <p className="text-sm font-medium">Contact Information</p>
              <div className="space-y-2 text-sm text-muted-foreground">
                <div className="flex items-center gap-2">
                  <Phone className="h-4 w-4" />
                  <span>Contact the facility directly for availability and specific services</span>
                </div>
                <div className="flex items-center gap-2">
                  <Mail className="h-4 w-4" />
                  <span>Use the address above to navigate or call ahead</span>
                </div>
              </div>
            </div>

            <div className="rounded-lg border border-warning/20 bg-warning/5 p-3">
              <p className="text-xs text-muted-foreground">
                <strong>Note:</strong> Contact information is not available from the facility database. 
                Please call the facility directly using publicly available phone numbers or visit their website 
                for the most up-to-date contact information and hours of operation.
              </p>
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}





