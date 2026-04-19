"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Facility } from "@/lib/api"
import { Building2, MapPin, Clock, Navigation, Eye } from "lucide-react"

interface FacilityCardProps {
  facility: Facility
  index: number
  isRecommended?: boolean
  onSelect?: () => void
  onViewDetails?: () => void
}

export function FacilityCard({
  facility,
  index,
  isRecommended = false,
  onSelect,
  onViewDetails,
}: FacilityCardProps) {
  return (
    <Card className="hover:shadow-md transition-shadow">
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <CardTitle className="text-base flex items-center gap-2">
              <Building2 className="h-4 w-4 text-muted-foreground" />
              {index + 1}. {facility.name}
            </CardTitle>
            <CardDescription className="mt-1 flex items-center gap-1">
              <MapPin className="h-3 w-3" />
              <span className="text-xs line-clamp-1">{facility.address}</span>
            </CardDescription>
          </div>
          {isRecommended && (
            <Badge variant="default" className="ml-2">
              Recommended
            </Badge>
          )}
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="flex items-center gap-4 text-sm">
          <div className="flex items-center gap-1">
            <Badge variant="outline" className="text-xs">
              {facility.kind}
            </Badge>
          </div>
          <div className="flex items-center gap-1 text-muted-foreground">
            <Navigation className="h-3 w-3" />
            <span>{facility.distance_km?.toFixed(1) || "N/A"} km</span>
          </div>
          {facility.eta_minutes && (
            <div className="flex items-center gap-1 text-muted-foreground">
              <Clock className="h-3 w-3" />
              <span>{facility.eta_minutes} min</span>
            </div>
          )}
        </div>
        <div className="flex gap-2">
          {onViewDetails && (
            <Button variant="outline" size="sm" className="flex-1" onClick={onViewDetails}>
              <Eye className="h-3 w-3 mr-1" />
              View Details
            </Button>
          )}
          {onSelect && (
            <Button size="sm" className="flex-1" onClick={onSelect}>
              Select for Booking
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  )
}

