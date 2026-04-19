"use client"

import { ScrollArea } from "@/components/ui/scroll-area"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Facility } from "@/lib/api"
import { FacilityCard } from "./facility-card"
import { Map, Building2 } from "lucide-react"

interface FacilitiesListProps {
  facilities: Facility[]
  onSelectFacility?: (facility: Facility) => void
  onViewDetails?: (facility: Facility) => void
}

export function FacilitiesList({
  facilities,
  onSelectFacility,
  onViewDetails,
}: FacilitiesListProps) {
  if (facilities.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Nearby Facilities</CardTitle>
          <CardDescription>No facilities found</CardDescription>
        </CardHeader>
      </Card>
    )
  }

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle>Nearby Facilities</CardTitle>
          <CardDescription>Top {facilities.length} recommended options</CardDescription>
        </CardHeader>
        <CardContent>
          <ScrollArea className="h-[400px]">
            <div className="space-y-3">
              {facilities.map((facility, index) => (
                <FacilityCard
                  key={`${facility.name}-${facility.lat}-${facility.lon}`}
                  facility={facility}
                  index={index}
                  isRecommended={index === 0}
                  onSelect={() => onSelectFacility?.(facility)}
                  onViewDetails={() => onViewDetails?.(facility)}
                />
              ))}
            </div>
          </ScrollArea>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Map className="h-5 w-5" />
            Map View
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-[200px] rounded-lg bg-muted flex items-center justify-center">
            <p className="text-sm text-muted-foreground">Map coming soon</p>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

