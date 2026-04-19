"use client"

import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Separator } from "@/components/ui/separator"
import { ClipboardList } from "lucide-react"

interface SBARPanelProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  handoffPacket: string
}

export function SBARPanel({ open, onOpenChange, handoffPacket }: SBARPanelProps) {
  const sections = handoffPacket.split("\n").reduce((acc, line) => {
    if (line.startsWith("S (")) {
      acc.situation = line
    } else if (line.startsWith("B (")) {
      acc.background = line
    } else if (line.startsWith("A (")) {
      acc.assessment = line
    } else if (line.startsWith("R (")) {
      acc.recommendation = line
    } else if (line.trim()) {
      acc.notes = acc.notes ? [...acc.notes, line] : [line]
    }
    return acc
  }, {} as { situation?: string; background?: string; assessment?: string; recommendation?: string; notes?: string[] })

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent className="w-full sm:max-w-2xl">
        <SheetHeader>
          <SheetTitle className="flex items-center gap-2">
            <ClipboardList className="h-5 w-5" />
            SBAR Handoff Packet
          </SheetTitle>
          <SheetDescription>
            Clinician-ready handoff information in SBAR format
          </SheetDescription>
        </SheetHeader>
        <ScrollArea className="h-[calc(100vh-100px)] mt-6">
          <div className="space-y-6 pr-4">
            {sections.situation && (
              <div>
                <h3 className="font-semibold text-sm text-muted-foreground mb-2">
                  SITUATION
                </h3>
                <div className="rounded-lg border bg-card p-4 font-mono text-sm">
                  {sections.situation.replace("S (Situation): ", "")}
                </div>
              </div>
            )}
            {sections.background && (
              <div>
                <h3 className="font-semibold text-sm text-muted-foreground mb-2">
                  BACKGROUND
                </h3>
                <div className="rounded-lg border bg-card p-4 font-mono text-sm">
                  {sections.background.replace("B (Background): ", "")}
                </div>
              </div>
            )}
            {sections.assessment && (
              <div>
                <h3 className="font-semibold text-sm text-muted-foreground mb-2">
                  ASSESSMENT
                </h3>
                <div className="rounded-lg border bg-card p-4 font-mono text-sm">
                  {sections.assessment.replace("A (Assessment): ", "")}
                </div>
              </div>
            )}
            {sections.recommendation && (
              <div>
                <h3 className="font-semibold text-sm text-muted-foreground mb-2">
                  RECOMMENDATION
                </h3>
                <div className="rounded-lg border bg-card p-4 font-mono text-sm">
                  {sections.recommendation.replace("R (Recommendation): ", "")}
                </div>
              </div>
            )}
            {sections.notes && sections.notes.length > 0 && (
              <>
                <Separator />
                <div>
                  <h3 className="font-semibold text-sm text-muted-foreground mb-2">
                    ADDITIONAL NOTES
                  </h3>
                  <div className="rounded-lg border bg-card p-4 font-mono text-sm space-y-1">
                    {sections.notes.map((note, i) => (
                      <p key={i}>{note}</p>
                    ))}
                  </div>
                </div>
              </>
            )}
            <Separator />
            <div className="rounded-lg border border-muted bg-muted/50 p-4">
              <p className="text-xs text-muted-foreground">
                This summary is generated for information support only; clinician judgement required.
              </p>
            </div>
          </div>
        </ScrollArea>
      </SheetContent>
    </Sheet>
  )
}





