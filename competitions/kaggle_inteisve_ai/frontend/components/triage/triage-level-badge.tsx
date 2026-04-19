import { Badge } from "@/components/ui/badge"

interface TriageLevelBadgeProps {
  level: "emergency" | "high" | "medium" | "low"
}

export function TriageLevelBadge({ level }: TriageLevelBadgeProps) {
  const variantMap = {
    emergency: "emergency" as const,
    high: "warning" as const,
    medium: "default" as const,
    low: "success" as const,
  }

  return (
    <Badge variant={variantMap[level]} className="text-sm font-semibold px-3 py-1">
      {level.toUpperCase()}
    </Badge>
  )
}





