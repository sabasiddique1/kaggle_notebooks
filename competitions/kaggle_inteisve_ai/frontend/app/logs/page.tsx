"use client"

import { AppShell } from "@/components/layout/app-shell"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Badge } from "@/components/ui/badge"
import { Activity, AlertTriangle, Zap, Clock } from "lucide-react"

// Mock data
const metrics = {
  toolCalls: 1247,
  llmCalls: 89,
  errors: 3,
  avgResponseTime: "1.2s",
}

const logs = [
  {
    timestamp: new Date("2024-01-15T10:30:00Z"),
    traceId: "abc123",
    event: "coordinator_start",
    message: "Intake received for patient",
  },
  {
    timestamp: new Date("2024-01-15T10:30:01Z"),
    traceId: "abc123",
    event: "tool_call",
    message: "Geocoding location: Karachi Pakistan",
  },
  {
    timestamp: new Date("2024-01-15T10:30:02Z"),
    traceId: "abc123",
    event: "tool_call",
    message: "Searching facilities near coordinates",
  },
  {
    timestamp: new Date("2024-01-15T10:30:03Z"),
    traceId: "abc123",
    event: "tool_call",
    message: "Calculating ETA for 5 facilities",
  },
  {
    timestamp: new Date("2024-01-15T10:30:04Z"),
    traceId: "abc123",
    event: "coordinator_done",
    message: "Triage complete, 5 facilities found",
  },
  {
    timestamp: new Date("2024-01-15T10:30:05Z"),
    traceId: "abc123",
    event: "booking_approved",
    message: "Booking confirmed for Aga Khan Hospital",
  },
  {
    timestamp: new Date("2024-01-15T09:15:00Z"),
    traceId: "def456",
    event: "tool_error",
    message: "OSRM API timeout, using distance-only ranking",
  },
]

const getEventBadgeVariant = (event: string) => {
  if (event.includes("error")) return "destructive"
  if (event.includes("tool")) return "default"
  if (event.includes("llm")) return "secondary"
  return "outline"
}

export default function LogsPage() {
  return (
    <AppShell>
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Observability & Logs</h1>
          <p className="text-muted-foreground">
            System metrics and event logs for monitoring and debugging
          </p>
        </div>

        {/* Metrics */}
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Tool Calls</CardTitle>
              <Zap className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{metrics.toolCalls.toLocaleString()}</div>
              <p className="text-xs text-muted-foreground">Total API calls</p>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">LLM Calls</CardTitle>
              <Activity className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{metrics.llmCalls}</div>
              <p className="text-xs text-muted-foreground">Gemini API calls</p>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Errors</CardTitle>
              <AlertTriangle className="h-4 w-4 text-destructive" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-destructive">{metrics.errors}</div>
              <p className="text-xs text-muted-foreground">Total errors</p>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Avg Response Time</CardTitle>
              <Clock className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{metrics.avgResponseTime}</div>
              <p className="text-xs text-muted-foreground">Per request</p>
            </CardContent>
          </Card>
        </div>

        {/* Logs Table */}
        <Card>
          <CardHeader>
            <CardTitle>Event Logs</CardTitle>
            <CardDescription>
              Structured event logs with timestamps and trace IDs
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Timestamp</TableHead>
                  <TableHead>Trace ID</TableHead>
                  <TableHead>Event Type</TableHead>
                  <TableHead>Message</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {logs.map((log, index) => (
                  <TableRow key={index}>
                    <TableCell className="font-mono text-xs">
                      {log.timestamp.toLocaleString()}
                    </TableCell>
                    <TableCell className="font-mono text-xs">{log.traceId}</TableCell>
                    <TableCell>
                      <Badge variant={getEventBadgeVariant(log.event)}>
                        {log.event}
                      </Badge>
                    </TableCell>
                    <TableCell>{log.message}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      </div>
    </AppShell>
  )
}





