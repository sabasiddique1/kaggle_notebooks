'use client';

import { useState, useEffect } from 'react';
import { AppShell } from '@/components/layout/app-shell';
import { ProtectedRoute } from '@/components/auth/protected-route';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { ScrollArea } from '@/components/ui/scroll-area';
import { 
  getHospitalFacilities, 
  getHospitalBookings, 
  approveBooking,
  ackAlert,
  approveAppointment,
  rejectBooking,
  type HospitalBooking,
  type HospitalFacility 
} from '@/lib/api';
import { useToast } from '@/hooks/use-toast';
import { Building2, Clock, User, AlertCircle, CheckCircle2, XCircle, RefreshCw, AlertTriangle, Calendar } from 'lucide-react';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogFooter } from '@/components/ui/dialog';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';

export default function HospitalPanel() {
  const [facilities, setFacilities] = useState<HospitalFacility[]>([]);
  const [selectedFacility, setSelectedFacility] = useState<string | null>(null);
  const [bookings, setBookings] = useState<HospitalBooking[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [rejectDialogOpen, setRejectDialogOpen] = useState(false);
  const [selectedBooking, setSelectedBooking] = useState<HospitalBooking | null>(null);
  const [rejectReason, setRejectReason] = useState('');
  const { toast } = useToast();

  useEffect(() => {
    loadFacilities();
  }, []);

  useEffect(() => {
    if (selectedFacility) {
      loadBookings(selectedFacility);
    }
    
    // Auto-refresh every 30 seconds for real-time updates
    const interval = setInterval(() => {
      loadFacilities();
      if (selectedFacility) {
        loadBookings(selectedFacility);
      }
    }, 30000); // 30 seconds
    
    return () => clearInterval(interval);
  }, [selectedFacility]);

  const loadFacilities = async () => {
    try {
      setLoading(true);
      const data = await getHospitalFacilities();
      setFacilities(data);
      if (data.length > 0 && !selectedFacility) {
        setSelectedFacility(data[0].name);
      }
    } catch (error) {
      toast({
        title: 'Error',
        description: 'Failed to load facilities',
        variant: 'destructive',
      });
    } finally {
      setLoading(false);
    }
  };

  const loadBookings = async (facilityName: string) => {
    try {
      setRefreshing(true);
      const data = await getHospitalBookings(facilityName);
      setBookings(data);
    } catch (error) {
      toast({
        title: 'Error',
        description: 'Failed to load bookings',
        variant: 'destructive',
      });
    } finally {
      setRefreshing(false);
    }
  };

  const handleApprove = async (booking: HospitalBooking) => {
    try {
      // Use appropriate endpoint based on request type
      if (booking.request_type === 'alert') {
        await ackAlert(booking.session_id);
        toast({
          title: 'Success',
          description: 'Pre-arrival alert acknowledged successfully',
          variant: 'success',
        });
      } else if (booking.request_type === 'appointment') {
        await approveAppointment(booking.session_id);
        toast({
          title: 'Success',
          description: 'Appointment booking approved successfully',
          variant: 'success',
        });
      } else {
        // Fallback to legacy endpoint
        await approveBooking(booking.session_id);
        toast({
          title: 'Success',
          description: 'Request approved successfully',
          variant: 'success',
        });
      }
      if (selectedFacility) {
        loadBookings(selectedFacility);
        loadFacilities();
      }
    } catch (error) {
      toast({
        title: 'Error',
        description: error instanceof Error ? error.message : 'Failed to approve request',
        variant: 'destructive',
      });
    }
  };

  const handleRejectClick = (booking: HospitalBooking) => {
    setSelectedBooking(booking);
    setRejectReason('');
    setRejectDialogOpen(true);
  };

  const handleRejectConfirm = async () => {
    if (!selectedBooking) return;
    
    try {
      await rejectBooking(selectedBooking.session_id, rejectReason || undefined);
      toast({
        title: 'Success',
        description: 'Booking rejected',
        variant: 'success',
      });
      setRejectDialogOpen(false);
      setSelectedBooking(null);
      setRejectReason('');
      if (selectedFacility) {
        loadBookings(selectedFacility);
        loadFacilities();
      }
    } catch (error) {
      toast({
        title: 'Error',
        description: 'Failed to reject booking',
        variant: 'destructive',
      });
    }
  };

  const getTriageBadgeVariant = (level: string) => {
    switch (level) {
      case 'emergency':
        return 'destructive';
      case 'high':
        return 'destructive';
      case 'medium':
        return 'default';
      case 'low':
        return 'secondary';
      default:
        return 'secondary';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'pending_approval':
        return <Clock className="h-4 w-4 text-yellow-500" />;
      case 'confirmed':
        return <CheckCircle2 className="h-4 w-4 text-green-500" />;
      case 'rejected':
        return <XCircle className="h-4 w-4 text-red-500" />;
      default:
        return <AlertCircle className="h-4 w-4" />;
    }
  };

  const pendingBookings = bookings.filter(b => 
    b.status === 'PENDING_ACK' || 
    b.status === 'PENDING_APPROVAL' || 
    b.status === 'pending_approval'
  );
  const otherBookings = bookings.filter(b => 
    b.status !== 'PENDING_ACK' && 
    b.status !== 'PENDING_APPROVAL' && 
    b.status !== 'pending_approval'
  );

  return (
    <ProtectedRoute requiredRole="hospital_staff">
      <AppShell>
        <div className="container mx-auto p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Hospital Panel</h1>
          <p className="text-muted-foreground mt-1">Manage patient booking requests</p>
        </div>
        <Button
          onClick={() => {
            if (selectedFacility) loadBookings(selectedFacility);
            loadFacilities();
          }}
          variant="outline"
          disabled={refreshing}
        >
          <RefreshCw className={`h-4 w-4 mr-2 ${refreshing ? 'animate-spin' : ''}`} />
          Refresh
        </Button>
      </div>

      {/* Facility Selection */}
      <Card>
        <CardHeader>
          <CardTitle>Select Hospital</CardTitle>
          <CardDescription>Choose a facility to view and manage bookings</CardDescription>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="text-center py-4">Loading facilities...</div>
          ) : facilities.length === 0 ? (
            <div className="text-center py-12 text-muted-foreground">
              <Building2 className="h-16 w-16 mx-auto mb-4 opacity-50" />
              <p className="text-lg font-medium mb-2">No facilities with bookings found</p>
              <p className="text-sm max-w-md mx-auto">
                Booking requests will appear here when patients submit triage forms with emergency or high-priority conditions. 
                Have a patient submit a triage form with emergency/high priority symptoms to see bookings.
              </p>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {facilities.map((facility) => (
                <Card
                  key={facility.name}
                  className={`cursor-pointer transition-all ${
                    selectedFacility === facility.name
                      ? 'ring-2 ring-primary'
                      : 'hover:shadow-md'
                  }`}
                  onClick={() => setSelectedFacility(facility.name)}
                >
                  <CardHeader className="pb-3">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Building2 className="h-5 w-5 text-primary" />
                        <CardTitle className="text-lg">{facility.name}</CardTitle>
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-muted-foreground">Pending</span>
                        <Badge variant={facility.pending_bookings > 0 ? 'destructive' : 'secondary'}>
                          {facility.pending_bookings}
                        </Badge>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-muted-foreground">Total</span>
                        <span className="text-sm font-medium">{facility.total_bookings}</span>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Bookings List */}
      {selectedFacility && (
        <div className="space-y-6">
          {/* Pending Bookings */}
          {pendingBookings.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <AlertCircle className="h-5 w-5 text-yellow-500" />
                  Pending Requests ({pendingBookings.length})
                </CardTitle>
                <CardDescription>
                  {pendingBookings.filter(b => b.request_type === 'alert').length > 0 && (
                    <span className="inline-flex items-center gap-1">
                      <AlertTriangle className="h-4 w-4 text-red-500" /> {pendingBookings.filter(b => b.request_type === 'alert').length} Alert(s) | 
                    </span>
                  )}
                  {pendingBookings.filter(b => b.request_type === 'appointment').length > 0 && (
                    <span className="inline-flex items-center gap-1">
                      <Calendar className="h-4 w-4 text-blue-500" /> {pendingBookings.filter(b => b.request_type === 'appointment').length} Appointment(s)
                    </span>
                  )}
                  {pendingBookings.length === 0 && 'No pending requests'}
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ScrollArea className="h-[600px]">
                  <div className="space-y-4">
                    {pendingBookings.map((booking) => (
                      <Card key={booking.id} className={`border-l-4 ${
                        booking.request_type === 'alert' ? 'border-l-red-500' : 'border-l-yellow-500'
                      }`}>
                        <CardHeader>
                          <div className="flex items-start justify-between">
                            <div className="space-y-1">
                              <div className="flex items-center gap-2">
                                <User className="h-4 w-4 text-muted-foreground" />
                                <CardTitle className="text-lg">{booking.patient_name}</CardTitle>
                              </div>
                              <div className="flex items-center gap-2 mt-2 flex-wrap">
                                <Badge variant={getTriageBadgeVariant(booking.triage_level)} className="whitespace-nowrap">
                                  {booking.triage_level.toUpperCase()}
                                </Badge>
                                {booking.request_type && (
                                  <Badge variant="outline" className="whitespace-nowrap flex items-center gap-1">
                                    {booking.request_type === 'alert' ? (
                                      <>
                                        <AlertTriangle className="h-3 w-3" /> Alert
                                      </>
                                    ) : (
                                      <>
                                        <Calendar className="h-3 w-3" /> Appointment
                                      </>
                                    )}
                                  </Badge>
                                )}
                                <span className="whitespace-nowrap">{getStatusIcon(booking.status)}</span>
                                <span className="text-sm text-muted-foreground whitespace-nowrap">
                                  {new Date(booking.requested_at).toLocaleString()}
                                </span>
                              </div>
                            </div>
                          </div>
                        </CardHeader>
                        <CardContent className="space-y-4">
                          {booking.symptoms && booking.symptoms.length > 0 && (
                            <div>
                              <Label className="text-sm font-medium">Symptoms</Label>
                              <div className="flex flex-wrap gap-2 mt-1">
                                {booking.symptoms.map((symptom, idx) => (
                                  <Badge key={idx} variant="outline">{symptom}</Badge>
                                ))}
                              </div>
                            </div>
                          )}
                          {booking.location && (
                            <div>
                              <Label className="text-sm font-medium">Location</Label>
                              <p className="text-sm text-muted-foreground mt-1">{booking.location}</p>
                            </div>
                          )}
                          {booking.eta_minutes && (
                            <div>
                              <Label className="text-sm font-medium">Estimated Arrival</Label>
                              <p className="text-sm text-muted-foreground mt-1">{booking.eta_minutes} minutes</p>
                            </div>
                          )}
                          {booking.handoff_packet && (
                            <div>
                              <Label className="text-sm font-medium">SBAR Handoff Packet</Label>
                              <div className="mt-2 p-3 bg-muted rounded-md">
                                <pre className="text-xs whitespace-pre-wrap font-mono">
                                  {booking.handoff_packet}
                                </pre>
                              </div>
                            </div>
                          )}
                          <Separator />
                          <div className="flex gap-2">
                            <Button
                              onClick={() => handleApprove(booking)}
                              className="flex-1"
                              variant="default"
                            >
                              <CheckCircle2 className="h-4 w-4 mr-2" />
                              Approve
                            </Button>
                            <Button
                              onClick={() => handleRejectClick(booking)}
                              className="flex-1"
                              variant="destructive"
                            >
                              <XCircle className="h-4 w-4 mr-2" />
                              Reject
                            </Button>
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                </ScrollArea>
              </CardContent>
            </Card>
          )}

          {/* Other Bookings */}
          {otherBookings.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle>All Bookings</CardTitle>
                <CardDescription>Complete booking history</CardDescription>
              </CardHeader>
              <CardContent>
                <ScrollArea className="h-[400px]">
                  <div className="space-y-4">
                    {otherBookings.map((booking) => (
                      <Card key={booking.id} className="border-l-4 border-l-gray-300">
                        <CardHeader>
                          <div className="flex items-start justify-between">
                            <div className="space-y-1">
                              <div className="flex items-center gap-2">
                                <User className="h-4 w-4 text-muted-foreground" />
                                <CardTitle className="text-lg">{booking.patient_name}</CardTitle>
                              </div>
                              <div className="flex items-center gap-2 mt-2 flex-wrap">
                                <Badge variant={getTriageBadgeVariant(booking.triage_level)} className="whitespace-nowrap">
                                  {booking.triage_level.toUpperCase()}
                                </Badge>
                                {booking.request_type && (
                                  <Badge variant="outline" className="whitespace-nowrap flex items-center gap-1">
                                    {booking.request_type === 'alert' ? (
                                      <>
                                        <AlertTriangle className="h-3 w-3" /> Alert
                                      </>
                                    ) : (
                                      <>
                                        <Calendar className="h-3 w-3" /> Appointment
                                      </>
                                    )}
                                  </Badge>
                                )}
                                <span className="whitespace-nowrap">{getStatusIcon(booking.status)}</span>
                                <span className="text-sm text-muted-foreground whitespace-nowrap">
                                  {new Date(booking.requested_at).toLocaleString()}
                                </span>
                              </div>
                            </div>
                          </div>
                        </CardHeader>
                        <CardContent>
                          {booking.approved_at && (
                            <p className="text-sm text-muted-foreground">
                              Approved: {new Date(booking.approved_at).toLocaleString()}
                            </p>
                          )}
                          {booking.rejected_at && (
                            <div>
                              <p className="text-sm text-muted-foreground">
                                Rejected: {new Date(booking.rejected_at).toLocaleString()}
                              </p>
                              {booking.rejection_reason && (
                                <p className="text-sm text-red-500 mt-1">
                                  Reason: {booking.rejection_reason}
                                </p>
                              )}
                            </div>
                          )}
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                </ScrollArea>
              </CardContent>
            </Card>
          )}

          {bookings.length === 0 && !refreshing && selectedFacility && (
            <Card>
              <CardContent className="py-12 text-center text-muted-foreground">
                <Building2 className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p className="text-lg font-medium mb-2">No bookings found for {selectedFacility}</p>
                <p className="text-sm max-w-md mx-auto">
                  Booking requests will appear here when patients submit triage forms with emergency or high-priority conditions 
                  and select this facility. All bookings are automatically routed to the appropriate hospital.
                </p>
              </CardContent>
            </Card>
          )}
        </div>
      )}

      {/* Reject Dialog */}
      <Dialog open={rejectDialogOpen} onOpenChange={setRejectDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Reject Booking</DialogTitle>
            <DialogDescription>
              Please provide a reason for rejecting this booking request.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label htmlFor="reason">Rejection Reason</Label>
              <Textarea
                id="reason"
                placeholder="Enter reason for rejection (optional)"
                value={rejectReason}
                onChange={(e) => setRejectReason(e.target.value)}
                rows={4}
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setRejectDialogOpen(false)}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={handleRejectConfirm}>
              Reject Booking
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
        </div>
      </AppShell>
    </ProtectedRoute>
  );
}

