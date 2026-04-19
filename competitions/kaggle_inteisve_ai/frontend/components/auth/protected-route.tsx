'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/contexts/auth-context';
import { Loader2 } from 'lucide-react';

interface ProtectedRouteProps {
  children: React.ReactNode;
  requiredRole?: 'patient' | 'hospital_staff';
}

export function ProtectedRoute({ children, requiredRole }: ProtectedRouteProps) {
  const { user, loading, isAuthenticated, isPatient, isHospitalStaff } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!loading) {
      if (!isAuthenticated) {
        router.push('/login');
        return;
      }
      
      if (requiredRole) {
        if (requiredRole === 'patient' && !isPatient) {
          router.push('/hospital');
          return;
        }
        if (requiredRole === 'hospital_staff' && !isHospitalStaff) {
          router.push('/triage');
          return;
        }
      }
    }
  }, [loading, isAuthenticated, isPatient, isHospitalStaff, requiredRole, router]);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    );
  }

  if (!isAuthenticated) {
    return null; // Will redirect
  }

  if (requiredRole) {
    if (requiredRole === 'patient' && !isPatient) {
      return null; // Will redirect
    }
    if (requiredRole === 'hospital_staff' && !isHospitalStaff) {
      return null; // Will redirect
    }
  }

  return <>{children}</>;
}





