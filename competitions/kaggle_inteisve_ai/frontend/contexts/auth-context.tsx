'use client';

import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { useRouter } from 'next/navigation';
import { User, login as apiLogin, register as apiRegister, getCurrentUser, logout as apiLogout, getAuthToken, getStoredUser, setStoredUser } from '@/lib/api';
import { useToast } from '@/hooks/use-toast';

interface AuthContextType {
  user: User | null;
  loading: boolean;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string, name: string, role: 'patient' | 'hospital_staff', facility_name?: string) => Promise<void>;
  logout: () => void;
  isAuthenticated: boolean;
  isPatient: boolean;
  isHospitalStaff: boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const router = useRouter();
  const { toast } = useToast();

  useEffect(() => {
    // Check for stored user and token
    const storedUser = getStoredUser();
    const token = getAuthToken();
    
    if (storedUser && token) {
      // Verify token is still valid
      getCurrentUser()
        .then((user) => {
          setUser(user);
        })
        .catch(() => {
          // Token invalid, clear storage
          apiLogout();
          setUser(null);
        })
        .finally(() => {
          setLoading(false);
        });
    } else {
      setLoading(false);
    }
  }, []);

  const login = async (email: string, password: string) => {
    try {
      const response = await apiLogin({ email, password });
      setUser(response.user);
      toast({
        title: 'Success',
        description: `Welcome back, ${response.user.name}!`,
        variant: 'success',
      });
      
      // Navigate to dashboard after login
      router.push('/');
    } catch (error) {
      toast({
        title: 'Login Failed',
        description: error instanceof Error ? error.message : 'Invalid email or password',
        variant: 'destructive',
      });
      throw error;
    }
  };

  const register = async (email: string, password: string, name: string, role: 'patient' | 'hospital_staff', facility_name?: string) => {
    try {
      const response = await apiRegister({ email, password, name, role, facility_name });
      setUser(response.user);
      toast({
        title: 'Success',
        description: `Account created! Welcome, ${response.user.name}!`,
        variant: 'success',
      });
      
      // Navigate to dashboard after registration
      router.push('/');
    } catch (error) {
      toast({
        title: 'Registration Failed',
        description: error instanceof Error ? error.message : 'Failed to create account',
        variant: 'destructive',
      });
      throw error;
    }
  };

  const logout = () => {
    apiLogout();
    setUser(null);
    router.push('/');
    toast({
      title: 'Logged Out',
      description: 'You have been logged out successfully',
      variant: 'default',
    });
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        loading,
        login,
        register,
        logout,
        isAuthenticated: !!user,
        isPatient: user?.role === 'patient',
        isHospitalStaff: user?.role === 'hospital_staff',
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}


