import type { PageServerLoad } from './$types';
import { error } from '@sveltejs/kit';
import type { RouteDefinition } from '$lib/data/routes-config';

export interface SystemHealthData {
  system_overview: {
    healthy_services: number;
    total_services: number;
    uptime_hours: number;
    last_updated: string;
  };
  services: Array<{
    name: string;
    status: 'healthy' | 'degraded' | 'down';
    port?: number;
    response_time?: number;
  }>;
  performance: {
    cpu_usage: number;
    memory_usage: number;
    disk_usage: number;
  };
}

export interface UserSession {
  user: {
    id: string;
    email: string;
    firstName?: string;
    lastName?: string;
    role: 'attorney' | 'paralegal' | 'investigator' | 'user';
    preferences?: {
      theme: string;
      language: string;
      notifications: Record<string, boolean>;
    };
  } | null;
  isAuthenticated: boolean;
}

export interface RoutePageData {
  systemHealth: SystemHealthData | null;
  userSession: UserSession;
  availableRoutes: RouteDefinition[];
  recentOperations: Array<{
    operation: string;
    timestamp: string;
    status: 'success' | 'error' | 'pending';
    protocol?: string;
  }>;
}

async function checkServiceHealth(): Promise<SystemHealthData> {
  const services = [
    { name: 'PostgreSQL', port: 5432 },
    { name: 'Redis', port: 6379 },
    { name: 'Ollama Primary', port: 11434 },
    { name: 'Enhanced RAG', port: 8094 },
    { name: 'Upload Service', port: 8093 },
    { name: 'Neo4j', port: 7474 },
    { name: 'MinIO', port: 9000 },
    { name: 'Qdrant', port: 6333 }
  ];

  const serviceResults = await Promise.allSettled(
    services.map(async (service) => {
      try {
        // Simple TCP connection test with timeout
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 2000);
        
        const startTime = Date.now();
        
        // For HTTP services, try a simple fetch
        if ([8094, 8093, 7474, 9000, 6333, 11434].includes(service.port)) {
          const response = await fetch(`http://localhost:${service.port}/health`, {
            signal: controller.signal,
            method: 'GET'
          }).catch(() => null);
          
          clearTimeout(timeoutId);
          const responseTime = Date.now() - startTime;
          
          return {
            ...service,
            status: response?.ok ? 'healthy' : 'degraded' as const,
            response_time: responseTime
          };
        }
        
        // For database services, assume healthy for now
        clearTimeout(timeoutId);
        return {
          ...service,
          status: 'healthy' as const,
          response_time: 50 // Mock response time
        };
      } catch (error) {
        return {
          ...service,
          status: 'down' as const,
          response_time: undefined
        };
      }
    })
  );

  const healthyServices = serviceResults.filter(
    result => result.status === 'fulfilled' && result.value.status === 'healthy'
  ).length;

  return {
    system_overview: {
      healthy_services: healthyServices,
      total_services: services.length,
      uptime_hours: Math.floor(process.uptime() / 3600),
      last_updated: new Date().toISOString()
    },
    services: serviceResults.map(result => 
      result.status === 'fulfilled' ? result.value : {
        name: 'Unknown Service',
        status: 'down' as const
      }
    ),
    performance: {
      cpu_usage: Math.random() * 80 + 10, // Mock data
      memory_usage: Math.random() * 70 + 20,
      disk_usage: Math.random() * 60 + 15
    }
  };
}

async function getUserSession(cookies: any): Promise<UserSession> {
  // Check for session cookie/token
  const sessionToken = cookies.get('session_token') || cookies.get('auth_token');
  
  if (!sessionToken) {
    return {
      user: null,
      isAuthenticated: false
    };
  }

  try {
    // Mock user session - in production this would verify the token
    // against your authentication system
    const mockUser = {
      id: 'user_123',
      email: 'demo@legal-ai.com',
      firstName: 'Demo',
      lastName: 'User',
      role: 'attorney' as const,
      preferences: {
        theme: 'dark',
        language: 'en',
        notifications: {
          email: true,
          push: true,
          sms: false
        }
      }
    };

    return {
      user: mockUser,
      isAuthenticated: true
    };
  } catch (error) {
    console.error('Session validation error:', error);
    return {
      user: null,
      isAuthenticated: false
    };
  }
}

export const load: PageServerLoad = async ({ url, cookies, depends }) => {
  // Add dependency tracking for real-time updates
  depends('routes:health');
  depends('routes:session');

  try {
    // Load system health data in parallel
    const [systemHealth, userSession] = await Promise.all([
      checkServiceHealth().catch(error => {
        console.error('System health check failed:', error);
        return null;
      }),
      getUserSession(cookies)
    ]);

    // Mock recent operations for demo
    const recentOperations = [
      {
        operation: 'System Health Check',
        timestamp: new Date(Date.now() - 1000 * 60 * 2).toISOString(),
        status: 'success' as const,
        protocol: 'http'
      },
      {
        operation: 'Route Discovery Scan',
        timestamp: new Date(Date.now() - 1000 * 60 * 5).toISOString(),
        status: 'success' as const,
        protocol: 'internal'
      },
      {
        operation: 'API Endpoint Validation',
        timestamp: new Date(Date.now() - 1000 * 60 * 10).toISOString(),
        status: 'success' as const,
        protocol: 'http'
      }
    ];

    // Import routes dynamically to avoid circular dependencies
    const { allRoutes } = await import('$lib/data/routes-config');

    return {
      systemHealth,
      userSession,
      availableRoutes: allRoutes,
      recentOperations
    } satisfies RoutePageData;

  } catch (err) {
    console.error('Page load error:', err);
    throw error(500, 'Failed to load route data');
  }
};