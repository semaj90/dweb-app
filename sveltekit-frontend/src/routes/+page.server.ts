import type { ServerLoad as PageServerLoad, Actions } from "@sveltejs/kit";
import { redirect, fail } from '@sveltejs/kit';

// Types for our API responses
interface SystemHealth {
  overall: {
    status: 'healthy' | 'degraded' | 'unhealthy';
    healthScore: number;
    healthyServices: number;
    totalServices: number;
    timestamp: string;
  };
  services: {
    databases: Record<string, { host: string; port: number; status: string }>;
    aiServices: Record<string, { host: string; port: number; status: string }>;
    gpuServices: Record<string, { status: string; vram?: string }>;
    orchestration: Record<string, { host: string; port: number; status: string }>;
    storage: Record<string, { host: string; port: number; status: string }>;
  };
  performance: {
    systemUptime: number;
    memoryUsage: {
      heapUsed: number;
      heapTotal: number;
      external: number;
      rss: number;
    };
  };
  architecture: {
    platform: string;
    version: string;
    gpuArchitecture: string;
    microservices: number;
    protocols: string[];
    features: string[];
  };
}

interface SystemInfo {
  platform: string;
  arch: string;
  cpus: number;
  gpuInfo: string;
  memoryUsage: string;
  nodeVersion: string;
  uptime: number;
}

export const load: PageServerLoad = async ({ locals, fetch, setHeaders }) => {
  // Set cache headers for performance
  setHeaders({
    'Cache-Control': 'public, max-age=30', // Cache for 30 seconds
  });

  // Session information for homepage display
  const sessionInfo = {
    userId: locals.session?.user?.id ?? null,
    sessionId: locals.session?.id ?? null,
    email: locals.session?.user?.email ?? null,
    isAuthenticated: !!locals.session?.user
  };

  try {
    // Parallel data fetching for optimal performance
    const [healthResponse, systemInfoResponse] = await Promise.allSettled([
      fetch('/api/health'),
      fetch('/api/system-info'),
    ]);

    // Process health data
    let health: SystemHealth | null = null;
    if (healthResponse.status === 'fulfilled' && healthResponse.value.ok) {
      health = await healthResponse.value.json();
    }

    // Process system info data
    let systemInfo: SystemInfo | null = null;
    if (systemInfoResponse.status === 'fulfilled' && systemInfoResponse.value.ok) {
      systemInfo = await systemInfoResponse.value.json();
    }

    // Dashboard metrics - simulated for demo
    const dashboardStats = {
      activeCases: 42,
      evidenceItems: 1337,
      aiAnalyses: 89,
      systemUptime: health?.performance.systemUptime || 0,
    };

    // Recent activities - YoRHa themed data
    const recentActivities = [
      {
        id: '001',
        type: 'case_created',
        title: 'Corporate Espionage Investigation',
        timestamp: new Date(Date.now() - 1000 * 60 * 15), // 15 minutes ago
        priority: 'high',
      },
      {
        id: '002',
        type: 'evidence_uploaded',
        title: 'Financial Records - Anomaly Detected',
        timestamp: new Date(Date.now() - 1000 * 60 * 45), // 45 minutes ago
        priority: 'medium',
      },
      {
        id: '003',
        type: 'ai_analysis',
        title: 'Pattern Recognition Complete',
        timestamp: new Date(Date.now() - 1000 * 60 * 120), // 2 hours ago
        priority: 'low',
      },
    ];

    return {
      // Session data
      ...sessionInfo,
      
      // API data
      health,
      systemInfo,
      dashboardStats,
      recentActivities,
      
      // Meta information
      loadedAt: new Date().toISOString(),
    };
  } catch (err) {
    console.error('Failed to load dashboard data:', err);
    
    // Return minimal fallback data instead of throwing
    return {
      ...sessionInfo,
      health: null,
      systemInfo: null,
      dashboardStats: {
        activeCases: 0,
        evidenceItems: 0,
        aiAnalyses: 0,
        systemUptime: 0,
      },
      recentActivities: [],
      loadedAt: new Date().toISOString(),
      error: 'Failed to load system data',
    };
  }
};

export const actions: Actions = {
  logout: async ({ cookies }) => {
    // Clear the auth-session cookie
    cookies.delete('auth-session', { path: '/' });

    // Redirect back to homepage after logout
    throw redirect(303, '/');
  },

  // Quick case creation action
  createQuickCase: async ({ request, fetch }) => {
    const data = await request.formData();
    const title = data.get('title')?.toString();
    const priority = data.get('priority')?.toString() || 'medium';

    if (!title) {
      return fail(400, { title, missing: true });
    }

    try {
      // Mock case creation - in real app would call API
      await new Promise(resolve => setTimeout(resolve, 500)); // Simulate API delay
      
      return {
        success: true,
        case: {
          id: Date.now().toString(),
          title,
          priority,
          status: 'open',
          created_at: new Date().toISOString(),
        },
      };
    } catch (err) {
      console.error('Case creation failed:', err);
      return fail(500, { 
        title, 
        error: 'Failed to create case. Please try again.' 
      });
    }
  },

  // System refresh action
  refreshSystem: async ({ fetch }) => {
    try {
      const healthResponse = await fetch('/api/health');
      if (!healthResponse.ok) {
        throw new Error(`Health check failed: ${healthResponse.status}`);
      }

      return {
        success: true,
        refreshedAt: new Date().toISOString(),
      };
    } catch (err) {
      console.error('System refresh failed:', err);
      return fail(500, { 
        error: 'Failed to refresh system status.' 
      });
    }
  },
};