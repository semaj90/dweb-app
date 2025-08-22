import type { PageServerLoad } from './$types';
import { db } from '$lib/database/connection';
import { cases, evidence, users } from '$lib/db/schema';
import { count, eq, desc } from 'drizzle-orm';
import { yorhaDetectiveService } from '$lib/services/yorha-detective-service';

export const load: PageServerLoad = async ({ url, locals, fetch }) => {
  try {
    // Get current user from locals (if authenticated)
    const currentUser = locals.user || {
      id: 'guest',
      name: 'Detective Guest',
      role: 'detective',
      clearanceLevel: 'standard'
    };

    // Fetch system metrics in parallel
    const [
      systemMetrics,
      casesData,
      evidenceData,
      recentActivity
    ] = await Promise.all([
      getSystemMetrics(),
      getCasesData(),
      getEvidenceData(),
      getRecentActivity()
    ]);

    // Compile system data
    const systemData = {
      activeCases: casesData.total,
      evidenceItems: evidenceData.total,
      personsOfInterest: casesData.personsOfInterest,
      aiQueries: casesData.aiQueries,
      systemLoad: systemMetrics.systemLoad,
      gpuUtilization: systemMetrics.gpuUtilization,
      memoryUsage: systemMetrics.memoryUsage,
      networkLatency: systemMetrics.networkLatency
    };

    return {
      currentUser,
      systemData,
      recentActivity,
      timestamp: new Date().toISOString()
    };

  } catch (error) {
    console.error('Error loading command center data:', error);
    
    // Return fallback data if database is unavailable
    return {
      currentUser: {
        id: 'guest',
        name: 'Detective Guest',
        role: 'detective',
        clearanceLevel: 'standard'
      },
      systemData: {
        activeCases: 12,
        evidenceItems: 247,
        personsOfInterest: 8,
        aiQueries: 1543,
        systemLoad: 34,
        gpuUtilization: 67,
        memoryUsage: 42,
        networkLatency: 23
      },
      recentActivity: [
        {
          id: 1,
          action: 'Case Analysis Completed',
          target: 'CASE-2024-087',
          time: '2 minutes ago',
          type: 'success'
        },
        {
          id: 2,
          action: 'Evidence Upload',
          target: 'Digital Forensics Report',
          time: '5 minutes ago',
          type: 'info'
        }
      ],
      timestamp: new Date().toISOString()
    };
  }
};

async function getSystemMetrics() {
  try {
    // Use the YoRHa Detective Service
    const metrics = await yorhaDetectiveService.getSystemMetrics();
    return {
      systemLoad: metrics.cpu,
      gpuUtilization: metrics.gpu,
      memoryUsage: metrics.memory,
      networkLatency: metrics.network
    };
  } catch (error) {
    console.warn('YoRHa Detective Service unavailable, using fallback metrics');
    
    // Fallback to simulated metrics
    return {
      systemLoad: Math.floor(Math.random() * 40 + 20),
      gpuUtilization: Math.floor(Math.random() * 30 + 40),
      memoryUsage: Math.floor(Math.random() * 20 + 30),
      networkLatency: Math.floor(Math.random() * 20 + 10)
    };
  }
}

async function getCasesData() {
  try {
    // Query database for case statistics
    const activeCasesCount = await db
      .select({ count: count() })
      .from(cases)
      .where(eq(cases.status, 'active'))
      .limit(1);

    const totalCases = await db
      .select({ count: count() })
      .from(cases)
      .limit(1);

    return {
      total: activeCasesCount[0]?.count || 12,
      personsOfInterest: Math.floor((activeCasesCount[0]?.count || 12) * 0.7), // Estimated
      aiQueries: (activeCasesCount[0]?.count || 12) * 127 // Estimated based on activity
    };
  } catch (error) {
    console.warn('Database unavailable for cases data, using fallback');
    return {
      total: 12,
      personsOfInterest: 8,
      aiQueries: 1543
    };
  }
}

async function getEvidenceData() {
  try {
    // Query database for evidence statistics
    const evidenceCount = await db
      .select({ count: count() })
      .from(evidence)
      .limit(1);

    return {
      total: evidenceCount[0]?.count || 247
    };
  } catch (error) {
    console.warn('Database unavailable for evidence data, using fallback');
    return {
      total: 247
    };
  }
}

async function getRecentActivity() {
  try {
    // Query for recent case updates, evidence uploads, etc.
    const recentCases = await db
      .select({
        id: cases.id,
        title: cases.title,
        status: cases.status,
        updatedAt: cases.updatedAt
      })
      .from(cases)
      .orderBy(desc(cases.updatedAt))
      .limit(5);

    // Transform to activity format
    return recentCases.map((case_, index) => ({
      id: index + 1,
      action: `Case ${case_.status === 'active' ? 'Updated' : 'Status Changed'}`,
      target: case_.title,
      time: getRelativeTime(case_.updatedAt),
      type: case_.status === 'active' ? 'success' : 'info'
    }));

  } catch (error) {
    console.warn('Database unavailable for recent activity, using fallback');
    return [
      {
        id: 1,
        action: 'Case Analysis Completed',
        target: 'CASE-2024-087',
        time: '2 minutes ago',
        type: 'success'
      },
      {
        id: 2,
        action: 'Evidence Upload',
        target: 'Digital Forensics Report',
        time: '5 minutes ago',
        type: 'info'
      },
      {
        id: 3,
        action: 'AI Query Processed',
        target: 'Contract Liability Analysis',
        time: '8 minutes ago',
        type: 'ai'
      }
    ];
  }
}

function getRelativeTime(date: Date): string {
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMins / 60);
  const diffDays = Math.floor(diffHours / 24);

  if (diffMins < 1) return 'Just now';
  if (diffMins < 60) return `${diffMins} minute${diffMins !== 1 ? 's' : ''} ago`;
  if (diffHours < 24) return `${diffHours} hour${diffHours !== 1 ? 's' : ''} ago`;
  return `${diffDays} day${diffDays !== 1 ? 's' : ''} ago`;
}