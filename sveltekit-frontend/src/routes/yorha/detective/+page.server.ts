import { error } from '@sveltejs/kit';
import type { PageServerLoad } from './$types';
// Import canonical DB exports (central index)
import { db, eq, desc, count } from '$lib/server/db';
import { cases, evidence, users } from '$lib/server/db/schema-unified';

export const load: PageServerLoad = async ({ locals }) => {
  try {
    // Mock user for now - replace with real auth when available
    const mockUser = {
      id: 'mock-user-id',
      firstName: 'Detective',
      lastName: 'Smith',
      role: 'detective'
    };

    // Mock data to avoid database calls when DB is not available
    const mockStats = {
      caseStats: [{ count: 12 }],
      evidenceStats: [{ count: 247 }],
      recentCases: [
        {
          id: 'case-1',
          title: 'Corporate Espionage Investigation',
          status: 'active',
          createdAt: new Date('2024-01-15T10:30:00'),
          priority: 'high'
        },
        {
          id: 'case-2',
          title: 'Missing Person: Dr. Sarah Chen',
          status: 'active',
          createdAt: new Date('2024-01-16T14:20:00'),
          priority: 'medium'
        }
      ],
      recentEvidence: [
        {
          id: 'evidence-1',
          title: 'Security Camera Footage',
          evidenceType: 'video',
          createdAt: new Date('2024-01-15T10:30:00'),
          caseTitle: 'Corporate Espionage Investigation'
        },
        {
          id: 'evidence-2',
          title: 'Witness Statement - John Doe',
          evidenceType: 'document',
          createdAt: new Date('2024-01-16T14:20:00'),
          caseTitle: 'Missing Person: Dr. Sarah Chen'
        }
      ]
    };

    // Use mock data for now (database integration can be added later)
    const caseStats = mockStats.caseStats;
    const evidenceStats = mockStats.evidenceStats;
    const recentCases = mockStats.recentCases;
    const recentEvidence = mockStats.recentEvidence;

    // Mock system health data for now (can be replaced with real monitoring)
    const systemData = {
      activeCases: caseStats[0]?.count || 0,
      evidenceItems: evidenceStats[0]?.count || 0,
      personsOfInterest: 42, // Mock data
      aiQueries: 156, // Mock data
      systemLoad: 35,
      gpuUtilization: 67,
      memoryUsage: 42,
      networkLatency: 23
    };

    return {
      systemData,
      recentCases: recentCases || [],
      recentEvidence: recentEvidence || [],
      user: mockUser
    };
  } catch (err) {
    console.error('Error loading detective dashboard:', err);
    throw error(500, 'Failed to load dashboard data');
  }
};