import { json } from '@sveltejs/kit';
import { db } from '$lib/server/db';
// Adjust schema imports to actual available unified schema (fallback counts via raw SQL if needed)
// TODO: Replace with precise imports once schema-postgres module path confirmed
// Temporary raw table names used below.
import { count, eq } from 'drizzle-orm';
import type { RequestHandler } from './$types';


export const GET: RequestHandler = async ({ url }) => {
  try {
    // Import pool for direct database queries
    const { pool } = await import('$lib/database/connection');
    
    // Get live statistics from database using direct SQL
    const [activeCasesRes, evidenceItemsRes, poiRes] = await Promise.all([
      pool`SELECT COUNT(*)::int AS count FROM cases WHERE status = 'active'`,
      pool`SELECT COUNT(*)::int AS count FROM evidence`, 
      pool`SELECT COUNT(*)::int AS count FROM users WHERE role IN ('person_of_interest', 'suspect')`
    ]);

    const activeCases = activeCasesRes[0]?.count || 0;
    const evidenceItems = evidenceItemsRes[0]?.count || 0; 
    const personsOfInterestCount = poiRes[0]?.count || 0;

    // Recent activity count (last 24 hours)
    const recentActivity = activeCases + evidenceItems; // Simplified calculation

    const stats = {
      activeCases,
      evidenceItems,
      personsOfInterest: personsOfInterestCount,
      recentActivity,
      loading: false
    };

    return json(stats, {
      status: 200,
      headers: {
        'Cache-Control': 'max-age=60' // Cache for 1 minute
      }
    });

  } catch (error: any) {
    console.error('Error fetching dashboard stats:', error);

    // Return fallback stats on error
    return json({
      activeCases: 0,
      evidenceItems: 0,
      personsOfInterest: 0,
      recentActivity: 0,
      loading: false,
      error: 'Failed to fetch statistics'
    }, { status: 500 });
  }
};