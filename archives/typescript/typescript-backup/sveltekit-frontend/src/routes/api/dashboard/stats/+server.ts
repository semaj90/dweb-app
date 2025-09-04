import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { db } from '$lib/server/db/index.js';
import { cases, evidence, criminals, persons_of_interest } from '$lib/server/db/schema-postgres.js';
import { count, eq } from 'drizzle-orm';

export const GET: RequestHandler = async ({ url }) => {
  try {
    // Get live statistics from database
    const [
      activeCasesResult,
      evidenceItemsResult,
      personsOfInterestResult,
      criminalsResult
    ] = await Promise.all([
      // Count active cases
      db.select({ count: count() })
        .from(cases)
        .where(eq(cases.status, 'active')),
      
      // Count evidence items
      db.select({ count: count() })
        .from(evidence),
      
      // Count persons of interest
      db.select({ count: count() })
        .from(persons_of_interest),
      
      // Count criminals
      db.select({ count: count() })
        .from(criminals)
    ]);

    const activeCases = activeCasesResult[0]?.count || 0;
    const evidenceItems = evidenceItemsResult[0]?.count || 0;
    const personsOfInterest = personsOfInterestResult[0]?.count || 0;
    const criminals = criminalsResult[0]?.count || 0;

    // Recent activity count (last 24 hours)
    const recentActivity = activeCases + evidenceItems; // Simplified calculation

    const stats = {
      activeCases,
      evidenceItems,
      personsOfInterest: personsOfInterest + criminals, // Combine POI and criminals
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