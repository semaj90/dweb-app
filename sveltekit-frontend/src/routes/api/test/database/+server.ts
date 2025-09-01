import type { RequestHandler } from './$types.js';

// Database CRUD Test API
// Tests PostgreSQL, pgvector, and Drizzle ORM integration

import { json } from '@sveltejs/kit';
import { db } from '$lib/server/db/index';
import { users, cases, reports, evidence, criminals, personsOfInterest } from '$lib/server/db/unified-schema';
import { eq, desc, sql } from 'drizzle-orm';
import { URL } from "url";

export interface TestResult {
  test: string;
  status: 'success' | 'error';
  data?: any;
  error?: string;
}

export const GET: RequestHandler = async ({ url }) => {
  const testType = url.searchParams.get('test') || 'all';
  const results: TestResult[] = [];

  try {
    // Test 1: Database Connection
    if (testType === 'all' || testType === 'connection') {
      try {
        const connectionTest = await db.execute(sql`SELECT 'Database Connected' as status, version() as version`);
        results.push({
          test: 'database_connection',
          status: 'success',
          data: connectionTest[0] || { status: 'Connected', count: connectionTest.length }
        });
      } catch (error: any) {
        results.push({
          test: 'database_connection',
          status: 'error',
          error: error instanceof Error ? error.message : String(error)
        });
      }
    }

    // Test 2: Users CRUD
    if (testType === 'all' || testType === 'users') {
      try {
        const usersList = await db.select({
          id: users.id,
          email: users.email,
          firstName: users.firstName,
          lastName: users.lastName,
          role: users.role,
          isActive: users.isActive
        }).from(users).limit(5);

        results.push({
          test: 'users_read',
          status: 'success',
          data: { count: usersList.length, users: usersList }
        });
      } catch (error: any) {
        results.push({
          test: 'users_read',
          status: 'error',
          error: error instanceof Error ? error.message : String(error)
        });
      }
    }

    // Test 3: Cases CRUD
    if (testType === 'all' || testType === 'cases') {
      try {
        const casesList = await db.select({
          id: cases.id,
          title: cases.title,
          status: cases.status,
          caseNumber: cases.caseNumber,
          createdAt: cases.createdAt
        }).from(cases).orderBy(desc(cases.createdAt)).limit(5);

        results.push({
          test: 'cases_read',
          status: 'success',
          data: { count: casesList.length, cases: casesList }
        });
      } catch (error: any) {
        results.push({
          test: 'cases_read',
          status: 'error',
          error: error instanceof Error ? error.message : String(error)
        });
      }
    }

    // Test 4: Reports CRUD
    if (testType === 'all' || testType === 'reports') {
      try {
        const reportsList = await db.select({
          id: reports.id,
          title: reports.title,
          status: reports.status,
          createdAt: reports.createdAt
        }).from(reports).orderBy(desc(reports.createdAt)).limit(5);

        results.push({
          test: 'reports_read',
          status: 'success',
          data: { count: reportsList.length, reports: reportsList }
        });
      } catch (error: any) {
        results.push({
          test: 'reports_read',
          status: 'error',
          error: error instanceof Error ? error.message : String(error)
        });
      }
    }

    // Test 5: Evidence CRUD
    if (testType === 'all' || testType === 'evidence') {
      try {
        const evidenceList = await db.select({
          id: evidence.id,
          title: evidence.title,
          type: evidence.type,
          status: evidence.status,
          createdAt: evidence.createdAt
        }).from(evidence).orderBy(desc(evidence.createdAt)).limit(5);

        results.push({
          test: 'evidence_read',
          status: 'success',
          data: { count: evidenceList.length, evidence: evidenceList }
        });
      } catch (error: any) {
        results.push({
          test: 'evidence_read',
          status: 'error',
          error: error instanceof Error ? error.message : String(error)
        });
      }
    }

    // Test 6: Persons of Interest CRUD
    if (testType === 'all' || testType === 'poi') {
      try {
        const poiList = await db.select({
          id: personsOfInterest.id,
          name: personsOfInterest.name,
          threatLevel: personsOfInterest.threatLevel,
          status: personsOfInterest.status,
          createdAt: personsOfInterest.createdAt
        }).from(personsOfInterest).orderBy(desc(personsOfInterest.createdAt)).limit(5);

        results.push({
          test: 'persons_of_interest_read',
          status: 'success',
          data: { count: poiList.length, personsOfInterest: poiList }
        });
      } catch (error: any) {
        results.push({
          test: 'persons_of_interest_read',
          status: 'error',
          error: error instanceof Error ? error.message : String(error)
        });
      }
    }

    // Test 7: Criminals CRUD
    if (testType === 'all' || testType === 'criminals') {
      try {
        const criminalsList = await db.select({
          id: criminals.id,
          firstName: criminals.firstName,
          lastName: criminals.lastName,
          threatLevel: criminals.threatLevel,
          status: criminals.status,
          createdAt: criminals.createdAt
        }).from(criminals).orderBy(desc(criminals.createdAt)).limit(5);

        results.push({
          test: 'criminals_read',
          status: 'success',
          data: { count: criminalsList.length, criminals: criminalsList }
        });
      } catch (error: any) {
        results.push({
          test: 'criminals_read',
          status: 'error',
          error: error instanceof Error ? error.message : String(error)
        });
      }
    }

    // Test 8: Vector Extension Test
    if (testType === 'all' || testType === 'vector') {
      try {
        const vectorTest = await db.execute(sql`SELECT 'pgvector extension available' as status WHERE EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')`);
        results.push({
          test: 'pgvector_extension',
          status: 'success',
          data: vectorTest.rows.length > 0 ? { available: true } : { available: false }
        });
      } catch (error: any) {
        results.push({
          test: 'pgvector_extension',
          status: 'error',
          error: error instanceof Error ? error.message : String(error)
        });
      }
    }

    return json({
      success: true,
      timestamp: new Date().toISOString(),
      tests: results,
      summary: {
        total: results.length,
        passed: results.filter(r => r.status === 'success').length,
        failed: results.filter(r => r.status === 'error').length
      }
    });

  } catch (error: any) {
    return json({
      success: false,
      error: error instanceof Error ? error.message : String(error),
      timestamp: new Date().toISOString()
    }, { status: 500 });
  }
};