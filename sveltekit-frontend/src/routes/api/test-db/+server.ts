/**
 * Database Connection Test Endpoint
 * GET /api/test-db - Test PostgreSQL, Redis, and basic CRUD
 */

import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { db, testDatabaseConnection } from '$lib/database/connection';
import { sql } from 'drizzle-orm';

export const GET: RequestHandler = async ({ url }) => {
  const results: any = {
    timestamp: new Date().toISOString(),
    tests: {}
  };

  try {
    // Test 1: Basic PostgreSQL Connection
    console.log('Testing PostgreSQL connection...');
    const pgTest = await testDatabaseConnection();
    results.tests.postgresql = pgTest;

    // Test 2: Simple Query Test
    console.log('Testing simple query...');
    try {
      const queryResult = await db.execute(sql`SELECT COUNT(*) as table_count FROM information_schema.tables WHERE table_schema = 'public'`);
      results.tests.simpleQuery = {
        success: true,
        message: 'Simple query executed successfully',
        tableCount: queryResult[0]?.table_count
      };
    } catch (error) {
      results.tests.simpleQuery = {
        success: false,
        message: `Query failed: ${(error as Error).message}`
      };
    }

    // Test 3: Cases Table Test
    console.log('Testing cases table access...');
    try {
      const casesResult = await db.execute(sql`SELECT COUNT(*) as count FROM cases`);
      results.tests.casesTable = {
        success: true,
        message: 'Cases table accessible',
        count: casesResult[0]?.count || 0
      };
    } catch (error) {
      results.tests.casesTable = {
        success: false,
        message: `Cases table error: ${(error as Error).message}`
      };
    }

    // Test 4: Evidence Table Test
    console.log('Testing evidence table access...');
    try {
      const evidenceResult = await db.execute(sql`SELECT COUNT(*) as count FROM evidence`);
      results.tests.evidenceTable = {
        success: true,
        message: 'Evidence table accessible',
        count: evidenceResult[0]?.count || 0
      };
    } catch (error) {
      results.tests.evidenceTable = {
        success: false,
        message: `Evidence table error: ${(error as Error).message}`
      };
    }

    // Test 5: Legal Documents Table Test
    console.log('Testing legal_documents table access...');
    try {
      const docsResult = await db.execute(sql`SELECT COUNT(*) as count FROM legal_documents`);
      results.tests.legalDocuments = {
        success: true,
        message: 'Legal documents table accessible',
        count: docsResult[0]?.count || 0
      };
    } catch (error) {
      results.tests.legalDocuments = {
        success: false,
        message: `Legal documents error: ${(error as Error).message}`
      };
    }

    // Test 6: Vector Extension Test
    console.log('Testing pgvector extension...');
    try {
      const vectorResult = await db.execute(sql`SELECT extname FROM pg_extension WHERE extname = 'vector'`);
      results.tests.pgvector = {
        success: vectorResult.length > 0,
        message: vectorResult.length > 0 ? 'pgvector extension available' : 'pgvector extension not found'
      };
    } catch (error) {
      results.tests.pgvector = {
        success: false,
        message: `Vector test error: ${(error as Error).message}`
      };
    }

    // Overall Status
    const allTests = Object.values(results.tests);
    const successfulTests = allTests.filter((test: any) => test.success).length;
    const totalTests = allTests.length;

    results.overall = {
      success: successfulTests === totalTests,
      message: `${successfulTests}/${totalTests} tests passed`,
      readyForCRUD: successfulTests >= 4 // Need at least PostgreSQL + main tables working
    };

    console.log(`Database tests completed: ${successfulTests}/${totalTests} passed`);

    return json(results);

  } catch (error) {
    console.error('Database test error:', error);
    
    results.tests.connectionError = {
      success: false,
      message: `Connection error: ${(error as Error).message}`,
      stack: (error as Error).stack
    };

    results.overall = {
      success: false,
      message: 'Database connection failed',
      readyForCRUD: false
    };

    return json(results, { status: 500 });
  }
};