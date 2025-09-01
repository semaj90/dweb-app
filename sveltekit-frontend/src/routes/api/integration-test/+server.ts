import { json } from '@sveltejs/kit';
import { db, testConnection, healthCheck } from '$lib/server/db';
import { users, cases, evidence, documentChunks } from '$lib/server/db/schema-postgres';
import { eq, sql } from 'drizzle-orm';
import type { RequestHandler } from './$types.js';

// import { mcpTools } from '../../../mcp/index.js'; // Temporarily disabled due to dependency issues
import bcrypt from 'bcrypt';
import crypto from "crypto";
import { URL } from "url";

export const GET: RequestHandler = async ({ url }) => {
  const testType = url.searchParams.get('type') || 'all';
  const results: Record<string, any> = {};

  try {
    // 1. Test Database Connection
    if (testType === 'all' || testType === 'connection') {
      results.connection = await testConnection();
      results.health = await healthCheck();
    }

    // 2. Test pgvector Extension
    if (testType === 'all' || testType === 'vector') {
      try {
        const vectorTest = await db.execute(sql`
          SELECT 
            extname as extension_name,
            extversion as version
          FROM pg_extension 
          WHERE extname = 'vector'
        `);
        
        results.vector = {
          installed: vectorTest.length > 0,
          details: vectorTest[0] || null
        };
      } catch (error: any) {
        results.vector = {
          installed: false,
          error: error.message
        };
      }
    }

    // 3. Test Table Creation and Schema
    if (testType === 'all' || testType === 'schema') {
      try {
        const tables = await db.execute(sql`
          SELECT table_name 
          FROM information_schema.tables 
          WHERE table_schema = 'public' 
          AND table_name IN ('users', 'cases', 'evidence', 'document_chunks')
        `);
        
        results.schema = {
          tables: tables,
          tablesCount: tables.length,
          expectedTables: ['users', 'cases', 'evidence', 'document_chunks']
        };
      } catch (error: any) {
        results.schema = {
          error: error.message
        };
      }
    }

    // 4. Test CRUD Operations
    if (testType === 'all' || testType === 'crud') {
      const testUser = {
        email: 'test-integration@legal.ai',
        hashed_password: await bcrypt.hash('test123', 10),
        first_name: 'Integration',
        last_name: 'Test',
        role: 'attorney'
      };

      try {
        // CREATE - Insert test user
        const newUser = await db.insert(users)
          .values(testUser)
          .returning()
          .catch(async (error) => {
            if (error.code === '23505') { // Unique constraint violation
              // User already exists, get it
              return await db.select()
                .from(users)
                .where(eq(users.email, testUser.email))
                .limit(1);
            }
            throw error;
          });

        const userId = Array.isArray(newUser) ? newUser[0]?.id : newUser.id;

        // CREATE - Insert test case
        const testCase = {
          title: 'Integration Test Case',
          description: 'Test case for database integration',
          assigned_attorney: userId,
          status: 'open'
        };

        const newCase = await db.insert(cases)
          .values(testCase)
          .returning();

        const caseId = newCase[0]?.id;

        // CREATE - Insert test evidence
        const testEvidence = {
          case_id: caseId,
          title: 'Integration Test Evidence',
          description: 'Test evidence for database integration',
          evidence_type: 'document'
        };

        const newEvidence = await db.insert(evidence)
          .values(testEvidence)
          .returning();

        // READ - Query with joins
        const caseWithDetails = await db.select({
          case: cases,
          attorney: users,
          evidence: evidence
        })
          .from(cases)
          .leftJoin(users, eq(cases.assigned_attorney, users.id))
          .leftJoin(evidence, eq(evidence.case_id, cases.id))
          .where(eq(cases.id, caseId))
          .limit(1);

        // UPDATE - Modify case
        const updatedCase = await db.update(cases)
          .set({ 
            description: 'Updated test case description',
            status: 'in_progress'
          })
          .where(eq(cases.id, caseId))
          .returning();

        results.crud = {
          success: true,
          operations: {
            create: {
              user: newUser[0] || newUser,
              case: newCase[0],
              evidence: newEvidence[0]
            },
            read: {
              caseWithDetails: caseWithDetails[0]
            },
            update: {
              updatedCase: updatedCase[0]
            }
          }
        };

        // Cleanup - DELETE test data
        await db.delete(evidence).where(eq(evidence.case_id, caseId));
        await db.delete(cases).where(eq(cases.id, caseId));
        await db.delete(users).where(eq(users.id, userId));

      } catch (error: any) {
        results.crud = {
          success: false,
          error: error.message,
          stack: error.stack
        };
      }
    }

    // 5. Test Vector Operations
    if (testType === 'all' || testType === 'vector-ops') {
      try {
        // Create a test embedding vector (384 dimensions)
        const testEmbedding = new Array(384).fill(0).map(() => Math.random() * 2 - 1);
        
        const testChunk = {
          document_id: crypto.randomUUID(),
          document_type: 'test',
          chunk_index: '0',
          content: 'This is a test document chunk for vector operations',
          embedding: JSON.stringify(testEmbedding)
        };

        // Insert test chunk with embedding
        const newChunk = await db.insert(documentChunks)
          .values(testChunk)
          .returning();

        // Test vector similarity search
        const similarChunks = await db.select({
          id: documentChunks.id,
          content: documentChunks.content,
          similarity: sql<number>`1 - (${documentChunks.embedding} <=> ${testEmbedding}::vector) as similarity`
        })
          .from(documentChunks)
          .where(sql`${documentChunks.embedding} IS NOT NULL`)
          .orderBy(sql`${documentChunks.embedding} <=> ${testEmbedding}::vector`)
          .limit(5);

        // Cleanup
        await db.delete(documentChunks).where(eq(documentChunks.id, newChunk[0].id));

        results.vectorOps = {
          success: true,
          operations: {
            create: newChunk[0],
            search: similarChunks
          }
        };

      } catch (error: any) {
        results.vectorOps = {
          success: false,
          error: error.message,
          stack: error.stack
        };
      }
    }

    // 6. Test MCP Tools Integration (temporarily disabled)
    if (testType === 'mcp') {
      results.mcp = {
        success: false,
        error: 'MCP tools temporarily disabled due to dependency issues - use direct database operations instead'
      };
    }

    return json({
      success: true,
      timestamp: new Date().toISOString(),
      testType,
      results
    });

  } catch (error: any) {
    return json({
      success: false,
      error: error.message,
      stack: error.stack,
      timestamp: new Date().toISOString()
    }, { status: 500 });
  }
};

export const POST: RequestHandler = async ({ request }) => {
  try {
    const { action, data } = await request.json();

    switch (action) {
      case 'create-test-data':
        // Create comprehensive test data set
        const testUser = await db.insert(users).values({
          email: `test-${Date.now()}@legal.ai`,
          hashed_password: await bcrypt.hash('test123', 10),
          first_name: 'Test',
          last_name: 'User',
          role: 'attorney',
          department: 'Integration Testing'
        }).returning();

        const testCase = await db.insert(cases).values({
          title: `Test Case ${Date.now()}`,
          description: 'Comprehensive integration test case',
          assigned_attorney: testUser[0].id,
          status: 'open'
        }).returning();

        return json({
          success: true,
          data: {
            user: testUser[0],
            case: testCase[0]
          }
        });

      case 'cleanup-test-data':
        // Cleanup all test data
        const deletedEvidence = await db.delete(evidence)
          .where(sql`${evidence.title} LIKE '%Test%' OR ${evidence.title} LIKE '%Integration%'`);
        
        const deletedCases = await db.delete(cases)
          .where(sql`${cases.title} LIKE '%Test%' OR ${cases.title} LIKE '%Integration%'`);
        
        const deletedUsers = await db.delete(users)
          .where(sql`${users.email} LIKE '%test%@legal.ai'`);

        return json({
          success: true,
          cleanup: {
            evidence: deletedEvidence,
            cases: deletedCases,
            users: deletedUsers
          }
        });

      default:
        return json({ error: 'Unknown action' }, { status: 400 });
    }

  } catch (error: any) {
    return json({
      success: false,
      error: error.message,
      stack: error.stack
    }, { status: 500 });
  }
};