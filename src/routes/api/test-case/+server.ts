import { json, error } from '@sveltejs/kit';
import type { RequestHandler } from './$types';
import { db } from '$lib/database/client';
import { 
  legalCases, 
  caseDocuments, 
  legalDocuments, 
  agentAnalysisCache
} from '$lib/database/schema/legal-documents';
import { eq, desc, and, or, like, sql } from 'drizzle-orm';
import { z } from 'zod';

// For session management - check if user table exists
import { getUserFromSession } from '$lib/server/auth';

// Validation schema for test case creation
const createTestCaseSchema = z.object({
  caseNumber: z.string().min(1, 'Case number is required'),
  title: z.string().min(1, 'Title is required'),
  description: z.string().optional(),
  clientName: z.string().min(1, 'Client name is required'),
  opposingParty: z.string().optional(),
  jurisdiction: z.string().default('federal'),
  courtName: z.string().optional(),
  judgeAssigned: z.string().optional(),
  caseType: z.enum(['civil', 'criminal', 'administrative', 'appellate', 'arbitration']).default('civil'),
  practiceArea: z.string().default('general'),
  priority: z.enum(['low', 'medium', 'high', 'critical']).default('medium'),
  status: z.enum(['active', 'pending', 'closed', 'archived', 'on_hold']).default('active'),
  filingDate: z.string().optional(),
  trialDate: z.string().optional(),
  estimatedValue: z.number().optional(),
  billingRate: z.number().optional(),
  caseSummary: z.string().optional(),
  legalStrategy: z.string().optional(),
  keyIssues: z.array(z.string()).optional(),
  precedents: z.array(z.object({
    caseNumber: z.string(),
    citation: z.string(),
    relevance: z.enum(['high', 'medium', 'low']),
    summary: z.string()
  })).optional()
});

// Helper to generate embeddings for pgvector
async function generateEmbedding(text: string): Promise<number[]> {
  try {
    // Call your embedding service (Ollama, OpenAI, etc.)
    const response = await fetch('http://localhost:11434/api/embeddings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'nomic-embed-text',
        prompt: text
      })
    });

    if (!response.ok) {
      console.error('Embedding generation failed');
      return null;
    }

    const data = await response.json();
    return data.embedding;
  } catch (error) {
    console.error('Error generating embedding:', error);
    return null;
  }
}

// GET: Retrieve test cases with PostgreSQL/pgvector
export const GET: RequestHandler = async ({ url, locals }) => {
  try {
    // Check authentication if available
    const user = locals.user || { id: 'system', role: 'admin' };

    const caseId = url.searchParams.get('id');
    const status = url.searchParams.get('status');
    const priority = url.searchParams.get('priority');
    const practiceArea = url.searchParams.get('practiceArea');
    const search = url.searchParams.get('search');
    const vectorSearch = url.searchParams.get('vectorSearch');
    const limit = parseInt(url.searchParams.get('limit') || '12');
    const offset = parseInt(url.searchParams.get('offset') || '0');

    if (caseId) {
      // Get specific case with related documents from PostgreSQL
      const caseData = await db
        .select()
        .from(legalCases)
        .where(eq(legalCases.id, caseId))
        .limit(1);
      
      if (caseData.length === 0) {
        return json({ error: 'Case not found' }, { status: 404 });
      }

      // Get related documents
      const documents = await db
        .select({
          id: caseDocuments.id,
          documentId: caseDocuments.documentId,
          relationship: caseDocuments.relationship,
          importance: caseDocuments.importance,
          notes: caseDocuments.notes,
          addedAt: caseDocuments.addedAt,
          document: legalDocuments
        })
        .from(caseDocuments)
        .leftJoin(legalDocuments, eq(caseDocuments.documentId, legalDocuments.id))
        .where(eq(caseDocuments.caseId, caseId));

      // Get AI analysis cache
      const aiAnalysis = await db
        .select()
        .from(agentAnalysisCache)
        .where(eq(agentAnalysisCache.caseId, caseId))
        .orderBy(desc(agentAnalysisCache.createdAt))
        .limit(5);

      // If vector search requested, find similar cases using pgvector
      let similarCases = [];
      if (vectorSearch && caseData[0].caseSummary) {
        const embedding = await generateEmbedding(vectorSearch);
        if (embedding) {
          // Use pgvector's <-> operator for cosine distance
          similarCases = await db.execute(sql`
            SELECT 
              id,
              case_number,
              title,
              priority,
              status,
              1 - (case_summary_embedding <-> ${embedding}::vector) as similarity
            FROM legal_cases
            WHERE id != ${caseId}
              AND case_summary_embedding IS NOT NULL
            ORDER BY case_summary_embedding <-> ${embedding}::vector
            LIMIT 5
          `);
        }
      }

      return json({
        success: true,
        data: {
          ...caseData[0],
          documents: documents.map(d => ({
            ...d,
            document: d.document
          })),
          aiAnalysis,
          similarCases
        }
      });
    }

    // Build query with filters for PostgreSQL
    let conditions = [];
    
    if (status) {
      conditions.push(eq(legalCases.status, status as any));
    }
    
    if (priority) {
      conditions.push(eq(legalCases.priority, priority as any));
    }
    
    if (practiceArea) {
      conditions.push(eq(legalCases.practiceArea, practiceArea));
    }
    
    if (search) {
      conditions.push(
        or(
          like(legalCases.caseNumber, `%${search}%`),
          like(legalCases.title, `%${search}%`),
          like(legalCases.clientName, `%${search}%`),
          like(legalCases.opposingParty, `%${search}%`)
        )
      );
    }

    // Execute query with Drizzle ORM
    const query = db
      .select()
      .from(legalCases)
      .orderBy(desc(legalCases.createdAt))
      .limit(limit)
      .offset(offset);

    if (conditions.length > 0) {
      query.where(and(...conditions));
    }

    const result = await query;

    // Get total count
    const countQuery = db
      .select({ count: sql<number>`count(*)` })
      .from(legalCases);

    if (conditions.length > 0) {
      countQuery.where(and(...conditions));
    }

    const totalCount = await countQuery;

    return json({
      success: true,
      data: result,
      pagination: {
        total: Number(totalCount[0]?.count || 0),
        limit,
        offset,
        hasMore: offset + limit < Number(totalCount[0]?.count || 0)
      }
    });
  } catch (error) {
    console.error('Error fetching test cases:', error);
    return json({ 
      error: 'Failed to fetch test cases',
      details: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 });
  }
};

// POST: Create a new test case in PostgreSQL with pgvector embeddings
export const POST: RequestHandler = async ({ request, locals }) => {
  try {
    // Get authenticated user or use system user
    const user = locals.user || { id: 'system', email: 'system@legal-ai.com' };

    // Parse and validate request body
    const body = await request.json();
    const validatedData = createTestCaseSchema.parse(body);

    // Generate unique case ID
    const caseId = crypto.randomUUID();
    const now = new Date();

    // Generate embedding for case summary if provided (for pgvector)
    let caseSummaryEmbedding = null;
    if (validatedData.caseSummary) {
      const embedding = await generateEmbedding(validatedData.caseSummary);
      if (embedding) {
        // Convert to PostgreSQL vector format
        caseSummaryEmbedding = `[${embedding.join(',')}]`;
      }
    }

    // Prepare case data for PostgreSQL
    const caseData: any = {
      id: caseId,
      caseNumber: validatedData.caseNumber,
      title: validatedData.title,
      description: validatedData.description || null,
      clientName: validatedData.clientName,
      opposingParty: validatedData.opposingParty || null,
      jurisdiction: validatedData.jurisdiction,
      courtName: validatedData.courtName || null,
      judgeAssigned: validatedData.judgeAssigned || null,
      caseType: validatedData.caseType,
      practiceArea: validatedData.practiceArea,
      priority: validatedData.priority,
      status: validatedData.status,
      filingDate: validatedData.filingDate ? new Date(validatedData.filingDate) : null,
      trialDate: validatedData.trialDate ? new Date(validatedData.trialDate) : null,
      estimatedValue: validatedData.estimatedValue || null,
      billingRate: validatedData.billingRate || null,
      caseSummary: validatedData.caseSummary || null,
      legalStrategy: validatedData.legalStrategy || null,
      keyIssues: validatedData.keyIssues || [],
      precedents: validatedData.precedents || null,
      createdAt: now,
      updatedAt: now,
      createdBy: user.id,
      assignedAttorney: null
    };

    // If we have an embedding, use raw SQL to insert with pgvector
    let newCase;
    if (caseSummaryEmbedding) {
      // Use raw SQL for pgvector insertion
      const result = await db.execute(sql`
        INSERT INTO legal_cases (
          id, case_number, title, description, client_name, opposing_party,
          jurisdiction, court_name, judge_assigned, case_type, practice_area,
          priority, status, filing_date, trial_date, estimated_value,
          billing_rate, case_summary, case_summary_embedding, legal_strategy,
          key_issues, precedents, created_at, updated_at, created_by
        ) VALUES (
          ${caseId}, ${caseData.caseNumber}, ${caseData.title}, 
          ${caseData.description}, ${caseData.clientName}, ${caseData.opposingParty},
          ${caseData.jurisdiction}, ${caseData.courtName}, ${caseData.judgeAssigned},
          ${caseData.caseType}, ${caseData.practiceArea}, ${caseData.priority},
          ${caseData.status}, ${caseData.filingDate}, ${caseData.trialDate},
          ${caseData.estimatedValue}, ${caseData.billingRate}, ${caseData.caseSummary},
          ${caseSummaryEmbedding}::vector, ${caseData.legalStrategy},
          ${caseData.keyIssues}, ${JSON.stringify(caseData.precedents)},
          ${caseData.createdAt}, ${caseData.updatedAt}, ${caseData.createdBy}
        ) RETURNING *
      `);
      newCase = result.rows;
    } else {
      // Use Drizzle ORM for regular insertion
      newCase = await db
        .insert(legalCases)
        .values(caseData)
        .returning();
    }

    // Trigger AI analysis for the new case
    if (validatedData.caseSummary || validatedData.description) {
      const analysisPrompt = `
        Analyze this legal case:
        Title: ${validatedData.title}
        Type: ${validatedData.caseType}
        Practice Area: ${validatedData.practiceArea}
        Summary: ${validatedData.caseSummary || validatedData.description}
        
        Provide:
        1. Key legal issues
        2. Potential risks
        3. Recommended precedents
        4. Strategic considerations
      `;
      
      // Store in cache for async processing
      await db.insert(agentAnalysisCache).values({
        id: crypto.randomUUID(),
        cacheKey: `case-analysis-${caseId}`,
        caseId: caseId,
        agentName: 'legal-assistant',
        analysisType: 'document_analysis',
        prompt: analysisPrompt,
        response: 'Analysis pending...',
        confidence: 0,
        expiresAt: new Date(Date.now() + 24 * 60 * 60 * 1000),
        createdAt: now,
        lastAccessed: now
      });

      // Trigger async AI processing
      fetch('http://localhost:8094/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          caseId,
          prompt: analysisPrompt
        })
      }).catch(err => console.error('AI analysis trigger failed:', err));
    }

    return json({
      success: true,
      message: 'Test case created successfully in PostgreSQL',
      data: Array.isArray(newCase) ? newCase[0] : newCase
    }, { status: 201 });

  } catch (error) {
    console.error('Error creating test case:', error);
    
    if (error instanceof z.ZodError) {
      return json({ 
        error: 'Validation failed',
        details: error.errors 
      }, { status: 400 });
    }

    // Check for unique constraint violations
    if (error instanceof Error && error.message.includes('unique')) {
      return json({ 
        error: 'Case number already exists',
        details: 'Please use a different case number' 
      }, { status: 409 });
    }

    return json({ 
      error: 'Failed to create test case',
      details: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 });
  }
};

// PUT: Update case in PostgreSQL
export const PUT: RequestHandler = async ({ request, url, locals }) => {
  try {
    const caseId = url.searchParams.get('id');
    
    if (!caseId) {
      return json({ error: 'Case ID is required' }, { status: 400 });
    }

    const body = await request.json();
    const now = new Date();

    // If case summary is updated, regenerate embedding
    let updateData: any = {
      ...body,
      updatedAt: now
    };

    if (body.caseSummary) {
      const embedding = await generateEmbedding(body.caseSummary);
      if (embedding) {
        // Update with pgvector embedding using raw SQL
        const result = await db.execute(sql`
          UPDATE legal_cases 
          SET 
            title = COALESCE(${body.title}, title),
            description = COALESCE(${body.description}, description),
            case_summary = ${body.caseSummary},
            case_summary_embedding = ${`[${embedding.join(',')}]`}::vector,
            updated_at = ${now}
          WHERE id = ${caseId}
          RETURNING *
        `);
        
        return json({
          success: true,
          message: 'Test case updated successfully',
          data: result.rows[0]
        });
      }
    }

    // Regular update without embedding
    const updatedCase = await db
      .update(legalCases)
      .set(updateData)
      .where(eq(legalCases.id, caseId))
      .returning();

    if (updatedCase.length === 0) {
      return json({ error: 'Case not found' }, { status: 404 });
    }

    return json({
      success: true,
      message: 'Test case updated successfully',
      data: updatedCase[0]
    });

  } catch (error) {
    console.error('Error updating test case:', error);
    return json({ 
      error: 'Failed to update test case',
      details: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 });
  }
};

// DELETE: Delete case from PostgreSQL
export const DELETE: RequestHandler = async ({ url, locals }) => {
  try {
    const caseId = url.searchParams.get('id');
    
    if (!caseId) {
      return json({ error: 'Case ID is required' }, { status: 400 });
    }

    // Use transaction for referential integrity
    await db.transaction(async (tx) => {
      // Delete related data first
      await tx.delete(caseDocuments).where(eq(caseDocuments.caseId, caseId));
      await tx.delete(agentAnalysisCache).where(eq(agentAnalysisCache.caseId, caseId));
      
      // Delete the case
      await tx.delete(legalCases).where(eq(legalCases.id, caseId));
    });

    return json({
      success: true,
      message: 'Test case deleted successfully from PostgreSQL',
      data: { id: caseId }
    });

  } catch (error) {
    console.error('Error deleting test case:', error);
    return json({ 
      error: 'Failed to delete test case',
      details: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 });
  }
};

// PATCH: Add/remove documents, perform vector search
export const PATCH: RequestHandler = async ({ request, url, locals }) => {
  try {
    const action = url.searchParams.get('action');
    
    if (action === 'add-document') {
      const { caseId, documentId, relationship, importance, notes } = await request.json();
      
      if (!caseId || !documentId) {
        return json({ error: 'Case ID and Document ID are required' }, { status: 400 });
      }

      const linkId = crypto.randomUUID();
      const link = await db
        .insert(caseDocuments)
        .values({
          id: linkId,
          caseId,
          documentId,
          relationship: relationship || 'evidence',
          importance: importance || 'medium',
          notes: notes || null,
          addedAt: new Date(),
          addedBy: locals.user?.id || 'system'
        })
        .returning();

      return json({
        success: true,
        message: 'Document linked to case successfully',
        data: link[0]
      });
    }

    if (action === 'vector-search') {
      const { query, limit = 10 } = await request.json();
      
      if (!query) {
        return json({ error: 'Search query is required' }, { status: 400 });
      }

      // Generate embedding for search query
      const embedding = await generateEmbedding(query);
      if (!embedding) {
        return json({ error: 'Failed to generate search embedding' }, { status: 500 });
      }

      // Perform vector similarity search using pgvector
      const results = await db.execute(sql`
        SELECT 
          id,
          case_number,
          title,
          client_name,
          priority,
          status,
          case_summary,
          1 - (case_summary_embedding <-> ${`[${embedding.join(',')}]`}::vector) as similarity
        FROM legal_cases
        WHERE case_summary_embedding IS NOT NULL
        ORDER BY case_summary_embedding <-> ${`[${embedding.join(',')}]`}::vector
        LIMIT ${limit}
      `);

      return json({
        success: true,
        message: 'Vector search completed',
        data: results.rows
      });
    }

    return json({ error: 'Invalid action' }, { status: 400 });

  } catch (error) {
    console.error('Error in PATCH operation:', error);
    return json({ 
      error: 'Failed to perform operation',
      details: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 });
  }
};
