import type { RequestHandler } from './$types.js';

// src/routes/api/vectors/sync/+server.ts
// Automatic vector synchronization to Qdrant after CUDA processing
// Triggered by Go microservice after successful vector generation

import { json } from '@sveltejs/kit';
import { drizzle } from 'drizzle-orm/node-postgres';
import postgres from 'postgres';
import { createClient } from 'redis';
import { vectors, vectorJobs, evidence, reports } from '$lib/server/db/schema-postgres.js';
import { eq } from 'drizzle-orm';

// Initialize connections
const sql = postgres(import.meta.env.DATABASE_URL || 'postgresql://legal_admin:123456@localhost:5432/legal_ai_db');
const db = drizzle(sql);

const redis = createClient({ 
  url: import.meta.env.REDIS_URL || 'redis://localhost:6379' 
});

let redisConnected = false;

async function connectRedis(): Promise<any> {
  if (!redisConnected) {
    await redis.connect();
    redisConnected = true;
  }
}

// Qdrant client (simple HTTP implementation)
class QdrantClient {
  private baseUrl: string;

  constructor(baseUrl = 'http://localhost:6333') {
    this.baseUrl = baseUrl;
  }

  async upsertPoint(collectionName: string, pointData: {
    id: string;
    vector: number[];
    payload: Record<string, any>;
  }) {
    const response = await fetch(`${this.baseUrl}/collections/${collectionName}/points`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        points: [pointData]
      })
    });

    if (!response.ok) {
      throw new Error(`Qdrant upsert failed: ${response.statusText}`);
    }

    return await response.json();
  }

  async deletePoint(collectionName: string, pointId: string) {
    const response = await fetch(`${this.baseUrl}/collections/${collectionName}/points/delete`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        points: [pointId]
      })
    });

    if (!response.ok) {
      throw new Error(`Qdrant delete failed: ${response.statusText}`);
    }

    return await response.json();
  }

  async ensureCollection(collectionName: string, vectorSize = 768) {
    try {
      // Check if collection exists
      const checkResponse = await fetch(`${this.baseUrl}/collections/${collectionName}`);
      if (checkResponse.ok) {
        return; // Collection already exists
      }

      // Create collection
      const createResponse = await fetch(`${this.baseUrl}/collections/${collectionName}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          vectors: {
            size: vectorSize,
            distance: "Cosine"
          }
        })
      });

      if (!createResponse.ok) {
        throw new Error(`Failed to create collection: ${createResponse.statusText}`);
      }

      console.log(`✅ Created Qdrant collection: ${collectionName}`);
    } catch (error: any) {
      console.error(`❌ Qdrant collection error for ${collectionName}:`, error);
      throw error;
    }
  }
}

const qdrant = new QdrantClient();

export const POST: RequestHandler = async ({ request }) => {
  let body: any;
  try {
    body = await request.json();
    const { jobId, vectorId, ownerType, ownerId, event } = body;

    console.log(`🔄 Syncing vector to Qdrant: ${jobId} (${event})`);

    // Validate required fields
    if (!jobId || !ownerType || !ownerId || !event) {
      return json({ 
        error: 'Missing required fields: jobId, ownerType, ownerId, event' 
      }, { status: 400 });
    }

    // Update job status to processing
    await db
      .update(vectorJobs)
      .set({ 
        status: 'processing',
        progress: 50,
        startedAt: new Date(),
      })
      .where(eq(vectorJobs.jobId, jobId));

    let result;

    if (event === 'delete') {
      // Handle deletion
      result = await handleVectorDeletion(ownerType, ownerId);
    } else {
      // Handle upsert/reembed
      result = await handleVectorUpsert(ownerType, ownerId, vectorId);
    }

    // Update job status to succeeded
    await db
      .update(vectorJobs)
      .set({ 
        status: 'succeeded',
        progress: 100,
        completedAt: new Date(),
        result: result,
      })
      .where(eq(vectorJobs.jobId, jobId));

    console.log(`✅ Vector sync completed: ${jobId}`);

    return json({
      success: true,
      jobId,
      result,
      message: `Vector ${event} completed successfully`,
    });

  } catch (error: any) {
    console.error('❌ Vector sync error:', error);
    
    // Update job status to failed
    if (body?.jobId) {
      await db
        .update(vectorJobs)
        .set({ 
          status: 'failed',
          error: error instanceof Error ? error.message : 'Unknown error',
          completedAt: new Date(),
        })
        .where(eq(vectorJobs.jobId, body.jobId))
        .catch(console.error);
    }
    
    return json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error',
    }, { status: 500 });
  }
};

async function handleVectorUpsert(ownerType: string, ownerId: string, vectorId?: string): Promise<any> {
  // Get vector from PostgreSQL
  const [vector] = await db
    .select()
    .from(vectors)
    .where(eq(vectors.ownerId, ownerId))
    .limit(1);

  if (!vector || !vector.embedding) {
    throw new Error('Vector not found or embedding missing');
  }

  // Get source data for payload
  let sourceData;
  let collectionName;

  switch (ownerType) {
    case 'evidence':
      [sourceData] = await db.select().from(evidence).where(eq(evidence.id, ownerId)).limit(1);
      collectionName = 'legal_evidence';
      break;
    
    case 'report':
      [sourceData] = await db.select().from(reports).where(eq(reports.id, ownerId)).limit(1);
      collectionName = 'legal_reports';
      break;
    
    default:
      throw new Error(`Unsupported owner type: ${ownerType}`);
  }

  if (!sourceData) {
    throw new Error(`Source data not found for ${ownerType}:${ownerId}`);
  }

  // Ensure Qdrant collection exists
  await qdrant.ensureCollection(collectionName);

  // Prepare point data for Qdrant
  const pointData = {
    id: ownerId,
    vector: Array.isArray(vector.embedding) ? vector.embedding : [],
    payload: {
      ownerType,
      title: sourceData.title || '',
      description: sourceData.description || '',
      createdAt: sourceData.createdAt?.toISOString(),
      updatedAt: sourceData.updatedAt?.toISOString(),
      metadata: sourceData.metadata || {},
      // Add specific fields based on type
      ...(ownerType === 'evidence' && {
        evidenceType: sourceData.evidenceType,
        caseId: sourceData.caseId,
        tags: sourceData.tags,
      }),
      ...(ownerType === 'report' && {
        reportType: sourceData.reportType,
        caseId: sourceData.caseId,
        status: sourceData.status,
      }),
    }
  };

  // Upsert to Qdrant
  const qdrantResult = await qdrant.upsertPoint(collectionName, pointData);

  return {
    action: 'upserted',
    collection: collectionName,
    pointId: ownerId,
    vectorDimensions: pointData.vector.length,
    qdrantResult,
  };
}

async function handleVectorDeletion(ownerType: string, ownerId: string): Promise<any> {
  const collectionName = ownerType === 'evidence' ? 'legal_evidence' : 'legal_reports';
  
  // Delete from Qdrant
  const qdrantResult = await qdrant.deletePoint(collectionName, ownerId);

  return {
    action: 'deleted',
    collection: collectionName,
    pointId: ownerId,
    qdrantResult,
  };
}

// Health check endpoint
export const GET: RequestHandler = async () => {
  try {
    // Check Qdrant connection
    const response = await fetch(`${qdrant.baseUrl || 'http://localhost:6333'}/collections`);
    const collections = response.ok ? await response.json() : null;

    // Check PostgreSQL connection
    const [pgTest] = await db.select().from(vectors).limit(1);

    // Check Redis connection
    await connectRedis();
    const redisInfo = await redis.ping();

    return json({
      success: true,
      services: {
        qdrant: {
          connected: response.ok,
          collections: collections?.result?.collections || [],
        },
        postgresql: {
          connected: !!pgTest,
        },
        redis: {
          connected: redisInfo === 'PONG',
        }
      },
      timestamp: new Date().toISOString(),
    });

  } catch (error: any) {
    return json({
      success: false,
      error: error instanceof Error ? error.message : 'Health check failed',
    }, { status: 500 });
  }
};