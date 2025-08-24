// workers/autotag-worker.ts
// Postgres-first auto-tagging worker that enriches evidence and reports
// Uses Redis Streams for coordination, Postgres as source of truth

import { createClient } from 'redis';
import { drizzle } from 'drizzle-orm/postgres-js';
import postgres from 'postgres';
import { eq, and } from 'drizzle-orm';
import { 
  evidenceTable, 
  reportsTable, 
  evidenceVectorsTable,
  type Evidence,
  type Report 
} from '../src/lib/server/schema.js';

// Initialize connections
const redis = createClient({ 
  url: process.env.REDIS_URL || 'redis://localhost:6379' 
});

const sql = postgres(process.env.DATABASE_URL || 'postgresql://postgres:123456@localhost:5432/legal_ai_db');
const db = drizzle(sql);

let isConnected = false;

// Connect to Redis
async function connectRedis() {
  if (!isConnected) {
    await redis.connect();
    isConnected = true;
    console.log('✅ AutoTag Worker: Connected to Redis');
  }
}

// Auto-tagging logic based on evidence metadata
function generateEvidenceTags(evidence: Evidence): string[] {
  const tags: string[] = [];
  
  // File type tags
  if (evidence.file_type?.includes('pdf')) tags.push('pdf', 'document');
  if (evidence.file_type?.includes('image')) tags.push('image', 'visual');
  if (evidence.file_type?.includes('video')) tags.push('video', 'multimedia');
  if (evidence.file_type?.includes('audio')) tags.push('audio', 'multimedia');
  
  // Size-based tags
  if (evidence.file_size) {
    if (evidence.file_size > 10 * 1024 * 1024) tags.push('large-file');
    if (evidence.file_size < 100 * 1024) tags.push('small-file');
  }
  
  // Title/filename analysis
  const filename = evidence.title?.toLowerCase() || evidence.file_path?.toLowerCase() || '';
  if (filename.includes('contract')) tags.push('contract', 'legal-document');
  if (filename.includes('invoice')) tags.push('invoice', 'financial');
  if (filename.includes('email')) tags.push('email', 'communication');
  if (filename.includes('photo')) tags.push('photograph', 'evidence');
  if (filename.includes('scan')) tags.push('scanned', 'document');
  
  // Metadata-based tags
  if (evidence.metadata && typeof evidence.metadata === 'object') {
    const meta = evidence.metadata as Record<string, any>;
    if (meta.pages && meta.pages > 1) tags.push('multi-page');
    if (meta.encrypted) tags.push('encrypted');
    if (meta.signed) tags.push('digitally-signed');
  }
  
  // Default tags
  tags.push('unprocessed', 'needs-review');
  
  return [...new Set(tags)]; // Remove duplicates
}

// Auto-summary generation for reports
function generateReportSummary(report: Report): string {
  if (!report.doc || typeof report.doc !== 'object') {
    return `Report: ${report.title}`;
  }
  
  // Extract text from rich text editor doc (assuming ProseMirror/TipTap format)
  const doc = report.doc as any;
  let text = '';
  
  if (doc.content && Array.isArray(doc.content)) {
    text = doc.content
      .map((node: any) => {
        if (node.type === 'paragraph' && node.content) {
          return node.content
            .filter((item: any) => item.type === 'text')
            .map((item: any) => item.text)
            .join('');
        }
        return '';
      })
      .join(' ')
      .trim();
  }
  
  // Generate summary (first 150 chars + intelligent truncation)
  if (text.length > 150) {
    const truncated = text.substring(0, 150);
    const lastPeriod = truncated.lastIndexOf('.');
    const lastSpace = truncated.lastIndexOf(' ');
    
    if (lastPeriod > 100) {
      return truncated.substring(0, lastPeriod + 1);
    } else if (lastSpace > 100) {
      return truncated.substring(0, lastSpace) + '...';
    }
  }
  
  return text || `Report: ${report.title}`;
}

// Process evidence tagging
async function processEvidence(evidenceId: string) {
  try {
    const [evidence] = await db
      .select()
      .from(evidenceTable)
      .where(eq(evidenceTable.id, evidenceId));
    
    if (!evidence) {
      console.warn(`❌ Evidence ${evidenceId} not found`);
      return;
    }
    
    // Generate tags
    const tags = generateEvidenceTags(evidence);
    
    // Update evidence with tags
    await db
      .update(evidenceTable)
      .set({ tags })
      .where(eq(evidenceTable.id, evidenceId));
    
    console.log(`✅ Tagged evidence ${evidenceId} with: ${tags.join(', ')}`);
    
    // Check if vector exists and optionally sync to Qdrant
    const [vector] = await db
      .select()
      .from(evidenceVectorsTable)
      .where(eq(evidenceVectorsTable.evidence_id, evidenceId));
    
    if (vector) {
      console.log(`🔍 Vector exists for evidence ${evidenceId} - ready for search`);
      // TODO: Optionally sync to Qdrant here if needed
    }
    
  } catch (error) {
    console.error(`❌ Error processing evidence ${evidenceId}:`, error);
  }
}

// Process report summarization
async function processReport(reportId: string) {
  try {
    const [report] = await db
      .select()
      .from(reportsTable)
      .where(eq(reportsTable.id, reportId));
    
    if (!report) {
      console.warn(`❌ Report ${reportId} not found`);
      return;
    }
    
    // Generate summary if empty
    if (!report.summary || report.summary.trim() === '') {
      const summary = generateReportSummary(report);
      
      await db
        .update(reportsTable)
        .set({ 
          summary,
          updated_at: new Date()
        })
        .where(eq(reportsTable.id, reportId));
      
      console.log(`✅ Generated summary for report ${reportId}: "${summary.substring(0, 50)}..."`);
    } else {
      console.log(`ℹ️ Report ${reportId} already has summary`);
    }
    
  } catch (error) {
    console.error(`❌ Error processing report ${reportId}:`, error);
  }
}

// Multi-threaded job processing with k-means clustering
async function processWithMultiThreading(type: string, id: string, data: any) {
  const { Worker } = await import('worker_threads');
  const kmeansWorkerPath = new URL('../src/lib/workers/kmeans-worker.js', import.meta.url);
  
  // Prepare data for k-means clustering if it's a batch job
  if (type === 'evidence_batch' || type === 'report_batch') {
    console.log(`🧮 Starting k-means clustering for ${type} batch of ${data?.items?.length || 0} items`);
    
    const worker = new Worker(kmeansWorkerPath);
    
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        worker.terminate();
        reject(new Error('K-means worker timeout'));
      }, 30000); // 30 second timeout
      
      worker.on('message', (result) => {
        clearTimeout(timeout);
        if (result.type === 'result') {
          console.log(`✅ K-means clustering completed: ${result.iterations} iterations, ${result.clusters.length} clusters`);
          resolve(result);
        } else if (result.type === 'error') {
          reject(new Error(result.error));
        }
        worker.terminate();
      });
      
      worker.on('error', (error) => {
        clearTimeout(timeout);
        reject(error);
        worker.terminate();
      });
      
      // Send data to worker
      worker.postMessage({
        data: data.items || [],
        k: Math.min(5, Math.max(2, Math.floor((data?.items?.length || 0) / 10))), // Dynamic k
        dimensions: 768, // Embedding dimensions
        options: {
          maxIterations: 50,
          convergenceThreshold: 0.001
        }
      });
    });
  }
  
  // For single items, process normally
  return null;
}

// Enhanced evidence processing with multi-threading support
async function processEvidence(evidenceId: string, batchData?: any) {
  try {
    // Check if this is a batch job with clustering
    if (batchData && batchData.items) {
      const clusterResult = await processWithMultiThreading('evidence_batch', evidenceId, batchData);
      
      if (clusterResult) {
        console.log(`🎯 Evidence batch clustered into ${clusterResult.clusters.length} groups`);
        
        // Process each cluster
        for (const cluster of clusterResult.clusters) {
          for (const dataIndex of cluster.dataIndices) {
            const item = batchData.items[dataIndex];
            if (item?.id) {
              await processSingleEvidence(item.id, { 
                clusterId: cluster.id,
                clusterCohesion: cluster.cohesion,
                clusterSilhouette: cluster.silhouette 
              });
            }
          }
        }
        return;
      }
    }
    
    // Single evidence processing
    await processSingleEvidence(evidenceId);
    
  } catch (error) {
    console.error(`❌ Error processing evidence ${evidenceId}:`, error);
  }
}

// Single evidence processing (extracted from original function)
async function processSingleEvidence(evidenceId: string, clusterInfo?: any) {
  const [evidence] = await db
    .select()
    .from(evidenceTable)
    .where(eq(evidenceTable.id, evidenceId));
  
  if (!evidence) {
    console.warn(`❌ Evidence ${evidenceId} not found`);
    return;
  }
  
  // Generate tags
  const tags = generateEvidenceTags(evidence);
  
  // Add cluster-based tags if available
  if (clusterInfo) {
    tags.push(`cluster-${clusterInfo.clusterId}`, 'ai-clustered');
    if (clusterInfo.clusterCohesion > 0.8) tags.push('high-cohesion');
    if (clusterInfo.clusterSilhouette > 0.5) tags.push('well-separated');
  }
  
  // Update evidence with tags
  await db
    .update(evidenceTable)
    .set({ tags })
    .where(eq(evidenceTable.id, evidenceId));
  
  console.log(`✅ Tagged evidence ${evidenceId} with: ${tags.join(', ')}`);
  
  // Submit to vector pipeline for CUDA processing
  try {
    const vectorJob = await submitToVectorPipeline({
      ownerType: 'evidence',
      ownerId: evidenceId,
      event: 'upsert',
      data: {
        title: evidence.title,
        description: evidence.description,
        tags,
        clusterInfo
      }
    });
    
    console.log(`🚀 Submitted evidence ${evidenceId} to vector pipeline: ${vectorJob.jobId}`);
    
  } catch (vectorError) {
    console.error(`❌ Failed to submit to vector pipeline:`, vectorError);
  }
  
  // Check if vector exists and optionally sync to Qdrant
  const [vector] = await db
    .select()
    .from(evidenceVectorsTable)
    .where(eq(evidenceVectorsTable.evidence_id, evidenceId));
  
  if (vector) {
    console.log(`🔍 Vector exists for evidence ${evidenceId} - ready for search`);
  }
}

// Vector pipeline integration
async function submitToVectorPipeline(params: {
  ownerType: 'evidence' | 'report';
  ownerId: string;
  event: 'upsert' | 'reembed';
  data: any;
}) {
  // Use SvelteKit API endpoint for vector processing
  const response = await fetch('http://localhost:5173/api/compute', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  
  if (!response.ok) {
    throw new Error(`Vector pipeline submission failed: ${response.statusText}`);
  }
  
  return await response.json();
}

// Main worker loop with multi-threading support
async function runWorker() {
  await connectRedis();
  
  console.log('🚀 AutoTag Worker started - listening for Redis streams with multi-threading...');
  
  let lastId = '$'; // Start from latest messages
  
  while (true) {
    try {
      // Read from Redis streams
      const streams = await redis.xRead({
        key: 'autotag:requests',
        id: lastId
      }, {
        COUNT: 10,
        BLOCK: 5000 // 5 second timeout
      });
      
      if (streams && streams.length > 0) {
        for (const stream of streams) {
          for (const message of stream.messages) {
            lastId = message.id;
            
            const { type, id, data } = message.message as { 
              type: string; 
              id: string; 
              data?: string;
            };
            
            console.log(`📨 Processing ${type} ${id}`);
            
            // Parse data if present
            let parsedData;
            try {
              parsedData = data ? JSON.parse(data) : null;
            } catch (parseError) {
              console.warn(`⚠️ Failed to parse message data:`, parseError);
            }
            
            switch (type) {
              case 'evidence':
              case 'evidence_batch':
                await processEvidence(id, parsedData);
                break;
              case 'report':
              case 'report_batch':
                await processReport(id, parsedData);
                break;
              default:
                console.warn(`❓ Unknown message type: ${type}`);
            }
          }
        }
      }
      
    } catch (error) {
      if (error.message?.includes('BLOCK')) {
        // Timeout is normal, continue
        continue;
      }
      console.error('❌ Worker error:', error);
      
      // Wait before retrying
      await new Promise(resolve => setTimeout(resolve, 5000));
    }
  }
}

// Graceful shutdown
process.on('SIGINT', async () => {
  console.log('🛑 Shutting down AutoTag Worker...');
  await redis.disconnect();
  await sql.end();
  process.exit(0);
});

process.on('SIGTERM', async () => {
  console.log('🛑 Shutting down AutoTag Worker...');
  await redis.disconnect();
  await sql.end();
  process.exit(0);
});

// Start the worker
runWorker().catch(console.error);