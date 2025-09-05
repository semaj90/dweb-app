/**
 * AI Enrichment Queue Service
 * Handles enqueueing work for Claude agents to process and enrich data
 */

import { db } from '../server/db/index';
import { cases, evidence, documentChunks } from '../server/db/schema-postgres';
import { eq, sql } from 'drizzle-orm';

export interface EnrichmentJob {
  id: string;
  type: 'case_analysis' | 'evidence_processing' | 'document_embedding' | 'legal_research';
  entityId: string;
  entityType: 'case' | 'evidence' | 'document';
  priority: 'low' | 'medium' | 'high' | 'urgent';
  metadata: Record<string, any>;
  status: 'queued' | 'processing' | 'completed' | 'failed';
  createdAt: Date;
  processedAt?: Date;
}

export class AIEnrichmentQueue {
  
  /**
   * Enqueue a case for AI analysis and enrichment
   */
  async enqueueCaseAnalysis(caseId: string, priority: EnrichmentJob['priority'] = 'medium'): Promise<string> {
    // Get case data
    const caseData = await db.select()
      .from(cases)
      .where(eq(cases.id, caseId))
      .limit(1);

    if (!caseData.length) {
      throw new Error(`Case ${caseId} not found`);
    }

    const case_ = caseData[0];
    
    // Create enrichment job entry in database
    const jobId = crypto.randomUUID();
    
    // Store job in processing queue table (you'll need to create this)
    await this.storeJob({
      id: jobId,
      type: 'case_analysis',
      entityId: caseId,
      entityType: 'case',
      priority,
      metadata: {
        title: case_.title,
        description: case_.description,
        status: case_.status,
        practiceArea: case_.practice_area,
        jurisdiction: case_.jurisdiction
      },
      status: 'queued',
      createdAt: new Date()
    });

    // Trigger Redis/external queue notification if available
    await this.notifyExternalQueue('case_analysis', jobId);

    return jobId;
  }

  /**
   * Enqueue evidence for AI processing and embedding generation
   */
  async enqueueEvidenceProcessing(evidenceId: string, priority: EnrichmentJob['priority'] = 'medium'): Promise<string> {
    // Get evidence data  
    const evidenceData = await db.select()
      .from(evidence)
      .where(eq(evidence.id, evidenceId))
      .limit(1);

    if (!evidenceData.length) {
      throw new Error(`Evidence ${evidenceId} not found`);
    }

    const evidenceItem = evidenceData[0];
    
    const jobId = crypto.randomUUID();
    
    await this.storeJob({
      id: jobId,
      type: 'evidence_processing',
      entityId: evidenceId,
      entityType: 'evidence',
      priority,
      metadata: {
        title: evidenceItem.title,
        description: evidenceItem.description,
        evidenceType: evidenceItem.evidence_type,
        fileUrl: evidenceItem.file_url,
        caseId: evidenceItem.case_id
      },
      status: 'queued',
      createdAt: new Date()
    });

    await this.notifyExternalQueue('evidence_processing', jobId);
    
    return jobId;
  }

  /**
   * Enqueue document for chunking and embedding generation
   */
  async enqueueDocumentEmbedding(documentId: string, content: string, priority: EnrichmentJob['priority'] = 'medium'): Promise<string> {
    const jobId = crypto.randomUUID();
    
    await this.storeJob({
      id: jobId,
      type: 'document_embedding',
      entityId: documentId,
      entityType: 'document',
      priority,
      metadata: {
        contentLength: content.length,
        contentPreview: content.substring(0, 500)
      },
      status: 'queued',
      createdAt: new Date()
    });

    await this.notifyExternalQueue('document_embedding', jobId);
    
    return jobId;
  }

  /**
   * Store enrichment job in database
   */
  private async storeJob(job: EnrichmentJob): Promise<void> {
    // Store in a dedicated jobs table (you can create this)
    await db.execute(sql`
      INSERT INTO ai_enrichment_jobs (
        id, type, entity_id, entity_type, priority, metadata, status, created_at
      ) VALUES (
        ${job.id}, ${job.type}, ${job.entityId}, ${job.entityType}, 
        ${job.priority}, ${JSON.stringify(job.metadata)}, ${job.status}, ${job.createdAt}
      )
      ON CONFLICT (id) DO NOTHING
    `);
  }

  /**
   * Notify external queue system (Redis, NATS, etc.)
   */
  private async notifyExternalQueue(jobType: string, jobId: string): Promise<void> {
    try {
      // Option 1: Redis pub/sub
      // await redis.publish('ai:enrichment:queue', JSON.stringify({ jobType, jobId }));
      
      // Option 2: HTTP trigger to Claude agent
      const response = await fetch('http://localhost:8094/api/enqueue', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          jobType,
          jobId,
          timestamp: new Date().toISOString()
        })
      });

      if (!response.ok) {
        console.warn(`Failed to notify external queue: ${response.statusText}`);
      }
    } catch (error: any) {
      console.warn('External queue notification failed:', error);
      // Don't throw - job is still queued in database
    }
  }

  /**
   * Process completed enrichment and update entities
   */
  async processCompletedEnrichment(jobId: string, result: any): Promise<void> {
    // Get job details
    const jobData = await db.execute(sql`
      SELECT * FROM ai_enrichment_jobs WHERE id = ${jobId}
    `);

    if (!jobData.length) {
      throw new Error(`Job ${jobId} not found`);
    }

    const job = jobData[0];

    switch (job.type) {
      case 'case_analysis':
        await this.updateCaseWithAIResults(job.entity_id, result);
        break;
      case 'evidence_processing':
        await this.updateEvidenceWithAIResults(job.entity_id, result);
        break;
      case 'document_embedding':
        await this.storeDocumentEmbedding(job.entity_id, result);
        break;
    }

    // Mark job as completed
    await db.execute(sql`
      UPDATE ai_enrichment_jobs 
      SET status = 'completed', processed_at = NOW() 
      WHERE id = ${jobId}
    `);
  }

  /**
   * Update case with AI analysis results
   */
  private async updateCaseWithAIResults(caseId: string, results: any): Promise<void> {
    const updates: any = {};

    if (results.embedding) {
      updates.case_embedding = JSON.stringify(results.embedding);
    }

    if (results.aiTags) {
      updates.metadata = sql`metadata || ${JSON.stringify({ aiTags: results.aiTags })}`;
    }

    if (results.riskScore) {
      updates.metadata = sql`metadata || ${JSON.stringify({ riskScore: results.riskScore })}`;
    }

    if (results.precedentMatches) {
      updates.metadata = sql`metadata || ${JSON.stringify({ precedents: results.precedentMatches })}`;
    }

    await db.update(cases)
      .set(updates)
      .where(eq(cases.id, caseId));
  }

  /**
   * Update evidence with AI processing results
   */
  private async updateEvidenceWithAIResults(evidenceId: string, results: any): Promise<void> {
    const updates: any = {};

    if (results.titleEmbedding) {
      updates.title_embedding = JSON.stringify(results.titleEmbedding);
    }

    if (results.contentEmbedding) {
      updates.content_embedding = JSON.stringify(results.contentEmbedding);
    }

    if (results.aiSummary) {
      updates.ai_summary = results.aiSummary;
    }

    if (results.aiTags) {
      updates.ai_tags = JSON.stringify(results.aiTags);
    }

    if (results.analysis) {
      updates.ai_analysis = JSON.stringify(results.analysis);
    }

    await db.update(evidence)
      .set(updates)
      .where(eq(evidence.id, evidenceId));
  }

  /**
   * Store document chunks with embeddings
   */
  private async storeDocumentEmbedding(documentId: string, results: any): Promise<void> {
    if (!results.chunks || !Array.isArray(results.chunks)) {
      throw new Error('Invalid chunk results');
    }

    // Store each chunk with its embedding
    for (let i = 0; i < results.chunks.length; i++) {
      const chunk = results.chunks[i];
      
      await db.insert(documentChunks).values({
        document_id: documentId,
        document_type: 'processed',
        chunk_index: i,
        content: chunk.content,
        embedding: JSON.stringify(chunk.embedding),
        metadata: JSON.stringify({
          tokenCount: chunk.tokenCount,
          overlaps: chunk.overlaps
        })
      });
    }
  }

  /**
   * Get job status
   */
  async getJobStatus(jobId: string): Promise<EnrichmentJob | null> {
    const result = await db.execute(sql`
      SELECT * FROM ai_enrichment_jobs WHERE id = ${jobId}
    `);

    return result.length ? result[0] as EnrichmentJob : null;
  }

  /**
   * Get pending jobs for external processing
   */
  async getPendingJobs(type?: string, limit: number = 10): Promise<EnrichmentJob[]> {
    const typeFilter = type ? sql`AND type = ${type}` : sql``;
    
    const result = await db.execute(sql`
      SELECT * FROM ai_enrichment_jobs 
      WHERE status = 'queued' ${typeFilter}
      ORDER BY 
        CASE priority 
          WHEN 'urgent' THEN 1
          WHEN 'high' THEN 2 
          WHEN 'medium' THEN 3
          ELSE 4
        END,
        created_at ASC
      LIMIT ${limit}
    `);

    return result as EnrichmentJob[];
  }
}

// Export singleton instance
export const aiEnrichmentQueue = new AIEnrichmentQueue();