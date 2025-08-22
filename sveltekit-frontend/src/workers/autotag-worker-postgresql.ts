/**
 * PostgreSQL-First Auto-Tagging Worker
 * 
 * Architecture:
 * 1. Evidence uploads → PostgreSQL evidence table
 * 2. Go ingest-service (port 8227) → PostgreSQL document_metadata + document_embeddings
 * 3. Worker → Reads PostgreSQL, enriches PostgreSQL, mirrors to Qdrant
 * 4. Single source of truth: PostgreSQL
 * 5. Qdrant: Search index only (rebuildable from PostgreSQL)
 */

import { createClient } from 'redis';
import { db } from '$lib/server/db/index';
import { evidence, documentMetadata, documentEmbeddings, workerEvents } from '$lib/server/db/schema-unified';
import { eq, and, isNull, sql, desc } from 'drizzle-orm';
import { QdrantClient } from '@qdrant/js-client-rest';

// Types for Redis events
interface AutoTagEvent {
  type: 'evidence' | 'document' | 'ingest_complete';
  id: string;
  action?: 'tag' | 'mirror' | 'process';
  caseId?: string;
  userId?: string;
  retry?: boolean;
  correlationId?: string;
}

// PostgreSQL notification types
interface PgNotification {
  channel: string;
  payload: string;
}

interface IngestCompletionPayload {
  document_id: string;
  case_id?: string;
  evidence_id?: string;
  processing_status: string;
  timestamp: string;
}

interface EvidenceUpdatePayload {
  evidence_id: string;
  case_id?: string;
  processing_status: string;
  ingest_status: string;
  timestamp: string;
}

export class PostgreSQLFirstWorker {
  private redis: ReturnType<typeof createClient>;
  private qdrant: QdrantClient;
  private isRunning = false;
  private streamName = 'autotag:requests';
  private pgNotificationClient: any; // PostgreSQL LISTEN client
  private embeddingCache = new Map<string, number[]>(); // In-memory embedding cache
  private processedEvents = new Set<string>(); // Deduplication

  constructor() {
    this.redis = createClient({ 
      url: process.env.REDIS_URL || 'redis://localhost:6379',
      socket: { connectTimeout: 5000, lazyConnect: true }
    });
    
    this.qdrant = new QdrantClient({ 
      url: process.env.QDRANT_URL || 'http://localhost:6333' 
    });
  }

  async start() {
    try {
      await this.redis.connect();
      console.log('🔌 Connected to Redis');
      
      // Setup PostgreSQL notification listener
      await this.setupPostgreSQLNotifications();
      
      // Ensure Qdrant collection exists
      await this.ensureQdrantCollection();
      
      this.isRunning = true;
      console.log('🚀 PostgreSQL-First Auto-Tagging Worker started');
      console.log('📡 Listening for Redis events on stream:', this.streamName);
      console.log('📡 Listening for PostgreSQL notifications: ingest_completed, evidence_updated');
      
      // Start both event processors concurrently
      await Promise.all([
        this.processEventStream(), // Redis streams
        this.processPostgreSQLNotifications() // PostgreSQL LISTEN/NOTIFY
      ]);
    } catch (error) {
      console.error('❌ Failed to start worker:', error);
      throw error;
    }
  }

  /**
   * Setup PostgreSQL LISTEN for real-time notifications
   */
  private async setupPostgreSQLNotifications() {
    try {
      // Create separate client for LISTEN (blocking operation)
      const { Client } = await import('pg');
      this.pgNotificationClient = new Client({
        connectionString: process.env.DATABASE_URL || 'postgresql://legal_admin:123456@localhost:5432/legal_ai_db'
      });
      
      await this.pgNotificationClient.connect();
      
      // Listen for ingest completion and evidence updates
      await this.pgNotificationClient.query('LISTEN ingest_completed');
      await this.pgNotificationClient.query('LISTEN evidence_updated');
      
      console.log('✅ PostgreSQL notifications setup complete');
    } catch (error) {
      console.warn('⚠️  PostgreSQL notifications setup failed (non-critical):', error.message);
    }
  }

  /**
   * Process PostgreSQL LISTEN/NOTIFY events
   */
  private async processPostgreSQLNotifications() {
    if (!this.pgNotificationClient) {
      console.log('📡 PostgreSQL notifications not available, using Redis-only mode');
      return;
    }
    
    this.pgNotificationClient.on('notification', async (notification: PgNotification) => {
      try {
        const payload = JSON.parse(notification.payload);
        
        switch (notification.channel) {
          case 'ingest_completed':
            await this.handleIngestCompletionNotification(payload as IngestCompletionPayload);
            break;
            
          case 'evidence_updated':
            await this.handleEvidenceUpdateNotification(payload as EvidenceUpdatePayload);
            break;
            
          default:
            console.log(`📡 Unknown PostgreSQL notification: ${notification.channel}`);
        }
      } catch (error) {
        console.error('❌ Failed to process PostgreSQL notification:', error);
      }
    });
    
    console.log('📡 PostgreSQL notification processor started');
  }

  /**
   * Handle ingest completion from PostgreSQL notifications
   */
  private async handleIngestCompletionNotification(payload: IngestCompletionPayload) {
    const correlationId = `pg_${Date.now()}_${Math.random().toString(36).slice(2)}`;
    
    console.log(`📡 PostgreSQL ingest completion: ${payload.document_id}`);
    
    // Avoid duplicate processing
    if (this.processedEvents.has(payload.document_id)) {
      console.log(`⚠️  Already processed document ${payload.document_id}, skipping`);
      return;
    }
    
    this.processedEvents.add(payload.document_id);
    
    // Log event in PostgreSQL
    await this.logWorkerEvent({
      eventType: 'ingest_complete',
      targetId: payload.document_id,
      caseId: payload.case_id,
      correlationId,
      metadata: { source: 'postgresql_notification', payload }
    });
    
    // Process the event
    await this.processIngestComplete({
      type: 'ingest_complete',
      id: payload.document_id,
      caseId: payload.case_id,
      correlationId
    });
  }

  /**
   * Handle evidence updates from PostgreSQL notifications
   */
  private async handleEvidenceUpdateNotification(payload: EvidenceUpdatePayload) {
    const correlationId = `pg_${Date.now()}_${Math.random().toString(36).slice(2)}`;
    
    console.log(`📡 PostgreSQL evidence update: ${payload.evidence_id} -> ${payload.processing_status}`);
    
    // Only process specific status changes
    if (payload.processing_status === 'completed' && payload.ingest_status === 'pending') {
      await this.processEvidenceEvent({
        type: 'evidence',
        id: payload.evidence_id,
        action: 'tag',
        caseId: payload.case_id,
        correlationId
      });
    }
  }

  private async processEventStream() {
    let lastId = '$'; // Start with latest events
    
    while (this.isRunning) {
      try {
        const streams = await this.redis.xRead([
          { key: this.streamName, id: lastId }
        ], { 
          COUNT: 10, 
          BLOCK: 5000 // 5 second timeout
        });
        
        if (!streams || streams.length === 0) {
          continue; // No new events
        }
        
        for (const stream of streams) {
          for (const message of stream.messages) {
            lastId = message.id;
            
            const event = this.parseRedisMessage(message.message);
            if (event) {
              await this.processEvent(event);
            }
          }
        }
        
      } catch (error) {
        console.error('❌ Stream processing error:', error);
        
        // Exponential backoff on errors
        await this.sleep(Math.min(5000, 1000 * Math.pow(2, 1)));
        
        // Reset lastId to continue from latest on reconnect
        lastId = '$';
      }
    }
  }

  private parseRedisMessage(message: Record<string, string>): AutoTagEvent | null {
    try {
      return {
        type: message.type as AutoTagEvent['type'],
        id: message.id,
        action: message.action as AutoTagEvent['action'],
        caseId: message.caseId,
        retry: message.retry === 'true'
      };
    } catch (error) {
      console.warn('⚠️  Failed to parse Redis message:', message);
      return null;
    }
  }

  private async processEvent(event: AutoTagEvent) {
    const startTime = Date.now();
    console.log(`📝 Processing ${event.type} event: ${event.id} (action: ${event.action})`);
    
    try {
      switch (event.type) {
        case 'evidence':
          await this.processEvidenceEvent(event);
          break;
          
        case 'document':
          await this.processDocumentEvent(event);
          break;
          
        case 'ingest_complete':
          await this.processIngestComplete(event);
          break;
          
        default:
          console.log(`⚠️  Unknown event type: ${event.type}`);
      }
      
      const duration = Date.now() - startTime;
      console.log(`✅ Processed ${event.type} ${event.id} in ${duration}ms`);
      
    } catch (error) {
      console.error(`❌ Failed to process ${event.type} ${event.id}:`, error);
      
      // Retry logic for failed events
      if (!event.retry) {
        await this.requeueEvent({ ...event, retry: true });
      }
    }
  }

  /**
   * Process evidence auto-tagging (PostgreSQL-First)
   * 1. Read evidence from PostgreSQL (single source of truth)
   * 2. Check if embedding already exists in PostgreSQL
   * 3. Generate auto-tags based on metadata (NO external API calls)
   * 4. Update PostgreSQL with enriched data
   * 5. Mirror to Qdrant asynchronously (non-blocking)
   */
  private async processEvidenceEvent(event: AutoTagEvent) {
    const startTime = Date.now();
    
    // Log event start
    await this.logWorkerEvent({
      eventType: 'evidence',
      targetId: event.id,
      caseId: event.caseId,
      userId: event.userId,
      correlationId: event.correlationId,
      metadata: { action: event.action }
    });
    // 1. Read evidence from PostgreSQL (single source of truth)
    const [evidenceItem] = await db
      .select()
      .from(evidence)
      .where(eq(evidence.id, event.id))
      .limit(1);
      
    if (!evidenceItem) {
      console.log(`⚠️  Evidence ${event.id} not found in PostgreSQL`);
      return;
    }
    
    console.log(`📄 Processing evidence: ${evidenceItem.title} (${evidenceItem.fileName})`);
    
    // 2. Generate auto-tags based on PostgreSQL evidence data
    const autoTags = this.generateEvidenceTags(evidenceItem);
    const aiAnalysis = this.generateEvidenceAnalysis(evidenceItem);
    
    // 3. Update PostgreSQL with enriched data (no external dependencies)
    const [updatedEvidence] = await db
      .update(evidence)
      .set({
        aiTags: autoTags,
        aiAnalysis: {
          ...evidenceItem.aiAnalysis,
          ...aiAnalysis,
          autoTagged: true,
          taggedAt: new Date().toISOString(),
          worker: 'postgresql-first-worker',
          version: '1.0'
        }
      })
      .where(eq(evidence.id, event.id))
      .returning();
      
    console.log(`✅ Updated evidence ${event.id} in PostgreSQL:`, {
      tags: autoTags.length,
      confidence: aiAnalysis.confidence
    });
    
    // 4. Check if embedding exists in PostgreSQL and mirror to Qdrant (async, non-blocking)
    setImmediate(() => this.mirrorEvidenceToQdrant(event.id).catch(console.warn));
    
    // 5. Log successful completion
    const processingTime = Date.now() - startTime;
    await this.logWorkerEvent({
      eventType: 'evidence',
      targetId: event.id,
      caseId: event.caseId,
      userId: event.userId,
      correlationId: event.correlationId,
      processingStatus: 'completed',
      processingTimeMs: processingTime,
      metadata: { 
        action: event.action, 
        tagsGenerated: autoTags.length,
        confidence: aiAnalysis.confidence
      }
    });
    
    return updatedEvidence;
  }

  /**
   * Process document from ingest service
   * 1. Verify document exists in PostgreSQL
   * 2. Check if embedding is available (from Go ingest-service)
   * 3. Mirror to Qdrant when ready
   */
  private async processDocumentEvent(event: AutoTagEvent) {
    // 1. Check document in PostgreSQL
    const [document] = await db
      .select()
      .from(documentMetadata)
      .where(eq(documentMetadata.id, event.id))
      .limit(1);
      
    if (!document) {
      console.log(`⚠️  Document ${event.id} not found in PostgreSQL`);
      return;
    }
    
    console.log(`📄 Processing document: ${document.originalFilename}`);
    
    // 2. Check if ingest service has created embedding
    const [embedding] = await db
      .select()
      .from(documentEmbeddings)
      .where(eq(documentEmbeddings.documentId, event.id))
      .limit(1);
      
    if (embedding) {
      console.log(`✅ Document ${event.id} has embedding (dims: ${embedding.embedding?.length || 'unknown'})`);
      
      // Mirror to Qdrant (async)
      setImmediate(() => this.mirrorDocumentToQdrant(event.id).catch(console.warn));
      
    } else {
      console.log(`⏳ Document ${event.id} waiting for ingest service embedding`);
      
      // Re-queue for later processing (with delay)
      setTimeout(() => {
        this.requeueEvent({
          type: 'document',
          id: event.id,
          action: 'process',
          retry: true
        }).catch(console.warn);
      }, 5000); // 5 second delay before retry
    }
    
    return document;
  }

  /**
   * Process ingest completion notification
   * Triggered when Go ingest-service finishes processing
   */
  private async processIngestComplete(event: AutoTagEvent) {
    console.log(`🎯 Ingest complete for document: ${event.id}`);
    
    // Process both document mirroring and related evidence
    await Promise.all([
      this.processDocumentEvent({ type: 'document', id: event.id, action: 'mirror' }),
      this.linkDocumentToEvidence(event.id, event.caseId)
    ]);
  }

  /**
   * Link ingested document to related evidence
   */
  private async linkDocumentToEvidence(documentId: string, caseId?: string) {
    if (!caseId) return;
    
    try {
      // Find evidence in the same case that might be related
      const relatedEvidence = await db
        .select()
        .from(evidence)
        .where(eq(evidence.caseId, caseId))
        .limit(10);
        
      for (const evidenceItem of relatedEvidence) {
        // Trigger evidence re-processing to include document context
        await this.requeueEvent({
          type: 'evidence',
          id: evidenceItem.id,
          action: 'tag',
          caseId
        });
      }
      
      console.log(`🔗 Linked document ${documentId} to ${relatedEvidence.length} evidence items`);
    } catch (error) {
      console.warn('⚠️  Failed to link document to evidence:', error);
    }
  }

  /**
   * Generate auto-tags based on evidence metadata
   * Uses only PostgreSQL data, no external API calls
   */
  private generateEvidenceTags(evidenceItem: any): string[] {
    const tags: string[] = [];
    
    // MIME type classification
    const mimeType = evidenceItem.mimeType?.toLowerCase() || '';
    if (mimeType.includes('pdf')) tags.push('pdf-document');
    if (mimeType.includes('image')) tags.push('image-evidence');
    if (mimeType.includes('video')) tags.push('video-evidence');
    if (mimeType.includes('audio')) tags.push('audio-evidence');
    if (mimeType.includes('text')) tags.push('text-document');
    
    // File name analysis
    const fileName = evidenceItem.fileName?.toLowerCase() || '';
    const titleLower = evidenceItem.title?.toLowerCase() || '';
    const descriptionLower = evidenceItem.description?.toLowerCase() || '';
    
    const textContent = `${fileName} ${titleLower} ${descriptionLower}`;
    
    // Legal document types
    if (textContent.includes('contract')) tags.push('contract', 'legal-document');
    if (textContent.includes('agreement')) tags.push('agreement', 'legal-document');
    if (textContent.includes('report')) tags.push('report');
    if (textContent.includes('statement')) tags.push('statement');
    if (textContent.includes('invoice')) tags.push('financial', 'billing');
    if (textContent.includes('receipt')) tags.push('financial', 'receipt');
    if (textContent.includes('email')) tags.push('correspondence', 'communication');
    if (textContent.includes('memo')) tags.push('internal-communication', 'memo');
    if (textContent.includes('photo')) tags.push('photograph', 'visual-evidence');
    if (textContent.includes('screenshot')) tags.push('digital-evidence', 'screenshot');
    
    // Evidence type mapping
    const evidenceType = evidenceItem.evidenceType?.toLowerCase();
    if (evidenceType) {
      tags.push(`type-${evidenceType}`);
      
      // Enhanced classification based on evidence type
      switch (evidenceType) {
        case 'document':
          tags.push('documentary-evidence');
          break;
        case 'physical':
          tags.push('physical-evidence');
          break;
        case 'digital':
          tags.push('digital-evidence');
          break;
        case 'testimony':
          tags.push('witness-statement');
          break;
      }
    }
    
    // File size analysis
    const fileSize = evidenceItem.fileSize || 0;
    if (fileSize > 10 * 1024 * 1024) tags.push('large-file'); // > 10MB
    if (fileSize < 1024) tags.push('small-file'); // < 1KB
    
    // Date-based tags
    const createdAt = new Date(evidenceItem.createdAt);
    const now = new Date();
    const daysDiff = Math.floor((now.getTime() - createdAt.getTime()) / (1000 * 60 * 60 * 24));
    
    if (daysDiff < 1) tags.push('recent');
    if (daysDiff < 7) tags.push('this-week');
    if (daysDiff < 30) tags.push('this-month');
    
    // Confidentiality level
    if (evidenceItem.confidentialityLevel) {
      tags.push(`confidentiality-${evidenceItem.confidentialityLevel}`);
    }
    
    return [...new Set(tags)]; // Remove duplicates
  }

  /**
   * Generate AI analysis metadata
   */
  private generateEvidenceAnalysis(evidenceItem: any): any {
    const analysis = {
      confidence: 0.85,
      processingMethod: 'rule-based',
      features: {
        hasFileExtension: !!evidenceItem.fileName?.includes('.'),
        hasDescription: !!evidenceItem.description,
        hasMetadata: !!evidenceItem.metadata,
        fileSize: evidenceItem.fileSize || 0,
        mimeTypeDetected: !!evidenceItem.mimeType
      },
      recommendations: [] as string[]
    };
    
    // Generate recommendations based on evidence
    if (!evidenceItem.description) {
      analysis.recommendations.push('Add description for better searchability');
    }
    
    if (!evidenceItem.tags || evidenceItem.tags.length === 0) {
      analysis.recommendations.push('Consider adding manual tags');
    }
    
    if (evidenceItem.mimeType?.includes('image') && !evidenceItem.metadata?.ocrText) {
      analysis.recommendations.push('OCR processing recommended for text extraction');
    }
    
    return analysis;
  }

  /**
   * Mirror evidence embedding to Qdrant (PostgreSQL -> Qdrant sync)
   * PostgreSQL is the authoritative source, Qdrant is search index only
   */
  private async mirrorEvidenceToQdrant(evidenceId: string) {
    try {
      // First check for evidence embedding in evidence table itself
      const [evidenceWithEmbedding] = await db
        .select({
          id: evidence.id,
          titleEmbedding: evidence.titleEmbedding,
          contentEmbedding: evidence.contentEmbedding,
          title: evidence.title,
          description: evidence.description,
          caseId: evidence.caseId,
          evidenceType: evidence.evidenceType,
          mimeType: evidence.mimeType,
          aiTags: evidence.aiTags,
          confidentialityLevel: evidence.confidentialityLevel
        })
        .from(evidence)
        .where(eq(evidence.id, evidenceId))
        .limit(1);
        
      if (!evidenceWithEmbedding) {
        console.log(`⚠️  Evidence ${evidenceId} not found for Qdrant mirroring`);
        return;
      }
      
      // Try content embedding first, fallback to title embedding
      let embeddingVector = evidenceWithEmbedding.contentEmbedding || evidenceWithEmbedding.titleEmbedding;
      let embeddingContent = evidenceWithEmbedding.description || evidenceWithEmbedding.title;
      
      // If no direct embeddings, check document_embeddings table
      if (!embeddingVector) {
        const [docEmbedding] = await db
          .select({
            embedding: documentEmbeddings.embedding,
            content: documentEmbeddings.content
          })
          .from(documentEmbeddings)
          .where(eq(documentEmbeddings.evidenceId, evidenceId))
          .orderBy(desc(documentEmbeddings.createdAt))
          .limit(1);
          
        if (docEmbedding?.embedding) {
          embeddingVector = docEmbedding.embedding;
          embeddingContent = docEmbedding.content;
        }
      }
        
      if (!embeddingVector) {
        console.log(`⚠️  No embedding available for evidence ${evidenceId}`);
        return; // No embedding available
      }
      
      // Mirror to Qdrant with PostgreSQL data as authoritative source
      await this.qdrant.upsert('legal_documents', {
        wait: false, // Async operation
        points: [{
          id: `evidence_${evidenceId}`,
          vector: embeddingVector,
          payload: {
            type: 'evidence',
            evidenceId: evidenceId,
            caseId: evidenceWithEmbedding.caseId,
            title: evidenceWithEmbedding.title,
            tags: evidenceWithEmbedding.aiTags || [],
            content: embeddingContent,
            metadata: {
              evidenceType: evidenceWithEmbedding.evidenceType,
              mimeType: evidenceWithEmbedding.mimeType,
              confidentialityLevel: evidenceWithEmbedding.confidentialityLevel,
              source: 'postgresql_mirror',
              mirroredAt: new Date().toISOString(),
              workerVersion: '2.0-postgresql-first'
            }
          }
        }]
      });
      
      console.log(`🔄 Mirrored evidence ${evidenceId} to Qdrant`);
      
    } catch (error) {
      console.warn(`⚠️  Qdrant mirror failed for evidence ${evidenceId}:`, error.message);
    }
  }

  /**
   * Mirror document embedding to Qdrant (async, non-blocking)
   */
  private async mirrorDocumentToQdrant(documentId: string) {
    try {
      // Get document and embedding from PostgreSQL
      const [result] = await db
        .select({
          document: documentMetadata,
          embedding: documentEmbeddings
        })
        .from(documentMetadata)
        .leftJoin(documentEmbeddings, eq(documentMetadata.id, documentEmbeddings.documentId))
        .where(eq(documentMetadata.id, documentId))
        .limit(1);
        
      if (!result?.embedding?.embedding) {
        return; // No embedding available
      }
      
      // Mirror to Qdrant
      await this.qdrant.upsert('legal_documents', {
        wait: false,
        points: [{
          id: `document_${documentId}`,
          vector: result.embedding.embedding,
          payload: {
            type: 'document',
            documentId: documentId,
            title: result.document.originalFilename,
            summary: result.document.summary,
            documentType: result.document.documentType,
            content: result.embedding.content,
            metadata: {
              ...result.embedding.metadata,
              processingStatus: result.document.processingStatus,
              fileSize: result.document.fileSize,
              contentType: result.document.contentType
            },
            source: 'postgresql_mirror',
            mirroredAt: new Date().toISOString()
          }
        }]
      });
      
      console.log(`🔄 Mirrored document ${documentId} to Qdrant`);
      
    } catch (error) {
      console.warn(`⚠️  Qdrant mirror failed for document ${documentId}:`, error.message);
    }
  }

  /**
   * Ensure Qdrant collection exists with proper configuration
   */
  private async ensureQdrantCollection() {
    try {
      const collections = await this.qdrant.getCollections();
      const exists = collections.collections.some(c => c.name === 'legal_documents');
      
      if (!exists) {
        await this.qdrant.createCollection('legal_documents', {
          vectors: {
            size: 384, // nomic-embed-text dimensions
            distance: 'Cosine'
          },
          optimizers_config: {
            default_segment_number: 2
          }
        });
        console.log('✅ Created Qdrant collection: legal_documents');
      } else {
        console.log('✅ Qdrant collection exists: legal_documents');
      }
    } catch (error) {
      console.warn('⚠️  Qdrant collection setup failed (non-critical):', error.message);
    }
  }

  /**
   * Re-queue event for retry
   */
  private async requeueEvent(event: AutoTagEvent) {
    try {
      await this.redis.xAdd(this.streamName, '*', {
        type: event.type,
        id: event.id,
        action: event.action || 'process',
        caseId: event.caseId || '',
        retry: 'true'
      });
      
      console.log(`🔄 Re-queued ${event.type} ${event.id} for retry`);
    } catch (error) {
      console.error(`❌ Failed to re-queue ${event.type} ${event.id}:`, error);
    }
  }

  /**
   * Utility: Sleep for specified milliseconds
   */
  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Log worker events to PostgreSQL for monitoring and debugging
   */
  private async logWorkerEvent(event: {
    eventType: string;
    targetId: string;
    caseId?: string;
    userId?: string;
    correlationId?: string;
    processingStatus?: string;
    processingTimeMs?: number;
    errorMessage?: string;
    metadata?: any;
  }) {
    try {
      await db.insert(workerEvents).values({
        eventType: event.eventType,
        eventAction: 'process',
        targetId: event.targetId,
        caseId: event.caseId,
        userId: event.userId,
        processingStatus: event.processingStatus || 'processing',
        processingTimeMs: event.processingTimeMs,
        errorMessage: event.errorMessage,
        metadata: {
          correlationId: event.correlationId,
          ...(event.metadata || {})
        }
      });
    } catch (error) {
      console.warn('⚠️  Failed to log worker event:', error.message);
    }
  }

  /**
   * Get cached embedding or return null
   */
  private getCachedEmbedding(textHash: string): number[] | null {
    return this.embeddingCache.get(textHash) || null;
  }

  /**
   * Cache embedding in memory and optionally in PostgreSQL
   */
  private async cacheEmbedding(textHash: string, embedding: number[]) {
    // Cache in memory
    this.embeddingCache.set(textHash, embedding);
    
    // Optionally cache in PostgreSQL for persistence
    try {
      await db.insert(embeddingCache).values({
        textHash,
        embedding,
        model: 'nomic-embed-text'
      }).onConflictDoNothing();
    } catch (error) {
      console.warn('⚠️  Failed to cache embedding in PostgreSQL:', error.message);
    }
  }

  /**
   * Graceful shutdown
   */
  async stop() {
    console.log('🛑 Stopping PostgreSQL-First Worker...');
    this.isRunning = false;
    
    try {
      await this.redis.disconnect();
      console.log('✅ Disconnected from Redis');
    } catch (error) {
      console.warn('⚠️  Redis disconnect error:', error);
    }
    
    try {
      if (this.pgNotificationClient) {
        await this.pgNotificationClient.end();
        console.log('✅ Disconnected from PostgreSQL notifications');
      }
    } catch (error) {
      console.warn('⚠️  PostgreSQL notification disconnect error:', error);
    }
    
    // Clear caches
    this.embeddingCache.clear();
    this.processedEvents.clear();
    
    console.log('✅ PostgreSQL-First Worker stopped');
  }

  /**
   * Health check
   */
  async healthCheck(): Promise<{
    status: 'healthy' | 'unhealthy';
    redis: boolean;
    qdrant: boolean;
    postgresql: boolean;
    uptime: number;
  }> {
    const startTime = Date.now();
    const health = {
      status: 'healthy' as const,
      redis: false,
      qdrant: false,
      postgresql: false,
      uptime: this.isRunning ? Date.now() - startTime : 0
    };
    
    try {
      // Check Redis
      await this.redis.ping();
      health.redis = true;
    } catch (error) {
      health.redis = false;
    }
    
    try {
      // Check Qdrant
      await this.qdrant.getCollections();
      health.qdrant = true;
    } catch (error) {
      health.qdrant = false;
    }
    
    try {
      // Check PostgreSQL
      await db.execute(sql`SELECT 1`);
      health.postgresql = true;
    } catch (error) {
      health.postgresql = false;
    }
    
    health.status = health.redis && health.postgresql ? 'healthy' : 'unhealthy';
    return health;
  }
}

// Export for use in other modules
export default PostgreSQLFirstWorker;

// Start worker if run directly
if (import.meta.url === `file://${process.argv[1]}`) {
  const worker = new PostgreSQLFirstWorker();
  
  // Graceful shutdown handling
  const shutdown = async (signal: string) => {
    console.log(`\n🛑 Received ${signal}, shutting down worker...`);
    await worker.stop();
    process.exit(0);
  };
  
  process.on('SIGINT', () => shutdown('SIGINT'));
  process.on('SIGTERM', () => shutdown('SIGTERM'));
  
  // Start worker
  worker.start().catch((error) => {
    console.error('❌ Worker startup failed:', error);
    process.exit(1);
  });
  
  console.log('🚀 PostgreSQL-First Worker started successfully');
}