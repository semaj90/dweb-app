// Phase 1: SvelteKit Frontend Cache-First Architecture
// LokiJS + Superforms + Zod Integration for Legal AI Platform

import Loki from 'lokijs';
import { z } from 'zod';
import { writable, derived, type Writable } from 'svelte/store'
import { superForm } from 'sveltekit-superforms';
import { zod } from 'sveltekit-superforms/adapters';
import crypto from "crypto";
import { URL } from "url";

// ===== CACHE-FIRST SCHEMAS =====

export const CaseSchema = z.object({
  id: z.string().uuid(),
  title: z.string().min(1, 'Case title required'),
  description: z.string().optional(),
  status: z.enum(['open', 'pending', 'closed', 'archived']),
  priority: z.enum(['low', 'medium', 'high', 'critical']),
  createdAt: z.date(),
  updatedAt: z.date(),
  userId: z.string().uuid(),
  metadata: z.record(z.any()).default({}),
  // Cache metadata
  _cached: z.boolean().default(true),
  _lastSync: z.date().default(new Date()),
  _dirty: z.boolean().default(false)
});

export const EvidenceSchema = z.object({
  id: z.string().uuid(),
  caseId: z.string().uuid(),
  title: z.string().min(1, 'Evidence title required'),
  evidenceType: z.enum(['document', 'image', 'video', 'audio', 'physical']),
  fileUrl: z.string().url().optional(),
  fileSize: z.number().optional(),
  analysisResults: z.record(z.any()).default({}),
  tags: z.array(z.string()).default([]),
  createdAt: z.date(),
  // Cache metadata
  _cached: z.boolean().default(true),
  _lastSync: z.date().default(new Date()),
  _dirty: z.boolean().default(false)
});

export const AIAnalysisSchema = z.object({
  id: z.string().uuid(),
  entityId: z.string().uuid(), // Case or Evidence ID
  entityType: z.enum(['case', 'evidence']),
  analysisType: z.enum(['summary', 'classification', 'extraction', 'recommendation']),
  prompt: z.string(),
  response: z.string(),
  confidence: z.number().min(0).max(1),
  model: z.string(),
  processingTime: z.number(),
  createdAt: z.date(),
  // Cache metadata
  _cached: z.boolean().default(true),
  _lastSync: z.date().default(new Date())
});

export type Case = z.infer<typeof CaseSchema>
export type Evidence = z.infer<typeof EvidenceSchema>
export type AIAnalysis = z.infer<typeof AIAnalysisSchema>

// ===== CACHE-FIRST SERVICE =====

export class CacheFirstService {
  private db: Loki;
  private casesCollection: Collection<Case>;
  private evidenceCollection: Collection<Evidence>;
  private aiAnalysisCollection: Collection<AIAnalysis>;
  
  // Reactive stores for real-time UI updates
  public cases: Writable<Case[]> = writable([]);
  public evidence: Writable<Evidence[]> = writable([]);
  public aiAnalyses: Writable<AIAnalysis[]> = writable([]);
  
  // Cache statistics
  public stats = writable({
    totalCases: 0,
    totalEvidence: 0,
    totalAnalyses: 0,
    cacheHitRate: 0,
    lastSyncTime: new Date(),
    pendingSync: 0
  });

  constructor() {
    this.db = new Loki('legal-ai-cache.db', {
      adapter: typeof window !== 'undefined' ? new LokiIndexedAdapter('legal-ai') : undefined,
      autoload: true,
      autoloadCallback: () => this.initializeCollections(),
      autosave: true,
      autosaveInterval: 10000 // 10 seconds
    });
  }

  private initializeCollections() {
    // Cases collection with indices for fast queries
    this.casesCollection = this.db.getCollection('cases') || 
      this.db.addCollection('cases', {
        indices: ['id', 'userId', 'status', 'priority', 'createdAt'],
        unique: ['id']
      });

    // Evidence collection with indices
    this.evidenceCollection = this.db.getCollection('evidence') || 
      this.db.addCollection('evidence', {
        indices: ['id', 'caseId', 'evidenceType', 'createdAt'],
        unique: ['id']
      });

    // AI Analysis collection
    this.aiAnalysisCollection = this.db.getCollection('ai_analyses') || 
      this.db.addCollection('ai_analyses', {
        indices: ['id', 'entityId', 'entityType', 'analysisType', 'createdAt'],
        unique: ['id']
      });

    this.refreshStores();
  }

  // ===== CACHE-FIRST CRUD OPERATIONS =====

  async createCase(caseData: Omit<Case, 'id' | 'createdAt' | 'updatedAt'>): Promise<Case> {
    const newCase: Case = CaseSchema.parse({
      ...caseData,
      id: crypto.randomUUID(),
      createdAt: new Date(),
      updatedAt: new Date(),
      _cached: true,
      _lastSync: new Date(),
      _dirty: true // Needs sync to server
    });

    // Insert into cache immediately
    this.casesCollection.insert(newCase);
    this.refreshStores();

    // Background sync to server
    this.syncToServer('cases', newCase);

    return newCase;
  }

  async getCaseById(caseId: string): Promise<Case | null> {
    // Cache-first lookup
    const cached = this.casesCollection.findOne({ id: caseId });
    
    if (cached) {
      this.updateCacheStats('hit');
      return cached;
    }

    // Cache miss - fetch from server
    this.updateCacheStats('miss');
    try {
      const response = await fetch(`/api/cases/${caseId}`);
      if (response.ok) {
        const serverCase = await response.json();
        const case_ = CaseSchema.parse({
          ...serverCase,
          _cached: true,
          _lastSync: new Date(),
          _dirty: false
        });
        
        this.casesCollection.insert(case_);
        this.refreshStores();
        return case_;
      }
    } catch (error) {
      console.error('Failed to fetch case from server:', error);
    }

    return null;
  }

  async updateCase(caseId: string, updates: Partial<Case>): Promise<Case | null> {
    const existing = this.casesCollection.findOne({ id: caseId });
    if (!existing) return null;

    const updated: Case = CaseSchema.parse({
      ...existing,
      ...updates,
      updatedAt: new Date(),
      _dirty: true // Mark for sync
    });

    this.casesCollection.update(updated);
    this.refreshStores();

    // Background sync
    this.syncToServer('cases', updated);

    return updated;
  }

  async getCasesForUser(userId: string): Promise<Case[]> {
    // Cache-first with user filter
    const cached = this.casesCollection.find({ userId });
    
    if (cached.length > 0) {
      this.updateCacheStats('hit');
      return cached;
    }

    // Fetch from server if cache is empty
    this.updateCacheStats('miss');
    try {
      const response = await fetch(`/api/cases?userId=${userId}`);
      if (response.ok) {
        const serverCases = await response.json();
        const validatedCases = serverCases.map((case_: any) => 
          CaseSchema.parse({
            ...case_,
            _cached: true,
            _lastSync: new Date(),
            _dirty: false
          })
        );

        // Bulk insert into cache
        validatedCases.forEach(case_ => this.casesCollection.insert(case_));
        this.refreshStores();
        
        return validatedCases;
      }
    } catch (error) {
      console.error('Failed to fetch cases from server:', error);
    }

    return [];
  }

  // ===== EVIDENCE OPERATIONS =====

  async createEvidence(evidenceData: Omit<Evidence, 'id' | 'createdAt'>): Promise<Evidence> {
    const newEvidence: Evidence = EvidenceSchema.parse({
      ...evidenceData,
      id: crypto.randomUUID(),
      createdAt: new Date(),
      _cached: true,
      _lastSync: new Date(),
      _dirty: true
    });

    this.evidenceCollection.insert(newEvidence);
    this.refreshStores();
    this.syncToServer('evidence', newEvidence);

    return newEvidence;
  }

  async getEvidenceForCase(caseId: string): Promise<Evidence[]> {
    const cached = this.evidenceCollection.find({ caseId });
    
    if (cached.length > 0) {
      this.updateCacheStats('hit');
      return cached;
    }

    // Fetch from server
    this.updateCacheStats('miss');
    try {
      const response = await fetch(`/api/evidence?caseId=${caseId}`);
      if (response.ok) {
        const serverEvidence = await response.json();
        const validatedEvidence = serverEvidence.map((ev: any) => 
          EvidenceSchema.parse({
            ...ev,
            _cached: true,
            _lastSync: new Date(),
            _dirty: false
          })
        );

        validatedEvidence.forEach(ev => this.evidenceCollection.insert(ev));
        this.refreshStores();
        
        return validatedEvidence;
      }
    } catch (error) {
      console.error('Failed to fetch evidence from server:', error);
    }

    return [];
  }

  // ===== AI ANALYSIS OPERATIONS =====

  async createAIAnalysis(analysisData: Omit<AIAnalysis, 'id' | 'createdAt'>): Promise<AIAnalysis> {
    const analysis: AIAnalysis = AIAnalysisSchema.parse({
      ...analysisData,
      id: crypto.randomUUID(),
      createdAt: new Date(),
      _cached: true,
      _lastSync: new Date()
    });

    this.aiAnalysisCollection.insert(analysis);
    this.refreshStores();

    return analysis;
  }

  async getAIAnalyses(entityId: string, entityType: 'case' | 'evidence'): Promise<AIAnalysis[]> {
    return this.aiAnalysisCollection.find({ entityId, entityType });
  }

  // ===== CACHE MANAGEMENT =====

  private refreshStores() {
    this.cases.set(this.casesCollection.data);
    this.evidence.set(this.evidenceCollection.data);
    this.aiAnalyses.set(this.aiAnalysisCollection.data);
    
    // Update stats
    this.stats.update(current => ({
      ...current,
      totalCases: this.casesCollection.count(),
      totalEvidence: this.evidenceCollection.count(),
      totalAnalyses: this.aiAnalysisCollection.count(),
      lastSyncTime: new Date(),
      pendingSync: this.casesCollection.find({ _dirty: true }).length + 
                   this.evidenceCollection.find({ _dirty: true }).length
    }));
  }

  private cacheHits = 0;
  private cacheMisses = 0;

  private updateCacheStats(type: 'hit' | 'miss') {
    if (type === 'hit') this.cacheHits++
    else this.cacheMisses++;

    const total = this.cacheHits + this.cacheMisses;
    const hitRate = total > 0 ? this.cacheHits / total : 0;

    this.stats.update(current => ({
      ...current,
      cacheHitRate: hitRate
    }));
  }

  // ===== BACKGROUND SYNC =====

  private async syncToServer(collection: 'cases' | 'evidence', data: any) {
    try {
      const endpoint = collection === 'cases' ? '/api/cases' : '/api/evidence';
      const method = data._dirty ? 'PUT' : 'POST';
      
      await fetch(endpoint, {
        method,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
      });

      // Mark as clean after successful sync
      if (collection === 'cases') {
        const item = this.casesCollection.findOne({ id: data.id });
        if (item) {
          item._dirty = false;
          item._lastSync = new Date();
          this.casesCollection.update(item);
        }
      } else {
        const item = this.evidenceCollection.findOne({ id: data.id });
        if (item) {
          item._dirty = false;
          item._lastSync = new Date();
          this.evidenceCollection.update(item);
        }
      }

      this.refreshStores();
    } catch (error) {
      console.error('Sync to server failed:', error);
    }
  }

  // ===== CACHE WARMING =====

  async warmCache(userId: string) {
    console.log('Warming cache for user:', userId);
    
    // Preload user's cases and evidence
    await this.getCasesForUser(userId);
    
    // Preload evidence for each case
    const userCases = this.casesCollection.find({ userId });
    for (const case_ of userCases) {
      await this.getEvidenceForCase(case_.id);
    }

    console.log('Cache warming completed');
  }

  // ===== CACHE CLEANUP =====

  async cleanupCache(maxAge: number = 24 * 60 * 60 * 1000) { // 24 hours
    const cutoff = new Date(Date.now() - maxAge);
    
    // Remove old cached items
    this.casesCollection.removeWhere((case_: Case) => 
      case_._lastSync < cutoff && !case_._dirty
    );
    
    this.evidenceCollection.removeWhere((evidence: Evidence) => 
      evidence._lastSync < cutoff && !evidence._dirty
    );

    this.refreshStores();
    console.log('Cache cleanup completed');
  }
}

// ===== GLOBAL CACHE INSTANCE =====

export const cacheFirstService = new CacheFirstService();
// ===== SUPERFORMS INTEGRATION =====

export function createCacheFirstForm<T extends z.ZodSchema>(
  schema: T,
  options?: {
    onSuccess?: (data: z.infer<T>) => void | Promise<void>;
    onError?: (errors: any) => void;
  }
) {
  return {
    schema,
    adapter: zod,
    ...options,
    // Enhanced with cache-first patterns
    onUpdate: async ({ form }) => {
      if (form.valid) {
        // Cache the form data immediately
        const validData = schema.parse(form.data);
        // Additional cache-first logic here
        options?.onSuccess?.(validData);
      }
    }
  };
}

// ===== REACTIVE CACHE QUERIES =====

export function createCacheQuery<T>(
  queryFn: () => Promise<T>,
  key: string,
  options?: {
    refetchInterval?: number;
    staleTime?: number;
  }
) {
  const data = writable<T | null>(null);
  const loading = writable(false);
  const error = writable<Error | null>(null);

  let cache: { data: T; timestamp: number } | null = null;
  const staleTime = options?.staleTime || 5 * 60 * 1000; // 5 minutes

  async function fetch() {
    // Check cache first
    if (cache && Date.now() - cache.timestamp < staleTime) {
      data.set(cache.data);
      return cache.data;
    }

    loading.set(true);
    error.set(null);

    try {
      const result = await queryFn();
      cache = { data: result, timestamp: Date.now() };
      data.set(result);
      return result;
    } catch (err) {
      error.set(err as Error);
      throw err;
    } finally {
      loading.set(false);
    }
  }

  // Auto-refetch if specified
  if (options?.refetchInterval) {
    setInterval(fetch, options.refetchInterval);
  }

  // Initial fetch
  fetch();

  return {
    data: { subscribe: data.subscribe },
    loading: { subscribe: loading.subscribe },
    error: { subscribe: error.subscribe },
    refetch: fetch,
    invalidate: () => {
      cache = null;
      fetch();
    }
  };
}