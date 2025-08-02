/**
 * Enhanced RAG System Type Definitions
 * Comprehensive types for multi-modal legal AI RAG system
 */

export interface RAGDocument {
  id: string;
  title: string;
  content: string;
  metadata: {
    source: string;
    type: "case" | "statute" | "regulation" | "evidence" | "memo";
    jurisdiction: string;
    practiceArea: string[];
    confidentialityLevel: number;
    lastModified: Date;
    fileSize: number;
    language: string;
    tags: string[];
    caseId?: string;
    evidenceType?: string;
    confidence?: number;
    timestamp?: number;
  };
  embeddings?: number[];
  summary?: string;
  keyPhrases?: string[];
  legalCitations?: string[];
  version: string;
}

export interface EmbeddingResponse {
  embeddings: number[];
  model: string;
  dimensions: number;
  processingTime: number;
  tokenCount: number;
}

export interface SearchResult {
  id: string;
  document: RAGDocument;
  score: number;
  relevantChunks: TextChunk[];
  highlights: string[];
  explanation: string;
  legalRelevance: LegalRelevanceScore;
  clusterId?: string;
  cacheLayer?: string;
  relevanceScore: number;
  rank: number;
  snippet?: string;
}

export interface TextChunk {
  id: string;
  content: string;
  startIndex: number;
  endIndex: number;
  score: number;
  embeddings: number[];
  chunkType: "paragraph" | "section" | "citation" | "header";
}

export interface LegalRelevanceScore {
  overall: number;
  factual: number;
  procedural: number;
  precedential: number;
  jurisdictional: number;
  confidence: number;
}

export interface RAGSystemStatus {
  isOnline: boolean;
  modelsLoaded: boolean;
  vectorDBConnected: boolean;
  lastSync: Date | null;
  version: string;
  health: "healthy" | "degraded" | "critical";
  activeConnections: number;
  memoryUsage: {
    current: number;
    peak: number;
    limit: number;
  };
  isInitialized: boolean;
  isIndexing: boolean;
  isSearching: boolean;
  documentsCount: number;
  lastUpdate: number;
  cacheHitRate: number;
  errorCount: number;
}

export interface MLCachingMetrics {
  hitRate: number;
  memoryUsageMB: number;
  predictionAccuracy: number;
  layersActive: string[];
  avgResponseTime: number;
  compressionRatio: number;
  evictionCount: number;
  predictiveHits: number;
  missRate: number;
  evictionRate: number;
  memoryPressure: number;
  clusterCount: number;
  averageSearchTime: number;
  cacheSize: number;
  recommendations: string[];
}

export interface QuerySuggestion {
  text: string;
  confidence: number;
  type: "typo_correction" | "semantic_expansion" | "legal_term" | "case_law";
  reasoning: string;
}

export interface RAGSearchParams {
  query: string;
  limit?: number;
  threshold?: number;
  includeMetadata?: boolean;
  useMLRanking?: boolean;
  caseId?: string;
  evidenceTypes?: string[];
}

export interface RecommendationResult {
  suggestions: QuerySuggestion[];
  relatedQueries: string[];
  hotWords: string[];
  confidence: number;
}

export interface SOMCluster {
  id: string;
  centroid: number[];
  documents: string[];
  keywords: string[];
  practiceAreas: string[];
  avgRelevance: number;
  lastUpdated: Date;
  size: number;
  coherence: number;
}

export interface NeuralPrediction {
  queryId: string;
  predictedRelevance: number;
  suggestedQueries: string[];
  optimalCacheLayer: string;
  confidence: number;
  modelVersion: string;
  features: Record<string, number>;
}

export interface RAGSearchOptions {
  limit?: number;
  threshold?: number;
  enableClustering?: boolean;
  enableReranking?: boolean;
  practiceAreas?: string[];
  jurisdictions?: string[];
  documentTypes?: string[];
  dateRange?: {
    start: Date;
    end: Date;
  };
  confidentialityLevel?: number;
  enableMLOptimization?: boolean;
  cacheStrategy?: "aggressive" | "balanced" | "minimal";
}

export interface RAGResponse {
  results: SearchResult[];
  totalFound: number;
  searchTime: number;
  query: string;
  suggestions: string[];
  didYouMean: string[];
  filters: SearchFilters;
  clusters: SOMCluster[];
  metadata: ResponseMetadata;
  // Additional properties for evidence synthesis
  analysis?: string;
  recommendations?: string[];
}

export interface SearchFilters {
  practiceAreas: string[];
  jurisdictions: string[];
  documentTypes: string[];
  dateRange: {
    start: Date;
    end: Date;
  } | null;
  confidentialityLevel: number;
  tags: string[];
}

export interface ResponseMetadata {
  cacheHit: boolean;
  cacheLayer?: string;
  modelUsed: string;
  embeddingDimensions: number;
  processingSteps: ProcessingStep[];
  optimizationApplied: string[];
  qualityScore: number;
}

export interface ProcessingStep {
  step: string;
  duration: number;
  status: "success" | "error" | "warning";
  details?: any;
}

export interface CacheEntry<T = any> {
  key: string;
  value: T;
  timestamp: number;
  accessCount: number;
  lastAccessed: number;
  size: number;
  ttl?: number;
  layer: "L1" | "L2" | "L3" | "L4" | "L5" | "L6" | "L7";
  compressionRatio?: number;
  predictionScore?: number;
}

export interface OptimizationConfig {
  enableNeuralCaching: boolean;
  enableSOMClustering: boolean;
  enablePredictiveLoading: boolean;
  cacheCompressionLevel: number;
  maxCacheSizeMB: number;
  autoOptimizationInterval: number;
  learningRate: number;
  clusteringThreshold: number;
}

export interface RecommendationEngine {
  type: "content" | "query" | "document" | "legal";
  suggestions: Recommendation[];
  confidence: number;
  reasoning: string;
  context: RecommendationContext;
}

export interface Recommendation {
  id: string;
  text: string;
  score: number;
  type: "similar" | "related" | "followup" | "clarification";
  metadata: Record<string, any>;
}

export interface RecommendationContext {
  currentQuery: string;
  searchHistory: string[];
  userProfile: UserProfile;
  sessionContext: SessionContext;
}

export interface UserProfile {
  id: string;
  practiceAreas: string[];
  jurisdictions: string[];
  experienceLevel: "junior" | "mid" | "senior" | "expert";
  preferences: UserPreferences;
  searchPatterns: SearchPattern[];
}

export interface UserPreferences {
  resultFormat: "detailed" | "summary" | "headlines";
  sortOrder: "relevance" | "date" | "authority";
  highlightStyle: "context" | "keywords" | "legal";
  enableAutoSuggestions: boolean;
  maxResults: number;
}

export interface SearchPattern {
  query: string;
  frequency: number;
  lastUsed: Date;
  avgRelevance: number;
  followupQueries: string[];
}

export interface SessionContext {
  startTime: Date;
  queriesCount: number;
  documentsViewed: string[];
  averageRelevance: number;
  currentPracticeArea?: string;
  workflowStage?: "research" | "analysis" | "drafting" | "review";
}

export interface WebAssemblyJSONProcessor {
  parseJSON: (input: string) => Promise<any>;
  compressData: (data: any) => Promise<Uint8Array>;
  decompressData: (compressed: Uint8Array) => Promise<any>;
  optimizeEmbeddings: (embeddings: number[]) => Promise<number[]>;
  calculateSimilarity: (a: number[], b: number[]) => Promise<number>;
  benchmark: () => Promise<PerformanceBenchmark>;
}

export interface PerformanceBenchmark {
  parsing: {
    speed: number; // operations per second
    accuracy: number;
    memoryUsage: number;
  };
  compression: {
    ratio: number;
    speed: number;
    quality: number;
  };
  similarity: {
    speed: number;
    accuracy: number;
    optimizationLevel: number;
  };
}
