// Type definitions for AI services

export interface OllamaResponse {
  model: string;
  created_at: string;
  response: string;
  done: boolean;
  context?: number[];
  total_duration?: number;
  load_duration?: number;
  prompt_eval_duration?: number;
  eval_duration?: number;
  eval_count?: number;
}

export interface OllamaEmbedding {
  embedding: number[];
}

export interface ModelInfo {
  name: string;
  modified_at: string;
  size: number;
  digest: string;
  details: {
    format: string;
    family: string;
    families: string[] | null;
    parameter_size: string;
    quantization_level: string;
  };
}

export interface Document {
  id: string;
  content: string;
  metadata: {
    title?: string;
    type?: string;
    source?: string;
    created_at?: Date;
    tags?: string[];
    [key: string]: any;
  };
  embedding?: number[];
}

export interface SearchResult {
  document: Document;
  score: number;
  highlights?: string[];
}

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp?: Date;
}

export interface LegalCase {
  id: string;
  title: string;
  description: string;
  client_name?: string;
  case_type: string;
  status: 'active' | 'closed' | 'pending';
  created_at: Date;
  updated_at: Date;
  documents: Document[];
  notes?: string;
  tags: string[];
}

export interface EmbeddingSearchOptions {
  limit?: number;
  threshold?: number;
  filter?: Record<string, any>;
}

// Cache-related types
export interface CacheEntry<T = any> {
  key: string;
  value: T;
  timestamp: number;
  expiresAt: number;
  accessCount: number;
  lastAccessed: number;
  size: number;
  tags?: string[];
}

export interface CacheStrategy {
  name: 'lru' | 'ttl' | 'size' | 'hybrid';
  maxSize?: number;
  maxAge?: number;
  checkPeriod?: number;
  priority?: number;
}

export interface CacheMetrics {
  hits: number;
  misses: number;
  hitRate: number;
  evictions: number;
  totalRequests: number;
  totalSizeBytes: number;
  averageResponseTime: number;
  lastCleanup: number;
}

export interface CacheConfiguration {
  strategies: CacheStrategy[];
  defaultTtl: number;
  maxMemoryUsage: number;
  cleanupInterval: number;
  persistToDisk: boolean;
  compression: boolean;
  encryption?: {
    enabled: boolean;
    algorithm: string;
  };
}

export interface CacheLayer {
  name: string;
  priority: number;
  strategy: CacheStrategy;
  enabled: boolean;
  statistics: CacheMetrics;
}

export type CacheKey = string;
export type CacheValue = any;

export interface CachePolicy {
  allowStale: boolean;
  refreshAhead: boolean;
  refreshThreshold: number;
  validateOnGet: boolean;
  serializeValues: boolean;
}

export interface CacheStats {
  layers: CacheLayer[];
  globalMetrics: CacheMetrics;
  memoryUsage: {
    used: number;
    total: number;
    percentage: number;
  };
  diskUsage?: {
    used: number;
    total: number;
    percentage: number;
  };
}

export interface CacheAnalytics {
  popularKeys: Array<{
    key: string;
    accessCount: number;
    lastAccessed: number;
  }>;
  performanceMetrics: {
    averageGetTime: number;
    averageSetTime: number;
    slowestOperations: Array<{
      operation: string;
      duration: number;
      timestamp: number;
    }>;
  };
  recommendations: Array<{
    type: 'optimization' | 'configuration' | 'cleanup';
    message: string;
    priority: 'low' | 'medium' | 'high';
  }>;
}

// Self-prompting engine types
export interface SelfPromptOptions {
  useMultiAgent: boolean;
  useGPU: boolean;
  useCUDA: boolean;
  parallelProcessing: boolean;
  maxAgents: number;
  timeout: number;
  retryAttempts: number;
  priority: 'low' | 'medium' | 'high';
  realTimeUpdates: boolean;
  enableCaching: boolean;
  cacheStrategy: string;
  contextWindow: number;
  temperature: number;
  maxProcessingTime: number;
  enableSelfReflection: boolean;
  iterativeImprovement: boolean;
  qualityThreshold: number;
  diversityBoost: number;
}

// Performance optimization types
export interface DatabaseOptimization {
  indexOptimization: boolean;
  queryOptimization: boolean;
  connectionPooling: {
    enabled: boolean;
    maxConnections: number;
    minConnections: number;
  };
  caching: {
    enabled: boolean;
    ttl: number;
    strategy: string;
  };
}

export interface ResourceOptimization {
  cpu: {
    maxUsage: number;
    scaling: boolean;
    cores: number;
  };
  memory: {
    maxUsage: number;
    gcOptimization: boolean;
    heapSize: number;
  };
  gpu: {
    enabled: boolean;
    maxUsage: number;
    memoryLimit: number;
  };
}

export interface OptimizationConfig {
  database: DatabaseOptimization;
  resources: ResourceOptimization;
  caching: CacheStrategy;
  monitoring: {
    enabled: boolean;
    interval: number;
    metrics: string[];
  };
}

export interface MetricsHistory {
  timestamp: number;
  cpuUsage: number;
  memoryUsage: number;
  responseTime: number;
  throughput: number;
  errorRate: number;
  customMetrics: Record<string, number>;
}

export interface BottleneckAnalysis {
  type: 'cpu' | 'memory' | 'io' | 'network' | 'database';
  severity: 'low' | 'medium' | 'high' | 'critical';
  description: string;
  impact: number;
  recommendations: string[];
  estimatedImprovement: number;
}

// Workflow and trigger types
export interface WorkflowTrigger {
  id: string;
  name: string;
  type: 'event' | 'schedule' | 'webhook' | 'manual';
  enabled: boolean;
  conditions: TriggerCondition[];
  actions: string[];
  priority: number;
}

export interface TriggerCondition {
  field: string;
  operator: 'equals' | 'contains' | 'greater' | 'less' | 'exists';
  value: any;
  metadata?: Record<string, any>;
}

export interface TriggerEvent {
  id: string;
  triggerId: string;
  type: string;
  payload: Record<string, any>;
  timestamp: number;
  processed: boolean;
}

export interface WorkflowExecution {
  id: string;
  workflowId: string;
  triggeredBy: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  startTime: number;
  endTime?: number;
  steps: WorkflowStep[];
  result?: any;
  error?: string;
}

export interface WorkflowStep {
  id: string;
  name: string;
  type: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  input: any;
  output?: any;
  error?: string;
  duration?: number;
}

export interface SystemAlert {
  id: string;
  level: 'info' | 'warning' | 'error' | 'critical';
  title: string;
  message: string;
  timestamp: number;
  source: string;
  metadata?: Record<string, any>;
  acknowledged: boolean;
}

export interface TriggerConfig {
  maxConcurrentExecutions: number;
  retryAttempts: number;
  retryDelay: number;
  timeout: number;
  enableLogging: boolean;
  enableMetrics: boolean;
}

export interface AutomationRule {
  id: string;
  name: string;
  description: string;
  trigger: WorkflowTrigger;
  actions: AutomationAction[];
  enabled: boolean;
  priority: number;
}

export interface AutomationAction {
  id: string;
  type: string;
  config: Record<string, any>;
  retry: {
    enabled: boolean;
    maxAttempts: number;
    delay: number;
  };
}

// Task routing types
export interface TaskRoute {
  id: string;
  pattern: string;
  service: string;
  endpoint: string;
  method: 'GET' | 'POST' | 'PUT' | 'DELETE';
  priority: number;
  enabled: boolean;
}

export interface RoutingDecision {
  selectedRoute: TaskRoute;
  confidence: number;
  alternativeRoutes: TaskRoute[];
  reasoningSteps: string[];
  executionTime: number;
}

export interface TaskFeatures {
  complexity: number;
  dataSize: number;
  requestType: string;
  userContext: Record<string, any>;
  performance: {
    expectedLatency: number;
    resourceRequirements: number;
  };
}

export interface ServiceEndpoint {
  id: string;
  name: string;
  url: string;
  method: string;
  capacity: ServiceCapacity;
  healthStatus: 'healthy' | 'degraded' | 'unhealthy';
  averageResponseTime: number;
}

export interface RoutingModel {
  name: string;
  version: string;
  accuracy: number;
  trainingData: {
    size: number;
    lastUpdated: number;
  };
  features: string[];
}

export interface TaskMetrics {
  totalRequests: number;
  successfulRoutes: number;
  failedRoutes: number;
  averageRoutingTime: number;
  accuracyRate: number;
}

export interface RoutingConfig {
  algorithm: 'roundRobin' | 'weightedRandom' | 'leastConnections' | 'ml';
  fallbackStrategy: 'retry' | 'alternative' | 'queue';
  healthCheckInterval: number;
  loadBalancing: boolean;
}

export interface ServiceCapacity {
  maxConcurrentRequests: number;
  currentLoad: number;
  resourceUtilization: {
    cpu: number;
    memory: number;
    network: number;
  };
}

export interface RoutingAnalytics {
  routingDecisions: RoutingDecision[];
  servicePerformance: ServiceEndpoint[];
  modelMetrics: {
    accuracy: number;
    precision: number;
    recall: number;
    f1Score: number;
  };
  recommendations: string[];
}

export interface PromptContext {
  originalPrompt: string;
  processedPrompt: string;
  metadata: Record<string, any>;
  timestamp: number;
  sessionId: string;
  userId?: string;
  processingTime: number;
  complexity: number;
  confidence: number;
  tags: string[];
}

export interface SynthesisResult {
  synthesizedResponse: string;
  confidence: number;
  sources: string[];
  methodology: string;
  processingTime: number;
  qualityScore: number;
  recommendations: Array<{
    type: string;
    description: string;
    priority: 'low' | 'medium' | 'high';
  }>;
  nextActions: string[];
}

export interface SelfPromptResult {
  response: string;
  confidence: number;
  processingTime: number;
  agentResults: Array<{
    agentType: string;
    response: string;
    confidence: number;
    processingTime: number;
  }>;
  synthesis: SynthesisResult;
  metadata: {
    sessionId: string;
    timestamp: number;
    options: SelfPromptOptions;
    metrics: {
      totalTime: number;
      agentProcessingTime: number;
      contextProcessingTime: number;
      synthesisTime: number;
      cacheHits: number;
      cacheMisses: number;
    };
  };
  recommendations: string[];
  nextActions: string[];
}

// GPU Workload Types
export interface GPUWorkload {
  id: string;
  type: string;
  priority: WorkloadPriority;
  data: any;
  requirements?: {
    memoryMB?: number;
    computeUnits?: number;
    maxDurationMs?: number;
  };
  timestamp: number;
  status: 'pending' | 'processing' | 'completed' | 'error';
}

export interface GPUWorkloadQueue {
  id: string;
  workloads: GPUWorkload[];
  capacity: number;
  processing: number;
  completed: number;
  failed: number;
}

export interface BatchProcessingJob {
  id: string;
  workloads: GPUWorkload[];
  batchSize: number;
  status: 'pending' | 'processing' | 'completed' | 'error';
  startTime?: number;
  endTime?: number;
  results?: any[];
}

export type WorkloadPriority = 'low' | 'medium' | 'high' | 'critical';

// Additional missing exports for GPU workload management

export interface GPUResourceAllocation {
  deviceId: number;
  memoryMB: number;
  computeUnits: number;
  priority: WorkloadPriority;
}

export interface WorkloadSchedulingStrategy {
  type: 'fifo' | 'priority' | 'round_robin' | 'weighted';
  config: Record<string, any>;
}

export interface GPUWorkloadManagerConfig {
  maxQueueSize: number;
  batchOptimizationEnabled: boolean;
  adaptiveScheduling: boolean;
  priorityQueues: boolean;
  resourcePreallocation: boolean;
  cacheIntegration: boolean;
  routingIntegration: boolean;
  performanceMonitoring: boolean;
  autoScaling: boolean;
  queueTimeoutMs: number;
  batchTimeoutMs: number;
  maxBatchSize: number;
  schedulingInterval: number;
}

// Additional missing types for automation
export interface AutomationConfig {
  enabled: boolean;
  rules: AutomationRule[];
  triggers: WorkflowTrigger[];
  monitoring: boolean;
  retryAttempts: number;
  timeout: number;
}

export interface ResourceMetrics {
  cpu: number;
  memory: number;
  disk: number;
  network: number;
  timestamp: Date;
}

export interface ServiceHealth {
  status: 'healthy' | 'degraded' | 'unhealthy' | 'unknown';
  lastCheck: Date;
  responseTime: number;
  errorRate: number;
  uptime: number;
}

export interface ServerHealth {
  status: 'healthy' | 'degraded' | 'unhealthy' | 'unknown';
  lastCheck: Date;
  responseTime: number;
  errorRate: number;
  uptime: number;
}

