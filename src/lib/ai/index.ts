// Barrel export file for AI types and modules
// This centralizes all AI-related type exports for easy importing

// Re-export all types from types.ts
export * from './types';

// Re-export specific modules for convenience
export * from './enhanced-self-prompting-engine';
export * from './extended-thinking-pipeline';
export * from './langchain-service';

// Type aliases for commonly used types
export type {
  OllamaResponse,
  OllamaEmbedding,
  ModelInfo,
  Document,
  SearchResult,
  ChatMessage,
  LegalCase,
  EmbeddingSearchOptions,
  CacheEntry,
  CacheStrategy,
  CacheMetrics,
  CacheConfiguration,
  CacheLayer,
  CachePolicy,
  CacheStats,
  CacheAnalytics,
  SelfPromptOptions,
  DatabaseOptimization,
  ResourceOptimization,
  OptimizationConfig,
  MetricsHistory,
  BottleneckAnalysis,
  WorkflowTrigger,
  TriggerCondition,
  TriggerEvent,
  WorkflowExecution,
  WorkflowStep,
  SystemAlert,
  TriggerConfig,
  AutomationRule,
  AutomationAction,
  TaskRoute,
  RoutingDecision,
  TaskFeatures,
  ServiceEndpoint,
  RoutingModel,
  TaskMetrics,
  RoutingConfig,
  ServiceCapacity,
  RoutingAnalytics,
  PromptContext,
  SynthesisResult,
  SelfPromptResult,
  GPUWorkload,
  GPUWorkloadQueue,
  BatchProcessingJob,
  WorkloadPriority,
  GPUResourceAllocation,
  WorkloadSchedulingStrategy,
  GPUWorkloadManagerConfig,
  AutomationConfig,
  ResourceMetrics,
  ServiceHealth,
  ServerHealth
} from './types';