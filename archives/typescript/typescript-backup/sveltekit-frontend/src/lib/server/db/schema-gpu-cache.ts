/**
 * GPU Shader Cache Database Schema - PostgreSQL + pgvector Integration
 * Supports reinforcement learning, predictive preloading, and multi-dimensional recall
 * Optimized for legal document visualization AI workflows
 */

import { 
  pgTable, 
  serial, 
  text, 
  jsonb, 
  timestamp, 
  boolean, 
  integer, 
  real,
  uuid,
  index,
  foreignKey
} from 'drizzle-orm/pg-core';
import { vector } from 'pgvector/drizzle-orm';
import { sql } from 'drizzle-orm';

// ============================================================================
// CORE SHADER CACHE TABLES
// ============================================================================

/**
 * Primary shader cache entries with source code, compiled binaries, and metadata
 */
export const shaderCacheEntries = pgTable('shader_cache_entries', {
  id: uuid('id').defaultRandom().primaryKey(),
  
  // Core identification
  shaderKey: text('shader_key').notNull().unique(), // Human-readable cache key
  shaderHash: text('shader_hash').notNull().unique(), // Content hash for integrity
  shaderType: text('shader_type').notNull(), // 'vertex', 'fragment', 'compute', 'wgsl', 'glsl'
  
  // Source code and compilation
  sourceCode: text('source_code').notNull(), // Original WGSL/GLSL source
  compiledBinary: text('compiled_binary'), // Base64 encoded compiled binary (if supported)
  compilationLog: text('compilation_log'), // Compiler output/warnings
  compilationSuccess: boolean('compilation_success').default(true),
  
  // Semantic embeddings for similarity search
  sourceEmbedding: vector('source_embedding', { dimensions: 384 }), // nomic-embed-text
  semanticTags: text('semantic_tags').array(), // ['legal-doc', 'timeline', 'evidence']
  
  // Legal workflow context
  legalContext: jsonb('legal_context'),
  
  // Performance metrics
  performanceMetrics: jsonb('performance_metrics'),
  
  // Reinforcement learning data
  reinforcementData: jsonb('reinforcement_data'),
  
  // Version and lifecycle
  version: integer('version').default(1),
  deprecated: boolean('deprecated').default(false),
  createdAt: timestamp('created_at', { withTimezone: true }).defaultNow(),
  updatedAt: timestamp('updated_at', { withTimezone: true }).defaultNow(),
  lastAccessedAt: timestamp('last_accessed_at', { withTimezone: true }),
  
  // MinIO integration for large assets
  minioPath: text('minio_path'), // Optional path for large shader assets
  assetBundle: jsonb('asset_bundle')
}, (table) => ({
  // Primary indexes for fast retrieval
  shaderKeyIdx: index('shader_key_idx').on(table.shaderKey),
  shaderHashIdx: index('shader_hash_idx').on(table.shaderHash),
  shaderTypeIdx: index('shader_type_idx').on(table.shaderType),
  
  // Semantic search indexes (pgvector HNSW)
  sourceEmbeddingIdx: index('source_embedding_hnsw_idx')
    .using('hnsw', table.sourceEmbedding.op('vector_cosine_ops')),
  
  // Performance and usage indexes
  lastAccessedIdx: index('last_accessed_idx').on(table.lastAccessedAt),
  usageCountIdx: index('usage_count_idx').on(sql`(reinforcement_data->>'usageCount')::integer`),
  successRateIdx: index('success_rate_idx').on(sql`(reinforcement_data->>'successRate')::real`),
  
  // Legal context indexes (GIN for jsonb)
  legalContextIdx: index('legal_context_gin_idx').using('gin', table.legalContext),
  semanticTagsIdx: index('semantic_tags_gin_idx').using('gin', table.semanticTags)
}));

/**
 * User shader access patterns for predictive preloading
 */
export const shaderUserPatterns = pgTable('shader_user_patterns', {
  id: serial('id').primaryKey(),
  
  // User and session context
  userId: text('user_id').notNull(), // From your auth system
  sessionId: text('session_id').notNull(),
  clientFingerprint: text('client_fingerprint'), // Browser/GPU fingerprint
  
  // Shader access tracking
  shaderId: uuid('shader_id').references(() => shaderCacheEntries.id),
  shaderKey: text('shader_key').notNull(),
  
  // Workflow context
  workflowStep: text('workflow_step'), // 'doc-load', 'evidence-view', 'timeline', 'analysis'
  previousStep: text('previous_step'),
  nextStepPrediction: text('next_step_prediction'),
  
  // Temporal patterns
  accessTimestamp: timestamp('access_timestamp', { withTimezone: true }).defaultNow(),
  timeOfDay: integer('time_of_day'), // Hour 0-23
  dayOfWeek: integer('day_of_week'), // 0=Sunday
  sessionDuration: integer('session_duration'), // Seconds since session start
  
  // Performance outcomes
  loadLatencyMs: integer('load_latency_ms'),
  cacheHit: boolean('cache_hit'),
  preloadSuccessful: boolean('preload_successful'),
  userSatisfaction: real('user_satisfaction'), // -1 to 1
  
  // Contextual metadata
  documentContext: jsonb('document_context'),
  
  // Reinforcement learning features
  stateVector: vector('state_vector', { dimensions: 64 }), // Compressed workflow state
  actionVector: vector('action_vector', { dimensions: 32 }), // Action embedding
  reward: real('reward'), // Computed reward for this access
  
  createdAt: timestamp('created_at', { withTimezone: true }).defaultNow()
}, (table) => ({
  // User and session indexes
  userIdIdx: index('user_patterns_user_id_idx').on(table.userId),
  sessionIdIdx: index('user_patterns_session_id_idx').on(table.sessionId),
  
  // Temporal indexes for pattern analysis
  accessTimeIdx: index('access_time_idx').on(table.accessTimestamp),
  timeOfDayIdx: index('time_of_day_idx').on(table.timeOfDay),
  workflowStepIdx: index('workflow_step_idx').on(table.workflowStep),
  
  // ML feature indexes
  stateVectorIdx: index('state_vector_hnsw_idx')
    .using('hnsw', table.stateVector.op('vector_cosine_ops')),
  actionVectorIdx: index('action_vector_hnsw_idx')
    .using('hnsw', table.actionVector.op('vector_cosine_ops')),
  
  // Performance indexes
  cacheHitIdx: index('cache_hit_idx').on(table.cacheHit),
  latencyIdx: index('load_latency_idx').on(table.loadLatencyMs),
  rewardIdx: index('reward_idx').on(table.reward)
}));

/**
 * Predictive preloading rules learned by the reinforcement system
 */
export const shaderPreloadRules = pgTable('shader_preload_rules', {
  id: serial('id').primaryKey(),
  
  // Rule identification
  ruleKey: text('rule_key').notNull().unique(),
  ruleName: text('rule_name').notNull(),
  ruleType: text('rule_type').notNull(), // 'sequential', 'conditional', 'temporal', 'similarity'
  
  // Condition matching
  triggerConditions: jsonb('trigger_conditions'),
  
  // Preload specifications
  preloadTargets: jsonb('preload_targets'),
  
  // ML model weights and thresholds
  modelWeights: vector('model_weights', { dimensions: 128 }), // Learned weights
  confidence: real('confidence').notNull(), // 0-1 rule confidence
  accuracy: real('accuracy').notNull(), // Historical accuracy
  
  // Performance metrics
  triggerCount: integer('trigger_count').default(0),
  successCount: integer('success_count').default(0),
  preloadSavingsMs: integer('preload_savings_ms').default(0),
  
  // Lifecycle
  active: boolean('active').default(true),
  learningRate: real('learning_rate').default(0.01),
  lastTriggered: timestamp('last_triggered', { withTimezone: true }),
  lastUpdated: timestamp('last_updated', { withTimezone: true }).defaultNow(),
  createdAt: timestamp('created_at', { withTimezone: true }).defaultNow()
}, (table) => ({
  ruleKeyIdx: index('rule_key_idx').on(table.ruleKey),
  ruleTypeIdx: index('rule_type_idx').on(table.ruleType),
  activeIdx: index('active_rules_idx').on(table.active),
  confidenceIdx: index('confidence_idx').on(table.confidence),
  accuracyIdx: index('accuracy_idx').on(table.accuracy),
  
  // ML model index
  modelWeightsIdx: index('model_weights_hnsw_idx')
    .using('hnsw', table.modelWeights.op('vector_cosine_ops')),
  
  // Performance indexes
  triggerCountIdx: index('trigger_count_idx').on(table.triggerCount),
  lastTriggeredIdx: index('last_triggered_idx').on(table.lastTriggered)
}));

/**
 * Shader dependency graph for intelligent preloading
 */
export const shaderDependencies = pgTable('shader_dependencies', {
  id: serial('id').primaryKey(),
  
  // Dependency relationship
  parentShaderId: uuid('parent_shader_id').references(() => shaderCacheEntries.id),
  childShaderId: uuid('child_shader_id').references(() => shaderCacheEntries.id),
  
  // Dependency metadata
  dependencyType: text('dependency_type').notNull(), // 'include', 'texture', 'uniform', 'buffer'
  dependencyStrength: real('dependency_strength'), // 0-1 how critical this dependency is
  loadOrder: integer('load_order'), // Relative load order
  
  // Performance impact
  parallelLoadSafe: boolean('parallel_load_safe').default(true),
  loadLatencyImpactMs: integer('load_latency_impact_ms'),
  
  // Usage statistics
  coUsageFrequency: real('co_usage_frequency'), // 0-1 how often used together
  lastCoUsed: timestamp('last_co_used', { withTimezone: true }),
  
  createdAt: timestamp('created_at', { withTimezone: true }).defaultNow(),
  updatedAt: timestamp('updated_at', { withTimezone: true }).defaultNow()
}, (table) => ({
  parentChildIdx: index('parent_child_idx').on(table.parentShaderId, table.childShaderId),
  dependencyTypeIdx: index('dependency_type_idx').on(table.dependencyType),
  dependencyStrengthIdx: index('dependency_strength_idx').on(table.dependencyStrength),
  loadOrderIdx: index('load_order_idx').on(table.loadOrder),
  coUsageFrequencyIdx: index('co_usage_frequency_idx').on(table.coUsageFrequency)
}));

/**
 * Real-time shader compilation queue for background processing
 */
export const shaderCompilationQueue = pgTable('shader_compilation_queue', {
  id: serial('id').primaryKey(),
  
  // Queue identification
  queueKey: text('queue_key').notNull().unique(),
  priority: text('priority').notNull(), // 'immediate', 'high', 'normal', 'low', 'preload'
  status: text('status').notNull(), // 'pending', 'processing', 'completed', 'failed', 'cancelled'
  
  // Shader information
  shaderKey: text('shader_key').notNull(),
  sourceCode: text('source_code').notNull(),
  shaderType: text('shader_type').notNull(),
  targetGPU: text('target_gpu'), // GPU-specific compilation
  
  // Processing context
  userId: text('user_id'),
  sessionId: text('session_id'),
  workflowContext: jsonb('workflow_context'),
  
  // Queue timing
  queuedAt: timestamp('queued_at', { withTimezone: true }).defaultNow(),
  startedAt: timestamp('started_at', { withTimezone: true }),
  completedAt: timestamp('completed_at', { withTimezone: true }),
  
  // Processing results
  compilationResult: jsonb('compilation_result'),
  
  // Retry logic
  retryCount: integer('retry_count').default(0),
  maxRetries: integer('max_retries').default(3),
  nextRetryAt: timestamp('next_retry_at', { withTimezone: true }),
  
  updatedAt: timestamp('updated_at', { withTimezone: true }).defaultNow()
}, (table) => ({
  queueKeyIdx: index('queue_key_idx').on(table.queueKey),
  statusIdx: index('queue_status_idx').on(table.status),
  priorityIdx: index('queue_priority_idx').on(table.priority),
  queuedAtIdx: index('queued_at_idx').on(table.queuedAt),
  userSessionIdx: index('user_session_queue_idx').on(table.userId, table.sessionId),
  retryIdx: index('retry_idx').on(table.nextRetryAt),
  
  // Composite indexes for queue processing
  statusPriorityIdx: index('status_priority_idx').on(table.status, table.priority),
  pendingQueueIdx: index('pending_queue_idx').on(table.status, table.priority, table.queuedAt)
}));

// ============================================================================
// VIEWS FOR COMMON QUERIES
// ============================================================================

/**
 * Materialized view for fast shader recommendations
 * Updated periodically by background jobs
 */
export const shaderRecommendationsView = pgTable('shader_recommendations_view', {
  id: serial('id').primaryKey(),
  userId: text('user_id').notNull(),
  shaderKey: text('shader_key').notNull(),
  recommendationType: text('recommendation_type').notNull(), // 'similar', 'next', 'popular', 'optimal'
  confidence: real('confidence').notNull(),
  reasoning: text('reasoning'),
  
  // Recommendation context
  baseContext: jsonb('base_context'), // Context that triggered this recommendation
  expectedBenefit: real('expected_benefit'), // Expected performance/satisfaction improvement
  
  // Metadata
  computedAt: timestamp('computed_at', { withTimezone: true }).defaultNow(),
  validUntil: timestamp('valid_until', { withTimezone: true }),
  
  // Performance tracking
  timesRecommended: integer('times_recommended').default(0),
  timesAccepted: integer('times_accepted').default(0),
  averageUserSatisfaction: real('average_user_satisfaction')
}, (table) => ({
  userRecommendationIdx: index('user_recommendation_idx').on(table.userId, table.recommendationType),
  confidenceIdx: index('recommendation_confidence_idx').on(table.confidence),
  validityIdx: index('recommendation_validity_idx').on(table.validUntil),
  shaderKeyIdx: index('recommendation_shader_key_idx').on(table.shaderKey)
}));

// ============================================================================
// EXPORT TYPES FOR TYPESCRIPT
// ============================================================================

export type ShaderCacheEntry = typeof shaderCacheEntries.$inferSelect;
export type InsertShaderCacheEntry = typeof shaderCacheEntries.$inferInsert;

export type ShaderUserPattern = typeof shaderUserPatterns.$inferSelect;
export type InsertShaderUserPattern = typeof shaderUserPatterns.$inferInsert;

export type ShaderPreloadRule = typeof shaderPreloadRules.$inferSelect;
export type InsertShaderPreloadRule = typeof shaderPreloadRules.$inferInsert;

export type ShaderDependency = typeof shaderDependencies.$inferSelect;
export type InsertShaderDependency = typeof shaderDependencies.$inferInsert;

export type ShaderCompilationQueue = typeof shaderCompilationQueue.$inferSelect;
export type InsertShaderCompilationQueue = typeof shaderCompilationQueue.$inferInsert;

export type ShaderRecommendation = typeof shaderRecommendationsView.$inferSelect;
export type InsertShaderRecommendation = typeof shaderRecommendationsView.$inferInsert;