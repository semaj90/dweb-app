/**
 * GPU Shader Cache Schema with Reinforcement Learning
 * PostgreSQL + pgvector integration for legal visualization AI
 */

import {
  pgTable,
  text,
  timestamp,
  uuid,
  integer,
  jsonb,
  boolean,
  decimal,
  varchar,
  index,
  primaryKey,
  relations
} from "drizzle-orm/pg-core";

// === Shader Cache Tables ===

// Main shader cache entries
export const shaderCacheEntries = pgTable("shader_cache_entries", {
  id: uuid("id").primaryKey().defaultRandom(),
  cacheKey: varchar("cache_key", { length: 255 }).notNull().unique(),
  shaderType: varchar("shader_type", { length: 50 }).notNull(), // 'vertex', 'fragment', 'compute', 'geometry'
  
  // Shader source and metadata
  sourceCode: text("source_code").notNull(), // WGSL/GLSL source
  shaderLanguage: varchar("shader_language", { length: 20 }).notNull(), // 'wgsl', 'glsl', 'hlsl'
  shaderVersion: varchar("shader_version", { length: 50 }), // shader model version
  
  // Compiled binaries (stored in MinIO, referenced here)
  compiledBinaryPath: text("compiled_binary_path"), // MinIO object path
  compiledBinarySize: integer("compiled_binary_size"), // bytes
  compilationTime: decimal("compilation_time", { precision: 8, scale: 3 }), // milliseconds
  
  // Context and metadata
  legalContext: varchar("legal_context", { length: 100 }), // 'case', 'evidence', 'timeline', 'graph'
  visualizationType: varchar("visualization_type", { length: 50 }), // 'scatter', 'network', 'heatmap', 'timeline'
  complexity: integer("complexity").default(0), // shader complexity score 0-100
  
  // Vector embedding for semantic similarity (384 dimensions for nomic-embed)
  embedding: text("embedding"), // pgvector embedding stored as text
  
  // Performance metrics
  averageRenderTime: decimal("avg_render_time", { precision: 8, scale: 3 }), // microseconds
  memoryFootprint: integer("memory_footprint"), // bytes
  gpuUtilization: decimal("gpu_utilization", { precision: 5, scale: 4 }), // 0.0-1.0
  
  // Usage statistics for reinforcement learning
  accessCount: integer("access_count").default(0),
  lastAccessedAt: timestamp("last_accessed_at"),
  
  // Reinforcement learning metrics
  reinforcementScore: decimal("reinforcement_score", { precision: 7, scale: 4 }).default("0.5"), // 0.0-1.0
  rewardHistory: jsonb("reward_history").default("[]"), // Array of rewards over time
  
  // Tags and categorization
  semanticTags: jsonb("semantic_tags").default("[]"), // AI-generated tags
  userTags: jsonb("user_tags").default("[]"), // User-defined tags
  
  // Metadata and dependencies
  dependencies: jsonb("dependencies").default("[]"), // Other shader dependencies
  parameters: jsonb("parameters").default("{}"), // Shader uniform parameters
  metadata: jsonb("metadata").default("{}"),
  
  // Audit fields
  createdBy: uuid("created_by"), // references users.id
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull()
}, (table) => ({
  // Indexes for performance
  cacheKeyIdx: index("shader_cache_key_idx").on(table.cacheKey),
  shaderTypeIdx: index("shader_type_idx").on(table.shaderType),
  contextIdx: index("legal_context_idx").on(table.legalContext),
  visualizationIdx: index("visualization_type_idx").on(table.visualizationType),
  accessCountIdx: index("access_count_idx").on(table.accessCount),
  reinforcementScoreIdx: index("reinforcement_score_idx").on(table.reinforcementScore),
  lastAccessedIdx: index("last_accessed_idx").on(table.lastAccessedAt),
  
  // pgvector index for semantic similarity
  embeddingIdx: index("shader_embedding_hnsw_idx").using("hnsw", table.embedding).with({ 
    m: 16, 
    ef_construction: 64 
  })
}));

// User shader access patterns for reinforcement learning
export const shaderUserPatterns = pgTable("shader_user_patterns", {
  id: uuid("id").primaryKey().defaultRandom(),
  userId: uuid("user_id").notNull(), // references users.id
  shaderCacheId: uuid("shader_cache_id").references(() => shaderCacheEntries.id).notNull(),
  
  // Access context
  sessionId: varchar("session_id", { length: 100 }),
  accessTimestamp: timestamp("access_timestamp").defaultNow().notNull(),
  
  // User workflow context
  workflowStep: varchar("workflow_step", { length: 50 }), // 'document_upload', 'evidence_review', 'case_analysis'
  previousShaders: jsonb("previous_shaders").default("[]"), // Shader sequence for workflow prediction
  nextShaders: jsonb("next_shaders").default("[]"), // Actual next shaders used (for learning)
  
  // Performance and satisfaction metrics
  renderQuality: decimal("render_quality", { precision: 3, scale: 2 }), // 1.0-5.0 user rating
  performanceSatisfaction: decimal("performance_satisfaction", { precision: 3, scale: 2 }), // 1.0-5.0
  userEngagementTime: integer("user_engagement_time"), // seconds spent with this shader active
  
  // Context vectors for correlation
  documentType: varchar("document_type", { length: 50 }), // 'contract', 'evidence', 'brief', 'statute'
  caseComplexity: integer("case_complexity"), // 1-10 scale
  dataSize: integer("data_size"), // number of objects being visualized
  
  // Reinforcement learning features
  reward: decimal("reward", { precision: 7, scale: 4 }), // calculated reward for this access
  prediction: jsonb("prediction").default("{}"), // ML model prediction data
  actualOutcome: jsonb("actual_outcome").default("{}"), // actual user behavior for training
  
  metadata: jsonb("metadata").default("{}")
}, (table) => ({
  // Indexes for ML queries
  userIdIdx: index("user_patterns_user_id_idx").on(table.userId),
  shaderCacheIdIdx: index("user_patterns_shader_id_idx").on(table.shaderCacheId),
  sessionIdx: index("user_patterns_session_idx").on(table.sessionId),
  workflowStepIdx: index("workflow_step_idx").on(table.workflowStep),
  accessTimeIdx: index("access_timestamp_idx").on(table.accessTimestamp),
  rewardIdx: index("reward_idx").on(table.reward),
  
  // Composite index for workflow prediction
  workflowPredictionIdx: index("workflow_prediction_idx").on(table.userId, table.workflowStep, table.accessTimestamp)
}));

// Shader dependency graph for optimization
export const shaderDependencies = pgTable("shader_dependencies", {
  id: uuid("id").primaryKey().defaultRandom(),
  parentShaderCacheId: uuid("parent_shader_cache_id").references(() => shaderCacheEntries.id).notNull(),
  dependentShaderCacheId: uuid("dependent_shader_cache_id").references(() => shaderCacheEntries.id).notNull(),
  
  dependencyType: varchar("dependency_type", { length: 30 }).notNull(), // 'uniform', 'texture', 'buffer', 'include'
  dependencyStrength: decimal("dependency_strength", { precision: 5, scale: 4 }), // 0.0-1.0 correlation
  
  // Performance impact
  loadOrderPriority: integer("load_order_priority").default(100), // lower = load first
  parallelizable: boolean("parallelizable").default(true),
  
  createdAt: timestamp("created_at").defaultNow().notNull(),
  metadata: jsonb("metadata").default("{}")
}, (table) => ({
  // Unique constraint and indexes
  uniqueDependency: primaryKey({ columns: [table.parentShaderCacheId, table.dependentShaderCacheId] }),
  parentIdx: index("parent_shader_idx").on(table.parentShaderCacheId),
  dependentIdx: index("dependent_shader_idx").on(table.dependentShaderCacheId),
  dependencyTypeIdx: index("dependency_type_idx").on(table.dependencyType),
  priorityIdx: index("load_order_priority_idx").on(table.loadOrderPriority)
}));

// Predictive preload cache for reinforcement learning
export const shaderPreloadQueue = pgTable("shader_preload_queue", {
  id: uuid("id").primaryKey().defaultRandom(),
  userId: uuid("user_id").notNull(), // references users.id
  shaderCacheId: uuid("shader_cache_id").references(() => shaderCacheEntries.id).notNull(),
  
  // Prediction metadata
  predictionConfidence: decimal("prediction_confidence", { precision: 5, scale: 4 }).notNull(), // 0.0-1.0
  predictedAtTimestamp: timestamp("predicted_at_timestamp").defaultNow().notNull(),
  predictionModel: varchar("prediction_model", { length: 50 }), // 'workflow_sequence', 'collaborative_filter', 'content_based'
  
  // Context for prediction
  currentWorkflowStep: varchar("current_workflow_step", { length: 50 }),
  userSessionContext: jsonb("user_session_context").default("{}"),
  legalCaseContext: jsonb("legal_case_context").default("{}"),
  
  // Priority and scheduling
  preloadPriority: integer("preload_priority").default(50), // 0-100, higher = more urgent
  scheduledFor: timestamp("scheduled_for"), // when to preload
  
  // Status tracking
  status: varchar("status", { length: 20 }).default("pending"), // 'pending', 'loading', 'loaded', 'expired', 'cancelled'
  preloadStartedAt: timestamp("preload_started_at"),
  preloadCompletedAt: timestamp("preload_completed_at"),
  
  // Accuracy tracking for model improvement
  wasUsed: boolean("was_used"), // did user actually use this shader?
  usedAtTimestamp: timestamp("used_at_timestamp"),
  actualDelay: integer("actual_delay"), // ms between preload and actual use
  
  metadata: jsonb("metadata").default("{}")
}, (table) => ({
  userIdIdx: index("preload_user_id_idx").on(table.userId),
  shaderCacheIdIdx: index("preload_shader_id_idx").on(table.shaderCacheId),
  confidenceIdx: index("prediction_confidence_idx").on(table.predictionConfidence),
  priorityIdx: index("preload_priority_idx").on(table.preloadPriority),
  statusIdx: index("preload_status_idx").on(table.status),
  scheduledForIdx: index("scheduled_for_idx").on(table.scheduledFor),
  
  // Composite indexes for queue processing
  queueProcessingIdx: index("queue_processing_idx").on(table.status, table.scheduledFor, table.preloadPriority),
  accuracyTrackingIdx: index("accuracy_tracking_idx").on(table.wasUsed, table.predictionModel)
}));

// Shader performance metrics for monitoring
export const shaderPerformanceMetrics = pgTable("shader_performance_metrics", {
  id: uuid("id").primaryKey().defaultRandom(),
  shaderCacheId: uuid("shader_cache_id").references(() => shaderCacheEntries.id).notNull(),
  userId: uuid("user_id"), // references users.id, null for system-wide metrics
  
  // Timing metrics
  compilationTimeMs: decimal("compilation_time_ms", { precision: 8, scale: 3 }),
  firstRenderTimeMs: decimal("first_render_time_ms", { precision: 8, scale: 3 }),
  averageRenderTimeMs: decimal("average_render_time_ms", { precision: 8, scale: 3 }),
  
  // Resource usage
  gpuMemoryUsageBytes: integer("gpu_memory_usage_bytes"),
  cpuUsagePercent: decimal("cpu_usage_percent", { precision: 5, scale: 2 }),
  gpuUtilizationPercent: decimal("gpu_utilization_percent", { precision: 5, scale: 2 }),
  
  // Quality metrics
  frameRate: decimal("frame_rate", { precision: 6, scale: 2 }), // FPS
  qualityScore: decimal("quality_score", { precision: 5, scale: 4 }), // 0.0-1.0 calculated quality
  errorCount: integer("error_count").default(0),
  
  // Context
  deviceInfo: jsonb("device_info").default("{}"), // GPU model, driver version, etc.
  renderContext: jsonb("render_context").default("{}"), // resolution, complexity settings
  
  recordedAt: timestamp("recorded_at").defaultNow().notNull(),
  metadata: jsonb("metadata").default("{}")
}, (table) => ({
  shaderCacheIdIdx: index("perf_shader_id_idx").on(table.shaderCacheId),
  userIdIdx: index("perf_user_id_idx").on(table.userId),
  recordedAtIdx: index("perf_recorded_at_idx").on(table.recordedAt),
  qualityScoreIdx: index("quality_score_idx").on(table.qualityScore),
  frameRateIdx: index("frame_rate_idx").on(table.frameRate),
  
  // Time series analysis
  timeSeriesIdx: index("perf_time_series_idx").on(table.shaderCacheId, table.recordedAt)
}));

// === Relations ===

export const shaderCacheEntriesRelations = relations(shaderCacheEntries, ({ many }) => ({
  userPatterns: many(shaderUserPatterns),
  dependencies: many(shaderDependencies, { relationName: "parent_dependencies" }),
  dependents: many(shaderDependencies, { relationName: "dependent_shaders" }),
  preloadQueue: many(shaderPreloadQueue),
  performanceMetrics: many(shaderPerformanceMetrics)
}));

export const shaderUserPatternsRelations = relations(shaderUserPatterns, ({ one }) => ({
  shaderCache: one(shaderCacheEntries, {
    fields: [shaderUserPatterns.shaderCacheId],
    references: [shaderCacheEntries.id]
  })
}));

export const shaderDependenciesRelations = relations(shaderDependencies, ({ one }) => ({
  parentShader: one(shaderCacheEntries, {
    fields: [shaderDependencies.parentShaderCacheId],
    references: [shaderCacheEntries.id],
    relationName: "parent_dependencies"
  }),
  dependentShader: one(shaderCacheEntries, {
    fields: [shaderDependencies.dependentShaderCacheId],
    references: [shaderCacheEntries.id],
    relationName: "dependent_shaders"
  })
}));

export const shaderPreloadQueueRelations = relations(shaderPreloadQueue, ({ one }) => ({
  shaderCache: one(shaderCacheEntries, {
    fields: [shaderPreloadQueue.shaderCacheId],
    references: [shaderCacheEntries.id]
  })
}));

export const shaderPerformanceMetricsRelations = relations(shaderPerformanceMetrics, ({ one }) => ({
  shaderCache: one(shaderCacheEntries, {
    fields: [shaderPerformanceMetrics.shaderCacheId],
    references: [shaderCacheEntries.id]
  })
}));

// === TypeScript Types ===

export type ShaderCacheEntry = typeof shaderCacheEntries.$inferSelect;
export type NewShaderCacheEntry = typeof shaderCacheEntries.$inferInsert;
export type ShaderUserPattern = typeof shaderUserPatterns.$inferSelect;
export type NewShaderUserPattern = typeof shaderUserPatterns.$inferInsert;
export type ShaderDependency = typeof shaderDependencies.$inferSelect;
export type NewShaderDependency = typeof shaderDependencies.$inferInsert;
export type ShaderPreloadQueue = typeof shaderPreloadQueue.$inferSelect;
export type NewShaderPreloadQueue = typeof shaderPreloadQueue.$inferInsert;
export type ShaderPerformanceMetric = typeof shaderPerformanceMetrics.$inferSelect;
export type NewShaderPerformanceMetric = typeof shaderPerformanceMetrics.$inferInsert;

// === Helper Types for API ===

export interface ShaderCacheConfig {
  maxCacheSize: number;
  preloadThreshold: number;
  reinforcementLearningEnabled: boolean;
  predictivePreloadEnabled: boolean;
}

export interface ShaderPredictionContext {
  userId: string;
  currentWorkflow: string;
  caseContext?: any;
  userHistory?: ShaderUserPattern[];
  sessionContext?: any;
}

export interface ReinforcementReward {
  timestamp: number;
  reward: number;
  context: string;
  performanceBonus: number;
  userSatisfactionScore: number;
}

export interface PredictionResult {
  shaderCacheId: string;
  confidence: number;
  model: string;
  features: Record<string, any>;
  priority: number;
}