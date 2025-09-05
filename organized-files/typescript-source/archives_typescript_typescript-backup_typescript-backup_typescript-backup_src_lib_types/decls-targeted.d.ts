// Targeted module declaration stubs to reduce missing-module/unknown errors
// Keep these minimal and narrow; replace with real types when available.

declare module '$lib/server/schema' {
  const schema: any;
  export = schema;
}

declare module '$lib/server/db/schema-postgres' {
  const pgSchema: any;
  export = pgSchema;
}

declare module '$lib/database/schema/legal-documents' {
  const legalDocuments: any;
  export = legalDocuments;
}

declare module '$lib/services/cognitive-cache-integration' {
  export const cognitiveCache: any;
  export default cognitiveCache;
}

declare module '$lib/services/ollamaService' {
  export const ollamaService: any;
  export default ollamaService;
}

declare module '$lib/server/minio/client' {
  export const minioClient: any;
  export default minioClient;
}

declare module '$lib/ai/types' {
  // Export commonly referenced type names as any so imports typecheck until proper types are added
  export type CacheValue = any;
  export type GPUWorkload = any;
  export type PerformanceMetrics = any;
  export type OptimizationRecommendation = any;
  export type PerformanceAnalysis = any;
  export type AgentPlacement = any;
  export type ScalingMetrics = any;
  export type AgentScalingConfig = any;
  export type ServerHealth = any;
  export type VectorSearchOptions = any;
  export type VectorDB = any;
  export default {} as any;
}

declare module '$lib/ai/ragStreamClient' {
  const ragStreamClient: any;
  export default ragStreamClient;
}

declare module '$lib/ai/llamacpp-service.js' {
  const ll: any;
  export default ll;
}

declare module '$lib/ai/semantic-analysis-pipeline' {
  export const semanticPipeline: any;
  export default semanticPipeline;
}

declare module '$lib/yorha/services/vector.service' {
  export type VectorService = any;
  export const VectorService: any;
  export default VectorService;
}
