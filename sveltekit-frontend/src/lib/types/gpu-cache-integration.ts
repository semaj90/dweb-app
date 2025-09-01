/**
 * Comprehensive TypeScript Types for Enhanced GPU Cache Integration
 * Covers Binary Encoding + GPU Shader Cache + NES Orchestrator + WebGPU + Legal Workflows
 */

// === Core Integration Types ===

export interface EnhancedGPUCacheEntry {
  id: string;
  cacheKey: string;
  data: any;
  metadata: GPUCacheMetadata;
  integrations: IntegrationStatus;
  performance: PerformanceMetrics;
  timestamp: number;
  ttl?: number;
}

export interface GPUCacheMetadata {
  workflowType?: LegalWorkflowType;
  encodingFormat: EncodingFormat;
  compressionRatio: number;
  memoryFootprint: number;
  nesRegion?: NESMemoryRegion;
  webgpuAccelerated: boolean;
  securityLevel: SecurityLevel;
  legalContext?: LegalContext;
  version: string;
}

export interface IntegrationStatus {
  binaryEncoding: BinaryEncodingStatus;
  nesCache: NESCacheStatus;
  webgpu: WebGPUStatus;
  shaderCache: ShaderCacheStatus;
  som?: SOMClusteringStatus;
}

// === Binary Encoding Types ===

export type EncodingFormat = 'cbor' | 'msgpack' | 'json';

export interface BinaryEncodingStatus {
  enabled: boolean;
  format: EncodingFormat;
  compressionRatio: number;
  encodingTime: number;
  decodingTime: number;
  sizeBefore: number;
  sizeAfter: number;
  error?: string;
}

export interface BinaryEncodingOptions {
  format?: EncodingFormat;
  compression?: boolean;
  validation?: boolean;
  fallback?: boolean;
  performance?: boolean;
  workflowOptimized?: boolean;
}

export interface EncodingMetrics {
  format: EncodingFormat;
  originalSize: number;
  encodedSize: number;
  compressionRatio: number;
  encodeTime: number;
  decodeTime: number;
  cacheHitRatio?: number;
}

// === NES Cache Types ===

export type NESMemoryRegion =
  | 'PRG_ROM'
  | 'CHR_ROM'
  | 'RAM'
  | 'PPU_MEMORY'
  | 'SPRITE_MEMORY'
  | 'PALETTE_MEMORY';

export interface NESCacheStatus {
  cached: boolean;
  region?: NESMemoryRegion;
  memoryUsage: number;
  priority: number;
  lastAccessed: number;
  optimized: boolean;
  error?: string;
}

export interface NESMemoryAllocation {
  nesRegions: NESMemoryRegion[];
  allocation: Record<NESMemoryRegion, number>;
  utilization: number;
  available: number;
  fragmentation: number;
}

export interface NESCacheConfig {
  enablePredictiveLoading: boolean;
  enableCompression: boolean;
  enableCoherence: boolean;
  memoryBudget: number;
  garbageCollectionThreshold: number;
  defaultPriority: number;
}

// === WebGPU Types ===

export interface WebGPUStatus {
  available: boolean;
  accelerated: boolean;
  device?: string;
  adapter?: string;
  features: string[];
  memoryUsage?: number;
  performance?: WebGPUPerformanceMetrics;
  error?: string;
}

export interface WebGPUPerformanceMetrics {
  processingTime: number;
  gpuUtilization: number;
  memoryBandwidth: number;
  computeUnits: number;
  shaderCompilationTime?: number;
}

export interface WebGPUShaderData {
  name: string;
  computeShader: string;
  vertexShader?: string;
  fragmentShader?: string;
  uniforms?: Record<string, any>;
  bindGroupLayout?: GPUBindGroupLayoutDescriptor;
  workgroupSize?: [number, number, number];
}

// === Shader Cache Types ===

export interface ShaderCacheStatus {
  cached: boolean;
  shaderType: 'vertex' | 'fragment' | 'compute' | 'geometry';
  shaderLanguage: 'wgsl' | 'glsl' | 'hlsl';
  compilationTime: string;
  memoryFootprint: number;
  optimized: boolean;
  error?: string;
}

export interface ShaderCacheEntry {
  id: string;
  cacheKey: string;
  shaderType: 'vertex' | 'fragment' | 'compute' | 'geometry';
  sourceCode: string;
  compiledBinary: ArrayBuffer;
  encodingFormat: EncodingFormat;
  compressionRatio: number;
  metadata: ShaderMetadata;
  dependencies: string[];
  performance: ShaderPerformanceMetrics;
}

export interface ShaderMetadata {
  legalContext?: string;
  visualizationType?: string;
  complexity: number;
  embedding?: Float32Array;
  parameters?: Record<string, any>;
  tags: string[];
  createdAt: Date;
  updatedAt: Date;
}

export interface ShaderPerformanceMetrics {
  averageRenderTime?: number;
  memoryFootprint: number;
  gpuUtilization?: number;
  compilationTime: number;
  accessCount: number;
  reinforcementScore: number;
}

// === Legal Workflow Types ===

export type LegalWorkflowType =
  | 'document_upload'
  | 'evidence_review'
  | 'case_analysis'
  | 'contract_review'
  | 'litigation_prep'
  | 'deposition_analysis'
  | 'discovery_management'
  | 'legal_research'
  | 'compliance_audit';

export interface LegalWorkflowContext {
  type: LegalWorkflowType;
  jurisdiction?: string;
  practiceArea?: string;
  complexity: ComplexityLevel;
  documentCount?: number;
  estimatedDataSize?: number;
  requiresEncryption?: boolean;
  retentionPeriod?: number;
  collaborators?: number;
  urgency: UrgencyLevel;
  clientId?: string;
  caseId?: string;
  metadata?: Record<string, any>;
}

export type ComplexityLevel = 'low' | 'medium' | 'high' | 'critical';
export type UrgencyLevel = 'low' | 'medium' | 'high' | 'emergency';
export type SecurityLevel = 'standard' | 'enhanced' | 'maximum';

export interface LegalContext {
  jurisdiction: string;
  practiceArea: string;
  caseType: string;
  confidentialityLevel: SecurityLevel;
  retentionRequirements: number;
  complianceStandards: string[];
}

// === SOM Clustering Types ===

export interface SOMClusteringStatus {
  enabled: boolean;
  clustered: boolean;
  clusterCount?: number;
  similarity?: number;
  trainingId?: string;
  performance?: SOMPerformanceMetrics;
  error?: string;
}

export interface SOMPerformanceMetrics {
  trainingTime: number;
  convergenceRate: number;
  clusterAccuracy: number;
  memoryUsage: number;
}

export interface SOMClusterResult {
  clusterId: number;
  centroid: number[];
  members: string[];
  similarity: number;
  legalRelevance?: number;
}

// === Performance and Analytics Types ===

export interface PerformanceMetrics {
  cacheHitRatio: number;
  averageRetrievalTime: number;
  averageStorageTime: number;
  memoryUtilization: number;
  compressionEfficiency: number;
  gpuAccelerationGain?: number;
  nesOptimizationGain?: number;
  overallPerformanceScore: number;
}

export interface CacheAnalytics {
  totalEntries: number;
  totalSize: number;
  hitRate: number;
  missRate: number;
  evictionRate: number;
  averageEntryAge: number;
  hotEntries: string[];
  coldEntries: string[];
  workflowDistribution: Record<LegalWorkflowType, number>;
}

export interface OptimizationRecommendations {
  encoding: string;
  caching: string;
  memory: string;
  webgpu: string;
  workflow: string;
  security: string;
  priority: 'low' | 'medium' | 'high' | 'critical';
  estimatedImprovement: number;
}

// === API Response Types ===

export interface EnhancedGPUCacheResponse<T = any> {
  success: boolean;
  data?: T;
  entry?: EnhancedGPUCacheEntry;
  integrations?: IntegrationStatus;
  performance?: PerformanceMetrics;
  recommendations?: OptimizationRecommendations;
  metadata: ResponseMetadata;
  error?: string;
}

export interface ResponseMetadata {
  timestamp: string;
  processingTime: number;
  version: string;
  requestId?: string;
  cacheStatus: 'hit' | 'miss' | 'partial';
  optimizations: string[];
}

export interface BatchOperationResult {
  successful: number;
  failed: number;
  results: Array<{
    key: string;
    success: boolean;
    data?: any;
    error?: string;
  }>;
  performance: PerformanceMetrics;
}

// === Configuration Types ===

export interface EnhancedGPUCacheConfig {
  // Binary encoding configuration
  binaryEncoding: BinaryEncodingOptions;

  // NES cache configuration
  nesCache: NESCacheConfig;

  // WebGPU configuration
  webgpu: {
    enabled: boolean;
    preferredAdapter?: string;
    memoryLimit?: number;
    features?: string[];
  };

  // Shader cache configuration
  shaderCache: {
    enabled: boolean;
    maxEntries: number;
    compressionEnabled: boolean;
    predictiveLoading: boolean;
  };

  // Legal workflow configuration
  legalWorkflows: {
    enabled: boolean;
    defaultComplexity: ComplexityLevel;
    securityLevel: SecurityLevel;
    retentionPeriod: number;
  };

  // Performance configuration
  performance: {
    enableAnalytics: boolean;
    metricsInterval: number;
    optimizationThreshold: number;
  };
}

// === Utility Types ===

export interface CacheOperationOptions {
  ttl?: number;
  priority?: number;
  workflowType?: LegalWorkflowType;
  enableBinaryEncoding?: boolean;
  enableNESCache?: boolean;
  enableWebGPU?: boolean;
  securityLevel?: SecurityLevel;
  tags?: string[];
  metadata?: Record<string, any>;
}

export interface RetrievalOptions {
  includeMetadata?: boolean;
  includePerformance?: boolean;
  includeIntegrations?: boolean;
  decodeFormat?: EncodingFormat;
  workflowContext?: LegalWorkflowContext;
}

export interface WorkflowOptimizationResult {
  workflowType: LegalWorkflowType;
  recommendedEncoding: EncodingFormat;
  cacheStrategy: 'memory' | 'nes' | 'hybrid' | 'distributed';
  compressionLevel: number;
  estimatedPerformanceGain: number;
  memoryOptimization: NESMemoryAllocation;
  webgpuAcceleration: boolean;
  securityLevel: SecurityLevel;
  configuration: any;
  predictions: any;
}

// === Event Types ===

export interface CacheEvent {
  type: 'store' | 'retrieve' | 'evict' | 'optimize' | 'error';
  key: string;
  timestamp: number;
  workflowType?: LegalWorkflowType;
  performance?: Partial<PerformanceMetrics>;
  metadata?: Record<string, any>;
}

export interface IntegrationEvent {
  type: 'binary_encode' | 'nes_cache' | 'webgpu_process' | 'shader_compile' | 'som_cluster';
  source: string;
  target: string;
  success: boolean;
  performance: number;
  timestamp: number;
  details?: Record<string, any>;
}

// === Error Types ===

export class EnhancedGPUCacheError extends Error {
  constructor(
    message: string,
    public code: string,
    public integration?: keyof IntegrationStatus,
    public context?: any
  ) {
    super(message);
    this.name = 'EnhancedGPUCacheError';
  }
}

export interface ErrorContext {
  operation: string;
  key?: string;
  workflowType?: LegalWorkflowType;
  integration?: keyof IntegrationStatus;
  timestamp: number;
  requestId?: string;
}

// === Type Guards ===

export function isBinaryEncodingFormat(value: string): value is EncodingFormat {
  return ['cbor', 'msgpack', 'json'].includes(value);
}

export function isLegalWorkflowType(value: string): value is LegalWorkflowType {
  return [
    'document_upload', 'evidence_review', 'case_analysis',
    'contract_review', 'litigation_prep', 'deposition_analysis',
    'discovery_management', 'legal_research', 'compliance_audit'
  ].includes(value);
}

export function isNESMemoryRegion(value: string): value is NESMemoryRegion {
  return [
    'PRG_ROM', 'CHR_ROM', 'RAM',
    'PPU_MEMORY', 'SPRITE_MEMORY', 'PALETTE_MEMORY'
  ].includes(value);
}

export function isComplexityLevel(value: string): value is ComplexityLevel {
  return ['low', 'medium', 'high', 'critical'].includes(value);
}

export function isUrgencyLevel(value: string): value is UrgencyLevel {
  return ['low', 'medium', 'high', 'emergency'].includes(value);
}

export function isSecurityLevel(value: string): value is SecurityLevel {
  return ['standard', 'enhanced', 'maximum'].includes(value);
}

// === Utility Functions ===

export function createDefaultEnhancedGPUCacheConfig(): EnhancedGPUCacheConfig {
  return {
    binaryEncoding: {
      format: 'msgpack',
      compression: true,
      validation: true,
      fallback: true,
      performance: true,
      workflowOptimized: true
    },
    nesCache: {
      enablePredictiveLoading: true,
      enableCompression: true,
      enableCoherence: true,
      memoryBudget: 59424, // NES total budget
      garbageCollectionThreshold: 0.8,
      defaultPriority: 1
    },
    webgpu: {
      enabled: true,
      memoryLimit: 1024 * 1024 * 1024, // 1GB
      features: ['gpu-accelerated-rag', 'vector-ops']
    },
    shaderCache: {
      enabled: true,
      maxEntries: 1000,
      compressionEnabled: true,
      predictiveLoading: true
    },
    legalWorkflows: {
      enabled: true,
      defaultComplexity: 'medium',
      securityLevel: 'standard',
      retentionPeriod: 365
    },
    performance: {
      enableAnalytics: true,
      metricsInterval: 60000, // 1 minute
      optimizationThreshold: 0.7
    }
  };
}

export function calculateOverallPerformanceScore(metrics: PerformanceMetrics): number {
  const weights = {
    cacheHitRatio: 0.3,
    averageRetrievalTime: 0.2,
    memoryUtilization: 0.2,
    compressionEfficiency: 0.15,
    gpuAccelerationGain: 0.1,
    nesOptimizationGain: 0.05
  };

  // Normalize metrics to 0-1 range
  const normalized = {
    cacheHitRatio: metrics.cacheHitRatio,
    averageRetrievalTime: Math.max(0, 1 - (metrics.averageRetrievalTime / 1000)), // Invert - lower is better
    memoryUtilization: Math.min(metrics.memoryUtilization, 1),
    compressionEfficiency: metrics.compressionEfficiency,
    gpuAccelerationGain: metrics.gpuAccelerationGain || 0,
    nesOptimizationGain: metrics.nesOptimizationGain || 0
  };

  const score = Object.entries(weights).reduce((total, [key, weight]) => {
    return total + (normalized[key as keyof typeof normalized] * weight);
  }, 0);

  return Math.min(Math.max(score, 0), 1);
}

export function createCacheKey(;
  baseKey: string,
  workflowType?: LegalWorkflowType,
  format?: EncodingFormat
): string {
  const parts = [baseKey];
  if (workflowType) parts.push(workflowType);
  if (format) parts.push(format);
  return parts.join(':');
}

export function estimateMemoryFootprint(;
  data: any,
  format: EncodingFormat = 'json'
): number {
  const jsonSize = JSON.stringify(data).length * 2; // UTF-16

  switch (format) {
    case 'cbor':
      return Math.floor(jsonSize * 0.7); // ~30% compression
    case 'msgpack':
      return Math.floor(jsonSize * 0.8); // ~20% compression
    default:
      return jsonSize;
  }
}

// === Default Export ===

export default {
  // Type guards
  isBinaryEncodingFormat,
  isLegalWorkflowType,
  isNESMemoryRegion,
  isComplexityLevel,
  isUrgencyLevel,
  isSecurityLevel,

  // Utility functions
  createDefaultEnhancedGPUCacheConfig,
  calculateOverallPerformanceScore,
  createCacheKey,
  estimateMemoryFootprint,

  // Error class
  EnhancedGPUCacheError
};