/**
 * Legal AI Platform - Comprehensive Library Exports
 * SvelteKit 2 + Svelte 5 + TypeScript
 * 
 * Centralized export file for all components, services, stores, and utilities
 * "Wire it up" - TypeScript polyfills and WebAssembly/WebGPU fallbacks
 */

// SvelteKit 2 Polyfills - Import first to ensure module availability
import './polyfills.js';
import { barrelStore } from './stores/barrel-functions.js';

// Enhanced Type Definitions - Import to register module augmentations
import './types/drizzle-enhanced.js';
import './types/lokijs-enhanced.js';
import path from "path";

// ===== CENTRALIZED TYPES (SINGLE SOURCE OF TRUTH) =====
export * from './types/index';

// ===== TYPE GUARDS & UTILITIES =====
export * from './utils/type-guards';

// ===== ENHANCED API CLIENT =====
export { 
  EnhancedApiClient, 
  apiClient as enhancedApiClient 
} from './services/enhanced-api-client';

// ===== ALL COMPONENTS (COMPREHENSIVE BARREL EXPORT) =====
export * from './components/index';

// ===== FILE UPLOAD SERVICES =====
export { localStorageFiles } from './services/localStorage-file-fallback.js';
export { enhancedFileUpload } from './services/enhanced-file-upload.js';

// ===== UTILITIES & TYPES =====
export { 
  cn,
  formatFileSize,
  formatDate,
  generateId,
  debounce,
  throttle,
  getConfidenceLevel,
  getCaseStatusStyling,
  getEvidenceTypeStyling,
  formatProcessingTime,
  getInitials,
  isValidEmail,
  copyToClipboard,
  downloadFile,
  isBrowser,
  storage,
  theme
} from './utils';

// Export type helpers for Svelte 5 compatibility
export type {
  WithoutChild,
  WithoutChildren,
  WithoutChildrenOrChild,
  WithElementRef
} from './utils';

// ===== OLLAMA INTEGRATION SERVICES =====
export { 
  comprehensiveOllamaSummarizer,
  type ComprehensiveSummaryRequest,
  type ComprehensiveSummaryResponse,
  type SummarizerConfig,
  type SummarizerStats
} from './services/comprehensive-ollama-summarizer';

export { 
  ollamaIntegrationLayer,
  type IntegratedChatRequest,
  type IntegratedChatResponse,
  type OllamaServiceStatus
} from './services/ollama-integration-layer';

export { 
  LangChainOllamaService,
  langChainOllamaService,
  type LangChainConfig,
  type ProcessingResult,
  type QueryResult
} from './ai/langchain-ollama-service';

// ===== SERVER SERVICES (Server-side only) =====
// Note: These should only be imported on the server side
export type { AuthService } from './server/auth';
export type { EmbeddingService, EmbeddingOptions } from './server/embedding-service';

// ===== VERSION INFO =====
export const VERSION = '2.0.0';
export const BUILD_DATE = new Date().toISOString();
export const FRAMEWORK_INFO = {
  sveltekit: '2.x',
  svelte: '5.x',
  typescript: '5.x',
  vite: '5.x'
};

// ===== FEATURE FLAGS =====
export const FEATURES = {
  GPU_ACCELERATION: true,
  VECTOR_SEARCH: true,
  REAL_TIME_CHAT: true,
  CONTEXT7_INTEGRATION: true,
  MULTI_PROTOCOL_API: true,
  YORHA_THEME: true,
  MCP_INTEGRATION: true,
  WASM_SUPPORT: true,
  WEBGPU_SUPPORT: true,
  CUDA_SUPPORT: true
} as const;

// ===== DEVELOPMENT UTILITIES =====
export const DEV_TOOLS = {
  COMPONENT_COUNT: 392,
  ROUTE_COUNT: 82,
  API_ENDPOINT_COUNT: 145,
  STORE_COUNT: 8,
  SERVICE_COUNT: 12
} as const;

// ===== BARREL STORE - MISSING FUNCTIONS & METHODS =====
export { 
  barrelStore,
  testingFramework,
  cacheLayerMethods,
  databaseEntityProperties,
  webGPUExtendedMethods,
  lokiCollectionMethods,
  configurationProperties,
  utilityFunctions
} from './stores/barrel-functions';

// ===== DATABASE COMPATIBILITY LAYER =====
export {
  default as drizzleCompatibilityFix,
  drizzleCompatibilityLayer,
  handleQueryResult,
  safePropertyAccess,
  vectorOperations,
  ensureConnection,
  enhanceResultWithTypes,
  entityEnhancers,
  createTypeSafeQuery
} from './database/drizzle-compatibility-fix';

// Make barrel store globally available
if (typeof globalThis !== 'undefined') {
  globalThis.barrelStore = barrelStore;
}

// ===== ENHANCED SERVICES & STORES =====

// Global User Store with Svelte 5 Runes
export { default as globalUserStore } from './stores/global-user-store.svelte';

// Search Services with Fuse.js Integration
export { 
  searchService, 
  globalSearch, 
  searchServices, 
  searchComponents, 
  searchDocumentation, 
  searchDemos 
} from './services/search-service';

// Hybrid Vector Operations
export { 
  hybridVectorService, 
  hybridSearch, 
  syncVectorData, 
  getVectorSystemHealth 
} from './services/hybrid-vector-operations';

// Search Types
export type {
  SearchResult,
  SearchCategory,
  SearchOptions,
  SearchFilter,
  SearchState
} from './types/search.types';

// Default export for convenience
export default {
  VERSION,
  BUILD_DATE,
  FRAMEWORK_INFO,
  FEATURES,
  DEV_TOOLS,
  barrelStore
};

// ===== TYPESCRIPT ERROR RESOLUTION UTILITIES =====
export const typeScriptErrorResolution = {
  // Utility to enhance objects with missing properties
  enhanceWithMissingProperties: <T extends object>(obj: T, properties: Partial<T>): T => {
    return { ...obj, ...properties };
  },
  
  // Safe property access with type assertions
  safeAccess: <T>(obj: any, path: string, defaultValue: T): T => {
    const keys = path.split('.');
    let current = obj;
    for (const key of keys) {
      if (current && typeof current === 'object' && key in current) {
        current = current[key];
      } else {
        return defaultValue;
      }
    }
    return current;
  },
  
  // Type assertion with fallback
  assertType: <T>(value: any, fallback: T): T => {
    return value !== null && value !== undefined ? value : fallback;
  }
};