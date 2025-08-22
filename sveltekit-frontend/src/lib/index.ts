/**
 * Legal AI Platform - Comprehensive Library Exports
 * SvelteKit 2 + Svelte 5 + TypeScript
 * 
 * Centralized export file for all components, services, stores, and utilities
 */

// ===== CORE UI COMPONENTS =====
export { default as Button } from './components/ui/Button.svelte';
export { default as Card } from './components/ui/Card.svelte';
// Essential exports only
export { cn } from './utils';

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

// Default export for convenience
export default {
  VERSION,
  BUILD_DATE,
  FRAMEWORK_INFO,
  FEATURES,
  DEV_TOOLS
};