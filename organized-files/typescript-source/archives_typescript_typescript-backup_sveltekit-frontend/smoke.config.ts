/**
 * Comprehensive Smoke Test Configuration
 * Legal AI Platform - Service Health & Route Validation
 */

export interface EndpointConfig {
  name: string;
  url: string;
  method?: 'GET' | 'HEAD' | 'POST';
  expectedStatus?: number | number[];
  timeout?: number;
  critical?: boolean;
  category: 'service' | 'api' | 'route' | 'debug';
}

export const serviceEndpoints: EndpointConfig[] = [
  // Backend Services
  {
    name: "Enhanced RAG Service",
    url: "http://localhost:8094/api/health",
    method: "GET",
    expectedStatus: 200,
    critical: true,
    category: "service"
  },
  {
    name: "Context7 MCP Server",
    url: "http://localhost:4000/health", 
    method: "GET",
    expectedStatus: 200,
    critical: true,
    category: "service"
  },
  {
    name: "Ollama AI Service",
    url: "http://localhost:11434/api/tags",
    method: "GET", 
    expectedStatus: 200,
    critical: true,
    category: "service"
  },
  {
    name: "Upload Service",
    url: "http://localhost:8093/health",
    method: "GET",
    expectedStatus: 200,
    critical: false,
    category: "service"
  }
];

export const apiEndpoints: EndpointConfig[] = [
  // Core API Routes
  {
    name: "AI Chat API",
    url: "http://localhost:5181/api/ai/chat",
    method: "GET",
    expectedStatus: [200, 405], // 405 for GET on POST endpoint is OK
    critical: true,
    category: "api"
  },
  {
    name: "Enhanced RAG API",
    url: "http://localhost:5181/api/enhanced-rag",
    method: "GET",
    expectedStatus: [200, 405],
    critical: true,
    category: "api"
  },
  {
    name: "Evidence Upload API",
    url: "http://localhost:5181/api/evidence",
    method: "GET",
    expectedStatus: [200, 405],
    critical: true,
    category: "api"
  }
];

// SvelteKit routes to test (critical paths)
export const criticalRoutes: EndpointConfig[] = [
  {
    name: "Homepage",
    url: "http://localhost:5181/",
    method: "GET",
    expectedStatus: 200,
    critical: true,
    category: "route"
  },
  {
    name: "Enhanced RAG Demo",
    url: "http://localhost:5181/demo/enhanced-rag-semantic",
    method: "GET",
    expectedStatus: 200,
    critical: true,
    category: "route"
  },
  {
    name: "YoRHa Command Center",
    url: "http://localhost:5181/yorha-command-center",
    method: "GET",
    expectedStatus: 200,
    critical: false,
    category: "route"
  },
  {
    name: "System Health Dashboard",
    url: "http://localhost:5181/system/health",
    method: "GET",
    expectedStatus: 200,
    critical: false,
    category: "route"
  }
];

// Legacy endpoints (for backward compatibility)
export const endpoints = [
  { name: 'SvelteKit (frontend)', url: 'http://localhost:5181/' },
  { name: 'Enhanced RAG', url: 'http://localhost:8094/api/health' },
  { name: 'Upload Service', url: 'http://localhost:8093/health' },
  { name: 'Context7 MCP', url: 'http://localhost:4000/health' },
  { name: 'Ollama', url: 'http://localhost:11434/api/tags' }
];

// Combine all endpoints
export const allEndpoints = [
  ...serviceEndpoints,
  ...apiEndpoints,
  ...criticalRoutes
];

// Smoke test configuration
export const smokeConfig = {
  timeout: 10000,
  concurrency: 5,
  retries: 2,
  baseUrl: "http://localhost:5181",
  criticalOnly: false,
  verbose: true
};
