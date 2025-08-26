# SvelteKit 2 + Svelte 5 + Bits UI v2 Best Practices
## Production Legal AI Platform Architecture

### 🎯 **Core Architecture Principles**

#### **1. Type Safety & Service Integration**
```typescript
// ✅ Proper type organization
export type {
  AITask,
  AIResponse,
  WorkerMessage,
  WorkerStatus,
  VectorSearchResult,
  RAGContext,
  RAGResponse,
  ProcessingContext,
  UserContext
} from '$lib/services/types/service-types.js';

// ✅ Enhanced service type definitions
export interface AITask {
  id: string;
  type: 'embedding' | 'summarization' | 'analysis' | 'search' | 'classification';
  priority: 'low' | 'medium' | 'high' | 'critical';
  data: Record<string, unknown>;
  context?: ProcessingContext;
  timestamp: number;
  retries: number;
  maxRetries: number;
}
```

#### **2. Environment Service Pattern**
```typescript
// ✅ SvelteKit 2 compatible environment detection
import { browser } from '$app/environment';
import { env } from '$env/dynamic/public';

export const CLIENT_ENV = {
  dev: import.meta.env.DEV,
  prod: import.meta.env.PROD,
  preview: import.meta.env.MODE === 'preview',
  browser: browser
};

// ✅ LLM endpoint health checking
export async function getHealthyLlmEndpoint(model?: string): Promise<string | null> {
  const endpoint = await healthChecker.getHealthyLlmEndpoint(model);
  return endpoint ? endpoint.url : null;
}
```

### 🔧 **Service Architecture Patterns**

#### **3. AI Worker Management**
```typescript
// ✅ Proper worker orchestration
export class AIWorkerManager implements AIServiceWorkerManager {
  private workerPool: WorkerPool;
  private activeTasks: Map<string, TaskContext>;
  
  async submitTask(task: AITask): Promise<string> {
    if (!this.isInitialized) {
      await this.initialize();
    }

    const taskId = (task as any).taskId || crypto.randomUUID();
    const enhancedTask: AITask = {
      ...task,
      timestamp: Date.now(),
    };

    return new Promise((resolve, reject) => {
      const workerId = this.selectWorker(enhancedTask);
      // ... worker selection and execution
    });
  }
}
```

#### **4. Vector Store Integration**
```typescript
// ✅ PGVectorStore proper initialization
async function initializePGVectorStore(): Promise<PGVectorStore> {
  try {
    const vectorStore = new PGVectorStore(embeddings, {
      pool: pgPool,
      tableName: "legal_document_embeddings",
      columns: {
        idColumnName: "id",
        vectorColumnName: "embedding",
        contentColumnName: "content",
        metadataColumnName: "metadata",
      },
    });
    
    await vectorStore.ensureTableInDatabase();
    return vectorStore;
  } catch (pgError) {
    // Fallback initialization
    return await (PGVectorStore as any).initialize(embeddings, config);
  }
}
```

### 🎨 **Component Architecture**

#### **5. Svelte 5 Component Structure**
```typescript
// ✅ Modern Svelte 5 component with proper typing
<script lang="ts">
  import type { LLMEndpoint, ClientEnvironment } from '$lib/services/types/service-types.js';
  import { getHealthyLlmEndpoint, CLIENT_ENV } from '$lib/services/utils/environment-service.js';
  
  interface Props {
    endpoint?: LLMEndpoint;
    environment?: ClientEnvironment;
    onHealthCheck?: (healthy: boolean) => void;
  }
  
  let { endpoint, environment = CLIENT_ENV, onHealthCheck }: Props = $props();
  
  let healthStatus = $state<boolean>(false);
  
  async function checkHealth() {
    const healthyEndpoint = await getHealthyLlmEndpoint();
    healthStatus = !!healthyEndpoint;
    onHealthCheck?.(healthStatus);
  }
</script>
```

#### **6. Bits UI v2 Integration**
```typescript
// ✅ Proper Bits UI v2 usage with Svelte 5
import { createDialog, melt } from '@melt-ui/svelte';
import type { CreateDialogProps } from '@melt-ui/svelte';

interface DialogProps extends CreateDialogProps {
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
}

let { open = false, onOpenChange, ...restProps }: DialogProps = $props();

const {
  elements: { trigger, overlay, content, title, description, close },
  states: { open: isOpen }
} = createDialog({
  open: open,
  onOpenChange: ({ next }) => {
    onOpenChange?.(next);
    return next;
  },
  ...restProps
});
```

### 📡 **API Layer Patterns**

#### **7. SvelteKit 2 Server Hooks**
```typescript
// ✅ Enhanced server hooks with service integration
import { sequence } from '@sveltejs/kit/hooks';
import type { Handle, RequestEvent } from '@sveltejs/kit';

const initializeServices: Handle = async ({ event, resolve }) => {
  // Initialize AI services
  event.locals.aiServices = {
    pipeline: aiPipeline,
    workerManager: aiWorkerManager,
    healthChecker: healthChecker
  };
  
  return resolve(event);
};

const apiContextHook: Handle = async ({ event, resolve }) => {
  if (event.url.pathname.startsWith('/api/')) {
    event.locals.context = {
      userId: event.locals.user?.id,
      timestamp: Date.now(),
      requestId: crypto.randomUUID()
    };
  }
  
  return resolve(event);
};

export const handle = sequence(initializeServices, apiContextHook);
```

#### **8. Type-Safe API Endpoints**
```typescript
// ✅ SvelteKit 2 API route with proper typing
import type { RequestHandler } from './$types.js';
import type { 
  AITask, 
  AIResponse, 
  ProcessingContext 
} from '$lib/services/types/service-types.js';

export const POST: RequestHandler = async ({ request, locals }) => {
  try {
    const { query, context }: {
      query: string;
      context: ProcessingContext;
    } = await request.json();

    const task: AITask = {
      id: crypto.randomUUID(),
      type: 'analysis',
      priority: 'medium',
      data: { query },
      context,
      timestamp: Date.now(),
      retries: 0,
      maxRetries: 3
    };

    const taskId = await locals.aiServices.workerManager.submitTask(task);
    
    return new Response(JSON.stringify({ 
      success: true, 
      taskId 
    }), {
      headers: { 'content-type': 'application/json' }
    });
  } catch (error) {
    return new Response(JSON.stringify({ 
      success: false, 
      error: error.message 
    }), {
      status: 500,
      headers: { 'content-type': 'application/json' }
    });
  }
};
```

### 🔄 **State Management Patterns**

#### **9. Svelte 5 Stores with Service Integration**
```typescript
// ✅ Modern Svelte 5 store pattern
import { writable, derived } from 'svelte/store';
import type { Writable } from 'svelte/store';
import type { AITask, WorkerStatus } from '$lib/services/types/service-types.js';

class AIServiceStore {
  private _tasks: Writable<AITask[]> = writable([]);
  private _status: Writable<WorkerStatus | null> = writable(null);
  
  readonly tasks = this._tasks.asReadonly();
  readonly status = this._status.asReadonly();
  
  readonly isHealthy = derived(
    this.status,
    ($status) => $status?.activeRequests !== undefined && $status.errors === 0
  );

  async submitTask(task: Omit<AITask, 'id' | 'timestamp'>) {
    const fullTask: AITask = {
      ...task,
      id: crypto.randomUUID(),
      timestamp: Date.now()
    };
    
    this._tasks.update(tasks => [...tasks, fullTask]);
    
    try {
      const taskId = await aiWorkerManager.submitTask(fullTask);
      return taskId;
    } catch (error) {
      this._tasks.update(tasks => 
        tasks.filter(t => t.id !== fullTask.id)
      );
      throw error;
    }
  }
  
  async refreshStatus() {
    const status = await aiWorkerManager.getStatus();
    this._status.set(status);
  }
}

export const aiServiceStore = new AIServiceStore();
```

### 🛠 **Error Handling Patterns**

#### **10. Production Error Handling**
```typescript
// ✅ Comprehensive error handling with type safety
interface ServiceError {
  code: string;
  message: string;
  details?: Record<string, unknown>;
  timestamp: number;
}

export class ServiceErrorHandler {
  static handle(error: unknown, context?: string): ServiceError {
    const timestamp = Date.now();
    
    if (error instanceof Error) {
      return {
        code: error.name || 'UNKNOWN_ERROR',
        message: error.message,
        details: { stack: error.stack, context },
        timestamp
      };
    }
    
    return {
      code: 'UNKNOWN_ERROR',
      message: String(error),
      details: { context },
      timestamp
    };
  }
  
  static async withRetry<T>(
    operation: () => Promise<T>, 
    maxRetries: number = 3,
    backoffMs: number = 1000
  ): Promise<T> {
    let lastError: unknown;
    
    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      try {
        return await operation();
      } catch (error) {
        lastError = error;
        
        if (attempt < maxRetries) {
          await new Promise(resolve => 
            setTimeout(resolve, backoffMs * Math.pow(2, attempt))
          );
        }
      }
    }
    
    throw ServiceErrorHandler.handle(lastError, 'retry_exhausted');
  }
}
```

### 📊 **Performance Optimization**

#### **11. Lazy Loading & Code Splitting**
```typescript
// ✅ Proper code splitting for SvelteKit 2
// Dynamic imports for heavy services
export const loadAIServices = async () => {
  const [
    { aiPipeline },
    { aiWorkerManager },
    { aiAssistantInputSynthesizer }
  ] = await Promise.all([
    import('$lib/services/ai-pipeline.js'),
    import('$lib/services/ai-worker-manager.js'),
    import('$lib/services/ai-assistant-input-synthesizer.js')
  ]);
  
  return {
    aiPipeline,
    aiWorkerManager,
    aiAssistantInputSynthesizer
  };
};

// Component-level lazy loading
<script lang="ts">
  import { onMount } from 'svelte';
  
  let AIComponent: any = null;
  
  onMount(async () => {
    const module = await import('$lib/components/AIComponent.svelte');
    AIComponent = module.default;
  });
</script>

{#if AIComponent}
  <svelte:component this={AIComponent} />
{:else}
  <div>Loading AI services...</div>
{/if}
```

### 🏗 **Build & Development**

#### **12. Vite Configuration for SvelteKit 2**
```typescript
// ✅ Production-optimized Vite config
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [sveltekit()],
  optimizeDeps: {
    include: [
      '@langchain/community',
      '@melt-ui/svelte',
      'drizzle-orm'
    ]
  },
  server: {
    proxy: {
      '/api/go/enhanced-rag': 'http://localhost:8094',
      '/api/go/upload': 'http://localhost:8093',
      '/api/ollama': 'http://localhost:11434',
      '/api/qdrant': 'http://localhost:6333'
    }
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          'ai-services': [
            '$lib/services/ai-pipeline',
            '$lib/services/ai-worker-manager'
          ],
          'vector-ops': [
            '@langchain/community/vectorstores/pgvector',
            '@langchain/community/embeddings/ollama'
          ]
        }
      }
    }
  }
});
```

### 🔒 **Security Best Practices**

#### **13. Environment Variable Management**
```typescript
// ✅ Secure environment handling
import { env } from '$env/dynamic/private';
import { PUBLIC_API_BASE } from '$env/static/public';

export function getServiceConfig(serviceName: string) {
  return {
    baseUrl: env[`${serviceName.toUpperCase()}_URL`] || `http://localhost:8094`,
    enabled: env[`${serviceName.toUpperCase()}_ENABLED`] !== 'false',
    timeout: parseInt(env[`${serviceName.toUpperCase()}_TIMEOUT`] || '30000', 10),
    retryAttempts: parseInt(env[`${serviceName.toUpperCase()}_RETRIES`] || '3', 10),
  };
}

// ✅ Input validation and sanitization
import { z } from 'zod';

const AITaskSchema = z.object({
  type: z.enum(['embedding', 'summarization', 'analysis', 'search', 'classification']),
  priority: z.enum(['low', 'medium', 'high', 'critical']),
  data: z.record(z.unknown()),
  maxRetries: z.number().min(0).max(10).default(3)
});

export function validateAITask(input: unknown): AITask {
  const result = AITaskSchema.parse(input);
  return {
    ...result,
    id: crypto.randomUUID(),
    timestamp: Date.now(),
    retries: 0
  };
}
```

### 🎯 **Testing Strategies**

#### **14. Component Testing with Svelte 5**
```typescript
// ✅ Modern testing setup
import { render, screen } from '@testing-library/svelte';
import { describe, it, expect, vi } from 'vitest';
import AIServiceComponent from '$lib/components/AIServiceComponent.svelte';

describe('AIServiceComponent', () => {
  it('should handle AI task submission', async () => {
    const mockSubmitTask = vi.fn().mockResolvedValue('task-123');
    
    render(AIServiceComponent, {
      props: {
        onTaskSubmit: mockSubmitTask
      }
    });
    
    const button = screen.getByRole('button', { name: /submit task/i });
    await button.click();
    
    expect(mockSubmitTask).toHaveBeenCalledWith(
      expect.objectContaining({
        type: 'analysis',
        priority: 'medium'
      })
    );
  });
});
```

### 🚀 **Deployment & Monitoring**

#### **15. Health Check Implementation**
```typescript
// ✅ Comprehensive health monitoring
export interface HealthCheckResult {
  status: 'healthy' | 'degraded' | 'unhealthy';
  services: Record<string, ServiceHealth>;
  timestamp: number;
  version: string;
}

export interface ServiceHealth {
  status: 'healthy' | 'unhealthy';
  latency?: number;
  error?: string;
  metadata?: Record<string, unknown>;
}

export class HealthMonitor {
  static async checkAllServices(): Promise<HealthCheckResult> {
    const checks = await Promise.allSettled([
      this.checkPostgreSQL(),
      this.checkOllama(),
      this.checkRedis(),
      this.checkVectorStore()
    ]);
    
    const services: Record<string, ServiceHealth> = {};
    const serviceNames = ['postgresql', 'ollama', 'redis', 'vectorstore'];
    
    checks.forEach((result, index) => {
      services[serviceNames[index]] = result.status === 'fulfilled' 
        ? result.value 
        : { status: 'unhealthy', error: result.reason?.message };
    });
    
    const healthyCount = Object.values(services).filter(s => s.status === 'healthy').length;
    const status = healthyCount === Object.keys(services).length 
      ? 'healthy' 
      : healthyCount > 0 ? 'degraded' : 'unhealthy';
    
    return {
      status,
      services,
      timestamp: Date.now(),
      version: '1.0.0'
    };
  }
}
```

---

## 🏆 **Production Checklist**

### ✅ **Essential Components**
- [x] **Type-safe service layer** with comprehensive interfaces
- [x] **Environment service** for configuration management  
- [x] **AI worker orchestration** with proper lifecycle management
- [x] **Vector store integration** with PGVectorStore fallbacks
- [x] **Error handling** with retry mechanisms and logging
- [x] **Health monitoring** across all services
- [x] **Svelte 5 compatibility** with modern reactive patterns
- [x] **Bits UI v2 integration** for accessible components
- [x] **Security patterns** with input validation and sanitization
- [x] **Performance optimization** with lazy loading and code splitting

### 🎯 **Architecture Benefits**
1. **Type Safety**: End-to-end TypeScript with proper service interfaces
2. **Modularity**: Clean separation of concerns with barrel exports
3. **Reliability**: Comprehensive error handling and health monitoring  
4. **Performance**: Optimized builds with strategic code splitting
5. **Scalability**: Worker-based AI processing with load balancing
6. **Security**: Proper environment variable management and input validation
7. **Maintainability**: Clear patterns and best practices throughout

This architecture provides a solid foundation for the Legal AI platform with SvelteKit 2, Svelte 5, and modern TypeScript patterns.