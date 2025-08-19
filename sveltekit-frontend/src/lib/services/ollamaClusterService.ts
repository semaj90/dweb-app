/**
 * Multi-Core Ollama Cluster Service
 * Handles load balancing and distribution across Ollama instances
 */

import { writable, derived } from 'svelte/store';

export interface OllamaInstance {
  id: string;
  url: string;
  port: number;
  status: 'healthy' | 'unhealthy' | 'starting' | 'stopping';
  cpu_cores: number;
  memory_gb: number;
  models: string[];
  current_load: number; // 0-100
  response_time: number; // ms
  last_health_check: number;
}

export interface OllamaClusterConfig {
  instances: OllamaInstance[];
  load_balancing_strategy: 'round_robin' | 'least_connections' | 'cpu_based' | 'response_time';
  health_check_interval: number;
  max_retries: number;
  timeout: number;
  preferred_models: {
    legal_analysis: string;
    embeddings: string;
    general_chat: string;
    document_summary: string;
  };
}

export interface ClusterStats {
  total_instances: number;
  healthy_instances: number;
  total_cpu_cores: number;
  total_memory_gb: number;
  average_load: number;
  average_response_time: number;
  requests_per_minute: number;
  uptime_percentage: number;
}

const initialConfig: OllamaClusterConfig = {
  instances: [
    {
      id: 'ollama-primary',
      url: 'http://localhost:11434',
      port: 11434,
      status: 'healthy',
      cpu_cores: 8,
      memory_gb: 16,
      models: ['gemma3-legal', 'nomic-embed-text', 'deeds-web'],
      current_load: 0,
      response_time: 0,
      last_health_check: Date.now()
    },
    {
      id: 'ollama-secondary',
      url: 'http://localhost:11435',
      port: 11435,
      status: 'starting',
      cpu_cores: 6,
      memory_gb: 12,
      models: ['gemma3-legal', 'nomic-embed-text'],
      current_load: 0,
      response_time: 0,
      last_health_check: Date.now()
    },
    {
      id: 'ollama-embeddings',
      url: 'http://localhost:11436',
      port: 11436,
      status: 'starting',
      cpu_cores: 4,
      memory_gb: 8,
      models: ['nomic-embed-text'],
      current_load: 0,
      response_time: 0,
      last_health_check: Date.now()
    }
  ],
  load_balancing_strategy: 'cpu_based',
  health_check_interval: 30000, // 30 seconds
  max_retries: 3,
  timeout: 60000, // 60 seconds
  preferred_models: {
    legal_analysis: 'gemma3-legal',
    embeddings: 'nomic-embed-text',
    general_chat: 'deeds-web',
    document_summary: 'gemma3-legal'
  }
};

// Core stores
export const ollamaClusterStore = writable<OllamaClusterConfig>(initialConfig);
export const clusterStatsStore = writable<ClusterStats>({
  total_instances: 3,
  healthy_instances: 1,
  total_cpu_cores: 18,
  total_memory_gb: 36,
  average_load: 0,
  average_response_time: 0,
  requests_per_minute: 0,
  uptime_percentage: 33.3
});

// Derived stores
export const healthyInstances = derived(
  ollamaClusterStore,
  $cluster => $cluster.instances.filter(instance => instance.status === 'healthy')
);

export const bestPerformingInstance = derived(
  healthyInstances,
  $instances => {
    if ($instances.length === 0) return null;
    
    // Sort by lowest load and response time
    return $instances.sort((a, b) => {
      const scoreA = a.current_load + (a.response_time / 10);
      const scoreB = b.current_load + (b.response_time / 10);
      return scoreA - scoreB;
    })[0];
  }
);

export class OllamaClusterService {
  private healthCheckInterval: NodeJS.Timeout | null = null;
  private requestCounter = 0;
  private lastMinuteRequests: number[] = [];

  constructor() {
    this.startHealthChecking();
  }

  /**
   * Start health checking for all instances
   */
  private startHealthChecking(): void {
    if (this.healthCheckInterval) {
      clearInterval(this.healthCheckInterval);
    }

    this.healthCheckInterval = setInterval(async () => {
      await this.checkAllInstancesHealth();
    }, initialConfig.health_check_interval);

    // Initial health check
    this.checkAllInstancesHealth();
  }

  /**
   * Check health of all Ollama instances
   */
  private async checkAllInstancesHealth(): Promise<void> {
    const config = await new Promise<OllamaClusterConfig>(resolve => {
      ollamaClusterStore.subscribe(value => resolve(value))();
    });

    const updatedInstances = await Promise.all(
      config.instances.map(async (instance) => {
        const startTime = Date.now();
        
        try {
          const response = await fetch(`${instance.url}/api/tags`, {
            signal: AbortSignal.timeout(5000)
          });

          const responseTime = Date.now() - startTime;
          
          if (response.ok) {
            const data = await response.json();
            return {
              ...instance,
              status: 'healthy' as const,
              models: data.models?.map((m: any) => m.name) || instance.models,
              response_time: responseTime,
              last_health_check: Date.now()
            };
          } else {
            return {
              ...instance,
              status: 'unhealthy' as const,
              response_time: responseTime,
              last_health_check: Date.now()
            };
          }
        } catch (error) {
          return {
            ...instance,
            status: 'unhealthy' as const,
            response_time: 9999,
            last_health_check: Date.now()
          };
        }
      })
    );

    // Update cluster configuration
    ollamaClusterStore.update(config => ({
      ...config,
      instances: updatedInstances
    }));

    // Update cluster stats
    this.updateClusterStats(updatedInstances);
  }

  /**
   * Update cluster statistics
   */
  private updateClusterStats(instances: OllamaInstance[]): void {
    const healthyInstances = instances.filter(i => i.status === 'healthy');
    const totalCores = instances.reduce((sum, i) => sum + i.cpu_cores, 0);
    const totalMemory = instances.reduce((sum, i) => sum + i.memory_gb, 0);
    const avgLoad = healthyInstances.length > 0 
      ? healthyInstances.reduce((sum, i) => sum + i.current_load, 0) / healthyInstances.length 
      : 0;
    const avgResponseTime = healthyInstances.length > 0
      ? healthyInstances.reduce((sum, i) => sum + i.response_time, 0) / healthyInstances.length
      : 0;

    // Calculate requests per minute
    const now = Date.now();
    this.lastMinuteRequests = this.lastMinuteRequests.filter(timestamp => now - timestamp < 60000);
    const requestsPerMinute = this.lastMinuteRequests.length;

    clusterStatsStore.set({
      total_instances: instances.length,
      healthy_instances: healthyInstances.length,
      total_cpu_cores: totalCores,
      total_memory_gb: totalMemory,
      average_load: avgLoad,
      average_response_time: avgResponseTime,
      requests_per_minute: requestsPerMinute,
      uptime_percentage: instances.length > 0 ? (healthyInstances.length / instances.length) * 100 : 0
    });
  }

  /**
   * Select best instance for a request
   */
  async selectInstance(preferredModel?: string): Promise<OllamaInstance | null> {
    const config = await new Promise<OllamaClusterConfig>(resolve => {
      ollamaClusterStore.subscribe(value => resolve(value))();
    });

    const healthyInstances = config.instances.filter(i => i.status === 'healthy');
    
    if (healthyInstances.length === 0) {
      console.warn('No healthy Ollama instances available');
      return null;
    }

    // Filter by model if specified
    let candidateInstances = healthyInstances;
    if (preferredModel) {
      const modelInstances = healthyInstances.filter(i => i.models.includes(preferredModel));
      if (modelInstances.length > 0) {
        candidateInstances = modelInstances;
      }
    }

    // Apply load balancing strategy
    switch (config.load_balancing_strategy) {
      case 'round_robin':
        return candidateInstances[this.requestCounter % candidateInstances.length];
      
      case 'least_connections':
      case 'cpu_based':
        return candidateInstances.sort((a, b) => a.current_load - b.current_load)[0];
      
      case 'response_time':
        return candidateInstances.sort((a, b) => a.response_time - b.response_time)[0];
      
      default:
        return candidateInstances[0];
    }
  }

  /**
   * Execute a request with automatic failover
   */
  async executeRequest(
    endpoint: string,
    payload: any,
    preferredModel?: string,
    retries: number = 3
  ): Promise<any> {
    const instance = await this.selectInstance(preferredModel);
    
    if (!instance) {
      throw new Error('No healthy Ollama instances available');
    }

    this.requestCounter++;
    this.lastMinuteRequests.push(Date.now());

    const startTime = Date.now();
    
    try {
      const response = await fetch(`${instance.url}${endpoint}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload),
        signal: AbortSignal.timeout(60000)
      });

      const responseTime = Date.now() - startTime;
      
      // Update instance response time
      ollamaClusterStore.update(config => ({
        ...config,
        instances: config.instances.map(i => 
          i.id === instance.id 
            ? { ...i, response_time: responseTime }
            : i
        )
      }));

      if (!response.ok) {
        throw new Error(`Request failed: ${response.status} ${response.statusText}`);
      }

      return await response.json();

    } catch (error) {
      console.error(`Request to ${instance.id} failed:`, error);
      
      // Mark instance as unhealthy if multiple failures
      ollamaClusterStore.update(config => ({
        ...config,
        instances: config.instances.map(i => 
          i.id === instance.id 
            ? { ...i, status: 'unhealthy' }
            : i
        )
      }));

      // Retry with different instance
      if (retries > 0) {
        return this.executeRequest(endpoint, payload, preferredModel, retries - 1);
      }

      throw error;
    }
  }

  /**
   * Generate text completion with cluster
   */
  async generateText(prompt: string, model?: string): Promise<string> {
    const selectedModel = model || initialConfig.preferred_models.legal_analysis;
    
    const response = await this.executeRequest('/api/generate', {
      model: selectedModel,
      prompt,
      stream: false
    }, selectedModel);

    return response.response || '';
  }

  /**
   * Generate embeddings with cluster
   */
  async generateEmbeddings(text: string): Promise<number[]> {
    const embeddingModel = initialConfig.preferred_models.embeddings;
    
    const response = await this.executeRequest('/api/embeddings', {
      model: embeddingModel,
      prompt: text
    }, embeddingModel);

    return response.embedding || [];
  }

  /**
   * Add new instance to cluster
   */
  async addInstance(instance: Omit<OllamaInstance, 'last_health_check'>): Promise<void> {
    ollamaClusterStore.update(config => ({
      ...config,
      instances: [...config.instances, { ...instance, last_health_check: Date.now() }]
    }));
  }

  /**
   * Remove instance from cluster
   */
  async removeInstance(instanceId: string): Promise<void> {
    ollamaClusterStore.update(config => ({
      ...config,
      instances: config.instances.filter(i => i.id !== instanceId)
    }));
  }

  /**
   * Update load balancing strategy
   */
  updateStrategy(strategy: OllamaClusterConfig['load_balancing_strategy']): void {
    ollamaClusterStore.update(config => ({
      ...config,
      load_balancing_strategy: strategy
    }));
  }

  /**
   * Get cluster status summary
   */
  async getClusterStatus(): Promise<{
    healthy: boolean;
    message: string;
    stats: ClusterStats;
  }> {
    const stats = await new Promise<ClusterStats>(resolve => {
      clusterStatsStore.subscribe(value => resolve(value))();
    });

    const healthy = stats.healthy_instances > 0;
    const message = healthy 
      ? `Cluster operational: ${stats.healthy_instances}/${stats.total_instances} instances healthy`
      : 'Cluster down: No healthy instances available';

    return { healthy, message, stats };
  }

  /**
   * Cleanup resources
   */
  destroy(): void {
    if (this.healthCheckInterval) {
      clearInterval(this.healthCheckInterval);
      this.healthCheckInterval = null;
    }
  }
}

// Singleton instance
export const ollamaCluster = new OllamaClusterService();