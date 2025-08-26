/**
 * Service Configuration and Management System
 * Handles dynamic configuration, service discovery, and runtime management
 */

import fs from 'fs/promises';
import path from 'path';
import yaml from 'js-yaml';
import { EventEmitter } from 'events';
import { createRequire } from 'module';
import { performance } from 'perf_hooks';

const require = createRequire(import.meta.url);

interface ServiceDefinition {
  name: string;
  version: string;
  description: string;
  type: 'go' | 'node' | 'python' | 'native' | 'windows';
  
  execution: {
    command: string;
    args: string[];
    cwd?: string;
    env?: Record<string, string>;
    shell?: boolean;
  };
  
  network: {
    port?: number;
    host?: string;
    protocol: 'http' | 'https' | 'grpc' | 'quic' | 'tcp';
    healthCheck?: {
      path: string;
      interval: number;
      timeout: number;
      retries: number;
    };
  };
  
  dependencies: {
    required: string[];
    optional: string[];
    services: string[];
    infrastructure: string[];
  };
  
  resources: {
    cpu?: {
      min: string;
      max: string;
    };
    memory?: {
      min: string;
      max: string;
    };
    gpu?: {
      required: boolean;
      memory?: string;
      cores?: number;
    };
    disk?: {
      space: string;
      type?: 'ssd' | 'hdd';
    };
  };
  
  scaling: {
    min: number;
    max: number;
    targetCpu: number;
    targetMemory: number;
  };
  
  lifecycle: {
    startup: {
      timeout: number;
      retries: number;
      delay: number;
    };
    shutdown: {
      timeout: number;
      gracePeriod: number;
    };
    restart: {
      policy: 'always' | 'on-failure' | 'unless-stopped' | 'never';
      maxRetries: number;
      backoff: 'linear' | 'exponential';
    };
  };
  
  monitoring: {
    metrics: {
      enabled: boolean;
      endpoint?: string;
      interval: number;
    };
    logging: {
      level: 'debug' | 'info' | 'warn' | 'error';
      file?: string;
      rotation?: {
        maxSize: string;
        maxFiles: number;
      };
    };
    alerts: {
      cpu: number;
      memory: number;
      errorRate: number;
      responseTime: number;
    };
  };
  
  security: {
    runAsUser?: string;
    capabilities?: string[];
    secrets?: string[];
    allowPrivileged?: boolean;
  };
  
  metadata: {
    labels: Record<string, string>;
    annotations: Record<string, string>;
    priority: number;
    category: 'infrastructure' | 'core' | 'api' | 'worker' | 'frontend';
  };
}

interface ServiceRegistry {
  services: Map<string, ServiceDefinition>;
  instances: Map<string, ServiceInstance[]>;
  endpoints: Map<string, ServiceEndpoint>;
}

interface ServiceInstance {
  id: string;
  serviceName: string;
  status: 'pending' | 'running' | 'stopping' | 'stopped' | 'error';
  startTime: Date;
  pid?: number;
  port?: number;
  health: 'unknown' | 'healthy' | 'unhealthy';
  lastHealthCheck: Date;
  restartCount: number;
  metrics: {
    cpu: number;
    memory: number;
    network: {
      bytesIn: number;
      bytesOut: number;
      connections: number;
    };
    uptime: number;
  };
}

interface ServiceEndpoint {
  serviceName: string;
  url: string;
  protocol: string;
  health: boolean;
  loadBalanced: boolean;
  instances: string[];
}

export class ServiceConfigManager extends EventEmitter {
  private configPath: string;
  private registry: ServiceRegistry;
  private configWatcher?: fs.FileHandle;
  
  constructor(configPath: string = './config/services.yaml') {
    super();
    this.configPath = configPath;
    this.registry = {
      services: new Map(),
      instances: new Map(),
      endpoints: new Map()
    };
  }

  async initialize(): Promise<void> {
    await this.ensureConfigDirectory();
    await this.loadConfiguration();
    await this.generateDefaultConfigs();
    this.startConfigWatcher();
    
    this.emit('initialized');
  }

  private async ensureConfigDirectory(): Promise<void> {
    const configDir = path.dirname(this.configPath);
    try {
      await fs.access(configDir);
    } catch {
      await fs.mkdir(configDir, { recursive: true });
    }
  }

  private async loadConfiguration(): Promise<void> {
    try {
      const configData = await fs.readFile(this.configPath, 'utf-8');
      const config = yaml.load(configData) as { services: ServiceDefinition[] };
      
      for (const service of config.services) {
        this.registry.services.set(service.name, service);
        this.registry.instances.set(service.name, []);
      }
      
      this.emit('configLoaded', this.registry.services.size);
    } catch (error) {
      console.log('No existing config found, will generate default');
    }
  }

  private async generateDefaultConfigs(): Promise<void> {
    if (this.registry.services.size === 0) {
      const defaultServices = this.createDefaultServiceConfigs();
      
      for (const service of defaultServices) {
        this.registry.services.set(service.name, service);
        this.registry.instances.set(service.name, []);
      }
      
      await this.saveConfiguration();
    }
  }

  private createDefaultServiceConfigs(): ServiceDefinition[] {
    return [
      // Infrastructure Services
      {
        name: 'postgresql',
        version: '17.0',
        description: 'PostgreSQL database with pgvector extension',
        type: 'windows',
        execution: {
          command: 'net',
          args: ['start', 'postgresql-x64-17'],
          shell: true
        },
        network: {
          port: 5432,
          protocol: 'tcp',
          healthCheck: {
            path: 'postgresql://legal_admin:123456@localhost:5432/legal_ai_db',
            interval: 30000,
            timeout: 5000,
            retries: 3
          }
        },
        dependencies: {
          required: [],
          optional: [],
          services: [],
          infrastructure: []
        },
        resources: {
          cpu: { min: '0.5', max: '2.0' },
          memory: { min: '512MB', max: '4GB' },
          disk: { space: '10GB', type: 'ssd' }
        },
        scaling: { min: 1, max: 1, targetCpu: 70, targetMemory: 80 },
        lifecycle: {
          startup: { timeout: 30000, retries: 3, delay: 5000 },
          shutdown: { timeout: 15000, gracePeriod: 10000 },
          restart: { policy: 'always', maxRetries: 3, backoff: 'exponential' }
        },
        monitoring: {
          metrics: { enabled: true, interval: 15000 },
          logging: { level: 'info', file: 'logs/postgresql.log' },
          alerts: { cpu: 80, memory: 85, errorRate: 5, responseTime: 1000 }
        },
        security: {
          runAsUser: 'postgres',
          allowPrivileged: true
        },
        metadata: {
          labels: { tier: 'data', critical: 'true' },
          annotations: { 'backup.enabled': 'true' },
          priority: 1,
          category: 'infrastructure'
        }
      },
      
      {
        name: 'redis',
        version: '7.2',
        description: 'Redis cache and session store',
        type: 'native',
        execution: {
          command: 'redis-server',
          args: ['--port', '6379', '--save', '60', '1000', '--appendonly', 'yes']
        },
        network: {
          port: 6379,
          protocol: 'tcp',
          healthCheck: {
            path: 'redis://localhost:6379',
            interval: 15000,
            timeout: 3000,
            retries: 3
          }
        },
        dependencies: {
          required: [],
          optional: [],
          services: [],
          infrastructure: []
        },
        resources: {
          cpu: { min: '0.2', max: '1.0' },
          memory: { min: '256MB', max: '2GB' }
        },
        scaling: { min: 1, max: 1, targetCpu: 60, targetMemory: 70 },
        lifecycle: {
          startup: { timeout: 15000, retries: 3, delay: 2000 },
          shutdown: { timeout: 10000, gracePeriod: 5000 },
          restart: { policy: 'always', maxRetries: 5, backoff: 'linear' }
        },
        monitoring: {
          metrics: { enabled: true, endpoint: '/metrics', interval: 15000 },
          logging: { level: 'info', file: 'logs/redis.log' },
          alerts: { cpu: 70, memory: 80, errorRate: 3, responseTime: 500 }
        },
        security: {},
        metadata: {
          labels: { tier: 'cache', critical: 'true' },
          annotations: { 'persistence.enabled': 'true' },
          priority: 1,
          category: 'infrastructure'
        }
      },

      // Go Microservices
      {
        name: 'enhanced-rag',
        version: '2.0.0',
        description: 'Enhanced RAG service with Context7 integration',
        type: 'go',
        execution: {
          command: 'go',
          args: ['run', 'cmd/enhanced-rag/main.go'],
          cwd: 'go-microservice',
          env: {
            'PORT': '8094',
            'DB_HOST': 'localhost',
            'REDIS_URL': 'redis://localhost:6379'
          }
        },
        network: {
          port: 8094,
          protocol: 'http',
          healthCheck: {
            path: '/health',
            interval: 30000,
            timeout: 5000,
            retries: 3
          }
        },
        dependencies: {
          required: ['postgresql', 'redis'],
          optional: ['qdrant'],
          services: ['vector-service'],
          infrastructure: ['ollama']
        },
        resources: {
          cpu: { min: '0.5', max: '2.0' },
          memory: { min: '512MB', max: '2GB' },
          gpu: { required: false }
        },
        scaling: { min: 1, max: 3, targetCpu: 70, targetMemory: 75 },
        lifecycle: {
          startup: { timeout: 30000, retries: 3, delay: 5000 },
          shutdown: { timeout: 15000, gracePeriod: 10000 },
          restart: { policy: 'always', maxRetries: 3, backoff: 'exponential' }
        },
        monitoring: {
          metrics: { enabled: true, endpoint: '/metrics', interval: 15000 },
          logging: { level: 'info', file: 'logs/enhanced-rag.log' },
          alerts: { cpu: 75, memory: 80, errorRate: 5, responseTime: 2000 }
        },
        security: {},
        metadata: {
          labels: { tier: 'api', service: 'rag' },
          annotations: { 'scaling.enabled': 'true' },
          priority: 2,
          category: 'core'
        }
      },

      {
        name: 'upload-service',
        version: '1.0.0',
        description: 'File upload and processing service',
        type: 'go',
        execution: {
          command: 'go',
          args: ['run', 'cmd/upload-service/main.go'],
          cwd: 'go-microservice',
          env: {
            'PORT': '8093',
            'MINIO_ENDPOINT': 'localhost:9000'
          }
        },
        network: {
          port: 8093,
          protocol: 'http',
          healthCheck: {
            path: '/health',
            interval: 30000,
            timeout: 5000,
            retries: 3
          }
        },
        dependencies: {
          required: ['minio', 'postgresql'],
          optional: [],
          services: [],
          infrastructure: []
        },
        resources: {
          cpu: { min: '0.3', max: '1.5' },
          memory: { min: '256MB', max: '1GB' },
          disk: { space: '5GB' }
        },
        scaling: { min: 1, max: 2, targetCpu: 65, targetMemory: 70 },
        lifecycle: {
          startup: { timeout: 25000, retries: 3, delay: 3000 },
          shutdown: { timeout: 15000, gracePeriod: 10000 },
          restart: { policy: 'always', maxRetries: 3, backoff: 'exponential' }
        },
        monitoring: {
          metrics: { enabled: true, endpoint: '/metrics', interval: 15000 },
          logging: { level: 'info', file: 'logs/upload-service.log' },
          alerts: { cpu: 70, memory: 75, errorRate: 3, responseTime: 3000 }
        },
        security: {
          secrets: ['minio-credentials']
        },
        metadata: {
          labels: { tier: 'api', service: 'upload' },
          annotations: { 'storage.type': 'object' },
          priority: 2,
          category: 'api'
        }
      },

      {
        name: 'gpu-tensor-service',
        version: '1.0.0',
        description: 'GPU-accelerated tensor processing service',
        type: 'go',
        execution: {
          command: 'go',
          args: ['run', 'cmd/gpu-tensor-service/main.go'],
          cwd: 'go-microservice',
          env: {
            'PORT': '8088',
            'CUDA_VISIBLE_DEVICES': '0'
          }
        },
        network: {
          port: 8088,
          protocol: 'http',
          healthCheck: {
            path: '/health',
            interval: 45000,
            timeout: 8000,
            retries: 2
          }
        },
        dependencies: {
          required: [],
          optional: ['cuda-service'],
          services: [],
          infrastructure: []
        },
        resources: {
          cpu: { min: '0.5', max: '2.0' },
          memory: { min: '1GB', max: '4GB' },
          gpu: { required: true, memory: '4GB', cores: 1 }
        },
        scaling: { min: 0, max: 1, targetCpu: 80, targetMemory: 85 },
        lifecycle: {
          startup: { timeout: 45000, retries: 2, delay: 10000 },
          shutdown: { timeout: 20000, gracePeriod: 15000 },
          restart: { policy: 'on-failure', maxRetries: 2, backoff: 'exponential' }
        },
        monitoring: {
          metrics: { enabled: true, endpoint: '/metrics', interval: 20000 },
          logging: { level: 'info', file: 'logs/gpu-tensor.log' },
          alerts: { cpu: 85, memory: 90, errorRate: 10, responseTime: 5000 }
        },
        security: {},
        metadata: {
          labels: { tier: 'compute', gpu: 'required' },
          annotations: { 'gpu.type': 'nvidia' },
          priority: 4,
          category: 'worker'
        }
      },

      // Frontend Service
      {
        name: 'sveltekit-frontend',
        version: '2.0.0',
        description: 'SvelteKit frontend application',
        type: 'node',
        execution: {
          command: 'npm',
          args: ['run', 'dev', '--', '--host', '0.0.0.0'],
          cwd: 'sveltekit-frontend'
        },
        network: {
          port: 5173,
          protocol: 'http',
          healthCheck: {
            path: '/',
            interval: 30000,
            timeout: 5000,
            retries: 3
          }
        },
        dependencies: {
          required: [],
          optional: [],
          services: ['enhanced-rag', 'upload-service'],
          infrastructure: []
        },
        resources: {
          cpu: { min: '0.3', max: '1.0' },
          memory: { min: '256MB', max: '1GB' }
        },
        scaling: { min: 1, max: 2, targetCpu: 60, targetMemory: 70 },
        lifecycle: {
          startup: { timeout: 30000, retries: 3, delay: 5000 },
          shutdown: { timeout: 10000, gracePeriod: 5000 },
          restart: { policy: 'always', maxRetries: 5, backoff: 'linear' }
        },
        monitoring: {
          metrics: { enabled: true, interval: 20000 },
          logging: { level: 'info', file: 'logs/frontend.log' },
          alerts: { cpu: 70, memory: 75, errorRate: 5, responseTime: 2000 }
        },
        security: {},
        metadata: {
          labels: { tier: 'frontend', public: 'true' },
          annotations: { 'external.access': 'true' },
          priority: 2,
          category: 'frontend'
        }
      }
    ];
  }

  private async saveConfiguration(): Promise<void> {
    const config = {
      version: '1.0',
      generated: new Date().toISOString(),
      services: Array.from(this.registry.services.values())
    };
    
    const yamlData = yaml.dump(config, { indent: 2, lineWidth: -1 });
    await fs.writeFile(this.configPath, yamlData, 'utf-8');
  }

  private startConfigWatcher(): void {
    // Watch for config file changes and reload
    fs.watchFile(this.configPath, { interval: 5000 }, async () => {
      try {
        await this.loadConfiguration();
        this.emit('configReloaded');
      } catch (error) {
        this.emit('configError', error);
      }
    });
  }

  // Service Management API
  getService(name: string): ServiceDefinition | undefined {
    return this.registry.services.get(name);
  }

  getAllServices(): ServiceDefinition[] {
    return Array.from(this.registry.services.values());
  }

  getServicesByCategory(category: string): ServiceDefinition[] {
    return this.getAllServices().filter(s => s.metadata.category === category);
  }

  getServicesByPriority(): ServiceDefinition[] {
    return this.getAllServices().sort((a, b) => a.metadata.priority - b.metadata.priority);
  }

  getDependencyGraph(): Map<string, string[]> {
    const graph = new Map<string, string[]>();
    
    for (const service of this.registry.services.values()) {
      const deps = [
        ...service.dependencies.required,
        ...service.dependencies.services,
        ...service.dependencies.infrastructure
      ];
      graph.set(service.name, deps);
    }
    
    return graph;
  }

  getStartupOrder(): string[] {
    const graph = this.getDependencyGraph();
    const visited = new Set<string>();
    const order: string[] = [];
    
    const visit = (name: string): void => {
      if (visited.has(name)) return;
      visited.add(name);
      
      const deps = graph.get(name) || [];
      for (const dep of deps) {
        if (graph.has(dep)) {
          visit(dep);
        }
      }
      
      order.push(name);
    };
    
    for (const name of graph.keys()) {
      visit(name);
    }
    
    return order;
  }

  validateConfiguration(): { valid: boolean; errors: string[] } {
    const errors: string[] = [];
    
    for (const service of this.registry.services.values()) {
      // Check required fields
      if (!service.name) {
        errors.push(`Service missing name`);
      }
      
      if (!service.execution?.command) {
        errors.push(`Service ${service.name} missing execution command`);
      }
      
      // Check dependencies exist
      for (const dep of service.dependencies.required) {
        if (!this.registry.services.has(dep)) {
          errors.push(`Service ${service.name} has unknown dependency: ${dep}`);
        }
      }
      
      // Check port conflicts
      if (service.network.port) {
        const otherServices = Array.from(this.registry.services.values())
          .filter(s => s.name !== service.name && s.network.port === service.network.port);
        
        if (otherServices.length > 0) {
          errors.push(`Port ${service.network.port} conflict between ${service.name} and ${otherServices[0].name}`);
        }
      }
      
      // Check resource constraints
      if (service.resources.gpu?.required) {
        // In a real implementation, check GPU availability
      }
    }
    
    return { valid: errors.length === 0, errors };
  }

  // Dynamic configuration updates
  async updateService(name: string, updates: Partial<ServiceDefinition>): Promise<void> {
    const service = this.registry.services.get(name);
    if (!service) {
      throw new Error(`Service ${name} not found`);
    }
    
    const updatedService = { ...service, ...updates };
    this.registry.services.set(name, updatedService);
    
    await this.saveConfiguration();
    this.emit('serviceUpdated', name, updatedService);
  }

  async addService(service: ServiceDefinition): Promise<void> {
    if (this.registry.services.has(service.name)) {
      throw new Error(`Service ${service.name} already exists`);
    }
    
    this.registry.services.set(service.name, service);
    this.registry.instances.set(service.name, []);
    
    await this.saveConfiguration();
    this.emit('serviceAdded', service.name);
  }

  async removeService(name: string): Promise<void> {
    if (!this.registry.services.has(name)) {
      throw new Error(`Service ${name} not found`);
    }
    
    this.registry.services.delete(name);
    this.registry.instances.delete(name);
    
    await this.saveConfiguration();
    this.emit('serviceRemoved', name);
  }

  // Instance management
  registerInstance(instance: ServiceInstance): void {
    const instances = this.registry.instances.get(instance.serviceName) || [];
    instances.push(instance);
    this.registry.instances.set(instance.serviceName, instances);
    
    this.emit('instanceRegistered', instance);
  }

  updateInstanceStatus(serviceId: string, instanceId: string, status: ServiceInstance['status']): void {
    const instances = this.registry.instances.get(serviceId) || [];
    const instance = instances.find(i => i.id === instanceId);
    
    if (instance) {
      instance.status = status;
      this.emit('instanceStatusChanged', instance);
    }
  }

  getServiceInstances(serviceName: string): ServiceInstance[] {
    return this.registry.instances.get(serviceName) || [];
  }

  // Service discovery and endpoints
  registerEndpoint(endpoint: ServiceEndpoint): void {
    this.registry.endpoints.set(endpoint.serviceName, endpoint);
    this.emit('endpointRegistered', endpoint);
  }

  getEndpoint(serviceName: string): ServiceEndpoint | undefined {
    return this.registry.endpoints.get(serviceName);
  }

  getAllEndpoints(): ServiceEndpoint[] {
    return Array.from(this.registry.endpoints.values());
  }

  // Configuration export/import
  async exportConfiguration(format: 'yaml' | 'json' = 'yaml'): Promise<string> {
    const config = {
      version: '1.0',
      exported: new Date().toISOString(),
      services: Array.from(this.registry.services.values())
    };
    
    if (format === 'json') {
      return JSON.stringify(config, null, 2);
    } else {
      return yaml.dump(config, { indent: 2, lineWidth: -1 });
    }
  }

  async importConfiguration(configData: string, format: 'yaml' | 'json' = 'yaml'): Promise<void> {
    let config: any;
    
    if (format === 'json') {
      config = JSON.parse(configData);
    } else {
      config = yaml.load(configData);
    }
    
    // Validate and update registry
    this.registry.services.clear();
    this.registry.instances.clear();
    
    for (const service of config.services) {
      this.registry.services.set(service.name, service);
      this.registry.instances.set(service.name, []);
    }
    
    await this.saveConfiguration();
    this.emit('configurationImported');
  }

  // Cleanup
  async cleanup(): Promise<void> {
    if (this.configWatcher) {
      fs.unwatchFile(this.configPath);
    }
    
    this.removeAllListeners();
  }
}

export default ServiceConfigManager;