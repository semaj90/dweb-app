// Service Validation Script
// Comprehensive validation of all external service connections

import postgres from 'postgres';
import { createClient } from 'redis';

interface ServiceStatus {
  name: string;
  status: 'healthy' | 'degraded' | 'unhealthy';
  responseTime?: number;
  version?: string;
  error?: string;
  details?: Record<string, any>;
}

interface ValidationReport {
  overall: 'healthy' | 'degraded' | 'unhealthy';
  services: ServiceStatus[];
  timestamp: string;
  environment: string;
  summary: {
    healthy: number;
    degraded: number;
    unhealthy: number;
    total: number;
  };
}

class ServiceValidator {
  private results: ServiceStatus[] = [];

  /**
   * Run comprehensive service validation
   */
  async validateAllServices(): Promise<ValidationReport> {
    console.log('🔍 Starting comprehensive service validation...\n');

    const services = [
      () => this.validatePostgreSQL(),
      () => this.validateQdrant(),
      () => this.validateRedis(),
      () => this.validateOllama(),
      () => this.validateNeo4j(),
      () => this.validateMinio(),
      () => this.validateRabbitMQ(),
      () => this.validateWebServices()
    ];

    // Run all validations in parallel with timeout
    const validationPromises = services.map(validator =>
      this.withTimeout(validator(), 30000)
    );

    this.results = await Promise.all(validationPromises);

    return this.generateReport();
  }

  /**
   * Validate PostgreSQL connection and configuration
   */
  async validatePostgreSQL(): Promise<ServiceStatus> {
    const service: ServiceStatus = {
      name: 'PostgreSQL',
      status: 'unhealthy'
    };

    try {
      const startTime = Date.now();
      
      const connectionString = process.env.DATABASE_URL || 
        `postgresql://${process.env.DB_USER || 'postgres'}:${process.env.DB_PASSWORD || 'postgres'}@${process.env.DB_HOST || 'localhost'}:${process.env.DB_PORT || 5432}/${process.env.DB_NAME || 'legal_ai'}`;

      const client = postgres(connectionString, {
        max: 1,
        connect_timeout: 10,
        idle_timeout: 10
      });

      // Test basic connection
      const result = await client`SELECT version(), current_database(), current_user`;
      const responseTime = Date.now() - startTime;

      // Test required extensions
      const extensions = await client`
        SELECT extname FROM pg_extension 
        WHERE extname IN ('vector', 'uuid-ossp')
      `;

      // Test table existence
      const tables = await client`
        SELECT table_name FROM information_schema.tables 
        WHERE table_schema = 'public' AND table_name IN (
          'legal_documents', 'content_embeddings', 'search_sessions'
        )
      `;

      await client.end();

      service.status = 'healthy';
      service.responseTime = responseTime;
      service.version = result[0].version.split(' ')[1];
      service.details = {
        database: result[0].current_database,
        user: result[0].current_user,
        extensions: extensions.map(e => e.extname),
        tables: tables.map(t => t.table_name),
        vectorSupport: extensions.some(e => e.extname === 'vector'),
        tablesCount: tables.length
      };

      console.log(`✅ PostgreSQL: Connected (${responseTime}ms)`);

    } catch (error: any) {
      service.error = error.message;
      console.log(`❌ PostgreSQL: ${error.message}`);
    }

    return service;
  }

  /**
   * Validate Qdrant vector database connection
   */
  async validateQdrant(): Promise<ServiceStatus> {
    const service: ServiceStatus = {
      name: 'Qdrant',
      status: 'unhealthy'
    };

    try {
      const startTime = Date.now();
      const baseUrl = process.env.QDRANT_URL || 'http://localhost:6333';

      // Test basic connection
      const response = await fetch(`${baseUrl}/collections`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' }
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const responseTime = Date.now() - startTime;
      const collections = await response.json();

      // Test legal_documents collection
      const collectionResponse = await fetch(`${baseUrl}/collections/legal_documents`);
      const collectionExists = collectionResponse.ok;
      let collectionInfo = null;

      if (collectionExists) {
        collectionInfo = await collectionResponse.json();
      }

      service.status = 'healthy';
      service.responseTime = responseTime;
      service.details = {
        url: baseUrl,
        collectionsCount: collections.result?.length || 0,
        legalCollectionExists: collectionExists,
        vectorCount: collectionInfo?.result?.vectors_count || 0,
        collectionStatus: collectionInfo?.result?.status || 'unknown'
      };

      console.log(`✅ Qdrant: Connected (${responseTime}ms)`);

    } catch (error: any) {
      service.error = error.message;
      console.log(`❌ Qdrant: ${error.message}`);
    }

    return service;
  }

  /**
   * Validate Redis connection and configuration
   */
  async validateRedis(): Promise<ServiceStatus> {
    const service: ServiceStatus = {
      name: 'Redis',
      status: 'unhealthy'
    };

    try {
      const startTime = Date.now();
      const redisUrl = process.env.REDIS_URL || 'redis://localhost:6379';

      const client = createClient({ url: redisUrl });
      await client.connect();

      // Test basic operations
      await client.set('health_check', 'ok', { EX: 10 });
      const value = await client.get('health_check');
      
      if (value !== 'ok') {
        throw new Error('Redis read/write test failed');
      }

      const responseTime = Date.now() - startTime;
      const info = await client.info();
      
      await client.disconnect();

      service.status = 'healthy';
      service.responseTime = responseTime;
      service.details = {
        url: redisUrl,
        connected: true,
        memory: this.parseRedisInfo(info, 'used_memory_human'),
        version: this.parseRedisInfo(info, 'redis_version')
      };

      console.log(`✅ Redis: Connected (${responseTime}ms)`);

    } catch (error: any) {
      service.error = error.message;
      console.log(`❌ Redis: ${error.message}`);
    }

    return service;
  }

  /**
   * Validate Ollama AI service connection
   */
  async validateOllama(): Promise<ServiceStatus> {
    const service: ServiceStatus = {
      name: 'Ollama',
      status: 'unhealthy'
    };

    try {
      const startTime = Date.now();
      const ollamaUrl = process.env.OLLAMA_URL || 'http://localhost:11434';

      // Test basic connection
      const response = await fetch(`${ollamaUrl}/api/tags`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const responseTime = Date.now() - startTime;
      const models = await response.json();

      // Test specific model availability
      const hasGemma = models.models?.some((m: any) => m.name.includes('gemma'));
      const hasEmbedding = models.models?.some((m: any) => m.name.includes('embed'));

      service.status = 'healthy';
      service.responseTime = responseTime;
      service.details = {
        url: ollamaUrl,
        modelsCount: models.models?.length || 0,
        hasLegalModel: hasGemma,
        hasEmbeddingModel: hasEmbedding,
        models: models.models?.map((m: any) => m.name) || []
      };

      console.log(`✅ Ollama: Connected (${responseTime}ms) - ${models.models?.length || 0} models`);

    } catch (error: any) {
      service.error = error.message;
      console.log(`❌ Ollama: ${error.message}`);
    }

    return service;
  }

  /**
   * Validate Neo4j graph database connection
   */
  async validateNeo4j(): Promise<ServiceStatus> {
    const service: ServiceStatus = {
      name: 'Neo4j',
      status: 'unhealthy'
    };

    try {
      const startTime = Date.now();
      const neo4jUrl = process.env.NEO4J_URI || 'bolt://localhost:7687';
      
      // For this validation, we'll use HTTP API if available
      const httpUrl = neo4jUrl.replace('bolt://', 'http://').replace(':7687', ':7474');
      
      const response = await fetch(`${httpUrl}/db/data/`, {
        headers: {
          'Authorization': `Basic ${Buffer.from(`${process.env.NEO4J_USER || 'neo4j'}:${process.env.NEO4J_PASSWORD || 'password'}`).toString('base64')}`
        }
      });

      const responseTime = Date.now() - startTime;

      if (response.ok) {
        const data = await response.json();
        service.status = 'healthy';
        service.responseTime = responseTime;
        service.details = {
          url: neo4jUrl,
          version: data.neo4j_version,
          connected: true
        };
        console.log(`✅ Neo4j: Connected (${responseTime}ms)`);
      } else {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

    } catch (error: any) {
      service.error = error.message;
      service.status = 'degraded'; // Neo4j is optional for core functionality
      console.log(`⚠️ Neo4j: ${error.message} (optional service)`);
    }

    return service;
  }

  /**
   * Validate MinIO object storage connection
   */
  async validateMinio(): Promise<ServiceStatus> {
    const service: ServiceStatus = {
      name: 'MinIO',
      status: 'unhealthy'
    };

    try {
      const startTime = Date.now();
      const minioUrl = process.env.MINIO_ENDPOINT || 'http://localhost:9000';

      const response = await fetch(`${minioUrl}/minio/health/live`);
      const responseTime = Date.now() - startTime;

      if (response.ok) {
        service.status = 'healthy';
        service.responseTime = responseTime;
        service.details = {
          url: minioUrl,
          connected: true
        };
        console.log(`✅ MinIO: Connected (${responseTime}ms)`);
      } else {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

    } catch (error: any) {
      service.error = error.message;
      service.status = 'degraded'; // MinIO is optional for core functionality
      console.log(`⚠️ MinIO: ${error.message} (optional service)`);
    }

    return service;
  }

  /**
   * Validate RabbitMQ message queue connection
   */
  async validateRabbitMQ(): Promise<ServiceStatus> {
    const service: ServiceStatus = {
      name: 'RabbitMQ',
      status: 'unhealthy'
    };

    try {
      const startTime = Date.now();
      const rabbitUrl = process.env.RABBITMQ_URL || 'http://localhost:15672';

      // Try to access RabbitMQ management API
      const response = await fetch(`${rabbitUrl}/api/overview`, {
        headers: {
          'Authorization': `Basic ${Buffer.from(`${process.env.RABBITMQ_USER || 'guest'}:${process.env.RABBITMQ_PASSWORD || 'guest'}`).toString('base64')}`
        }
      });

      const responseTime = Date.now() - startTime;

      if (response.ok) {
        const data = await response.json();
        service.status = 'healthy';
        service.responseTime = responseTime;
        service.details = {
          url: rabbitUrl,
          version: data.rabbitmq_version,
          connected: true
        };
        console.log(`✅ RabbitMQ: Connected (${responseTime}ms)`);
      } else {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

    } catch (error: any) {
      service.error = error.message;
      service.status = 'degraded'; // RabbitMQ is optional for core functionality
      console.log(`⚠️ RabbitMQ: ${error.message} (optional service)`);
    }

    return service;
  }

  /**
   * Validate web services and external APIs
   */
  async validateWebServices(): Promise<ServiceStatus> {
    const service: ServiceStatus = {
      name: 'Web Services',
      status: 'unhealthy'
    };

    try {
      const startTime = Date.now();
      const checks = [];

      // Check internet connectivity
      try {
        const googleResponse = await fetch('https://www.google.com', {
          method: 'HEAD',
          signal: AbortSignal.timeout(5000)
        });
        checks.push({ name: 'Internet', status: googleResponse.ok });
      } catch {
        checks.push({ name: 'Internet', status: false });
      }

      // Check if local web server is running (if in development)
      try {
        const localResponse = await fetch('http://localhost:5173/api/health', {
          method: 'GET',
          signal: AbortSignal.timeout(3000)
        });
        checks.push({ name: 'Local Server', status: localResponse.ok });
      } catch {
        checks.push({ name: 'Local Server', status: false });
      }

      const responseTime = Date.now() - startTime;
      const healthyChecks = checks.filter(c => c.status).length;

      service.status = healthyChecks > 0 ? 'healthy' : 'unhealthy';
      service.responseTime = responseTime;
      service.details = {
        checks,
        healthyCount: healthyChecks,
        totalCount: checks.length
      };

      console.log(`${healthyChecks > 0 ? '✅' : '❌'} Web Services: ${healthyChecks}/${checks.length} checks passed`);

    } catch (error: any) {
      service.error = error.message;
      console.log(`❌ Web Services: ${error.message}`);
    }

    return service;
  }

  /**
   * Generate comprehensive validation report
   */
  private generateReport(): ValidationReport {
    const summary = this.results.reduce(
      (acc, service) => {
        acc[service.status]++;
        acc.total++;
        return acc;
      },
      { healthy: 0, degraded: 0, unhealthy: 0, total: 0 }
    );

    let overall: 'healthy' | 'degraded' | 'unhealthy' = 'healthy';
    
    if (summary.unhealthy > 0) {
      const criticalServices = this.results.filter(s => 
        s.status === 'unhealthy' && ['PostgreSQL', 'Qdrant'].includes(s.name)
      );
      overall = criticalServices.length > 0 ? 'unhealthy' : 'degraded';
    } else if (summary.degraded > 0) {
      overall = 'degraded';
    }

    return {
      overall,
      services: this.results,
      timestamp: new Date().toISOString(),
      environment: process.env.NODE_ENV || 'development',
      summary
    };
  }

  /**
   * Utility: Add timeout to promise
   */
  private async withTimeout<T>(promise: Promise<T>, timeoutMs: number): Promise<T> {
    const timeoutPromise = new Promise<never>((_, reject) =>
      setTimeout(() => reject(new Error('Timeout')), timeoutMs)
    );

    return Promise.race([promise, timeoutPromise]);
  }

  /**
   * Utility: Parse Redis info string
   */
  private parseRedisInfo(info: string, key: string): string {
    const lines = info.split('\r\n');
    const line = lines.find(l => l.startsWith(`${key}:`));
    return line ? line.split(':')[1] : 'unknown';
  }

  /**
   * Print detailed validation report
   */
  printReport(report: ValidationReport): void {
    console.log('\n📊 Service Validation Report');
    console.log('='.repeat(50));
    console.log(`Overall Status: ${this.getStatusEmoji(report.overall)} ${report.overall.toUpperCase()}`);
    console.log(`Environment: ${report.environment}`);
    console.log(`Timestamp: ${report.timestamp}`);
    console.log(`Summary: ${report.summary.healthy} healthy, ${report.summary.degraded} degraded, ${report.summary.unhealthy} unhealthy\n`);

    report.services.forEach(service => {
      console.log(`${this.getStatusEmoji(service.status)} ${service.name}`);
      console.log(`   Status: ${service.status}`);
      if (service.responseTime) {
        console.log(`   Response Time: ${service.responseTime}ms`);
      }
      if (service.version) {
        console.log(`   Version: ${service.version}`);
      }
      if (service.error) {
        console.log(`   Error: ${service.error}`);
      }
      if (service.details) {
        Object.entries(service.details).forEach(([key, value]) => {
          console.log(`   ${key}: ${JSON.stringify(value)}`);
        });
      }
      console.log('');
    });

    console.log('='.repeat(50));
  }

  private getStatusEmoji(status: string): string {
    switch (status) {
      case 'healthy': return '✅';
      case 'degraded': return '⚠️';
      case 'unhealthy': return '❌';
      default: return '❓';
    }
  }
}

// CLI interface
async function main() {
  const validator = new ServiceValidator();
  
  try {
    const report = await validator.validateAllServices();
    validator.printReport(report);

    // Exit with appropriate code
    if (report.overall === 'unhealthy') {
      console.log('❌ Validation failed - critical services are unavailable');
      process.exit(1);
    } else if (report.overall === 'degraded') {
      console.log('⚠️ Validation passed with warnings - some optional services are unavailable');
      process.exit(0);
    } else {
      console.log('✅ All services validated successfully');
      process.exit(0);
    }

  } catch (error) {
    console.error('❌ Service validation failed:', error);
    process.exit(1);
  }
}

// Export for use in other modules
export { ServiceValidator, type ValidationReport, type ServiceStatus };

// Run CLI if this file is executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}
