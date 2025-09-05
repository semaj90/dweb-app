// Neo4j Desktop Connection Service
// Handles connection to Neo4j Desktop with fallback to Redis caching

import neo4j, { type Driver, type Session } from 'neo4j-driver'

export interface Neo4jConfig {
  uri: string;
  username: string;  
  password: string;
  database?: string;
  enabled: boolean;
}

export interface Neo4jHealth {
  connected: boolean;
  version?: string;
  database?: string;
  error?: string;
  mode: 'neo4j-desktop' | 'redis-fallback' | 'disabled';
}

export class Neo4jDesktopService {
  private driver: Driver | null = null;
  private config: Neo4jConfig;
  private connectionAttempted = false;

  constructor(config: Neo4jConfig) {
    this.config = config;
  }

  /**
   * Initialize connection to Neo4j Desktop
   */
  async initialize(): Promise<boolean> {
    if (!this.config.enabled) {
      console.log('🔌 Neo4j Desktop: Disabled in configuration');
      return false;
    }

    try {
      console.log(`🔌 Neo4j Desktop: Connecting to ${this.config.uri}...`);
      
      this.driver = neo4j.driver(
        this.config.uri,
        neo4j.auth.basic(this.config.username, this.config.password),
        {
          // Connection settings optimized for Neo4j Desktop
          maxConnectionLifetime: 3 * 60 * 60 * 1000, // 3 hours
          maxConnectionPoolSize: 50,
          connectionAcquisitionTimeout: 2 * 60 * 1000, // 2 minutes
          disableLosslessIntegers: true
        }
      );

      // Test the connection
      await this.testConnection();
      this.connectionAttempted = true;

      console.log('✅ Neo4j Desktop: Connected successfully');
      return true;

    } catch (error) {
      console.error('❌ Neo4j Desktop: Connection failed:', error);
      this.connectionAttempted = true;
      
      // Don't throw - let the app continue with Redis fallback
      return false;
    }
  }

  /**
   * Test connection to Neo4j Desktop
   */
  private async testConnection(): Promise<void> {
    if (!this.driver) {
      throw new Error('No driver available');
    }

    const session = this.driver.session({
      database: this.config.database || 'neo4j',
      defaultAccessMode: neo4j.session.READ
    });

    try {
      const result = await session.run('RETURN 1 as test');
      const testValue = result.records[0]?.get('test')?.toNumber();
      
      if (testValue !== 1) {
        throw new Error('Invalid test result');
      }
    } finally {
      await session.close();
    }
  }

  /**
   * Get health status of Neo4j Desktop connection
   */
  async getHealthStatus(): Promise<Neo4jHealth> {
    if (!this.config.enabled) {
      return {
        connected: false,
        mode: 'disabled',
        error: 'Neo4j disabled in configuration'
      };
    }

    if (!this.driver) {
      if (this.connectionAttempted) {
        return {
          connected: false,
          mode: 'redis-fallback',
          error: 'Connection failed - using Redis fallback'
        };
      } else {
        // Try to initialize if not attempted yet
        const connected = await this.initialize();
        if (!connected) {
          return {
            connected: false,
            mode: 'redis-fallback', 
            error: 'Failed to connect - using Redis fallback'
          };
        }
      }
    }

    // Test current connection
    try {
      const session = this.driver!.session({
        database: this.config.database || 'neo4j',
        defaultAccessMode: neo4j.session.READ
      });

      try {
        // Get Neo4j version and database info
        const versionResult = await session.run('CALL dbms.components() YIELD name, versions RETURN name, versions[0] as version');
        const dbInfoResult = await session.run('CALL db.info() YIELD name RETURN name');

        const version = versionResult.records[0]?.get('version') || 'Unknown';
        const database = dbInfoResult.records[0]?.get('name') || this.config.database;

        return {
          connected: true,
          version,
          database,
          mode: 'neo4j-desktop'
        };

      } finally {
        await session.close();
      }

    } catch (error) {
      console.error('Neo4j health check failed:', error);
      return {
        connected: false,
        mode: 'redis-fallback',
        error: error instanceof Error ? error.message : 'Unknown error'
      };
    }
  }

  /**
   * Execute a Cypher query (with Redis fallback awareness)
   */
  async executeQuery(cypher: string, parameters = {}): Promise<any[]> {
    if (!this.driver) {
      throw new Error('Neo4j not available - query should be cached in Redis');
    }

    const session = this.driver.session({
      database: this.config.database || 'neo4j',
      defaultAccessMode: cypher.trim().toUpperCase().startsWith('CREATE') || 
                        cypher.trim().toUpperCase().startsWith('MERGE') ||
                        cypher.trim().toUpperCase().startsWith('DELETE') ||
                        cypher.trim().toUpperCase().startsWith('SET') 
                        ? neo4j.session.WRITE 
                        : neo4j.session.READ
    });

    try {
      const result = await session.run(cypher, parameters);
      return result.records.map(record => record.toObject());
    } finally {
      await session.close();
    }
  }

  /**
   * Create nodes and relationships for legal AI data
   */
  async createLegalCase(caseData: {
    caseId: string;
    title: string;
    description?: string;
    status: string;
  }): Promise<void> {
    const cypher = `
      MERGE (c:Case {caseId: $caseId})
      SET c.title = $title,
          c.description = $description,
          c.status = $status,
          c.updatedAt = datetime()
      RETURN c
    `;

    await this.executeQuery(cypher, caseData);
  }

  /**
   * Find related cases using graph traversal
   */
  async findRelatedCases(caseId: string, maxDepth = 2): Promise<any[]> {
    const cypher = `
      MATCH (c:Case {caseId: $caseId})
      MATCH path = (c)-[*1..${maxDepth}]-(related:Case)
      WHERE related.caseId <> $caseId
      RETURN DISTINCT related.caseId as relatedCaseId, 
             related.title as title,
             related.status as status,
             length(path) as depth
      ORDER BY depth ASC, related.title ASC
      LIMIT 10
    `;

    return await this.executeQuery(cypher, { caseId });
  }

  /**
   * Close connection
   */
  async close(): Promise<void> {
    if (this.driver) {
      await this.driver.close();
      this.driver = null;
      console.log('🔌 Neo4j Desktop: Connection closed');
    }
  }
}

// Factory function to create Neo4j service from environment
export function createNeo4jService(): Neo4jDesktopService {
  const config: Neo4jConfig = {
    uri: process.env.NEO4J_URL || 'neo4j://localhost:7687',
    username: process.env.NEO4J_USERNAME || 'neo4j',
    password: process.env.NEO4J_PASSWORD || 'neo4j', 
    database: process.env.NEO4J_DATABASE || 'neo4j',
    enabled: process.env.NEO4J_ENABLED === 'true' && process.env.NEO4J_DESKTOP_MODE === 'true'
  };

  return new Neo4jDesktopService(config);
}

// Singleton instance for app-wide use
let neo4jServiceInstance: Neo4jDesktopService | null = null;

export function getNeo4jService(): Neo4jDesktopService {
  if (!neo4jServiceInstance) {
    neo4jServiceInstance = createNeo4jService();
  }
  return neo4jServiceInstance;
}

export default Neo4jDesktopService;