
import pg from 'pg';
import { createClient } from 'redis';
import neo4j from 'neo4j-driver';
import type { QueryResult } from 'pg';

/**
 * Unified Database Service
 * Consolidates PostgreSQL, Redis, Neo4j, and Qdrant operations
 */
export class UnifiedDatabaseService {
  private pgPool: pg.Pool | null = null;
  private redisClient: unknown = null;
  private neo4jDriver: unknown = null;
  private qdrantConfig: unknown = {};
  private initialized = false;

  constructor(config: unknown = {}) {
    // PostgreSQL configuration
    this.pgPool = new pg.Pool({
      host: config.pg?.host || process.env.DB_HOST || 'localhost',
      port: config.pg?.port || parseInt(process.env.DB_PORT || '5432'),
      database: config.pg?.database || process.env.DB_NAME || 'legal_ai_db',
      user: config.pg?.user || process.env.DB_USER || 'legal_admin',
      password: config.pg?.password || process.env.POSTGRES_PASSWORD || '123456',
      max: 20,
      idleTimeoutMillis: 30000,
      connectionTimeoutMillis: 2000,
    });

    // Neo4j configuration
    this.neo4jDriver = neo4j.driver(
      config.neo4j?.uri || process.env.NEO4J_URI || 'bolt://localhost:7687',
      neo4j.auth.basic(
        config.neo4j?.user || process.env.NEO4J_USER || 'neo4j',
        config.neo4j?.password || process.env.NEO4J_PASSWORD || 'password'
      )
    );

    // Qdrant configuration
    this.qdrantConfig = {
      baseUrl: config.qdrant?.url || process.env.QDRANT_URL || 'http://localhost:6333',
      collection: config.qdrant?.collection || 'legal_documents'
    };
  }

  async initialize(): Promise<boolean> {
    try {
      // Test PostgreSQL connection
      if (this.pgPool) {
        const client = await this.pgPool.connect();
        await client.query('SELECT NOW()');
        client.release();
        console.log('✓ PostgreSQL connected');
      }

      // Initialize Redis
      this.redisClient = createClient({ 
        url: process.env.REDIS_URL || 'redis://localhost:6379' 
      });
      this.redisClient.on('error', (err: Error) => console.error('Redis Error:', err));
      await this.redisClient.connect();
      console.log('✓ Redis connected');

      // Test Neo4j
      const neo4jSession = this.neo4jDriver.session();
      await neo4jSession.run('RETURN 1');
      await neo4jSession.close();
      console.log('✓ Neo4j connected');

      // Test Qdrant
      const qdrantResponse = await fetch(`${this.qdrantConfig.baseUrl}/collections`);
      if (qdrantResponse.ok) {
        console.log('✓ Qdrant connected');
      }

      this.initialized = true;
      return true;
    } catch (error) {
      console.error('Database initialization failed:', error);
      throw error;
    }
  }

  // ============ PostgreSQL Methods ============
  async query(text: string, params: unknown[] = []): Promise<QueryResult> {
    if (!this.pgPool) throw new Error('PostgreSQL not initialized');
    return await this.pgPool.query(text, params);
  }

  async transaction<T>(callback: (client: pg.PoolClient) => Promise<T>): Promise<T> {
    if (!this.pgPool) throw new Error('PostgreSQL not initialized');
    
    const client = await this.pgPool.connect();
    try {
      await client.query('BEGIN');
      const result = await callback(client);
      await client.query('COMMIT');
      return result;
    } catch (error) {
      await client.query('ROLLBACK');
      throw error;
    } finally {
      client.release();
    }
  }

  // Legal document operations
  async insertLegalDocument(document: unknown): Promise<any> {
    const query = `
      INSERT INTO legal_documents (id, title, content, metadata, embedding, case_id, created_at)
      VALUES ($1, $2, $3, $4, $5, $6, $7)
      RETURNING *
    `;
    
    const values = [
      document.id,
      document.title,
      document.content,
      JSON.stringify(document.metadata),
      document.embedding ? JSON.stringify(document.embedding) : null,
      document.case_id,
      new Date()
    ];

    return await this.query(query, values);
  }

  async searchLegalDocuments(query: string, caseId?: string): Promise<unknown[]> {
    let sqlQuery = `
      SELECT id, title, content, metadata, case_id, 
             ts_rank(to_tsvector('english', content), plainto_tsquery('english', $1)) as rank
      FROM legal_documents
      WHERE to_tsvector('english', content) @@ plainto_tsquery('english', $1)
    `;
    
    const params = [query];
    
    if (caseId) {
      sqlQuery += ` AND case_id = $2`;
      params.push(caseId);
    }
    
    sqlQuery += ` ORDER BY rank DESC LIMIT 20`;

    const result = await this.query(sqlQuery, params);
    return result.rows;
  }

  // ============ Redis Methods ============
  async getCached(key: string): Promise<any> {
    if (!this.redisClient) return null;
    try {
      const value = await this.redisClient.get(key);
      return value ? JSON.parse(value) : null;
    } catch (error) {
      console.error('Redis get error:', error);
      return null;
    }
  }

  async setCached(key: string, value: unknown, ttl: number = 3600): Promise<boolean> {
    if (!this.redisClient) return false;
    try {
      const serialized = JSON.stringify(value);
      if (ttl > 0) {
        await this.redisClient.setEx(key, ttl, serialized);
      } else {
        await this.redisClient.set(key, serialized);
      }
      return true;
    } catch (error) {
      console.error('Redis set error:', error);
      return false;
    }
  }

  async invalidateCache(pattern: string): Promise<void> {
    if (!this.redisClient) return;
    try {
      const keys = await this.redisClient.keys(pattern);
      if (keys.length > 0) {
        await this.redisClient.del(keys);
      }
    } catch (error) {
      console.error('Redis invalidate error:', error);
    }
  }

  // ============ Neo4j Methods ============
  async runCypher(query: string, params: unknown = {}): Promise<unknown[]> {
    const session = this.neo4jDriver.session();
    try {
      const result = await session.run(query, params);
      return result.records;
    } finally {
      await session.close();
    }
  }

  async createNode(label: string, properties: unknown): Promise<any> {
    return await this.runCypher(
      `CREATE (n:${label} $props) RETURN n`,
      { props: properties }
    );
  }

  async createRelationship(fromId: string, toId: string, type: string, properties: unknown = {}): Promise<any> {
    return await this.runCypher(
      `
      MATCH (a {id: $fromId})
      MATCH (b {id: $toId})
      CREATE (a)-[r:${type} $props]->(b)
      RETURN r
      `,
      { fromId, toId, props: properties }
    );
  }

  // Legal case operations
  async createLegalCase(caseData: unknown): Promise<any> {
    return await this.createNode('LegalCase', {
      id: caseData.id,
      title: caseData.title,
      status: caseData.status || 'active',
      createdAt: new Date().toISOString(),
      ...caseData.properties
    });
  }

  async linkDocumentToCase(documentId: string, caseId: string): Promise<any> {
    return await this.createRelationship(documentId, caseId, 'BELONGS_TO', {
      linkedAt: new Date().toISOString()
    });
  }

  async getCaseGraph(caseId: string): Promise<any> {
    return await this.runCypher(
      `
      MATCH (c:LegalCase {id: $caseId})
      OPTIONAL MATCH (c)<-[:BELONGS_TO]-(d:Document)
      OPTIONAL MATCH (c)-[:HAS_EVIDENCE]->(e:Evidence)
      OPTIONAL MATCH (c)-[:INVOLVES]->(p:Person)
      RETURN c, collect(DISTINCT d) as documents, 
             collect(DISTINCT e) as evidence, 
             collect(DISTINCT p) as people
      `,
      { caseId }
    );
  }

  // ============ Qdrant Methods ============
  async vectorSearch(vector: number[], limit: number = 10, filter: unknown = {}): Promise<unknown[]> {
    const response = await fetch(
      `${this.qdrantConfig.baseUrl}/collections/${this.qdrantConfig.collection}/points/search`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          vector,
          limit,
          filter,
          with_payload: true
        })
      }
    );

    if (response.ok) {
      const data = await response.json();
      return data.result || [];
    }
    return [];
  }

  async upsertVector(id: string, vector: number[], payload: unknown): Promise<boolean> {
    const response = await fetch(
      `${this.qdrantConfig.baseUrl}/collections/${this.qdrantConfig.collection}/points`,
      {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          points: [{ id, vector, payload }]
        })
      }
    );
    return response.ok;
  }

  async hybridSearch(query: string, vector: number[], caseId?: string): Promise<unknown[]> {
    // Combine text search with vector similarity
    const [textResults, vectorResults] = await Promise.all([
      this.searchLegalDocuments(query, caseId),
      this.vectorSearch(vector, 10, caseId ? { case_id: caseId } : {})
    ]);

    // Merge and rank results
    const combinedResults = new Map();

    // Add text search results
    textResults.forEach((doc, index) => {
      combinedResults.set(doc.id, {
        ...doc,
        textScore: 1 - (index * 0.1),
        vectorScore: 0,
        source: 'text'
      });
    });

    // Add vector search results
    vectorResults.forEach((result, index) => {
      const existing = combinedResults.get(result.id);
      if (existing) {
        existing.vectorScore = result.score;
        existing.source = 'hybrid';
      } else {
        combinedResults.set(result.id, {
          ...result.payload,
          id: result.id,
          textScore: 0,
          vectorScore: result.score,
          source: 'vector'
        });
      }
    });

    // Sort by combined score
    return Array.from(combinedResults.values())
      .map(result => ({
        ...result,
        combinedScore: (result.textScore * 0.4) + (result.vectorScore * 0.6)
      }))
      .sort((a, b) => b.combinedScore - a.combinedScore)
      .slice(0, 10);
  }

  // ============ Health Check Methods ============
  async getHealthStatus(): Promise<any> {
    const health = {
      postgresql: false,
      redis: false,
      neo4j: false,
      qdrant: false,
      overall: 'unhealthy'
    };

    try {
      // Check PostgreSQL
      if (this.pgPool) {
        const client = await this.pgPool.connect();
        await client.query('SELECT 1');
        client.release();
        health.postgresql = true;
      }

      // Check Redis
      if (this.redisClient) {
        await this.redisClient.ping();
        health.redis = true;
      }

      // Check Neo4j
      const session = this.neo4jDriver.session();
      await session.run('RETURN 1');
      await session.close();
      health.neo4j = true;

      // Check Qdrant
      const qdrantResponse = await fetch(`${this.qdrantConfig.baseUrl}/health`);
      health.qdrant = qdrantResponse.ok;

      health.overall = Object.values(health).slice(0, 4).every(Boolean) ? 'healthy' : 'partial';
    } catch (error) {
      console.error('Health check error:', error);
    }

    return health;
  }

  // ============ Cleanup ============
  async close(): Promise<void> {
    if (this.pgPool) await this.pgPool.end();
    if (this.redisClient) await this.redisClient.quit();
    if (this.neo4jDriver) await this.neo4jDriver.close();
    console.log('All database connections closed');
  }
}

// Export singleton instance
export const db = new UnifiedDatabaseService();