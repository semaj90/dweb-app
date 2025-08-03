import * as vscode from 'vscode';
import neo4j, { Driver, Session, Record } from 'neo4j-driver';

export interface EmbeddingNode {
    id: string;
    text: string;
    embedding: number[];
    metadata: {
        source: string;
        chunk_index: number;
        file_path: string;
        timestamp: string;
        model: string;
        [key: string]: any;
    };
}

export interface SearchResult {
    node: EmbeddingNode;
    score: number;
    similarity: number;
}

export interface Neo4jConfig {
    uri: string;
    user: string;
    password: string;
    database?: string;
}

export class Neo4jService {
    private driver: Driver | null = null;
    private config: Neo4jConfig;

    constructor() {
        this.config = this.loadConfiguration();
    }

    private loadConfiguration(): Neo4jConfig {
        const config = vscode.workspace.getConfiguration('mcpContext7');
        return {
            uri: config.get('neo4jUri', 'bolt://localhost:7687'),
            user: config.get('neo4jUser', 'neo4j'),
            password: config.get('neo4jPassword', 'password'),
            database: config.get('neo4jDatabase', 'neo4j')
        };
    }

    async connect(): Promise<void> {
        try {
            this.driver = neo4j.driver(
                this.config.uri,
                neo4j.auth.basic(this.config.user, this.config.password),
                {
                    connectionTimeout: 30000,
                    maxConnectionPoolSize: 10
                }
            );

            // Test connection
            await this.driver.verifyConnectivity();
            
            // Initialize schema
            await this.initializeSchema();
            
            console.log('Connected to Neo4j successfully');
        } catch (error) {
            console.error('Failed to connect to Neo4j:', error);
            throw new Error(`Neo4j connection failed: ${error instanceof Error ? error.message : String(error)}`);
        }
    }

    private async initializeSchema(): Promise<void> {
        const session = this.getSession();
        
        try {
            // Create constraints and indexes
            await session.run(`
                CREATE CONSTRAINT embedding_id_unique IF NOT EXISTS
                FOR (e:Embedding) REQUIRE e.id IS UNIQUE
            `);

            await session.run(`
                CREATE INDEX embedding_source_index IF NOT EXISTS
                FOR (e:Embedding) ON (e.source)
            `);

            await session.run(`
                CREATE INDEX embedding_timestamp_index IF NOT EXISTS  
                FOR (e:Embedding) ON (e.timestamp)
            `);

            // Create vector index for similarity search (if supported)
            try {
                await session.run(`
                    CREATE VECTOR INDEX embedding_vector_index IF NOT EXISTS
                    FOR (e:Embedding) ON (e.embedding)
                    OPTIONS {indexConfig: {
                        \`vector.dimensions\`: 768,
                        \`vector.similarity_function\`: 'cosine'
                    }}
                `);
            } catch (vectorError) {
                console.warn('Vector index creation failed (may not be supported):', vectorError);
                // Vector search will fall back to manual cosine similarity
            }

            console.log('Neo4j schema initialized successfully');
        } finally {
            session.close();
        }
    }

    private getSession(): Session {
        if (!this.driver) {
            throw new Error('Neo4j driver not connected. Call connect() first.');
        }
        return this.driver.session({ database: this.config.database });
    }

    /**
     * Store embedding nodes in Neo4j
     */
    async storeEmbeddings(nodes: EmbeddingNode[]): Promise<number> {
        const session = this.getSession();
        let storedCount = 0;

        try {
            await session.writeTransaction(async (tx) => {
                for (const node of nodes) {
                    const result = await tx.run(`
                        MERGE (e:Embedding {id: $id})
                        SET e.text = $text,
                            e.embedding = $embedding,
                            e.source = $source,
                            e.chunk_index = $chunk_index,
                            e.file_path = $file_path,
                            e.timestamp = $timestamp,
                            e.model = $model,
                            e.metadata = $metadata
                        RETURN e
                    `, {
                        id: node.id,
                        text: node.text,
                        embedding: node.embedding,
                        source: node.metadata.source,
                        chunk_index: node.metadata.chunk_index,
                        file_path: node.metadata.file_path,
                        timestamp: node.metadata.timestamp,
                        model: node.metadata.model,
                        metadata: node.metadata
                    });

                    if (result.records.length > 0) {
                        storedCount++;
                    }
                }
            });

            console.log(`Stored ${storedCount} embedding nodes in Neo4j`);
            return storedCount;
        } finally {
            session.close();
        }
    }

    /**
     * Search for similar embeddings using vector similarity
     */
    async searchSimilar(
        queryEmbedding: number[],
        limit: number = 10,
        minSimilarity: number = 0.7
    ): Promise<SearchResult[]> {
        const session = this.getSession();

        try {
            // Try vector index search first (if available)
            try {
                const vectorResult = await session.run(`
                    CALL db.index.vector.queryNodes('embedding_vector_index', $k, $queryVector)
                    YIELD node, score
                    WHERE score >= $minSimilarity
                    RETURN node, score
                    ORDER BY score DESC
                    LIMIT $limit
                `, {
                    k: limit,
                    queryVector: queryEmbedding,
                    minSimilarity,
                    limit
                });

                return vectorResult.records.map(record => this.recordToSearchResult(record));
            } catch (vectorError) {
                console.warn('Vector index search failed, falling back to manual similarity:', vectorError);
                // Fall back to manual cosine similarity calculation
            }

            // Manual cosine similarity search
            const result = await session.run(`
                MATCH (e:Embedding)
                WITH e, 
                     reduce(dot = 0.0, i IN range(0, size($queryVector)-1) | 
                         dot + ($queryVector[i] * e.embedding[i])
                     ) AS dotProduct,
                     sqrt(reduce(sumA = 0.0, i IN range(0, size($queryVector)-1) | 
                         sumA + ($queryVector[i] * $queryVector[i])
                     )) AS queryMagnitude,
                     sqrt(reduce(sumB = 0.0, i IN range(0, size(e.embedding)-1) | 
                         sumB + (e.embedding[i] * e.embedding[i])
                     )) AS embeddingMagnitude
                WITH e, (dotProduct / (queryMagnitude * embeddingMagnitude)) AS similarity
                WHERE similarity >= $minSimilarity
                RETURN e, similarity AS score
                ORDER BY similarity DESC
                LIMIT $limit
            `, {
                queryVector: queryEmbedding,
                minSimilarity,
                limit
            });

            return result.records.map(record => this.recordToSearchResult(record));
        } finally {
            session.close();
        }
    }

    private recordToSearchResult(record: Record): SearchResult {
        const node = record.get('e') || record.get('node');
        const score = record.get('score');

        return {
            node: {
                id: node.properties.id,
                text: node.properties.text,
                embedding: node.properties.embedding,
                metadata: {
                    source: node.properties.source,
                    chunk_index: node.properties.chunk_index,
                    file_path: node.properties.file_path,
                    timestamp: node.properties.timestamp,
                    model: node.properties.model,
                    ...node.properties.metadata
                }
            },
            score: typeof score === 'number' ? score : score.toNumber(),
            similarity: typeof score === 'number' ? score : score.toNumber()
        };
    }

    /**
     * Search evidence by text query (semantic search)
     */
    async searchEvidence(
        query: string,
        queryEmbedding: number[],
        filters?: {
            source?: string;
            file_path?: string;
            date_range?: { start: string; end: string };
        },
        limit: number = 20
    ): Promise<SearchResult[]> {
        const session = this.getSession();

        try {
            let cypher = `
                MATCH (e:Embedding)
                WHERE 1=1
            `;

            const params: any = {
                queryVector: queryEmbedding,
                limit
            };

            // Apply filters
            if (filters?.source) {
                cypher += ` AND e.source = $source`;
                params.source = filters.source;
            }

            if (filters?.file_path) {
                cypher += ` AND e.file_path CONTAINS $file_path`;
                params.file_path = filters.file_path;
            }

            if (filters?.date_range) {
                cypher += ` AND e.timestamp >= $start_date AND e.timestamp <= $end_date`;
                params.start_date = filters.date_range.start;
                params.end_date = filters.date_range.end;
            }

            // Add similarity calculation and ordering
            cypher += `
                WITH e, 
                     reduce(dot = 0.0, i IN range(0, size($queryVector)-1) | 
                         dot + ($queryVector[i] * e.embedding[i])
                     ) AS dotProduct,
                     sqrt(reduce(sumA = 0.0, i IN range(0, size($queryVector)-1) | 
                         sumA + ($queryVector[i] * $queryVector[i])
                     )) AS queryMagnitude,
                     sqrt(reduce(sumB = 0.0, i IN range(0, size(e.embedding)-1) | 
                         sumB + (e.embedding[i] * e.embedding[i])
                     )) AS embeddingMagnitude
                WITH e, (dotProduct / (queryMagnitude * embeddingMagnitude)) AS similarity
                WHERE similarity >= 0.3
                RETURN e, similarity AS score
                ORDER BY similarity DESC
                LIMIT $limit
            `;

            const result = await session.run(cypher, params);
            return result.records.map(record => this.recordToSearchResult(record));
        } finally {
            session.close();
        }
    }

    /**
     * Get embedding statistics
     */
    async getStats(): Promise<{
        totalEmbeddings: number;
        sourceBreakdown: { source: string; count: number }[];
        modelBreakdown: { model: string; count: number }[];
        dateRange: { earliest: string; latest: string };
    }> {
        const session = this.getSession();

        try {
            // Total count
            const totalResult = await session.run(`
                MATCH (e:Embedding)
                RETURN count(e) AS total
            `);
            const totalEmbeddings = totalResult.records[0]?.get('total')?.toNumber() || 0;

            // Source breakdown
            const sourceResult = await session.run(`
                MATCH (e:Embedding)
                RETURN e.source AS source, count(e) AS count
                ORDER BY count DESC
            `);
            const sourceBreakdown = sourceResult.records.map(record => ({
                source: record.get('source'),
                count: record.get('count').toNumber()
            }));

            // Model breakdown
            const modelResult = await session.run(`
                MATCH (e:Embedding)
                RETURN e.model AS model, count(e) AS count
                ORDER BY count DESC
            `);
            const modelBreakdown = modelResult.records.map(record => ({
                model: record.get('model'),
                count: record.get('count').toNumber()
            }));

            // Date range
            const dateResult = await session.run(`
                MATCH (e:Embedding)
                RETURN min(e.timestamp) AS earliest, max(e.timestamp) AS latest
            `);
            const dateRecord = dateResult.records[0];
            const dateRange = {
                earliest: dateRecord?.get('earliest') || '',
                latest: dateRecord?.get('latest') || ''
            };

            return {
                totalEmbeddings,
                sourceBreakdown,
                modelBreakdown,
                dateRange
            };
        } finally {
            session.close();
        }
    }

    /**
     * Delete embeddings by source or file path
     */
    async deleteEmbeddings(criteria: {
        source?: string;
        file_path?: string;
        before_date?: string;
    }): Promise<number> {
        const session = this.getSession();

        try {
            let cypher = `MATCH (e:Embedding) WHERE 1=1`;
            const params: any = {};

            if (criteria.source) {
                cypher += ` AND e.source = $source`;
                params.source = criteria.source;
            }

            if (criteria.file_path) {
                cypher += ` AND e.file_path = $file_path`;
                params.file_path = criteria.file_path;
            }

            if (criteria.before_date) {
                cypher += ` AND e.timestamp < $before_date`;
                params.before_date = criteria.before_date;
            }

            cypher += ` DELETE e RETURN count(e) AS deleted`;

            const result = await session.run(cypher, params);
            const deletedCount = result.records[0]?.get('deleted')?.toNumber() || 0;

            console.log(`Deleted ${deletedCount} embedding nodes from Neo4j`);
            return deletedCount;
        } finally {
            session.close();
        }
    }

    async disconnect(): Promise<void> {
        if (this.driver) {
            await this.driver.close();
            this.driver = null;
            console.log('Disconnected from Neo4j');
        }
    }

    dispose() {
        this.disconnect().catch(console.error);
    }
}