/**
 * LLM Data Generation Script for Legal Dataset Creation
 * Processes legal documents to extract semantic phrases and prosecution correlations
 */

const fs = require('fs').promises;
const path = require('path');
const { Pool } = require('pg');
const Redis = require('ioredis');
const { z } = require('zod');

// Configuration
const CONFIG = {
    database: {
        user: process.env.DB_USER || 'postgres',
        password: process.env.DB_PASSWORD || 'password',
        host: process.env.DB_HOST || 'localhost',
        port: process.env.DB_PORT || 5432,
        database: process.env.DB_NAME || 'prosecutor_db'
    },
    redis: {
        url: process.env.REDIS_URL || 'redis://localhost:6379'
    },
    ollama: {
        url: process.env.OLLAMA_URL || 'http://localhost:11434',
        model: process.env.LLM_MODEL || 'gemma3-legal'
    },
    embedding: {
        model: 'nomic-embed-text',
        dimensions: 384
    },
    processing: {
        batchSize: 10,
        maxRetries: 3,
        pauseBetweenBatches: 1000
    }
};

// Zod schemas for validation
const LegalDocumentSchema = z.object({
    document_id: z.string(),
    jurisdiction: z.enum(['federal', 'state', 'local', 'international']),
    text_chunk: z.string().min(50),
    semantic_phrases: z.array(z.string()).min(1),
    entities: z.object({
        defendant: z.string().optional(),
        plaintiff: z.string().optional(),
        judge: z.string().optional(),
        court: z.string().optional()
    }),
    prosecution_strength_score: z.number().min(0).max(100),
    judgement_outcome: z.enum(['Guilty', 'Not Guilty', 'Dismissed', 'Plea Deal', 'Pending']),
    sentencing_factors: z.array(z.string()),
    case_type: z.enum(['criminal', 'civil', 'administrative', 'constitutional']),
    legal_precedents: z.array(z.string()).optional(),
    confidence_score: z.number().min(0).max(1)
});

const ProcessingStatsSchema = z.object({
    total_documents: z.number(),
    successful_extractions: z.number(),
    failed_extractions: z.number(),
    average_prosecution_score: z.number(),
    top_semantic_phrases: z.array(z.string()),
    processing_time_ms: z.number()
});

class LegalDatasetGenerator {
    constructor() {
        this.db = new Pool(CONFIG.database);
        this.redis = new Redis(CONFIG.redis.url);
        this.stats = {
            total_documents: 0,
            successful_extractions: 0,
            failed_extractions: 0,
            semantic_phrases: new Map(),
            prosecution_scores: [],
            start_time: Date.now()
        };
    }

    async initialize() {
        console.log('🚀 Initializing Legal Dataset Generator...');
        
        // Test database connection
        try {
            await this.db.query('SELECT NOW()');
            console.log('✅ Database connected');
        } catch (error) {
            throw new Error(`Database connection failed: ${error.message}`);
        }

        // Test Redis connection
        try {
            await this.redis.ping();
            console.log('✅ Redis connected');
        } catch (error) {
            throw new Error(`Redis connection failed: ${error.message}`);
        }

        // Create tables if not exist
        await this.createTables();
        
        console.log('✅ Legal Dataset Generator initialized');
    }

    async createTables() {
        const createTablesSQL = `
            -- Legal documents processed table
            CREATE TABLE IF NOT EXISTS legal_documents_processed (
                id SERIAL PRIMARY KEY,
                document_id VARCHAR(255) UNIQUE NOT NULL,
                jurisdiction VARCHAR(50) NOT NULL,
                text_chunk TEXT NOT NULL,
                semantic_phrases JSONB NOT NULL,
                entities JSONB NOT NULL,
                prosecution_strength_score INTEGER NOT NULL CHECK (prosecution_strength_score >= 0 AND prosecution_strength_score <= 100),
                judgement_outcome VARCHAR(50) NOT NULL,
                sentencing_factors JSONB NOT NULL,
                case_type VARCHAR(50) NOT NULL,
                legal_precedents JSONB,
                confidence_score DECIMAL(3,2) NOT NULL,
                embedding VECTOR(384),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            -- Semantic phrases ranking table
            CREATE TABLE IF NOT EXISTS semantic_phrases_ranking (
                id SERIAL PRIMARY KEY,
                phrase TEXT NOT NULL,
                frequency INTEGER DEFAULT 1,
                avg_prosecution_score DECIMAL(5,2),
                correlation_strength DECIMAL(3,2),
                jurisdiction_distribution JSONB,
                outcome_correlation JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(phrase)
            );

            -- Auto-complete suggestions table
            CREATE TABLE IF NOT EXISTS autocomplete_suggestions (
                id SERIAL PRIMARY KEY,
                prefix VARCHAR(100) NOT NULL,
                suggestion TEXT NOT NULL,
                score DECIMAL(3,2) NOT NULL,
                context_type VARCHAR(50) NOT NULL,
                usage_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX(prefix, score DESC)
            );

            -- Processing statistics
            CREATE TABLE IF NOT EXISTS processing_statistics (
                id SERIAL PRIMARY KEY,
                batch_id VARCHAR(100) NOT NULL,
                total_documents INTEGER NOT NULL,
                successful_extractions INTEGER NOT NULL,
                failed_extractions INTEGER NOT NULL,
                average_prosecution_score DECIMAL(5,2),
                top_semantic_phrases JSONB,
                processing_time_ms BIGINT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            -- Create indexes for performance
            CREATE INDEX IF NOT EXISTS idx_legal_documents_jurisdiction ON legal_documents_processed(jurisdiction);
            CREATE INDEX IF NOT EXISTS idx_legal_documents_prosecution_score ON legal_documents_processed(prosecution_strength_score);
            CREATE INDEX IF NOT EXISTS idx_legal_documents_outcome ON legal_documents_processed(judgement_outcome);
            CREATE INDEX IF NOT EXISTS idx_semantic_phrases_frequency ON semantic_phrases_ranking(frequency DESC);
            CREATE INDEX IF NOT EXISTS idx_semantic_phrases_correlation ON semantic_phrases_ranking(correlation_strength DESC);
        `;

        await this.db.query(createTablesSQL);
        console.log('✅ Database tables created/verified');
    }

    async processLegalDocuments(documentsPath) {
        console.log(`📄 Processing legal documents from: ${documentsPath}`);
        
        const files = await fs.readdir(documentsPath);
        const legalFiles = files.filter(file => 
            file.endsWith('.txt') || file.endsWith('.md') || file.endsWith('.pdf')
        );

        console.log(`Found ${legalFiles.length} legal documents to process`);

        // Process in batches
        for (let i = 0; i < legalFiles.length; i += CONFIG.processing.batchSize) {
            const batch = legalFiles.slice(i, i + CONFIG.processing.batchSize);
            console.log(`📊 Processing batch ${Math.floor(i / CONFIG.processing.batchSize) + 1}/${Math.ceil(legalFiles.length / CONFIG.processing.batchSize)}`);
            
            await this.processBatch(batch, documentsPath);
            
            // Pause between batches to avoid overwhelming the LLM
            if (i + CONFIG.processing.batchSize < legalFiles.length) {
                await new Promise(resolve => setTimeout(resolve, CONFIG.processing.pauseBetweenBatches));
            }
        }

        // Generate final statistics
        await this.generateStatistics();
        await this.buildAutocompleteSuggestions();
    }

    async processBatch(files, documentsPath) {
        const batchPromises = files.map(async (filename) => {
            try {
                const fullPath = path.join(documentsPath, filename);
                const content = await fs.readFile(fullPath, 'utf-8');
                
                if (content.length < 100) {
                    console.log(`⚠️ Skipping ${filename} - too short`);
                    return null;
                }

                return await this.processDocument(filename, content);
            } catch (error) {
                console.error(`❌ Error processing ${filename}:`, error.message);
                this.stats.failed_extractions++;
                return null;
            }
        });

        const results = await Promise.allSettled(batchPromises);
        const successful = results.filter(r => r.status === 'fulfilled' && r.value).map(r => r.value);
        
        if (successful.length > 0) {
            await this.saveToDatabase(successful);
        }
    }

    async processDocument(filename, content) {
        console.log(`🔍 Processing: ${filename}`);
        
        // Create document chunks (split long documents)
        const chunks = this.createChunks(content, 2000);
        const documentResults = [];

        for (let i = 0; i < chunks.length; i++) {
            const chunk = chunks[i];
            const documentId = `${filename.replace(/\.[^/.]+$/, "")}_chunk_${i}`;
            
            try {
                // Generate LLM analysis
                const analysis = await this.generateLLMAnalysis(chunk);
                
                // Generate embedding
                const embedding = await this.generateEmbedding(chunk);
                
                // Validate the analysis
                const validatedData = LegalDocumentSchema.parse({
                    document_id: documentId,
                    ...analysis,
                    text_chunk: chunk
                });

                // Add embedding
                validatedData.embedding = embedding;
                
                documentResults.push(validatedData);
                this.stats.successful_extractions++;
                
                // Update phrase frequency tracking
                for (const phrase of validatedData.semantic_phrases) {
                    const count = this.stats.semantic_phrases.get(phrase) || 0;
                    this.stats.semantic_phrases.set(phrase, count + 1);
                }
                
                this.stats.prosecution_scores.push(validatedData.prosecution_strength_score);
                
            } catch (error) {
                console.error(`❌ Failed to process chunk ${i} of ${filename}:`, error.message);
                this.stats.failed_extractions++;
            }
        }

        this.stats.total_documents++;
        return documentResults;
    }

    createChunks(text, maxLength = 2000) {
        const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0);
        const chunks = [];
        let currentChunk = '';

        for (const sentence of sentences) {
            if ((currentChunk + sentence).length > maxLength && currentChunk.length > 0) {
                chunks.push(currentChunk.trim());
                currentChunk = sentence;
            } else {
                currentChunk += sentence + '. ';
            }
        }

        if (currentChunk.trim().length > 0) {
            chunks.push(currentChunk.trim());
        }

        return chunks.length > 0 ? chunks : [text.substring(0, maxLength)];
    }

    async generateLLMAnalysis(textChunk) {
        const systemPrompt = `You are a legal analyst AI specializing in prosecution strategy and legal outcomes. Your task is to analyze legal document text and extract structured information that correlates language patterns with prosecution success and legal outcomes.

Analyze the provided legal text and respond with ONLY a valid JSON object (no other text) in this exact format:

{
  "jurisdiction": "federal|state|local|international",
  "semantic_phrases": ["phrase1", "phrase2", "phrase3"],
  "entities": {
    "defendant": "name or null",
    "plaintiff": "name or null", 
    "judge": "name or null",
    "court": "name or null"
  },
  "prosecution_strength_score": 0-100,
  "judgement_outcome": "Guilty|Not Guilty|Dismissed|Plea Deal|Pending",
  "sentencing_factors": ["factor1", "factor2"],
  "case_type": "criminal|civil|administrative|constitutional",
  "legal_precedents": ["precedent1", "precedent2"],
  "confidence_score": 0.0-1.0
}

Focus on:
- Identifying language that correlates with prosecution success
- Extracting key legal phrases that influence outcomes
- Assessing strength of legal arguments (prosecution_strength_score)
- Recognizing legal entities and precedents
- Determining case jurisdiction and type

Text to analyze:
${textChunk}`;

        let retries = 0;
        while (retries < CONFIG.processing.maxRetries) {
            try {
                const response = await fetch(`${CONFIG.ollama.url}/api/generate`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        model: CONFIG.ollama.model,
                        prompt: systemPrompt,
                        stream: false,
                        options: {
                            temperature: 0.3,
                            top_p: 0.9,
                            repeat_penalty: 1.1
                        }
                    })
                });

                if (!response.ok) {
                    throw new Error(`LLM request failed: ${response.status}`);
                }

                const data = await response.json();
                const analysisText = data.response.trim();
                
                // Try to extract JSON from the response
                const jsonMatch = analysisText.match(/\{[\s\S]*\}/);
                if (!jsonMatch) {
                    throw new Error('No JSON found in LLM response');
                }

                const analysis = JSON.parse(jsonMatch[0]);
                
                // Validate required fields
                if (!analysis.semantic_phrases || !Array.isArray(analysis.semantic_phrases)) {
                    throw new Error('Invalid semantic_phrases in LLM response');
                }

                return analysis;

            } catch (error) {
                retries++;
                console.error(`❌ LLM analysis attempt ${retries} failed:`, error.message);
                
                if (retries >= CONFIG.processing.maxRetries) {
                    // Return fallback analysis
                    return this.generateFallbackAnalysis(textChunk);
                }
                
                // Wait before retry
                await new Promise(resolve => setTimeout(resolve, 2000 * retries));
            }
        }
    }

    generateFallbackAnalysis(textChunk) {
        console.log('⚠️ Using fallback analysis');
        
        // Simple keyword-based analysis as fallback
        const legalKeywords = [
            'contract', 'liability', 'negligence', 'breach', 'damages',
            'guilty', 'innocent', 'evidence', 'testimony', 'verdict',
            'statute', 'regulation', 'precedent', 'jurisdiction', 'appeal'
        ];

        const foundPhrases = legalKeywords.filter(keyword => 
            textChunk.toLowerCase().includes(keyword)
        );

        return {
            jurisdiction: 'federal',
            semantic_phrases: foundPhrases.length > 0 ? foundPhrases.slice(0, 5) : ['legal document'],
            entities: {
                defendant: null,
                plaintiff: null,
                judge: null,
                court: null
            },
            prosecution_strength_score: Math.floor(Math.random() * 40) + 30, // 30-70 range
            judgement_outcome: 'Pending',
            sentencing_factors: [],
            case_type: 'civil',
            legal_precedents: [],
            confidence_score: 0.3
        };
    }

    async generateEmbedding(text) {
        try {
            const response = await fetch(`${CONFIG.ollama.url}/api/embeddings`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model: CONFIG.embedding.model,
                    prompt: text
                })
            });

            if (!response.ok) {
                throw new Error(`Embedding request failed: ${response.status}`);
            }

            const data = await response.json();
            return data.embedding;

        } catch (error) {
            console.error('❌ Embedding generation failed:', error.message);
            // Return random embedding as fallback
            return Array.from({ length: CONFIG.embedding.dimensions }, () => Math.random() - 0.5);
        }
    }

    async saveToDatabase(documents) {
        const client = await this.db.connect();
        
        try {
            await client.query('BEGIN');

            for (const docBatch of documents) {
                if (Array.isArray(docBatch)) {
                    for (const doc of docBatch) {
                        await this.insertDocument(client, doc);
                    }
                } else {
                    await this.insertDocument(client, docBatch);
                }
            }

            await client.query('COMMIT');
            console.log(`✅ Saved ${documents.length} documents to database`);

        } catch (error) {
            await client.query('ROLLBACK');
            console.error('❌ Database save failed:', error.message);
            throw error;
        } finally {
            client.release();
        }
    }

    async insertDocument(client, doc) {
        const insertSQL = `
            INSERT INTO legal_documents_processed (
                document_id, jurisdiction, text_chunk, semantic_phrases,
                entities, prosecution_strength_score, judgement_outcome,
                sentencing_factors, case_type, legal_precedents,
                confidence_score, embedding
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            ON CONFLICT (document_id) DO UPDATE SET
                jurisdiction = EXCLUDED.jurisdiction,
                text_chunk = EXCLUDED.text_chunk,
                semantic_phrases = EXCLUDED.semantic_phrases,
                entities = EXCLUDED.entities,
                prosecution_strength_score = EXCLUDED.prosecution_strength_score,
                judgement_outcome = EXCLUDED.judgement_outcome,
                sentencing_factors = EXCLUDED.sentencing_factors,
                case_type = EXCLUDED.case_type,
                legal_precedents = EXCLUDED.legal_precedents,
                confidence_score = EXCLUDED.confidence_score,
                embedding = EXCLUDED.embedding,
                updated_at = CURRENT_TIMESTAMP
        `;

        await client.query(insertSQL, [
            doc.document_id,
            doc.jurisdiction,
            doc.text_chunk,
            JSON.stringify(doc.semantic_phrases),
            JSON.stringify(doc.entities),
            doc.prosecution_strength_score,
            doc.judgement_outcome,
            JSON.stringify(doc.sentencing_factors),
            doc.case_type,
            JSON.stringify(doc.legal_precedents || []),
            doc.confidence_score,
            JSON.stringify(doc.embedding)
        ]);
    }

    async generateStatistics() {
        console.log('📊 Generating processing statistics...');

        const batchId = `batch_${Date.now()}`;
        const processingTime = Date.now() - this.stats.start_time;
        const avgProsecutionScore = this.stats.prosecution_scores.length > 0 
            ? this.stats.prosecution_scores.reduce((a, b) => a + b, 0) / this.stats.prosecution_scores.length
            : 0;

        // Get top semantic phrases
        const topPhrases = Array.from(this.stats.semantic_phrases.entries())
            .sort(([,a], [,b]) => b - a)
            .slice(0, 20)
            .map(([phrase]) => phrase);

        const stats = {
            batch_id: batchId,
            total_documents: this.stats.total_documents,
            successful_extractions: this.stats.successful_extractions,
            failed_extractions: this.stats.failed_extractions,
            average_prosecution_score: avgProsecutionScore,
            top_semantic_phrases: topPhrases,
            processing_time_ms: processingTime
        };

        // Validate with schema
        const validatedStats = ProcessingStatsSchema.parse(stats);

        // Save to database
        await this.db.query(`
            INSERT INTO processing_statistics (
                batch_id, total_documents, successful_extractions,
                failed_extractions, average_prosecution_score,
                top_semantic_phrases, processing_time_ms
            ) VALUES ($1, $2, $3, $4, $5, $6, $7)
        `, [
            validatedStats.batch_id,
            validatedStats.total_documents,
            validatedStats.successful_extractions,
            validatedStats.failed_extractions,
            validatedStats.average_prosecution_score,
            JSON.stringify(validatedStats.top_semantic_phrases),
            validatedStats.processing_time_ms
        ]);

        // Cache in Redis for quick access
        await this.redis.setex(`stats:${batchId}`, 3600, JSON.stringify(validatedStats));

        console.log('✅ Statistics generated and saved');
        this.printStatistics(validatedStats);
    }

    async buildAutocompleteSuggestions() {
        console.log('🔍 Building autocomplete suggestions...');

        // Update semantic phrases ranking
        for (const [phrase, frequency] of this.stats.semantic_phrases) {
            const avgScore = await this.db.query(`
                SELECT AVG(prosecution_strength_score) as avg_score
                FROM legal_documents_processed 
                WHERE semantic_phrases::text LIKE '%"${phrase}"%'
            `);

            const avgProsecutionScore = avgScore.rows[0]?.avg_score || 0;
            const correlationStrength = frequency > 5 ? Math.min(frequency / 100, 1) : 0.1;

            await this.db.query(`
                INSERT INTO semantic_phrases_ranking (
                    phrase, frequency, avg_prosecution_score, correlation_strength
                ) VALUES ($1, $2, $3, $4)
                ON CONFLICT (phrase) DO UPDATE SET
                    frequency = EXCLUDED.frequency,
                    avg_prosecution_score = EXCLUDED.avg_prosecution_score,
                    correlation_strength = EXCLUDED.correlation_strength,
                    updated_at = CURRENT_TIMESTAMP
            `, [phrase, frequency, avgProsecutionScore, correlationStrength]);
        }

        // Generate autocomplete suggestions
        const suggestions = await this.generateAutocompleteSuggestions();
        await this.cacheAutocompleteSuggestions(suggestions);

        console.log(`✅ Built ${suggestions.length} autocomplete suggestions`);
    }

    async generateAutocompleteSuggestions() {
        const highValuePhrases = await this.db.query(`
            SELECT phrase, avg_prosecution_score, correlation_strength
            FROM semantic_phrases_ranking
            WHERE frequency >= 3 AND correlation_strength > 0.2
            ORDER BY avg_prosecution_score DESC, frequency DESC
            LIMIT 1000
        `);

        const suggestions = [];

        for (const row of highValuePhrases.rows) {
            const phrase = row.phrase;
            const words = phrase.split(' ');
            
            // Create prefix suggestions (1-3 words)
            for (let i = 1; i <= Math.min(3, words.length); i++) {
                const prefix = words.slice(0, i).join(' ').toLowerCase();
                const score = (row.avg_prosecution_score / 100) * row.correlation_strength;
                
                suggestions.push({
                    prefix,
                    suggestion: phrase,
                    score,
                    context_type: 'legal_phrase'
                });
            }
        }

        return suggestions;
    }

    async cacheAutocompleteSuggestions(suggestions) {
        // Group by prefix
        const groupedSuggestions = suggestions.reduce((acc, sugg) => {
            if (!acc[sugg.prefix]) {
                acc[sugg.prefix] = [];
            }
            acc[sugg.prefix].push(sugg);
            return acc;
        }, {});

        // Cache top suggestions for each prefix
        for (const [prefix, prefixSuggestions] of Object.entries(groupedSuggestions)) {
            const topSuggestions = prefixSuggestions
                .sort((a, b) => b.score - a.score)
                .slice(0, 10);

            await this.redis.setex(
                `autocomplete:${prefix}`,
                7200, // 2 hours
                JSON.stringify(topSuggestions)
            );
        }
    }

    printStatistics(stats) {
        console.log('\n📊 === PROCESSING STATISTICS ===');
        console.log(`📄 Total Documents: ${stats.total_documents}`);
        console.log(`✅ Successful Extractions: ${stats.successful_extractions}`);
        console.log(`❌ Failed Extractions: ${stats.failed_extractions}`);
        console.log(`📈 Average Prosecution Score: ${stats.average_prosecution_score.toFixed(2)}`);
        console.log(`⏱️ Processing Time: ${(stats.processing_time_ms / 1000).toFixed(2)}s`);
        console.log(`🔝 Top Phrases: ${stats.top_semantic_phrases.slice(0, 5).join(', ')}`);
        console.log('===============================\n');
    }

    async cleanup() {
        await this.db.end();
        await this.redis.disconnect();
        console.log('✅ Cleanup completed');
    }
}

// CLI interface
async function main() {
    if (process.argv.length < 3) {
        console.log('Usage: node llm-data-generation.js <path-to-legal-documents>');
        console.log('Example: node llm-data-generation.js ./legal-documents/');
        process.exit(1);
    }

    const documentsPath = process.argv[2];
    
    if (!await fs.access(documentsPath).then(() => true).catch(() => false)) {
        console.error(`❌ Documents path not found: ${documentsPath}`);
        process.exit(1);
    }

    const generator = new LegalDatasetGenerator();
    
    try {
        await generator.initialize();
        await generator.processLegalDocuments(documentsPath);
        console.log('🎉 Legal dataset generation completed successfully!');
    } catch (error) {
        console.error('❌ Dataset generation failed:', error);
        process.exit(1);
    } finally {
        await generator.cleanup();
    }
}

// Run if called directly
if (require.main === module) {
    main().catch(console.error);
}

module.exports = { LegalDatasetGenerator, CONFIG };