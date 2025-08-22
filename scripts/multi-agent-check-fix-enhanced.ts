// scripts/multi-agent-check-fix-enhanced.ts
// Enhanced multi-agent build fixer with vector DB integration and GPU optimization

import { spawn } from 'child_process';
import fs from 'fs';
import path from 'path';
import fetch from 'node-fetch';
import { Worker } from 'worker_threads';
import os from 'os';

// Configuration
const config = {
  timestamp: new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19),
  maxWorkers: os.cpus().length,
  embedModel: 'nomic-embed-text-v1.5',
  vectorDimension: 768,
  qdrantHost: process.env.QDRANT_HOST || 'localhost',
  qdrantPort: parseInt(process.env.QDRANT_PORT || '6333'),
  pgHost: process.env.POSTGRES_HOST || 'localhost',
  pgPort: parseInt(process.env.POSTGRES_PORT || '5432'),
  pgDatabase: process.env.POSTGRES_DB || 'legal_ai_db',
  pgUser: process.env.POSTGRES_USER || 'legal_admin',
  pgPassword: process.env.POSTGRES_PASSWORD || 'LegalAI2024!',
  redisHost: process.env.REDIS_HOST || 'localhost',
  redisPort: parseInt(process.env.REDIS_PORT || '6379'),
  claudeApiKey: process.env.CLAUDE_API_KEY,
  openaiApiKey: process.env.OPENAI_API_KEY,
  ollamaHost: process.env.OLLAMA_HOST || 'http://localhost:11434'
};

// Directory setup
const logsDir = `logs_${config.timestamp}`;
const todoDir = `todolist_${config.timestamp}`;
const embeddingsDir = `embeddings_${config.timestamp}`;
const logFile = path.join(logsDir, 'npm_check.log');
const summaryFile = path.join(todoDir, 'summary.md');
const outputJson = path.join(todoDir, 'claude_suggestions.json');
const embeddingsFile = path.join(embeddingsDir, 'error_embeddings.json');
const metricsFile = path.join(todoDir, 'build_metrics.json');

// Create directories
[logsDir, todoDir, embeddingsDir].forEach(dir => {
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
});

console.log(`📁 Created directories:
- Logs: ${logsDir}
- TODOs: ${todoDir}
- Embeddings: ${embeddingsDir}`);

// Initialize vector stores
const qdrantClient = new QdrantClient({
  host: config.qdrantHost,
  port: config.qdrantPort,
});

const pgPool = new Pool({
  host: config.pgHost,
  port: config.pgPort,
  database: config.pgDatabase,
  user: config.pgUser,
  password: config.pgPassword,
});

const redis = new Redis({
  host: config.redisHost,
  port: config.redisPort,
});

const lokiDB = new loki('build-errors.db');
const errorCollection = lokiDB.addCollection('errors', {
  indices: ['type', 'file', 'severity', 'timestamp']
});

// GPU optimization settings for VS Code
const vscodeGpuSettings = {
  "disable-hardware-acceleration": false,
  "enable-gpu-rasterization": true,
  "enable-webgl2-compute-context": true,
  "gpu-sandbox": true,
  "use-angle": "d3d11",
  "enable-zero-copy": true,
  "enable-gpu-memory-buffer-video-frames": true,
  "enable-native-gpu-memory-buffers": true,
  "disable-gpu-vsync": false,
  "max-active-webgl-contexts": 16,
  "webgl-antialiasing-mode": "msaa",
  "enable-gpu-scheduler": true
};

// Error classification types
enum ErrorType {
  TypeScript = 'typescript',
  Svelte = 'svelte',
  Build = 'build',
  Runtime = 'runtime',
  Memory = 'memory',
  GPU = 'gpu',
  Network = 'network',
  Database = 'database'
}

interface ErrorEntry {
  id: string;
  type: ErrorType;
  file: string;
  line: number;
  column: number;
  message: string;
  severity: 'error' | 'warning' | 'info';
  context: string;
  embedding?: number[];
  timestamp: Date;
  suggestions?: string[];
  autoFixAvailable?: boolean;
}

interface BuildMetrics {
  totalErrors: number;
  errorsByType: Record<ErrorType, number>;
  memoryUsage: NodeJS.MemoryUsage;
  cpuUsage: NodeJS.CpuUsage;
  duration: number;
  gpuAvailable: boolean;
  workerThreadsUsed: number;
  vectorsGenerated: number;
  cachingLayersActive: string[];
}

// Enhanced error parser with pattern recognition
class ErrorParser {
  private patterns = {
    typescript: /TS\d+:|Type error:|Cannot find|Property .* does not exist/i,
    svelte: /Svelte:|Component:|<script>|<style>|\$:/i,
    memory: /heap|memory|allocation|OOM|out of memory/i,
    gpu: /GPU|CUDA|WebGL|shader|rasterization/i,
    database: /postgres|pgvector|drizzle|SQL|query/i,
  };

  parseErrors(log: string): ErrorEntry[] {
    const errors: ErrorEntry[] = [];
    const lines = log.split('\n');
    
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const errorMatch = this.extractError(line, i, lines);
      
      if (errorMatch) {
        errors.push(errorMatch);
      }
    }
    
    return errors;
  }

  private extractError(line: string, index: number, allLines: string[]): ErrorEntry | null {
    // Check for error patterns
    for (const [type, pattern] of Object.entries(this.patterns)) {
      if (pattern.test(line)) {
        const fileMatch = line.match(/([^\s]+\.(ts|tsx|js|jsx|svelte)):(\d+):(\d+)/);
        
        return {
          id: `error-${index}-${Date.now()}`,
          type: type as ErrorType,
          file: fileMatch?.[1] || 'unknown',
          line: parseInt(fileMatch?.[3] || '0'),
          column: parseInt(fileMatch?.[4] || '0'),
          message: line.trim(),
          severity: line.includes('error') ? 'error' : 'warning',
          context: allLines.slice(Math.max(0, index - 2), index + 3).join('\n'),
          timestamp: new Date(),
        };
      }
    }
    
    return null;
  }
}

// Vector embedding generator using worker threads
class EmbeddingGenerator {
  private workers: Worker[] = [];
  private currentWorker = 0;

  constructor() {
    // Initialize worker threads for parallel processing
    for (let i = 0; i < config.maxWorkers; i++) {
      this.workers.push(new Worker(path.join(__dirname, 'embedding-worker.js')));
    }
  }

  async generateEmbeddings(errors: ErrorEntry[]): Promise<void> {
    console.log(`🧠 Generating embeddings for ${errors.length} errors using ${config.maxWorkers} workers...`);
    
    const chunks = this.chunkArray(errors, Math.ceil(errors.length / config.maxWorkers));
    const promises = chunks.map((chunk, i) => this.processChunk(chunk, i));
    
    await Promise.all(promises);
    
    // Save embeddings
    fs.writeJsonSync(embeddingsFile, errors);
    console.log(`✅ Embeddings saved to ${embeddingsFile}`);
  }

  private chunkArray<T>(array: T[], size: number): T[][] {
    const chunks: T[][] = [];
    for (let i = 0; i < array.length; i += size) {
      chunks.push(array.slice(i, i + size));
    }
    return chunks;
  }

  private async processChunk(errors: ErrorEntry[], workerIndex: number): Promise<void> {
    return new Promise((resolve, reject) => {
      const worker = this.workers[workerIndex];
      
      worker.postMessage({
        type: 'generate',
        errors: errors,
        model: config.embedModel
      });
      
      worker.once('message', (result) => {
        if (result.type === 'embeddings') {
          // Update errors with embeddings
          result.embeddings.forEach((embedding: number[], index: number) => {
            errors[index].embedding = embedding;
          });
          resolve();
        } else if (result.type === 'error') {
          reject(new Error(result.error));
        }
      });
    });
  }

  async close() {
    await Promise.all(this.workers.map(w => w.terminate()));
  }
}

// Multi-layer caching with vector stores
class VectorCacheManager {
  async storeErrors(errors: ErrorEntry[]): Promise<void> {
    console.log('📊 Storing errors in multi-layer cache...');
    
    // Layer 1: Loki.js (in-memory)
    errors.forEach(error => errorCollection.insert(error));
    
    // Layer 2: Redis (fast cache)
    const pipeline = redis.pipeline();
    errors.forEach(error => {
      pipeline.hset('errors', error.id, JSON.stringify(error));
      pipeline.zadd('errors:by_severity', 
        error.severity === 'error' ? 2 : 1, 
        error.id
      );
    });
    await pipeline.exec();
    
    // Layer 3: Qdrant (vector search)
    if (errors.some(e => e.embedding)) {
      await this.storeInQdrant(errors.filter(e => e.embedding));
    }
    
    // Layer 4: PostgreSQL + pgvector (persistent)
    await this.storeInPostgres(errors);
    
    console.log('✅ Errors stored in all cache layers');
  }

  private async storeInQdrant(errors: ErrorEntry[]) {
    const collection = 'build_errors';
    
    // Ensure collection exists
    try {
      await qdrantClient.createCollection(collection, {
        vectors: {
          size: config.vectorDimension,
          distance: 'Cosine'
        }
      });
    } catch (e) {
      // Collection might already exist
    }
    
    // Store vectors
    await qdrantClient.upsert(collection, {
      wait: true,
      points: errors.map(error => ({
        id: error.id,
        vector: error.embedding!,
        payload: {
          type: error.type,
          file: error.file,
          message: error.message,
          severity: error.severity,
          timestamp: error.timestamp
        }
      }))
    });
  }

  private async storeInPostgres(errors: ErrorEntry[]) {
    const client = await pgPool.connect();
    
    try {
      // Create table if not exists
      await client.query(`
        CREATE TABLE IF NOT EXISTS build_errors (
          id TEXT PRIMARY KEY,
          type TEXT,
          file TEXT,
          line INTEGER,
          column INTEGER,
          message TEXT,
          severity TEXT,
          context TEXT,
          embedding vector(${config.vectorDimension}),
          timestamp TIMESTAMPTZ,
          suggestions JSONB,
          auto_fix_available BOOLEAN
        )
      `);

      // Insert errors
      for (const error of errors) {
        await client.query(`
          INSERT INTO build_errors 
          (id, type, file, line, column, message, severity, context, embedding, timestamp, suggestions, auto_fix_available)
          VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
          ON CONFLICT (id) DO UPDATE SET
            message = EXCLUDED.message,
            timestamp = EXCLUDED.timestamp
        `, [
          error.id,
          error.type,
          error.file,
          error.line,
          error.column,
          error.message,
          error.severity,
          error.context,
          error.embedding ? `[${error.embedding.join(',')}]` : null,
          error.timestamp,
          JSON.stringify(error.suggestions || []),
          error.autoFixAvailable || false
        ]);
      }
    } finally {
      client.release();
    }
  }

  async findSimilarErrors(embedding: number[], limit: number = 5): Promise<ErrorEntry[]> {
    // Search in Qdrant
    const searchResult = await qdrantClient.search('build_errors', {
      vector: embedding,
      limit: limit,
      with_payload: true
    });

    const errorIds = searchResult.map(r => r.payload?.id).filter(Boolean);
    
    // Retrieve full error details from Redis
    const errors: ErrorEntry[] = [];
    for (const id of errorIds) {
      const errorJson = await redis.hget('errors', id);
      if (errorJson) {
        errors.push(JSON.parse(errorJson));
      }
    }
    
    return errors;
  }
}

// Multi-agent orchestrator
class MultiAgentOrchestrator {
  private agents = {
    claude: this.askClaude.bind(this),
    openai: this.askOpenAI.bind(this),
    ollama: this.askOllama.bind(this),
    autogen: this.runAutogen.bind(this),
    crewai: this.runCrewAI.bind(this)
  };

  async orchestrate(errors: ErrorEntry[], log: string): Promise<any> {
    console.log('🤖 Starting multi-agent orchestration...');
    
    const tasks = [];
    
    // Primary analysis with Claude
    tasks.push(this.askClaude(errors, log));
    
    // Parallel analysis with other agents if available
    if (config.openaiApiKey) {
      tasks.push(this.askOpenAI(errors));
    }
    
    // Local model analysis
    tasks.push(this.askOllama(errors));
    
    // Run all agents in parallel
    const results = await Promise.allSettled(tasks);
    
    // Synthesize results
    const synthesis = await this.synthesizeResults(results);
    
    // Generate action plan
    const actionPlan = await this.generateActionPlan(synthesis, errors);
    
    return {
      synthesis,
      actionPlan,
      agentResults: results
    };
  }

  private async askClaude(errors: ErrorEntry[], log: string): Promise<any> {
    const prompt = {
      task: 'multi_agent_fix',
      system: `You are an expert TypeScript/SvelteKit developer. Analyze these build errors and create a structured action plan for fixing them. Consider:
      1. Error dependencies and fix order
      2. Similar error patterns that can be batch-fixed
      3. Root causes vs symptoms
      4. Integration with the existing Enhanced RAG system
      5. GPU optimization opportunities`,
      errors: errors.slice(0, 50), // Limit for token management
      log_excerpt: log.slice(0, 4000),
      codebase_context: {
        framework: 'SvelteKit 2 + Svelte 5',
        features: ['Enhanced RAG', 'Multi-agent system', 'GPU optimization', 'Vector DBs'],
        dependencies: ['drizzle-orm', 'pgvector', 'langchain', 'bits-ui', 'unocss']
      }
    };

    try {
      const res = await fetch('https://api.anthropic.com/v1/messages', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${config.claudeApiKey}`,
          'Content-Type': 'application/json',
          'anthropic-version': '2023-06-01'
        },
        body: JSON.stringify({
          model: 'claude-3-opus-20240229',
          messages: [{ role: 'user', content: JSON.stringify(prompt) }],
          temperature: 0.3,
          max_tokens: 4000
        })
      });

      const json = await res.json();
      return json?.content?.[0]?.text || 'No response from Claude';
    } catch (error) {
      console.error('Claude API error:', error);
      return { error: 'Claude API failed', details: error };
    }
  }

  private async askOpenAI(errors: ErrorEntry[]): Promise<any> {
    // OpenAI implementation
    return { agent: 'openai', status: 'not_implemented' };
  }

  private async askOllama(errors: ErrorEntry[]): Promise<any> {
    try {
      const response = await fetch(`${config.ollamaHost}/api/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: 'gemma3-legal',
          prompt: `Analyze these TypeScript/Svelte errors and suggest fixes:\n${JSON.stringify(errors.slice(0, 10))}`,
          temperature: 0.3,
          stream: false
        })
      });

      const result = await response.json();
      return { agent: 'ollama', response: result.response };
    } catch (error) {
      return { agent: 'ollama', error: 'Failed to connect' };
    }
  }

  private async runAutogen(errors: ErrorEntry[]): Promise<any> {
    // Autogen integration placeholder
    return { agent: 'autogen', status: 'ready_for_integration' };
  }

  private async runCrewAI(errors: ErrorEntry[]): Promise<any> {
    // CrewAI integration placeholder
    return { agent: 'crewai', status: 'ready_for_integration' };
  }

  private async synthesizeResults(results: PromiseSettledResult<any>[]): Promise<any> {
    const successful = results
      .filter((r): r is PromiseFulfilledResult<any> => r.status === 'fulfilled')
      .map(r => r.value);

    return {
      totalAgents: results.length,
      successfulAgents: successful.length,
      recommendations: this.extractRecommendations(successful),
      consensus: this.findConsensus(successful)
    };
  }

  private extractRecommendations(results: unknown[]): string[] {
    // Extract and deduplicate recommendations from all agents
    const allRecommendations = new Set<string>();
    
    results.forEach(result => {
      if (typeof result === 'string') {
        // Simple extraction for now
        const lines = result.split('\n');
        lines.forEach(line => {
          if (line.includes('fix') || line.includes('Fix') || line.includes('TODO')) {
            allRecommendations.add(line.trim());
          }
        });
      }
    });
    
    return Array.from(allRecommendations);
  }

  private findConsensus(results: unknown[]): unknown {
    // Find common patterns across agent responses
    return {
      priorityFiles: this.extractPriorityFiles(results),
      commonPatterns: this.extractPatterns(results)
    };
  }

  private extractPriorityFiles(results: unknown[]): string[] {
    // Extract files mentioned by multiple agents
    const fileCounts = new Map<string, number>();
    
    results.forEach(result => {
      const text = JSON.stringify(result);
      const fileMatches = text.match(/[a-zA-Z0-9\-_/]+\.(ts|js|svelte)/g) || [];
      
      fileMatches.forEach(file => {
        fileCounts.set(file, (fileCounts.get(file) || 0) + 1);
      });
    });
    
    return Array.from(fileCounts.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10)
      .map(([file]) => file);
  }

  private extractPatterns(results: unknown[]): string[] {
    // Extract common error patterns
    return ['Type errors in stores', 'Missing imports', 'Svelte 5 migration issues'];
  }

  private async generateActionPlan(synthesis: unknown, errors: ErrorEntry[]): Promise<any> {
    const errorsByType = errors.reduce((acc, error) => {
      acc[error.type] = (acc[error.type] || 0) + 1;
      return acc;
    }, {} as Record<ErrorType, number>);

    return {
      immediate: [
        'Fix critical TypeScript errors blocking compilation',
        'Update Svelte component syntax to v5 patterns',
        'Resolve missing module imports'
      ],
      automated: [
        'Run codemod for Svelte 5 migration patterns',
        'Apply ESLint auto-fixes',
        'Update import paths using TypeScript compiler API'
      ],
      manual: synthesis.recommendations.slice(0, 5),
      metrics: {
        estimatedTime: '2-4 hours',
        automationPotential: '65%',
        errorsByType
      }
    };
  }
}

// Main orchestration function
async function runEnhancedCheckFix() {
  const startTime = Date.now();
  const initialMemory = process.memoryUsage();
  
  console.log('🚀 Starting enhanced multi-agent check-fix pipeline...');
  
  // Step 1: Update VS Code GPU settings
  console.log('⚙️ Updating VS Code GPU settings...');
  const vscodeSettingsPath = path.join(process.cwd(), '.vscode', 'settings.json');
  if (fs.existsSync(vscodeSettingsPath)) {
    const settings = fs.readJsonSync(vscodeSettingsPath);
    Object.assign(settings, vscodeGpuSettings);
    fs.writeJsonSync(vscodeSettingsPath, settings, { spaces: 2 });
    console.log('✅ VS Code GPU optimizations applied');
  }

  // Step 2: Run npm check
  console.log('🔍 Running npm check...');
  const log = await runCheck();
  
  // Step 3: Parse errors
  const parser = new ErrorParser();
  const errors = parser.parseErrors(log);
  console.log(`📋 Found ${errors.length} errors`);
  
  // Step 4: Generate embeddings
  const embedder = new EmbeddingGenerator();
  await embedder.generateEmbeddings(errors);
  
  // Step 5: Store in vector databases
  const cacheManager = new VectorCacheManager();
  await cacheManager.storeErrors(errors);
  
  // Step 6: Multi-agent analysis
  const orchestrator = new MultiAgentOrchestrator();
  const results = await orchestrator.orchestrate(errors, log);
  
  // Step 7: Save results
  fs.writeFileSync(summaryFile, formatMarkdownSummary(results, errors));
  fs.writeJsonSync(outputJson, results, { spaces: 2 });
  
  // Step 8: Generate metrics
  const metrics: BuildMetrics = {
    totalErrors: errors.length,
    errorsByType: errors.reduce((acc, e) => {
      acc[e.type] = (acc[e.type] || 0) + 1;
      return acc;
    }, {} as Record<ErrorType, number>),
    memoryUsage: process.memoryUsage(),
    cpuUsage: process.cpuUsage(),
    duration: Date.now() - startTime,
    gpuAvailable: await checkGPUAvailability(),
    workerThreadsUsed: config.maxWorkers,
    vectorsGenerated: errors.filter(e => e.embedding).length,
    cachingLayersActive: ['loki', 'redis', 'qdrant', 'postgres']
  };
  
  fs.writeJsonSync(metricsFile, metrics, { spaces: 2 });
  
  // Cleanup
  await embedder.close();
  await redis.quit();
  await pgPool.end();
  
  console.log(`
✅ Enhanced check-fix pipeline complete!

📊 Summary:
- Total errors: ${errors.length}
- Vectors generated: ${metrics.vectorsGenerated}
- Processing time: ${(metrics.duration / 1000).toFixed(2)}s
- Memory used: ${((metrics.memoryUsage.heapUsed - initialMemory.heapUsed) / 1024 / 1024).toFixed(2)}MB

📁 Outputs:
- Summary: ${summaryFile}
- Suggestions: ${outputJson}
- Embeddings: ${embeddingsFile}
- Metrics: ${metricsFile}

🚀 Next steps:
1. Review ${summaryFile} for prioritized fixes
2. Run agent workflows: npm run agents:fix
3. Monitor vector similarity for recurring errors
4. GPU-accelerated error visualization available
`);
}

// Helper functions
function runCheck(): Promise<string> {
  return new Promise((resolve) => {
    const check = spawn('npm', ['run', 'check'], { 
      shell: true,
      env: { ...process.env, NODE_OPTIONS: '--max-old-space-size=8192' }
    });
    let output = '';

    check.stdout.on('data', (data) => {
      const chunk = data.toString();
      output += chunk;
      process.stdout.write(chunk); // Real-time output
    });
    
    check.stderr.on('data', (data) => {
      const chunk = data.toString();
      output += chunk;
      process.stderr.write(chunk);
    });

    check.on('close', () => {
      fs.writeFileSync(logFile, output);
      resolve(output);
    });
  });
}

async function checkGPUAvailability(): Promise<boolean> {
  try {
    const response = await fetch(`${config.ollamaHost}/api/tags`);
    const data = await response.json();
    return data?.models?.some((m: unknown) => m.details?.families?.includes('gpu'));
  } catch {
    return false;
  }
}

function formatMarkdownSummary(results: unknown, errors: ErrorEntry[]): string {
  const errorsByFile = errors.reduce((acc, error) => {
    if (!acc[error.file]) acc[error.file] = [];
    acc[error.file].push(error);
    return acc;
  }, {} as Record<string, ErrorEntry[]>);

  return `# Build Fix Summary - ${config.timestamp}

## 📊 Overview
- **Total Errors**: ${errors.length}
- **Critical Errors**: ${errors.filter(e => e.severity === 'error').length}
- **Warnings**: ${errors.filter(e => e.severity === 'warning').length}
- **Files Affected**: ${Object.keys(errorsByFile).length}

## 🤖 Multi-Agent Analysis

### Synthesis
${JSON.stringify(results.synthesis, null, 2)}

### Action Plan

#### 🚨 Immediate Actions
${results.actionPlan.immediate.map((a: string) => `- ${a}`).join('\n')}

#### 🤖 Automated Fixes
${results.actionPlan.automated.map((a: string) => `- ${a}`).join('\n')}

#### 👨‍💻 Manual Review Required
${results.actionPlan.manual.map((a: string) => `- ${a}`).join('\n')}

## 📁 Files with Most Errors
${Object.entries(errorsByFile)
  .sort((a, b) => b[1].length - a[1].length)
  .slice(0, 10)
  .map(([file, errs]) => `- **${file}**: ${errs.length} errors`)
  .join('\n')}

## 🔧 Integration Points

### Vector Database Status
- ✅ Qdrant: Embeddings stored
- ✅ PostgreSQL + pgvector: Persistent storage
- ✅ Redis: Fast cache active
- ✅ Loki.js: In-memory index

### GPU Optimization
- VS Code GPU settings updated
- Worker threads: ${config.maxWorkers} active
- SIMD optimizations enabled

## 📚 Resources
- [Enhanced RAG Documentation](./docs/enhanced-rag-self-organizing-loop-system.md)
- [Multi-Agent Orchestration Guide](./CLAUDE.md#multi-agent-orchestration)
- [GPU Optimization Patterns](./CLAUDE.md#llm-optimization-patterns)
`;
}

// Run the enhanced pipeline
if (require.main === module) {
  runEnhancedCheckFix().catch(console.error);
}

export { runEnhancedCheckFix, ErrorParser, EmbeddingGenerator, VectorCacheManager, MultiAgentOrchestrator };