#!/usr/bin/env node

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";

// Load env for service URLs/models
import "dotenv/config";

// Prefer GPU-accelerated Go services when provided
const GO_SIMD_PARSER_URL =
  process.env.GO_SIMD_PARSER_URL ||
  process.env.SIMD_PARSER_URL ||
  process.env.GO_SERVICE_SIMD_URL ||
  "";

// Route embeddings to gollama (Go-based Ollama) if available
const GOLLAMA_URL =
  process.env.GOLLAMA_URL ||
  process.env.GO_LLAMA_URL ||
  process.env.GO_LLAMA_EMBEDDINGS_URL ||
  "";

// Use legal-bert by default for semantic analysis (chat/generation)
if (!process.env.OLLAMA_MODEL) {
  process.env.OLLAMA_MODEL = process.env.LEGAL_BERT_MODEL || "gemma3-legal";
}

// Embedding model decoupled from chat model (default to nomic-embed-text)
if (!process.env.OLLAMA_EMBED_MODEL) {
  process.env.OLLAMA_EMBED_MODEL = "nomic-embed-text";
}

// Point embeddings base URL at gollama if provided
if (!process.env.OLLAMA_URL && GOLLAMA_URL) {
  process.env.OLLAMA_URL = GOLLAMA_URL;
}

// Expose SIMD parser URL globally for downstream usage
globalThis.GO_SIMD_PARSER_URL = GO_SIMD_PARSER_URL;

// Optional helper that calls the Go SIMD parser (GPU-accelerated) if needed elsewhere
globalThis.parseWithGoSimd = async function parseWithGoSimd(input) {
  if (!globalThis.GO_SIMD_PARSER_URL) {
    throw new Error("GO_SIMD_PARSER_URL not configured");
  }
  const res = await fetch(`${globalThis.GO_SIMD_PARSER_URL}/api/parse`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: input }),
  });
  if (!res.ok) {
    const txt = await res.text();
    throw new Error(`SIMD parser error: ${res.status} ${txt}`);
  }
  return res.json();
};
import express from "express";
import bodyParser from "body-parser";
import { PGVectorStore } from "@langchain/community/vectorstores/pgvector";
import { OpenAIEmbeddings } from "@langchain/openai";
import fs from "fs/promises";
import path from "path";
// chokidar is optional; we'll load it dynamically only if WATCH_FILES=true
let _chokidar = null;
import crypto from "crypto";

// Enhanced Meta-Data Embedder and Auto-Encoder Neural Network
class MetaDataEmbedder {
  constructor(nomicEmbeddings) {
    this.embeddings = nomicEmbeddings;
    this.fileMetaCache = new Map();
    this.autoEncoder = new AutoEncoderNN();
  }

  async extractFileMetadata(filePath, content) {
    const stats = await fs.stat(filePath).catch(() => null);
    const extension = path.extname(filePath);
    const size = content.length;

    return {
      filePath,
      extension,
      size,
      lastModified: stats?.mtime || new Date(),
      hash: crypto.createHash('md5').update(content).digest('hex'),
      lineCount: content.split('\n').length,
      language: this.detectLanguage(extension),
      complexity: this.calculateComplexity(content),
      imports: this.extractImports(content, extension),
      exports: this.extractExports(content, extension),
      functions: this.extractFunctions(content, extension),
      classes: this.extractClasses(content, extension)
    };
  }

  detectLanguage(extension) {
    const langMap = {
      '.js': 'javascript', '.ts': 'typescript', '.svelte': 'svelte',
      '.py': 'python', '.go': 'go', '.rs': 'rust', '.java': 'java',
      '.cs': 'csharp', '.cpp': 'cpp', '.c': 'c', '.h': 'c_header',
      '.sql': 'sql', '.json': 'json', '.md': 'markdown', '.html': 'html',
      '.css': 'css', '.scss': 'scss', '.yaml': 'yaml', '.yml': 'yaml'
    };
    return langMap[extension] || 'text';
  }

  calculateComplexity(content) {
    const lines = content.split('\n');
    let complexity = 0;

    // Cyclomatic complexity estimation
    const complexityKeywords = [
      'if', 'else', 'while', 'for', 'switch', 'case', 'try', 'catch',
      '?', '&&', '||', 'async', 'await', 'function', 'class'
    ];

    lines.forEach(line => {
      complexityKeywords.forEach(keyword => {
        const matches = (line.match(new RegExp(keyword, 'g')) || []).length;
        complexity += matches;
      });
    });

    return Math.min(complexity, 100); // Cap at 100
  }

  extractImports(content, extension) {
    const imports = [];
    if (['.js', '.ts', '.svelte'].includes(extension)) {
      const importRegex = /import\s+.*?\s+from\s+['"`]([^'"`]+)['"`]/g;
      let match;
      while ((match = importRegex.exec(content)) !== null) {
        imports.push(match[1]);
      }
    }
    return imports;
  }

  extractExports(content, extension) {
    const exports = [];
    if (['.js', '.ts', '.svelte'].includes(extension)) {
      const exportRegex = /export\s+(function|class|const|let|var)\s+(\w+)/g;
      let match;
      while ((match = exportRegex.exec(content)) !== null) {
        exports.push({ type: match[1], name: match[2] });
      }
    }
    return exports;
  }

  extractFunctions(content, extension) {
    const functions = [];
    if (['.js', '.ts', '.svelte'].includes(extension)) {
      const functionRegex = /(function\s+(\w+)|(\w+)\s*=\s*(?:async\s+)?(?:function|\([^)]*\)\s*=>))/g;
      let match;
      while ((match = functionRegex.exec(content)) !== null) {
        functions.push(match[2] || match[3]);
      }
    }
    return functions;
  }

  extractClasses(content, extension) {
    const classes = [];
    if (['.js', '.ts', '.svelte'].includes(extension)) {
      const classRegex = /class\s+(\w+)/g;
      let match;
      while ((match = classRegex.exec(content)) !== null) {
        classes.push(match[1]);
      }
    }
    return classes;
  }

  async embedFileWithMetadata(filePath, content) {
    const metadata = await this.extractFileMetadata(filePath, content);

    // Create rich context for embedding
    const contextualContent = this.createContextualContent(metadata, content);

    // Generate embeddings for both content and metadata
    const contentEmbedding = await this.embeddings.embedQuery(content);
    const metadataEmbedding = await this.embeddings.embedQuery(JSON.stringify(metadata));

    // Use auto-encoder to compress and enhance embeddings
    const compressedEmbedding = await this.autoEncoder.encode(contentEmbedding);

    this.fileMetaCache.set(metadata.hash, {
      metadata,
      contentEmbedding,
      metadataEmbedding,
      compressedEmbedding,
      contextualContent
    });

    return {
      metadata,
      contentEmbedding,
      metadataEmbedding,
      compressedEmbedding,
      contextualContent
    };
  }

  createContextualContent(metadata, content) {
    const summary = content.length > 1000 ? content.substring(0, 1000) + '...' : content;

    return `
File: ${metadata.filePath}
Language: ${metadata.language}
Size: ${metadata.size} bytes
Complexity: ${metadata.complexity}
Imports: ${metadata.imports.join(', ')}
Exports: ${metadata.exports.map(e => `${e.type} ${e.name}`).join(', ')}
Functions: ${metadata.functions.join(', ')}
Classes: ${metadata.classes.join(', ')}

Content Summary:
${summary}
    `.trim();
  }
}

// Auto-Encoder Neural Network for Context Compression
class AutoEncoderNN {
  constructor() {
    this.inputSize = 768; // nomic-embed-text dimensions
    this.hiddenSize = 256; // Compressed representation
    this.weights = this.initializeWeights();
  }

  initializeWeights() {
    // Simple weight initialization (in production, use proper ML library)
    return {
      encoder: Array(this.hiddenSize).fill().map(() =>
        Array(this.inputSize).fill().map(() => (Math.random() - 0.5) * 0.1)
      ),
      decoder: Array(this.inputSize).fill().map(() =>
        Array(this.hiddenSize).fill().map(() => (Math.random() - 0.5) * 0.1)
      )
    };
  }

  async encode(embedding) {
    // Simple matrix multiplication for encoding
    const encoded = new Array(this.hiddenSize).fill(0);

    for (let i = 0; i < this.hiddenSize; i++) {
      for (let j = 0; j < this.inputSize && j < embedding.length; j++) {
        encoded[i] += embedding[j] * this.weights.encoder[i][j];
      }
      encoded[i] = this.relu(encoded[i]); // ReLU activation
    }

    return encoded;
  }

  async decode(encoded) {
    // Simple matrix multiplication for decoding
    const decoded = new Array(this.inputSize).fill(0);

    for (let i = 0; i < this.inputSize; i++) {
      for (let j = 0; j < this.hiddenSize && j < encoded.length; j++) {
        decoded[i] += encoded[j] * this.weights.decoder[i][j];
      }
      decoded[i] = this.relu(decoded[i]); // ReLU activation
    }

    return decoded;
  }

  relu(x) {
    return Math.max(0, x);
  }

  // Training method (placeholder for real implementation)
  async train(trainingData) {
    console.log(`Training auto-encoder with ${trainingData.length} samples`);
    // In production, implement proper backpropagation
  }
}

// File Monitoring and Auto-Update System
class FileMonitor {
  constructor(metaDataEmbedder, vectorStore) {
    this.embedder = metaDataEmbedder;
    this.vectorStore = vectorStore;
    this.watchers = new Map();
    this.updateQueue = [];
    this.isProcessing = false;
  }

  watchDirectory(dirPath, patterns = ['**/*.js', '**/*.ts', '**/*.svelte', '**/*.md']) {
    const watcher = chokidar.watch(patterns, {
      cwd: dirPath,
      ignored: /node_modules|\.git|\.svelte-kit/,
      persistent: true
    });

    watcher
      .on('add', (filePath) => this.queueUpdate('add', path.join(dirPath, filePath)))
      .on('change', (filePath) => this.queueUpdate('change', path.join(dirPath, filePath)))
      .on('unlink', (filePath) => this.queueUpdate('delete', path.join(dirPath, filePath)));

    this.watchers.set(dirPath, watcher);
    return watcher;
  }

  queueUpdate(action, filePath) {
    this.updateQueue.push({ action, filePath, timestamp: Date.now() });
    this.processQueue();
  }

  async processQueue() {
    if (this.isProcessing || this.updateQueue.length === 0) return;

    this.isProcessing = true;

    try {
      while (this.updateQueue.length > 0) {
        const update = this.updateQueue.shift();
        await this.processUpdate(update);
      }
    } finally {
      this.isProcessing = false;
    }
  }

  async processUpdate(update) {
    const { action, filePath } = update;

    try {
      if (action === 'delete') {
        await this.removeFromIndex(filePath);
      } else {
        const content = await fs.readFile(filePath, 'utf-8');
        const embeddingData = await this.embedder.embedFileWithMetadata(filePath, content);
        await this.updateIndex(filePath, embeddingData);
      }

      console.log(`Processed ${action} for ${filePath}`);
    } catch (error) {
      console.error(`Error processing ${action} for ${filePath}:`, error.message);
    }
  }

  async updateIndex(filePath, embeddingData) {
    if (!this.vectorStore) return;

    try {
      // Remove existing entry
      await this.removeFromIndex(filePath);

      // Add new entry
      await this.vectorStore.addDocuments([{
        pageContent: embeddingData.contextualContent,
        metadata: {
          ...embeddingData.metadata,
          embeddings: {
            content: embeddingData.contentEmbedding,
            metadata: embeddingData.metadataEmbedding,
            compressed: embeddingData.compressedEmbedding
          }
        }
      }]);
    } catch (error) {
      console.error(`Error updating index for ${filePath}:`, error);
    }
  }

  async removeFromIndex(filePath) {
    // Note: PGVectorStore doesn't have built-in delete by metadata
    // This would need custom implementation with direct SQL
    console.log(`Removing ${filePath} from index (custom implementation needed)`);
  }

  stopWatching(dirPath) {
    const watcher = this.watchers.get(dirPath);
    if (watcher) {
      watcher.close();
      this.watchers.delete(dirPath);
    }
  }

  stopAllWatching() {
    for (const [dirPath, watcher] of this.watchers) {
      watcher.close();
    }
    this.watchers.clear();
  }
}

// Enhanced Summarization Service
class SummarizationService {
  constructor(nomicEmbeddings) {
    this.embeddings = nomicEmbeddings;
    this.summaryCache = new Map();
  }

  async generateSummary(content, context = {}) {
    const cacheKey = crypto.createHash('md5').update(content + JSON.stringify(context)).digest('hex');

    if (this.summaryCache.has(cacheKey)) {
      return this.summaryCache.get(cacheKey);
    }

    const summary = await this.createContextualSummary(content, context);
    this.summaryCache.set(cacheKey, summary);

    return summary;
  }

  async createContextualSummary(content, context) {
    // Extract key sentences using embedding similarity
    const sentences = content.split(/[.!?]+/).filter(s => s.trim().length > 10);

    if (sentences.length <= 3) {
      return content; // Already short enough
    }

    // Calculate embeddings for all sentences
    const sentenceEmbeddings = await Promise.all(
      sentences.map(sentence => this.embeddings.embedQuery(sentence.trim()))
    );

    // Find most representative sentences using centrality
    const scores = sentenceEmbeddings.map((embedding, i) => {
      const similarity = sentenceEmbeddings.reduce((sum, other, j) => {
        if (i === j) return sum;
        return sum + this.cosineSimilarity(embedding, other);
      }, 0);
      return { index: i, score: similarity / (sentenceEmbeddings.length - 1) };
    });

    // Select top sentences
    const topSentences = scores
      .sort((a, b) => b.score - a.score)
      .slice(0, Math.max(3, Math.ceil(sentences.length * 0.3)))
      .sort((a, b) => a.index - b.index)
      .map(item => sentences[item.index]);

    return topSentences.join('. ') + '.';
  }

  cosineSimilarity(a, b) {
    if (!a || !b || a.length !== b.length) return 0;

    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    if (normA === 0 || normB === 0) return 0;
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
  }
}

// Custom Context7-like MCP Server for Legal AI
class CustomContext7Server {
  constructor() {
    this.server = new Server(
      {
        name: "custom-context7-enhanced",
        version: "2.0.0",
      },
      {
        capabilities: {
          tools: {},
        },
      }
    );

    // Initialize enhanced services
    this.metaDataEmbedder = null;
    this.fileMonitor = null;
    this.summarizationService = null;
    this.vectorStore = null;
    this.nomicEmbeddings = null;

    // Available libraries and frameworks for our legal AI stack
    this.libraries = {
      sveltekit: {
        id: "/svelte/sveltekit",
        name: "SvelteKit",
        description: "Full-stack Svelte framework with file-based routing",
        version: "2.0.0",
        topics: ["routing", "ssr", "api-routes", "forms", "stores"],
        codeSnippets: 150,
        trustScore: 10,
      },
      "bits-ui": {
        id: "/huntabyte/bits-ui",
        name: "Bits UI",
        description: "Headless component library for Svelte",
        version: "0.21.0",
        topics: ["components", "accessibility", "headless-ui", "svelte5"],
        codeSnippets: 89,
        trustScore: 9,
      },
      "melt-ui": {
        id: "/melt-ui/melt-ui",
        name: "Melt UI",
        description: "Builder library for Svelte components",
        version: "0.84.0",
        topics: ["builders", "accessibility", "components"],
        codeSnippets: 67,
        trustScore: 9,
      },
      "drizzle-orm": {
        id: "/drizzle-team/drizzle-orm",
        name: "Drizzle ORM",
        description: "TypeScript ORM for SQL databases",
        version: "0.36.0",
        topics: ["orm", "typescript", "sql", "migrations", "postgresql"],
        codeSnippets: 124,
        trustScore: 9,
      },
      xstate: {
        id: "/statelyai/xstate",
        name: "XState",
        description: "State management with state machines",
        version: "5.0.0",
        topics: ["state-management", "finite-state-machines", "actors"],
        codeSnippets: 203,
        trustScore: 10,
      },
      unocss: {
        id: "/unocss/unocss",
        name: "UnoCSS",
        description: "Instant on-demand atomic CSS engine",
        version: "0.64.0",
        topics: ["css", "atomic-css", "utilities", "vite"],
        codeSnippets: 78,
        trustScore: 9,
      },
      ollama: {
        id: "/ollama-project/ollama",
        name: "Ollama",
        description:
          "High-throughput and memory-efficient inference engine for LLMs",
        version: "0.6.0",
        topics: ["llm", "inference", "gpu", "batching", "quantization"],
        codeSnippets: 156,
        trustScore: 9,
      },
      ollama: {
        id: "/ollama/ollama",
        name: "Ollama",
        description: "Run large language models locally",
        version: "0.3.0",
        topics: ["llm", "local-ai", "gguf", "quantization", "api"],
        codeSnippets: 89,
        trustScore: 9,
      },
    };

    this.setupHandlers();
  }

  setupHandlers() {
    this.server.setRequestHandler(ListToolsRequestSchema, async () => {
      return {
        tools: [
          {
            name: "resolve-library-id",
            description:
              "Resolve library name to Context7-compatible library ID",
            inputSchema: {
              type: "object",
              properties: {
                libraryName: {
                  type: "string",
                  description: "Library name to search for",
                },
              },
              required: ["libraryName"],
            },
          },
          {
            name: "get-library-docs",
            description: "Get documentation for a library",
            inputSchema: {
              type: "object",
              properties: {
                context7CompatibleLibraryID: {
                  type: "string",
                  description: "Context7-compatible library ID",
                },
                topic: {
                  type: "string",
                  description: "Topic to focus documentation on",
                },
                tokens: {
                  type: "number",
                  description: "Maximum tokens to retrieve",
                  default: 10000,
                },
              },
              required: ["context7CompatibleLibraryID"],
            },
          },
        ],
      };
    });

    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      const { name, arguments: args } = request.params;

      if (name === "resolve-library-id") {
        return this.resolveLibraryId(args.libraryName);
      } else if (name === "get-library-docs") {
        return this.getLibraryDocs(args);
      } else {
        throw new Error(`Unknown tool: ${name}`);
      }
    });
  }

  async resolveLibraryId(libraryName) {
    const searchTerm = libraryName.toLowerCase();
    const matches = Object.entries(this.libraries)
      .filter(
        ([key, lib]) =>
          key.includes(searchTerm) ||
          lib.name.toLowerCase().includes(searchTerm) ||
          lib.description.toLowerCase().includes(searchTerm)
      )
      .map(([_, lib]) => lib)
      .sort((a, b) => b.trustScore - a.trustScore);

    if (matches.length === 0) {
      return {
        content: [
          {
            type: "text",
            text: `No libraries found matching "${libraryName}". Available libraries: ${Object.keys(
              this.libraries
            ).join(", ")}`,
          },
        ],
      };
    }

    const selectedLibrary = matches[0];
    return {
      content: [
        {
          type: "text",
          text: `Selected Library ID: ${selectedLibrary.id}\n\nLibrary: ${selectedLibrary.name}\nDescription: ${selectedLibrary.description}\nVersion: ${selectedLibrary.version}\nTrust Score: ${selectedLibrary.trustScore}/10\nCode Snippets: ${selectedLibrary.codeSnippets}\n\nThis library was chosen as the most relevant match for "${libraryName}".`,
        },
      ],
    };
  }

  async getLibraryDocs(args) {
    const { context7CompatibleLibraryID, topic, tokens = 10000 } = args;

    // Find library by ID
    const library = Object.values(this.libraries).find(
      (lib) => lib.id === context7CompatibleLibraryID
    );

    if (!library) {
      return {
        content: [
          {
            type: "text",
            text: `Library not found: ${context7CompatibleLibraryID}`,
          },
        ],
      };
    }

    // Generate contextual documentation based on our legal AI stack
    let docs = this.generateLibraryDocs(library, topic);

    // Truncate if needed
    if (docs.length > tokens * 4) {
      // Rough estimate: 4 chars per token
      docs = docs.substring(0, tokens * 4) + "...";
    }

    return {
      content: [
        {
          type: "text",
          text: docs,
        },
      ],
    };
  }

  generateLibraryDocs(library, topic) {
    const docs = {
      "/svelte/sveltekit": {
        default: `# SvelteKit Documentation

## Overview
SvelteKit is a full-stack Svelte framework that provides:
- File-based routing
- Server-side rendering (SSR)
- API routes
- Form handling
- Store management

## Key Features for Legal AI Applications

### API Routes
\`\`\`typescript
// src/routes/api/ai/chat/+server.ts
import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

export const POST: RequestHandler = async ({ request }) => {
  const { message } = await request.json();

  // Process with Ollama/Gemma3
  const response = await fetch('http://localhost:11434/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: 'gemma3-legal',
      messages: [{ role: 'user', content: message }],
      stream: true
    })
  });

  return new Response(response.body, {
    headers: { 'Content-Type': 'text/plain' }
  });
};
\`\`\`

### Stores with XState Integration
\`\`\`typescript
// src/lib/stores/chatStore.ts
import { writable } from 'svelte/store';
import { createActor } from 'xstate';
import { chatMachine } from './chatMachine';

const chatActor = createActor(chatMachine);
export const chatStore = writable(chatActor.getSnapshot());

chatActor.subscribe(snapshot => {
  chatStore.set(snapshot);
});

export const useChatActor = () => ({
  send: chatActor.send,
  getSnapshot: () => chatActor.getSnapshot()
});
\`\`\``,
        routing: `# SvelteKit Routing for Legal AI

## File-based Routing Structure
\`\`\`
src/routes/
├── +layout.svelte              # Root layout
├── +page.svelte               # Home page
├── api/
│   ├── ai/
│   │   ├── chat/+server.ts    # AI chat endpoint
│   │   └── models/+server.ts  # Model management
│   └── cases/
│       ├── +server.ts         # CRUD operations
│       └── [id]/+server.ts    # Specific case
├── dashboard/
│   ├── +page.svelte          # Dashboard
│   └── cases/
│       ├── +page.svelte      # Cases list
│       └── [id]/+page.svelte # Case detail
└── gaming-ai-demo/
    └── +page.svelte          # Gaming interface demo
\`\`\`

## Dynamic Routes
\`\`\`typescript
// src/routes/cases/[id]/+page.ts
import type { PageLoad } from './$types';

export const load: PageLoad = async ({ params, fetch }) => {
  const response = await fetch(\`/api/cases/\${params.id}\`);
  const caseData = await response.json();

  return {
    case: caseData
  };
};
\`\`\``,
      },
      "/huntabyte/bits-ui": {
        default: `# Bits UI Documentation

## Overview
Bits UI is a headless component library for Svelte that provides:
- Accessible components
- Customizable styling
- TypeScript support
- Svelte 5 compatibility

## Key Components for Legal AI

### Dialog/Modal Components
\`\`\`svelte
<script lang="ts">
  import { Dialog } from 'bits-ui';

  let open = false;
</script>

<Dialog.Root bind:open>
  <Dialog.Trigger class="btn-primary">
    Open Case Details
  </Dialog.Trigger>

  <Dialog.Portal>
    <Dialog.Overlay class="overlay" />
    <Dialog.Content class="modal-content">
      <Dialog.Title>Case Information</Dialog.Title>
      <Dialog.Description>
        Review case details and evidence
      </Dialog.Description>

      <!-- Case content here -->

      <Dialog.Close class="btn-secondary">Close</Dialog.Close>
    </Dialog.Content>
  </Dialog.Portal>
</Dialog.Root>
\`\`\`

### Select/Dropdown Components
\`\`\`svelte
<script lang="ts">
  import { Select } from 'bits-ui';

  let selectedModel = 'gemma3-legal';
  const models = [
    { value: 'gemma3-legal', label: 'Gemma 3 Legal' },
    { value: 'llama3.1', label: 'Llama 3.1' },
    { value: 'mistral', label: 'Mistral 7B' }
  ];
</script>

<Select.Root bind:selected={selectedModel}>
  <Select.Trigger class="select-trigger">
    <Select.Value placeholder="Select AI Model" />
  </Select.Trigger>

  <Select.Content class="select-content">
    {#each models as model}
      <Select.Item value={model.value}>
        {model.label}
      </Select.Item>
    {/each}
  </Select.Content>
</Select.Root>
\`\`\``,
      },
      "/drizzle-team/drizzle-orm": {
        default: `# Drizzle ORM Documentation

## Schema Definition for Legal AI
\`\`\`typescript
// src/lib/db/schema.ts
import { pgTable, serial, text, timestamp, jsonb, boolean } from 'drizzle-orm/pg-core';

export const cases = pgTable('cases', {
  id: serial('id').primaryKey(),
  title: text('title').notNull(),
  description: text('description'),
  status: text('status').default('open'),
  evidence: jsonb('evidence'),
  aiAnalysis: jsonb('ai_analysis'),
  createdAt: timestamp('created_at').defaultNow(),
  updatedAt: timestamp('updated_at').defaultNow()
});

export const aiInteractions = pgTable('ai_interactions', {
  id: serial('id').primaryKey(),
  caseId: serial('case_id').references(() => cases.id),
  userMessage: text('user_message').notNull(),
  aiResponse: text('ai_response').notNull(),
  model: text('model').notNull(),
  timestamp: timestamp('timestamp').defaultNow(),
  tokens: serial('tokens'),
  processingTime: serial('processing_time')
});
\`\`\`

## Database Operations
\`\`\`typescript
// src/lib/db/queries.ts
import { db } from './connection';
import { cases, aiInteractions } from './schema';
import { eq, desc } from 'drizzle-orm';

export async function createCase(data: {
  title: string;
  description?: string;
  evidence?: any;
}) {
  return await db.insert(cases).values(data).returning();
}

export async function getCaseWithAI(caseId: number) {
  return await db
    .select()
    .from(cases)
    .leftJoin(aiInteractions, eq(aiInteractions.caseId, cases.id))
    .where(eq(cases.id, caseId))
    .orderBy(desc(aiInteractions.timestamp));
}
\`\`\``,
      },
    };

    const libraryDocs = docs[library.id] || {};
    const topicDocs = topic ? libraryDocs[topic] : null;

    return (
      topicDocs ||
      libraryDocs.default ||
      `# ${library.name}\n\n${library.description}\n\nVersion: ${
        library.version
      }\nTopics: ${library.topics.join(
        ", "
      )}\n\nDocumentation for this library is being generated...`
    );
  }

  async run() {
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    console.error("Custom Context7 MCP server running on stdio");

    const app = express();
    app.use(bodyParser.json());

      // --- Vector Store Integration (Nomic Embed via Ollama) ---
      // Uses local Ollama embeddings: nomic-embed-text by default.
      const OLLAMA_URL = process.env.OLLAMA_URL || "http://localhost:11434";
      const OLLAMA_MODEL = process.env.OLLAMA_MODEL || "gemma3-legal"; // chat/generation
      const OLLAMA_EMBED_MODEL = process.env.OLLAMA_EMBED_MODEL || "nomic-embed-text";
      const PG_CONN =
        process.env.PG_VECTOR_URL ||
        process.env.PGVECTOR_CONNECTION_STRING ||
        process.env.DATABASE_URL;

      class NomicOllamaEmbeddings {
        constructor({ baseUrl = OLLAMA_URL, model = OLLAMA_EMBED_MODEL } = {}) {
        this.baseUrl = baseUrl;
        this.model = model;
        }
        async embedQuery(input) {
        const res = await fetch(`${this.baseUrl}/api/embeddings`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          // Ollama embeddings expects { input }
          body: JSON.stringify({ model: this.model, input }),
        });
        if (!res.ok) {
          const text = await res.text();
          throw new Error(`Ollama embed error: ${res.status} ${text}`);
        }
        const data = await res.json();
        // Support multiple shapes: { embedding }, { embeddings }, { data: [{ embedding }] }
        if (Array.isArray(data?.embedding)) return data.embedding;
        if (Array.isArray(data?.embeddings)) {
          if (Array.isArray(data.embeddings[0])) return data.embeddings[0];
          return data.embeddings;
        }
        if (Array.isArray(data?.data) && Array.isArray(data.data[0]?.embedding)) {
          return data.data[0].embedding;
        }
        return [];
        }
        async embedDocuments(texts) {
        return Promise.all(texts.map((t) => this.embedQuery(t)));
        }
      }

      // --- Simple metadata-aware embedder ---
      function buildIndexText({ title, content, metadata = {} }) {
        const metaBits = [];
        // Common metadata fields
        if (metadata.caseId) metaBits.push(`Case: ${metadata.caseId}`);
        if (metadata.tags) metaBits.push(`Tags: ${[].concat(metadata.tags).join(", ")}`);
        if (metadata.keywords) metaBits.push(`Keywords: ${[].concat(metadata.keywords).join(", ")}`);
        if (metadata.entities) metaBits.push(`Entities: ${[].concat(metadata.entities).join(", ")}`);
        if (metadata.summary) metaBits.push(`Summary: ${metadata.summary}`);
        // Flatten any extra top-level metadata keys
        const extra = Object.entries(metadata)
          .filter(([k]) => !["caseId","tags","keywords","entities","summary"].includes(k))
          .map(([k,v]) => `${k}: ${typeof v === "string" ? v : JSON.stringify(v)}`);
        if (extra.length) metaBits.push(`Meta: ${extra.join("; ")}`);
        const header = [title ? `Title: ${title}` : null, ...metaBits].filter(Boolean).join("\n");
        return `${header}\n\n${content || ""}`.trim();
      }

      function normalize(vec) {
        if (!Array.isArray(vec) || vec.length === 0) return vec || [];
        const norm = Math.sqrt(vec.reduce((s, x) => s + x * x, 0)) || 1;
        return vec.map((x) => x / norm);
      }

      function combineEmbeddings(contentVec = [], metaVec = [], wContent = 0.8, wMeta = 0.2) {
        // Pad to same length
        const len = Math.max(contentVec.length || 0, metaVec.length || 0);
        const a = Array.from({ length: len }, (_, i) => contentVec[i] ?? 0);
        const b = Array.from({ length: len }, (_, i) => metaVec[i] ?? 0);
        const out = a.map((x, i) => wContent * x + wMeta * b[i]);
        return normalize(out);
      }

      let vectorStore = null;
      try {
        if (!PG_CONN) {
        console.warn("PG connection string not set; semantic search disabled.");
        } else {
        const { Pool } = await import("pg");
        const pool = new Pool({ connectionString: PG_CONN });
        // Ensure your pgvector table matches your embedding dimensions.
        const embeddings = new NomicOllamaEmbeddings();
        vectorStore = await PGVectorStore.initialize(embeddings, {
          pool,
          tableName: process.env.PGVECTOR_TABLE || "documents",
          // schema: process.env.PGVECTOR_SCHEMA || "public",
        });
        }
      } catch (err) {
        console.error("Vector store initialization failed:", err);
        vectorStore = null;
      }

      // --- Semantic search ---
      app.post("/api/semantic-search", async (req, res) => {
        const { query, k = 4 } = req.body || {};
        if (!query) {
        return res.status(400).json({ error: "Query is required" });
        }
        try {
        if (!vectorStore) {
          return res.status(503).json({ error: "Vector store not configured" });
        }
        const results = await vectorStore.similaritySearch(query, Number(k));
        res.json({ results });
        } catch (err) {
        res.status(500).json({ error: err.message });
        }
      });

      // --- Metadata-aware indexing ---
      // Accepts: { documents: [{ id?, title?, content, metadata? }] }
      app.post("/api/index", async (req, res) => {
        try {
          if (!vectorStore) return res.status(503).json({ error: "Vector store not configured" });
          const { documents } = req.body || {};
          if (!Array.isArray(documents) || documents.length === 0) {
            return res.status(400).json({ error: "documents[] required" });
          }
          const docs = documents.map((d) => ({
            pageContent: buildIndexText({ title: d.title, content: d.content, metadata: d.metadata }),
            metadata: { ...(d.metadata || {}), _title: d.title || null, _id: d.id || null },
          }));
          await vectorStore.addDocuments(docs);
          res.json({ indexed: docs.length });
        } catch (err) {
          console.error("/api/index error:", err);
          res.status(500).json({ error: err.message });
        }
      });

      // --- Summarize using local Ollama (fallback: simple extract) ---
      app.post("/api/summarize", async (req, res) => {
        const { text, maxTokens = 256, model = OLLAMA_MODEL } = req.body || {};
        if (!text) return res.status(400).json({ error: "text is required" });
        try {
          // Try Ollama /api/generate
          const r = await fetch(`${OLLAMA_URL}/api/generate`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              model,
              prompt: `Summarize the following legal text in ${Math.max(64, Math.min(512, maxTokens))} tokens, preserving legal context and key entities.\n\n===\n${text}`,
              stream: false,
            }),
          });
          if (r.ok) {
            const data = await r.json();
            const summary = data?.response || data?.content || "";
            return res.json({ summary });
          }
          // Fallback: extract top sentences by length
          const sentences = String(text).split(/(?<=[.!?])\s+/).slice(0, 8);
          const summary = sentences.sort((a, b) => b.length - a.length).slice(0, 3).join(" ");
          return res.json({ summary });
        } catch (err) {
          const sentences = String(text).split(/(?<=[.!?])\s+/).slice(0, 8);
          const summary = sentences.sort((a, b) => b.length - a.length).slice(0, 3).join(" ");
          return res.json({ summary, note: "fallback" });
        }
      });

      // --- Optional file watcher to auto-index text/markdown files ---
      const CONTENT_DIR = process.env.CONTENT_DIR || path.resolve(process.cwd(), "docs");
      const WATCH_FILES = (process.env.WATCH_FILES || "false").toLowerCase() === "true";

      async function indexFile(filePath) {
        try {
          if (!vectorStore) return;
          const ext = path.extname(filePath).toLowerCase();
          if (![".md", ".txt"].includes(ext)) return; // Only text/markdown for now
          const raw = await fs.readFile(filePath, "utf8");
          const title = path.basename(filePath).replace(ext, "");
          const metadata = { path: filePath, ext };
          const doc = { pageContent: buildIndexText({ title, content: raw, metadata }), metadata: { ...metadata, _title: title } };
          await vectorStore.addDocuments([doc]);
          console.log(`Indexed: ${filePath}`);
        } catch (e) {
          console.warn(`Index failed for ${filePath}:`, e.message || e);
        }
      }

      async function startWatcher() {
        try {
          await fs.mkdir(CONTENT_DIR, { recursive: true });
          if (!_chokidar) {
            try {
              const mod = await import("chokidar");
              _chokidar = mod?.default || mod;
            } catch (e) {
              console.warn("chokidar not installed; file watching disabled");
              return;
            }
          }
          const watcher = _chokidar.watch([`${CONTENT_DIR}/**/*.md`, `${CONTENT_DIR}/**/*.txt`], { ignoreInitial: false });
          watcher.on("add", indexFile);
          watcher.on("change", indexFile);
          console.log(`Watching content dir: ${CONTENT_DIR}`);
        } catch (e) {
          console.warn("Watcher failed:", e.message || e);
        }
      }

      if (WATCH_FILES) {
        startWatcher();
      }

      app.listen(3000, () => {
        console.log("Custom Context7 MCP server running on port 3000");
      });
      }
    }

    const server = new CustomContext7Server();
    server.run().catch(console.error);
