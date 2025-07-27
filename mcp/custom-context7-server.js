#!/usr/bin/env node

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import express from "express";
import bodyParser from "body-parser";
import { PGVectorStore } from "@langchain/community/vectorstores/pgvector";
import { OpenAIEmbeddings } from "@langchain/openai";

// Custom Context7-like MCP Server for Legal AI
class CustomContext7Server {
  constructor() {
    this.server = new Server(
      {
        name: "custom-context7",
        version: "0.1.0",
      },
      {
        capabilities: {
          tools: {},
        },
      }
    );

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
      vllm: {
        id: "/vllm-project/vllm",
        name: "vLLM",
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

    const embeddings = new OpenAIEmbeddings({
      modelName: "nomic-embed-text",
      openAIApiKey: "N/A",
      baseURL: "http://localhost:8000/v1",
    });
    // You must provide your actual pool config here
    // const vectorStore = new PGVectorStore({ pool });

    app.post("/api/semantic-search", async (req, res) => {
      const { query } = req.body;
      if (!query) {
        return res.status(400).json({ error: "Query is required" });
      }
      try {
        // Uncomment and configure vectorStore before using
        // const results = await vectorStore.similaritySearch(query, 4);
        // res.json({ results });
        res.json({ results: [] }); // Placeholder
      } catch (err) {
        res.status(500).json({ error: err.message });
      }
    });

    app.listen(3000, () => {
      console.log("Custom Context7 MCP server running on port 3000");
    });
  }
}

const server = new CustomContext7Server();
server.run().catch(console.error);
