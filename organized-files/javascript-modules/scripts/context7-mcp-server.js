#!/usr/bin/env node

/**
 * Context7 MCP Server for Legal AI Integration
 * Enhanced with stack config, vLLM routing, and SvelteKit best practices
 */

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { 
  CallToolRequestSchema, 
  ErrorCode, 
  ListToolsRequestSchema, 
  McpError 
} from '@modelcontextprotocol/sdk/types.js';
import fs from 'fs/promises';
import path from 'path';

const PROJECT_ROOT = process.env.PROJECT_ROOT || process.cwd();
const DOCS_PATH = process.env.DOCS_PATH || path.join(PROJECT_ROOT, 'context7-docs');
const VLLM_ENDPOINT = process.env.VLLM_ENDPOINT || 'http://localhost:8000';
const OLLAMA_ENDPOINT = process.env.OLLAMA_ENDPOINT || 'http://localhost:11434';
const VLLM_ENABLED = process.env.VLLM_ENABLED === 'true';

// Stack configuration for Context7
const STACK_CONFIG = {
  frontend: {
    framework: 'SvelteKit 5',
    styling: 'UnoCSS + TailwindCSS',
    components: 'Bits UI v2',
    stateManagement: 'XState + Svelte stores',
    icons: 'Lucide Svelte',
    canvas: 'Fabric.js'
  },
  backend: {
    database: 'PostgreSQL + Drizzle ORM',
    vector: 'pgvector + Qdrant',
    ai: 'Gemma3 Legal + Ollama + vLLM',
    agents: 'Autogen + CrewAI',
    rag: 'RAG Backend with FastAPI'
  },
  docs: {
    sveltekit: 'SvelteKit 2 best practices',
    drizzle: 'Drizzle ORM patterns',
    unocss: 'UnoCSS utility classes',
    'bits-ui': 'Accessible UI components',
    xstate: 'State machine patterns',
    'fabric-js': 'Canvas manipulation',
    rag: 'RAG system integration patterns'
  }
};

// RAG System configuration
const RAG_CONFIG = {
  endpoint: process.env.RAG_ENDPOINT || 'http://localhost:8000',
  enabled: process.env.RAG_ENABLED !== 'false',
  vectorStore: 'pgvector + Qdrant',
  embeddingModel: 'sentence-transformers/all-MiniLM-L6-v2',
  queryThreshold: 0.7
};

class Context7MCPServer {
  constructor() {
    this.server = new Server(
      {
        name: 'context7-legal-mcp',
        version: '1.0.0',
      },
      {
        capabilities: {
          tools: {},
        },
      }
    );

    this.setupToolHandlers();
  }

  setupToolHandlers() {
    // List available tools
    this.server.setRequestHandler(ListToolsRequestSchema, async () => {
      return {
        tools: [
          {
            name: 'analyze-stack',
            description: 'Analyze any component with context-aware suggestions for the legal AI stack',
            inputSchema: {
              type: 'object',
              properties: {
                component: {
                  type: 'string',
                  description: 'Component to analyze (sveltekit, drizzle, unocss, bits-ui, xstate, etc.)'
                },
                context: {
                  type: 'string',
                  enum: ['legal-ai', 'gaming-ui', 'performance'],
                  description: 'Analysis context'
                }
              },
              required: ['component']
            }
          },
          {
            name: 'generate-best-practices',
            description: 'Generate best practices for specific areas of the legal AI application',
            inputSchema: {
              type: 'object',
              properties: {
                area: {
                  type: 'string',
                  enum: ['performance', 'security', 'ui-ux'],
                  description: 'Area to generate best practices for'
                }
              },
              required: ['area']
            }
          },
          {
            name: 'suggest-integration',
            description: 'Suggest integration patterns for new features with legal AI requirements',
            inputSchema: {
              type: 'object',
              properties: {
                feature: {
                  type: 'string',
                  description: 'Feature to integrate'
                },
                requirements: {
                  type: 'string',
                  description: 'Specific requirements or constraints'
                }
              },
              required: ['feature']
            }
          },
          {
            name: 'resolve-library-id',
            description: 'Find Context7-compatible library IDs for the stack',
            inputSchema: {
              type: 'object',
              properties: {
                library: {
                  type: 'string',
                  description: 'Library name to resolve'
                }
              },
              required: ['library']
            }
          },
          {
            name: 'get-library-docs',
            description: 'Retrieve specific documentation with topic filtering',
            inputSchema: {
              type: 'object',
              properties: {
                libraryId: {
                  type: 'string',
                  description: 'Library ID to get docs for'
                },
                topic: {
                  type: 'string',
                  description: 'Specific topic within the library docs'
                }
              },
              required: ['libraryId']
            }
          },
          {
            name: 'get_legal_docs',
            description: 'Get legal documentation and best practices',
            inputSchema: {
              type: 'object',
              properties: {
                docType: {
                  type: 'string',
                  enum: ['sveltekit', 'drizzle', 'melt-ui', 'legal-integration', 'project-overview'],
                  description: 'Type of documentation to retrieve'
                }
              },
              required: ['docType']
            }
          },
          {
            name: 'get_project_status',
            description: 'Get current project status and integration plan',
            inputSchema: {
              type: 'object',
              properties: {
                component: {
                  type: 'string',
                  enum: ['database', 'api', 'frontend', 'ai-integration', 'overall'],
                  description: 'Project component to check status for'
                }
              }
            }
          },
          {
            name: 'get_legal_schema',
            description: 'Get legal database schema and table definitions',
            inputSchema: {
              type: 'object',
              properties: {
                tableType: {
                  type: 'string',
                  enum: ['legal-documents', 'precedents', 'analysis-sessions', 'all'],
                  description: 'Type of legal tables to describe'
                }
              }
            }
          },
          {
            name: 'get_api_endpoints',
            description: 'Get available legal AI API endpoints and their usage',
            inputSchema: {
              type: 'object',
              properties: {
                category: {
                  type: 'string',
                  enum: ['legal-chat', 'precedent-search', 'document-analysis', 'all'],
                  description: 'Category of API endpoints to list'
                }
              }
            }
          },
          {
            name: 'get_gemma3_config',
            description: 'Get Gemma3 legal model configuration and setup',
            inputSchema: {
              type: 'object',
              properties: {}
            }
          },
          {
            name: 'rag-query',
            description: 'Perform RAG query against legal documents with semantic search',
            inputSchema: {
              type: 'object',
              properties: {
                query: {
                  type: 'string',
                  description: 'Legal query to search for'
                },
                maxResults: {
                  type: 'number',
                  description: 'Maximum number of results to return',
                  default: 5
                },
                confidenceThreshold: {
                  type: 'number',
                  description: 'Minimum confidence threshold (0.0-1.0)',
                  default: 0.7
                },
                caseId: {
                  type: 'string',
                  description: 'Optional case ID to filter results'
                },
                documentTypes: {
                  type: 'array',
                  items: { type: 'string' },
                  description: 'Optional document types to filter'
                }
              },
              required: ['query']
            }
          },
          {
            name: 'rag-upload-document',
            description: 'Upload and index a legal document in the RAG system',
            inputSchema: {
              type: 'object',
              properties: {
                filePath: {
                  type: 'string',
                  description: 'Path to the document file'
                },
                caseId: {
                  type: 'string',
                  description: 'Case ID to associate with the document'
                },
                documentType: {
                  type: 'string',
                  description: 'Type of legal document',
                  default: 'general'
                },
                title: {
                  type: 'string',
                  description: 'Document title'
                }
              },
              required: ['filePath']
            }
          },
          {
            name: 'rag-get-stats',
            description: 'Get RAG system statistics and collection information',
            inputSchema: {
              type: 'object',
              properties: {}
            }
          },
          {
            name: 'rag-analyze-relevance',
            description: 'Analyze document relevance for a specific query',
            inputSchema: {
              type: 'object',
              properties: {
                query: {
                  type: 'string',
                  description: 'Query to analyze relevance for'
                },
                documentId: {
                  type: 'string',
                  description: 'Document ID to analyze'
                }
              },
              required: ['query', 'documentId']
            }
          },
          {
            name: 'rag-integration-guide',
            description: 'Get guidance on integrating RAG with SvelteKit frontend',
            inputSchema: {
              type: 'object',
              properties: {
                integrationType: {
                  type: 'string',
                  enum: ['api-integration', 'component-integration', 'search-ui', 'document-upload'],
                  description: 'Type of integration guidance needed'
                }
              },
              required: ['integrationType']
            }
          }
        ]
      };
    });

    // Handle tool calls
    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      const { name, arguments: args } = request.params;

      try {
        switch (name) {
          case 'analyze-stack':
            return await this.analyzeStack(args.component, args.context);
          case 'generate-best-practices':
            return await this.generateBestPractices(args.area);
          case 'suggest-integration':
            return await this.suggestIntegration(args.feature, args.requirements);
          case 'resolve-library-id':
            return await this.resolveLibraryId(args.library);
          case 'get-library-docs':
            return await this.getLibraryDocs(args.libraryId, args.topic);
          case 'get_legal_docs':
            return await this.getLegalDocs(args.docType);
          case 'get_project_status':
            return await this.getProjectStatus(args.component);
          case 'get_legal_schema':
            return await this.getLegalSchema(args.tableType);
          case 'get_api_endpoints':
            return await this.getApiEndpoints(args.category);
          case 'get_gemma3_config':
            return await this.getGemma3Config();
          case 'rag-query':
            return await this.ragQuery(args);
          case 'rag-upload-document':
            return await this.ragUploadDocument(args);
          case 'rag-get-stats':
            return await this.ragGetStats();
          case 'rag-analyze-relevance':
            return await this.ragAnalyzeRelevance(args);
          case 'rag-integration-guide':
            return await this.ragIntegrationGuide(args);
          default:
            throw new McpError(ErrorCode.MethodNotFound, `Unknown tool: ${name}`);
        }
      } catch (error) {
        console.error(`Error in tool ${name}:`, error);
        throw new McpError(ErrorCode.InternalError, `Tool execution failed: ${error.message}`);
      }
    });
  }

  // Context7 Stack Analysis Tools
  async analyzeStack(component, context = 'legal-ai') {
    const stackData = STACK_CONFIG;
    const analysisPrompt = `Analyze ${component} in the context of ${context} for a legal AI application.
    
Current Stack: ${JSON.stringify(stackData, null, 2)}

Provide specific recommendations for:
1. Best practices for ${component} in legal AI context
2. Performance optimizations
3. Integration patterns with other stack components
4. Security considerations for legal data
5. Code examples and patterns`;

    // Use vLLM if enabled, fallback to structured response
    if (VLLM_ENABLED) {
      try {
        const response = await this.queryVLLM(analysisPrompt);
        return {
          content: [{
            type: 'text',
            text: `# Stack Analysis: ${component} (${context})\n\n${response}`
          }]
        };
      } catch (error) {
        console.warn('vLLM query failed, using structured response:', error);
      }
    }

    // Structured fallback response
    const componentAnalysis = this.getComponentAnalysis(component, context);
    return {
      content: [{
        type: 'text',
        text: `# Stack Analysis: ${component} (${context})\n\n${componentAnalysis}`
      }]
    };
  }

  async generateBestPractices(area) {
    const practicesMap = {
      performance: `# Performance Best Practices for Legal AI

## SvelteKit Optimization
- Use server-side rendering for legal document pages
- Implement progressive enhancement for offline access
- Optimize bundle size with dynamic imports for AI features
- Use Svelte stores efficiently for case data management

## Database Performance (Drizzle + PostgreSQL)
- Index case_id, evidence_id, and user_id columns
- Use prepared statements for frequent legal queries
- Implement connection pooling for multi-user access
- Optimize vector similarity searches with proper indexing

## AI Model Performance
- Use vLLM for high-throughput legal analysis
- Implement model caching for repeated document analysis
- Batch similar requests for evidence processing
- Monitor GPU memory usage with legal model workloads

## UnoCSS Optimization
- Use JIT compilation for legal UI components
- Minimize CSS bundle size with purging
- Implement design tokens for consistent legal branding
- Cache compiled styles for production deployment`,

      security: `# Security Best Practices for Legal AI

## Data Protection
- Encrypt sensitive legal documents at rest and in transit
- Implement row-level security (RLS) for case data
- Use secure authentication for prosecutor access
- Audit all evidence access and modifications

## API Security
- Implement JWT authentication with short expiration
- Use HTTPS for all legal data transmission
- Validate and sanitize all evidence uploads
- Rate limit AI analysis requests to prevent abuse

## Legal Compliance
- Ensure GDPR compliance for EU legal data
- Implement data retention policies for case evidence
- Secure backup procedures for legal documents
- Monitor access logs for compliance auditing

## AI Model Security
- Validate AI model outputs for legal accuracy
- Implement content filtering for sensitive information
- Secure model endpoints with authentication
- Monitor AI usage for unusual patterns`,

      'ui-ux': `# UI/UX Best Practices for Legal AI

## Legal Professional Workflow
- Design case-centric navigation for prosecutors
- Implement quick evidence search and filtering
- Use progressive disclosure for complex legal data
- Provide clear AI confidence indicators

## Accessibility
- Follow WCAG 2.1 AA standards for legal professionals
- Implement keyboard navigation for evidence review
- Use high contrast modes for document analysis
- Provide screen reader support for case summaries

## Bits UI Integration
- Use Dialog components for evidence details
- Implement Command palette for quick case navigation
- Use Select components for legal category filtering
- Apply consistent focus management throughout

## Legal Document Handling
- Implement PDF viewer with annotation support
- Use infinite scroll for large evidence lists
- Provide thumbnail previews for document evidence
- Enable bulk operations for evidence management`
    };

    const practices = practicesMap[area] || `Best practices for ${area} not yet implemented.`;
    
    return {
      content: [{
        type: 'text',
        text: practices
      }]
    };
  }

  async suggestIntegration(feature, requirements = '') {
    const integrationPrompt = `Suggest integration patterns for "${feature}" in a SvelteKit legal AI application.

Requirements: ${requirements}

Current Stack:
- SvelteKit 5 + TypeScript
- UnoCSS + Bits UI v2
- Drizzle ORM + PostgreSQL + pgvector
- Gemma3 Legal AI + vLLM
- Autogen + CrewAI multi-agent systems

Provide:
1. Recommended integration approach
2. Code structure and file organization
3. Database schema considerations
4. API endpoint design
5. Frontend component patterns
6. Security and performance considerations`;

    if (VLLM_ENABLED) {
      try {
        const response = await this.queryVLLM(integrationPrompt);
        return {
          content: [{
            type: 'text',
            text: `# Integration Suggestion: ${feature}\n\n${response}`
          }]
        };
      } catch (error) {
        console.warn('vLLM query failed, using template response:', error);
      }
    }

    // Template-based integration suggestions
    const integrationTemplate = this.getIntegrationTemplate(feature, requirements);
    return {
      content: [{
        type: 'text',
        text: `# Integration Suggestion: ${feature}\n\n${integrationTemplate}`
      }]
    };
  }

  async resolveLibraryId(library) {
    const libraryMap = {
      'sveltekit': 'sveltekit',
      'svelte': 'sveltekit',
      'drizzle': 'drizzle',
      'drizzle-orm': 'drizzle',
      'unocss': 'unocss',
      'uno': 'unocss',
      'tailwind': 'unocss', // We use UnoCSS instead
      'bits-ui': 'bits-ui',
      'melt-ui': 'bits-ui', // Bits UI is based on Melt UI
      'xstate': 'xstate',
      'fabric': 'fabric-js',
      'fabric.js': 'fabric-js',
      'lucide': 'lucide-svelte',
      'postgres': 'drizzle',
      'postgresql': 'drizzle'
    };

    const resolved = libraryMap[library.toLowerCase()] || library;
    
    return {
      content: [{
        type: 'text',
        text: `# Library ID Resolution\n\nLibrary: ${library}\nResolved ID: ${resolved}\n\nAvailable documentation:\n${Object.keys(libraryMap).join(', ')}`
      }]
    };
  }

  async getLibraryDocs(libraryId, topic = 'overview') {
    const docsMap = {
      'sveltekit': {
        'routing': 'SvelteKit file-based routing with +page.svelte and +page.ts patterns',
        'ssr': 'Server-side rendering configuration and optimization',
        'forms': 'Form handling with actions and validation',
        'stores': 'Svelte store patterns and reactive state management'
      },
      'drizzle': {
        'schema': 'Database schema definition with TypeScript types',
        'queries': 'Type-safe query building and execution',
        'migrations': 'Database migration management with drizzle-kit',
        'relations': 'Table relationships and joins'
      },
      'unocss': {
        'utilities': 'Utility class patterns and JIT compilation',
        'presets': 'UnoCSS preset configuration and customization',
        'performance': 'Build-time optimization and purging',
        'integration': 'SvelteKit integration and configuration'
      },
      'bits-ui': {
        'dialog': 'Modal dialog components with accessibility',
        'command': 'Command palette and search interfaces',
        'select': 'Dropdown and multi-select components',
        'forms': 'Form components with validation'
      }
    };

    const libraryDocs = docsMap[libraryId];
    if (!libraryDocs) {
      return {
        content: [{
          type: 'text',
          text: `# Documentation: ${libraryId}\n\nDocumentation for ${libraryId} not yet available.\n\nAvailable libraries: ${Object.keys(docsMap).join(', ')}`
        }]
      };
    }

    const topicDoc = libraryDocs[topic] || `Topic '${topic}' not found. Available topics: ${Object.keys(libraryDocs).join(', ')}`;
    
    return {
      content: [{
        type: 'text',
        text: `# ${libraryId} Documentation: ${topic}\n\n${topicDoc}`
      }]
    };
  }

  // Helper methods
  getComponentAnalysis(component, context) {
    const analyses = {
      'sveltekit': `## SvelteKit Analysis for Legal AI

### Recommended Patterns
- Use +page.server.ts for secure legal data loading
- Implement form actions for evidence uploads
- Use load functions for case data hydration
- Apply proper error boundaries for legal workflows

### Legal AI Integration
- Create dedicated routes for AI analysis (/cases/[id]/ai-analysis)
- Use server-side AI processing for sensitive legal data
- Implement streaming for long-running legal analysis
- Cache AI results with proper invalidation

### Performance Considerations
- Lazy load heavy legal components
- Use Svelte's reactive statements for case data updates
- Implement proper loading states for AI operations
- Optimize for mobile legal professional workflows`,

      'drizzle': `## Drizzle ORM Analysis for Legal AI

### Schema Best Practices
- Use UUID primary keys for case and evidence tables
- Implement proper foreign key relationships
- Add indexes for legal search queries
- Use JSONB for flexible evidence metadata

### Legal Data Patterns
- Implement soft deletes for legal audit trails
- Use row-level security for prosecutor access
- Create views for common legal queries
- Implement versioning for evidence changes

### Vector Integration
- Use pgvector extension for legal document similarity
- Index embeddings for fast legal precedent search
- Implement hybrid search (text + vector)
- Optimize for legal document clustering`,

      'unocss': `## UnoCSS Analysis for Legal AI

### Legal UI Patterns
- Use consistent color schemes for case urgency
- Implement responsive layouts for evidence review
- Create utility classes for legal document styling
- Use dark mode variants for long reading sessions

### Performance Optimization
- Configure JIT compilation for legal components
- Use preset configurations for legal branding
- Minimize bundle size with selective imports
- Implement efficient CSS purging`,
    };

    return analyses[component] || `Analysis for ${component} in ${context} context not yet implemented.`;
  }

  getIntegrationTemplate(feature, requirements) {
    return `## Integration Approach for ${feature}

### 1. File Structure
\`\`\`
src/
├── routes/
│   └── api/${feature.toLowerCase()}/
│       └── +server.ts
├── lib/
│   ├── components/${feature}/
│   ├── stores/${feature}Store.ts
│   └── types/${feature}.ts
└── database/
    └── schema/${feature}.sql
\`\`\`

### 2. Database Schema
- Add tables for ${feature} data
- Implement proper relationships with existing case/evidence tables
- Add indexes for query performance
- Consider audit trail requirements

### 3. API Design
- RESTful endpoints following SvelteKit conventions
- Proper error handling and validation
- Authentication/authorization checks
- Rate limiting for AI features

### 4. Frontend Components
- Use Bits UI components for accessibility
- Implement proper loading and error states
- Follow legal UI patterns and branding
- Ensure mobile responsiveness

### 5. Requirements Analysis
${requirements || 'No specific requirements provided'}

### 6. Security Considerations
- Input validation and sanitization
- Proper authentication for legal data
- Audit logging for compliance
- Data encryption for sensitive information`;
  }

  async queryVLLM(prompt) {
    const response = await fetch(`${VLLM_ENDPOINT}/legal-analysis`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt,
        max_tokens: 2048,
        temperature: 0.3,
        stream: false
      })
    });

    if (!response.ok) {
      throw new Error(`vLLM request failed: ${response.statusText}`);
    }

    const data = await response.json();
    return data.analysis || data.response || 'No response from vLLM';
  }

  async getLegalDocs(docType) {
    const docFiles = {
      'sveltekit': 'sveltekit2.md',
      'drizzle': 'drizzle.md', 
      'melt-ui': 'melt-ui.md',
      'legal-integration': 'CONTEXT7-INTEGRATION-PLAN.md',
      'project-overview': 'PROJECT-OVERVIEW-CONTEXT7.md'
    };

    const fileName = docFiles[docType];
    if (!fileName) {
      throw new Error(`Unknown doc type: ${docType}`);
    }

    const filePath = path.join(DOCS_PATH, fileName);
    const content = await fs.readFile(filePath, 'utf-8');

    return {
      content: [{
        type: 'text',
        text: `# ${docType.toUpperCase()} Documentation\n\n${content}`
      }]
    };
  }

  async getProjectStatus(component = 'overall') {
    const statusInfo = {
      database: {
        status: 'Enhanced with Context7 legal tables',
        tables: ['legal_documents', 'legal_precedents', 'legal_analysis_sessions'],
        features: ['Vector embeddings', 'Legal categorization', 'Precedent linking']
      },
      api: {
        status: 'Legal endpoints implemented',
        endpoints: ['/api/legal/chat', '/api/legal/precedents', '/api/legal/documents'],
        features: ['Gemma3 integration', 'Vector search', 'Analysis sessions']
      },
      frontend: {
        status: 'Legal components created',
        components: ['LegalAnalysisDialog', 'LegalPrecedentSearch'],
        features: ['Real-time analysis', 'Precedent search', 'Source citation']
      },
      'ai-integration': {
        status: 'Gemma3 Legal model integrated',
        model: 'mohf16-Q4_K_M.gguf',
        features: ['Legal chat', 'Document analysis', 'Precedent matching']
      },
      overall: {
        status: 'Context7 integration complete',
        phase: 'Phase 3 - AI Integration (Complete)',
        readiness: '95% - Ready for production testing'
      }
    };

    const info = statusInfo[component] || statusInfo.overall;

    return {
      content: [{
        type: 'text',
        text: `# Project Status: ${component.toUpperCase()}\n\n` +
              `**Status**: ${info.status}\n\n` +
              (info.tables ? `**Tables**: ${info.tables.join(', ')}\n\n` : '') +
              (info.endpoints ? `**Endpoints**: ${info.endpoints.join(', ')}\n\n` : '') +
              (info.components ? `**Components**: ${info.components.join(', ')}\n\n` : '') +
              (info.model ? `**Model**: ${info.model}\n\n` : '') +
              (info.phase ? `**Phase**: ${info.phase}\n\n` : '') +
              (info.readiness ? `**Readiness**: ${info.readiness}\n\n` : '') +
              `**Features**: ${info.features.join(', ')}`
      }]
    };
  }

  async getLegalSchema(tableType = 'all') {
    const schemas = {
      'legal-documents': `
CREATE TABLE legal_documents (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  content TEXT NOT NULL,
  embedding TEXT, -- 1536-dimensional vector for legal embeddings
  legal_category VARCHAR(50),
  case_reference VARCHAR(100),
  jurisdiction VARCHAR(50),
  document_type VARCHAR(50), -- statute, precedent, brief, motion
  confidence DECIMAL(3,2) DEFAULT 0.85,
  case_id UUID REFERENCES cases(id) ON DELETE CASCADE,
  evidence_id UUID REFERENCES evidence(id) ON DELETE CASCADE,
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);`,
      'precedents': `
CREATE TABLE legal_precedents (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  case_title VARCHAR(255) NOT NULL,
  citation VARCHAR(255) NOT NULL,
  court VARCHAR(100),
  year INTEGER,
  jurisdiction VARCHAR(50),
  summary TEXT,
  full_text TEXT,
  embedding TEXT, -- Legal document vector
  relevance_score DECIMAL(3,2),
  legal_principles JSONB DEFAULT '[]',
  linked_cases JSONB DEFAULT '[]',
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);`,
      'analysis-sessions': `
CREATE TABLE legal_analysis_sessions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  case_id UUID REFERENCES cases(id) ON DELETE CASCADE,
  user_id UUID REFERENCES users(id) ON DELETE CASCADE,
  session_type VARCHAR(50) DEFAULT 'case_analysis',
  analysis_prompt TEXT,
  analysis_result TEXT,
  confidence_level DECIMAL(3,2),
  sources_used JSONB DEFAULT '[]',
  model VARCHAR(100) DEFAULT 'gemma3-legal',
  processing_time INTEGER, -- milliseconds
  is_active BOOLEAN DEFAULT true,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);`
    };

    let schemaText = '';
    if (tableType === 'all') {
      schemaText = Object.values(schemas).join('\n\n');
    } else {
      schemaText = schemas[tableType] || 'Unknown table type';
    }

    return {
      content: [{
        type: 'text',
        text: `# Legal Database Schema: ${tableType.toUpperCase()}\n\n\`\`\`sql\n${schemaText}\n\`\`\``
      }]
    };
  }

  async getApiEndpoints(category = 'all') {
    const endpoints = {
      'legal-chat': `
## Legal Chat API (/api/legal/chat)

**POST** - Perform legal analysis
- Request: { prompt, caseId?, userId, sessionType?, context? }
- Response: { sessionId, analysis, confidence, sources, recommendations, processingTime }

**GET** - Get analysis history
- Query params: caseId, userId, sessionType, limit
- Response: Array of analysis sessions`,

      'precedent-search': `
## Precedent Search API (/api/legal/precedents)

**GET** - Search legal precedents
- Query params: query, jurisdiction?, court?, yearFrom?, yearTo?, limit?, offset?
- Response: { precedents, totalCount, searchTerms, processingTime }

**POST** - Add new precedent
- Request: { caseTitle, citation, court?, year?, jurisdiction?, summary?, ... }
- Response: Created precedent object

**PUT** - Find similar precedents
- Request: { precedentId, queryVector }
- Response: { basePrecedent, similarPrecedents, similarityMethod, count }`,

      'document-analysis': `
## Document Analysis API (/api/legal/documents)

**GET** - List legal documents
- Query params: caseId?, type?, status?, search?, limit?, offset?
- Response: Array of legal documents with word counts

**POST** - Create legal document
- Request: { title, content, documentType, status?, caseId?, metadata? }
- Response: Created document object

**PUT** - Update document
- Request: { title?, content?, documentType?, status?, metadata? }
- Response: Updated document object

**DELETE** - Delete document
- Response: { success: true }`
    };

    let endpointText = '';
    if (category === 'all') {
      endpointText = Object.values(endpoints).join('\n\n');
    } else {
      endpointText = endpoints[category] || 'Unknown endpoint category';
    }

    return {
      content: [{
        type: 'text',
        text: `# Legal AI API Endpoints: ${category.toUpperCase()}\n\n${endpointText}`
      }]
    };
  }

  async getGemma3Config() {
    const config = `
# Gemma3 Legal Model Configuration

## Model Details
- **File**: mohf16-Q4_K_M.gguf
- **Location**: C:\\Users\\james\\Desktop\\deeds-web\\deeds-web-app\\gemma3Q4_K_M\\
- **Quantization**: Q4_K_M (4-bit quantization with high quality)
- **Context Length**: 8192 tokens
- **Specialized For**: Legal document analysis and case law research

## Ollama Integration
- **Model Name**: gemma3-legal
- **Modelfile**: ./Modelfile-gemma3-legal
- **Template Format**: Gemma instruction format
- **System Prompt**: Specialized for legal analysis and prosecution support

## Usage Examples
\`\`\`bash
# Load model in Ollama
ollama create gemma3-legal -f ./Modelfile-gemma3-legal

# Test model
ollama run gemma3-legal "Analyze the legal implications of..."

# API Usage
curl -X POST http://localhost:11434/api/generate \\
  -d '{"model": "gemma3-legal", "prompt": "Legal query here"}'
\`\`\`

## Integration Points
- **Legal Chat API**: /api/legal/chat
- **Document Analysis**: Automatic legal document processing
- **Precedent Matching**: Vector similarity search with legal embeddings
- **Case Analysis**: Comprehensive case review and recommendations
`;

    return {
      content: [{
        type: 'text',
        text: config
      }]
    };
  }

  // RAG System Integration Methods
  async ragQuery(args) {
    const { query, maxResults = 5, confidenceThreshold = 0.7, caseId, documentTypes } = args;
    
    if (!RAG_CONFIG.enabled) {
      return {
        content: [{
          type: 'text',
          text: 'RAG system is not enabled. Please check RAG_ENABLED environment variable.'
        }]
      };
    }

    try {
      // Call RAG backend API
      const response = await fetch(`${RAG_CONFIG.endpoint}/api/v1/rag/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          max_results: maxResults,
          confidence_threshold: confidenceThreshold,
          case_id: caseId,
          document_types: documentTypes
        })
      });

      if (!response.ok) {
        throw new Error(`RAG API error: ${response.status}`);
      }

      const result = await response.json();
      
      const formattedResult = `# RAG Query Results

**Query**: ${query}
**Confidence Score**: ${result.confidence_score}
**Processing Time**: ${result.processing_time_ms}ms
**Sources Found**: ${result.sources.length}

## Response
${result.response}

## Sources
${result.sources.map((source, i) => `
### ${i + 1}. ${source.title} (${source.document_type})
- **Similarity**: ${(source.similarity_score * 100).toFixed(1)}%
- **Excerpt**: ${source.excerpt}
`).join('')}`;

      return {
        content: [{
          type: 'text',
          text: formattedResult
        }]
      };

    } catch (error) {
      return {
        content: [{
          type: 'text',
          text: `RAG Query Failed: ${error.message}\n\nEnsure RAG backend is running at ${RAG_CONFIG.endpoint}`
        }]
      };
    }
  }

  async ragUploadDocument(args) {
    const { filePath, caseId, documentType = 'general', title } = args;

    if (!RAG_CONFIG.enabled) {
      return {
        content: [{
          type: 'text',
          text: 'RAG system is not enabled. Please check RAG_ENABLED environment variable.'
        }]
      };
    }

    try {
      // Read file content
      const fileContent = await fs.readFile(filePath);
      const fileName = path.basename(filePath);
      
      // Create form data
      const formData = new FormData();
      formData.append('file', new Blob([fileContent]), fileName);
      if (caseId) formData.append('case_id', caseId);
      formData.append('document_type', documentType);

      const response = await fetch(`${RAG_CONFIG.endpoint}/api/v1/documents/upload`, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error(`Upload failed: ${response.status}`);
      }

      const result = await response.json();

      return {
        content: [{
          type: 'text',
          text: `# Document Upload Success

**Document ID**: ${result.document_id}
**File**: ${fileName}
**Status**: ${result.status}
**Message**: ${result.message}
**Processing Job ID**: ${result.processing_job_id || 'N/A'}

The document has been uploaded and indexed in the RAG system. You can now query it using the rag-query tool.`
        }]
      };

    } catch (error) {
      return {
        content: [{
          type: 'text',
          text: `Document Upload Failed: ${error.message}\n\nEnsure RAG backend is running and file exists at: ${filePath}`
        }]
      };
    }
  }

  async ragGetStats() {
    if (!RAG_CONFIG.enabled) {
      return {
        content: [{
          type: 'text',
          text: 'RAG system is not enabled. Please check RAG_ENABLED environment variable.'
        }]
      };
    }

    try {
      const response = await fetch(`${RAG_CONFIG.endpoint}/health`);
      const health = await response.json();

      return {
        content: [{
          type: 'text',
          text: `# RAG System Statistics

## Health Status
- **Overall Status**: ${health.status}
- **Timestamp**: ${health.timestamp}

## Services Status
${Object.entries(health.services).map(([service, status]) => 
  `- **${service}**: ${status}`
).join('\n')}

## Model Status
- **Loaded**: ${health.model_status.loaded ? 'Yes' : 'No'}
- **Model Info**: ${JSON.stringify(health.model_status, null, 2)}

## Configuration
- **RAG Endpoint**: ${RAG_CONFIG.endpoint}
- **Vector Store**: ${RAG_CONFIG.vectorStore}
- **Embedding Model**: ${RAG_CONFIG.embeddingModel}
- **Query Threshold**: ${RAG_CONFIG.queryThreshold}

## Integration Points
- Connect to SvelteKit via \`/api/legal/rag\` endpoints
- Real-time updates via WebSocket at \`/ws/events\`
- Document processing via \`/api/v1/documents/upload\`
`
        }]
      };

    } catch (error) {
      return {
        content: [{
          type: 'text',
          text: `RAG Stats Failed: ${error.message}\n\nEnsure RAG backend is running at ${RAG_CONFIG.endpoint}`
        }]
      };
    }
  }

  async ragAnalyzeRelevance(args) {
    const { query, documentId } = args;

    if (!RAG_CONFIG.enabled) {
      return {
        content: [{
          type: 'text',
          text: 'RAG system is not enabled. Please check RAG_ENABLED environment variable.'
        }]
      };
    }

    try {
      // This would need to be implemented in the RAG backend
      return {
        content: [{
          type: 'text',
          text: `# Document Relevance Analysis

**Query**: ${query}
**Document ID**: ${documentId}

This feature analyzes how relevant a specific document is to your query using semantic similarity and legal keyword matching.

**Note**: Document relevance analysis endpoint needs to be implemented in RAG backend.
Suggested endpoint: \`GET /api/v1/documents/{documentId}/relevance?query={query}\`

**Implementation**: Add this method to RAGService.analyze_document_relevance() in rag-backend/services/rag_service.py`
        }]
      };

    } catch (error) {
      return {
        content: [{
          type: 'text',
          text: `Relevance Analysis Failed: ${error.message}`
        }]
      };
    }
  }

  async ragIntegrationGuide(args) {
    const { integrationType } = args;

    const guides = {
      'api-integration': `# RAG API Integration with SvelteKit

## Create RAG Service
\`\`\`typescript
// src/lib/services/rag-service.ts
export class RAGService {
  private baseUrl = 'http://localhost:8000/api/v1';

  async query(query: string, options?: {
    maxResults?: number;
    confidenceThreshold?: number;
    caseId?: string;
    documentTypes?: string[];
  }) {
    const response = await fetch(\`\${this.baseUrl}/rag/query\`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, ...options })
    });
    return response.json();
  }

  async uploadDocument(file: File, caseId?: string, documentType = 'general') {
    const formData = new FormData();
    formData.append('file', file);
    if (caseId) formData.append('case_id', caseId);
    formData.append('document_type', documentType);

    const response = await fetch(\`\${this.baseUrl}/documents/upload\`, {
      method: 'POST',
      body: formData
    });
    return response.json();
  }
}
\`\`\`

## SvelteKit API Route
\`\`\`typescript
// src/routes/api/legal/rag/+server.ts
import { RAGService } from '$lib/services/rag-service';
import { json } from '@sveltejs/kit';

const ragService = new RAGService();

export async function POST({ request }) {
  const { query, options } = await request.json();
  const result = await ragService.query(query, options);
  return json(result);
}
\`\`\``,

      'component-integration': `# RAG Search Component

\`\`\`svelte
<!-- src/lib/components/legal/RAGSearch.svelte -->
<script lang="ts">
  import { RAGService } from '$lib/services/rag-service';
  import { writable } from 'svelte/store';

  export let caseId: string | undefined = undefined;
  
  let query = '';
  let results = writable([]);
  let loading = false;
  let confidence = 0.7;

  const ragService = new RAGService();

  async function performSearch() {
    if (!query.trim()) return;
    
    loading = true;
    try {
      const response = await ragService.query(query, {
        confidenceThreshold: confidence,
        caseId,
        maxResults: 10
      });
      
      results.set(response.sources || []);
    } catch (error) {
      console.error('RAG search failed:', error);
    } finally {
      loading = false;
    }
  }
</script>

<div class="rag-search">
  <div class="search-form">
    <input 
      bind:value={query}
      placeholder="Search legal documents..."
      on:keydown={(e) => e.key === 'Enter' && performSearch()}
    />
    <button on:click={performSearch} disabled={loading}>
      {loading ? 'Searching...' : 'Search'}
    </button>
  </div>

  <div class="confidence-slider">
    <label>Confidence Threshold: {confidence}</label>
    <input type="range" bind:value={confidence} min="0.1" max="1" step="0.1" />
  </div>

  {#if $results.length > 0}
    <div class="results">
      {#each $results as result}
        <div class="result-item">
          <h3>{result.title}</h3>
          <p class="similarity">Similarity: {(result.similarity_score * 100).toFixed(1)}%</p>
          <p class="excerpt">{result.excerpt}</p>
        </div>
      {/each}
    </div>
  {/if}
</div>
\`\`\``,

      'search-ui': `# Advanced RAG Search UI

## Features
- Real-time search suggestions
- Filter by document type and case
- Relevance scoring visualization
- Export search results

## Implementation
\`\`\`svelte
<!-- src/lib/components/legal/AdvancedRAGSearch.svelte -->
<script lang="ts">
  import { debounce } from '$lib/utils/debounce';
  import { RAGService } from '$lib/services/rag-service';
  
  export let caseId: string | undefined = undefined;
  
  let query = '';
  let filters = {
    documentTypes: [],
    confidenceThreshold: 0.7,
    maxResults: 20
  };
  
  // Debounced search
  const performSearch = debounce(async () => {
    // Search implementation
  }, 300);
  
  // Real-time suggestions
  $: if (query.length > 2) {
    performSearch();
  }
</script>

<div class="advanced-search">
  <!-- Search input with autocomplete -->
  <!-- Filter controls -->
  <!-- Results with relevance visualization -->
  <!-- Export functionality -->
</div>
\`\`\``,

      'document-upload': `# Document Upload Integration

## Upload Component
\`\`\`svelte
<!-- src/lib/components/legal/DocumentUpload.svelte -->
<script lang="ts">
  import { RAGService } from '$lib/services/rag-service';
  import { createEventDispatcher } from 'svelte';
  
  export let caseId: string;
  export let allowedTypes = ['pdf', 'docx', 'txt'];
  
  const dispatch = createEventDispatcher();
  const ragService = new RAGService();
  
  let files: FileList;
  let uploading = false;
  let uploadProgress = 0;

  async function handleUpload() {
    if (!files?.length) return;
    
    uploading = true;
    uploadProgress = 0;
    
    try {
      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        
        await ragService.uploadDocument(file, caseId, 'evidence');
        uploadProgress = ((i + 1) / files.length) * 100;
        
        dispatch('uploaded', { 
          file: file.name, 
          caseId 
        });
      }
    } catch (error) {
      dispatch('error', { error: error.message });
    } finally {
      uploading = false;
      uploadProgress = 0;
    }
  }
</script>

<div class="document-upload">
  <input 
    type="file" 
    bind:files 
    multiple 
    accept=".pdf,.docx,.txt"
    disabled={uploading}
  />
  
  {#if uploading}
    <div class="progress">
      <div class="progress-bar" style="width: {uploadProgress}%"></div>
    </div>
  {/if}
  
  <button on:click={handleUpload} disabled={!files?.length || uploading}>
    {uploading ? 'Uploading...' : 'Upload Documents'}
  </button>
</div>
\`\`\`

## Processing Status
Monitor document processing status:
\`\`\`typescript
// Check processing job status
async function checkProcessingStatus(jobId: string) {
  const response = await fetch(\`/api/v1/jobs/\${jobId}\`);
  return response.json();
}
\`\`\``
    };

    const guide = guides[integrationType] || 'Unknown integration type';

    return {
      content: [{
        type: 'text',
        text: guide
      }]
    };
  }

  async run() {
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    console.error('Context7 Legal MCP Server running on stdio');
  }
}

const server = new Context7MCPServer();
server.run().catch(console.error);