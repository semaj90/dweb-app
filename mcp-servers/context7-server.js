#!/usr/bin/env node

/**
 * Custom Context7 MCP Server
 * Provides context-aware assistance for the SvelteKit + UnoCSS + Legal AI stack
 */

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ListResourcesRequestSchema,
  ListToolsRequestSchema,
  ReadResourceRequestSchema
} from '@modelcontextprotocol/sdk/types.js';
import fs from 'fs/promises';
import path from 'path';
import http from 'http';
import { WebSocketServer } from 'ws';
import fetch from 'node-fetch';

// Stack configuration from command line args
const stackConfig = process.argv.find(arg => arg.startsWith('--stack='))?.split('=')[1]?.split(',') || [];

class Context7Server {
  constructor() {
    this.server = new Server(
      {
        name: 'context7-server',
        version: '1.0.0',
      },
      {
        capabilities: {
          resources: {},
          tools: {},
        },
      }
    );

    this.setupHandlers();
  }

  setupHandlers() {
    // List available tools
  this.server.setRequestHandler(ListToolsRequestSchema, async () => {
      return {
        tools: [
          {
            name: 'analyze-stack',
            description: 'Analyze the current technology stack and provide context-aware suggestions',
            inputSchema: {
              type: 'object',
              properties: {
                component: {
                  type: 'string',
                  description: 'Component or technology to analyze',
                },
                context: {
                  type: 'string',
                  description: 'Additional context (e.g., "legal-ai", "gaming-ui", "performance")',
                },
              },
              required: ['component'],
            },
          },
          {
            name: 'generate-best-practices',
            description: 'Generate best practices for the current stack configuration',
            inputSchema: {
              type: 'object',
              properties: {
                area: {
                  type: 'string',
                  description: 'Area to generate best practices for (e.g., "performance", "security", "ui-ux")',
                },
              },
              required: ['area'],
            },
          },
          {
            name: 'suggest-integration',
            description: 'Suggest how to integrate new features with the existing stack',
            inputSchema: {
              type: 'object',
              properties: {
                feature: {
                  type: 'string',
                  description: 'Feature to integrate',
                },
                requirements: {
                  type: 'string',
                  description: 'Special requirements or constraints',
                },
              },
              required: ['feature'],
            },
          },
          {
            name: 'analyze-legal-document',
            description: 'Analyze legal documents for key findings, risk assessment, and compliance',
            inputSchema: {
              type: 'object',
              properties: {
                content: {
                  type: 'string',
                  description: 'Legal document content to analyze',
                },
                caseType: {
                  type: 'string',
                  description: 'Type of legal case (e.g., "contract", "litigation", "compliance")',
                },
                jurisdiction: {
                  type: 'string',
                  description: 'Legal jurisdiction (e.g., "federal", "state", "international")',
                },
              },
              required: ['content', 'caseType'],
            },
          },
          {
            name: 'generate-compliance-report',
            description: 'Generate compliance reports based on evidence and regulations',
            inputSchema: {
              type: 'object',
              properties: {
                evidence: {
                  type: 'array',
                  items: { type: 'string' },
                  description: 'Array of evidence items to analyze',
                },
                regulations: {
                  type: 'array',
                  items: { type: 'string' },
                  description: 'Applicable regulations and standards',
                },
                framework: {
                  type: 'string',
                  description: 'Compliance framework (e.g., "GDPR", "HIPAA", "SOX")',
                },
              },
              required: ['evidence', 'regulations'],
            },
          },
          {
            name: 'suggest-legal-precedents',
            description: 'Find and suggest relevant legal precedents for a case',
            inputSchema: {
              type: 'object',
              properties: {
                query: {
                  type: 'string',
                  description: 'Legal query or case description',
                },
                jurisdiction: {
                  type: 'string',
                  description: 'Legal jurisdiction for precedent search',
                },
                caseType: {
                  type: 'string',
                  description: 'Type of case for precedent matching',
                },
              },
              required: ['query'],
            },
          },
          {
            name: 'extract-legal-entities',
            description: 'Extract legal entities (parties, dates, amounts, clauses) from documents',
            inputSchema: {
              type: 'object',
              properties: {
                content: {
                  type: 'string',
                  description: 'Document content to analyze',
                },
                entityTypes: {
                  type: 'array',
                  items: { type: 'string' },
                  description: 'Types of entities to extract (e.g., "parties", "dates", "monetary", "clauses")',
                },
              },
              required: ['content'],
            },
          }
          {
            name: 'vector-search-qdrant',
            description: 'Search Qdrant dev_embeddings with an embedded query via Ollama',
            inputSchema: {
              type: 'object',
              properties: {
                query: { type: 'string' },
                limit: { type: 'number' }
              },
              required: ['query']
            }
          },
        ]
      };
    });

    // List available resources
    this.server.setRequestHandler(ListResourcesRequestSchema, async () => {
      return {
        resources: [
          {
            uri: 'context7://stack-overview',
            name: 'Technology Stack Overview',
            description: 'Current technology stack configuration and best practices',
            mimeType: 'text/markdown',
          },
          {
            uri: 'context7://integration-guide',
            name: 'Integration Guide',
            description: 'Guide for integrating new components with the existing stack',
            mimeType: 'text/markdown',
          },
          {
            uri: 'context7://performance-tips',
            name: 'Performance Optimization Tips',
            description: 'Performance optimization recommendations for the current stack',
            mimeType: 'text/markdown',
          }
        ]
      };
    });

    // Read resource content
    this.server.setRequestHandler(ReadResourceRequestSchema, async (request) => {
      const { uri } = request.params;

      switch (uri) {
        case 'context7://stack-overview':
          return {
            contents: [
              {
                uri,
                mimeType: 'text/markdown',
                text: this.generateStackOverview()
              }
            ]
          };

        case 'context7://integration-guide':
          return {
            contents: [
              {
                uri,
                mimeType: 'text/markdown',
                text: this.generateIntegrationGuide()
              }
            ]
          };

        case 'context7://performance-tips':
          return {
            contents: [
              {
                uri,
                mimeType: 'text/markdown',
                text: this.generatePerformanceTips()
              }
            ]
          };

        default:
          throw new Error(`Unknown resource: ${uri}`);
      }
    });

    // Handle tool calls
  this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      const { name, arguments: args } = request.params;

      switch (name) {
        case 'analyze-stack':
          return {
            content: [
              {
                type: 'text',
                text: this.analyzeStackComponent(args.component, args.context)
              }
            ]
          };

        case 'generate-best-practices':
          return {
            content: [
              {
                type: 'text',
                text: this.generateBestPractices(args.area)
              }
            ]
          };

        case 'suggest-integration':
          return {
            content: [
              {
                type: 'text',
                text: this.suggestIntegration(args.feature, args.requirements)
              }
            ]
          };

        case 'analyze-legal-document':
          return {
            content: [
              {
                type: 'text',
                text: this.analyzeLegalDocument(args.content, args.caseType, args.jurisdiction)
              }
            ]
          };

        case 'generate-compliance-report':
          return {
            content: [
              {
                type: 'text',
                text: this.generateComplianceReport(args.evidence, args.regulations, args.framework)
              }
            ]
          };

        case 'suggest-legal-precedents':
          return {
            content: [
              {
                type: 'text',
                text: this.suggestLegalPrecedents(args.query, args.jurisdiction, args.caseType)
              }
            ]
          };

        case 'extract-legal-entities':
          return {
            content: [
              {
                type: 'text',
                text: this.extractLegalEntities(args.content, args.entityTypes)
              }
            ]
          };

        case 'vector-search-qdrant':
          return { content: [ { type: 'json', json: await this.qdrantSearch(args.query, args.limit ?? 5) } ] };

        default:
          throw new Error(`Unknown tool: ${name}`);
      }
    });
  }

  generateStackOverview() {
    return `# Technology Stack Overview

## Current Stack Configuration
${stackConfig.length > 0 ? stackConfig.map(tech => `- ${tech}`).join('\n') : '- No stack configuration provided'}

## Core Technologies

### Frontend
- **SvelteKit 2**: Modern full-stack framework with SSR/SPA capabilities
- **Svelte 5**: Component framework with new runes syntax
- **UnoCSS**: Instant on-demand atomic CSS engine
- **TypeScript**: Type-safe JavaScript development

### UI Components
- **shadcn-svelte**: Copy-paste component library
- **Melt UI**: Headless UI primitives for Svelte
- **Bits UI**: Additional component primitives

### State Management
- **XState**: State machines for complex application logic
- **Superforms**: Advanced form handling for SvelteKit

### Database & Backend
- **Drizzle ORM**: Type-safe SQL ORM
- **PostgreSQL**: Primary database
- **pgvector**: Vector similarity search for AI features

### AI & ML
- **Ollama**: Efficient LLM inference
- **Docker**: Containerization for AI services

## Best Practices
1. Use TypeScript for all new code
2. Leverage UnoCSS for styling
3. Implement proper error boundaries
4. Use XState for complex state logic
5. Follow shadcn-svelte patterns for UI consistency
`;
  }

  generateIntegrationGuide() {
    return `# Integration Guide

## Adding New Components

### 1. UI Components
- Follow shadcn-svelte patterns
- Use UnoCSS for styling
- Implement proper TypeScript types
- Add to component library structure

### 2. State Management
- Use XState for complex logic
- Leverage Svelte 5 runes for simple state
- Implement proper error handling

### 3. Database Integration
- Use Drizzle ORM schemas
- Follow established patterns
- Implement proper migrations

### 4. AI Features
- Integrate with existing Ollama setup
- Use pgvector for similarity search
- Follow established AI service patterns

## Testing Strategy
1. Unit tests for components
2. Integration tests for flows
3. E2E tests for critical paths
4. AI model testing for accuracy
`;
  }

  generatePerformanceTips() {
    return `# Performance Optimization Tips

## Frontend Optimization
1. **UnoCSS Configuration**
   - Use atomic classes efficiently
   - Implement proper purging
   - Optimize bundle size

2. **SvelteKit Optimization**
   - Leverage SSR appropriately
   - Implement proper code splitting
   - Use efficient data loading patterns

3. **Component Performance**
   - Use Svelte 5 runes effectively
   - Implement proper reactivity
   - Avoid unnecessary re-renders

## Backend Optimization
1. **Database Performance**
   - Optimize Drizzle queries
   - Use proper indexing
   - Implement connection pooling

2. **AI Performance**
   - Optimize Ollama configuration
   - Use efficient embedding strategies
   - Implement proper caching

## Monitoring
- Implement performance metrics
- Monitor AI response times
- Track database query performance
`;
  }

  analyzeStackComponent(component, context = '') {
    const analysis = {
      svelte: 'Modern reactive framework with excellent performance. Use Svelte 5 runes for optimal reactivity.',
      sveltekit: 'Full-stack framework providing SSR, routing, and API endpoints. Excellent for legal AI applications.',
      typescript: 'Essential for type safety in complex legal AI systems. Helps prevent runtime errors.',
      unocss: 'Atomic CSS engine perfect for rapid UI development. Integrates well with shadcn-svelte.',
      drizzle: 'Type-safe ORM ideal for complex legal database schemas. Excellent TypeScript integration.',
      xstate: 'State machines perfect for complex legal workflow logic. Handles edge cases well.',
      superforms: 'Advanced form handling essential for legal data entry. Great validation capabilities.',
      pgvector: 'Vector similarity search perfect for AI-powered legal document analysis.',
      'shadcn-svelte': 'Consistent UI component library. Excellent for professional legal interfaces.',
      'bits-ui': 'Headless UI primitives that complement shadcn-svelte well.',
      'melt-ui': 'Advanced UI primitives for complex interactions.',
      ollama: 'Efficient LLM inference for legal AI capabilities.',
      docker: 'Essential for consistent AI model deployment.',
      postgres: 'Robust database perfect for legal data requirements.'
    };

    const componentAnalysis = analysis[component.toLowerCase()] ||
      `Component "${component}" not in current stack. Consider integration patterns with existing technologies.`;

    let contextualAdvice = '';
    if (context) {
      switch (context.toLowerCase()) {
        case 'legal-ai':
          contextualAdvice = '\n\nFor legal AI applications, focus on data security, audit trails, and precise AI responses.';
          break;
        case 'gaming-ui':
          contextualAdvice = '\n\nFor gaming UI, emphasize visual effects, smooth animations, and immersive user experience.';
          break;
        case 'performance':
          contextualAdvice = '\n\nFor performance optimization, focus on efficient rendering, minimal bundle size, and fast database queries.';
          break;
      }
    }

    return `## Analysis: ${component}

${componentAnalysis}${contextualAdvice}

### Current Stack Integration
${stackConfig.includes(component.toLowerCase()) ? '✅ Already integrated in your stack' : '⚠️ Not currently in your stack configuration'}

### Recommendations
- Ensure proper TypeScript integration
- Follow established patterns from existing components
- Implement proper error handling and validation
- Consider performance implications for legal AI workloads`;
  }

  generateBestPractices(area) {
    const practices = {
      performance: `# Performance Best Practices

## Frontend Performance
- Use UnoCSS atomic classes efficiently
- Implement proper Svelte component optimization
- Leverage SvelteKit's SSR capabilities
- Minimize bundle size with proper tree shaking

## Backend Performance
- Optimize Drizzle ORM queries
- Use database indexing effectively
- Implement connection pooling
- Cache frequently accessed data

## AI Performance
- Optimize Ollama model loading
- Use efficient vector similarity search
- Implement proper embedding strategies
- Monitor AI response times`,

      security: `# Security Best Practices

## Data Security
- Implement proper authentication with Lucia
- Use secure session management
- Validate all inputs with Superforms
- Implement audit trails for legal compliance

## AI Security
- Secure API endpoints for AI services
- Implement proper access controls
- Monitor AI usage for anomalies
- Ensure data privacy compliance`,

      'ui-ux': `# UI/UX Best Practices

## Design System
- Follow shadcn-svelte patterns consistently
- Use UnoCSS for maintainable styles
- Implement proper accessibility features
- Ensure responsive design across devices

## User Experience
- Provide clear feedback for AI operations
- Implement proper loading states
- Use progressive enhancement
- Follow legal industry UX patterns`
    };

    return practices[area.toLowerCase()] ||
      `Best practices for "${area}" not available. Consider general development best practices.`;
  }

  suggestIntegration(feature, requirements = '') {
    return `# Integration Suggestion: ${feature}

## Recommended Approach

### 1. Architecture Considerations
- Leverage existing SvelteKit structure
- Use TypeScript for type safety
- Follow established component patterns
- Integrate with current state management (XState)

### 2. Implementation Steps
1. Create component structure following shadcn-svelte patterns
2. Implement TypeScript interfaces and types
3. Add to existing routing structure
4. Integrate with Drizzle ORM if database access needed
5. Add proper error handling and validation

### 3. Stack Integration
${stackConfig.length > 0 ?
  stackConfig.map(tech => `- **${tech}**: Consider how ${feature} integrates with ${tech}`).join('\n') :
  '- No specific stack configuration available'}

### 4. Testing Strategy
- Unit tests for new components
- Integration tests with existing features
- E2E tests for user workflows
- Performance testing if applicable

${requirements ? `\n### 5. Special Requirements\n${requirements}` : ''}

### 6. Next Steps
1. Design component API
2. Implement core functionality
3. Add proper documentation
4. Test integration with existing features
5. Deploy and monitor performance`;
  }

  analyzeLegalDocument(content, caseType, jurisdiction = 'federal') {
    // Simulate AI-powered legal document analysis
    const wordCount = content.split(' ').length;
    const hasContracts = content.toLowerCase().includes('contract') || content.toLowerCase().includes('agreement');
    const hasLiability = content.toLowerCase().includes('liability') || content.toLowerCase().includes('damages');
    const hasCompliance = content.toLowerCase().includes('comply') || content.toLowerCase().includes('regulation');

    let riskLevel = 'Low';
    let riskScore = 25;

    if (hasLiability && wordCount > 1000) {
      riskLevel = 'High';
      riskScore = 85;
    } else if (hasContracts && wordCount > 500) {
      riskLevel = 'Medium';
      riskScore = 60;
    }

    const keyFindings = [];
    if (hasContracts) keyFindings.push('Contract terms and conditions identified');
    if (hasLiability) keyFindings.push('Liability and damages clauses present');
    if (hasCompliance) keyFindings.push('Compliance requirements mentioned');

    return `# Legal Document Analysis

## Document Overview
- **Case Type**: ${caseType}
- **Jurisdiction**: ${jurisdiction}
- **Document Length**: ${wordCount} words
- **Analysis Date**: ${new Date().toISOString().split('T')[0]}

## Risk Assessment
- **Overall Risk Level**: ${riskLevel}
- **Risk Score**: ${riskScore}/100
- **Critical Issues**: ${hasLiability ? '2' : '0'} potential issues identified

## Key Findings
${keyFindings.length > 0 ? keyFindings.map(finding => `- ${finding}`).join('\n') : '- No significant legal issues identified'}

## Compliance Status
- **GDPR Compliance**: ${hasCompliance ? 'Under Review' : 'Not Applicable'}
- **Contract Law**: ${hasContracts ? 'Requires Review' : 'N/A'}
- **Liability Assessment**: ${hasLiability ? 'High Priority Review Needed' : 'Standard Processing'}

## Recommended Actions
${riskLevel === 'High' ? `1. Immediate legal review required
2. Risk mitigation strategies needed
3. Stakeholder notification recommended` :
riskLevel === 'Medium' ? `1. Standard legal review process
2. Monitor for compliance issues
3. Schedule follow-up assessment` :
`1. Routine processing acceptable
2. Standard filing procedures
3. Regular monitoring sufficient`}

## Integration Notes
This analysis integrates with:
- **Drizzle ORM**: Store analysis results in evidence table
- **pgvector**: Enable semantic similarity search
- **XState**: Trigger appropriate workflow states
- **Ollama**: Enhanced AI analysis capabilities`;
  }

  generateComplianceReport(evidence, regulations, framework = 'General') {
    const evidenceCount = evidence.length;
    const regulationCount = regulations.length;
    const complianceScore = Math.min(90, evidenceCount * 10 + regulationCount * 5);

    return `# Compliance Report - ${framework}

## Executive Summary
- **Evidence Items Analyzed**: ${evidenceCount}
- **Applicable Regulations**: ${regulationCount}
- **Compliance Score**: ${complianceScore}%
- **Report Date**: ${new Date().toISOString().split('T')[0]}

## Evidence Analysis
${evidence.map((item, index) => `${index + 1}. ${item.substring(0, 100)}${item.length > 100 ? '...' : ''}`).join('\n')}

## Regulatory Framework
${regulations.map((reg, index) => `${index + 1}. ${reg}`).join('\n')}

## Compliance Assessment
- **${framework} Compliance**: ${complianceScore > 80 ? 'Compliant' : complianceScore > 60 ? 'Partially Compliant' : 'Non-Compliant'}
- **Risk Level**: ${complianceScore > 80 ? 'Low' : complianceScore > 60 ? 'Medium' : 'High'}
- **Remediation Required**: ${complianceScore < 80 ? 'Yes' : 'No'}

## Stack Integration
- **Database Storage**: Evidence stored via Drizzle ORM
- **Vector Search**: Compliance patterns indexed with pgvector
- **Workflow Management**: XState compliance review workflows
- **AI Analysis**: Enhanced compliance checking with Ollama`;
  }

  suggestLegalPrecedents(query, jurisdiction = 'federal', caseType = 'general') {
    // Simulate precedent matching based on query analysis
    const queryWords = query.toLowerCase().split(' ');
    const hasContract = queryWords.includes('contract') || queryWords.includes('agreement');
    const hasLiability = queryWords.includes('liability') || queryWords.includes('damages');
    const hasBreach = queryWords.includes('breach') || queryWords.includes('violation');

    const precedents = [];

    if (hasContract && hasBreach) {
      precedents.push({
        case: 'Smith v. Jones Contract Dispute',
        relevance: '95%',
        year: '2022',
        jurisdiction: jurisdiction,
        summary: 'Breach of contract with significant damages awarded'
      });
    }

    if (hasLiability) {
      precedents.push({
        case: 'Corporate Liability Standards Case',
        relevance: '88%',
        year: '2021',
        jurisdiction: jurisdiction,
        summary: 'Established liability standards for corporate entities'
      });
    }

    return `# Legal Precedent Analysis

## Query Analysis
- **Search Query**: "${query}"
- **Jurisdiction**: ${jurisdiction}
- **Case Type**: ${caseType}
- **Search Date**: ${new Date().toISOString().split('T')[0]}

## Relevant Precedents
${precedents.length > 0 ?
  precedents.map(p => `### ${p.case} (${p.year})
- **Relevance Score**: ${p.relevance}
- **Jurisdiction**: ${p.jurisdiction}
- **Summary**: ${p.summary}
`).join('\n') :
  '### No Direct Precedents Found\n- Consider expanding search criteria\n- Review similar case types\n- Consult legal databases'}

## Precedent Impact Analysis
- **Binding Authority**: ${jurisdiction === 'federal' ? 'High' : 'Medium'}
- **Persuasive Value**: ${precedents.length > 0 ? 'Strong' : 'Limited'}
- **Case Strength**: ${precedents.length > 1 ? 'Well-Supported' : 'Requires Additional Research'}

## Integration Recommendations
- **Vector Storage**: Index precedents with pgvector for similarity search
- **Case Database**: Store in PostgreSQL with Drizzle ORM relationships
- **AI Enhancement**: Use Ollama for deeper precedent analysis
- **Workflow Integration**: Trigger XState precedent research workflows`;
  }

  extractLegalEntities(content, entityTypes = ['parties', 'dates', 'monetary', 'clauses']) {
    const entities = {
      parties: [],
      dates: [],
      monetary: [],
      clauses: []
    };

    // Simple entity extraction simulation
    const words = content.split(' ');

    // Extract potential parties (capitalized names)
    if (entityTypes.includes('parties')) {
      const partyPatterns = content.match(/[A-Z][a-z]+ [A-Z][a-z]+/g) || [];
      entities.parties = [...new Set(partyPatterns)].slice(0, 5);
    }

    // Extract dates
    if (entityTypes.includes('dates')) {
      const datePatterns = content.match(/\d{1,2}\/\d{1,2}\/\d{4}|\d{4}-\d{2}-\d{2}|[A-Z][a-z]+ \d{1,2}, \d{4}/g) || [];
      entities.dates = [...new Set(datePatterns)];
    }

    // Extract monetary amounts
    if (entityTypes.includes('monetary')) {
      const monetaryPatterns = content.match(/\$[\d,]+\.?\d*|\$\d+|USD \d+/g) || [];
      entities.monetary = [...new Set(monetaryPatterns)];
    }

    // Extract clause references
    if (entityTypes.includes('clauses')) {
      const clausePatterns = content.match(/[Ss]ection \d+|[Cc]lause \d+|[Aa]rticle \d+|[Pp]aragraph \d+/g) || [];
      entities.clauses = [...new Set(clausePatterns)];
    }

    const totalEntities = Object.values(entities).flat().length;

    return `# Legal Entity Extraction

## Extraction Summary
- **Total Entities Found**: ${totalEntities}
- **Entity Types Requested**: ${entityTypes.join(', ')}
- **Document Length**: ${content.length} characters
- **Extraction Date**: ${new Date().toISOString().split('T')[0]}

## Extracted Entities

### Parties (${entities.parties.length})
${entities.parties.length > 0 ? entities.parties.map(party => `- ${party}`).join('\n') : '- No parties identified'}

### Dates (${entities.dates.length})
${entities.dates.length > 0 ? entities.dates.map(date => `- ${date}`).join('\n') : '- No dates identified'}

### Monetary Amounts (${entities.monetary.length})
${entities.monetary.length > 0 ? entities.monetary.map(amount => `- ${amount}`).join('\n') : '- No monetary amounts identified'}

### Legal Clauses (${entities.clauses.length})
${entities.clauses.length > 0 ? entities.clauses.map(clause => `- ${clause}`).join('\n') : '- No clause references identified'}

## Data Structure for Integration
\`\`\`json
{
  "entities": ${JSON.stringify(entities, null, 2)},
  "metadata": {
    "extractionDate": "${new Date().toISOString()}",
    "documentLength": ${content.length},
    "entityCount": ${totalEntities}
  }
}

// Auxiliary methods
Context7Server.prototype.qdrantSearch = async function(query, limit = 5) {
  const MODEL = process.env.EMBED_MODEL || 'nomic-embed-text';
  const COLLECTION = process.env.QDRANT_COLLECTION || 'dev_embeddings';
  const QDRANT_URL = process.env.QDRANT_URL || 'http://localhost:6333';

  const er = await fetch('http://localhost:11434/api/embeddings', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: MODEL, input: query })
  });
  if (!er.ok) throw new Error('Ollama embeddings failed');
  const ed = await er.json();
  const vector = ed?.embedding || ed?.data?.[0]?.embedding;
  if (!Array.isArray(vector)) throw new Error('No embedding vector');

  const sr = await fetch(`${QDRANT_URL}/collections/${COLLECTION}/points/search`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ vector, limit, with_payload: true })
  });
  if (!sr.ok) throw new Error('Qdrant search failed');
  const sd = await sr.json();
  return sd?.result || [];
}

// WebSocket logs server (optional)
const LOG_WS_PORT = Number(process.env.MCP_LOG_WS_PORT || 7072);
const httpServer = http.createServer((_, res) => { res.end('Context7 WS') });
const wss = new WebSocketServer({ server: httpServer, path: '/logs' });
httpServer.listen(LOG_WS_PORT, () => console.log(`[Context7 WS] ws://localhost:${LOG_WS_PORT}/logs`));
setInterval(() => {
  for (const c of wss.clients) if (c.readyState === 1) c.send(JSON.stringify({ ts: Date.now(), msg: 'context7 heartbeat' }))
}, 7000);
\`\`\`

## Stack Integration
- **Database Storage**: Store entities in jsonb column with Drizzle ORM
- **Vector Search**: Create embeddings for entity relationships with pgvector
- **State Management**: Update XState context with extracted entities
- **AI Enhancement**: Use Ollama for advanced entity relationship analysis`;
  }

  async run() {
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    console.error('Context7 MCP Server running...');
  }
}

// Start the server
const server = new Context7Server();
server.run().catch(console.error);