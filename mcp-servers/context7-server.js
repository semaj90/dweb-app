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
          }
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
- **VLLM**: Efficient LLM inference
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
- Integrate with existing VLLM setup
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
   - Optimize VLLM configuration
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
      vllm: 'Efficient LLM inference for legal AI capabilities.',
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
- Optimize VLLM model loading
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

  async run() {
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    console.error('Context7 MCP Server running...');
  }
}

// Start the server
const server = new Context7Server();
server.run().catch(console.error);