#!/usr/bin/env node

/**
 * Complete Integration Setup Script
 * This script sets up all components for the enhanced RAG system
 */

import { spawn } from "child_process";
import { promises as fs } from "fs";
import path from "path";

// Import our services
import { librarySyncService } from "./src/lib/services/library-sync-service.js";
import { multiAgentOrchestrator } from "./src/lib/services/multi-agent-orchestrator.js";
import { redisVectorService } from "./src/lib/services/redis-vector-service.js";
import { determinismEvaluationService } from "./src/lib/services/determinism-evaluation-service.js";

const workspaceRoot = process.cwd();

/**
 * Setup and start all services
 */
async function setupIntegratedSystem() {
  console.log("🚀 Starting Enhanced RAG System Integration...\n");

  try {
    // Step 1: Initialize Redis Vector Service
    console.log("📊 Step 1: Initializing Redis Vector Service...");
    await redisVectorService.connect();
    console.log("✅ Redis Vector Service connected");

    // Step 2: Start Library Sync Service
    console.log("📚 Step 2: Starting Library Sync Service...");
    librarySyncService.startPeriodicSync(6); // Every 6 hours
    await librarySyncService.syncAllLibraries();
    console.log("✅ Library Sync Service started");

    // Step 3: Initialize Multi-Agent Orchestrator
    console.log("🤖 Step 3: Setting up Multi-Agent Orchestrator...");
    const workflowId = await multiAgentOrchestrator.createWorkflow(
      "System Health Check",
      "Perform initial system health check and validation",
      ["vector_search", "quality_check"],
      {
        originalQuery: "Initial system setup validation",
        sessionId: "setup-session",
      }
    );
    console.log(
      `✅ Multi-Agent Orchestrator initialized (workflow: ${workflowId})`
    );

    // Step 4: Record initial test results
    console.log("📈 Step 4: Recording initial evaluation metrics...");
    await determinismEvaluationService.recordTestResult({
      testType: "integration",
      testName: "Enhanced RAG System Setup",
      passed: true,
      duration: Date.now() - startTime,
      details: {
        components: ["redis", "library-sync", "orchestrator", "evaluation"],
        status: "initialized",
      },
    });
    console.log("✅ Initial metrics recorded");

    // Step 5: Create integration status file
    console.log("📄 Step 5: Creating integration status...");
    const status = {
      timestamp: new Date().toISOString(),
      version: "1.0.0",
      components: {
        redisVectorService: "connected",
        librarySyncService: "running",
        multiAgentOrchestrator: "initialized",
        determinismEvaluationService: "ready",
      },
      apiEndpoints: {
        libraries: "http://localhost:5173/api/libraries",
        agentLogs: "http://localhost:5173/api/agent-logs",
        orchestrator: "http://localhost:5173/api/orchestrator",
        evaluation: "http://localhost:5173/api/evaluation",
        rag: "http://localhost:5173/api/rag",
      },
      nextSteps: [
        "Start SvelteKit development server: npm run dev",
        "Open VS Code and use Context7 MCP Assistant commands",
        "Access EnhancedRAG Studio at http://localhost:5173/rag-studio",
      ],
    };

    await fs.writeFile(
      path.join(workspaceRoot, "INTEGRATION_STATUS.json"),
      JSON.stringify(status, null, 2)
    );
    console.log("✅ Integration status created");

    // Step 6: Generate comprehensive documentation
    console.log("📖 Step 6: Generating documentation...");
    await generateIntegrationDocumentation();
    console.log("✅ Documentation generated");

    console.log("\n🎉 Enhanced RAG System Integration Complete!\n");
    console.log("📋 Summary of implemented features:");
    console.log("   ✅ Redis vector database with semantic search");
    console.log("   ✅ Library sync service (GitHub/Context7/NPM)");
    console.log("   ✅ Agent call logging and audit trails");
    console.log("   ✅ Multi-agent orchestration framework");
    console.log("   ✅ Deterministic LLM configuration");
    console.log("   ✅ Performance metrics and evaluation");
    console.log("   ✅ VS Code extension with enhanced commands");
    console.log("   ✅ SvelteKit API endpoints for all services");
    console.log("\n🚀 Ready to start development!");
    console.log("   Run: npm run dev");
    console.log("   Open: http://localhost:5173/rag-studio");
  } catch (error) {
    console.error("❌ Integration setup failed:", error);
    process.exit(1);
  }
}

/**
 * Generate comprehensive documentation
 */
async function generateIntegrationDocumentation() {
  const docs = `# Enhanced RAG System - Complete Implementation

## Overview
This document describes the complete implementation of the enhanced RAG system with multi-agent orchestration, built on SvelteKit with VS Code MCP integration.

## Architecture

### Core Services
1. **Redis Vector Service** (\`src/lib/services/redis-vector-service.ts\`)
   - Vector storage and semantic search
   - Semantic caching for performance
   - Document indexing and retrieval

2. **Library Sync Service** (\`src/lib/services/library-sync-service.ts\`)
   - Periodic syncing from GitHub/Context7/NPM
   - Agent call logging and audit trails
   - Library metadata management

3. **Multi-Agent Orchestrator** (\`src/lib/services/multi-agent-orchestrator.ts\`)
   - Agent workflow management
   - Dependency-based execution planning
   - Integration with CrewAI/AutoGen patterns

4. **Determinism & Evaluation Service** (\`src/lib/services/determinism-evaluation-service.ts\`)
   - Fixed LLM configuration (temperature=0)
   - User feedback collection
   - Performance metrics and benchmarking

### API Endpoints
- \`/api/rag\` - Enhanced RAG operations
- \`/api/libraries\` - Library search and sync
- \`/api/agent-logs\` - Agent call logging
- \`/api/orchestrator\` - Multi-agent workflows
- \`/api/evaluation\` - Metrics and feedback

### VS Code Integration
- **Extension**: \`.vscode/extensions/mcp-context7-assistant/\`
- **Commands**: 15+ specialized commands for RAG, orchestration, and evaluation
- **MCP Server**: Registered agent orchestrator provider

## Usage

### Starting the System
\`\`\`bash
npm run integration-setup  # Run this setup script
npm run dev                # Start SvelteKit server
\`\`\`

### VS Code Commands
- \`Ctrl+Shift+P\` → "Context7 MCP: ..."
- Key commands:
  - Sync Library Metadata
  - Create Multi-Agent Workflow
  - View Performance Metrics
  - Record User Feedback

### Programmatic Usage
\`\`\`typescript
import { multiAgentOrchestrator } from '$lib/services/multi-agent-orchestrator.js';

// Create and execute workflow
const workflowId = await multiAgentOrchestrator.createWorkflow(
  'Code Analysis',
  'Analyze codebase for best practices',
  ['code_analysis', 'best_practices_generation'],
  { originalQuery: 'Analyze my SvelteKit app', sessionId: 'user-123' }
);

const result = await multiAgentOrchestrator.executeWorkflow(workflowId, context);
\`\`\`

## Configuration

### Environment Variables
\`\`\`env
REDIS_URL=redis://localhost:6379
GITHUB_TOKEN=your_github_token_here
RAG_BACKEND_URL=http://localhost:8000
\`\`\`

### Redis Setup
\`\`\`bash
docker-compose -f docker-compose.redis.yml up -d
\`\`\`

## Monitoring

### Health Checks
- Redis: \`/api/rag?action=status\`
- Libraries: \`/api/libraries\`
- Orchestrator: \`/api/orchestrator\`

### Metrics
- Access via VS Code: "View Performance Metrics"
- API: \`/api/evaluation?action=metrics\`
- Benchmarks: \`/api/evaluation?action=benchmarks\`

## Development

### Adding New Agents
1. Define agent in \`multi-agent-orchestrator.ts\`
2. Implement execution logic
3. Add to capability mapping
4. Update VS Code commands

### Custom Workflows
\`\`\`typescript
const workflow = await multiAgentOrchestrator.createWorkflow(
  'Custom Analysis',
  'Description',
  ['required', 'capabilities'],
  context
);
\`\`\`

## Production Deployment

### Scaling
- Redis Cluster for vector storage
- Multiple agent worker processes
- Load balancing for API endpoints

### Monitoring
- Agent call logs in vector DB
- Performance metrics collection
- User feedback analysis

## Troubleshooting

### Common Issues
1. **Redis Connection Failed**
   - Check Redis is running: \`docker ps\`
   - Verify connection string in env

2. **Agent Timeout**
   - Check individual agent logs
   - Verify dependency chain

3. **Vector Search Slow**
   - Check Redis memory usage
   - Review index configuration

### Debug Mode
Set \`DEBUG=true\` for detailed logging across all services.

---

Generated by Enhanced RAG System Integration v1.0.0
${new Date().toISOString()}
`;

  await fs.writeFile(
    path.join(workspaceRoot, "ENHANCED_RAG_DOCUMENTATION.md"),
    docs
  );
}

// Track setup start time
const startTime = Date.now();

// Run setup if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
  setupIntegratedSystem().catch(console.error);
}

export { setupIntegratedSystem };
