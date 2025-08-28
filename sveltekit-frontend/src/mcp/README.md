# MCP Tools Layer - Legal AI Platform

## Overview

The MCP (Model Context Protocol) Tools Layer provides a clean abstraction layer for database operations in the Legal AI Platform. This layer follows the Context7 architectural patterns and integrates seamlessly with PostgreSQL + pgvector, Drizzle ORM, and the enhanced AI services.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                Frontend Layer                       │
│  (SvelteKit Components, XState Machines)           │
└─────────────────────┬───────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────┐
│                MCP Tools Layer                      │
│  • cases.mcp.ts     • evidence.mcp.ts             │
│  • users.mcp.ts     • ai-analysis.mcp.ts          │
│  • agentShellMachine.mcp.ts (XState integration)   │
└─────────────────────┬───────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────┐
│               Database Layer                        │
│  PostgreSQL + pgvector + Drizzle ORM               │
└─────────────────────────────────────────────────────┘
```

## Available Tools

### 1. Cases MCP Tool (`cases.mcp.ts`)
- **createCase**: Create new legal cases with validation
- **loadCases**: Query cases with filtering and pagination  
- **updateCase**: Update existing case details
- **addEvidence**: Link evidence to cases with vector embeddings
- **findSimilarCases**: Vector similarity search using pgvector
- **getCaseAnalytics**: Generate case statistics and insights

### 2. Evidence MCP Tool (`evidence.mcp.ts`)
- **createEvidence**: Create evidence records with metadata
- **loadEvidence**: Query evidence with case filtering
- **updateEvidence**: Update evidence details and classifications
- **findSimilarEvidence**: Vector-based evidence similarity search
- **getEvidenceAnalytics**: Evidence statistics by type and case
- **deleteEvidence**: Safe evidence removal (soft delete)

### 3. Users MCP Tool (`users.mcp.ts`)
- **createUser**: Create user accounts with role-based permissions
- **loadUsers**: Query users with department/jurisdiction filters
- **updateUser**: Update user profiles and permissions
- **findSimilarUsers**: AI-powered user matching by profile embeddings
- **getUserAnalytics**: User activity and role statistics
- **authenticateUser**: Secure password-based authentication
- **getUserById**: Retrieve user details without sensitive data

### 4. AI Analysis MCP Tool (`ai-analysis.mcp.ts`)
- **analyzeDocument**: AI-powered legal document analysis
- **performLegalAnalysis**: Comprehensive legal research and analysis
- **findSimilarDocuments**: Vector similarity search across document chunks
- **performBatchAnalysis**: Batch processing for multiple documents
- **assessCaseRisk**: Legal risk assessment with AI insights

## Usage Examples

### Basic Case Operations
```typescript
import { mcpTools } from '$lib/mcp';

// Create a new case
const caseResult = await mcpTools.cases.createCase({
  title: "Contract Dispute - ABC vs XYZ",
  description: "Commercial contract breach involving software licensing",
  userId: "user-123",
  priority: "high"
});

if (caseResult.success) {
  console.log("Case created:", caseResult.data);
}
```

### Evidence Management with Vector Search
```typescript
// Add evidence with embedding generation
const evidenceResult = await mcpTools.evidence.createEvidence({
  caseId: "case-456",
  title: "Signed Contract Document",
  description: "Original signed contract with disputed terms",
  evidenceType: "document",
  tags: ["contract", "signed", "disputed"]
});

// Find similar evidence using vector embeddings
const similarEvidence = await mcpTools.evidence.findSimilarEvidence({
  embedding: documentEmbedding, // 384-dimensional vector from AI model
  caseId: "case-456",
  threshold: 0.8,
  limit: 5
});
```

### AI-Powered Legal Analysis
```typescript
// Analyze a legal document
const analysisResult = await mcpTools.aiAnalysis.analyzeDocument({
  content: documentText,
  documentType: "contract",
  caseId: "case-456",
  userId: "user-123",
  generateEmbedding: true
});

if (analysisResult.success) {
  const analysis = analysisResult.data;
  console.log("Risk Level:", analysis.riskLevel);
  console.log("Key Findings:", analysis.keyFindings);
  console.log("Legal Implications:", analysis.legalImplications);
}

// Perform comprehensive case risk assessment
const riskAssessment = await mcpTools.aiAnalysis.assessCaseRisk({
  caseId: "case-456",
  factors: {
    evidenceQuality: true,
    legalPrecedents: true,
    jurisdictionalRisks: true,
    timelineAnalysis: true
  },
  userId: "user-123"
});
```

### XState Machine Integration
```typescript
import { agentShellMachineMCP, agentShellServicesMCP } from '$lib/machines/agentShellMachine.mcp';
import { createActor } from 'xstate';

// Create machine actor with MCP services
const agent = createActor(agentShellMachineMCP, {
  input: { services: agentShellServicesMCP }
});

agent.start();

// Trigger MCP operations through state machine
agent.send({
  type: "MCP_LOAD_CASE",
  caseId: "case-456"
});

agent.send({
  type: "MCP_CREATE_EVIDENCE",
  evidenceData: {
    title: "New Evidence",
    evidenceType: "document"
  },
  caseId: "case-456"
});
```

## Response Format

All MCP tools return a standardized response format:

```typescript
interface MCPToolResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  metadata?: {
    tool: string;           // e.g., "cases.createCase"
    timestamp: number;      // Unix timestamp
    [key: string]: any;     // Additional metadata
  };
}
```

### Success Response Example
```typescript
{
  success: true,
  data: {
    id: "case-123",
    title: "New Legal Case",
    status: "active",
    // ... other case data
  },
  metadata: {
    tool: "cases.createCase",
    timestamp: 1703123456789,
    hasEmbedding: false
  }
}
```

### Error Response Example
```typescript
{
  success: false,
  error: "User not found",
  metadata: {
    tool: "users.updateUser",
    userId: "invalid-user-id",
    timestamp: 1703123456789
  }
}
```

## Integration with Services

The MCP layer integrates with several backend services:

- **Enhanced RAG Service** (Port 8094): AI analysis and vector operations
- **Upload Service** (Port 8093): File processing and evidence management  
- **PostgreSQL**: Primary database with pgvector extension
- **Ollama Models**: Legal AI analysis (gemma3-legal, nomic-embed-text)

## Vector Operations

The MCP layer leverages PostgreSQL's pgvector extension for similarity searches:

- **Embedding Dimensions**: 384 (nomic-embed-text model)
- **Similarity Methods**: Cosine distance, L2 distance
- **Index Types**: HNSW for high-performance vector search
- **Threshold Tuning**: Configurable similarity thresholds (default: 0.7)

## Performance Considerations

- **Batch Operations**: Use batch analysis for multiple documents
- **Lazy Loading**: Implement pagination for large result sets
- **Embedding Caching**: Store generated embeddings for reuse
- **Index Optimization**: Proper vector index configuration for fast searches
- **Connection Pooling**: Database connections managed by Drizzle ORM

## Security Features

- **Input Validation**: All parameters validated before database operations
- **SQL Injection Prevention**: Drizzle ORM provides query parameterization
- **Role-Based Access**: User permissions checked before sensitive operations
- **Password Security**: bcrypt hashing for user authentication
- **Data Sanitization**: Sensitive data (passwords) excluded from responses

## Future Enhancements

- **Real-time Updates**: WebSocket integration for live case updates
- **Audit Logging**: Track all MCP operations for compliance
- **Caching Layer**: Redis integration for performance optimization  
- **Multi-tenant Support**: Organization-level data isolation
- **API Rate Limiting**: Prevent abuse of AI analysis operations
- **Backup Integration**: Automated backup triggers for critical operations

## Development Workflow

1. **Import Tools**: `import { mcpTools } from '$lib/mcp'`
2. **Call Operations**: Use appropriate MCP tool methods
3. **Handle Responses**: Check `success` flag and handle errors
4. **Integrate with UI**: Use data in Svelte components
5. **State Management**: Integrate with XState machines for complex workflows

This MCP Tools Layer provides a robust, type-safe, and performant foundation for all database operations in the Legal AI Platform, following industry best practices and Context7 architectural patterns.