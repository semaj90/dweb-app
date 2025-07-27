# Legal AI Case Management - CRUD CMS Documentation

## Database Schema Overview

### Core Tables
- **users**: User accounts (admin, prosecutor, detective, user)
- **cases**: Legal cases with status tracking
- **evidence**: Evidence files and metadata
- **documents**: AI-processed documents with embeddings
- **notes**: Case and evidence annotations
- **ai_history**: AI interaction tracking
- **collaboration_sessions**: Real-time collaboration

### Key Relations
```sql
cases.created_by → users.id
evidence.case_id → cases.id
evidence.created_by → users.id
documents.case_id → cases.id
documents.evidence_id → evidence.id
```

## CRUD Operations

### Cases CRUD
```javascript
// Create
const newCase = await db.insert(cases).values({
  title: "Case Title",
  description: "Description",
  status: "active",
  priority: "high",
  createdBy: userId
});

// Read with relations
const caseWithEvidence = await db.query.cases.findFirst({
  where: eq(cases.id, caseId),
  with: {
    evidence: true,
    creator: true
  }
});

// Update
await db.update(cases)
  .set({ status: "closed" })
  .where(eq(cases.id, caseId));

// Delete
await db.delete(cases).where(eq(cases.id, caseId));
```

### Evidence CRUD
```javascript
// Create
const newEvidence = await db.insert(evidence).values({
  caseId: caseId,
  title: "Evidence Title", 
  type: "document",
  content: "Extracted text",
  createdBy: userId
});

// Read with case
const evidenceWithCase = await db.query.evidence.findFirst({
  where: eq(evidence.id, evidenceId),
  with: {
    case: true,
    creator: true
  }
});

// Update
await db.update(evidence)
  .set({ title: "Updated Title" })
  .where(eq(evidence.id, evidenceId));

// Delete
await db.delete(evidence).where(eq(evidence.id, evidenceId));
```

## Testing Commands

```bash
# Test PostgreSQL connection
node verify-database.mjs

# Test CRUD operations
node test-crud.mjs

# Validate schema
node validate-schema.mjs

# Full system test
TEST-FULL-SYSTEM.bat
```

## API Integration

### SvelteKit Routes
- `/api/cases` - Cases CRUD API
- `/api/evidence` - Evidence CRUD API  
- `/api/users` - User management API

### Type Safety
All operations use Drizzle ORM types:
- `Case`, `NewCase`
- `Evidence`, `NewEvidence`
- `User`, `NewUser`

## Vector Search (AI Features)

```javascript
// Document with embeddings
const document = await db.insert(documents).values({
  caseId: caseId,
  filename: "document.pdf",
  extractedText: text,
  embeddings: vectorEmbedding // 384-dimensional vector
});

// Similarity search
const similarDocs = await db
  .select()
  .from(documents)
  .where(sql`embeddings <-> ${queryVector} < 0.5`);
```

## Real-time Features

### Collaboration Sessions
```javascript
// Start collaboration
const session = await db.insert(collaborationSessions).values({
  caseId: caseId,
  userId: userId,
  sessionId: generateSessionId(),
  isActive: true
});
```

### AI History Tracking
```javascript
// Log AI interaction
const aiLog = await db.insert(aiHistory).values({
  caseId: caseId,
  userId: userId,
  prompt: userPrompt,
  response: aiResponse,
  model: "gpt-4",
  tokensUsed: 1500
});
```

## Development Workflow

1. **Setup**: Run `setup-database.mjs --seed`
2. **Test**: Run `TEST-FULL-SYSTEM.bat`
3. **Develop**: Use `npm run dev`
4. **Debug**: Check logs in generated `.txt` files

## Production Considerations

- Use connection pooling for PostgreSQL
- Enable row-level security (RLS)
- Regular backups of case data
- Monitor AI token usage costs
- Implement audit logging for evidence changes

## RAG System Integration

The RAG (Retrieval Augmented Generation) system is now fully integrated with MCP tools for seamless legal document search and analysis.

### RAG-MCP Tools Available:

1. **`rag-query`** - Semantic search across legal documents
   ```
   Usage: Ask Claude to "rag query '[legal question]' for case [case-id]"
   Example: "rag query 'contract liability clauses' for case CASE-2024-001"
   ```

2. **`rag-upload-document`** - Upload and index legal documents
   ```
   Usage: Ask Claude to "upload document '[file-path]' to case [case-id] as [document-type]"
   Example: "upload document '/legal/contract.pdf' to case CASE-001 as contract"
   ```

3. **`rag-get-stats`** - Get RAG system health and statistics
   ```
   Usage: Ask Claude to "get rag system statistics"
   ```

4. **`rag-analyze-relevance`** - Analyze document relevance for queries
   ```
   Usage: Ask Claude to "analyze relevance of document [doc-id] for query '[query]'"
   Example: "analyze relevance of document doc-123 for query 'liability clauses'"
   ```

5. **`rag-integration-guide`** - Get integration guidance for SvelteKit
   ```
   Usage: Ask Claude to "get rag integration guide for [type]"
   Types: api-integration, component-integration, search-ui, document-upload
   Example: "get rag integration guide for search-ui"
   ```

### RAG Common Queries (TypeScript):

```typescript
import { commonMCPQueries } from '$lib/utils/mcp-helpers'

// Legal document search
const legalQuery = commonMCPQueries.ragLegalQuery('contract terms', 'CASE-001')

// Contract analysis
const contractQuery = commonMCPQueries.ragContractAnalysis('liability provisions')

// Case law search
const caseLawQuery = commonMCPQueries.ragCaseLawSearch('employment disputes')

// Evidence search
const evidenceQuery = commonMCPQueries.ragEvidenceSearch('digital forensics', 'CASE-001')

// Integration guides
const apiGuide = commonMCPQueries.ragApiIntegration()
const uiGuide = commonMCPQueries.ragSearchUI()
```

### RAG Configuration:

```bash
# Environment variables for RAG system
RAG_ENDPOINT=http://localhost:8000
RAG_ENABLED=true
DATABASE_URL=postgresql://user:pass@localhost:5432/legal_ai
QDRANT_URL=http://localhost:6333  # Optional vector database
```

### RAG Demo Interface:

Access the interactive RAG testing interface at:
```
http://localhost:5173/dev/mcp-tools
```

Features:
- Legal document search testing
- Document upload simulation
- System statistics monitoring
- Integration code examples
- Relevance analysis tools

## MCP Context7 Tools (Available via Claude Code)

### Available MCP Tools for Stack Development:

1. **`analyze-stack`** - Analyze any component with context-aware suggestions
   ```
   Usage: Ask Claude to "analyze [component] with context [legal-ai|gaming-ui|performance]"
   Example: "analyze sveltekit with context legal-ai"
   ```

2. **`generate-best-practices`** - Get best practices for specific areas
   ```
   Usage: Ask Claude to "generate best practices for [area]"
   Areas: performance, security, ui-ux
   Example: "generate best practices for performance"
   ```

3. **`suggest-integration`** - Integration patterns for new features
   ```
   Usage: Ask Claude to "suggest integration for [feature] with requirements [details]"
   Example: "suggest integration for document upload with requirements security and audit trails"
   ```

4. **`resolve-library-id`** - Find Context7-compatible library IDs
   ```
   Usage: Ask Claude to "resolve library id for [library-name]"
   Example: "resolve library id for drizzle"
   ```

5. **`get-library-docs`** - Retrieve specific documentation with topic filtering
   ```
   Usage: Ask Claude to "get library docs for [library-id] topic [topic]"
   Example: "get library docs for sveltekit topic routing"
   ```

### Available Resources:

- **`context7://stack-overview`** - Complete technology stack configuration
- **`context7://integration-guide`** - Component integration guide
- **`context7://performance-tips`** - Performance optimization recommendations

### Quick Reference Commands:

```bash
# Ask for stack analysis
"analyze typescript with context legal-ai"

# Get performance guidance  
"generate best practices for performance"

# Integration help
"suggest integration for AI chat component"

# Library documentation
"get library docs for bits-ui topic dialog"

# Stack overview
"show me the stack overview resource"
```

### JSON Function Helper:

For complex queries, use this JSON structure:
```json
{
  "tool": "analyze-stack|generate-best-practices|suggest-integration|resolve-library-id|get-library-docs",
  "component": "string",
  "context": "legal-ai|gaming-ui|performance",
  "area": "performance|security|ui-ux",
  "feature": "string",
  "requirements": "string",
  "library": "string",
  "topic": "string"
}
```

### MCP Helper Functions (TypeScript):

Location: `src/lib/utils/mcp-helpers.ts`

```typescript
import { generateMCPPrompt, commonMCPQueries } from '$lib/utils/mcp-helpers'

// Quick access to common queries
const svelteKitQuery = commonMCPQueries.analyzeSvelteKit()
const performanceQuery = commonMCPQueries.performanceBestPractices()
const aiChatQuery = commonMCPQueries.aiChatIntegration()

// Generate prompts programmatically
const customPrompt = generateMCPPrompt({
  tool: 'analyze-stack',
  component: 'typescript',
  context: 'legal-ai'
})
```

### Common Pre-built Queries:

```typescript
// Stack Analysis
commonMCPQueries.analyzeSvelteKit()      // SvelteKit for legal AI
commonMCPQueries.analyzeDrizzle()        // Drizzle ORM for legal data
commonMCPQueries.analyzeUnoCSS()         // UnoCSS performance

// Best Practices
commonMCPQueries.performanceBestPractices()  // Performance optimization
commonMCPQueries.securityBestPractices()     // Security guidelines
commonMCPQueries.uiUxBestPractices()         // UI/UX patterns

// Integration Help
commonMCPQueries.aiChatIntegration()         // AI chat components
commonMCPQueries.documentUploadIntegration() // Document upload system
commonMCPQueries.gamingUIIntegration()       // Gaming-style UI

// Documentation
commonMCPQueries.svelteKitRouting()      // SvelteKit routing docs
commonMCPQueries.bitsUIDialog()          // Bits UI dialog components
commonMCPQueries.drizzleSchema()         // Drizzle schema patterns
```

## MCP Servers Status:

- ✅ **context7-custom** - Stack-aware analysis and best practices
- ✅ **figma-http** - Figma API integration with design tokens
- ⚠️ **serena** - Semantic code analysis (requires setup)
- ❌ **filesystem** - Local file access (connection issues)
- ❌ **puppeteer** - Browser automation (deprecated package)
