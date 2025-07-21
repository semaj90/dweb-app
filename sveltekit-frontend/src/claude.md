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
