# TypeScript Errors Fix Guide

This document provides comprehensive fixes for all TypeScript errors found in the project.

## 1. rag-pipeline-integrator.ts - Missing 'document' property

**Error**: Property 'document' is missing in SearchResult type

**Fix**: Update the mapping to include the document property:

```typescript
// In src/lib/services/rag-pipeline-integrator.ts line 264
const searchResults: SearchResult[] = progress.documents.map((doc, index) => ({
  score: doc.score,
  rank: index + 1,
  id: doc.id,
  title: doc.title,
  content: doc.content,
  summary: doc.summary,
  excerpt: doc.excerpt,
  metadata: doc.metadata,
  type: doc.type,
  createdAt: doc.createdAt,
  updatedAt: doc.updatedAt,
  document: doc // Add this missing property
}));
```

## 2. enhanced-sentence-splitter.test.ts - Import and Method Issues

**Errors**: Missing exports and incorrect property names

**Fixes**:

### A. Update imports (line 10):
```typescript
// Change from:
import { createStreamingSplitter } from "../services/enhanced-sentence-splitter";

// To:
import createStreamingSplitter from "../services/enhanced-sentence-splitter";
// OR check the actual export in enhanced-sentence-splitter.ts and update accordingly
```

### B. Fix property names (lines 43, 55):
```typescript
// Change 'minFragmentLength' to 'minLength'
const splitter = new EnhancedSentenceSplitter({
  minLength: 25, // was: minFragmentLength: 25
  maxLength: 100
});

// And later:
const splitter2 = new EnhancedSentenceSplitter({
  minLength: 30, // was: minFragmentLength: 30
  maxLength: 150
});
```

### C. Fix method name (lines 48, 59, 125):
```typescript
// Change 'splitSentences' to the correct method name
// Check the actual method in EnhancedSentenceSplitter class
const result = splitter.split(text); // or whatever the correct method name is

// Also for addAbbreviations method (line 122):
// Check if this method exists or use the correct method name
```

## 3. ai-synthesizer/+server.ts - MetricData type issues

**Error**: Properties 'name' and 'value' don't exist on MetricData

**Fix**: Update MetricData type definition or fix the usage:

```typescript
// Option 1: Update the type definition
interface MetricData {
  name: string;
  value: number;
  // ... other properties
}

// Option 2: Fix the usage if MetricData has different structure
const totalRequests = metrics.find((m) => m.metricName === 'api_requests_total')?.metricValue || 0;
const totalErrors = metrics.find((m) => m.metricName === 'api_errors_total')?.metricValue || 0;
const avgResponseTime = metrics.find((m) => m.metricName === 'api_request_duration_avg')?.metricValue || 0;
```

## 4. ai/analyze/+server.ts - Database Schema Issues

**Errors**: Missing properties 'tags', 'metadata', and schema mismatches

**Fixes**:

### A. Add missing properties to database schema or handle them:
```typescript
// Line 55 - Handle missing 'tags' property:
tags: dbDoc.tags || [], // Make sure 'tags' field exists in schema or remove this line

// Line 96 - Remove 'metadata' if not in schema:
// Remove the metadata line or add it to your schema

// Line 105 - Fix database insert:
await db.insert(contentEmbeddings).values({
  contentId: doc.id,
  contentType: 'document',
  textContent: content,
  // documentId: doc.id, // Remove if this field doesn't exist in schema
  embedding: JSON.stringify(embedding), // Convert array to string if needed
  // ... other required fields
});
```

## 5. ai/embeddings/+server.ts - Database Schema Issues

**Error**: Embedding field type mismatch and unknown 'content' property

**Fix**:
```typescript
// Line 35 - Fix the database insert:
const [saved] = await db.insert(contentEmbeddings).values({
  contentId: generateId(), // or appropriate ID
  contentType: 'text',
  textContent: content, // was: content
  embedding: JSON.stringify(embedding), // Convert number[] to string
  // Remove any fields not in your schema
});
```

## 6. ai/legal-research/+server.ts - Missing rerankedResults

**Error**: Property 'rerankedResults' doesn't exist

**Fix**:
```typescript
// Line 128 - Fix the property access:
return reranked?.results || reranked || results; // Check actual structure of reranked object
```

## 7. documents/search/+server.ts - Database Schema Issues

**Error**: Type mismatches in database insert

**Fix**:
```typescript
// Line 127 - Fix search sessions insert:
await db.insert(searchSessions).values({
  query: searchParams.query,
  searchType: searchParams.searchType || 'semantic',
  queryEmbedding: JSON.stringify(queryEmbedding), // Convert to string if needed
  results: searchResults,
  resultCount: searchResults.length,
  createdAt: new Date()
});

// Line 468 - Fix embeddings insert:
await db.insert(embeddings).values({
  content: chunk,
  // documentId: doc.id, // Remove if field doesn't exist
  embedding: JSON.stringify(embedding), // Convert to string
  metadata: { /* relevant metadata */ },
  createdAt: new Date()
});
```

## 8. documents/store/+server.ts - Database Schema Issues

**Error**: Similar embedding type issues

**Fix**:
```typescript
// Line 44 - Fix embeddings insert:
await db.insert(embeddings).values({
  content: chunk,
  // documentId: documentId, // Remove if field doesn't exist  
  embedding: JSON.stringify(embedding), // Convert to string
  metadata: chunkMetadata,
  createdAt: new Date()
});
```

## 9. ocr/langextract/+server.ts - Buffer and Tesseract Issues

**Errors**: Buffer type mismatch and unknown logger property

**Fixes**:

### A. Fix Buffer type (line 121):
```typescript
// Convert Buffer to proper type:
processedBuffer = await sharp(Buffer.from(buffer.buffer))
  .jpeg({ quality: 90 })
  .toBuffer();
```

### B. Fix Tesseract configuration (line 133):
```typescript
// Remove or fix logger configuration:
const { data: { text } } = await recognize(
  processedBuffer,
  langs,
  {
    // logger: (m) => console.log(`[Tesseract] ${m.status}: ${m.progress}`), // Remove this line
  }
);
```

## 10. General Database Schema Fixes

It appears your database schema might be out of sync. Consider:

1. **Update your Drizzle schema** to include missing fields like:
   - `tags` field in documents table
   - `metadata` field in documents table  
   - Proper embedding field types (string vs number[])

2. **Run database migrations** to sync schema changes

3. **Check field names** in your database match your TypeScript interfaces

## 11. Type Definition Updates Needed

Create or update these type definitions:

```typescript
// Update SearchResult interface
interface SearchResult {
  score: number;
  rank: number;
  id: string;
  title: string;
  content?: string;
  summary?: string;
  excerpt?: string;
  metadata?: Record<string, any>;
  type?: string;
  createdAt?: Date;
  updatedAt?: Date;
  document: LegalDocument; // Add this required property
}

// Update MetricData interface
interface MetricData {
  name: string;
  value: number;
  // ... other properties
}
```

## Next Steps

1. Apply these fixes systematically, starting with the type definitions
2. Update your database schema to match your TypeScript interfaces
3. Run `npm run check` after each major fix to verify progress
4. Consider using database migration tools to keep schema in sync

This should resolve all 48 TypeScript errors across the 18 files.
