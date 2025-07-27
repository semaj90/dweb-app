import { GraphQLSchema } from 'graphql';
import { createYoga } from 'graphql-yoga';
import { drizzle } from 'drizzle-orm/node-postgres';
import SchemaBuilder from '@pothos/core';
import DrizzlePlugin from '@pothos/plugin-drizzle';
import { db } from '$lib/db';
import type { DB } from '$lib/db/types';
import { LocalLLMService } from '$lib/ai/local-llm-service';
import { vectorSearch, extractRelationships } from '$lib/services/vectorService';

// Initialize builder with plugins
const builder = new SchemaBuilder<{
  DrizzleSchema: typeof import('$lib/db/schema');
  Context: {
    user?: { id: string };
    db: DB;
    llm: LocalLLMService;
  };
}>({
  plugins: [DrizzlePlugin],
  drizzle: {
    client: db,
  },
});

// Define GraphQL types
const CaseResult = builder.objectType('CaseResult', {
  fields: (t) => ({
    id: t.exposeString('id'),
    title: t.exposeString('title'),
    content: t.exposeString('content', { nullable: true }),
    relevanceScore: t.exposeFloat('relevanceScore', { nullable: true }),
    metadata: t.field({
      type: 'JSON',
      nullable: true,
      resolve: (parent) => parent.metadata,
    }),
  }),
});

const AnalysisResult = builder.objectType('AnalysisResult', {
  fields: (t) => ({
    caseId: t.exposeString('caseId'),
    analysisType: t.exposeString('analysisType'),
    result: t.exposeString('result'),
    confidence: t.exposeFloat('confidence'),
    metadata: t.field({
      type: 'JSON',
      nullable: true,
      resolve: (parent) => parent.metadata,
    }),
  }),
});

// JSON scalar type
builder.scalarType('JSON', {
  serialize: (value) => value,
  parseValue: (value) => value,
});

// Query type
builder.queryType({
  fields: (t) => ({
    // Legal case queries with local embeddings
    searchCases: t.field({
      type: [CaseResult],
      args: {
        query: t.arg.string({ required: true }),
        limit: t.arg.int({ defaultValue: 10 }),
      },
      resolve: async (parent, args, ctx) => {
        // Use local Gemma3 embeddings via Ollama
        const embedding = await ctx.llm.embed(args.query);
        return vectorSearch(embedding, args.limit);
      },
    }),
    
    // AI analysis with Gemma3
    analyzeCaseWithAI: t.field({
      type: AnalysisResult,
      args: {
        caseId: t.arg.string({ required: true }),
        analysisType: t.arg.string({ required: true }),
      },
      resolve: async (parent, args, ctx) => {
        return ctx.llm.analyzeCase(args.caseId, args.analysisType);
      },
    }),
    
    // Get case by ID
    getCase: t.field({
      type: CaseResult,
      nullable: true,
      args: {
        id: t.arg.string({ required: true }),
      },
      resolve: async (parent, args, ctx) => {
        const result = await ctx.db
          .select()
          .from(cases)
          .where(eq(cases.id, args.id))
          .limit(1);
        return result[0] || null;
      },
    }),
  }),
});

// Mutation type
builder.mutationType({
  fields: (t) => ({
    // Upload and process document
    uploadDocument: t.field({
      type: builder.objectType('UploadResult', {
        fields: (t) => ({
          success: t.exposeBoolean('success'),
          documentId: t.exposeString('documentId', { nullable: true }),
          documentCount: t.exposeInt('documentCount'),
          error: t.exposeString('error', { nullable: true }),
        }),
      }),
      args: {
        content: t.arg.string({ required: true }),
        title: t.arg.string({ required: true }),
        metadata: t.arg({ type: 'JSON', required: false }),
      },
      resolve: async (parent, args, ctx) => {
        try {
          // Process document with local LLM
          const chunks = await ctx.llm.splitText(args.content);
          
          // Batch embed with local model
          const embeddings = await ctx.llm.embedBatch(chunks);
          
          // Store in database
          const documents = chunks.map((chunk, i) => ({
            content: chunk,
            embedding: embeddings[i],
            title: args.title,
            metadata: args.metadata || {},
          }));
          
          const inserted = await ctx.db.insert(documentsTable).values(documents).returning();
          
          // Extract relationships for graph processing
          const relationships = await extractRelationships(args.content);
          
          return {
            success: true,
            documentId: inserted[0]?.id,
            documentCount: documents.length,
            error: null,
          };
        } catch (error) {
          console.error('Upload error:', error);
          return {
            success: false,
            documentId: null,
            documentCount: 0,
            error: error.message,
          };
        }
      },
    }),
    
    // Create new case
    createCase: t.field({
      type: CaseResult,
      args: {
        title: t.arg.string({ required: true }),
        content: t.arg.string({ required: false }),
        metadata: t.arg({ type: 'JSON', required: false }),
      },
      resolve: async (parent, args, ctx) => {
        const [newCase] = await ctx.db
          .insert(cases)
          .values({
            title: args.title,
            content: args.content || '',
            status: 'active',
            metadata: args.metadata || {},
          })
          .returning();
        return newCase;
      },
    }),
  }),
});

// Subscription type
builder.subscriptionType({
  fields: (t) => ({
    analysisProgress: t.field({
      type: builder.objectType('AnalysisProgress', {
        fields: (t) => ({
          taskId: t.exposeString('taskId'),
          progress: t.exposeFloat('progress'),
          status: t.exposeString('status'),
          message: t.exposeString('message', { nullable: true }),
        }),
      }),
      args: {
        taskId: t.arg.string({ required: true }),
      },
      subscribe: async function* (parent, args, ctx) {
        // Subscribe to Redis channel for progress updates
        const channel = `analysis:${args.taskId}`;
        
        // Mock subscription for now - replace with actual Redis subscription
        for (let i = 0; i <= 100; i += 10) {
          yield {
            taskId: args.taskId,
            progress: i / 100,
            status: i < 100 ? 'processing' : 'complete',
            message: `Processing... ${i}%`,
          };
          await new Promise(resolve => setTimeout(resolve, 1000));
        }
      },
      resolve: (payload) => payload,
    }),
  }),
});

// Build and export schema
export const schema = builder.toSchema();

// Import required from schema
import { cases, documents as documentsTable } from '$lib/db/schema';
import { eq } from 'drizzle-orm';
