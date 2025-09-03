// Enhanced Drizzle Configuration - Best Practices Implementation
// PostgreSQL + pgvector + Component Auto-Discovery + Production Ready

import { defineConfig } from 'drizzle-kit';
import * as dotenv from 'dotenv';
import path from 'path';

// Load environment variables with fallback chain
dotenv.config({ path: '../.env' });
dotenv.config({ path: '.env.local' });
dotenv.config({ path: '.env' });

// Database connection with fallback values for development  
const databaseUrl = process.env.DATABASE_URL || 'postgresql://postgres:123456@localhost:5432/legal_ai_db';

console.log('Drizzle using DATABASE_URL:', databaseUrl);

export default defineConfig({
  // ===== SCHEMA DISCOVERY - BEST PRACTICES =====

  // Use lightweight unified schema aggregator (unified-schema.ts)
  schema: ['./src/lib/server/db/unified-schema.ts'],

  // Output directory for migrations and introspection
  out: './drizzle',

  // ===== DATABASE CONFIGURATION =====

  dialect: 'postgresql',

  dbCredentials: {
    url: databaseUrl,
    // Enhanced connection options for production
    ssl: process.env.NODE_ENV === 'production' ? { rejectUnauthorized: false } : false
  },

  // ===== INTROSPECTION - PRODUCTION SETTINGS =====

  // Advanced introspection for existing database
  introspect: {
    casing: 'snake_case',           // Match PostgreSQL conventions
    breakpoints: true,              // Enable migration breakpoints
    bundle: true                    // Bundle related changes
  },

  // ===== MIGRATION MANAGEMENT =====

  migrations: {
    prefix: 'timestamp',                    // Use timestamp prefix for ordering
    table: '__drizzle_migrations__',       // Custom migration table name
    schema: 'public'                       // Target schema for migrations
  },

  // ===== FILTERING & OPTIMIZATION =====

  // Schema and table filtering for large databases
  schemaFilter: ['public'],               // Only public schema
  tablesFilter: '*',                      // All tables (can be restricted)

  // Extension support filters
  extensionsFilters: ['postgresjs'],      // PostgreSQL extensions

  // ===== TYPE GENERATION =====

  // Generate comprehensive TypeScript definitions
  breakpoints: true,                      // Support for migration breakpoints

  // ===== LOGGING & DEBUGGING =====

  verbose: true,                          // Detailed logging
  strict: true,                           // Strict validation

  // ===== CUSTOM CONFIGURATION =====

  // Entity-specific settings
  entities: {
    roles: false                          // Exclude role entities from generation
  },

  // ===== PGVECTOR EXTENSION SUPPORT =====

  // Custom SQL types for pgvector
  custom: {
    vector: {
      fromDriver: (value: string) => {
        // Parse pgvector format [1,2,3] to number[]
        return value.replace(/^\[|\]$/g, '').split(',').map(n => parseFloat(n.trim()));
      },
      toDriver: (value: number[]) => {
        // Convert number[] to pgvector format
        return `[${value.join(',')}]`;
      }
    }
  }
});

