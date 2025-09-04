#!/usr/bin/env node
/**
 * Enhanced Database Setup Script
 * Sets up PostgreSQL with pgvector, generates Drizzle migrations, and configures Qdrant
 */

import { execSync } from 'child_process';
import { readFileSync, writeFileSync, existsSync } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import postgres from 'postgres';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Color output utilities
const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  dim: '\x1b[2m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  magenta: '\x1b[35m',
  cyan: '\x1b[36m',
  white: '\x1b[37m'
};

const log = {
  info: (msg) => console.log(`${colors.cyan}ℹ${colors.reset} ${msg}`),
  success: (msg) => console.log(`${colors.green}✅${colors.reset} ${msg}`),
  warning: (msg) => console.log(`${colors.yellow}⚠${colors.reset} ${msg}`),
  error: (msg) => console.log(`${colors.red}❌${colors.reset} ${msg}`),
  step: (msg) => console.log(`${colors.blue}🔄${colors.reset} ${msg}`),
};

// Configuration
const config = {
  DATABASE_URL: process.env.DATABASE_URL || 
    `postgresql://${process.env.DATABASE_USER || 'legal_admin'}:${process.env.DATABASE_PASSWORD || '123456'}@${process.env.DATABASE_HOST || 'localhost'}:${process.env.DATABASE_PORT || '5432'}/${process.env.DATABASE_NAME || 'legal_ai_db'}`,
  QDRANT_HOST: process.env.QDRANT_HOST || 'localhost',
  QDRANT_PORT: process.env.QDRANT_PORT || '6333',
};

/**
 * Execute command with error handling
 */
function runCommand(command, description, options = {}) {
  log.step(description);
  try {
    const result = execSync(command, { 
      stdio: options.silent ? 'pipe' : 'inherit', 
      encoding: 'utf8',
      ...options 
    });
    log.success(`${description} - Complete`);
    return result;
  } catch (error) {
    log.error(`${description} - Failed: ${error.message}`);
    if (options.exitOnError !== false) {
      process.exit(1);
    }
    return null;
  }
}

/**
 * Check if PostgreSQL is running and accessible
 */
async function checkPostgreSQL() {
  log.step('Checking PostgreSQL connection...');
  try {
    const sql = postgres(config.DATABASE_URL, { max: 1 });
    await sql`SELECT version()`;
    await sql.end();
    log.success('PostgreSQL connection verified');
    return true;
  } catch (error) {
    log.error(`PostgreSQL connection failed: ${error.message}`);
    return false;
  }
}

/**
 * Check if pgvector extension is available
 */
async function checkPgVector() {
  log.step('Checking pgvector extension...');
  try {
    const sql = postgres(config.DATABASE_URL, { max: 1 });
    
    // Check if extension exists
    const result = await sql`
      SELECT * FROM pg_available_extensions WHERE name = 'vector'
    `;
    
    if (result.length === 0) {
      await sql.end();
      log.error('pgvector extension not available. Please install pgvector.');
      return false;
    }

    // Check if extension is installed
    const installed = await sql`
      SELECT * FROM pg_extension WHERE extname = 'vector'
    `;

    await sql.end();
    
    if (installed.length > 0) {
      log.success('pgvector extension is installed and available');
    } else {
      log.warning('pgvector extension available but not installed (will be installed during migration)');
    }
    return true;
  } catch (error) {
    log.error(`pgvector check failed: ${error.message}`);
    return false;
  }
}

/**
 * Check Qdrant availability
 */
async function checkQdrant() {
  log.step('Checking Qdrant connection...');
  try {
    const response = await fetch(`http://${config.QDRANT_HOST}:${config.QDRANT_PORT}/collections`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
    });
    
    if (response.ok) {
      log.success('Qdrant connection verified');
      return true;
    } else {
      log.warning(`Qdrant responded with status ${response.status}`);
      return false;
    }
  } catch (error) {
    log.warning(`Qdrant connection failed: ${error.message}`);
    log.info('Qdrant is optional - proceeding with PostgreSQL-only setup');
    return false;
  }
}

/**
 * Generate Drizzle migrations
 */
function generateMigrations() {
  log.step('Generating Drizzle migrations...');
  
  // Check if drizzle.config.ts exists
  if (!existsSync('drizzle.config.ts')) {
    log.error('drizzle.config.ts not found');
    return false;
  }

  try {
    runCommand('npx drizzle-kit generate', 'Generating database migrations');
    return true;
  } catch (error) {
    log.error(`Migration generation failed: ${error.message}`);
    return false;
  }
}

/**
 * Run migrations
 */
function runMigrations() {
  log.step('Running database migrations...');
  
  try {
    // First run the SQL migration for enhanced schema
    const migrationPath = path.join(__dirname, '..', 'src', 'lib', 'server', 'db', 'migrations', '002_enhanced_schema_with_qdrant.sql');
    
    if (existsSync(migrationPath)) {
      log.step('Running enhanced schema SQL migration...');
      runCommand(`psql "${config.DATABASE_URL}" -f "${migrationPath}"`, 'Applying enhanced schema migration');
    }

    // Then run Drizzle migrations
    runCommand('npx drizzle-kit push', 'Pushing schema changes to database');
    return true;
  } catch (error) {
    log.error(`Migration execution failed: ${error.message}`);
    return false;
  }
}

/**
 * Test database operations
 */
async function testDatabaseOperations() {
  log.step('Testing database operations...');
  
  try {
    const sql = postgres(config.DATABASE_URL, { 
      max: 1,
      types: {
        vector: {
          to: 1184,
          from: [1184],
          serialize: (x) => `[${x.join(',')}]`,
          parse: (x) => x.slice(1, -1).split(',').map(Number),
        },
      },
    });

    // Test basic table access
    const tables = await sql`
      SELECT table_name 
      FROM information_schema.tables 
      WHERE table_schema = 'public' 
      AND table_type = 'BASE TABLE'
    `;
    
    log.success(`Found ${tables.length} tables in database`);
    
    // Test vector operations if pgvector is available
    try {
      const vectorTest = await sql`SELECT '[1,2,3]'::vector(3) as test_vector`;
      log.success('Vector operations working correctly');
    } catch (vectorError) {
      log.warning('Vector operations not available - pgvector may not be installed');
    }

    await sql.end();
    return true;
  } catch (error) {
    log.error(`Database operation test failed: ${error.message}`);
    return false;
  }
}

/**
 * Initialize Qdrant collections
 */
async function initializeQdrantCollections() {
  log.step('Initializing Qdrant collections...');
  
  const collections = [
    {
      name: 'legal_documents',
      config: {
        vectors: {
          size: 384,
          distance: 'Cosine',
        },
        optimizers_config: {
          default_segment_number: 2,
          memmap_threshold: 20000,
          indexing_threshold: 20000,
        },
        hnsw_config: {
          m: 16,
          ef_construct: 64,
          full_scan_threshold: 10000,
        },
      },
    },
    {
      name: 'cases',
      config: {
        vectors: {
          size: 384,
          distance: 'Cosine',
        },
      },
    },
    {
      name: 'users',
      config: {
        vectors: {
          size: 384,
          distance: 'Cosine',
        },
      },
    },
  ];

  for (const collection of collections) {
    try {
      const response = await fetch(`http://${config.QDRANT_HOST}:${config.QDRANT_PORT}/collections/${collection.name}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(collection.config),
      });

      if (response.ok) {
        log.success(`Created Qdrant collection: ${collection.name}`);
      } else if (response.status === 409) {
        log.info(`Qdrant collection already exists: ${collection.name}`);
      } else {
        log.warning(`Failed to create Qdrant collection ${collection.name}: ${response.status}`);
      }
    } catch (error) {
      log.warning(`Error creating Qdrant collection ${collection.name}: ${error.message}`);
    }
  }

  return true;
}

/**
 * Create database connection test script
 */
function createConnectionTestScript() {
  log.step('Creating database connection test script...');
  
  const testScript = `#!/usr/bin/env node
import { createQdrantService } from '../src/lib/server/db/qdrant-integration.js';

async function testConnections() {
  console.log('🔄 Testing database connections...');
  
  try {
    const qdrantService = createQdrantService();
    const health = await qdrantService.healthCheck();
    
    console.log('\\n📊 Health Check Results:');
    console.log(\`PostgreSQL: \${health.postgresql ? '✅' : '❌'}\`);
    console.log(\`Qdrant: \${health.qdrant ? '✅' : '❌'}\`);
    console.log(\`Collections: \${health.collections.join(', ')}\`);
    console.log(\`Total Documents: \${health.syncStatus.totalDocuments}\`);
    console.log(\`Synced Documents: \${health.syncStatus.syncedDocuments}\`);
    console.log(\`Pending Syncs: \${health.syncStatus.pendingSyncs}\`);
    
    await qdrantService.close();
    console.log('\\n✅ Connection test completed successfully');
  } catch (error) {
    console.error('❌ Connection test failed:', error.message);
    process.exit(1);
  }
}

testConnections();
`;

  writeFileSync('scripts/test-db-connections.mjs', testScript);
  log.success('Created database connection test script: scripts/test-db-connections.mjs');
}

/**
 * Main setup function
 */
async function main() {
  console.log(`${colors.bright}🚀 Enhanced Database Setup for Legal AI Platform${colors.reset}\n`);
  
  // Step 1: Check prerequisites
  log.info('Step 1: Checking prerequisites...');
  const postgresOk = await checkPostgreSQL();
  if (!postgresOk) {
    log.error('PostgreSQL is not accessible. Please ensure PostgreSQL is running and DATABASE_URL is correct.');
    process.exit(1);
  }

  const pgvectorOk = await checkPgVector();
  const qdrantOk = await checkQdrant();

  // Step 2: Generate migrations
  log.info('\\nStep 2: Generating database migrations...');
  const migrationsGenerated = generateMigrations();
  if (!migrationsGenerated) {
    log.error('Failed to generate migrations');
    process.exit(1);
  }

  // Step 3: Run migrations
  log.info('\\nStep 3: Running database migrations...');
  const migrationsRun = runMigrations();
  if (!migrationsRun) {
    log.error('Failed to run migrations');
    process.exit(1);
  }

  // Step 4: Test database operations
  log.info('\\nStep 4: Testing database operations...');
  const dbTestOk = await testDatabaseOperations();
  if (!dbTestOk) {
    log.error('Database operations test failed');
    process.exit(1);
  }

  // Step 5: Initialize Qdrant (if available)
  if (qdrantOk) {
    log.info('\\nStep 5: Initializing Qdrant collections...');
    await initializeQdrantCollections();
  } else {
    log.info('\\nStep 5: Skipping Qdrant initialization (not available)');
  }

  // Step 6: Create test scripts
  log.info('\\nStep 6: Creating utility scripts...');
  createConnectionTestScript();

  // Final summary
  console.log(`\\n${colors.green}${colors.bright}✅ Database setup completed successfully!${colors.reset}\\n`);
  console.log('📋 Summary:');
  console.log(`   PostgreSQL: ${postgresOk ? '✅' : '❌'} Connected`);
  console.log(`   pgvector: ${pgvectorOk ? '✅' : '❌'} Available`);
  console.log(`   Qdrant: ${qdrantOk ? '✅' : '❌'} Connected`);
  console.log(`   Migrations: ✅ Applied`);
  console.log(`   Schema: ✅ Enhanced with vector support`);
  
  console.log('\\n🎯 Next steps:');
  console.log('   1. Run: npm run test-db-connections');
  console.log('   2. Run: npm run db:studio (to explore the database)');
  console.log('   3. Start your SvelteKit application');
  
  console.log(`\\n${colors.cyan}Happy coding! 🚀${colors.reset}`);
}

// Handle errors
process.on('uncaughtException', (error) => {
  log.error(`Uncaught exception: ${error.message}`);
  process.exit(1);
});

process.on('unhandledRejection', (reason, promise) => {
  log.error(`Unhandled rejection: ${reason}`);
  process.exit(1);
});

// Run main function
main().catch((error) => {
  log.error(`Setup failed: ${error.message}`);
  process.exit(1);
});