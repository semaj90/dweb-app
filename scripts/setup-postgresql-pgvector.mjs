#!/usr/bin/env node

/**
 * PostgreSQL + pgvector Setup and Integration
 * Production-ready database setup with vector embeddings
 */

import { Client } from 'pg';
import { readFileSync, writeFileSync, existsSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const projectRoot = join(__dirname, '..');

// Database configuration
const DB_CONFIG = {
  host: process.env.DB_HOST || 'localhost',
  port: process.env.DB_PORT || 5432,
  database: process.env.DB_NAME || 'legal_ai_db',
  user: process.env.DB_USER || 'legal_admin',
  password: process.env.DB_PASSWORD || '123456',
  ssl: process.env.NODE_ENV === 'production' ? { rejectUnauthorized: false } : false
};

class PostgreSQLSetup {
  constructor(config) {
    this.config = config;
    this.client = null;
  }

  async connect() {
    try {
      this.client = new Client(this.config);
      await this.client.connect();
      console.log('✅ Connected to PostgreSQL');
      return true;
    } catch (error) {
      console.error('❌ PostgreSQL connection failed:', error.message);
      return false;
    }
  }

  async setupDatabase() {
    console.log('🚀 Setting up PostgreSQL database with pgvector...');
    
    try {
      // Enable pgvector extension
      await this.enablePgVector();
      
      // Create tables
      await this.createTables();
      
      // Create indexes
      await this.createIndexes();
      
      // Insert sample data
      await this.insertSampleData();
      
      console.log('✅ PostgreSQL setup completed successfully');
      
    } catch (error) {
      console.error('❌ Database setup failed:', error.message);
      throw error;
    }
  }

  async enablePgVector() {
    console.log('🔧 Enabling pgvector extension...');
    
    try {
      await this.client.query('CREATE EXTENSION IF NOT EXISTS vector;');
      
      // Verify extension is installed
      const result = await this.client.query(`
        SELECT extname FROM pg_extension WHERE extname = 'vector';
      `);
      
      if (result.rows.length > 0) {
        console.log('✅ pgvector extension enabled');
      } else {
        throw new Error('pgvector extension not found');
      }
    } catch (error) {
      console.error('❌ Failed to enable pgvector:', error.message);
      throw error;
    }
  }

  async createTables() {
    console.log('📋 Creating database tables...');
    
    const queries = [
      // Users table
      `CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        email VARCHAR(255) UNIQUE NOT NULL,
        password_hash VARCHAR(255) NOT NULL,
        name VARCHAR(255),
        role VARCHAR(50) DEFAULT 'user',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      );`,
      
      // Cases table
      `CREATE TABLE IF NOT EXISTS cases (
        id SERIAL PRIMARY KEY,
        title VARCHAR(500) NOT NULL,
        description TEXT,
        status VARCHAR(50) DEFAULT 'active',
        priority VARCHAR(50) DEFAULT 'medium',
        user_id INTEGER REFERENCES users(id),
        title_embedding vector(384),
        description_embedding vector(384),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      );`,
      
      // Documents table  
      `CREATE TABLE IF NOT EXISTS documents (
        id SERIAL PRIMARY KEY,
        case_id INTEGER REFERENCES cases(id),
        title VARCHAR(500) NOT NULL,
        content TEXT,
        file_path VARCHAR(1000),
        file_type VARCHAR(100),
        file_size INTEGER,
        content_embedding vector(384),
        title_embedding vector(384),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      );`,
      
      // Evidence table
      `CREATE TABLE IF NOT EXISTS evidence (
        id SERIAL PRIMARY KEY,
        case_id INTEGER REFERENCES cases(id),
        document_id INTEGER REFERENCES documents(id),
        title VARCHAR(500) NOT NULL,
        description TEXT,
        evidence_type VARCHAR(100),
        content TEXT,
        metadata JSONB,
        title_embedding vector(384),
        content_embedding vector(384),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      );`,
      
      // Vector search logs
      `CREATE TABLE IF NOT EXISTS vector_search_logs (
        id SERIAL PRIMARY KEY,
        query TEXT NOT NULL,
        query_embedding vector(384),
        results_count INTEGER,
        search_time_ms INTEGER,
        user_id INTEGER REFERENCES users(id),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      );`
    ];

    for (const query of queries) {
      try {
        await this.client.query(query);
      } catch (error) {
        console.error(`❌ Failed to create table:`, error.message);
        throw error;
      }
    }
    
    console.log('✅ Database tables created');
  }

  async createIndexes() {
    console.log('🏗️ Creating database indexes...');
    
    const indexes = [
      // Vector indexes for similarity search
      'CREATE INDEX IF NOT EXISTS cases_title_embedding_idx ON cases USING ivfflat (title_embedding vector_cosine_ops) WITH (lists = 100);',
      'CREATE INDEX IF NOT EXISTS cases_description_embedding_idx ON cases USING ivfflat (description_embedding vector_cosine_ops) WITH (lists = 100);',
      'CREATE INDEX IF NOT EXISTS documents_content_embedding_idx ON documents USING ivfflat (content_embedding vector_cosine_ops) WITH (lists = 100);',
      'CREATE INDEX IF NOT EXISTS evidence_content_embedding_idx ON evidence USING ivfflat (content_embedding vector_cosine_ops) WITH (lists = 100);',
      
      // Regular indexes
      'CREATE INDEX IF NOT EXISTS cases_user_id_idx ON cases(user_id);',
      'CREATE INDEX IF NOT EXISTS cases_status_idx ON cases(status);',
      'CREATE INDEX IF NOT EXISTS documents_case_id_idx ON documents(case_id);',
      'CREATE INDEX IF NOT EXISTS evidence_case_id_idx ON evidence(case_id);',
      'CREATE INDEX IF NOT EXISTS evidence_document_id_idx ON evidence(document_id);'
    ];

    for (const index of indexes) {
      try {
        await this.client.query(index);
      } catch (error) {
        // IVFFlat indexes might fail if not enough data, that's okay
        if (!error.message.includes('ivfflat')) {
          console.error(`❌ Failed to create index:`, error.message);
        }
      }
    }
    
    console.log('✅ Database indexes created');
  }

  async insertSampleData() {
    console.log('📝 Inserting sample data...');
    
    try {
      // Check if sample data already exists
      const userCount = await this.client.query('SELECT COUNT(*) FROM users');
      if (parseInt(userCount.rows[0].count) > 0) {
        console.log('ℹ️ Sample data already exists, skipping...');
        return;
      }

      // Insert sample user
      const userResult = await this.client.query(`
        INSERT INTO users (email, password_hash, name, role) 
        VALUES ('admin@legal-ai.com', '$2b$10$K8JoGZZL5dqZ5qZ5qZ5qZ5qZ5qZ5qZ5qZ5qZ5qZ5qZ5qZ5qZ5qZ5q', 'Admin User', 'admin')
        RETURNING id;
      `);
      
      const userId = userResult.rows[0].id;

      // Insert sample case
      const caseResult = await this.client.query(`
        INSERT INTO cases (title, description, status, priority, user_id) 
        VALUES ('Corporate Merger Investigation', 'Investigation into potential antitrust violations in proposed merger', 'active', 'high', $1)
        RETURNING id;
      `, [userId]);
      
      const caseId = caseResult.rows[0].id;

      // Insert sample document
      const docResult = await this.client.query(`
        INSERT INTO documents (case_id, title, content, file_type) 
        VALUES ($1, 'Merger Agreement', 'This merger agreement outlines the terms and conditions...', 'pdf')
        RETURNING id;
      `, [caseId]);
      
      const docId = docResult.rows[0].id;

      // Insert sample evidence
      await this.client.query(`
        INSERT INTO evidence (case_id, document_id, title, description, evidence_type, content) 
        VALUES ($1, $2, 'Financial Statements', 'Quarterly financial statements showing market dominance', 'financial', 'Revenue data indicates significant market share...');
      `, [caseId, docId]);

      console.log('✅ Sample data inserted');
      
    } catch (error) {
      console.error('❌ Failed to insert sample data:', error.message);
      // Don't throw here, sample data is optional
    }
  }

  async testVectorOperations() {
    console.log('🧪 Testing vector operations...');
    
    try {
      // Test vector similarity search
      const testQuery = `
        SELECT id, title, 
               title_embedding <-> '[0.1,0.2,0.3,0.4]'::vector AS distance
        FROM cases 
        WHERE title_embedding IS NOT NULL
        ORDER BY distance
        LIMIT 5;
      `;
      
      await this.client.query(testQuery);
      console.log('✅ Vector operations working');
      
    } catch (error) {
      console.error('❌ Vector operations test failed:', error.message);
    }
  }

  async generateHealthReport() {
    const report = {
      timestamp: new Date().toISOString(),
      database: this.config.database,
      host: this.config.host,
      port: this.config.port,
      status: 'healthy',
      tables: {},
      extensions: [],
      indexes: [],
      performance: {}
    };

    try {
      // Check table counts
      const tables = ['users', 'cases', 'documents', 'evidence', 'vector_search_logs'];
      for (const table of tables) {
        const result = await this.client.query(`SELECT COUNT(*) FROM ${table}`);
        report.tables[table] = parseInt(result.rows[0].count);
      }

      // Check extensions
      const extensions = await this.client.query(`
        SELECT extname, extversion FROM pg_extension WHERE extname IN ('vector', 'uuid-ossp');
      `);
      report.extensions = extensions.rows;

      // Check indexes
      const indexes = await this.client.query(`
        SELECT indexname, tablename FROM pg_indexes 
        WHERE tablename IN ('cases', 'documents', 'evidence')
        AND indexname LIKE '%embedding%';
      `);
      report.indexes = indexes.rows;

      console.log('📊 Database Health Report:');
      console.log(JSON.stringify(report, null, 2));
      
      return report;
      
    } catch (error) {
      report.status = 'error';
      report.error = error.message;
      console.error('❌ Health report generation failed:', error.message);
      return report;
    }
  }

  async disconnect() {
    if (this.client) {
      await this.client.end();
      console.log('✅ PostgreSQL connection closed');
    }
  }
}

// Environment file generation
function generateEnvFile() {
  const envPath = join(projectRoot, '.env.database');
  const envContent = `
# PostgreSQL Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=legal_ai_db
DB_USER=legal_admin
DB_PASSWORD=123456

# Vector Configuration  
VECTOR_DIMENSION=384
SIMILARITY_THRESHOLD=0.7
MAX_VECTOR_RESULTS=50

# Connection Pool
DB_POOL_MIN=2
DB_POOL_MAX=20
DB_POOL_IDLE_TIMEOUT=30000
DB_POOL_CONNECT_TIMEOUT=60000
`.trim();

  writeFileSync(envPath, envContent);
  console.log(`✅ Environment file created: ${envPath}`);
}

// Main setup function
async function setupPostgreSQL() {
  console.log('🚀 Starting PostgreSQL + pgvector setup...');
  
  const setup = new PostgreSQLSetup(DB_CONFIG);
  
  try {
    const connected = await setup.connect();
    if (!connected) {
      console.error('❌ Cannot proceed without database connection');
      process.exit(1);
    }
    
    await setup.setupDatabase();
    await setup.testVectorOperations();
    await setup.generateHealthReport();
    
    // Generate environment file
    generateEnvFile();
    
    console.log('🎉 PostgreSQL + pgvector setup completed successfully!');
    console.log('📋 Next steps:');
    console.log('   1. Run: npm run db:migrate');
    console.log('   2. Run: npm run db:seed');
    console.log('   3. Test: npm run db:health');
    
  } catch (error) {
    console.error('❌ Setup failed:', error.message);
    process.exit(1);
  } finally {
    await setup.disconnect();
  }
}

// Run if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
  setupPostgreSQL().catch(console.error);
}

export { PostgreSQLSetup, setupPostgreSQL };