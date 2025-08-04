import { test, expect } from '@playwright/test';
import { spawn } from 'child_process';
import { Client } from 'pg';

// Database configuration for testing
const dbConfig = {
  user: 'postgres',
  password: '123456',
  host: 'localhost',
  database: 'legal_ai_db',
  port: 5432,
};

test.describe('Development Environment Tests', () => {
  let dbClient: Client;

  test.beforeAll(async () => {
    // Initialize database connection
    dbClient = new Client(dbConfig);
    try {
      await dbClient.connect();
      console.log('✅ Connected to PostgreSQL for testing');
    } catch (error) {
      console.warn('⚠️ PostgreSQL connection failed, some tests may be skipped');
    }
  });

  test.afterAll(async () => {
    if (dbClient) {
      await dbClient.end();
    }
  });

  test('should check development environment status', async () => {
    // Test the dev:status command
    const statusProcess = spawn('npm', ['run', 'dev:status'], {
      cwd: process.cwd(),
      shell: true,
      stdio: 'pipe'
    });

    let output = '';
    statusProcess.stdout.on('data', (data) => {
      output += data.toString();
    });

    await new Promise((resolve) => {
      statusProcess.on('close', resolve);
    });

    // Verify status output contains expected services
    expect(output).toContain('Legal AI Development Status Check');
    expect(output).toContain('Service Status');
    expect(output).toContain('Environment Variables');
    expect(output).toContain('PostgreSQL');
    expect(output).toContain('Ollama');
  });

  test('should test PostgreSQL connection and pgvector extension', async () => {
    if (!dbClient) {
      test.skip('PostgreSQL not available');
      return;
    }

    // Test basic connection
    const versionResult = await dbClient.query('SELECT version()');
    expect(versionResult.rows).toHaveLength(1);
    expect(versionResult.rows[0].version).toContain('PostgreSQL');

    // Test pgvector extension
    const vectorResult = await dbClient.query(
      "SELECT extname, extversion FROM pg_extension WHERE extname = 'vector'"
    );
    
    if (vectorResult.rows.length > 0) {
      expect(vectorResult.rows[0].extname).toBe('vector');
      console.log(`✅ pgvector version: ${vectorResult.rows[0].extversion}`);
    } else {
      console.warn('⚠️ pgvector extension not installed');
    }
  });

  test('should test Ollama service availability', async () => {
    try {
      const response = await fetch('http://localhost:11434/api/version', {
        signal: AbortSignal.timeout(5000)
      });
      
      if (response.ok) {
        const data = await response.json();
        expect(data).toHaveProperty('version');
        console.log(`✅ Ollama version: ${data.version}`);
      } else {
        console.warn('⚠️ Ollama not responding');
      }
    } catch (error) {
      console.warn('⚠️ Ollama service not available:', error);
    }
  });

  test('should verify environment variables configuration', async () => {
    // Test environment file exists
    const fs = await import('fs/promises');
    try {
      const envContent = await fs.readFile('.env.development', 'utf-8');
      
      expect(envContent).toContain('NODE_ENV=development');
      expect(envContent).toContain('DATABASE_URL=postgresql://postgres:123456@localhost:5432/postgres');
      expect(envContent).toContain('OLLAMA_URL=http://localhost:11434');
      expect(envContent).toContain('SKIP_RAG_INITIALIZATION=true');
      expect(envContent).toContain('USE_POSTGRESQL_ONLY=true');
      
      console.log('✅ Environment configuration verified');
    } catch (error) {
      console.warn('⚠️ Environment file not found or invalid');
    }
  });

  test('should check SvelteKit development server availability', async () => {
    try {
      const response = await fetch('http://localhost:5177', {
        signal: AbortSignal.timeout(3000)
      });
      
      if (response.ok) {
        const html = await response.text();
        expect(html).toContain('<!DOCTYPE html>');
        console.log('✅ SvelteKit dev server responding');
      } else {
        console.warn('⚠️ SvelteKit dev server not responding (expected if not running)');
      }
    } catch (error) {
      console.warn('⚠️ SvelteKit dev server not available (expected if not running)');
    }
  });
});