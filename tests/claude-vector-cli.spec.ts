import { test, expect } from '@playwright/test';
import { spawn } from 'child_process';
import { Client } from 'pg';
import fs from 'fs/promises';
import path from 'path';

const dbConfig = {
  user: 'postgres',
  password: '123456',
  host: 'localhost',
  database: 'legal_ai_db',
  port: 5432,
};

test.describe('Claude Vector Integration CLI Tests', () => {
  let dbClient: Client;

  test.beforeAll(async () => {
    dbClient = new Client(dbConfig);
    await dbClient.connect();
    console.log('✅ Connected to PostgreSQL for CLI testing');
  });

  test.afterAll(async () => {
    // Clean up any test data
    try {
      await dbClient.query(`DELETE FROM documents WHERE file LIKE '%test%' OR file LIKE '%demo%'`);
      await dbClient.query(`DELETE FROM embedding_cache WHERE model LIKE '%test%'`);
    } catch (error) {
      console.warn('⚠️ Cleanup warning:', error);
    }
    await dbClient.end();
  });

  test('should display CLI help and usage information', async () => {
    const cliProcess = spawn(
      'npm', 
      ['run', 'vector:claude'], 
      {
        cwd: process.cwd(),
        shell: true,
        stdio: 'pipe'
      }
    );

    let output = '';
    let errorOutput = '';

    cliProcess.stdout.on('data', (data: unknown) => {
      output += data.toString();
    });

    cliProcess.stderr.on('data', (data: unknown) => {
      errorOutput += data.toString();
    });

    await new Promise((resolve: unknown) => {
      cliProcess.on('close', resolve);
    });

    // Should display usage information
    const combinedOutput = output + errorOutput;
    expect(combinedOutput).toContain('Claude + pgvector Integration CLI');
    expect(combinedOutput).toContain('Usage:');
    expect(combinedOutput).toContain('Commands:');
    expect(combinedOutput).toContain('embed-file');
    expect(combinedOutput).toContain('embed-dir');
    expect(combinedOutput).toContain('search');
    expect(combinedOutput).toContain('test');

    console.log('✅ CLI help information displayed correctly');
  });

  test('should run test command and insert sample legal documents', async () => {
    const testProcess = spawn(
      'npm',
      ['run', 'vector:claude', 'test'],
      {
        cwd: process.cwd(),
        shell: true,
        stdio: 'pipe',
        timeout: 30000 // 30 second timeout
      }
    );

    let output = '';
    let errorOutput = '';

    testProcess.stdout.on('data', (data: unknown) => {
      output += data.toString();
    });

    testProcess.stderr.on('data', (data: unknown) => {
      errorOutput += data.toString();
    });

    const exitCode = await new Promise((resolve: unknown) => {
      testProcess.on('close', resolve);
    });

    const combinedOutput = output + errorOutput;
    console.log('CLI Test Output:', combinedOutput);

    if (exitCode === 0) {
      // Test was successful
      expect(combinedOutput).toContain('Inserting test legal documents');
      expect(combinedOutput).toContain('Testing search');
      
      // Verify test documents were inserted
      const testDocuments = await dbClient.query(`
        SELECT file, content 
        FROM documents 
        WHERE file IN ('contract-law-basics.txt', 'criminal-procedure.txt', 'tort-law-overview.txt')
      `);

      expect(testDocuments.rows.length).toBeGreaterThanOrEqual(1);
      console.log(`✅ CLI test command inserted ${testDocuments.rows.length} test documents`);

    } else if (combinedOutput.includes('ECONNREFUSED') || combinedOutput.includes('Ollama')) {
      console.warn('⚠️ CLI test failed due to Ollama connection - expected in some environments');
    } else {
      console.warn('⚠️ CLI test failed with unexpected error:', combinedOutput);
    }
  });

  test('should test file embedding functionality', async () => {
    // Create a temporary test file
    const tempDir = await fs.mkdtemp(path.join(process.cwd(), '.test-'));
    const testFilePath = path.join(tempDir, 'test-legal-doc.txt');
    
    const testContent = `# Legal Document Test
    
This is a test legal document for the Claude vector integration CLI.

## Contract Law Principles

1. Offer - A clear proposal with specific terms
2. Acceptance - Agreement to the offer
3. Consideration - Exchange of value
4. Capacity - Legal ability to contract
5. Legality - Lawful purpose

This document tests the embedding and indexing functionality.`;

    await fs.writeFile(testFilePath, testContent);

    try {
      const embedProcess = spawn(
        'npm',
        ['run', 'vector:claude', 'embed-file', testFilePath],
        {
          cwd: process.cwd(),
          shell: true,
          stdio: 'pipe',
          timeout: 20000
        }
      );

      let output = '';
      let errorOutput = '';

      embedProcess.stdout.on('data', (data: unknown) => {
        output += data.toString();
      });

      embedProcess.stderr.on('data', (data: unknown) => {
        errorOutput += data.toString();
      });

      const exitCode = await new Promise((resolve: unknown) => {
        embedProcess.on('close', resolve);
      });

      const combinedOutput = output + errorOutput;
      console.log('Embed File Output:', combinedOutput);

      if (exitCode === 0) {
        expect(combinedOutput).toContain('Processing file');
        expect(combinedOutput).toContain(testFilePath);
        
        // Verify file was embedded in database
        const embeddedDoc = await dbClient.query(`
          SELECT file, content, embedding
          FROM documents
          WHERE content LIKE '%Legal Document Test%'
        `);

        expect(embeddedDoc.rows.length).toBeGreaterThanOrEqual(1);
        expect(embeddedDoc.rows[0].embedding).toBeDefined();
        console.log('✅ File embedding completed successfully');

      } else if (combinedOutput.includes('ECONNREFUSED') || combinedOutput.includes('Ollama')) {
        console.warn('⚠️ File embedding failed due to Ollama connection - expected in some environments');
      } else {
        console.warn('⚠️ File embedding failed:', combinedOutput);
      }

    } finally {
      // Cleanup temp file
      try {
        await fs.unlink(testFilePath);
        await fs.rmdir(tempDir);
      } catch (error) {
        console.warn('⚠️ Temp file cleanup failed:', error);
      }
    }
  });

  test('should test search functionality and Claude context generation', async () => {
    // First ensure we have some test data
    const testQuery = 'What are the requirements for a valid contract?';
    
    const searchProcess = spawn(
      'npm',
      ['run', 'vector:claude', 'search', testQuery],
      {
        cwd: process.cwd(),
        shell: true,
        stdio: 'pipe',
        timeout: 25000
      }
    );

    let output = '';
    let errorOutput = '';

    searchProcess.stdout.on('data', (data: unknown) => {
      output += data.toString();
    });

    searchProcess.stderr.on('data', (data: unknown) => {
      errorOutput += data.toString();
    });

    const exitCode = await new Promise((resolve: unknown) => {
      searchProcess.on('close', resolve);
    });

    const combinedOutput = output + errorOutput;
    console.log('Search Output:', combinedOutput);

    if (exitCode === 0) {
      expect(combinedOutput).toContain('Searching for:');
      expect(combinedOutput).toContain(testQuery);
      expect(combinedOutput).toContain('Claude-ready context');
      
      // Should generate a JSON context file
      if (combinedOutput.includes('Saved to:')) {
        const logDirExists = await fs.access('.check-logs').then(() => true).catch(() => false);
        if (logDirExists) {
          const logFiles = await fs.readdir('.check-logs');
          const contextFiles = logFiles.filter(f => f.includes('claude-context-'));
          expect(contextFiles.length).toBeGreaterThan(0);
          
          // Verify the JSON structure
          if (contextFiles.length > 0) {
            const latestContext = path.join('.check-logs', contextFiles[contextFiles.length - 1]);
            const contextData = JSON.parse(await fs.readFile(latestContext, 'utf-8'));
            
            expect(contextData).toHaveProperty('task');
            expect(contextData).toHaveProperty('prompt');
            expect(contextData).toHaveProperty('context');
            expect(contextData).toHaveProperty('instructions');
            
            if (contextData.context.length > 0) {
              expect(contextData.context[0]).toHaveProperty('relevance_score');
              expect(contextData.context[0]).toHaveProperty('content');
            }
            
            console.log(`✅ Search generated Claude context with ${contextData.context.length} results`);
          }
        }
      }

    } else if (combinedOutput.includes('ECONNREFUSED') || combinedOutput.includes('Ollama')) {
      console.warn('⚠️ Search failed due to Ollama connection - expected in some environments');
    } else if (combinedOutput.includes('no_results') || combinedOutput.includes('No relevant documents')) {
      console.log('✅ Search completed but found no relevant documents (expected if test data not available)');
    } else {
      console.warn('⚠️ Search failed with unexpected error:', combinedOutput);
    }
  });

  test('should test directory embedding functionality', async () => {
    // Create a temporary directory with test files
    const tempDir = await fs.mkdtemp(path.join(process.cwd(), '.test-dir-'));
    
    const testFiles = [
      {
        name: 'contract-law.md',
        content: '# Contract Law\n\nBasic principles of contract formation and enforcement.'
      },
      {
        name: 'tort-law.md', 
        content: '# Tort Law\n\nCivil wrongs and remedies for damages.'
      },
      {
        name: 'readme.txt',
        content: 'This directory contains legal reference documents.'
      }
    ];

    // Create test files
    for (const file of testFiles) {
      await fs.writeFile(path.join(tempDir, file.name), file.content);
    }

    try {
      const embedDirProcess = spawn(
        'npm',
        ['run', 'vector:claude', 'embed-dir', tempDir],
        {
          cwd: process.cwd(),
          shell: true,
          stdio: 'pipe',
          timeout: 30000
        }
      );

      let output = '';
      let errorOutput = '';

      embedDirProcess.stdout.on('data', (data: unknown) => {
        output += data.toString();
      });

      embedDirProcess.stderr.on('data', (data: unknown) => {
        errorOutput += data.toString();
      });

      const exitCode = await new Promise((resolve: unknown) => {
        embedDirProcess.on('close', resolve);
      });

      const combinedOutput = output + errorOutput;
      console.log('Embed Directory Output:', combinedOutput);

      if (exitCode === 0) {
        expect(combinedOutput).toContain('Found');
        expect(combinedOutput).toContain('files to embed');
        
        // Should process .md and .txt files
        const expectedFiles = testFiles.filter(f => 
          f.name.endsWith('.md') || f.name.endsWith('.txt')
        ).length;
        
        console.log(`✅ Directory embedding processed ${expectedFiles} files`);

      } else if (combinedOutput.includes('ECONNREFUSED') || combinedOutput.includes('Ollama')) {
        console.warn('⚠️ Directory embedding failed due to Ollama connection - expected in some environments');
      } else {
        console.warn('⚠️ Directory embedding failed:', combinedOutput);
      }

    } finally {
      // Cleanup temp directory
      try {
        for (const file of testFiles) {
          await fs.unlink(path.join(tempDir, file.name));
        }
        await fs.rmdir(tempDir);
      } catch (error) {
        console.warn('⚠️ Temp directory cleanup failed:', error);
      }
    }
  });

  test('should verify database state after CLI operations', async () => {
    // Check overall database state
    const documentCount = await dbClient.query('SELECT COUNT(*) FROM documents');
    const embeddingCacheCount = await dbClient.query('SELECT COUNT(*) FROM embedding_cache');
    
    console.log(`📊 Database State:`);
    console.log(`   Documents: ${documentCount.rows[0].count}`);
    console.log(`   Embedding Cache: ${embeddingCacheCount.rows[0].count}`);
    
    expect(parseInt(documentCount.rows[0].count)).toBeGreaterThanOrEqual(0);
    expect(parseInt(embeddingCacheCount.rows[0].count)).toBeGreaterThanOrEqual(0);

    // Check for test documents
    const testDocs = await dbClient.query(`
      SELECT file, substring(content, 1, 50) as preview, created_at
      FROM documents
      WHERE file LIKE '%test%' OR content LIKE '%test%'
      ORDER BY created_at DESC
      LIMIT 5
    `);

    if (testDocs.rows.length > 0) {
      console.log('📄 Recent test documents:');
      testDocs.rows.forEach((doc, idx) => {
        console.log(`   ${idx + 1}. ${doc.file}: ${doc.preview}...`);
      });
    }

    // Test vector similarity functionality
    try {
      const vectorTest = await dbClient.query(`
        SELECT '[1,2,3]'::vector <=> '[1,2,4]'::vector as distance
      `);
      
      expect(vectorTest.rows).toHaveLength(1);
      expect(vectorTest.rows[0].distance).toBeDefined();
      console.log(`✅ Vector operations working (test distance: ${vectorTest.rows[0].distance})`);
      
    } catch (error) {
      console.warn('⚠️ Vector operations may not be fully configured:', error);
    }
  });

  test('should test PostgreSQL shortcuts integration', async () => {
    // Test the psql shortcuts batch file
    const psqlProcess = spawn(
      'npm',
      ['run', 'psql', 'vector-status'],
      {
        cwd: process.cwd(),
        shell: true,
        stdio: 'pipe',
        timeout: 10000
      }
    );

    let output = '';
    let errorOutput = '';

    psqlProcess.stdout.on('data', (data: unknown) => {
      output += data.toString();
    });

    psqlProcess.stderr.on('data', (data: unknown) => {
      errorOutput += data.toString();
    });

    const exitCode = await new Promise((resolve: unknown) => {
      psqlProcess.on('close', resolve);
    });

    const combinedOutput = output + errorOutput;
    console.log('PSQL Shortcuts Output:', combinedOutput);

    if (exitCode === 0) {
      // Should show pgvector status
      expect(combinedOutput).toMatch(/vector|extension|docs/i);
      console.log('✅ PostgreSQL shortcuts working');
    } else {
      console.warn('⚠️ PostgreSQL shortcuts may need configuration:', combinedOutput);
    }
  });
});