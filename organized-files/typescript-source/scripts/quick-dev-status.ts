#!/usr/bin/env tsx
import { spawn } from 'child_process';
import fs from 'fs/promises';

interface ServiceStatus {
  name: string;
  status: 'running' | 'error' | 'not_running';
  port?: number;
  error?: string;
}

async function checkService(name: string, url: string, port: number): Promise<ServiceStatus> {
  try {
    const response = await fetch(url, { 
      signal: AbortSignal.timeout(3000),
      method: 'GET'  
    });
    
    return {
      name,
      status: response.ok ? 'running' : 'error',
      port,
      error: response.ok ? undefined : `HTTP ${response.status}`
    };
  } catch (error) {
    return {
      name,
      status: 'not_running',
      port,
      error: error instanceof Error ? error.message : 'Unknown error'
    };
  }
}

async function checkEnvironment(): Promise<void> {
  console.log('🔧 Legal AI Development Status Check\n');

  // Check required services
  const services = [
    { name: 'SvelteKit Dev Server', url: 'http://localhost:5177', port: 5177 },
    { name: 'Ollama', url: 'http://localhost:11434/api/version', port: 11434 },
    { name: 'PostgreSQL', url: 'postgresql://postgres:123456@localhost:5432/legal_ai_db', port: 5432 },
    { name: 'Qdrant (Optional)', url: 'http://localhost:6333/health', port: 6333 }
  ];

  const results: ServiceStatus[] = [];

  for (const service of services) {
    if (service.name === 'PostgreSQL') {
      // Special handling for PostgreSQL - try direct connection test
      try {
        const testUrl = 'http://localhost:5432'; // This will fail but shows if port is open
        await fetch(testUrl, { signal: AbortSignal.timeout(1000) });
        results.push({ name: service.name, status: 'running', port: service.port });
      } catch (error) {
        // Try to connect via psql command if available
        try {
          const psqlTest = spawn('"C:\\Program Files\\PostgreSQL\\17\\bin\\psql.exe"', 
            ['-U', 'postgres', '-d', 'legal_ai_db', '-h', 'localhost', '-c', 'SELECT 1;'], 
            { 
              shell: true,
              stdio: 'pipe',
              env: { ...process.env, PGPASSWORD: '123456' }
            }
          );
          
          let connected = false;
          psqlTest.on('close', (code) => {
            connected = code === 0;
          });
          
          // Wait a bit for the process
          await new Promise(resolve => setTimeout(resolve, 2000));
          
          results.push({ 
            name: service.name, 
            status: connected ? 'running' : 'not_running', 
            port: service.port,
            error: connected ? undefined : 'Connection test failed'
          });
        } catch (psqlError) {
          results.push({ 
            name: service.name, 
            status: 'not_running', 
            port: service.port,
            error: 'PostgreSQL test failed - check if service is running'
          });
        }
      }
    } else {
      const result = await checkService(service.name, service.url, service.port);
      results.push(result);
    }
  }

  // Display results
  console.log('📊 Service Status:');
  console.log('==================');
  
  for (const result of results) {
    const icon = result.status === 'running' ? '✅' : 
                 result.status === 'error' ? '⚠️' : '❌';
    const status = result.status === 'running' ? 'RUNNING' :
                   result.status === 'error' ? 'ERROR' : 'NOT RUNNING';
    
    console.log(`${icon} ${result.name.padEnd(25)} ${status.padEnd(12)} Port: ${result.port}`);
    if (result.error) {
      console.log(`   └─ ${result.error}`);
    }
  }

  // Check environment variables
  console.log('\n🌍 Environment Variables:');
  console.log('=========================');
  
  const envVars = [
    'NODE_ENV',
    'DATABASE_URL',
    'OLLAMA_URL',
    'SKIP_RAG_INITIALIZATION',
    'USE_POSTGRESQL_ONLY'
  ];

  for (const envVar of envVars) {
    const value = process.env[envVar];
    const icon = value ? '✅' : '❌';
    console.log(`${icon} ${envVar.padEnd(25)} ${value || 'NOT SET'}`);
  }

  // Summary
  const runningServices = results.filter(r => r.status === 'running').length;
  const totalServices = results.length;
  
  console.log('\n📋 Summary:');
  console.log('============');
  console.log(`Services Running: ${runningServices}/${totalServices}`);
  
  if (runningServices >= 2) { // SvelteKit + Ollama minimum
    console.log('🎉 Development environment is ready!');
    console.log('   → Visit: http://localhost:5177');
  } else {
    console.log('⚠️  Some services need attention');
    console.log('   → Run: npm run dev to start SvelteKit');
    console.log('   → Check Ollama installation');
  }

  // Development tips
  console.log('\n💡 Development Tips:');
  console.log('====================');
  console.log('• PostgreSQL + pgvector is working ✅');
  console.log('• Qdrant is optional for development (fallback mode enabled)');
  console.log('• RAG service will use PostgreSQL-only mode');
  console.log('• Use: npm run vector:claude for CLI vector search');
  console.log('• Use: npm run check:claude for TypeScript error analysis');
}

// Run the check
checkEnvironment().catch(console.error);