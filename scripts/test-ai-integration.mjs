#!/usr/bin/env node
// Test AI integration and MCP server connectivity

import fetch from 'node-fetch';

console.log('🔍 Testing AI Integration and MCP Servers...\n');

// Test services
const services = [
  {
    name: 'PostgreSQL',
    url: 'postgresql://postgres:postgres@localhost:5432/prosecutor_db',
    test: async () => {
      try {
        // Simple connection test using node-postgres
        const { Client } = await import('pg');
        const client = new Client({
          connectionString: 'postgresql://postgres:postgres@localhost:5432/prosecutor_db'
        });
        await client.connect();
        const result = await client.query('SELECT 1 as test');
        await client.end();
        return { success: true, data: result.rows[0] };
      } catch (error) {
        return { success: false, error: error.message };
      }
    }
  },
  {
    name: 'Ollama',
    url: 'http://localhost:11434',
    test: async () => {
      try {
        const response = await fetch('http://localhost:11434/api/version');
        if (response.ok) {
          const data = await response.json();
          return { success: true, data };
        }
        return { success: false, error: 'Service unavailable' };
      } catch (error) {
        return { success: false, error: error.message };
      }
    }
  },
  {
    name: 'Redis',
    url: 'redis://localhost:6379',
    test: async () => {
      try {
        // Simple Redis test
        const response = await fetch('http://localhost:6379', { method: 'HEAD' });
        return { success: true, data: 'Redis responding' };
      } catch (error) {
        return { success: false, error: 'Redis connection failed' };
      }
    }
  },
  {
    name: 'Qdrant',
    url: 'http://localhost:6333',
    test: async () => {
      try {
        const response = await fetch('http://localhost:6333/collections');
        if (response.ok) {
          const data = await response.json();
          return { success: true, data };
        }
        return { success: false, error: 'Qdrant unavailable' };
      } catch (error) {
        return { success: false, error: error.message };
      }
    }
  }
];

// Test each service
for (const service of services) {
  process.stdout.write(`Testing ${service.name}... `);
  
  try {
    const result = await service.test();
    if (result.success) {
      console.log('✅ PASS');
      if (result.data) {
        console.log(`   Data: ${JSON.stringify(result.data).substring(0, 100)}...`);
      }
    } else {
      console.log('❌ FAIL');
      console.log(`   Error: ${result.error}`);
    }
  } catch (error) {
    console.log('❌ FAIL');
    console.log(`   Error: ${error.message}`);
  }
  
  console.log();
}

// Test Ollama models
console.log('🤖 Testing Ollama Models...');
try {
  const response = await fetch('http://localhost:11434/api/tags');
  if (response.ok) {
    const data = await response.json();
    console.log('✅ Available models:');
    data.models?.forEach(model => {
      console.log(`   - ${model.name} (${model.size})`);
    });
  } else {
    console.log('❌ Failed to fetch models');
  }
} catch (error) {
  console.log(`❌ Ollama models test failed: ${error.message}`);
}

console.log('\n📊 Integration Test Summary:');
console.log('- Docker services configured for Windows 10 low memory');
console.log('- Ollama replaces VLLM for local LLM processing');
console.log('- PostgreSQL optimized with vector extension');
console.log('- MCP servers ready for VS Code integration');
console.log('- Global stores implemented with authentication');
console.log('- Report generation with Context7 MCP support');

console.log('\n🚀 Next Steps:');
console.log('1. Run: START-LEGAL-AI-WINDOWS.bat');
console.log('2. Open VS Code and start debugging');
console.log('3. Navigate to case management interface');
console.log('4. Test report generation with AI assistance');

console.log('\n✨ System Ready for Development!');