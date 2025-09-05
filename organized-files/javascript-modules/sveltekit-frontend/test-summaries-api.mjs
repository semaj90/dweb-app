#!/usr/bin/env node

/**
 * Test script for /api/summaries endpoint
 * Tests the comprehensive AI synthesis system
 */

import { execSync } from 'child_process';

console.log('🧪 Testing /api/summaries API endpoint...\n');

// Test data for the API
const testRequest = {
  type: "case",
  targetId: "test-case-123",
  depth: "comprehensive",
  includeRAG: true,
  includeUserActivity: true,
  enableStreaming: false,
  chunkSize: 2000,
  userId: "test-user"
};

// Function to test the API
async function testSummariesAPI() {
  try {
    console.log('📋 Test Request:');
    console.log(JSON.stringify(testRequest, null, 2));
    console.log('\n🚀 Starting SvelteKit dev server...\n');

    // Start the dev server in background
    const serverProcess = execSync('npm run dev &', { 
      cwd: process.cwd(),
      stdio: 'inherit',
      timeout: 5000 
    });

    // Wait for server to start
    console.log('⏱️  Waiting for server to initialize...');
    await new Promise(resolve => setTimeout(resolve, 10000));

    console.log('🌐 Testing API endpoint...\n');

    // Test with curl
    const curlCommand = `curl -X POST http://localhost:5173/api/summaries \\
      -H "Content-Type: application/json" \\
      -d '${JSON.stringify(testRequest)}' \\
      --max-time 30 \\
      -v`;

    console.log('📡 Executing curl command:');
    console.log(curlCommand);
    console.log('\n📨 API Response:');

    try {
      const response = execSync(curlCommand, { 
        encoding: 'utf8',
        timeout: 30000,
        stdio: 'pipe'
      });
      
      console.log('✅ Success! Raw Response:');
      console.log(response);
      
      // Try to parse JSON response
      try {
        const parsed = JSON.parse(response);
        console.log('\n🎯 Parsed Response:');
        console.log(JSON.stringify(parsed, null, 2));
        
        if (parsed.success && parsed.result) {
          console.log('\n✨ AI Synthesis Summary:');
          console.log(parsed.result.summary);
          
          if (parsed.result.sources) {
            console.log('\n📚 Sources:');
            parsed.result.sources.forEach((source, i) => {
              console.log(`${i + 1}. ${source.type}: ${Math.round(source.contribution * 100)}%`);
            });
          }
        }
        
      } catch (parseError) {
        console.log('⚠️  Response is not valid JSON, showing raw response above');
      }
      
    } catch (curlError) {
      console.error('❌ curl command failed:', curlError.message);
      console.log('\n🔍 Troubleshooting:');
      console.log('1. Make sure the dev server is running');
      console.log('2. Check if port 5173 is available');
      console.log('3. Verify the API endpoint exists');
      console.log('4. Check for authentication requirements');
    }

  } catch (error) {
    console.error('❌ Test failed:', error.message);
  } finally {
    console.log('\n🧹 Cleaning up...');
    try {
      // Kill any remaining dev server processes
      execSync('pkill -f "vite.*dev"', { stdio: 'ignore' });
    } catch (e) {
      // Ignore cleanup errors
    }
  }
}

// Health check function
function checkSystemHealth() {
  console.log('🏥 System Health Check:');
  
  const checks = [
    {
      name: 'Node.js Version',
      command: 'node --version',
      required: true
    },
    {
      name: 'npm Version', 
      command: 'npm --version',
      required: true
    },
    {
      name: 'Package.json exists',
      command: 'test -f package.json && echo "✅ Found" || echo "❌ Missing"',
      required: true
    },
    {
      name: 'SvelteKit Dependencies',
      command: 'npm list @sveltejs/kit --depth=0 2>/dev/null | grep @sveltejs/kit || echo "❌ Missing"',
      required: true
    },
    {
      name: 'Ollama Service (Optional)',
      command: 'curl -s http://localhost:11434/api/tags >/dev/null && echo "✅ Running" || echo "⚠️  Not available"',
      required: false
    }
  ];

  let allPassed = true;
  
  checks.forEach(check => {
    try {
      const result = execSync(check.command, { encoding: 'utf8', stdio: 'pipe' }).trim();
      console.log(`  ${check.name}: ${result}`);
    } catch (error) {
      console.log(`  ${check.name}: ❌ Failed`);
      if (check.required) {
        allPassed = false;
      }
    }
  });

  if (!allPassed) {
    console.log('\n❌ Some required checks failed. Please fix before testing.');
    process.exit(1);
  }
  
  console.log('\n✅ Health check passed!\n');
}

// Main execution
async function main() {
  checkSystemHealth();
  await testSummariesAPI();
}

main().catch(console.error);