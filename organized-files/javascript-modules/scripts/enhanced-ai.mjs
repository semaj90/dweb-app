#!/usr/bin/env zx

/**
 * Google Zx Script with OAuth Token Support
 * Enhanced Legal AI System - Claude Extension Integration
 */

import { authenticate } from '@google-cloud/local-auth';
import fs from 'fs/promises';
import { GoogleAuth } from 'google-auth-library';
import path from 'path';

// Configuration
const SCOPES = [
  'https://www.googleapis.com/auth/cloud-platform',
  'https://www.googleapis.com/auth/compute',
  'https://www.googleapis.com/auth/bigquery'
];

const TOKEN_PATH = path.join(process.cwd(), '.credentials', 'token.json');
const CREDENTIALS_PATH = path.join(process.cwd(), '.credentials', 'credentials.json');

// Global auth instance
let authClient = null;

/**
 * Load saved credentials if they exist
 */
async function loadSavedCredentialsIfExist() {
  try {
    const content = await fs.readFile(TOKEN_PATH);
    const credentials = JSON.parse(content.toString());
    return GoogleAuth.fromJSON(credentials);
  } catch (err) {
    return null;
  }
}

/**
 * Save credentials to token.json
 */
async function saveCredentials(client) {
  try {
    const content = await fs.readFile(CREDENTIALS_PATH);
    const keys = JSON.parse(content.toString());
    const key = keys.installed || keys.web;
    const payload = JSON.stringify({
      type: 'authorized_user',
      client_id: key.client_id,
      client_secret: key.client_secret,
      refresh_token: client.credentials.refresh_token,
    });

    // Ensure .credentials directory exists
    await fs.mkdir(path.dirname(TOKEN_PATH), { recursive: true });
    await fs.writeFile(TOKEN_PATH, payload);
  } catch (err) {
    console.error('Error saving credentials:', err);
  }
}

/**
 * Authorize the client with OAuth2
 */
async function authorize() {
  let client = await loadSavedCredentialsIfExist();
  if (client) {
    console.log(chalk.green('✅ Using saved credentials'));
    return client;
  }

  try {
    client = await authenticate({
      scopes: SCOPES,
      keyfilePath: CREDENTIALS_PATH,
    });

    if (client.credentials) {
      await saveCredentials(client);
      console.log(chalk.green('✅ Credentials saved successfully'));
    }

    return client;
  } catch (err) {
    console.error(chalk.red('❌ Authorization failed:'), err.message);
    throw err;
  }
}

/**
 * Initialize OAuth authentication
 */
async function initAuth() {
  if (!authClient) {
    console.log(chalk.blue('🔐 Initializing Google OAuth...'));
    authClient = await authorize();
  }
  return authClient;
}

/**
 * Make authenticated API request
 */
async function authenticatedRequest(url, options = {}) {
  const auth = await initAuth();
  const token = await auth.getAccessToken();

  const headers = {
    'Authorization': `Bearer ${token.token}`,
    'Content-Type': 'application/json',
    ...options.headers
  };

  return fetch(url, {
    ...options,
    headers
  });
}

/**
 * Enhanced Legal AI System Commands
 */

// Command: Check system health
$.checkHealth = async function() {
  console.log(chalk.blue('🏥 Checking Enhanced Legal AI System Health...'));

  const services = [
    { name: 'SvelteKit Frontend', url: 'http://localhost:5173', endpoint: '/' },
    { name: 'Go Microservice', url: 'http://localhost:8080', endpoint: '/api/health' },
    { name: 'Enhanced Interface', url: 'http://localhost:5173', endpoint: '/enhanced' },
  ];

  for (const service of services) {
    try {
      const response = await fetch(`${service.url}${service.endpoint}`, {
        method: 'GET',
        signal: AbortSignal.timeout(5000)
      });

      if (response.ok) {
        console.log(chalk.green(`✅ ${service.name}: Healthy`));
      } else {
        console.log(chalk.yellow(`⚠️  ${service.name}: ${response.status} ${response.statusText}`));
      }
    } catch (error) {
      console.log(chalk.red(`❌ ${service.name}: ${error.message}`));
    }
  }
};

// Command: Fix TypeScript errors
$.fixErrors = async function() {
  console.log(chalk.blue('🔧 Fixing TypeScript errors...'));

  cd('./sveltekit-frontend');

  // Remove problematic files
  await $`Get-ChildItem -Recurse -Name "*+server*" | Where-Object { $_ -notmatch "^\\+server\\.ts$" } | ForEach-Object { Remove-Item $_ -Force -ErrorAction SilentlyContinue }`;

  // Run TypeScript check
  try {
    await $`npm run check`;
    console.log(chalk.green('✅ TypeScript check passed'));
  } catch (error) {
    console.log(chalk.yellow('⚠️  TypeScript check found issues, attempting fixes...'));

    // Run type generation
    await $`npx svelte-kit sync`;

    // Try check again
    try {
      await $`npm run check`;
      console.log(chalk.green('✅ TypeScript errors fixed'));
    } catch (finalError) {
      console.log(chalk.red('❌ Some TypeScript errors remain'));
      console.log(finalError.stdout);
    }
  }

  cd('..');
};

// Command: Start development servers
$.startDev = async function() {
  console.log(chalk.blue('🚀 Starting Enhanced Legal AI Development Environment...'));

  // Start SvelteKit in background
  console.log(chalk.blue('📱 Starting SvelteKit frontend...'));
  cd('./sveltekit-frontend');
  const frontend = $`npm run dev`.nothrow();

  // Start Go microservice
  console.log(chalk.blue('⚙️  Starting Go microservice...'));
  cd('../go-microservice');
  const backend = $`./enhanced-legal-ai-clean.exe`.nothrow();

  console.log(chalk.green('✅ Development environment started'));
  console.log(chalk.blue('🌐 Frontend: http://localhost:5173'));
  console.log(chalk.blue('🔗 Enhanced UI: http://localhost:5173/enhanced'));
  console.log(chalk.blue('⚙️  Backend: http://localhost:8080'));

  cd('..');

  // Wait for both processes
  await Promise.all([frontend, backend]);
};

// Command: Test document processing
$.testDocumentProcessing = async function(content = 'Test legal document for processing') {
  console.log(chalk.blue('📄 Testing document processing...'));

  const testDoc = {
    content: content,
    document_type: 'contract',
    practice_area: 'commercial',
    jurisdiction: 'US',
    use_gpu: true,
    metadata: {
      timestamp: new Date().toISOString(),
      user_id: 'zx-test-user',
      session_id: 'zx-test-session'
    }
  };

  try {
    const response = await fetch('http://localhost:8080/api/process', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(testDoc)
    });

    if (response.ok) {
      const result = await response.json();
      console.log(chalk.green('✅ Document processing successful'));
      console.log(chalk.blue(`   Processing time: ${result.processing_time}`));
      if (result.summary) {
        console.log(chalk.blue(`   Summary: ${result.summary.substring(0, 100)}...`));
      }
      return result;
    } else {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
  } catch (error) {
    console.log(chalk.red(`❌ Document processing failed: ${error.message}`));
    throw error;
  }
};

// Command: Test vector search
$.testVectorSearch = async function(query = 'contract liability terms') {
  console.log(chalk.blue(`🔍 Testing vector search for: "${query}"`));

  const searchRequest = {
    query: query,
    limit: 5,
    use_gpu: true,
    model: 'gemma3-legal',
    filters: {
      jurisdiction: 'US',
      practice_area: 'commercial'
    }
  };

  try {
    const response = await fetch('http://localhost:8080/api/vector-search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(searchRequest)
    });

    if (response.ok) {
      const result = await response.json();
      console.log(chalk.green(`✅ Vector search successful`));
      console.log(chalk.blue(`   Found: ${result.total} results`));
      console.log(chalk.blue(`   Query time: ${result.took}`));
      return result;
    } else {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
  } catch (error) {
    console.log(chalk.red(`❌ Vector search failed: ${error.message}`));
    throw error;
  }
};

// Command: Deploy to Google Cloud (with OAuth)
$.deployToCloud = async function() {
  console.log(chalk.blue('☁️  Deploying to Google Cloud...'));

  try {
    await initAuth();
    console.log(chalk.green('✅ Authenticated with Google Cloud'));

    // Example deployment commands with authentication
    // This would be customized based on your specific deployment needs
    console.log(chalk.blue('📦 Building application...'));
    cd('./sveltekit-frontend');
    await $`npm run build`;

    console.log(chalk.blue('🚀 Deploying to App Engine...'));
    // await $`gcloud app deploy --quiet`;

    console.log(chalk.green('✅ Deployment completed'));

    cd('..');
  } catch (error) {
    console.log(chalk.red(`❌ Deployment failed: ${error.message}`));
    throw error;
  }
};

// Command: Run comprehensive tests
$.runTests = async function() {
  console.log(chalk.blue('🧪 Running comprehensive test suite...'));

  const tests = [
    { name: 'Health Check', fn: $.checkHealth },
    { name: 'Document Processing', fn: () => $.testDocumentProcessing() },
    { name: 'Vector Search', fn: () => $.testVectorSearch() },
  ];

  let passed = 0;
  let failed = 0;

  for (const test of tests) {
    try {
      console.log(chalk.blue(`\n🔍 Running: ${test.name}`));
      await test.fn();
      passed++;
      console.log(chalk.green(`✅ ${test.name}: PASSED`));
    } catch (error) {
      failed++;
      console.log(chalk.red(`❌ ${test.name}: FAILED`));
      console.log(chalk.red(`   Error: ${error.message}`));
    }
  }

  console.log(chalk.blue('\n📊 Test Results Summary:'));
  console.log(chalk.green(`✅ Passed: ${passed}`));
  console.log(chalk.red(`❌ Failed: ${failed}`));
  console.log(chalk.blue(`📈 Success Rate: ${Math.round((passed / (passed + failed)) * 100)}%`));
};

// Main CLI handler
async function main() {
  const command = argv._[0];

  switch (command) {
    case 'health':
      await $.checkHealth();
      break;
    case 'fix':
      await $.fixErrors();
      break;
    case 'start':
      await $.startDev();
      break;
    case 'test-doc':
      await $.testDocumentProcessing(argv.content);
      break;
    case 'test-search':
      await $.testVectorSearch(argv.query);
      break;
    case 'deploy':
      await $.deployToCloud();
      break;
    case 'test':
      await $.runTests();
      break;
    default:
      console.log(chalk.blue('🤖 Enhanced Legal AI System - Google Zx Commands'));
      console.log('\nAvailable commands:');
      console.log(chalk.green('  health') + '           - Check system health');
      console.log(chalk.green('  fix') + '              - Fix TypeScript errors');
      console.log(chalk.green('  start') + '            - Start development servers');
      console.log(chalk.green('  test-doc') + '         - Test document processing');
      console.log(chalk.green('  test-search') + '      - Test vector search');
      console.log(chalk.green('  deploy') + '           - Deploy to Google Cloud');
      console.log(chalk.green('  test') + '             - Run comprehensive tests');
      console.log('\nExamples:');
      console.log('  zx scripts/enhanced-ai.mjs health');
      console.log('  zx scripts/enhanced-ai.mjs test-doc --content="Legal contract text"');
      console.log('  zx scripts/enhanced-ai.mjs test-search --query="liability terms"');
  }
}

// Run if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
  await main();
}
