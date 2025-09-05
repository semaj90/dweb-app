// setup-environment.mjs
// Quick setup script for the development environment

import { exec } from 'child_process';
import { promisify } from 'util';
import fs from 'fs/promises';
import path from 'path';

const execAsync = promisify(exec);

const COLORS = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  cyan: '\x1b[36m'
};

async function log(message, type = 'info') {
  const prefix = {
    info: `${COLORS.cyan}ℹ️ `,
    success: `${COLORS.green}✅ `,
    error: `${COLORS.red}❌ `,
    warn: `${COLORS.yellow}⚠️  `
  };
  
  console.log(`${prefix[type] || ''}${message}${COLORS.reset}`);
}

async function checkCommand(command) {
  try {
    await execAsync(`${command} --version`);
    return true;
  } catch {
    return false;
  }
}

async function installNpmDependencies() {
  await log('Installing npm dependencies...', 'info');
  
  // Required dev dependencies for scripts
  const devDeps = [
    'chalk',
    'ora',
    'glob',
    'concurrently',
    'ws',
    'rimraf'
  ];
  
  try {
    await execAsync(`npm install --save-dev ${devDeps.join(' ')}`);
    await log('Dev dependencies installed', 'success');
  } catch (error) {
    await log(`Failed to install dev dependencies: ${error.message}`, 'error');
  }
  
  // Install main dependencies
  try {
    await execAsync('npm install');
    await log('All dependencies installed', 'success');
  } catch (error) {
    await log(`Failed to install dependencies: ${error.message}`, 'error');
  }
}

async function createEnvFile() {
  const envPath = path.join(process.cwd(), '.env.development');
  
  const envContent = `# Development Environment Configuration
NODE_ENV=development
VITE_LEGAL_AI_API=http://localhost:8084
VITE_OLLAMA_URL=http://localhost:11434
VITE_REDIS_URL=redis://localhost:6379
VITE_ENABLE_GPU=true
VITE_MAX_WORKERS=4
PUBLIC_SUPABASE_URL=
PUBLIC_SUPABASE_ANON_KEY=
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/legal_ai_db
`;

  try {
    await fs.access(envPath);
    await log('.env.development already exists', 'info');
  } catch {
    await fs.writeFile(envPath, envContent);
    await log('.env.development created', 'success');
  }
}

async function setupTypeScriptConfig() {
  const tsconfigPath = path.join(process.cwd(), 'tsconfig.json');
  
  try {
    const existing = JSON.parse(await fs.readFile(tsconfigPath, 'utf-8'));
    
    // Ensure incremental compilation is enabled
    if (!existing.compilerOptions.incremental) {
      existing.compilerOptions.incremental = true;
      await fs.writeFile(tsconfigPath, JSON.stringify(existing, null, 2));
      await log('TypeScript incremental compilation enabled', 'success');
    }
  } catch (error) {
    await log('Could not update tsconfig.json', 'warn');
  }
}

async function main() {
  console.clear();
  console.log(`${COLORS.blue}${COLORS.bright}╔════════════════════════════════════════════════╗${COLORS.reset}`);
  console.log(`${COLORS.blue}${COLORS.bright}║        DEVELOPMENT ENVIRONMENT SETUP           ║${COLORS.reset}`);
  console.log(`${COLORS.blue}${COLORS.bright}╚════════════════════════════════════════════════╝${COLORS.reset}`);
  console.log();
  
  // Check Node.js version
  await log('Checking Node.js version...', 'info');
  const { stdout: nodeVersion } = await execAsync('node --version');
  const majorVersion = parseInt(nodeVersion.match(/v(\d+)/)?.[1] || '0');
  
  if (majorVersion < 18) {
    await log(`Node.js 18+ required (current: ${nodeVersion.trim()})`, 'error');
    process.exit(1);
  }
  await log(`Node.js ${nodeVersion.trim()} detected`, 'success');
  
  // Check optional tools
  await log('Checking optional tools...', 'info');
  
  if (await checkCommand('go')) {
    await log('Go detected - API features available', 'success');
  } else {
    await log('Go not installed - API features will be limited', 'warn');
  }
  
  if (await checkCommand('redis-cli')) {
    await log('Redis detected - caching available', 'success');
  } else {
    await log('Redis not installed - will use memory cache', 'warn');
  }
  
  if (await checkCommand('ollama')) {
    await log('Ollama detected - AI features available', 'success');
  } else {
    await log('Ollama not installed - AI features will be limited', 'warn');
  }
  
  if (await checkCommand('nvidia-smi')) {
    await log('NVIDIA GPU detected - GPU acceleration available', 'success');
  } else {
    await log('No NVIDIA GPU detected - CPU mode will be used', 'warn');
  }
  
  // Install dependencies
  await installNpmDependencies();
  
  // Create environment file
  await createEnvFile();
  
  // Setup TypeScript config
  await setupTypeScriptConfig();
  
  console.log();
  console.log(`${COLORS.green}${COLORS.bright}════════════════════════════════════════════════${COLORS.reset}`);
  console.log(`${COLORS.green}${COLORS.bright}        Setup completed successfully!           ${COLORS.reset}`);
  console.log(`${COLORS.green}${COLORS.bright}════════════════════════════════════════════════${COLORS.reset}`);
  console.log();
  console.log('Quick start commands:');
  console.log(`  ${COLORS.cyan}npm run dev:full${COLORS.reset}     - Start all services`);
  console.log(`  ${COLORS.cyan}npm run dev:enhanced${COLORS.reset} - Start frontend + Go API`);
  console.log(`  ${COLORS.cyan}npm run dev:windows${COLORS.reset}  - Windows-optimized startup`);
  console.log(`  ${COLORS.cyan}npm run check:all${COLORS.reset}    - Run all checks`);
  console.log(`  ${COLORS.cyan}npm run test:health${COLORS.reset}  - Health check all services`);
  console.log();
}

main().catch(error => {
  console.error(`${COLORS.red}Setup failed: ${error.message}${COLORS.reset}`);
  process.exit(1);
});
