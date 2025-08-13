#!/usr/bin/env node

/**
 * YoRHa Legal AI - System Setup Orchestrator
 *
 * Automated setup and configuration for the entire Legal AI system:
 * - Environment validation and prerequisite installation
 * - Database setup with schema and sample data
 * - Service configuration and initial setup
 * - Development and production environment preparation
 *
 * @author YoRHa Legal AI Team
 * @version 2.0.0
 */

import 'zx/globals';
import chalk from 'chalk';
import ora from 'ora';
import inquirer from 'inquirer';
import { program } from 'commander';
import fetch from 'node-fetch';

// Setup configuration
const SETUP_CONFIG = {
  prerequisites: {
    // Force PowerShell commands to avoid Git Bash path/quoting issues on Windows
    nodejs: { command: 'powershell -NoProfile -Command node --version', minVersion: '18.0.0', required: true },
    postgresql: { path: 'C:\\Program Files\\PostgreSQL\\17\\bin\\psql.exe', required: true },
    redis: { path: './redis-windows/redis-server.exe', required: true },
    ollama: { command: 'powershell -NoProfile -Command ollama --version', required: true },
    go: { command: 'powershell -NoProfile -Command go version', required: false },
    git: { command: 'powershell -NoProfile -Command git --version', required: false }
  },

  services: {
    postgresql: {
      name: 'PostgreSQL + pgvector',
      setupSteps: ['createDatabase', 'installExtensions', 'createSchema', 'seedData']
    },
    redis: {
      name: 'Redis Cache',
      setupSteps: ['configureRedis', 'startService']
    },
    ollama: {
      name: 'Ollama LLM',
      setupSteps: ['pullModels', 'configureService']
    },
    go: {
      name: 'Go Microservice',
      setupSteps: ['buildService', 'configureEnvironment']
    },
    frontend: {
      name: 'SvelteKit Frontend',
      setupSteps: ['installDependencies', 'buildAssets', 'configureEnvironment']
    }
  },

  models: {
    required: ['llama3.1:8b', 'nomic-embed-text'],
    optional: ['gemma2:9b', 'llama3.2:3b']
  }
};

// Enhanced logging
const log = {
  timestamp: () => new Date().toISOString(),
  info: (msg) => console.log(`[${log.timestamp()}]`, chalk.blue('ℹ'), msg),
  success: (msg) => console.log(`[${log.timestamp()}]`, chalk.green('✓'), msg),
  error: (msg) => console.log(`[${log.timestamp()}]`, chalk.red('✗'), msg),
  warn: (msg) => console.log(`[${log.timestamp()}]`, chalk.yellow('⚠'), msg),
  debug: (msg) => process.env.DEBUG && console.log(`[${log.timestamp()}]`, chalk.gray('🔍'), msg),
  step: (msg) => console.log(chalk.cyan.bold(`\n🔧 ${msg}\n`))
};

// Prerequisite checking
async function checkPrerequisites() {
  log.step('Checking System Prerequisites');

  const results = {};
  const missing = [];

  for (const [name, config] of Object.entries(SETUP_CONFIG.prerequisites)) {
    const spinner = ora(`Checking ${name}...`).start();

    try {
      let checkResult = false;

      if (config.command) {
        const result = await $`${config.command}`.catch(() => ({ exitCode: 1 }));
        checkResult = result.exitCode === 0;

        if (checkResult && config.minVersion) {
          const version = extractVersion(result.stdout);
          checkResult = isVersionAtLeast(version, config.minVersion);
        }
      } else if (config.path) {
        checkResult = await fs.pathExists(config.path);
      }

      results[name] = { available: checkResult, required: config.required };

      if (checkResult) {
        spinner.succeed(`${name} available`);
      } else {
        spinner.fail(`${name} not found`);
        if (config.required) {
          missing.push(name);
        }
      }

    } catch (error) {
      results[name] = { available: false, required: config.required, error: error.message };
      spinner.fail(`${name} check failed`);
      if (config.required) {
        missing.push(name);
      }
    }
  }

  if (missing.length > 0) {
    log.error(`Missing required prerequisites: ${missing.join(', ')}`);
    console.log(chalk.yellow('\n📋 Installation Guide:'));

    missing.forEach(item => {
      switch (item) {
        case 'nodejs':
          console.log('  Node.js: Download from https://nodejs.org/ (v18+ required)');
          break;
        case 'postgresql':
          console.log('  PostgreSQL: Download from https://www.postgresql.org/download/windows/');
          break;
        case 'redis':
          console.log('  Redis: Extract redis-windows.zip to ./redis-windows/');
          break;
        case 'ollama':
          console.log('  Ollama: Download from https://ollama.com/download');
          break;
      }
    });

    const { continueAnyway } = await inquirer.prompt([{
      type: 'confirm',
      name: 'continueAnyway',
      message: 'Continue setup without missing prerequisites?',
      default: false
    }]);

    if (!continueAnyway) {
      process.exit(1);
    }
  }

  return results;
}

// Database setup
async function setupDatabase() {
  log.step('Setting up PostgreSQL Database');

  const steps = [
    {
      name: 'Create Database',
      execute: async () => {
        try {
          await $`powershell -NoProfile -Command "& 'C:\\Program Files\\PostgreSQL\\17\\bin\\createdb.exe' -U postgres -h localhost legal_ai_db"`;
          return { success: true, message: 'Database created' };
        } catch (error) {
          if (String(error.stderr || error.stdout || '').includes('already exists')) {
            return { success: true, message: 'Database already exists' };
          }
          return { success: false, error: error.message };
        }
      }
    },
    {
      name: 'Create User',
      execute: async () => {
        try {
          const createUserSQL = `
            DO \\$\\$
            BEGIN
              IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'legal_admin') THEN
                CREATE USER legal_admin WITH PASSWORD 'LegalAI2024!';
                GRANT ALL PRIVILEGES ON DATABASE legal_ai_db TO legal_admin;
              END IF;
            END
            \\$\\$;
          `;
          const ps = `& 'C:\\Program Files\\PostgreSQL\\17\\bin\\psql.exe' -U postgres -h localhost -d legal_ai_db -c \"${createUserSQL}\"`;
          await $`powershell -NoProfile -Command ${ps}`;
          return { success: true, message: 'User created and configured' };
        } catch (error) {
          return { success: false, error: error.message };
        }
      }
    },
    {
      name: 'Install pgvector Extension',
      execute: async () => {
        try {
          const ps = `& 'C:\\Program Files\\PostgreSQL\\17\\bin\\psql.exe' -U legal_admin -d legal_ai_db -h localhost -c \"CREATE EXTENSION IF NOT EXISTS vector;\"`;
          await $`powershell -NoProfile -Command ${ps}`;
          return { success: true, message: 'pgvector extension installed' };
        } catch (error) {
          return { success: false, error: error.message };
        }
      }
    },
    {
      name: 'Create Schema',
      execute: async () => {
        try {
          const schemaSQL = `
            -- Legal AI Database Schema
            CREATE TABLE IF NOT EXISTS legal_cases (
              id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
              title VARCHAR(500) NOT NULL,
              description TEXT,
              status VARCHAR(50) DEFAULT 'active',
              priority VARCHAR(20) DEFAULT 'medium',
              jurisdiction VARCHAR(100),
              created_by UUID,
              created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
              updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS legal_documents (
              id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
              case_id UUID REFERENCES legal_cases(id) ON DELETE CASCADE,
              title VARCHAR(500) NOT NULL,
              content TEXT NOT NULL,
              document_type VARCHAR(50) DEFAULT 'legal',
              file_path TEXT,
              embedding vector(384),
              created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
              processed_at TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS evidence (
              id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
              case_id UUID REFERENCES legal_cases(id) ON DELETE CASCADE,
              title VARCHAR(500) NOT NULL,
              description TEXT,
              evidence_type VARCHAR(50) NOT NULL,
              file_path TEXT,
              chain_of_custody JSONB,
              created_by UUID,
              created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS ai_interactions (
              id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
              case_id UUID REFERENCES legal_cases(id),
              user_prompt TEXT NOT NULL,
              ai_response TEXT NOT NULL,
              model_used VARCHAR(100),
              tokens_used INTEGER,
              created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            -- Create indexes for performance
            CREATE INDEX IF NOT EXISTS idx_legal_documents_case_id ON legal_documents(case_id);
            CREATE INDEX IF NOT EXISTS idx_legal_documents_embedding ON legal_documents USING ivfflat (embedding vector_cosine_ops);
            CREATE INDEX IF NOT EXISTS idx_evidence_case_id ON evidence(case_id);
            CREATE INDEX IF NOT EXISTS idx_ai_interactions_case_id ON ai_interactions(case_id);
          `;
          const ps = `& 'C:\\Program Files\\PostgreSQL\\17\\bin\\psql.exe' -U legal_admin -d legal_ai_db -h localhost -c \"${schemaSQL}\"`;
          await $`powershell -NoProfile -Command ${ps}`;
          return { success: true, message: 'Database schema created' };
        } catch (error) {
          return { success: false, error: error.message };
        }
      }
    },
    {
      name: 'Seed Sample Data',
      execute: async () => {
        try {
          const sampleDataSQL = `
            -- Insert sample legal case
            INSERT INTO legal_cases (title, description, status, priority, jurisdiction)
            VALUES
              ('Sample Legal Case 001', 'Initial setup and testing case for system validation', 'active', 'low', 'Test Jurisdiction')
            ON CONFLICT DO NOTHING;

            -- Insert sample document
            INSERT INTO legal_documents (case_id, title, content, document_type)
            SELECT
              lc.id,
              'Sample Legal Document',
              'This is a sample legal document for testing purposes. It contains standard legal language and formatting to validate the document processing pipeline.',
              'legal'
            FROM legal_cases lc
            WHERE lc.title = 'Sample Legal Case 001'
            ON CONFLICT DO NOTHING;
          `;
          const ps = `& 'C:\\Program Files\\PostgreSQL\\17\\bin\\psql.exe' -U legal_admin -d legal_ai_db -h localhost -c \"${sampleDataSQL}\"`;
          await $`powershell -NoProfile -Command ${ps}`;
          return { success: true, message: 'Sample data inserted' };
        } catch (error) {
          return { success: false, error: error.message };
        }
      }
    }
  ];

  for (const step of steps) {
    const spinner = ora(`${step.name}...`).start();
    const result = await step.execute();

    if (result.success) {
      spinner.succeed(`${step.name} - ${result.message}`);
    } else {
      spinner.fail(`${step.name} - ${result.error}`);
      log.warn(`Database setup step failed but continuing...`);
    }
  }
}

// Ollama model setup
async function setupOllamaModels() {
  log.step('Setting up Ollama LLM Models');

  // Check if Ollama is running
  const spinner = ora('Checking Ollama service...').start();
  try {
    const response = await fetch('http://localhost:11434/api/version', { timeout: 5000 });
    if (!response.ok) {
      throw new Error('Ollama not responding');
    }
    spinner.succeed('Ollama service is running');
  } catch (error) {
    spinner.warn('Ollama service not responding - attempting to start...');

    try {
      // Try to start Ollama
      $`start /B ollama serve`;
      await sleep(5000); // Wait for startup

      const response = await fetch('http://localhost:11434/api/version', { timeout: 10000 });
      if (response.ok) {
        log.success('Ollama service started successfully');
      } else {
        throw new Error('Failed to start Ollama');
      }
    } catch (startError) {
      log.error('Could not start Ollama service. Please start it manually: ollama serve');
      return;
    }
  }

  // Pull required models
  for (const model of SETUP_CONFIG.models.required) {
    const modelSpinner = ora(`Pulling required model: ${model}...`).start();

    try {
      await $`ollama pull ${model}`;
      modelSpinner.succeed(`Model ${model} installed`);
    } catch (error) {
      modelSpinner.fail(`Failed to install ${model}: ${error.message}`);
    }
  }

  // Ask about optional models
  const { installOptional } = await inquirer.prompt([{
    type: 'confirm',
    name: 'installOptional',
    message: 'Install optional models? (This will take additional time and disk space)',
    default: false
  }]);

  if (installOptional) {
    for (const model of SETUP_CONFIG.models.optional) {
      const modelSpinner = ora(`Pulling optional model: ${model}...`).start();

      try {
        await $`ollama pull ${model}`;
        modelSpinner.succeed(`Model ${model} installed`);
      } catch (error) {
        modelSpinner.warn(`Failed to install optional model ${model}`);
      }
    }
  }
}

// Go service setup
async function setupGoService() {
  log.step('Setting up Go Microservice');

  // Check if Go is available
  try {
    await $`go version`;
  } catch (error) {
    log.warn('Go compiler not available - using pre-built binary');

    if (!(await fs.pathExists('./legal-ai-server.exe'))) {
      log.error('No Go compiler and no pre-built binary found');
      return;
    }

    log.success('Using existing pre-built binary');
    return;
  }

  // Build the Go service
  const buildSpinner = ora('Building Go microservice...').start();

  try {
    // Set up environment for Windows build
    process.env.GOOS = 'windows';
    process.env.GOARCH = 'amd64';
    process.env.CGO_ENABLED = '1';

    await $`go build -ldflags "-s -w" -o legal-ai-server.exe ./go-microservice/enhanced-grpc-legal-server.go`;
    buildSpinner.succeed('Go microservice built successfully');
  } catch (error) {
    buildSpinner.fail(`Failed to build Go service: ${error.message}`);

    // Check if we have the source file
    if (!(await fs.pathExists('./go-microservice/enhanced-grpc-legal-server.go'))) {
      log.error('Go source file not found. Ensure the Go microservice source is available.');
      return;
    }

    log.warn('Build failed - check Go environment and dependencies');
  }
}

// Frontend setup
async function setupFrontend() {
  log.step('Setting up SvelteKit Frontend');

  const frontendDir = './sveltekit-frontend';

  if (!(await fs.pathExists(frontendDir))) {
    log.error('Frontend directory not found');
    return;
  }

  // Install dependencies
  const depsSpinner = ora('Installing frontend dependencies...').start();

  try {
    await $`cd ${frontendDir} && npm install`;
    depsSpinner.succeed('Frontend dependencies installed');
  } catch (error) {
    depsSpinner.fail(`Failed to install dependencies: ${error.message}`);
    return;
  }

  // Build for production (optional)
  const { buildProduction } = await inquirer.prompt([{
    type: 'confirm',
    name: 'buildProduction',
    message: 'Build frontend for production?',
    default: false
  }]);

  if (buildProduction) {
    const buildSpinner = ora('Building frontend for production...').start();

    try {
      await $`cd ${frontendDir} && npm run build`;
      buildSpinner.succeed('Frontend built for production');
    } catch (error) {
      buildSpinner.warn(`Build failed: ${error.message}`);
    }
  }
}

// Environment configuration
async function setupEnvironment() {
  log.step('Configuring Environment');

  const envConfig = {
    NODE_ENV: process.env.NODE_ENV || 'development',
    DATABASE_URL: 'postgresql://legal_admin:LegalAI2024!@localhost:5432/legal_ai_db',
    REDIS_URL: 'redis://localhost:6379',
    OLLAMA_URL: 'http://localhost:11434',
    QDRANT_URL: 'http://localhost:6333',
    PORT: '8080',
    FRONTEND_PORT: process.env.NODE_ENV === 'production' ? '3000' : '5173'
  };

  // Create .env file
  const envContent = Object.entries(envConfig)
    .map(([key, value]) => `${key}=${value}`)
    .join('\n');

  await fs.writeFile('.env', envContent);
  log.success('.env file created');

  // Create production.env if needed
  if (process.env.NODE_ENV === 'production') {
    const prodEnvContent = envContent.replace('development', 'production');
    await fs.writeFile('production.env', prodEnvContent);
    log.success('production.env file created');
  }
}

// Final validation
async function validateSetup() {
  log.step('Validating Setup');

  const validationSteps = [
    {
      name: 'Database Connection',
      test: async () => {
        const result = await $`"C:\\Program Files\\PostgreSQL\\17\\bin\\pg_isready.exe" -h localhost -p 5432`;
        return result.exitCode === 0;
      }
    },
    {
      name: 'Database Schema',
      test: async () => {
        const result = await $`"C:\\Program Files\\PostgreSQL\\17\\bin\\psql.exe" -U legal_admin -d legal_ai_db -h localhost -c "\\dt" -t`;
        return result.stdout.includes('legal_cases');
      }
    },
    {
      name: 'Ollama Models',
      test: async () => {
        try {
          const response = await fetch('http://localhost:11434/api/tags');
          const data = await response.json();
          return data.models && data.models.length > 0;
        } catch {
          return false;
        }
      }
    },
    {
      name: 'Go Service Binary',
      test: async () => {
        return await fs.pathExists('./legal-ai-server.exe');
      }
    },
    {
      name: 'Frontend Dependencies',
      test: async () => {
        return await fs.pathExists('./sveltekit-frontend/node_modules');
      }
    }
  ];

  let allValid = true;

  for (const step of validationSteps) {
    const spinner = ora(`Validating ${step.name}...`).start();

    try {
      const isValid = await step.test();

      if (isValid) {
        spinner.succeed(`${step.name} - OK`);
      } else {
        spinner.fail(`${step.name} - Failed`);
        allValid = false;
      }
    } catch (error) {
      spinner.fail(`${step.name} - Error: ${error.message}`);
      allValid = false;
    }
  }

  return allValid;
}

// Main setup function
async function main() {
  console.log(chalk.cyan.bold('🚀 YoRHa Legal AI - System Setup Orchestrator\n'));

  program
    .option('--skip-prereq', 'Skip prerequisite checking')
    .option('--skip-db', 'Skip database setup')
    .option('--skip-models', 'Skip Ollama model installation')
    .option('--skip-build', 'Skip building services')
    .option('--production', 'Setup for production environment')
    .option('--quick', 'Quick setup with minimal prompts')
    .parse();

  const options = program.opts();

  if (options.production) {
    process.env.NODE_ENV = 'production';
  }

  const startTime = Date.now();

  try {
    // Prerequisite checking
    if (!options.skipPrereq) {
      await checkPrerequisites();
    }

    // Interactive setup options
    let setupChoices = {
      database: true,
      models: true,
      goService: true,
      frontend: true,
      environment: true
    };

    if (!options.quick && !options.production) {
      const { components } = await inquirer.prompt([{
        type: 'checkbox',
        name: 'components',
        message: 'Select components to set up:',
        choices: [
          { name: 'PostgreSQL Database + Schema', value: 'database', checked: true },
          { name: 'Ollama LLM Models', value: 'models', checked: true },
          { name: 'Go Microservice', value: 'goService', checked: true },
          { name: 'SvelteKit Frontend', value: 'frontend', checked: true },
          { name: 'Environment Configuration', value: 'environment', checked: true }
        ]
      }]);

      setupChoices = {
        database: components.includes('database'),
        models: components.includes('models'),
        goService: components.includes('goService'),
        frontend: components.includes('frontend'),
        environment: components.includes('environment')
      };
    }

    // Run setup steps
    if (setupChoices.database && !options.skipDb) {
      await setupDatabase();
    }

    if (setupChoices.models && !options.skipModels) {
      await setupOllamaModels();
    }

    if (setupChoices.goService && !options.skipBuild) {
      await setupGoService();
    }

    if (setupChoices.frontend) {
      await setupFrontend();
    }

    if (setupChoices.environment) {
      await setupEnvironment();
    }

    // Final validation
    log.step('Final Validation');
    const isValid = await validateSetup();

    const duration = ((Date.now() - startTime) / 1000).toFixed(1);

    if (isValid) {
      log.success(`🎯 Setup completed successfully in ${duration}s`);

      console.log(chalk.cyan('\n🚀 Next Steps:'));
      console.log('  1. Start services: npm run dev');
      console.log('  2. Check status: npm run status');
      console.log('  3. Run health check: npm run health');
      console.log('  4. Access frontend: http://localhost:5173 (dev) or http://localhost:3000 (prod)');

    } else {
      log.warn(`⚠ Setup completed with warnings in ${duration}s`);
      console.log(chalk.yellow('\n💡 Some components may need manual configuration'));
      console.log('  Run: npm run health for detailed diagnostics');
    }

  } catch (error) {
    log.error(`Setup failed: ${error.message}`);
    process.exit(1);
  }
}

// Utility functions
function extractVersion(versionString) {
  const match = versionString.match(/(\d+\.\d+\.\d+)/);
  return match ? match[1] : '0.0.0';
}

function isVersionAtLeast(current, required) {
  const currentParts = current.split('.').map(Number);
  const requiredParts = required.split('.').map(Number);

  for (let i = 0; i < 3; i++) {
    if ((currentParts[i] || 0) > (requiredParts[i] || 0)) return true;
    if ((currentParts[i] || 0) < (requiredParts[i] || 0)) return false;
  }

  return true;
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// Handle CLI help
if (process.argv.includes('--help')) {
  console.log(`
YoRHa Legal AI System Setup Orchestrator

Usage: npm run setup [options]

Options:
  --skip-prereq    Skip prerequisite checking
  --skip-db        Skip database setup
  --skip-models    Skip Ollama model installation
  --skip-build     Skip building services
  --production     Setup for production environment
  --quick          Quick setup with minimal prompts
  --help           Show this help message

Examples:
  npm run setup                          # Interactive full setup
  npm run setup --quick                  # Quick setup with defaults
  npm run setup --production             # Production environment setup
  npm run setup --skip-models --skip-db  # Skip time-consuming steps

Prerequisites:
  - Node.js 18+ (required)
  - PostgreSQL 17+ (required)
  - Redis for Windows (required)
  - Ollama LLM runtime (required)
  - Go compiler (optional, for building from source)
  - Git (optional, for development)

Setup Components:
  - PostgreSQL database with pgvector extension
  - Database schema and sample data
  - Ollama LLM models (llama3.1:8b, nomic-embed-text)
  - Go microservice build
  - SvelteKit frontend dependencies
  - Environment configuration files
`);
  process.exit(0);
}

// Run the setup orchestrator
main();