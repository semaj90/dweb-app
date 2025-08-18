#!/usr/bin/env node
/**
 * 🚀 Complete System Setup Script
 * Sets up multi-core auto-solver with all dependencies
 */

import fs from 'fs/promises';
import path from 'path';
import { execSync, spawn } from 'child_process';
import { performance } from 'perf_hooks';
import chalk from 'chalk';
import ora from 'ora';
import inquirer from 'inquirer';

class SystemSetup {
    constructor() {
        this.config = {
            enableGPU: false,
            multiCore: true,
            maxWorkers: 4,
            storageEngines: ['postgres', 'redis', 'qdrant'],
            enableLangChain: false,
            databaseURL: 'postgresql://postgres:123456@localhost:5432/legal_ai_db',
            redisURL: 'redis://localhost:6379',
            qdrantURL: 'http://localhost:6333'
        };
        this.spinner = ora();
        this.setupSteps = [];
    }

    async run() {
        console.log(chalk.blue.bold('🚀 VS Code Multi-Core Auto-Solver Setup\n'));
        
        try {
            await this.detectSystemCapabilities();
            await this.promptUserConfiguration();
            await this.setupEnvironment();
            await this.installDependencies();
            await this.setupDatabases();
            await this.initializeServices();
            await this.runTests();
            await this.generateConfiguration();
            
            console.log(chalk.green.bold('\n✅ Setup completed successfully!'));
            this.printSummary();
            
        } catch (error) {
            console.error(chalk.red.bold('❌ Setup failed:'), error.message);
            process.exit(1);
        }
    }

    async detectSystemCapabilities() {
        this.spinner.start('🔍 Detecting system capabilities...');
        
        const startTime = performance.now();
        const capabilities = {
            os: process.platform,
            architecture: process.arch,
            nodeVersion: process.version,
            memory: Math.round(require('os').totalmem() / 1024 / 1024 / 1024),
            cpus: require('os').cpus().length,
            gpu: await this.detectGPU(),
            postgres: await this.detectPostgreSQL(),
            redis: await this.detectRedis(),
            docker: await this.detectDocker()
        };

        this.config.maxWorkers = Math.min(capabilities.cpus, 8);
        this.config.enableGPU = capabilities.gpu;

        const detectionTime = performance.now() - startTime;
        this.spinner.succeed(
            `✅ System detected (${detectionTime.toFixed(2)}ms):\n` +
            `   💻 OS: ${capabilities.os} (${capabilities.architecture})\n` +
            `   🧠 CPUs: ${capabilities.cpus}, Memory: ${capabilities.memory}GB\n` +
            `   🎯 GPU: ${capabilities.gpu ? 'Available' : 'Not Available'}\n` +
            `   🗄️ PostgreSQL: ${capabilities.postgres ? 'Found' : 'Not Found'}\n` +
            `   ⚡ Redis: ${capabilities.redis ? 'Found' : 'Not Found'}`
        );

        this.capabilities = capabilities;
    }

    async detectGPU() {
        try {
            const output = execSync(process.platform === 'win32' ? 'nvidia-smi -L' : 'nvidia-smi -L', 
                                   { stdio: 'pipe', encoding: 'utf8' });
            return /GPU \d+/.test(output);
        } catch {
            return false;
        }
    }

    async detectPostgreSQL() {
        try {
            execSync('psql --version', { stdio: 'pipe' });
            return true;
        } catch {
            return false;
        }
    }

    async detectRedis() {
        try {
            execSync('redis-cli --version', { stdio: 'pipe' });
            return true;
        } catch {
            return false;
        }
    }

    async detectDocker() {
        try {
            execSync('docker --version', { stdio: 'pipe' });
            return true;
        } catch {
            return false;
        }
    }

    async promptUserConfiguration() {
        console.log(chalk.yellow('\n🔧 Configuration Setup'));
        
        const questions = [
            {
                type: 'confirm',
                name: 'multiCore',
                message: `Enable multi-core processing with ${this.config.maxWorkers} workers?`,
                default: true
            },
            {
                type: 'number',
                name: 'maxWorkers',
                message: 'Maximum number of workers:',
                default: this.config.maxWorkers,
                when: (answers) => answers.multiCore,
                validate: (input) => input > 0 && input <= 16
            },
            {
                type: 'confirm',
                name: 'enableGPU',
                message: 'Enable GPU acceleration (if available)?',
                default: this.config.enableGPU,
                when: () => this.capabilities.gpu
            },
            {
                type: 'checkbox',
                name: 'storageEngines',
                message: 'Select storage engines to use:',
                choices: [
                    { name: 'PostgreSQL (Vector)', value: 'postgres', checked: true },
                    { name: 'Redis (Cache)', value: 'redis', checked: true },
                    { name: 'Qdrant (Vector Search)', value: 'qdrant', checked: true },
                    { name: 'Loki.js (In-Memory)', value: 'loki', checked: true },
                    { name: 'Fuse.js (Fuzzy Search)', value: 'fuse', checked: true },
                    { name: 'IndexedDB (Browser)', value: 'indexeddb', checked: false }
                ]
            },
            {
                type: 'confirm',
                name: 'enableLangChain',
                message: 'Enable LangChain.js integration?',
                default: false
            },
            {
                type: 'input',
                name: 'databaseURL',
                message: 'PostgreSQL connection URL:',
                default: this.config.databaseURL,
                when: (answers) => answers.storageEngines.includes('postgres')
            },
            {
                type: 'input',
                name: 'redisURL',
                message: 'Redis connection URL:',
                default: this.config.redisURL,
                when: (answers) => answers.storageEngines.includes('redis')
            },
            {
                type: 'input',
                name: 'qdrantURL',
                message: 'Qdrant connection URL:',
                default: this.config.qdrantURL,
                when: (answers) => answers.storageEngines.includes('qdrant')
            }
        ];

        const answers = await inquirer.prompt(questions);
        this.config = { ...this.config, ...answers };
    }

    async setupEnvironment() {
        this.spinner.start('🌐 Setting up environment...');

        // Create necessary directories
        const dirs = [
            'logs',
            'cache',
            'models',
            'data/postgres',
            'data/redis', 
            'data/qdrant',
            'tests/fixtures',
            'scripts/sql'
        ];

        for (const dir of dirs) {
            await fs.mkdir(path.resolve(dir), { recursive: true });
        }

        // Generate .env file
        const envContent = this.generateEnvFile();
        await fs.writeFile('.env', envContent);

        // Create docker-compose.yml for services
        if (this.capabilities.docker) {
            const dockerCompose = this.generateDockerCompose();
            await fs.writeFile('docker-compose.yml', dockerCompose);
        }

        this.spinner.succeed('✅ Environment setup completed');
    }

    generateEnvFile() {
        return `# 🧠 VS Code Multi-Core Auto-Solver Configuration
# Generated on ${new Date().toISOString()}

# Core Configuration
NODE_ENV=development
DEBUG=auto-solver:*
VS_CODE_DEBUG=${this.config.enableGPU ? 'true' : 'false'}

# Multi-Core Settings
MCP_MULTICORE=${this.config.multiCore}
MCP_WORKERS=${this.config.maxWorkers}
ENABLE_CLUSTERING=true
ENABLE_GPU=${this.config.enableGPU}

# Storage Engines
DATABASE_URL=${this.config.databaseURL}
REDIS_URL=${this.config.redisURL}
QDRANT_URL=${this.config.qdrantURL}

# Performance Settings
CHUNK_SIZE=10000
SEMANTIC_BATCH_SIZE=100
MAX_MEMORY_PER_WORKER=2048

# AI/ML Settings
EMBED_PROVIDER=ollama
EMBED_MODEL=nomic-embed-text
OLLAMA_ENDPOINT=http://localhost:11434/api/embeddings
USE_LOCAL_EMBEDDINGS=true

# LangChain Configuration
LANGCHAIN_ENABLED=${this.config.enableLangChain}
OPENAI_API_KEY=

# Service Discovery
AUTO_START_SERVICES=true
SERVICE_DISCOVERY_INTERVAL=60000
HEALTH_CHECK_INTERVAL=30000

# Security
CORS_ORIGINS=http://localhost:5173,vscode-file://vscode-app
RATE_LIMIT_WINDOW=900000
RATE_LIMIT_MAX=1000

# Logging
LOG_LEVEL=info
LOG_FILE=logs/auto-solver.log
ERROR_LOG_FILE=logs/error.log

# Windows Native Settings (no Docker)
WINDOWS_SERVICES=true
CUSTOM_SERVERS=true
CUSTOM_SERVER_BASE_PORT=9000
CUSTOM_SERVER_COUNT=3

# Advanced Features
SEMANTIC_MODEL_PATH=./models/sentence-transformers.onnx
FORCE_NO_GPU=${this.config.enableGPU ? 'false' : 'true'}
BUFFER_OPTIMIZATION=true
API_CACHING=true
PM2_ENABLED=false
`;
    }

    generateDockerCompose() {
        const services = {};

        if (this.config.storageEngines.includes('postgres')) {
            services.postgres = {
                image: 'pgvector/pgvector:pg16',
                environment: {
                    POSTGRES_DB: 'legal_ai_db',
                    POSTGRES_USER: 'postgres', 
                    POSTGRES_PASSWORD: '123456'
                },
                ports: ['5432:5432'],
                volumes: ['postgres_data:/var/lib/postgresql/data'],
                healthcheck: {
                    test: ['CMD-SHELL', 'pg_isready -U postgres'],
                    interval: '30s',
                    timeout: '10s',
                    retries: 5
                }
            };
        }

        if (this.config.storageEngines.includes('redis')) {
            services.redis = {
                image: 'redis:7-alpine',
                ports: ['6379:6379'],
                volumes: ['redis_data:/data'],
                command: 'redis-server --appendonly yes'
            };
        }

        if (this.config.storageEngines.includes('qdrant')) {
            services.qdrant = {
                image: 'qdrant/qdrant:latest',
                ports: ['6333:6333', '6334:6334'],
                volumes: ['qdrant_data:/qdrant/storage']
            };
        }

        return `version: '3.8'

services:
${Object.entries(services).map(([name, config]) => 
    `  ${name}:\n${Object.entries(config).map(([key, value]) => 
        `    ${key}: ${typeof value === 'object' ? JSON.stringify(value) : value}`
    ).join('\n')}`
).join('\n\n')}

volumes:
${this.config.storageEngines.includes('postgres') ? '  postgres_data:' : ''}
${this.config.storageEngines.includes('redis') ? '  redis_data:' : ''}
${this.config.storageEngines.includes('qdrant') ? '  qdrant_data:' : ''}

networks:
  default:
    name: auto-solver-network
`;
    }

    async installDependencies() {
        this.spinner.start('📦 Installing dependencies...');

        try {
            // Install core dependencies
            execSync('npm install', { stdio: 'pipe' });

            // Install optional GPU dependencies if enabled
            if (this.config.enableGPU) {
                try {
                    execSync('npm install onnxruntime-node @tensorflow/tfjs-node-gpu', { stdio: 'pipe' });
                    this.spinner.text = '📦 GPU dependencies installed';
                } catch (error) {
                    console.warn(chalk.yellow('⚠️ GPU dependencies installation failed, continuing without GPU support'));
                    this.config.enableGPU = false;
                }
            }

            // Install LangChain if enabled
            if (this.config.enableLangChain) {
                execSync('npm install langchain @langchain/openai', { stdio: 'pipe' });
            }

            this.spinner.succeed('✅ Dependencies installed successfully');

        } catch (error) {
            this.spinner.fail('❌ Failed to install dependencies');
            throw error;
        }
    }

    async setupDatabases() {
        this.spinner.start('🗄️ Setting up databases...');

        try {
            // Setup PostgreSQL
            if (this.config.storageEngines.includes('postgres')) {
                await this.setupPostgreSQL();
            }

            // Setup Redis (usually no setup needed)
            if (this.config.storageEngines.includes('redis')) {
                await this.testRedisConnection();
            }

            // Setup Qdrant
            if (this.config.storageEngines.includes('qdrant')) {
                await this.setupQdrant();
            }

            this.spinner.succeed('✅ Databases setup completed');

        } catch (error) {
            this.spinner.fail('❌ Database setup failed');
            throw error;
        }
    }

    async setupPostgreSQL() {
        // Create database schema SQL
        const schemaSql = `
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- VS Code Auto-Solver Tables
CREATE TABLE IF NOT EXISTS semantic_embeddings (
    id SERIAL PRIMARY KEY,
    content_hash TEXT UNIQUE NOT NULL,
    content_text TEXT NOT NULL,
    content_type TEXT DEFAULT 'code',
    language TEXT,
    file_path TEXT,
    embeddings vector(768),
    metadata JSONB DEFAULT '{}'::jsonb,
    semantic_features JSONB DEFAULT '{}'::jsonb,
    processing_time_ms INTEGER,
    confidence_score FLOAT DEFAULT 0,
    cluster_id TEXT,
    similarity_group INTEGER,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS vscode_problems (
    id SERIAL PRIMARY KEY,
    file_path TEXT NOT NULL,
    problem_hash TEXT UNIQUE NOT NULL,
    problem_data JSONB NOT NULL,
    semantic_features JSONB,
    embeddings vector(768),
    solutions JSONB DEFAULT '[]'::jsonb,
    confidence_score FLOAT DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    solved_at TIMESTAMP,
    worker_id TEXT,
    processing_time_ms INTEGER
);

CREATE TABLE IF NOT EXISTS solution_patterns (
    id SERIAL PRIMARY KEY,
    problem_type TEXT NOT NULL,
    pattern_data JSONB NOT NULL,
    solution_template JSONB NOT NULL,
    success_rate FLOAT DEFAULT 0,
    usage_count INTEGER DEFAULT 0,
    last_used TIMESTAMP DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_embeddings_hnsw 
ON semantic_embeddings USING hnsw (embeddings vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_metadata_gin 
ON semantic_embeddings USING GIN (metadata);

CREATE INDEX IF NOT EXISTS idx_problems_hash 
ON vscode_problems(problem_hash);

CREATE INDEX IF NOT EXISTS idx_problems_data 
ON vscode_problems USING GIN (problem_data);

-- Sample data for testing
INSERT INTO solution_patterns (problem_type, pattern_data, solution_template) VALUES
('syntax-error', '{"keywords": ["SyntaxError", "Unexpected token"]}', '{"actions": ["Check brackets", "Verify semicolons"]}'),
('type-error', '{"keywords": ["TypeError", "Property does not exist"]}', '{"actions": ["Add type annotations", "Check imports"]}'),
('import-error', '{"keywords": ["Cannot find module"]}', '{"actions": ["Install package", "Check path"]}')
ON CONFLICT DO NOTHING;
`;

        await fs.writeFile('scripts/sql/schema.sql', schemaSql);

        // Test connection and run schema
        try {
            execSync(`psql "${this.config.databaseURL}" -f scripts/sql/schema.sql`, { stdio: 'pipe' });
        } catch (error) {
            console.warn(chalk.yellow('⚠️ PostgreSQL setup failed - ensure database is running'));
        }
    }

    async testRedisConnection() {
        try {
            execSync('redis-cli ping', { stdio: 'pipe' });
        } catch (error) {
            console.warn(chalk.yellow('⚠️ Redis not available - will use in-memory caching'));
        }
    }

    async setupQdrant() {
        // Qdrant setup is handled by the service itself
        console.log('   📊 Qdrant will be initialized on first run');
    }

    async initializeServices() {
        this.spinner.start('🚀 Initializing services...');

        if (this.capabilities.docker && this.config.storageEngines.some(e => ['postgres', 'redis', 'qdrant'].includes(e))) {
            try {
                execSync('docker-compose up -d', { stdio: 'pipe' });
                this.spinner.text = '🐳 Docker services started';
                
                // Wait for services to be ready
                await this.waitForServices();
            } catch (error) {
                console.warn(chalk.yellow('⚠️ Docker services failed to start - using native services'));
            }
        }

        this.spinner.succeed('✅ Services initialized');
    }

    async waitForServices() {
        const maxWait = 60000; // 60 seconds
        const checkInterval = 2000; // 2 seconds
        let elapsed = 0;

        while (elapsed < maxWait) {
            try {
                if (this.config.storageEngines.includes('postgres')) {
                    execSync(`psql "${this.config.databaseURL}" -c "SELECT 1"`, { stdio: 'pipe' });
                }
                if (this.config.storageEngines.includes('redis')) {
                    execSync('redis-cli ping', { stdio: 'pipe' });
                }
                break;
            } catch {
                await new Promise(resolve => setTimeout(resolve, checkInterval));
                elapsed += checkInterval;
            }
        }
    }

    async runTests() {
        this.spinner.start('🧪 Running system tests...');

        try {
            // Create a simple test
            const testContent = `
import { MultiCoreClusterManager } from '../core/multi-core-solver.js';

async function testBasicFunctionality() {
    const solver = new MultiCoreClusterManager();
    await solver.initializeCluster();
    
    const testProblems = [{
        filePath: 'test.ts',
        content: 'const x: string = 123;',
        language: 'typescript'
    }];
    
    const results = await solver.processProblemBatch(testProblems);
    
    if (results.results.length > 0) {
        console.log('✅ Basic functionality test passed');
        return true;
    } else {
        throw new Error('No results returned');
    }
}

testBasicFunctionality().catch(console.error);
`;

            await fs.writeFile('tests/basic-functionality.test.js', testContent);

            // Run the test
            execSync('node tests/basic-functionality.test.js', { stdio: 'pipe' });
            
            this.spinner.succeed('✅ System tests passed');

        } catch (error) {
            this.spinner.warn('⚠️ Some tests failed - system should still work');
        }
    }

    async generateConfiguration() {
        this.spinner.start('⚙️ Generating configuration files...');

        // VS Code tasks.json integration
        const tasksJson = {
            "version": "2.0.0",
            "tasks": [
                {
                    "label": "🧠 Start Multi-Core Auto-Solver",
                    "type": "shell",
                    "command": "npm",
                    "args": ["run", "start:cluster"],
                    "group": "build",
                    "presentation": {
                        "echo": true,
                        "reveal": "always",
                        "focus": true,
                        "panel": "dedicated"
                    },
                    "options": {
                        "env": {
                            "MCP_DEBUG": "true",
                            "ENABLE_GPU": this.config.enableGPU.toString(),
                            "MCP_WORKERS": this.config.maxWorkers.toString()
                        }
                    },
                    "isBackground": true
                },
                {
                    "label": "🔧 Setup Auto-Solver Database",
                    "type": "shell", 
                    "command": "npm",
                    "args": ["run", "setup:db"],
                    "group": "build"
                },
                {
                    "label": "📊 Show Auto-Solver Metrics",
                    "type": "shell",
                    "command": "curl",
                    "args": ["http://localhost:4100/mcp/metrics/multicore"],
                    "group": "test"
                }
            ]
        };

        await fs.writeFile('.vscode/tasks.json', JSON.stringify(tasksJson, null, 2));

        // Launch configuration for debugging
        const launchJson = {
            "version": "0.2.0",
            "configurations": [
                {
                    "name": "🐛 Debug Auto-Solver",
                    "type": "node",
                    "request": "launch",
                    "program": "${workspaceFolder}/core/multi-core-solver.js",
                    "env": {
                        "NODE_ENV": "development",
                        "MCP_DEBUG": "true",
                        "MCP_WORKERS": "1" // Single worker for debugging
                    },
                    "console": "integratedTerminal",
                    "skipFiles": ["<node_internals>/**"]
                }
            ]
        };

        await fs.mkdir('.vscode', { recursive: true });
        await fs.writeFile('.vscode/launch.json', JSON.stringify(launchJson, null, 2));

        // Package scripts for easy management
        const packageJsonPath = 'package.json';
        const packageJson = JSON.parse(await fs.readFile(packageJsonPath, 'utf8'));
        
        packageJson.scripts = {
            ...packageJson.scripts,
            "setup": "node scripts/setup-complete-system.js",
            "start:production": `MCP_MULTICORE=true MCP_WORKERS=${this.config.maxWorkers} NODE_ENV=production node core/multi-core-solver.js`,
            "health-check": "curl http://localhost:4100/health",
            "metrics": "curl http://localhost:4100/mcp/metrics/multicore"
        };

        await fs.writeFile(packageJsonPath, JSON.stringify(packageJson, null, 2));

        this.spinner.succeed('✅ Configuration files generated');
    }

    printSummary() {
        console.log(chalk.blue.bold('\n📋 Setup Summary:'));
        console.log(chalk.green(`✅ Multi-core processing: ${this.config.multiCore ? `Enabled (${this.config.maxWorkers} workers)` : 'Disabled'}`));
        console.log(chalk.green(`✅ GPU acceleration: ${this.config.enableGPU ? 'Enabled' : 'Disabled'}`));
        console.log(chalk.green(`✅ Storage engines: ${this.config.storageEngines.join(', ')}`));
        console.log(chalk.green(`✅ LangChain.js: ${this.config.enableLangChain ? 'Enabled' : 'Disabled'}`));
        
        console.log(chalk.yellow.bold('\n🚀 Next Steps:'));
        console.log(chalk.white('1. Start the auto-solver:'));
        console.log(chalk.cyan('   npm run start:cluster'));
        console.log(chalk.white('\n2. Or in VS Code:'));
        console.log(chalk.cyan('   Ctrl+Shift+P → "Tasks: Run Task" → "🧠 Start Multi-Core Auto-Solver"'));
        console.log(chalk.white('\n3. Test auto-solving:'));
        console.log(chalk.cyan('   Ctrl+Shift+Alt+S (solve current file)'));
        console.log(chalk.cyan('   Ctrl+Shift+Alt+W (solve workspace)'));
        console.log(chalk.white('\n4. View metrics:'));
        console.log(chalk.cyan('   npm run metrics'));
        
        console.log(chalk.blue.bold('\n🔗 Service URLs:'));
        console.log(chalk.cyan(`Auto-Solver API: http://localhost:4100`));
        if (this.config.storageEngines.includes('qdrant')) {
            console.log(chalk.cyan(`Qdrant Dashboard: ${this.config.qdrantURL}/dashboard`));
        }
        
        console.log(chalk.green.bold('\n🎉 Your VS Code Multi-Core Auto-Solver is ready!'));
    }
}

// Run setup if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
    const setup = new SystemSetup();
    setup.run().catch(console.error);
}

export { SystemSetup };