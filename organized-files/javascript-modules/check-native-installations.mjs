#!/usr/bin/env node
// Comprehensive Native Windows Installation Checker
// This script checks for all required services and components

import { exec } from 'child_process';
import { promisify } from 'util';
import fs from 'fs';
import path from 'path';
import net from 'net';
import pg from 'pg';

const execAsync = promisify(exec);

console.log(`
╔══════════════════════════════════════════════════════════════╗
║      NATIVE WINDOWS INSTALLATION STATUS CHECK                ║
║           Legal AI Platform - YoRHa Interface                ║
╚══════════════════════════════════════════════════════════════╝
`);

// Color codes for console output
const colors = {
    reset: '\x1b[0m',
    green: '\x1b[32m',
    red: '\x1b[31m',
    yellow: '\x1b[33m',
    cyan: '\x1b[36m',
    magenta: '\x1b[35m'
};

// Helper functions
function printSection(title) {
    console.log(`\n${colors.cyan}━━━ ${title} ━━━${colors.reset}`);
}

function printSuccess(message) {
    console.log(`${colors.green}✓${colors.reset} ${message}`);
}

function printError(message) {
    console.log(`${colors.red}✗${colors.reset} ${message}`);
}

function printWarning(message) {
    console.log(`${colors.yellow}⚠${colors.reset} ${message}`);
}

function printInfo(message) {
    console.log(`${colors.magenta}ℹ${colors.reset} ${message}`);
}

// Check if a port is in use
async function checkPort(port, serviceName) {
    return new Promise((resolve) => {
        const server = net.createServer();
        
        server.once('error', (err) => {
            if (err.code === 'EADDRINUSE') {
                printSuccess(`Port ${port} is in use (${serviceName} likely running)`);
                resolve(true);
            } else {
                printError(`Port ${port} error for ${serviceName}: ${err.message}`);
                resolve(false);
            }
        });
        
        server.once('listening', () => {
            server.close();
            printWarning(`Port ${port} is available (${serviceName} not running)`);
            resolve(false);
        });
        
        server.listen(port, '127.0.0.1');
    });
}

// Check if a Windows service exists and is running
async function checkWindowsService(serviceName, displayName) {
    try {
        const { stdout } = await execAsync(`sc query "${serviceName}" 2>nul`);
        if (stdout.includes('RUNNING')) {
            printSuccess(`${displayName} service is running`);
            return true;
        } else if (stdout.includes('STOPPED')) {
            printWarning(`${displayName} service exists but is stopped`);
            return false;
        }
    } catch {
        printError(`${displayName} service not found`);
        return false;
    }
}

// Check if a process is running
async function checkProcess(processName, displayName) {
    try {
        const { stdout } = await execAsync(`tasklist /FI "IMAGENAME eq ${processName}" 2>nul`);
        if (stdout.includes(processName)) {
            printSuccess(`${displayName} is running`);
            return true;
        } else {
            printWarning(`${displayName} is not running`);
            return false;
        }
    } catch {
        printError(`Could not check ${displayName}`);
        return false;
    }
}

// Check if a program is installed
async function checkProgram(command, displayName, versionFlag = '--version') {
    try {
        const { stdout } = await execAsync(`${command} ${versionFlag} 2>&1`);
        const version = stdout.split('\n')[0].trim();
        printSuccess(`${displayName} installed: ${version}`);
        return true;
    } catch {
        printError(`${displayName} not installed or not in PATH`);
        return false;
    }
}

// Check file/directory existence
function checkPath(pathToCheck, description) {
    if (fs.existsSync(pathToCheck)) {
        printSuccess(`${description} found at: ${pathToCheck}`);
        return true;
    } else {
        printWarning(`${description} not found at: ${pathToCheck}`);
        return false;
    }
}

// Main check function
async function checkAllInstallations() {
    const results = {
        core: {},
        databases: {},
        services: {},
        ai: {},
        frontend: {},
        project: {}
    };

    // ============================================
    // CORE DEVELOPMENT TOOLS
    // ============================================
    printSection('CORE DEVELOPMENT TOOLS');
    
    results.core.node = await checkProgram('node', 'Node.js', '--version');
    results.core.npm = await checkProgram('npm', 'npm', '--version');
    results.core.git = await checkProgram('git', 'Git', '--version');
    results.core.python = await checkProgram('python', 'Python', '--version');
    results.core.powershell = await checkProgram('powershell', 'PowerShell', '-Version');

    // ============================================
    // DATABASE SERVICES
    // ============================================
    printSection('DATABASE SERVICES');
    
    // PostgreSQL
    results.databases.postgresqlInstalled = await checkProgram('psql', 'PostgreSQL', '--version');
    results.databases.postgresqlService = await checkWindowsService('postgresql-x64-17', 'PostgreSQL 17');
    if (!results.databases.postgresqlService) {
        results.databases.postgresqlService = await checkWindowsService('postgresql-x64-15', 'PostgreSQL 15');
    }
    results.databases.postgresqlPort = await checkPort(5432, 'PostgreSQL');
    
    // Test actual database connection
    if (results.databases.postgresqlPort) {
        printInfo('Testing database connection...');
        try {
            const client = new pg.Client({
                connectionString: 'postgresql://legal_admin:123456@localhost:5432/legal_ai_db'
            });
            await client.connect();
            const result = await client.query('SELECT COUNT(*) FROM cases');
            printSuccess(`Database connected! Cases in DB: ${result.rows[0].count}`);
            results.databases.dbConnection = true;
            await client.end();
        } catch (error) {
            printError(`Database connection failed: ${error.message}`);
            results.databases.dbConnection = false;
        }
    }

    // Redis
    printInfo('Checking Redis...');
    results.databases.redisService = await checkWindowsService('Redis', 'Redis');
    results.databases.redisProcess = await checkProcess('redis-server.exe', 'Redis Server');
    results.databases.redisPort = await checkPort(6379, 'Redis');
    checkPath('C:\\Redis', 'Redis installation');
    checkPath('C:\\Program Files\\Redis', 'Redis installation (alternative)');

    // Neo4j
    printInfo('Checking Neo4j...');
    results.databases.neo4jService = await checkWindowsService('neo4j', 'Neo4j');
    results.databases.neo4jProcess = await checkProcess('java.exe', 'Neo4j (Java)');
    results.databases.neo4jBrowser = await checkPort(7474, 'Neo4j Browser');
    results.databases.neo4jBolt = await checkPort(7687, 'Neo4j Bolt');
    checkPath('C:\\neo4j', 'Neo4j installation');
    checkPath('.\\neo4j-community-5.23.0', 'Neo4j local installation');

    // ============================================
    // STORAGE & MESSAGING SERVICES
    // ============================================
    printSection('STORAGE & MESSAGING SERVICES');
    
    // MinIO
    results.services.minioProcess = await checkProcess('minio.exe', 'MinIO');
    results.services.minioApi = await checkPort(9000, 'MinIO API');
    results.services.minioConsole = await checkPort(9001, 'MinIO Console');
    checkPath('C:\\minio\\minio.exe', 'MinIO installation');
    checkPath('.\\minio.exe', 'MinIO local file');

    // RabbitMQ
    results.services.rabbitmqService = await checkWindowsService('RabbitMQ', 'RabbitMQ');
    results.services.rabbitmqPort = await checkPort(5672, 'RabbitMQ');
    results.services.rabbitmqManagement = await checkPort(15672, 'RabbitMQ Management');

    // ============================================
    // AI & ML SERVICES
    // ============================================
    printSection('AI & ML SERVICES');
    
    // Ollama
    results.ai.ollamaInstalled = await checkProgram('ollama', 'Ollama', 'version');
    results.ai.ollamaProcess = await checkProcess('ollama.exe', 'Ollama');
    results.ai.ollamaPort = await checkPort(11434, 'Ollama API');
    
    if (results.ai.ollamaInstalled) {
        try {
            const { stdout } = await execAsync('ollama list 2>nul');
            console.log('  Available models:');
            const models = stdout.split('\n').slice(1).filter(line => line.trim());
            models.forEach(model => {
                if (model.includes('NAME')) return;
                const modelName = model.split(/\s+/)[0];
                if (modelName) printInfo(`    - ${modelName}`);
            });
        } catch {
            printWarning('  Could not list Ollama models');
        }
    }

    // CUDA check for GPU support
    results.ai.cuda = await checkProgram('nvidia-smi', 'NVIDIA CUDA', '');
    checkPath('C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA', 'CUDA Toolkit');

    // ============================================
    // FRONTEND & APPLICATION
    // ============================================
    printSection('FRONTEND & APPLICATION');
    
    // Check if dev server is running
    results.frontend.devServer = await checkPort(5173, 'Vite Dev Server');
    results.frontend.alternatePort = await checkPort(3000, 'Alternative Dev Server');
    
    // Check Go services
    results.frontend.gpuOrchestrator = await checkPort(8084, 'GPU Orchestrator');
    results.frontend.ragService = await checkPort(8085, 'RAG Service');

    // ============================================
    // PROJECT FILES & STRUCTURE
    // ============================================
    printSection('PROJECT FILES & STRUCTURE');
    
    // Check critical project files
    results.project.packageJson = checkPath('./package.json', 'package.json');
    results.project.nodeModules = checkPath('./node_modules', 'node_modules');
    results.project.envFile = checkPath('./.env', 'Environment file');
    results.project.drizzleConfig = checkPath('./drizzle.config.ts', 'Drizzle config');
    
    // Check YoRHa components
    results.project.yorhaDashboard = checkPath('./src/routes/yorha-dashboard', 'YoRHa Dashboard route');
    results.project.mainPage = checkPath('./src/routes/+page.svelte', 'Main page');
    
    // Check critical scripts
    results.project.startScript = checkPath('./START-LEGAL-AI.bat', 'Start script');
    results.project.nativeScript = checkPath('./START-NATIVE-WINDOWS-COMPLETE.ps1', 'Native Windows script');

    // ============================================
    // SUMMARY & RECOMMENDATIONS
    // ============================================
    printSection('INSTALLATION SUMMARY');
    
    const working = [];
    const notWorking = [];
    const needsAttention = [];

    // Analyze results
    if (results.databases.dbConnection) working.push('PostgreSQL Database');
    else if (results.databases.postgresqlPort) needsAttention.push('PostgreSQL (running but connection failed)');
    else notWorking.push('PostgreSQL');

    if (results.databases.redisPort) working.push('Redis');
    else notWorking.push('Redis');

    if (results.databases.neo4jBrowser) working.push('Neo4j');
    else notWorking.push('Neo4j');

    if (results.services.minioApi) working.push('MinIO');
    else notWorking.push('MinIO');

    if (results.ai.ollamaPort) working.push('Ollama');
    else if (results.ai.ollamaInstalled) needsAttention.push('Ollama (installed but not running)');
    else notWorking.push('Ollama');

    if (results.frontend.devServer || results.frontend.alternatePort) working.push('Dev Server');
    else needsAttention.push('Dev Server (not running)');

    // Print summary
    console.log(`\n${colors.green}✅ WORKING SERVICES (${working.length}):${colors.reset}`);
    working.forEach(service => console.log(`   • ${service}`));

    if (needsAttention.length > 0) {
        console.log(`\n${colors.yellow}⚠️  NEEDS ATTENTION (${needsAttention.length}):${colors.reset}`);
        needsAttention.forEach(service => console.log(`   • ${service}`));
    }

    if (notWorking.length > 0) {
        console.log(`\n${colors.red}❌ NOT WORKING (${notWorking.length}):${colors.reset}`);
        notWorking.forEach(service => console.log(`   • ${service}`));
    }

    // ============================================
    // RECOMMENDED ACTIONS
    // ============================================
    printSection('RECOMMENDED ACTIONS');
    
    if (notWorking.length > 0 || needsAttention.length > 0) {
        console.log('\nTo fix missing services, run as Administrator:');
        console.log(`${colors.cyan}  .\\START-NATIVE-WINDOWS-COMPLETE.ps1${colors.reset}`);
        
        if (!results.databases.redisPort) {
            console.log('\nFor Redis specifically:');
            console.log('  1. Download: https://github.com/microsoftarchive/redis/releases');
            console.log('  2. Install to C:\\Redis');
            console.log('  3. Run: redis-server.exe');
        }
        
        if (!results.ai.ollamaPort && results.ai.ollamaInstalled) {
            console.log('\nStart Ollama:');
            console.log('  ollama serve');
        }
        
        if (!results.frontend.devServer && !results.frontend.alternatePort) {
            console.log('\nStart development server:');
            console.log('  npm run dev');
        }
    } else {
        console.log(`\n${colors.green}🎉 All services are running! Your system is ready.${colors.reset}`);
        console.log('\nAccess your application at:');
        console.log(`  ${colors.cyan}http://localhost:${results.frontend.devServer ? '5173' : '3000'}${colors.reset}`);
    }

    return results;
}

// Run the checks
checkAllInstallations().catch(console.error);
