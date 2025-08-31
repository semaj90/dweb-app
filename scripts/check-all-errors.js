#!/usr/bin/env node

// Comprehensive Error Checker for Legal AI System
// Checks all components and reports any errors

const fs = require('fs');
const path = require('path');
const { exec } = require('child_process');
const { promisify } = require('util');
const execAsync = promisify(exec);

const colors = {
    reset: '\x1b[0m',
    red: '\x1b[31m',
    green: '\x1b[32m',
    yellow: '\x1b[33m',
    blue: '\x1b[36m',
    magenta: '\x1b[35m'
};

let totalErrors = 0;
let totalWarnings = 0;
const report = {
    timestamp: new Date().toISOString(),
    errors: [],
    warnings: [],
    status: {}
};

async function checkService(name, port) {
    try {
        const response = await fetch(`http://localhost:${port}`, { 
            method: 'GET',
            signal: AbortSignal.timeout(2000)
        });
        report.status[name] = 'running';
        return true;
    } catch (error) {
        report.status[name] = 'offline';
        return false;
    }
}

async function checkNpmErrors() {
    console.log(`${colors.blue}📦 Checking NPM Dependencies...${colors.reset}`);
    
    try {
        const { stdout, stderr } = await execAsync('npm list --depth=0 --json');
        const result = JSON.parse(stdout);
        
        if (result.problems && result.problems.length > 0) {
            console.log(`${colors.red}❌ NPM dependency issues found:${colors.reset}`);
            result.problems.forEach(problem => {
                console.log(`  ${colors.red}• ${problem}${colors.reset}`);
                report.errors.push({ type: 'npm', message: problem });
                totalErrors++;
            });
        } else {
            console.log(`${colors.green}✅ NPM dependencies OK${colors.reset}`);
        }
    } catch (error) {
        console.log(`${colors.yellow}⚠️ Could not check NPM dependencies${colors.reset}`);
        report.warnings.push({ type: 'npm', message: 'Check failed' });
        totalWarnings++;
    }
}

async function checkTypeScriptErrors() {
    console.log(`\n${colors.blue}📝 Checking TypeScript...${colors.reset}`);
    
    if (!fs.existsSync('tsconfig.json')) {
        console.log(`${colors.yellow}⚠️ No TypeScript configuration found${colors.reset}`);
        report.warnings.push({ type: 'typescript', message: 'No tsconfig.json' });
        totalWarnings++;
        return;
    }
    
    try {
        const { stdout, stderr } = await execAsync('npx tsc --noEmit --pretty false');
        console.log(`${colors.green}✅ TypeScript OK - No errors${colors.reset}`);
    } catch (error) {
        const errorCount = (error.stdout.match(/error TS/g) || []).length;
        if (errorCount > 0) {
            console.log(`${colors.red}❌ TypeScript errors: ${errorCount}${colors.reset}`);
            
            // Show first 5 errors
            const errors = error.stdout.split('\n').filter(line => line.includes('error TS')).slice(0, 5);
            errors.forEach(err => {
                console.log(`  ${colors.red}• ${err.trim()}${colors.reset}`);
                report.errors.push({ type: 'typescript', message: err.trim() });
            });
            
            totalErrors += errorCount;
            
            if (errorCount > 5) {
                console.log(`  ${colors.yellow}... and ${errorCount - 5} more errors${colors.reset}`);
            }
        }
    }
}

async function checkServices() {
    console.log(`\n${colors.blue}🔍 Checking Services...${colors.reset}`);
    
    const services = [
        { name: 'PostgreSQL', port: 5432 },
        { name: 'Ollama', port: 11434 },
        { name: 'Neo4j', port: 7474 },
        { name: 'Redis', port: 6379 },
        { name: 'MinIO', port: 9000 },
        { name: 'Memory API', port: 3456 },
        { name: 'Frontend', port: 5173 },
        { name: 'Enhanced RAG', port: 8094 }
    ];
    
    for (const service of services) {
        const isRunning = await checkService(service.name, service.port);
        if (isRunning) {
            console.log(`${colors.green}✅ ${service.name.padEnd(15)} (port ${service.port})${colors.reset}`);
        } else {
            console.log(`${colors.red}❌ ${service.name.padEnd(15)} (port ${service.port}) - Not running${colors.reset}`);
            report.errors.push({ type: 'service', message: `${service.name} not running` });
            totalErrors++;
        }
    }
}

async function checkFileSystem() {
    console.log(`\n${colors.blue}📁 Checking File System...${colors.reset}`);
    
    const requiredFiles = [
        { path: 'admin-dashboard.html', critical: true },
        { path: 'scripts/memory-monitor-api.js', critical: true },
        { path: 'scripts/memory-optimizer.ps1', critical: true },
        { path: '.env', critical: true },
        { path: 'package.json', critical: true },
        { path: 'START-LEGAL-AI-FIXED.bat', critical: false },
        { path: 'MEMORY-OPTIMIZER.bat', critical: false }
    ];
    
    for (const file of requiredFiles) {
        if (fs.existsSync(file.path)) {
            console.log(`${colors.green}✅ Found: ${file.path}${colors.reset}`);
        } else {
            if (file.critical) {
                console.log(`${colors.red}❌ Missing: ${file.path}${colors.reset}`);
                report.errors.push({ type: 'file', message: `Missing ${file.path}` });
                totalErrors++;
            } else {
                console.log(`${colors.yellow}⚠️ Missing (optional): ${file.path}${colors.reset}`);
                report.warnings.push({ type: 'file', message: `Missing optional ${file.path}` });
                totalWarnings++;
            }
        }
    }
}

async function checkLogErrors() {
    console.log(`\n${colors.blue}📋 Checking Error Logs...${colors.reset}`);
    
    const logsDir = path.join(__dirname, '..', 'logs');
    
    if (!fs.existsSync(logsDir)) {
        console.log(`${colors.yellow}⚠️ Logs directory not found${colors.reset}`);
        report.warnings.push({ type: 'logs', message: 'Logs directory missing' });
        totalWarnings++;
        return;
    }
    
    // Check error log files
    const errorFiles = fs.readdirSync(logsDir).filter(file => file.includes('.err.log'));
    
    for (const file of errorFiles) {
        const filePath = path.join(logsDir, file);
        const content = fs.readFileSync(filePath, 'utf8');
        
        if (content.trim().length > 0) {
            const lines = content.trim().split('\n');
            const recentErrors = lines.slice(-5); // Last 5 errors
            
            console.log(`${colors.red}❌ Errors in ${file}:${colors.reset}`);
            recentErrors.forEach(err => {
                if (err.trim()) {
                    console.log(`  ${colors.red}• ${err.trim().substring(0, 100)}...${colors.reset}`);
                    report.errors.push({ type: 'log', file, message: err.trim() });
                    totalErrors++;
                }
            });
        }
    }
    
    if (errorFiles.length === 0 || totalErrors === 0) {
        console.log(`${colors.green}✅ No errors in log files${colors.reset}`);
    }
}

async function checkMemoryStatus() {
    console.log(`\n${colors.blue}💾 Checking Memory Status...${colors.reset}`);
    
    try {
        const response = await fetch('http://localhost:3456/status', {
            signal: AbortSignal.timeout(2000)
        });
        
        if (response.ok) {
            const data = await response.json();
            const usage = parseFloat(data.memory.usagePercent);
            
            if (usage > 85) {
                console.log(`${colors.red}❌ Critical memory usage: ${usage}%${colors.reset}`);
                report.errors.push({ type: 'memory', message: `Critical usage: ${usage}%` });
                totalErrors++;
            } else if (usage > 70) {
                console.log(`${colors.yellow}⚠️ High memory usage: ${usage}%${colors.reset}`);
                report.warnings.push({ type: 'memory', message: `High usage: ${usage}%` });
                totalWarnings++;
            } else {
                console.log(`${colors.green}✅ Memory usage normal: ${usage}%${colors.reset}`);
            }
            
            report.status.memory = {
                usage: usage,
                total: data.memory.totalGB,
                free: data.memory.freeGB
            };
        }
    } catch (error) {
        console.log(`${colors.yellow}⚠️ Memory API not available${colors.reset}`);
        report.warnings.push({ type: 'memory', message: 'API not available' });
        totalWarnings++;
    }
}

async function checkSvelteErrors() {
    console.log(`\n${colors.blue}🎨 Checking Svelte...${colors.reset}`);
    
    const sveltekitPath = path.join(__dirname, '..', 'sveltekit-frontend');
    
    if (!fs.existsSync(sveltekitPath)) {
        console.log(`${colors.yellow}⚠️ SvelteKit frontend not found${colors.reset}`);
        report.warnings.push({ type: 'svelte', message: 'Frontend directory missing' });
        totalWarnings++;
        return;
    }
    
    try {
        process.chdir(sveltekitPath);
        const { stdout } = await execAsync('npx svelte-check --output machine');
        
        const errors = (stdout.match(/"severity":"error"/g) || []).length;
        const warnings = (stdout.match(/"severity":"warning"/g) || []).length;
        
        if (errors > 0) {
            console.log(`${colors.red}❌ Svelte errors: ${errors}${colors.reset}`);
            report.errors.push({ type: 'svelte', message: `${errors} errors found` });
            totalErrors += errors;
        }
        
        if (warnings > 0) {
            console.log(`${colors.yellow}⚠️ Svelte warnings: ${warnings}${colors.reset}`);
            report.warnings.push({ type: 'svelte', message: `${warnings} warnings found` });
            totalWarnings += warnings;
        }
        
        if (errors === 0 && warnings === 0) {
            console.log(`${colors.green}✅ Svelte OK${colors.reset}`);
        }
        
        process.chdir('..');
    } catch (error) {
        console.log(`${colors.yellow}⚠️ Could not check Svelte${colors.reset}`);
        report.warnings.push({ type: 'svelte', message: 'Check failed' });
        totalWarnings++;
    }
}

async function generateReport() {
    console.log(`\n${colors.magenta}${'='.repeat(50)}${colors.reset}`);
    console.log(`${colors.magenta}           Error Check Summary${colors.reset}`);
    console.log(`${colors.magenta}${'='.repeat(50)}${colors.reset}\n`);
    
    if (totalErrors === 0 && totalWarnings === 0) {
        console.log(`${colors.green}🎉 PERFECT! No errors or warnings found!${colors.reset}`);
        console.log(`\n${colors.green}✅ All systems operational${colors.reset}`);
        console.log(`${colors.green}✅ All files in place${colors.reset}`);
        console.log(`${colors.green}✅ No TypeScript errors${colors.reset}`);
        console.log(`${colors.green}✅ Memory usage normal${colors.reset}`);
        
        report.summary = 'PERFECT - No issues';
    } else if (totalErrors === 0) {
        console.log(`${colors.green}✅ No critical errors found${colors.reset}`);
        console.log(`${colors.yellow}⚠️ ${totalWarnings} warning(s) detected${colors.reset}`);
        console.log(`\n${colors.yellow}System is functional with minor issues${colors.reset}`);
        
        report.summary = `GOOD - ${totalWarnings} warnings`;
    } else {
        console.log(`${colors.red}❌ Errors found: ${totalErrors}${colors.reset}`);
        console.log(`${colors.yellow}⚠️ Warnings: ${totalWarnings}${colors.reset}`);
        
        console.log(`\n${colors.yellow}🔧 Recommended Actions:${colors.reset}`);
        
        // Service errors
        const serviceErrors = report.errors.filter(e => e.type === 'service');
        if (serviceErrors.length > 0) {
            console.log(`${colors.yellow}  • Start services: npm run dev:fix${colors.reset}`);
        }
        
        // TypeScript errors
        const tsErrors = report.errors.filter(e => e.type === 'typescript');
        if (tsErrors.length > 0) {
            console.log(`${colors.yellow}  • Fix TypeScript: npm run check:auto:solve${colors.reset}`);
        }
        
        // Memory errors
        const memErrors = report.errors.filter(e => e.type === 'memory');
        if (memErrors.length > 0) {
            console.log(`${colors.yellow}  • Optimize memory: npm run memory:optimize${colors.reset}`);
        }
        
        // Missing files
        const fileErrors = report.errors.filter(e => e.type === 'file');
        if (fileErrors.length > 0) {
            console.log(`${colors.yellow}  • Install system: .\\INSTALL-MEMORY.bat${colors.reset}`);
        }
        
        report.summary = `NEEDS ATTENTION - ${totalErrors} errors, ${totalWarnings} warnings`;
    }
    
    // Save report
    const reportPath = path.join(__dirname, '..', 'logs', `error-report-${Date.now()}.json`);
    fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
    
    console.log(`\n${colors.blue}📄 Full report saved to: ${reportPath}${colors.reset}`);
    console.log(`\n${colors.cyan}Dashboard: admin-dashboard.html${colors.reset}`);
    console.log(`${colors.cyan}Memory API: http://localhost:3456/health${colors.reset}`);
}

// Main execution
async function main() {
    console.log(`${colors.cyan}${'='.repeat(50)}${colors.reset}`);
    console.log(`${colors.cyan}       Legal AI System - Error Check${colors.reset}`);
    console.log(`${colors.cyan}${'='.repeat(50)}${colors.reset}\n`);
    
    await checkNpmErrors();
    await checkTypeScriptErrors();
    await checkServices();
    await checkFileSystem();
    await checkLogErrors();
    await checkMemoryStatus();
    await checkSvelteErrors();
    await generateReport();
    
    // Exit with error code if errors found
    process.exit(totalErrors > 0 ? 1 : 0);
}

// Run the checker
main().catch(error => {
    console.error(`${colors.red}Fatal error during check: ${error.message}${colors.reset}`);
    process.exit(1);
});