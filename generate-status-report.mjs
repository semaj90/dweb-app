#!/usr/bin/env node
// Generate a status report for debugging/sharing

import fs from 'fs';
import { exec } from 'child_process';
import { promisify } from 'util';
import net from 'net';

const execAsync = promisify(exec);

async function generateStatusReport() {
    const report = [];
    const timestamp = new Date().toISOString();
    
    report.push('YoRHa Legal AI Platform - System Status Report');
    report.push(`Generated: ${timestamp}`);
    report.push('=' .repeat(60));
    report.push('');
    
    // System Information
    report.push('SYSTEM INFORMATION:');
    try {
        const { stdout: osInfo } = await execAsync('systeminfo | findstr /B /C:"OS Name" /C:"OS Version"');
        report.push(osInfo.trim());
    } catch {
        report.push('Unable to get OS information');
    }
    report.push('');
    
    // Check ports
    report.push('SERVICE STATUS:');
    const services = [
        { name: 'PostgreSQL', port: 5432 },
        { name: 'Redis', port: 6379 },
        { name: 'Neo4j Browser', port: 7474 },
        { name: 'Neo4j Bolt', port: 7687 },
        { name: 'MinIO API', port: 9000 },
        { name: 'MinIO Console', port: 9001 },
        { name: 'Ollama API', port: 11434 },
        { name: 'Vite Dev Server', port: 5173 },
        { name: 'Alt Dev Server', port: 3000 },
        { name: 'GPU Orchestrator', port: 8084 },
        { name: 'RAG Service', port: 8085 }
    ];
    
    for (const service of services) {
        const status = await checkPort(service.port);
        report.push(`${service.name.padEnd(20)} Port ${service.port.toString().padEnd(5)} : ${status ? 'ONLINE' : 'OFFLINE'}`);
    }
    report.push('');
    
    // Check installations
    report.push('INSTALLED SOFTWARE:');
    const software = [
        { name: 'Node.js', cmd: 'node --version' },
        { name: 'npm', cmd: 'npm --version' },
        { name: 'Git', cmd: 'git --version' },
        { name: 'Python', cmd: 'python --version' },
        { name: 'Ollama', cmd: 'ollama version' },
        { name: 'PostgreSQL', cmd: 'psql --version' }
    ];
    
    for (const app of software) {
        try {
            const { stdout } = await execAsync(`${app.cmd} 2>&1`);
            const version = stdout.trim().split('\n')[0];
            report.push(`${app.name.padEnd(15)} : ${version}`);
        } catch {
            report.push(`${app.name.padEnd(15)} : Not installed`);
        }
    }
    report.push('');
    
    // Check project files
    report.push('PROJECT STATUS:');
    const files = [
        'package.json',
        'node_modules',
        '.env',
        'drizzle.config.ts',
        'src/routes/yorha-dashboard',
        'src/routes/+page.svelte',
        'START-LEGAL-AI.bat',
        'START-NATIVE-WINDOWS-COMPLETE.ps1'
    ];
    
    for (const file of files) {
        const exists = fs.existsSync(file);
        report.push(`${file.padEnd(40)} : ${exists ? 'EXISTS' : 'MISSING'}`);
    }
    report.push('');
    
    // Database check
    report.push('DATABASE STATUS:');
    try {
        const { stdout } = await execAsync('psql -U legal_admin -d legal_ai_db -h localhost -t -c "SELECT \'Connected\' as status;" 2>&1');
        if (stdout.includes('Connected')) {
            report.push('PostgreSQL connection: SUCCESS');
            
            // Get counts
            try {
                const { stdout: cases } = await execAsync('psql -U legal_admin -d legal_ai_db -h localhost -t -c "SELECT COUNT(*) FROM cases;" 2>&1');
                const { stdout: docs } = await execAsync('psql -U legal_admin -d legal_ai_db -h localhost -t -c "SELECT COUNT(*) FROM legal_documents;" 2>&1');
                const { stdout: evidence } = await execAsync('psql -U legal_admin -d legal_ai_db -h localhost -t -c "SELECT COUNT(*) FROM evidence;" 2>&1');
                
                report.push(`Cases: ${cases.trim()}`);
                report.push(`Documents: ${docs.trim()}`);
                report.push(`Evidence: ${evidence.trim()}`);
            } catch {
                report.push('Could not get table counts');
            }
        } else {
            report.push('PostgreSQL connection: FAILED');
        }
    } catch {
        report.push('PostgreSQL: Not accessible');
    }
    report.push('');
    
    // Ollama models
    report.push('AI MODELS:');
    try {
        const { stdout } = await execAsync('ollama list 2>&1');
        const models = stdout.split('\n').slice(1).filter(line => line.trim() && !line.includes('NAME'));
        if (models.length > 0) {
            models.forEach(model => {
                const modelName = model.split(/\s+/)[0];
                if (modelName) report.push(`- ${modelName}`);
            });
        } else {
            report.push('No models loaded');
        }
    } catch {
        report.push('Ollama not available');
    }
    report.push('');
    
    // Summary
    report.push('=' .repeat(60));
    report.push('SUMMARY:');
    
    const onlineServices = services.filter(async s => await checkPort(s.port)).length;
    report.push(`Services Online: ${onlineServices}/${services.length}`);
    
    const installedSoftware = software.filter(async s => {
        try {
            await execAsync(s.cmd);
            return true;
        } catch {
            return false;
        }
    }).length;
    report.push(`Software Installed: ${installedSoftware}/${software.length}`);
    
    const existingFiles = files.filter(f => fs.existsSync(f)).length;
    report.push(`Project Files: ${existingFiles}/${files.length}`);
    
    report.push('');
    report.push('Report saved to: system-status-report.txt');
    
    // Save report
    const reportContent = report.join('\n');
    fs.writeFileSync('system-status-report.txt', reportContent);
    console.log(reportContent);
    
    return reportContent;
}

async function checkPort(port) {
    return new Promise((resolve) => {
        const server = net.createServer();
        server.once('error', () => resolve(true));
        server.once('listening', () => {
            server.close();
            resolve(false);
        });
        server.listen(port, '127.0.0.1');
    });
}

// Set PGPASSWORD for database checks
process.env.PGPASSWORD = '123456';

// Run the report
generateStatusReport().catch(console.error);
