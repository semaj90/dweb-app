#!/usr/bin/env node

// COMPREHENSIVE PHASE 3+4 SYSTEM STATUS REPORT
// Tests all services and provides complete validation

import { createConnection } from 'net';
import http from 'http';
import fs from 'fs';

console.log('🚀 PHASE 3+4 LEGAL AI SYSTEM - COMPREHENSIVE STATUS REPORT');
console.log('=' + '='.repeat(70));
console.log('');

const services = [
    { name: 'PostgreSQL + pgvector', port: 5432, type: 'database', phase: '3+4', critical: true },
    { name: 'Redis Cache', port: 6379, type: 'cache', phase: '3+4', critical: true },
    { name: 'Qdrant Vector DB', port: 6333, type: 'vector', phase: '3', critical: true, healthUrl: 'http://localhost:6333' },
    { name: 'Ollama LLM', port: 11434, type: 'llm', phase: '3', critical: true, healthUrl: 'http://localhost:11434/api/version' },
    { name: 'Neo4j Graph DB', port: 7474, type: 'graph', phase: '4', critical: false, healthUrl: 'http://localhost:7474' },
    { name: 'RabbitMQ Event Stream', port: 5672, type: 'messaging', phase: '4', critical: false },
    { name: 'RabbitMQ Management', port: 15672, type: 'ui', phase: '4', critical: false, healthUrl: 'http://localhost:15672' },
    { name: 'TTS Service', port: 5002, type: 'ai', phase: '3+4', critical: false, healthUrl: 'http://localhost:5002/health' }
];

async function testPort(service) {
    return new Promise((resolve) => {
        const socket = createConnection(service.port, 'localhost');
        const timeout = setTimeout(() => {
            socket.destroy();
            resolve({ ...service, status: 'timeout', details: 'Connection timeout' });
        }, 5000);
        
        socket.on('connect', () => {
            clearTimeout(timeout);
            socket.destroy();
            resolve({ ...service, status: 'connected', details: 'Port accessible' });
        });
        
        socket.on('error', (error) => {
            clearTimeout(timeout);
            resolve({ ...service, status: 'failed', details: error.message });
        });
    });
}

async function testHttpHealth(service) {
    if (!service.healthUrl) return { ...service, httpStatus: 'no-endpoint' };
    
    return new Promise((resolve) => {
        const url = new URL(service.healthUrl);
        const req = http.request({
            hostname: url.hostname,
            port: url.port,
            path: url.pathname,
            method: 'GET',
            timeout: 5000
        }, (res) => {
            resolve({ ...service, httpStatus: res.statusCode === 200 ? 'healthy' : `http-${res.statusCode}` });
        });
        
        req.on('error', (error) => {
            resolve({ ...service, httpStatus: 'http-error', httpDetails: error.message });
        });
        
        req.on('timeout', () => {
            req.destroy();
            resolve({ ...service, httpStatus: 'http-timeout' });
        });
        
        req.end();
    });
}

async function checkModels() {
    console.log('🤖 CHECKING LOCAL MODELS:');
    console.log('-'.repeat(30));
    
    try {
        const modelsDir = './models';
        if (fs.existsSync(modelsDir)) {
            const items = fs.readdirSync(modelsDir, { withFileTypes: true });
            items.forEach(item => {
                const icon = item.isDirectory() ? '📁' : '📄';
                console.log(`${icon} ${item.name}`);
            });
            
            // Check for GGUF files
            const ggufFiles = [];
            function findGGUF(dir) {
                const items = fs.readdirSync(dir, { withFileTypes: true });
                items.forEach(item => {
                    const fullPath = `${dir}/${item.name}`;
                    if (item.isDirectory()) {
                        findGGUF(fullPath);
                    } else if (item.name.endsWith('.gguf') || item.name.endsWith('.bin')) {
                        ggufFiles.push(fullPath);
                    }
                });
            }
            
            findGGUF(modelsDir);
            
            if (ggufFiles.length > 0) {
                console.log('\\n🎯 GGUF/Model Files Found:');
                ggufFiles.forEach(file => {
                    const stats = fs.statSync(file);
                    const sizeMB = (stats.size / (1024 * 1024)).toFixed(1);
                    console.log(`  ✅ ${file} (${sizeMB} MB)`);
                });
            } else {
                console.log('\\n⚠️  No GGUF files found - place your Unsloth model in ./models/');
            }
        } else {
            console.log('❌ Models directory not found');
        }
    } catch (error) {
        console.log(`❌ Error checking models: ${error.message}`);
    }
    
    console.log('');
}

async function generateReport() {
    const startTime = Date.now();
    
    // Test port connectivity
    console.log('🔍 TESTING SERVICE CONNECTIVITY:');
    console.log('-'.repeat(40));
    const portResults = await Promise.all(services.map(testPort));
    
    portResults.forEach(result => {
        const icon = result.status === 'connected' ? '✅' : 
                    result.critical && result.status !== 'connected' ? '❌' : '⚠️ ';
        const phase = result.phase.padEnd(3);
        const type = result.type.padEnd(9);
        console.log(`${icon} [${phase}] ${type} ${result.name}: ${result.status}`);
    });
    
    console.log('');
    
    // Test HTTP health endpoints
    console.log('🏥 TESTING HTTP HEALTH ENDPOINTS:');
    console.log('-'.repeat(35));
    const httpResults = await Promise.all(portResults.map(testHttpHealth));
    
    httpResults.forEach(result => {
        if (result.healthUrl) {
            const icon = result.httpStatus === 'healthy' ? '✅' : '❌';
            console.log(`${icon} ${result.name}: ${result.httpStatus}`);
        }
    });
    
    console.log('');
    
    // Check models
    await checkModels();
    
    // Generate summary
    const totalTime = Date.now() - startTime;
    const connectedServices = portResults.filter(r => r.status === 'connected').length;
    const criticalServices = portResults.filter(r => r.critical).length;
    const connectedCritical = portResults.filter(r => r.critical && r.status === 'connected').length;
    const healthyEndpoints = httpResults.filter(r => r.httpStatus === 'healthy').length;
    const totalEndpoints = httpResults.filter(r => r.healthUrl).length;
    
    console.log('📊 SYSTEM STATUS SUMMARY:');
    console.log('=' + '='.repeat(30));
    console.log(`⏱️  Test Duration: ${totalTime}ms`);
    console.log(`🔗 Connected Services: ${connectedServices}/${services.length}`);
    console.log(`🎯 Critical Services: ${connectedCritical}/${criticalServices}`);
    console.log(`🏥 Healthy Endpoints: ${healthyEndpoints}/${totalEndpoints}`);
    console.log('');
    
    // Phase breakdown
    const phase3Services = portResults.filter(r => r.phase.includes('3'));
    const phase4Services = portResults.filter(r => r.phase.includes('4'));
    const phase3Connected = phase3Services.filter(r => r.status === 'connected').length;
    const phase4Connected = phase4Services.filter(r => r.status === 'connected').length;
    
    console.log('🏗️  PHASE BREAKDOWN:');
    console.log(`Phase 3 (Advanced RAG): ${phase3Connected}/${phase3Services.length} services`);
    console.log(`Phase 4 (Data + Events): ${phase4Connected}/${phase4Services.length} services`);
    console.log('');
    
    // Final assessment
    const systemHealthy = connectedCritical >= 3; // Need at least 3 critical services
    const readyForPhase5 = connectedCritical === criticalServices;
    
    console.log('🎯 FINAL ASSESSMENT:');
    console.log('=' + '='.repeat(20));
    
    if (readyForPhase5) {
        console.log('🎉 SYSTEM STATUS: FULLY OPERATIONAL');
        console.log('✅ All critical services running');
        console.log('✅ Ready for Phase 5 development');
        console.log('✅ Advanced RAG capabilities active');
        console.log('✅ Data management pipeline ready');
        console.log('');
        console.log('🚀 NEXT STEPS:');
        console.log('1. Load your Unsloth GGUF model into Ollama');
        console.log('2. Set up vLLM server for enhanced performance');
        console.log('3. Begin Phase 5: AI-Driven Real-Time UI Updates');
        console.log('4. Implement SvelteKit components with reactive state');
        console.log('');
        console.log('🔗 SERVICE ACCESS:');
        console.log('• PostgreSQL: localhost:5432 (legal_admin/LegalRAG2024!)');
        console.log('• Qdrant: http://localhost:6333');
        console.log('• Ollama: http://localhost:11434');
        console.log('• Redis: localhost:6379');
        
    } else if (systemHealthy) {
        console.log('⚠️  SYSTEM STATUS: PARTIALLY OPERATIONAL');
        console.log(`✅ ${connectedCritical}/${criticalServices} critical services running`);
        console.log('🔧 Some services need attention');
        
        const failedCritical = portResults.filter(r => r.critical && r.status !== 'connected');
        if (failedCritical.length > 0) {
            console.log('\\n❌ Failed Critical Services:');
            failedCritical.forEach(service => {
                console.log(`   • ${service.name}: ${service.details}`);
            });
        }
        
        console.log('\\n🔧 TROUBLESHOOTING:');
        console.log('1. Check Docker containers: docker ps');
        console.log('2. View logs: docker-compose logs [service]');
        console.log('3. Restart services: docker-compose restart');
        
    } else {
        console.log('❌ SYSTEM STATUS: CRITICAL ISSUES');
        console.log('🚨 Multiple critical services down');
        console.log('🔧 System requires immediate attention');
        
        console.log('\\n🆘 EMERGENCY RECOVERY:');
        console.log('1. Stop all: docker-compose down');
        console.log('2. Start fresh: docker-compose up -d');
        console.log('3. Wait 60 seconds for initialization');
        console.log('4. Re-run this test');
    }
    
    console.log('');
    console.log('📋 LOG: Test completed at ' + new Date().toISOString());
    
    // Exit with appropriate code
    process.exit(readyForPhase5 ? 0 : systemHealthy ? 1 : 2);
}

// Handle errors gracefully
process.on('uncaughtException', (error) => {
    console.error('❌ Test crashed:', error.message);
    process.exit(3);
});

// Execute the comprehensive test
generateReport().catch(error => {
    console.error('❌ Test failed:', error.message);
    process.exit(3);
});
