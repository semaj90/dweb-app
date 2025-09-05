#!/usr/bin/env node
// System Integration Verification - Comprehensive Health Check

import { exec } from 'child_process';
import { promisify } from 'util';
import fs from 'fs/promises';

const execAsync = promisify(exec);

class SystemIntegrationChecker {
    constructor() {
        this.results = {
            critical: [],
            warnings: [],
            passed: [],
            performance: {}
        };
    }

    async runCheck() {
        console.log('🔍 Running comprehensive system integration check...\n');

        await this.checkDatabaseIntegration();
        await this.checkGPUServices();
        await this.checkServiceOrchestration();
        await this.checkAPIRouting();
        await this.checkPerformanceBaseline();
        await this.checkLoggingIntegration();

        this.generateReport();
    }

    async checkDatabaseIntegration() {
        try {
            // Check schema migration status
            const { stdout } = await execAsync(`psql -U legal_admin -d legal_ai_db -c "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'indexed_files' AND column_name = 'embedding'"`);
            
            if (stdout.includes('vector(768)')) {
                this.results.passed.push('✅ Database schema migrated to 768-dim embeddings');
            } else {
                this.results.critical.push('❌ Database schema migration required - run gpu-schema-migration.sql');
            }

            // Check table existence
            const tables = ['indexed_files', 'user_activities', 'processing_jobs'];
            for (const table of tables) {
                try {
                    await execAsync(`psql -U legal_admin -d legal_ai_db -c "SELECT COUNT(*) FROM ${table} LIMIT 1"`);
                    this.results.passed.push(`✅ Table ${table} exists`);
                } catch {
                    this.results.critical.push(`❌ Missing table: ${table}`);
                }
            }
        } catch (error) {
            this.results.critical.push('❌ Database connection failed');
        }
    }

    async checkGPUServices() {
        try {
            // Check GPU service health
            const response = await fetch('http://localhost:8080/health');
            const health = await response.json();
            
            if (health.gpu_enabled) {
                this.results.passed.push('✅ GPU service online with acceleration enabled');
                this.results.performance.gpu_memory = health.memory_usage;
            } else {
                this.results.warnings.push('⚠️ GPU service online but acceleration disabled');
            }

            // Check CUDA availability
            try {
                const { stdout } = await execAsync('nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits');
                this.results.passed.push('✅ CUDA/GPU detected');
                this.results.performance.gpu_memory_status = stdout.trim();
            } catch {
                this.results.warnings.push('⚠️ CUDA/GPU not available - will use CPU fallback');
            }
        } catch {
            this.results.critical.push('❌ GPU service offline - check port 8080');
        }
    }

    async checkServiceOrchestration() {
        const services = [
            { name: 'Redis', cmd: 'redis-windows\\redis-cli.exe ping', expected: 'PONG' },
            { name: 'Auto-indexer', port: 8081 },
            { name: 'BullMQ Producer', port: 3001 }
        ];

        for (const service of services) {
            if (service.cmd) {
                try {
                    const { stdout } = await execAsync(service.cmd);
                    if (stdout.includes(service.expected)) {
                        this.results.passed.push(`✅ ${service.name} responsive`);
                    } else {
                        this.results.warnings.push(`⚠️ ${service.name} unexpected response`);
                    }
                } catch {
                    this.results.critical.push(`❌ ${service.name} offline`);
                }
            }

            if (service.port) {
                try {
                    await fetch(`http://localhost:${service.port}/health`);
                    this.results.passed.push(`✅ ${service.name} online`);
                } catch {
                    this.results.warnings.push(`⚠️ ${service.name} not responding on port ${service.port}`);
                }
            }
        }
    }

    async checkAPIRouting() {
        const endpoints = [
            '/api/legal',
            '/api/legal/gpu',
            '/health',
            '/metrics'
        ];

        for (const endpoint of endpoints) {
            try {
                const response = await fetch(`http://localhost:8080${endpoint}`);
                if (response.ok) {
                    this.results.passed.push(`✅ API endpoint ${endpoint} responding`);
                } else {
                    this.results.warnings.push(`⚠️ API endpoint ${endpoint} returned ${response.status}`);
                }
            } catch {
                this.results.critical.push(`❌ API endpoint ${endpoint} unreachable`);
            }
        }
    }

    async checkPerformanceBaseline() {
        try {
            // Test similarity search performance
            const startTime = Date.now();
            const testData = {
                endpoint: 'similarity-search',
                queryEmbedding: new Array(768).fill(0.1),
                documentEmbeddings: [new Array(768).fill(0.2), new Array(768).fill(0.3)],
                documentIds: ['test1', 'test2'],
                topK: 2,
                useGPU: true
            };

            const response = await fetch('http://localhost:8080/api/legal', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(testData)
            });

            const endTime = Date.now();
            const responseTime = endTime - startTime;

            if (response.ok) {
                this.results.performance.similarity_search_ms = responseTime;
                if (responseTime < 1000) {
                    this.results.passed.push(`✅ Similarity search performance: ${responseTime}ms`);
                } else {
                    this.results.warnings.push(`⚠️ Slow similarity search: ${responseTime}ms`);
                }
            } else {
                this.results.critical.push('❌ Similarity search test failed');
            }
        } catch (error) {
            this.results.critical.push('❌ Performance test failed');
        }
    }

    async checkLoggingIntegration() {
        try {
            // Check if log files exist and are being written
            const logFiles = [
                './logs/gpu-legal-processor.log',
                './logs/auto-indexer.log',
                './logs/bullmq-combined.log'
            ];

            for (const logFile of logFiles) {
                try {
                    const stats = await fs.stat(logFile);
                    const ageMinutes = (Date.now() - stats.mtime.getTime()) / (1000 * 60);
                    
                    if (ageMinutes < 10) {
                        this.results.passed.push(`✅ Log file ${logFile} actively written`);
                    } else {
                        this.results.warnings.push(`⚠️ Log file ${logFile} stale (${ageMinutes.toFixed(1)}min old)`);
                    }
                } catch {
                    this.results.warnings.push(`⚠️ Log file ${logFile} missing`);
                }
            }
        } catch (error) {
            this.results.warnings.push('⚠️ Logging check failed');
        }
    }

    generateReport() {
        console.log('\n📊 SYSTEM INTEGRATION REPORT');
        console.log('=' .repeat(50));

        if (this.results.critical.length > 0) {
            console.log('\n🚨 CRITICAL ISSUES (Fix immediately):');
            this.results.critical.forEach(issue => console.log(`  ${issue}`));
        }

        if (this.results.warnings.length > 0) {
            console.log('\n⚠️  WARNINGS (Address when possible):');
            this.results.warnings.forEach(warning => console.log(`  ${warning}`));
        }

        if (this.results.passed.length > 0) {
            console.log('\n✅ PASSING CHECKS:');
            this.results.passed.forEach(check => console.log(`  ${check}`));
        }

        console.log('\n📈 PERFORMANCE METRICS:');
        Object.entries(this.results.performance).forEach(([key, value]) => {
            console.log(`  ${key}: ${value}`);
        });

        // System status
        const criticalCount = this.results.critical.length;
        if (criticalCount === 0) {
            console.log('\n🎯 SYSTEM STATUS: READY FOR PRODUCTION');
        } else {
            console.log(`\n🔧 SYSTEM STATUS: ${criticalCount} critical issues require attention`);
        }

        // Next steps
        console.log('\n📋 IMMEDIATE NEXT STEPS:');
        if (criticalCount > 0) {
            console.log('  1. Fix critical issues listed above');
            console.log('  2. Run database migration: psql -f database/gpu-schema-migration.sql');
            console.log('  3. Restart services: START-INTEGRATED-SYSTEM.bat');
            console.log('  4. Re-run this check');
        } else {
            console.log('  1. Monitor performance metrics');
            console.log('  2. Run load testing');
            console.log('  3. Configure production deployment');
        }
    }
}

// Run the check
const checker = new SystemIntegrationChecker();
checker.runCheck().catch(console.error);