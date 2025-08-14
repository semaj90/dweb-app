// test-simd-redis-vite.mjs
// Comprehensive test suite for SIMD Redis Vite integration

import fetch from 'node-fetch';
import WebSocket from 'ws';
import { performance } from 'perf_hooks';

const BASE_URL = 'http://localhost:8080';
const WS_URL = 'ws://localhost:8080/ws';

// Color codes for console output
const colors = {
    reset: '\x1b[0m',
    bright: '\x1b[1m',
    green: '\x1b[32m',
    yellow: '\x1b[33m',
    blue: '\x1b[34m',
    magenta: '\x1b[35m',
    cyan: '\x1b[36m',
    red: '\x1b[31m'
};

// Test data generators
function generateLargeJSON(size = 1000) {
    const data = {
        timestamp: Date.now(),
        records: []
    };
    
    for (let i = 0; i < size; i++) {
        data.records.push({
            id: `record_${i}`,
            name: `Test Record ${i}`,
            value: Math.random() * 1000,
            nested: {
                field1: `value_${i}_1`,
                field2: `value_${i}_2`,
                array: Array(10).fill(0).map((_, j) => ({
                    index: j,
                    data: `nested_${i}_${j}`
                }))
            },
            metadata: {
                created: new Date().toISOString(),
                tags: [`tag_${i % 10}`, `category_${i % 5}`],
                priority: i % 3
            }
        });
    }
    
    return data;
}

function generateLegalDocument() {
    return {
        case_id: `CASE-${Date.now()}`,
        title: "Legal Document Analysis Test",
        content: {
            summary: "This is a test legal document for SIMD processing validation.",
            sections: [
                {
                    heading: "Introduction",
                    text: "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
                    citations: ["Case A v. B", "State v. C"]
                },
                {
                    heading: "Evidence",
                    text: "The evidence clearly shows that the defendant...",
                    exhibits: ["Exhibit A", "Exhibit B", "Exhibit C"]
                },
                {
                    heading: "Legal Analysis",
                    text: "Based on precedent established in similar cases...",
                    references: ["Law Code 123.45", "Statute 678.90"]
                }
            ],
            metadata: {
                court: "District Court",
                date_filed: new Date().toISOString(),
                parties: ["Plaintiff A", "Defendant B"],
                attorneys: ["Attorney X", "Attorney Y"]
            }
        }
    };
}

// Test functions
async function testHealthCheck() {
    console.log(`${colors.cyan}Testing Health Check...${colors.reset}`);
    
    try {
        const response = await fetch(`${BASE_URL}/health`);
        const data = await response.json();
        
        console.log(`${colors.green}✅ Health Check Response:${colors.reset}`);
        console.log(JSON.stringify(data, null, 2));
        
        if (data.status === 'healthy' && data.simd === true) {
            console.log(`${colors.green}✅ Server is healthy with SIMD support${colors.reset}`);
            return true;
        }
    } catch (error) {
        console.error(`${colors.red}❌ Health check failed:${colors.reset}`, error.message);
        return false;
    }
}

async function testSIMDParsing() {
    console.log(`\n${colors.cyan}Testing SIMD JSON Parsing...${colors.reset}`);
    
    const testData = generateLargeJSON(500);
    const jsonStr = JSON.stringify(testData);
    
    console.log(`  Data size: ${(jsonStr.length / 1024).toFixed(2)} KB`);
    
    try {
        const start = performance.now();
        const response = await fetch(`${BASE_URL}/simd-parse?key=test_simd_${Date.now()}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: jsonStr
        });
        const result = await response.json();
        const elapsed = performance.now() - start;
        
        console.log(`${colors.green}✅ SIMD Parse Results:${colors.reset}`);
        console.log(`  Parse Time: ${(result.parse_time_ns / 1000000).toFixed(3)} ms`);
        console.log(`  Request Time: ${elapsed.toFixed(2)} ms`);
        console.log(`  Cached: ${result.cached}`);
        console.log(`  Success: ${result.success}`);
        
        return result.success;
    } catch (error) {
        console.error(`${colors.red}❌ SIMD parsing failed:${colors.reset}`, error.message);
        return false;
    }
}

async function testBatchProcessing() {
    console.log(`\n${colors.cyan}Testing Batch SIMD Processing...${colors.reset}`);
    
    const batchSize = 10;
    const batch = Array(batchSize).fill(0).map((_, i) => 
        generateLargeJSON(100 + i * 10)
    );
    
    try {
        const start = performance.now();
        const response = await fetch(`${BASE_URL}/simd-batch`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(batch)
        });
        const result = await response.json();
        const elapsed = performance.now() - start;
        
        console.log(`${colors.green}✅ Batch Processing Results:${colors.reset}`);
        console.log(`  Batch Size: ${result.batch_size}`);
        console.log(`  Total Time: ${elapsed.toFixed(2)} ms`);
        console.log(`  Avg Time per Item: ${(elapsed / batchSize).toFixed(2)} ms`);
        
        // Check individual results
        const successful = result.results.filter(r => r.success).length;
        console.log(`  Successful: ${successful}/${batchSize}`);
        
        if (result.results.length > 0) {
            const avgParseTime = result.results.reduce((sum, r) => 
                sum + (r.parse_time_ns || 0), 0) / result.results.length;
            console.log(`  Avg Parse Time: ${(avgParseTime / 1000000).toFixed(3)} ms`);
        }
        
        return successful === batchSize;
    } catch (error) {
        console.error(`${colors.red}❌ Batch processing failed:${colors.reset}`, error.message);
        return false;
    }
}

async function testDocumentProcessing() {
    console.log(`\n${colors.cyan}Testing Document Processing...${colors.reset}`);
    
    const document = {
        document_id: `DOC-${Date.now()}`,
        content: generateLegalDocument(),
        metadata: {
            type: "legal_brief",
            author: "Test System",
            created_at: new Date().toISOString()
        }
    };
    
    try {
        const start = performance.now();
        const response = await fetch(`${BASE_URL}/process-document`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(document)
        });
        const result = await response.json();
        const elapsed = performance.now() - start;
        
        console.log(`${colors.green}✅ Document Processing Results:${colors.reset}`);
        console.log(`  Document ID: ${result.document_id}`);
        console.log(`  Processed: ${result.processed}`);
        console.log(`  Parse Time: ${(result.parse_time / 1000000).toFixed(3)} ms`);
        console.log(`  Total Time: ${elapsed.toFixed(2)} ms`);
        console.log(`  Cached: ${result.cached}`);
        
        return result.processed;
    } catch (error) {
        console.error(`${colors.red}❌ Document processing failed:${colors.reset}`, error.message);
        return false;
    }
}

async function testLegalAnalysis() {
    console.log(`\n${colors.cyan}Testing Legal Analysis Endpoint...${colors.reset}`);
    
    const analysisRequest = {
        case_id: `CASE-${Date.now()}`,
        documents: [
            generateLegalDocument(),
            generateLegalDocument(),
            generateLegalDocument()
        ],
        analysis_type: "precedent_analysis"
    };
    
    try {
        const start = performance.now();
        const response = await fetch(`${BASE_URL}/legal/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(analysisRequest)
        });
        const result = await response.json();
        const elapsed = performance.now() - start;
        
        console.log(`${colors.green}✅ Legal Analysis Results:${colors.reset}`);
        console.log(`  Case ID: ${result.case_id}`);
        console.log(`  Analysis Type: ${result.analysis_type}`);
        console.log(`  Documents Processed: ${result.documents}`);
        console.log(`  Total Time: ${elapsed.toFixed(2)} ms`);
        
        // Check processing results
        const successful = result.results.filter(r => r.success).length;
        console.log(`  Successful: ${successful}/${result.documents}`);
        
        return successful === result.documents;
    } catch (error) {
        console.error(`${colors.red}❌ Legal analysis failed:${colors.reset}`, error.message);
        return false;
    }
}

async function testCaching() {
    console.log(`\n${colors.cyan}Testing Cache Performance...${colors.reset}`);
    
    const cacheKey = `cache_test_${Date.now()}`;
    const testData = generateLargeJSON(200);
    
    try {
        // First request (should cache)
        console.log(`  Making first request (uncached)...`);
        const start1 = performance.now();
        const response1 = await fetch(`${BASE_URL}/simd-parse?key=${cacheKey}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(testData)
        });
        const result1 = await response1.json();
        const time1 = performance.now() - start1;
        
        console.log(`    Time: ${time1.toFixed(2)} ms`);
        console.log(`    Cached: ${result1.cached} (should be false)`);
        
        // Second request (should be cached)
        console.log(`  Making second request (cached)...`);
        const start2 = performance.now();
        const response2 = await fetch(`${BASE_URL}/simd-parse?key=${cacheKey}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(testData)
        });
        const result2 = await response2.json();
        const time2 = performance.now() - start2;
        
        console.log(`    Time: ${time2.toFixed(2)} ms`);
        console.log(`    Cached: ${result2.cached} (should be true)`);
        
        const speedup = (time1 / time2).toFixed(2);
        console.log(`${colors.green}✅ Cache Speedup: ${speedup}x faster${colors.reset}`);
        
        // Clean up
        await fetch(`${BASE_URL}/cache/${cacheKey}`, { method: 'DELETE' });
        
        return result2.cached === true && time2 < time1;
    } catch (error) {
        console.error(`${colors.red}❌ Cache test failed:${colors.reset}`, error.message);
        return false;
    }
}

async function testMetrics() {
    console.log(`\n${colors.cyan}Testing Metrics Endpoint...${colors.reset}`);
    
    try {
        const response = await fetch(`${BASE_URL}/metrics`);
        const metrics = await response.json();
        
        console.log(`${colors.green}✅ Current Metrics:${colors.reset}`);
        console.log(`  Parse Count: ${metrics.parse_count}`);
        console.log(`  Cache Hits: ${metrics.cache_hits}`);
        console.log(`  Cache Misses: ${metrics.cache_misses}`);
        console.log(`  Avg Parse Time: ${(metrics.avg_parse_time_ns / 1000000).toFixed(3)} ms`);
        console.log(`  Active Workers: ${metrics.worker_pool_active}`);
        console.log(`  Connected Clients: ${metrics.connected_clients}`);
        console.log(`  Total Workers: ${metrics.workers}`);
        
        return true;
    } catch (error) {
        console.error(`${colors.red}❌ Metrics retrieval failed:${colors.reset}`, error.message);
        return false;
    }
}

async function testWebSocket() {
    console.log(`\n${colors.cyan}Testing WebSocket Connection...${colors.reset}`);
    
    return new Promise((resolve) => {
        const ws = new WebSocket(WS_URL);
        let messageCount = 0;
        const timeout = setTimeout(() => {
            ws.close();
            console.log(`${colors.yellow}⚠️  WebSocket timeout after 5 seconds${colors.reset}`);
            resolve(messageCount > 0);
        }, 5000);
        
        ws.on('open', () => {
            console.log(`${colors.green}✅ WebSocket connected${colors.reset}`);
        });
        
        ws.on('message', (data) => {
            messageCount++;
            const message = JSON.parse(data.toString());
            
            if (message.type === 'metrics') {
                console.log(`  Received metrics update #${messageCount}`);
                console.log(`    Active Workers: ${message.worker_pool_active}`);
                console.log(`    Parse Count: ${message.parse_count}`);
                
                if (messageCount >= 2) {
                    clearTimeout(timeout);
                    ws.close();
                    console.log(`${colors.green}✅ WebSocket real-time updates working${colors.reset}`);
                    resolve(true);
                }
            }
        });
        
        ws.on('error', (error) => {
            console.error(`${colors.red}❌ WebSocket error:${colors.reset}`, error.message);
            clearTimeout(timeout);
            resolve(false);
        });
        
        ws.on('close', () => {
            console.log(`  WebSocket closed`);
        });
    });
}

async function testPerformanceBenchmark() {
    console.log(`\n${colors.cyan}Running Performance Benchmark...${colors.reset}`);
    
    const sizes = [100, 500, 1000, 2000];
    const results = [];
    
    for (const size of sizes) {
        const data = generateLargeJSON(size);
        const jsonStr = JSON.stringify(data);
        const sizeKB = (jsonStr.length / 1024).toFixed(2);
        
        const times = [];
        const iterations = 3;
        
        for (let i = 0; i < iterations; i++) {
            const start = performance.now();
            const response = await fetch(`${BASE_URL}/simd-parse?key=bench_${size}_${i}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: jsonStr
            });
            const result = await response.json();
            const elapsed = performance.now() - start;
            
            if (result.success && !result.cached) {
                times.push(elapsed);
            }
        }
        
        if (times.length > 0) {
            const avgTime = times.reduce((a, b) => a + b, 0) / times.length;
            const throughput = (parseFloat(sizeKB) / (avgTime / 1000)).toFixed(2);
            
            results.push({
                records: size,
                sizeKB: sizeKB,
                avgTime: avgTime.toFixed(2),
                throughput: throughput
            });
            
            console.log(`  ${size} records (${sizeKB} KB): ${avgTime.toFixed(2)} ms | ${throughput} KB/s`);
        }
    }
    
    console.log(`\n${colors.green}✅ Benchmark Summary:${colors.reset}`);
    console.table(results);
    
    return true;
}

// Main test runner
async function runAllTests() {
    console.log(`${colors.bright}${colors.magenta}
╔══════════════════════════════════════════════════════════════╗
║     SIMD JSON + Redis + Vite Integration Test Suite         ║
╚══════════════════════════════════════════════════════════════╝
${colors.reset}`);
    
    const tests = [
        { name: 'Health Check', fn: testHealthCheck },
        { name: 'SIMD Parsing', fn: testSIMDParsing },
        { name: 'Batch Processing', fn: testBatchProcessing },
        { name: 'Document Processing', fn: testDocumentProcessing },
        { name: 'Legal Analysis', fn: testLegalAnalysis },
        { name: 'Caching', fn: testCaching },
        { name: 'Metrics', fn: testMetrics },
        { name: 'WebSocket', fn: testWebSocket },
        { name: 'Performance Benchmark', fn: testPerformanceBenchmark }
    ];
    
    const results = [];
    let passed = 0;
    let failed = 0;
    
    for (const test of tests) {
        try {
            const success = await test.fn();
            results.push({ test: test.name, status: success ? 'PASSED' : 'FAILED' });
            if (success) passed++;
            else failed++;
        } catch (error) {
            console.error(`${colors.red}Test "${test.name}" threw an error:${colors.reset}`, error.message);
            results.push({ test: test.name, status: 'ERROR' });
            failed++;
        }
    }
    
    // Print summary
    console.log(`\n${colors.bright}${colors.blue}
╔══════════════════════════════════════════════════════════════╗
║                        TEST SUMMARY                          ║
╚══════════════════════════════════════════════════════════════╝
${colors.reset}`);
    
    console.table(results);
    
    console.log(`\n${colors.bright}Total: ${tests.length} | ${colors.green}Passed: ${passed}${colors.reset} | ${colors.red}Failed: ${failed}${colors.reset}`);
    
    if (failed === 0) {
        console.log(`\n${colors.bright}${colors.green}🎉 All tests passed successfully!${colors.reset}`);
    } else {
        console.log(`\n${colors.bright}${colors.yellow}⚠️  Some tests failed. Please check the logs above.${colors.reset}`);
    }
    
    process.exit(failed === 0 ? 0 : 1);
}

// Check if server is running before starting tests
async function checkServerAndRun() {
    try {
        const response = await fetch(`${BASE_URL}/health`);
        if (response.ok) {
            console.log(`${colors.green}✅ Server is running. Starting tests...${colors.reset}\n`);
            await runAllTests();
        }
    } catch (error) {
        console.error(`${colors.red}❌ Server is not running at ${BASE_URL}${colors.reset}`);
        console.error(`Please start the server first with: go run simd-redis-vite-server.go`);
        process.exit(1);
    }
}

// Run tests
checkServerAndRun();
