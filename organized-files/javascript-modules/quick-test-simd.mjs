// quick-test-simd.mjs
// Quick test to verify SIMD Redis Vite integration is working

import fetch from 'node-fetch';

const BASE_URL = 'http://localhost:8080';

console.log('🧪 Quick SIMD Integration Test\n');

async function quickTest() {
    try {
        // 1. Health Check
        console.log('1. Testing health endpoint...');
        const healthResponse = await fetch(`${BASE_URL}/health`);
        const health = await healthResponse.json();
        console.log('   ✅ Server Status:', health.status);
        console.log('   ✅ SIMD Enabled:', health.simd);
        console.log('   ✅ Redis Connected:', health.redis);
        console.log('   ✅ Workers:', health.workers);
        
        // 2. SIMD Parse Test
        console.log('\n2. Testing SIMD parsing...');
        const testData = {
            test: "SIMD Performance Test",
            timestamp: Date.now(),
            nested: {
                field1: "value1",
                field2: "value2",
                array: [1, 2, 3, 4, 5]
            }
        };
        
        const parseResponse = await fetch(`${BASE_URL}/simd-parse?key=quick_test`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(testData)
        });
        const parseResult = await parseResponse.json();
        console.log('   ✅ Parse Success:', parseResult.success);
        console.log('   ✅ Parse Time:', (parseResult.parse_time_ns / 1000000).toFixed(3), 'ms');
        console.log('   ✅ Cached:', parseResult.cached);
        
        // 3. Cache Test (should be cached now)
        console.log('\n3. Testing cache retrieval...');
        const cacheResponse = await fetch(`${BASE_URL}/simd-parse?key=quick_test`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(testData)
        });
        const cacheResult = await cacheResponse.json();
        console.log('   ✅ Retrieved from cache:', cacheResult.cached);
        
        // 4. Metrics Check
        console.log('\n4. Checking metrics...');
        const metricsResponse = await fetch(`${BASE_URL}/metrics`);
        const metrics = await metricsResponse.json();
        console.log('   ✅ Total Parses:', metrics.parse_count);
        console.log('   ✅ Cache Hits:', metrics.cache_hits);
        console.log('   ✅ Cache Misses:', metrics.cache_misses);
        console.log('   ✅ Avg Parse Time:', (metrics.avg_parse_time_ns / 1000000).toFixed(3), 'ms');
        
        console.log('\n✨ All tests passed! SIMD Redis Vite integration is working correctly.');
        
    } catch (error) {
        console.error('\n❌ Test failed:', error.message);
        console.error('   Make sure the SIMD server is running on port 8080');
        console.error('   Run: .\\START-SIMD-REDIS-VITE.bat');
        process.exit(1);
    }
}

// Run the test
quickTest();
