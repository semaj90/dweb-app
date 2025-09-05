// Comprehensive System Test Script
import fetch from 'node-fetch';
import { exec } from 'child_process';
import { promisify } from 'util';
import { readFileSync, writeFileSync } from 'fs';

const execAsync = promisify(exec);

console.log('🔍 Running Comprehensive System Tests...\n');

const tests = {
    docker: { passed: false, message: '' },
    postgres: { passed: false, message: '' },
    ollama: { passed: false, message: '' },
    qdrant: { passed: false, message: '' },
    redis: { passed: false, message: '' },
    typescript: { passed: false, message: '' },
    frontend: { passed: false, message: '' },
    llmIntegration: { passed: false, message: '' }
};

// Test 1: Docker Services
async function testDocker() {
    try {
        const { stdout } = await execAsync('docker ps --format "{{.Names}}"');
        const runningContainers = stdout.trim().split('\n');
        
        const requiredContainers = [
            'prosecutor_postgres',
            'prosecutor_ollama',
            'prosecutor_qdrant',
            'prosecutor_redis'
        ];
        
        const allRunning = requiredContainers.every(container => 
            runningContainers.some(running => running.includes(container))
        );
        
        tests.docker.passed = allRunning;
        tests.docker.message = allRunning 
            ? 'All Docker services are running'
            : `Missing containers: ${requiredContainers.filter(c => !runningContainers.some(r => r.includes(c)))}`;
    } catch (error) {
        tests.docker.message = `Docker test failed: ${error.message}`;
    }
}

// Test 2: PostgreSQL Connection
async function testPostgres() {
    try {
        const { stdout } = await execAsync('docker exec prosecutor_postgres pg_isready -U postgres');
        tests.postgres.passed = stdout.includes('accepting connections');
        tests.postgres.message = tests.postgres.passed 
            ? 'PostgreSQL is ready and accepting connections'
            : 'PostgreSQL is not ready';
    } catch (error) {
        tests.postgres.message = `PostgreSQL test failed: ${error.message}`;
    }
}

// Test 3: Ollama API
async function testOllama() {
    try {
        const response = await fetch('http://localhost:11434/api/tags');
        const data = await response.json();
        
        tests.ollama.passed = response.ok && data.models && data.models.length > 0;
        tests.ollama.message = tests.ollama.passed
            ? `Ollama is running with ${data.models.length} models: ${data.models.map(m => m.name).join(', ')}`
            : 'Ollama is running but no models are installed';
    } catch (error) {
        tests.ollama.message = `Ollama test failed: ${error.message}`;
    }
}

// Test 4: Qdrant Vector Database
async function testQdrant() {
    try {
        const response = await fetch('http://localhost:6333/');
        tests.qdrant.passed = response.ok;
        tests.qdrant.message = tests.qdrant.passed
            ? 'Qdrant vector database is accessible'
            : 'Qdrant is not responding';
    } catch (error) {
        tests.qdrant.message = `Qdrant test failed: ${error.message}`;
    }
}

// Test 5: Redis Cache
async function testRedis() {
    try {
        const { stdout, stderr } = await execAsync('docker exec prosecutor_redis redis-cli ping');
        tests.redis.passed = stdout.trim() === 'PONG';
        tests.redis.message = tests.redis.passed
            ? 'Redis cache is responding'
            : 'Redis is not responding correctly';
    } catch (error) {
        tests.redis.message = `Redis test failed: ${error.message}`;
    }
}

// Test 6: TypeScript Compilation
async function testTypeScript() {
    try {
        process.chdir('sveltekit-frontend');
        const { stdout, stderr } = await execAsync('npm run check');
        
        // Check if there are any errors in the output
        const hasErrors = stderr.includes('Error:') || stdout.includes('Error:');
        tests.typescript.passed = !hasErrors;
        tests.typescript.message = tests.typescript.passed
            ? 'TypeScript compilation successful, no errors found'
            : 'TypeScript compilation has errors';
        
        process.chdir('..');
    } catch (error) {
        tests.typescript.message = `TypeScript test failed: ${error.message}`;
        process.chdir('..');
    }
}

// Test 7: Frontend Build
async function testFrontend() {
    try {
        // Check if dependencies are installed
        const packageJson = JSON.parse(readFileSync('./sveltekit-frontend/package.json', 'utf-8'));
        const hasRequiredDeps = [
            '@sveltejs/kit',
            'svelte',
            'drizzle-orm',
            'vite'
        ].every(dep => packageJson.dependencies[dep] || packageJson.devDependencies[dep]);
        
        tests.frontend.passed = hasRequiredDeps;
        tests.frontend.message = hasRequiredDeps
            ? 'Frontend dependencies are properly installed'
            : 'Some frontend dependencies are missing';
    } catch (error) {
        tests.frontend.message = `Frontend test failed: ${error.message}`;
    }
}

// Test 8: LLM Integration
async function testLLMIntegration() {
    try {
        const testPrompt = {
            model: "gemma3",
            prompt: "Respond with just 'OK' if you receive this message.",
            stream: false,
            options: {
                temperature: 0.1,
                num_predict: 10
            }
        };
        
        const response = await fetch('http://localhost:11434/api/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(testPrompt)
        });
        
        const data = await response.json();
        tests.llmIntegration.passed = response.ok && data.response && data.response.includes('OK');
        tests.llmIntegration.message = tests.llmIntegration.passed
            ? `LLM integration working: "${data.response.trim()}"`
            : 'LLM integration failed to respond correctly';
    } catch (error) {
        tests.llmIntegration.message = `LLM integration test failed: ${error.message}`;
    }
}

// Run all tests
async function runAllTests() {
    console.log('Running system tests...\n');
    
    await testDocker();
    await testPostgres();
    await testOllama();
    await testQdrant();
    await testRedis();
    await testTypeScript();
    await testFrontend();
    await testLLMIntegration();
    
    // Generate report
    console.log('\n📊 TEST RESULTS\n' + '='.repeat(50));
    
    let passedCount = 0;
    for (const [testName, result] of Object.entries(tests)) {
        const status = result.passed ? '✅ PASS' : '❌ FAIL';
        const name = testName.charAt(0).toUpperCase() + testName.slice(1);
        console.log(`${status} ${name}: ${result.message}`);
        if (result.passed) passedCount++;
    }
    
    console.log('\n' + '='.repeat(50));
    console.log(`\n📈 Summary: ${passedCount}/${Object.keys(tests).length} tests passed`);
    
    // Generate detailed report file
    const report = {
        timestamp: new Date().toISOString(),
        summary: {
            total: Object.keys(tests).length,
            passed: passedCount,
            failed: Object.keys(tests).length - passedCount
        },
        tests: tests,
        recommendations: []
    };
    
    // Add recommendations based on failures
    if (!tests.docker.passed) {
        report.recommendations.push('Ensure Docker Desktop is running and all containers are started');
    }
    if (!tests.ollama.passed) {
        report.recommendations.push('Run: docker exec prosecutor_ollama ollama pull gemma3');
    }
    if (!tests.typescript.passed) {
        report.recommendations.push('Run: cd sveltekit-frontend && node fix-all-typescript-errors.mjs');
    }
    
    writeFileSync('system-test-report.json', JSON.stringify(report, null, 2));
    console.log('\n📄 Detailed report saved to: system-test-report.json');
    
    // Exit code based on test results
    process.exit(passedCount === Object.keys(tests).length ? 0 : 1);
}

// Run tests
runAllTests().catch(error => {
    console.error('Test runner failed:', error);
    process.exit(1);
});
