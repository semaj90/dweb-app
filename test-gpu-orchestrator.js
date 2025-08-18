// Simple test for GPU orchestrator components
const { spawn } = require('child_process');
const path = require('path');

console.log('🧪 Testing GPU Orchestrator Components...');

// Test 1: Check if Node.js works
console.log('✅ Node.js is working');

// Test 2: Check CUDA worker exists
const cudaWorkerPath = path.join(__dirname, 'cuda-worker', 'cuda-worker.exe');
const fs = require('fs');

if (fs.existsSync(cudaWorkerPath)) {
    console.log('✅ CUDA worker executable found');
    
    // Test 3: Try to run CUDA worker
    const testData = JSON.stringify({
        jobId: 'test-123',
        type: 'embedding',
        data: [1, 2, 3, 4]
    });
    
    console.log('🧪 Testing CUDA worker execution...');
    const child = spawn(cudaWorkerPath, [], { stdio: ['pipe', 'pipe', 'pipe'] });
    
    child.stdin.write(testData);
    child.stdin.end();
    
    let output = '';
    child.stdout.on('data', (data) => output += data.toString());
    
    child.on('close', (code) => {
        if (code === 0 && output.trim()) {
            try {
                const result = JSON.parse(output.trim());
                console.log('✅ CUDA worker test successful');
                console.log('📊 Result:', result);
            } catch (e) {
                console.log('⚠️ CUDA worker returned non-JSON:', output.trim());
            }
        } else {
            console.log('❌ CUDA worker failed with code:', code);
        }
        
        // Test 4: Check dependencies availability
        console.log('\n📦 Checking required packages...');
        try {
            require('cluster');
            console.log('✅ cluster module available');
            
            require('os');
            console.log('✅ os module available');
            
            console.log('\n🎯 GPU Orchestrator basic components are ready!');
            console.log('\n📋 Next steps:');
            console.log('1. Install missing dependencies');
            console.log('2. Setup Redis and RabbitMQ services');
            console.log('3. Compile CUDA worker if needed');
            console.log('4. Start the orchestrator system');
            
        } catch (error) {
            console.log('❌ Missing core dependencies:', error.message);
        }
    });
    
} else {
    console.log('⚠️ CUDA worker not found - needs compilation');
    console.log('📍 Expected location:', cudaWorkerPath);
    
    // Check if NVCC is available
    const nvccTest = spawn('nvcc', ['--version'], { stdio: 'pipe' });
    nvccTest.on('close', (code) => {
        if (code === 0) {
            console.log('✅ NVCC compiler available - can build CUDA worker');
        } else {
            console.log('❌ NVCC not found - install CUDA Toolkit');
        }
    });
}

// Test 5: Check current system status
console.log('\n🏥 System Status Check:');
console.log('- Working Directory:', __dirname);
console.log('- Node.js Version:', process.version);
console.log('- Platform:', process.platform);
console.log('- Architecture:', process.arch);

// Test 6: Check package.json
try {
    const packageJson = require('./package.json');
    console.log('✅ package.json found');
    console.log('📦 Project:', packageJson.name);
    console.log('🔢 Version:', packageJson.version);
} catch (error) {
    console.log('❌ package.json issue:', error.message);
}