// Simple test for GPU orchestrator components
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

console.log('🧪 Testing GPU Orchestrator Components...');

// Test 1: Check if Node.js works
console.log('✅ Node.js is working');

// Test 2: Check CUDA worker exists
const cudaWorkerPath = path.join(__dirname, 'cuda-worker', 'cuda-worker.exe');

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
        finishTests();
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
        finishTests();
    });
}

function finishTests() {
    // Test 4: Check dependencies availability
    console.log('\n📦 Checking required modules...');
    try {
        require('cluster');
        console.log('✅ cluster module available');
        
        require('os');
        console.log('✅ os module available');
        
        // Check if XState dependencies exist
        try {
            // These will likely fail but that's expected
            require('xstate');
            console.log('✅ xstate available');
        } catch (e) {
            console.log('⚠️ xstate not installed (expected)');
        }
        
        try {
            require('amqplib');
            console.log('✅ amqplib available');
        } catch (e) {
            console.log('⚠️ amqplib not installed (expected)');
        }
        
        try {
            require('ioredis');
            console.log('✅ ioredis available');
        } catch (e) {
            console.log('⚠️ ioredis not installed (expected)');
        }
        
    } catch (error) {
        console.log('❌ Missing core dependencies:', error.message);
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
    
    console.log('\n🎯 GPU Orchestrator Test Summary:');
    console.log('\n📋 Next steps:');
    console.log('1. Install missing dependencies: npm install amqplib ioredis xstate');
    console.log('2. Setup Redis and RabbitMQ services');
    console.log('3. Compile CUDA worker if needed');
    console.log('4. Start the orchestrator system');
    console.log('\n🚀 Ready to proceed with full orchestrator setup!');
}