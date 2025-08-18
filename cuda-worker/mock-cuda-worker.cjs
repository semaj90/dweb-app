#!/usr/bin/env node
// Mock CUDA worker for development when NVCC is not available
// Simulates GPU processing with CPU-based operations

function mockEmbedding(data) {
    // Simple transformation that simulates embedding
    return data.map((x, i) => x * 1.2345 + Math.sin(i * 0.1));
}

function mockSimilarity(vec1, vec2) {
    // Element-wise similarity computation
    return vec1.map((a, i) => a * vec2[i]);
}

function mockAutoIndex(data) {
    const processed = mockEmbedding(data);
    processed.push(Date.now() / 1000); // timestamp
    processed.push(data.length);       // original size
    processed.push(1.0);               // auto-index flag
    return processed;
}

function processJob(job) {
    const { jobId, type, data } = job;
    
    let result;
    switch (type) {
        case 'embedding':
            result = mockEmbedding(data);
            break;
        case 'similarity':
            // Split data into two vectors at midpoint
            const mid = Math.floor(data.length / 2);
            const vec1 = data.slice(0, mid);
            const vec2 = data.slice(mid);
            result = mockSimilarity(vec1, vec2);
            break;
        case 'autoindex':
            result = mockAutoIndex(data);
            break;
        default:
            result = mockEmbedding(data);
    }
    
    return {
        jobId,
        type,
        vector: result,
        status: 'success',
        timestamp: Date.now(),
        processingTime: Math.floor(Math.random() * 50) + 10, // 10-60ms
        workerId: 'mock-cuda-worker',
        device: 'CPU-Simulation'
    };
}

// Main execution
try {
    let input = '';
    
    // Read from stdin
    process.stdin.setEncoding('utf8');
    process.stdin.on('readable', () => {
        const chunk = process.stdin.read();
        if (chunk !== null) {
            input += chunk;
        }
    });
    
    process.stdin.on('end', () => {
        try {
            if (!input.trim()) {
                console.error('No input received');
                process.exit(1);
            }
            
            const job = JSON.parse(input.trim());
            const result = processJob(job);
            
            // Output JSON result
            console.log(JSON.stringify(result));
            process.exit(0);
            
        } catch (error) {
            const errorResponse = {
                jobId: 'error',
                error: error.message,
                status: 'failed',
                timestamp: Date.now()
            };
            console.log(JSON.stringify(errorResponse));
            process.exit(1);
        }
    });
    
} catch (error) {
    console.error('Mock CUDA worker error:', error.message);
    process.exit(1);
}