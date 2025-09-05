// test-enhanced-rag-v2.mjs
// Quick test script for Enhanced RAG V2 integration

import fetch from 'node-fetch';

const BASE_URL = 'http://localhost:8097';
const TEST_USER = 'test-user-' + Date.now();

console.log('🧪 Testing Enhanced RAG V2 Integration...\n');

async function testEndpoint(name, method, endpoint, body = null) {
    console.log(`Testing: ${name}`);
    
    try {
        const options = {
            method,
            headers: {
                'Content-Type': 'application/json'
            }
        };
        
        if (body) {
            options.body = JSON.stringify(body);
        }
        
        const response = await fetch(`${BASE_URL}${endpoint}`, options);
        const data = await response.json();
        
        if (response.ok) {
            console.log(`  ✅ ${name} - Status: ${response.status}`);
            console.log(`     Response:`, JSON.stringify(data).substring(0, 100) + '...\n');
            return data;
        } else {
            console.log(`  ❌ ${name} - Status: ${response.status}`);
            console.log(`     Error:`, data, '\n');
            return null;
        }
    } catch (error) {
        console.log(`  ❌ ${name} - Error: ${error.message}\n`);
        return null;
    }
}

async function runTests() {
    // Test 1: Health Check
    await testEndpoint('Health Check', 'GET', '/health');
    
    // Test 2: Create User Intent
    const intent = await testEndpoint('Create User Intent', 'POST', '/api/intents', {
        user_id: TEST_USER,
        intent: 'contract_review',
        keywords: ['liability', 'indemnification', 'warranty'],
        confidence: 0.92,
        context: {
            document_type: 'SaaS Agreement',
            test: true
        }
    });
    
    // Test 3: List User Intents
    await testEndpoint('List User Intents', 'GET', `/api/intents?user_id=${TEST_USER}`);
    
    // Test 4: Create Todo Item
    const todo = await testEndpoint('Create Todo Item', 'POST', '/api/todos', {
        user_id: TEST_USER,
        title: 'Review Contract Section 3.2',
        description: 'Check liability limitations and indemnification clauses',
        priority: 8,
        status: 'pending'
    });
    
    // Test 5: Generate Recommendations
    await testEndpoint('Generate Recommendations', 'POST', '/api/recommendations/generate', {
        user_id: TEST_USER,
        context: {
            current_task: 'contract_review',
            document_type: 'legal'
        }
    });
    
    // Test 6: Auto-Solve Todos
    await testEndpoint('Auto-Solve Todos', 'POST', '/api/todos/solve', {
        user_id: TEST_USER,
        auto_solve: true
    });
    
    // Test 7: Update Session
    await testEndpoint('Update User Session', 'POST', '/api/sessions', {
        user_id: TEST_USER,
        state: 'active',
        last_activity: new Date().toISOString(),
        context: {
            current_view: 'dashboard',
            active_document: 'test.pdf'
        }
    });
    
    // Test 8: Track Analytics Event
    await testEndpoint('Track Analytics Event', 'POST', '/api/analytics/event', {
        user_id: TEST_USER,
        event_type: 'page_view',
        event_data: {
            page: 'test',
            duration: 1500
        }
    });
    
    // Test 9: Get Session
    await testEndpoint('Get User Session', 'GET', `/api/sessions/${TEST_USER}`);
    
    // Test 10: List Pending Todos
    await testEndpoint('List Pending Todos', 'GET', `/api/todos/pending?user_id=${TEST_USER}`);
    
    // Cleanup: Delete test data
    if (intent && intent.id) {
        await testEndpoint('Delete Test Intent', 'DELETE', `/api/intents/${intent.id}`);
    }
    
    if (todo && todo.id) {
        await testEndpoint('Delete Test Todo', 'DELETE', `/api/todos/${todo.id}`);
    }
    
    console.log('========================================');
    console.log('✅ All tests completed!');
    console.log('========================================\n');
    
    // Test WebSocket connection
    console.log('Testing WebSocket connection...');
    try {
        const WebSocket = (await import('ws')).default;
        const ws = new WebSocket('ws://localhost:8097/ws');
        
        ws.on('open', () => {
            console.log('✅ WebSocket connected successfully');
            ws.close();
        });
        
        ws.on('error', (error) => {
            console.log('❌ WebSocket error:', error.message);
        });
        
        setTimeout(() => {
            ws.terminate();
            process.exit(0);
        }, 3000);
    } catch (error) {
        console.log('⚠️  WebSocket test skipped (ws package not installed)');
        process.exit(0);
    }
}

// Check if service is running
fetch(`${BASE_URL}/health`)
    .then(response => {
        if (response.ok) {
            console.log('✅ Enhanced RAG V2 service is running!\n');
            runTests();
        } else {
            throw new Error('Service not healthy');
        }
    })
    .catch(error => {
        console.log('❌ Enhanced RAG V2 service is not running!');
        console.log('   Please start it with: .\\RUN-ENHANCED-RAG-V2-COMPLETE.bat\n');
        process.exit(1);
    });
