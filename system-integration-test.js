// Comprehensive System Integration Test
// Tests all 5 core services of the Legal AI Platform

const axios = require('axios');
const fs = require('fs');

class LegalAISystemTest {
    constructor() {
        this.services = {
            sveltekit: 'http://localhost:5173',
            goLlama: 'http://localhost:4101',
            enhancedRAG: 'http://localhost:8094',
            uploadService: 'http://localhost:8093',
            ollama: 'http://localhost:11434'
        };
        
        this.results = {};
    }

    async runAllTests() {
        console.log('🧪 Starting Comprehensive Legal AI Platform Integration Tests\n');
        
        try {
            // Test 1: SvelteKit Custom JSON API
            await this.testSvelteKitJSONAPI();
            
            // Test 2: Go Llama Integration Service  
            await this.testGoLlamaIntegration();
            
            // Test 3: Enhanced RAG Service
            await this.testEnhancedRAG();
            
            // Test 4: Upload Service
            await this.testUploadService();
            
            // Test 5: Ollama AI Models
            await this.testOllamaModels();
            
            // Test 6: End-to-End Integration
            await this.testEndToEndIntegration();
            
            this.printSummary();
            
        } catch (error) {
            console.error('❌ System test failed:', error.message);
            process.exit(1);
        }
    }

    async testSvelteKitJSONAPI() {
        console.log('🔄 Testing SvelteKit Custom JSON API...');
        
        const testData = {
            ocrData: {
                filename: 'test-contract.pdf',
                text: 'This is a test legal contract with various clauses and provisions.',
                pages: 1,
                totalCharacters: 73,
                averageConfidence: 95,
                extractedAt: new Date().toISOString()
            }
        };
        
        try {
            const response = await axios.post(
                `${this.services.sveltekit}/api/convert/to-json`,
                testData,
                { timeout: 10000 }
            );
            
            this.results.sveltekit = {
                status: 'SUCCESS',
                responseTime: response.headers['response-time'] || 'N/A',
                dataSize: JSON.stringify(response.data).length,
                features: {
                    legalAnalysis: !!response.data.data.document.legalAnalysis,
                    vectorization: !!response.data.data.document.vectorization,
                    structureExtraction: !!response.data.data.document.structure
                }
            };
            
            console.log('✅ SvelteKit JSON API - WORKING');
            
        } catch (error) {
            this.results.sveltekit = { status: 'FAILED', error: error.message };
            console.log('❌ SvelteKit JSON API - FAILED:', error.message);
        }
    }

    async testGoLlamaIntegration() {
        console.log('🔄 Testing Go Llama Integration Service...');
        
        try {
            // Health check
            const healthResponse = await axios.get(`${this.services.goLlama}/health`, { timeout: 5000 });
            
            // Process job
            const jobData = {
                type: 'ollama_chat',
                payload: {
                    prompt: 'Analyze potential legal risks in a software licensing agreement',
                    model: 'gemma3-legal',
                    use_custom_json: true
                }
            };
            
            const jobResponse = await axios.post(`${this.services.goLlama}/api/process`, jobData, { timeout: 10000 });
            
            // Wait for processing
            await new Promise(resolve => setTimeout(resolve, 3000));
            
            // Get results
            const resultResponse = await axios.get(
                `${this.services.goLlama}/api/results/${jobResponse.data.job_id}`,
                { timeout: 5000 }
            );
            
            this.results.goLlama = {
                status: 'SUCCESS',
                workerID: healthResponse.data.worker_id,
                jobProcessed: resultResponse.data.status === 'completed',
                legalAnalysis: !!resultResponse.data.result.processing,
                customJSONIntegration: !!resultResponse.data.result.json_optimization
            };
            
            console.log('✅ Go Llama Integration - WORKING');
            
        } catch (error) {
            this.results.goLlama = { status: 'FAILED', error: error.message };
            console.log('❌ Go Llama Integration - FAILED:', error.message);
        }
    }

    async testEnhancedRAG() {
        console.log('🔄 Testing Enhanced RAG Service...');
        
        try {
            const response = await axios.get(`${this.services.enhancedRAG}/health`, { timeout: 5000 });
            
            this.results.enhancedRAG = {
                status: 'SUCCESS',
                websocketConnections: response.data.websocket_connections,
                context7Connected: response.data.context7_connected,
                activePatterns: response.data.active_patterns
            };
            
            console.log('✅ Enhanced RAG Service - WORKING');
            
        } catch (error) {
            this.results.enhancedRAG = { status: 'FAILED', error: error.message };
            console.log('❌ Enhanced RAG Service - FAILED:', error.message);
        }
    }

    async testUploadService() {
        console.log('🔄 Testing Upload Service...');
        
        try {
            const response = await axios.get(`${this.services.uploadService}/health`, { timeout: 5000 });
            
            this.results.uploadService = {
                status: 'SUCCESS',
                databaseConnected: response.data.services.database,
                ollamaConnected: response.data.services.ollama,
                redisConnected: response.data.services.redis
            };
            
            console.log('✅ Upload Service - WORKING');
            
        } catch (error) {
            this.results.uploadService = { status: 'FAILED', error: error.message };
            console.log('❌ Upload Service - FAILED:', error.message);
        }
    }

    async testOllamaModels() {
        console.log('🔄 Testing Ollama AI Models...');
        
        try {
            const response = await axios.get(`${this.services.ollama}/api/tags`, { timeout: 5000 });
            
            const models = response.data.models;
            const hasLegalModel = models.some(m => m.name.includes('gemma3-legal'));
            const hasEmbeddingModel = models.some(m => m.name.includes('nomic-embed-text'));
            
            this.results.ollama = {
                status: 'SUCCESS',
                totalModels: models.length,
                legalModelAvailable: hasLegalModel,
                embeddingModelAvailable: hasEmbeddingModel,
                models: models.map(m => ({ name: m.name, size: m.size }))
            };
            
            console.log('✅ Ollama AI Models - WORKING');
            
        } catch (error) {
            this.results.ollama = { status: 'FAILED', error: error.message };
            console.log('❌ Ollama AI Models - FAILED:', error.message);
        }
    }

    async testEndToEndIntegration() {
        console.log('🔄 Testing End-to-End Integration Pipeline...');
        
        try {
            // Step 1: Process through Go Llama with custom JSON
            const jobData = {
                type: 'ollama_chat',
                payload: {
                    prompt: 'Review this employment contract for compliance issues and potential legal risks',
                    model: 'gemma3-legal',
                    use_custom_json: true
                }
            };
            
            const jobResponse = await axios.post(`${this.services.goLlama}/api/process`, jobData);
            
            // Step 2: Wait and get results
            await new Promise(resolve => setTimeout(resolve, 3000));
            
            const resultResponse = await axios.get(`${this.services.goLlama}/api/results/${jobResponse.data.job_id}`);
            
            // Step 3: Verify pipeline worked
            const result = resultResponse.data.result;
            
            this.results.endToEnd = {
                status: 'SUCCESS',
                pipelineCompleted: resultResponse.data.status === 'completed',
                legalConceptsExtracted: result.processing.legal_concepts.length > 0,
                documentTypeClassified: !!result.processing.document_type,
                semanticChunksCreated: result.processing.semantic_chunks.length > 0,
                customJSONAttempted: result.processing.custom_json_used,
                totalTokensProcessed: result.tokens
            };
            
            console.log('✅ End-to-End Integration Pipeline - WORKING');
            
        } catch (error) {
            this.results.endToEnd = { status: 'FAILED', error: error.message };
            console.log('❌ End-to-End Integration Pipeline - FAILED:', error.message);
        }
    }

    printSummary() {
        console.log('\n🎯 LEGAL AI PLATFORM - SYSTEM INTEGRATION TEST RESULTS');
        console.log('=' .repeat(70));
        
        let totalTests = 0;
        let passedTests = 0;
        
        Object.entries(this.results).forEach(([service, result]) => {
            totalTests++;
            const status = result.status === 'SUCCESS' ? '✅ PASS' : '❌ FAIL';
            if (result.status === 'SUCCESS') passedTests++;
            
            console.log(`\n🔧 ${service.toUpperCase()}: ${status}`);
            
            if (result.status === 'SUCCESS') {
                delete result.status;
                Object.entries(result).forEach(([key, value]) => {
                    console.log(`   ${key}: ${JSON.stringify(value)}`);
                });
            } else {
                console.log(`   Error: ${result.error}`);
            }
        });
        
        console.log('\n' + '=' .repeat(70));
        console.log(`📊 SUMMARY: ${passedTests}/${totalTests} services operational`);
        
        if (passedTests === totalTests) {
            console.log('🎉 ALL SYSTEMS OPERATIONAL - PRODUCTION READY!');
            process.exit(0);
        } else {
            console.log('⚠️  Some services need attention before production deployment');
            process.exit(1);
        }
    }
}

// Run the test suite
const tester = new LegalAISystemTest();
tester.runAllTests().catch(error => {
    console.error('Test suite failed to start:', error);
    process.exit(1);
});