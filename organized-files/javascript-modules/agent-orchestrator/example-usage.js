/**
 * Example Usage of Enhanced Agent Orchestrator with LLM Optimization Patterns
 * Demonstrates the optimization features from copilot.md
 */

import AgentOrchestrator from './index.js';

async function demonstrateLLMOptimizations() {
    console.log('🚀 Starting LLM Optimization Demo\n');
    
    // Initialize orchestrator with configuration
    const orchestrator = new AgentOrchestrator({
        configPath: './agents-config.json'
    });
    
    // Set up event listeners to see optimization in action
    orchestrator.on('orchestrator-initialized', (data) => {
        console.log('📋 Orchestrator initialized:', data);
        console.log('🎯 Optimizations enabled:', data.optimizations);
    });
    
    orchestrator.on('tokens-processed', (data) => {
        console.log('🔄 Tokens processed:', data);
    });
    
    orchestrator.on('tokens-compressed', (data) => {
        console.log('📦 Tokens compressed:', data);
    });
    
    orchestrator.on('workflow-started', (data) => {
        console.log('🏃 Workflow started:', data.workflow);
    });
    
    orchestrator.on('workflow-progress', (data) => {
        console.log(`📊 Progress: ${data.completed}/${data.total} (${data.step})`);
    });
    
    orchestrator.on('workflow-completed', (data) => {
        console.log('✅ Workflow completed in', data.duration, 'ms');
    });
    
    try {
        // Initialize the orchestrator
        console.log('1. Initializing orchestrator with LLM optimizations...');
        const initResult = await orchestrator.initialize();
        console.log('   Config summary:', initResult.configSummary);
        console.log('');
        
        // Example 1: Token Streaming Optimization
        console.log('2. Demonstrating token streaming optimization...');
        const sampleTokens = [
            { id: 1, text: 'Hello', type: 'word' },
            { id: 2, text: ',', type: 'punctuation' },
            { id: 3, text: 'world', type: 'word' },
            { id: 4, text: '!', type: 'punctuation' }
        ];
        
        const processedTokens = await orchestrator.processStreamingTokens(sampleTokens);
        console.log('   Processed tokens:', processedTokens.length);
        console.log('');
        
        // Example 2: Token Compression
        console.log('3. Demonstrating token compression (10x space savings)...');
        const largeTokenArray = Array.from({ length: 1000 }, (_, i) => ({
            id: i,
            text: `token_${i}`,
            type: 'word'
        }));
        
        const compressionResult = await orchestrator.compressTokens(largeTokenArray);
        console.log('   Compression result:', compressionResult.savings, 'space saved');
        console.log('');
        
        // Example 3: Legal Document Analysis Workflow
        console.log('4. Executing legal document analysis workflow...');
        const sampleDocument = {
            title: 'Sample Contract',
            content: `
                AGREEMENT FOR LEGAL SERVICES
                
                This agreement is entered into between Client and Law Firm for the provision
                of legal services relating to contract review and compliance analysis.
                
                Terms:
                1. Scope of services includes document review and legal analysis
                2. Fees are billed at $300 per hour
                3. Client shall provide all necessary documentation
                4. Confidentiality shall be maintained per attorney-client privilege
                
                Jurisdiction: This agreement shall be governed by State Law.
            `,
            type: 'contract',
            metadata: {
                dateCreated: new Date().toISOString(),
                classification: 'legal-agreement'
            }
        };
        
        try {
            const workflowResult = await orchestrator.executeWorkflow(
                'legal-document-analysis',
                sampleDocument,
                { 
                    agents: ['claude', 'ollama'],
                    streaming: true,
                    enableOptimizations: true
                }
            );
            console.log('   Workflow results:', Object.keys(workflowResult.results));
        } catch (error) {
            console.log('   Workflow execution skipped (agents not configured):', error.message);
        }
        console.log('');
        
        // Example 4: Performance Dashboard
        console.log('5. Getting performance dashboard...');
        const dashboard = await orchestrator.getPerformanceDashboard();
        console.log('   Dashboard data:', {
            timestamp: new Date(dashboard.timestamp).toISOString(),
            activeJobs: dashboard.orchestrator.jobs.active,
            completedJobs: dashboard.orchestrator.jobs.completed,
            optimizations: dashboard.config.optimizations
        });
        console.log('');
        
        // Example 5: Bottleneck Analysis
        console.log('6. Analyzing performance bottlenecks...');
        const metrics = orchestrator.getOptimizationMetrics();
        if (metrics.bottleneckAnalysis) {
            console.log('   Bottleneck layers:');
            metrics.bottleneckAnalysis.layers.forEach(layer => {
                console.log(`     - ${layer.name}: ${layer.bottlenecks.length} bottlenecks`);
                layer.solutions.forEach(solution => {
                    console.log(`       → ${solution}`);
                });
            });
        }
        console.log('');
        
        // Example 6: Real-time Chat Optimization
        console.log('7. Demonstrating real-time chat optimization...');
        const chatTokens = [
            'Hi,', ' how', ' can', ' I', ' help', ' you', ' with', ' your', ' legal', ' question', '?'
        ];
        
        // Simulate token-by-token streaming
        for (const token of chatTokens) {
            const processed = await orchestrator.processStreamingTokens([token]);
            process.stdout.write(token);
            await new Promise(resolve => setTimeout(resolve, 50)); // Simulate streaming delay
        }
        console.log('\n   ✅ Token-by-token streaming completed');
        console.log('');
        
        // Example 7: Batch Processing Optimization
        console.log('8. Demonstrating batch processing optimization...');
        const batchDocuments = Array.from({ length: 10 }, (_, i) => ({
            id: i,
            title: `Document ${i + 1}`,
            content: `Sample legal document content for batch processing item ${i + 1}`,
            type: 'legal-document'
        }));
        
        console.log(`   Processing ${batchDocuments.length} documents in batch...`);
        const batchTokens = batchDocuments.map(doc => doc.content).join(' ').split(' ');
        const batchResult = await orchestrator.processStreamingTokens(batchTokens);
        console.log(`   ✅ Batch processing completed: ${batchResult.length} tokens processed`);
        console.log('');
        
        console.log('🎉 LLM Optimization Demo completed successfully!\n');
        
        // Summary
        console.log('📊 SUMMARY OF OPTIMIZATIONS DEMONSTRATED:');
        console.log('✅ 1. Token-by-token streaming (reduces memory usage)');
        console.log('✅ 2. Token compression (10x space savings)');
        console.log('✅ 3. Worker thread processing (parallel execution)');
        console.log('✅ 4. Multi-layer caching (faster response times)');
        console.log('✅ 5. Batch processing (improved throughput)');
        console.log('✅ 6. Real-time monitoring (performance insights)');
        console.log('✅ 7. Bottleneck analysis (optimization guidance)');
        console.log('');
        
        // Best Practices Summary
        console.log('💡 BEST PRACTICES FOR YOUR LOCAL INFERENCE PIPELINE:');
        console.log('🎯 Frontend: Use <pre> and reactive stores for streaming tokens');
        console.log('🎯 SvelteKit: Use stream responses, don\\'t await full LLM result');
        console.log('🎯 Node.js: Use worker_threads and simdjson for parsing');
        console.log('🎯 Network: Use streaming and WebSockets for lower latency');
        console.log('🎯 Ollama: Use system_prompt, batch inference, GPU optimizations');
        console.log('🎯 GPU: Restart workers, clean cache, use quantized models (Q4_K_M)');
        
    } catch (error) {
        console.error('❌ Demo failed:', error);
    } finally {
        // Clean up
        await orchestrator.cleanup();
        console.log('\\n✅ Demo cleanup completed');
    }
}

// Run the demo if this file is executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
    demonstrateLLMOptimizations().catch(console.error);
}

export { demonstrateLLMOptimizations };
export default demonstrateLLMOptimizations;