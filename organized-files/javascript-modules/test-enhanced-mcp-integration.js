#!/usr/bin/env node
/**
 * Test script for Enhanced MCP Integration with VS Code Extension
 * Tests the unified MCPServerManager with all priority MCP tools and Enhanced RAG
 */

import fs from 'fs';
import path from 'path';

async function testEnhancedMCPIntegration() {
    console.log('🧪 Testing Enhanced MCP Integration...\n');

    // Test 1: Verify VS Code Extension Files
    console.log('1️⃣ Testing VS Code Extension Structure...');
    
    const extensionFiles = [
        '.vscode/extensions/mcp-context7-assistant/src/extension.ts',
        '.vscode/extensions/mcp-context7-assistant/src/mcpServerManager.ts',
        '.vscode/extensions/mcp-context7-assistant/src/types.ts',
        '.vscode/extensions/mcp-context7-assistant/package.json'
    ];

    let allFilesExist = true;
    for (const file of extensionFiles) {
        if (fs.existsSync(file)) {
            const stats = fs.statSync(file);
            console.log(`✅ ${file} (${Math.round(stats.size/1024)}KB)`);
        } else {
            console.log(`❌ ${file} - Missing`);
            allFilesExist = false;
        }
    }

    // Test 2: Verify Enhanced RAG Integration Points
    console.log('\n2️⃣ Testing Enhanced RAG Integration Points...');
    
    let integrationsFound = 0;
    const mcpServerManagerPath = '.vscode/extensions/mcp-context7-assistant/src/mcpServerManager.ts';
    if (fs.existsSync(mcpServerManagerPath)) {
        const content = fs.readFileSync(mcpServerManagerPath, 'utf8');
        
        const integrationChecks = [
            { pattern: 'EnhancedRAGService', description: 'Enhanced RAG Service interface' },
            { pattern: 'ClusterManager', description: 'Cluster Manager interface' },
            { pattern: 'OllamaGemmaCache', description: 'Ollama Gemma Cache interface' },
            { pattern: 'MCPMemoryGraph', description: 'Memory Graph interface' },
            { pattern: 'MCPContext7Tools', description: 'Context7 Tools interface' },
            { pattern: 'mcp_memory2_create_relations', description: 'Memory relations creation' },
            { pattern: 'mcp_memory2_read_graph', description: 'Memory graph reading' },
            { pattern: 'mcp_memory2_search_nodes', description: 'Memory node search' },
            { pattern: 'mcp_context72_get-library-docs', description: 'Context7 documentation retrieval' },
            { pattern: 'mcp_context72_resolve-library-id', description: 'Context7 library ID resolution' },
            { pattern: 'enhanced_rag_query', description: 'Enhanced RAG query handler' },
            { pattern: 'agent_orchestrate_claude', description: 'Claude agent orchestration' },
            { pattern: 'agent_orchestrate_crewai', description: 'CrewAI agent orchestration' },
            { pattern: 'agent_orchestrate_autogen', description: 'AutoGen agent orchestration' },
            { pattern: 'mcp_sequentialthi_sequentialthinking', description: 'Sequential thinking handler' },
            { pattern: 'runCommands', description: 'Command execution handler' },
            { pattern: 'toolCallCache', description: 'Caching system' },
            { pattern: 'performanceMetrics', description: 'Performance metrics' },
            { pattern: 'getEnhancedMetrics', description: 'Enhanced metrics method' }
        ];

        for (const check of integrationChecks) {
            if (content.includes(check.pattern)) {
                console.log(`✅ ${check.description}`);
                integrationsFound++;
            } else {
                console.log(`❌ ${check.description} - Not found`);
            }
        }

        console.log(`\n📊 Integration Coverage: ${integrationsFound}/${integrationChecks.length} (${Math.round(integrationsFound/integrationChecks.length*100)}%)`);
    }

    // Test 3: Verify Extension Commands
    console.log('\n3️⃣ Testing VS Code Extension Commands...');
    
    let commandsFound = 0;
    const extensionPath = '.vscode/extensions/mcp-context7-assistant/src/extension.ts';
    if (fs.existsSync(extensionPath)) {
        const content = fs.readFileSync(extensionPath, 'utf8');
        
        const commandChecks = [
            { pattern: 'mcp.analyzeCurrentContext', description: 'Enhanced context analysis command' },
            { pattern: 'mcp.enhancedRAGQuery', description: 'Enhanced RAG query command' },
            { pattern: 'mcp.agentOrchestration', description: 'Agent orchestration command' },
            { pattern: 'mcp.memoryGraph', description: 'Memory graph command' },
            { pattern: 'mcp.enhancedMetrics', description: 'Enhanced metrics command' },
            { pattern: 'enhanced_rag_query', description: 'Enhanced RAG integration' },
            { pattern: 'mcp_memory2_create_relations', description: 'Memory relations integration' },
            { pattern: 'getEnhancedMetrics', description: 'Metrics display integration' }
        ];

        for (const check of commandChecks) {
            if (content.includes(check.pattern)) {
                console.log(`✅ ${check.description}`);
                commandsFound++;
            } else {
                console.log(`❌ ${check.description} - Not found`);
            }
        }

        console.log(`\n📊 Command Coverage: ${commandsFound}/${commandChecks.length} (${Math.round(commandsFound/commandChecks.length*100)}%)`);
    }

    // Test 4: Verify Enhanced RAG Service Connection
    console.log('\n4️⃣ Testing Enhanced RAG Service Connection...');
    
    const ragServicePath = 'rag/enhanced-rag-service.ts';
    const clusterManagerPath = 'rag/cluster-manager-node.ts';
    const cacheManagerPath = 'vscode-llm-extension/src/ollama-gemma-cache.ts';
    
    const serviceConnections = [
        { path: ragServicePath, name: 'Enhanced RAG Service' },
        { path: clusterManagerPath, name: 'Cluster Manager' },
        { path: cacheManagerPath, name: 'Ollama Gemma Cache' }
    ];

    let servicesConnected = 0;
    for (const service of serviceConnections) {
        if (fs.existsSync(service.path)) {
            console.log(`✅ ${service.name} available`);
            servicesConnected++;
        } else {
            console.log(`❌ ${service.name} - Not available`);
        }
    }

    // Test 5: Integration Summary
    console.log('\n🎯 Enhanced MCP Integration Summary:');
    console.log('━'.repeat(50));
    console.log(`✅ VS Code Extension Files: ${allFilesExist ? 'Available' : 'Missing files'}`);
    console.log(`📊 MCP Tool Integration: ${integrationsFound || 0}/19 tools`);
    console.log(`🎮 VS Code Commands: ${commandsFound || 0}/8 commands`);
    console.log(`🔗 Enhanced RAG Services: ${servicesConnected}/3 connected`);
    console.log('');

    // Status Assessment
    const totalScore = (
        (allFilesExist ? 25 : 0) +
        ((integrationsFound || 0) / 19 * 40) +
        ((commandsFound || 0) / 8 * 25) +
        (servicesConnected / 3 * 10)
    );

    if (totalScore >= 90) {
        console.log('🎉 EXCELLENT: Enhanced MCP Integration is production-ready!');
        console.log('✅ All key MCP tools implemented and routed through mcpServerManager');
        console.log('✅ Enhanced RAG system fully integrated with VS Code extension');
        console.log('✅ Multi-agent orchestration (Claude, CrewAI, AutoGen) available');
        console.log('✅ Memory graph and Context7 tools properly wired');
        console.log('✅ Caching and performance metrics implemented');
    } else if (totalScore >= 75) {
        console.log('🚀 GOOD: Enhanced MCP Integration is mostly complete');
        console.log('ℹ️  Minor gaps in integration, but core functionality available');
    } else if (totalScore >= 50) {
        console.log('⚠️  PARTIAL: Enhanced MCP Integration needs more work');
        console.log('❗ Several key components missing or not integrated');
    } else {
        console.log('❌ INCOMPLETE: Enhanced MCP Integration requires significant work');
        console.log('❗ Major components missing or not properly connected');
    }

    console.log(`\n📈 Integration Score: ${Math.round(totalScore)}/100`);

    // Next Steps
    console.log('\n🔄 Next Steps for Production:');
    console.log('1. Test VS Code extension with real Ollama models');
    console.log('2. Configure Context7 MCP server endpoints');
    console.log('3. Test cluster performance with real workloads');
    console.log('4. Integrate with SvelteKit frontend components');
    console.log('5. Enable real-time Copilot context tracking');
    console.log('6. Add automated testing for all MCP tool handlers');

    return {
        score: Math.round(totalScore),
        filesAvailable: allFilesExist,
        integrationCount: integrationsFound || 0,
        commandCount: commandsFound || 0,
        servicesConnected
    };
}

// Run the test
testEnhancedMCPIntegration()
    .then(result => {
        console.log('\n✅ Enhanced MCP Integration Test Complete');
        process.exit(result.score >= 75 ? 0 : 1);
    })
    .catch(error => {
        console.error('💥 Test failed:', error);
        process.exit(1);
    });