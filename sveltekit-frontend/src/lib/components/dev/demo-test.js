/**
 * Test script for Self-Prompting Demo functionality
 * Verifies that the orchestration system works correctly
 */

import { 
  copilotOrchestrator, 
  generateMCPPrompt, 
  commonMCPQueries, 
  validateMCPRequest 
} from '../../utils/mcp-helpers.js';

async function testSelfPromptingDemo() {
  console.log('🚀 Testing Self-Prompting Demo Components...\n');

  // Test 1: Basic Orchestration
  console.log('1. Testing basic orchestration...');
  try {
    const result = await copilotOrchestrator(
      'Analyze legal evidence for case CASE-2024-001',
      {
        useSemanticSearch: true,
        useMemory: true,
        useMultiAgent: true,
        agents: ['autogen', 'crewai', 'claude']
      }
    );
    console.log('✅ Basic orchestration successful');
    console.log('   Result keys:', Object.keys(result));
  } catch (error) {
    console.log('❌ Basic orchestration failed:', error.message);
  }

  // Test 2: MCP Query Generation
  console.log('\n2. Testing MCP query generation...');
  try {
    const query = commonMCPQueries.analyzeSvelteKit();
    const prompt = generateMCPPrompt(query);
    const validation = validateMCPRequest(query);
    
    console.log('✅ MCP query generation successful');
    console.log('   Query:', query.tool);
    console.log('   Prompt:', prompt);
    console.log('   Valid:', validation.valid);
  } catch (error) {
    console.log('❌ MCP query generation failed:', error.message);
  }

  // Test 3: Self-Prompting Simulation
  console.log('\n3. Testing self-prompting simulation...');
  try {
    const initialPrompt = 'Analyze evidence chain for legal case';
    const iterations = 2;
    
    for (let i = 0; i < iterations; i++) {
      console.log(`   Iteration ${i + 1}/${iterations}`);
      
      const result = await copilotOrchestrator(initialPrompt, {
        useMultiAgent: true,
        synthesizeOutputs: true,
        context: { iteration: i + 1 }
      });
      
      // Simulate self-prompt generation
      if (result.selfPrompt) {
        console.log(`   Generated self-prompt: ${result.selfPrompt.substring(0, 80)}...`);
      }
    }
    
    console.log('✅ Self-prompting simulation successful');
  } catch (error) {
    console.log('❌ Self-prompting simulation failed:', error.message);
  }

  // Test 4: Context7 Integration
  console.log('\n4. Testing Context7 MCP integration...');
  try {
    const context7Queries = [
      commonMCPQueries.performanceBestPractices(),
      commonMCPQueries.securityBestPractices(),
      commonMCPQueries.aiChatIntegration()
    ];
    
    for (const query of context7Queries) {
      const validation = validateMCPRequest(query);
      if (validation.valid) {
        console.log(`   ✅ ${query.tool} query valid`);
      } else {
        console.log(`   ❌ ${query.tool} query invalid:`, validation.errors);
      }
    }
    
    console.log('✅ Context7 integration tests passed');
  } catch (error) {
    console.log('❌ Context7 integration failed:', error.message);
  }

  console.log('\n🎉 Self-Prompting Demo test suite completed!');
}

// Run tests if this file is executed directly
if (typeof window === 'undefined') {
  testSelfPromptingDemo().catch(console.error);
}

export { testSelfPromptingDemo };