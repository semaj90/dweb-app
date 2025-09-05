#!/usr/bin/env node

/**
 * Check Local LLM Status and Configuration
 * Validates all local LLM installations and models
 */

import { execSync } from 'child_process';
import fs from 'fs';
import path from 'path';

console.log('🤖 Checking Local LLM Status...\n');

async function checkOllamaStatus() {
  console.log('📋 Ollama Status Check:');
  
  try {
    // Check if Ollama is running
    const response = execSync('curl -s http://localhost:11434/api/tags', { 
      encoding: 'utf8',
      timeout: 5000
    });
    
    const data = JSON.parse(response);
    console.log('  ✅ Ollama service is running');
    console.log(`  📊 Available models: ${data.models?.length || 0}`);
    
    if (data.models && data.models.length > 0) {
      console.log('  📚 Installed models:');
      data.models.forEach(model => {
        const sizeGB = model.size ? (model.size / (1024 * 1024 * 1024)).toFixed(1) : 'Unknown';
        console.log(`    - ${model.name} (${sizeGB}GB) - Modified: ${model.modified_at || 'Unknown'}`);
      });
      
      // Check for recommended models
      const gemmaModels = data.models.filter(m => m.name.includes('gemma'));
      const llamaModels = data.models.filter(m => m.name.includes('llama'));
      
      console.log(`  🔹 Gemma models: ${gemmaModels.length}`);
      console.log(`  🔹 Llama models: ${llamaModels.length}`);
      
      if (gemmaModels.length === 0) {
        console.log('  ⚠️  No Gemma models found - consider installing gemma2:2b');
        console.log('  💡 Run: ollama pull gemma2:2b');
      }
      
      return { 
        success: true, 
        models: data.models,
        hasGemma: gemmaModels.length > 0,
        hasLlama: llamaModels.length > 0
      };
    } else {
      console.log('  ⚠️  No models installed');
      console.log('  💡 Install a model with: ollama pull gemma2:2b');
      return { success: true, models: [], hasGemma: false, hasLlama: false };
    }
    
  } catch (error) {
    console.log('  ❌ Ollama service not running or accessible');
    console.log('  💡 Start with: ollama serve');
    return { success: false, error: 'Ollama not accessible' };
  }
}

async function checkLocalConfig() {
  console.log('\n⚙️ Local LLM Configuration Check:');
  
  const configFiles = [
    './src/lib/config/local-llm.ts',
    './src/lib/services/local-llm-manager.ts',
    './src/lib/services/ollama-service.ts'
  ];
  
  let configsFound = 0;
  
  for (const config of configFiles) {
    if (fs.existsSync(config)) {
      console.log(`  ✅ ${config} exists`);
      configsFound++;
    } else {
      console.log(`  ❌ ${config} missing`);
    }
  }
  
  console.log(`  📊 Configuration files: ${configsFound}/${configFiles.length}`);
  
  return { configsFound, total: configFiles.length };
}

async function checkAPIIntegration() {
  console.log('\n🔗 API Integration Check:');
  
  const apiFiles = [
    './src/routes/api/ai/chat/+server.ts',
    './src/routes/api/ai/ollama-gemma3/+server.ts', 
    './src/routes/api/summaries/+server.ts',
    './src/routes/api/llm/chat/+server.ts'
  ];
  
  let apisFound = 0;
  
  for (const api of apiFiles) {
    if (fs.existsSync(api)) {
      console.log(`  ✅ ${api} exists`);
      apisFound++;
    } else {
      console.log(`  ❌ ${api} missing`);
    }
  }
  
  console.log(`  📊 API endpoints: ${apisFound}/${apiFiles.length}`);
  
  return { apisFound, total: apiFiles.length };
}

async function checkUIComponents() {
  console.log('\n🎨 UI Components Check:');
  
  const uiFiles = [
    './src/lib/components/LLMAssistant.svelte',
    './src/lib/components/ai/LLMSelector.svelte',
    './src/lib/components/ai/MultiLLMOrchestrator.svelte',
    './src/lib/components/ai/ComprehensiveSummaryEngine.svelte'
  ];
  
  let uiFound = 0;
  
  for (const ui of uiFiles) {
    if (fs.existsSync(ui)) {
      console.log(`  ✅ ${ui} exists`);
      uiFound++;
    } else {
      console.log(`  ❌ ${ui} missing`);
    }
  }
  
  console.log(`  📊 UI components: ${uiFound}/${uiFiles.length}`);
  
  return { uiFound, total: uiFiles.length };
}

async function testOllamaAPI() {
  console.log('\n🧪 Ollama API Test:');
  
  try {
    // Test basic API functionality
    const healthResponse = execSync('curl -s http://localhost:11434/api/tags', { 
      encoding: 'utf8',
      timeout: 3000
    });
    
    const data = JSON.parse(healthResponse);
    if (data.models && data.models.length > 0) {
      const firstModel = data.models[0].name;
      console.log(`  ✅ API accessible`);
      console.log(`  🎯 Testing with model: ${firstModel}`);
      
      // Test a simple chat completion
      const testPrompt = JSON.stringify({
        model: firstModel,
        prompt: "Hello, respond with just 'OK' if you can understand this.",
        stream: false,
        options: { temperature: 0.1, num_predict: 10 }
      });
      
      try {
        const chatResponse = execSync(`curl -s -X POST http://localhost:11434/api/generate -H "Content-Type: application/json" -d '${testPrompt}'`, {
          encoding: 'utf8',
          timeout: 30000
        });
        
        const result = JSON.parse(chatResponse);
        if (result.response) {
          console.log(`  ✅ Chat API working - Response: "${result.response.trim()}"`);
          return { success: true, model: firstModel, response: result.response };
        } else {
          console.log(`  ⚠️  Chat API responded but no content: ${JSON.stringify(result)}`);
          return { success: false, error: 'No response content' };
        }
      } catch (chatError) {
        console.log(`  ❌ Chat API test failed: ${chatError.message}`);
        return { success: false, error: 'Chat test failed' };
      }
      
    } else {
      console.log('  ❌ No models available for testing');
      return { success: false, error: 'No models available' };
    }
    
  } catch (error) {
    console.log(`  ❌ API test failed: ${error.message}`);
    return { success: false, error: error.message };
  }
}

// Main execution
async function main() {
  console.log('🏁 Starting Local LLM Status Check\n');
  
  const results = {
    ollama: await checkOllamaStatus(),
    config: await checkLocalConfig(),
    apis: await checkAPIIntegration(),
    ui: await checkUIComponents(),
    apiTest: null
  };
  
  // Only test API if Ollama is working and has models
  if (results.ollama.success && results.ollama.models && results.ollama.models.length > 0) {
    results.apiTest = await testOllamaAPI();
  }
  
  // Generate summary
  console.log('\n' + '='.repeat(60));
  console.log('📊 LOCAL LLM SYSTEM STATUS SUMMARY');
  console.log('='.repeat(60));
  
  console.log(`🤖 Ollama Service: ${results.ollama.success ? '✅ RUNNING' : '❌ NOT AVAILABLE'}`);
  if (results.ollama.success) {
    console.log(`   Models Installed: ${results.ollama.models?.length || 0}`);
    console.log(`   Gemma Available: ${results.ollama.hasGemma ? '✅' : '❌'}`);
    console.log(`   Llama Available: ${results.ollama.hasLlama ? '✅' : '❌'}`);
  }
  
  console.log(`⚙️  Configuration: ${results.config.configsFound}/${results.config.total} files`);
  console.log(`🔗 API Integration: ${results.apis.apisFound}/${results.apis.total} endpoints`);
  console.log(`🎨 UI Components: ${results.ui.uiFound}/${results.ui.total} components`);
  
  if (results.apiTest) {
    console.log(`🧪 API Test: ${results.apiTest.success ? '✅ WORKING' : '❌ FAILED'}`);
  }
  
  // Overall status
  const isFullyOperational = results.ollama.success && 
                            results.ollama.models && 
                            results.ollama.models.length > 0 &&
                            results.config.configsFound >= 2 &&
                            results.apis.apisFound >= 2;
  
  console.log('\n🎯 OVERALL STATUS:');
  if (isFullyOperational) {
    console.log('🟢 LOCAL LLM SYSTEM FULLY OPERATIONAL');
    console.log('🚀 Ready for AI-powered legal analysis');
    
    if (results.apiTest?.success) {
      console.log('🏆 PERFECT - All systems tested and working!');
    }
  } else {
    console.log('🟡 LOCAL LLM SYSTEM PARTIALLY OPERATIONAL');
    console.log('⚠️  Some components need attention');
    
    // Provide specific recommendations
    if (!results.ollama.success) {
      console.log('💡 Start Ollama: ollama serve');
    }
    if (!results.ollama.hasGemma && !results.ollama.hasLlama) {
      console.log('💡 Install a model: ollama pull gemma2:2b');
    }
  }
  
  return isFullyOperational;
}

main()
  .then(success => {
    process.exit(success ? 0 : 1);
  })
  .catch(error => {
    console.error('🚨 Status check failed:', error);
    process.exit(1);
  });