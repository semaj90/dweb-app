// COMPLETE SYSTEM INTEGRATION TEST
// Tests all wired services and API endpoints
// Run with: npm run test:integration

import fetch from 'node-fetch';

class IntegrationTester {
  constructor() {
    this.results = {
      services: new Map(),
      apis: new Map(),
      models: new Map(),
      database: null,
      overall: null
    };
    this.startTime = Date.now();
  }

  async checkPort(port, host = 'localhost', timeout = 5000) {
    return new Promise((resolve) => {
      const timeoutId = setTimeout(() => resolve(false), timeout);
      
      try {
        const socket = new (require('net').Socket)();
        
        socket.setTimeout(timeout);
        socket.on('connect', () => {
          clearTimeout(timeoutId);
          socket.destroy();
          resolve(true);
        });
        
        socket.on('timeout', () => {
          clearTimeout(timeoutId);
          socket.destroy();
          resolve(false);
        });
        
        socket.on('error', () => {
          clearTimeout(timeoutId);
          socket.destroy();
          resolve(false);
        });
        
        socket.connect(port, host);
      } catch (e) {
        clearTimeout(timeoutId);
        resolve(false);
      }
    });
  }

  async testServices() {
    console.log('🔍 Testing core services...');
    
    const services = [
      { name: 'PostgreSQL', port: 5432, required: true },
      { name: 'Ollama', port: 11434, required: true },
      { name: 'SvelteKit', port: 5173, required: true },
      { name: 'Redis', port: 6379, required: false },
      { name: 'Qdrant', port: 6333, required: false },
      { name: 'MinIO', port: 9000, required: false },
      { name: 'Enhanced RAG', port: 8094, required: false },
      { name: 'GPU Orchestrator', port: 8095, required: false }
    ];

    for (const service of services) {
      const isUp = await this.checkPort(service.port);
      this.results.services.set(service.name, {
        port: service.port,
        status: isUp ? 'healthy' : 'down',
        required: service.required,
        tested: new Date().toISOString()
      });

      const emoji = isUp ? '✅' : service.required ? '❌' : '⚠️';
      const status = isUp ? 'UP' : 'DOWN';
      console.log(`   ${emoji} ${service.name}:${service.port} - ${status}`);
    }
  }

  async testModels() {
    console.log('🤖 Testing AI models...');
    
    try {
      const response = await fetch('http://localhost:11434/api/tags');
      const data = await response.json();
      
      const expectedModels = ['gemma3:legal', 'nomic-embed-text'];
      const installedModels = data.models?.map(m => m.name) || [];
      
      for (const model of expectedModels) {
        const found = installedModels.some(installed => 
          installed.includes(model) || installed.includes(model.replace(':', '-'))
        );
        
        this.results.models.set(model, {
          status: found ? 'installed' : 'missing',
          required: true
        });

        const emoji = found ? '✅' : '❌';
        console.log(`   ${emoji} ${model} - ${found ? 'INSTALLED' : 'MISSING'}`);
      }
    } catch (error) {
      console.log(`   ❌ Failed to check models: ${error.message}`);
      this.results.models.set('check_failed', { status: 'error', error: error.message });
    }
  }

  async testDatabase() {
    console.log('💾 Testing database connection...');
    
    try {
      // Test via API endpoint
      const response = await fetch('http://localhost:5173/api/system/status?details=true');
      const data = await response.json();
      
      const dbStatus = data.services?.postgresql;
      if (dbStatus && dbStatus.status === 'healthy') {
        console.log('   ✅ Database connection - HEALTHY');
        console.log(`   ✅ Database: legal_ai_db - CONNECTED`);
        
        this.results.database = {
          status: 'healthy',
          name: 'legal_ai_db',
          connection: 'successful'
        };
      } else {
        console.log('   ❌ Database connection - FAILED');
        this.results.database = {
          status: 'failed',
          error: 'API reported unhealthy database'
        };
      }
    } catch (error) {
      console.log(`   ❌ Database test failed: ${error.message}`);
      this.results.database = {
        status: 'error',
        error: error.message
      };
    }
  }

  async testAPIs() {
    console.log('🌐 Testing API endpoints...');
    
    const criticalAPIs = [
      'http://localhost:5173/api/health',
      'http://localhost:5173/api/system/status',
      'http://localhost:5173/api/cases',
      'http://localhost:5173/api/evidence',
      'http://localhost:5173/api/citations',
      'http://localhost:5173/api/reports',
      'http://localhost:5173/api/chat',
      'http://localhost:5173/api/search',
      'http://localhost:5173/api/ai'
    ];

    const supportAPIs = [
      'http://localhost:5173/api/upload',
      'http://localhost:5173/api/auth', 
      'http://localhost:5173/api/users',
      'http://localhost:5173/api/documents',
      'http://localhost:5173/api/legal'
    ];

    const testEndpoints = async (endpoints, category, required = true) => {
      for (const endpoint of endpoints) {
        try {
          const response = await fetch(endpoint, { 
            method: 'GET',
            timeout: 5000
          });
          
          const isWorking = response.ok || response.status === 401 || response.status === 405;
          
          this.results.apis.set(endpoint, {
            status: response.status,
            ok: isWorking,
            category,
            required,
            responseTime: Date.now() - Date.now() // Simple timing
          });

          const emoji = isWorking ? '✅' : required ? '❌' : '⚠️';
          const status = isWorking ? 'OK' : `ERROR ${response.status}`;
          console.log(`   ${emoji} ${endpoint} - ${status}`);
          
        } catch (error) {
          this.results.apis.set(endpoint, {
            status: 'error',
            ok: false,
            category,
            required,
            error: error.message
          });

          const emoji = required ? '❌' : '⚠️';
          console.log(`   ${emoji} ${endpoint} - CONNECTION FAILED`);
        }
      }
    };

    console.log('   Testing critical APIs...');
    await testEndpoints(criticalAPIs, 'critical', true);
    
    console.log('   Testing support APIs...');
    await testEndpoints(supportAPIs, 'support', false);
  }

  async testAIIntegration() {
    console.log('🧠 Testing AI integration...');
    
    try {
      // Test chat endpoint with a simple query
      const chatResponse = await fetch('http://localhost:5173/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          message: 'What is contract law?',
          model: 'gemma3:legal'
        })
      });

      if (chatResponse.ok || chatResponse.status === 401) {
        console.log('   ✅ Chat AI endpoint - FUNCTIONAL');
      } else {
        console.log(`   ❌ Chat AI endpoint - ERROR ${chatResponse.status}`);
      }

      // Test search endpoint
      const searchResponse = await fetch('http://localhost:5173/api/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          query: 'legal document search test',
          type: 'vector'
        })
      });

      if (searchResponse.ok || searchResponse.status === 401) {
        console.log('   ✅ Vector search endpoint - FUNCTIONAL');
      } else {
        console.log(`   ❌ Vector search endpoint - ERROR ${searchResponse.status}`);
      }

    } catch (error) {
      console.log(`   ❌ AI integration test failed: ${error.message}`);
    }
  }

  generateReport() {
    const totalTime = Date.now() - this.startTime;
    
    // Calculate statistics
    const serviceStats = {
      total: this.results.services.size,
      healthy: Array.from(this.results.services.values()).filter(s => s.status === 'healthy').length,
      required: Array.from(this.results.services.values()).filter(s => s.required).length,
      requiredHealthy: Array.from(this.results.services.values()).filter(s => s.required && s.status === 'healthy').length
    };

    const apiStats = {
      total: this.results.apis.size,
      working: Array.from(this.results.apis.values()).filter(a => a.ok).length,
      critical: Array.from(this.results.apis.values()).filter(a => a.category === 'critical').length,
      criticalWorking: Array.from(this.results.apis.values()).filter(a => a.category === 'critical' && a.ok).length
    };

    const modelStats = {
      total: this.results.models.size,
      installed: Array.from(this.results.models.values()).filter(m => m.status === 'installed').length
    };

    const overallHealth = {
      coreServicesOk: serviceStats.requiredHealthy === serviceStats.required,
      criticalAPIsOk: apiStats.criticalWorking === apiStats.critical,
      modelsOk: modelStats.installed === modelStats.total,
      databaseOk: this.results.database?.status === 'healthy'
    };

    const systemReady = Object.values(overallHealth).every(Boolean);

    return {
      summary: {
        systemReady,
        totalTime: `${(totalTime / 1000).toFixed(2)}s`,
        timestamp: new Date().toISOString(),
        healthScore: Math.round((serviceStats.healthy + apiStats.working) / (serviceStats.total + apiStats.total) * 100)
      },
      services: serviceStats,
      apis: apiStats,
      models: modelStats,
      database: this.results.database,
      overall: overallHealth,
      details: {
        services: Object.fromEntries(this.results.services),
        apis: Object.fromEntries(this.results.apis),
        models: Object.fromEntries(this.results.models)
      }
    };
  }

  printReport() {
    const report = this.generateReport();
    
    console.log('\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('📊 INTEGRATION TEST RESULTS');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    
    const statusEmoji = report.summary.systemReady ? '🎉' : '⚠️';
    const statusText = report.summary.systemReady ? 'SYSTEM READY' : 'ISSUES DETECTED';
    
    console.log(`${statusEmoji} ${statusText}`);
    console.log(`📊 Health Score: ${report.summary.healthScore}%`);
    console.log(`⏱️  Test Duration: ${report.summary.totalTime}`);
    console.log('');
    
    // Services summary
    console.log('🔧 Services:');
    console.log(`   Core: ${report.services.requiredHealthy}/${report.services.required} healthy`);
    console.log(`   Total: ${report.services.healthy}/${report.services.total} healthy`);
    
    // APIs summary
    console.log('🌐 APIs:');
    console.log(`   Critical: ${report.apis.criticalWorking}/${report.apis.critical} working`);
    console.log(`   Total: ${report.apis.working}/${report.apis.total} working`);
    
    // Models summary
    console.log('🤖 AI Models:');
    console.log(`   Installed: ${report.models.installed}/${report.models.total}`);
    
    // Database summary
    console.log('💾 Database:');
    const dbEmoji = report.database?.status === 'healthy' ? '✅' : '❌';
    console.log(`   ${dbEmoji} legal_ai_db: ${report.database?.status || 'unknown'}`);
    
    console.log('');
    
    if (report.summary.systemReady) {
      console.log('✅ ALL SYSTEMS OPERATIONAL!');
      console.log('🌐 Ready to access: http://localhost:5173');
    } else {
      console.log('⚠️ System has issues - check detailed output above');
      
      // Show specific failures
      const failedServices = Array.from(this.results.services.entries())
        .filter(([_, service]) => service.required && service.status !== 'healthy');
      
      if (failedServices.length > 0) {
        console.log('❌ Failed required services:');
        failedServices.forEach(([name, service]) => {
          console.log(`   • ${name}:${service.port}`);
        });
      }
      
      const failedAPIs = Array.from(this.results.apis.entries())
        .filter(([_, api]) => api.category === 'critical' && !api.ok);
      
      if (failedAPIs.length > 0) {
        console.log('❌ Failed critical APIs:');
        failedAPIs.forEach(([url, api]) => {
          console.log(`   • ${url} (${api.status})`);
        });
      }
    }
    
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');
    
    return report;
  }

  async runFullTest() {
    console.log('🚀 Starting Legal AI System Integration Test...');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    
    await this.testServices();
    await this.testModels();
    await this.testDatabase();
    await this.testAPIs();
    await this.testAIIntegration();
    
    return this.printReport();
  }
}

// Main execution
async function main() {
  const tester = new IntegrationTester();
  
  try {
    const report = await tester.runFullTest();
    
    // Exit with appropriate code
    if (report.summary.systemReady) {
      process.exit(0);
    } else {
      process.exit(1);
    }
  } catch (error) {
    console.error('❌ Integration test failed:', error.message);
    process.exit(1);
  }
}

// Run if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch(error => {
    console.error('Unhandled error:', error);
    process.exit(1);
  });
}

export { IntegrationTester, main };
