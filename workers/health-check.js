// workers/health-check.js
import { checkOcrHealth } from './services/ocr.js';
import { checkEmbeddingHealth } from './services/embeddings.js';
import { checkRagHealth } from './services/rag.js';
import { healthCheck as rabbitHealthCheck } from '../sveltekit-frontend/src/lib/server/rabbitmq.js';
import { wsHealthCheck } from '../sveltekit-frontend/src/lib/server/wsBroker.js';

async function runHealthCheck() {
  console.log('🏥 Running Evidence Processing System Health Check\n');
  
  const checks = {
    ocr: await checkOcrHealth(),
    embedding: await checkEmbeddingHealth(),
    rag: await checkRagHealth(),
    rabbitmq: await rabbitHealthCheck(),
    websocket: wsHealthCheck()
  };
  
  console.log('📊 Health Check Results:');
  console.log('========================');
  
  Object.entries(checks).forEach(([service, status]) => {
    if (typeof status === 'object') {
      console.log(`${service.toUpperCase()}:`);
      Object.entries(status).forEach(([component, componentStatus]) => {
        const icon = componentStatus ? '✅' : '❌';
        console.log(`  ${icon} ${component}: ${componentStatus ? 'healthy' : 'unhealthy'}`);
      });
    } else {
      const icon = status ? '✅' : '❌';
      console.log(`${icon} ${service.toUpperCase()}: ${status ? 'healthy' : 'unhealthy'}`);
    }
  });
  
  console.log('\n📋 System Status Summary:');
  const allHealthy = Object.values(checks).every(status => {
    if (typeof status === 'object') {
      return Object.values(status).every(Boolean);
    }
    return Boolean(status);
  });
  
  if (allHealthy) {
    console.log('🎉 All systems operational!');
    process.exit(0);
  } else {
    console.log('⚠️ Some systems need attention');
    process.exit(1);
  }
}

runHealthCheck().catch(error => {
  console.error('❌ Health check failed:', error);
  process.exit(1);
});
