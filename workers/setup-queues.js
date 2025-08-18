// workers/setup-queues.js
import { setupQueues } from '../sveltekit-frontend/src/lib/server/rabbitmq.js';
import { initializeWsBroker } from '../sveltekit-frontend/src/lib/server/wsBroker.js';

async function setupSystem() {
  console.log('🚀 Setting up Evidence Processing System\n');
  
  try {
    console.log('📦 Setting up RabbitMQ queues...');
    await setupQueues();
    console.log('✅ RabbitMQ queues configured\n');
    
    console.log('🔌 Initializing WebSocket broker...');
    await initializeWsBroker();
    console.log('✅ WebSocket broker initialized\n');
    
    console.log('🎉 System setup complete!');
    console.log('\n📋 Next steps:');
    console.log('  1. Start the worker: npm start');
    console.log('  2. Check health: npm run health');
    console.log('  3. Test with evidence upload');
    
  } catch (error) {
    console.error('❌ Setup failed:', error);
    process.exit(1);
  }
}

setupSystem();
