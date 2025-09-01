/**
 * Standalone Auto-Tagging Worker
 * Runs without Redis dependency - for testing and development
 */

console.log('🚀 Starting Standalone Auto-Tagging Worker');
console.log('📋 Worker Configuration:');
console.log('   - Mode: Standalone (no Redis dependency)');
console.log('   - PostgreSQL: Target data store');  
console.log('   - AI Service: http://localhost:11434 (Ollama)');
console.log('   - Purpose: Evidence auto-tagging and AI analysis');

// Simulate worker activity
let isRunning = true;
let processedCount = 0;

// Mock evidence processing function
async function processEvidence(evidenceId) {
  console.log(`🔍 Processing evidence ${evidenceId}...`);
  
  // Simulate AI analysis delay
  await new Promise(resolve => setTimeout(resolve, 1000));
  
  // Mock AI tagging results
  const mockTags = [
    'document_type:legal_brief',
    'confidence:high', 
    'category:evidence',
    'ai_processed:true'
  ];
  
  console.log(`✅ Evidence ${evidenceId} processed with tags:`, mockTags);
  processedCount++;
  
  return {
    evidenceId,
    tags: mockTags,
    confidence: 0.85,
    processedAt: new Date().toISOString()
  };
}

// Main worker loop
async function workerLoop() {
  console.log('🔄 Worker loop started');
  
  while (isRunning) {
    try {
      // Simulate finding new evidence to process
      if (Math.random() > 0.7) { // 30% chance of new evidence
        const evidenceId = `evidence_${Date.now()}`;
        await processEvidence(evidenceId);
      }
      
      // Log status every 10 processes
      if (processedCount > 0 && processedCount % 5 === 0) {
        console.log(`📊 Worker Status: ${processedCount} items processed`);
      }
      
      // Wait before next check
      await new Promise(resolve => setTimeout(resolve, 2000));
      
    } catch (error) {
      console.error('❌ Worker loop error:', error);
      await new Promise(resolve => setTimeout(resolve, 5000)); // Wait longer on error
    }
  }
  
  console.log('🛑 Worker loop stopped');
}

// Health check function
function logWorkerHealth() {
  console.log('💓 Worker Health Check:');
  console.log(`   - Status: ${isRunning ? 'Running' : 'Stopped'}`);
  console.log(`   - Processed: ${processedCount} items`);
  console.log(`   - Uptime: ${process.uptime().toFixed(1)}s`);
  console.log(`   - Memory: ${Math.round(process.memoryUsage().heapUsed / 1024 / 1024)}MB`);
}

// Start health check interval
setInterval(logWorkerHealth, 10000); // Every 10 seconds

// Graceful shutdown handlers
process.on('SIGINT', () => {
  console.log('🛑 Received SIGINT - Worker shutting down gracefully...');
  isRunning = false;
  setTimeout(() => process.exit(0), 1000);
});

process.on('SIGTERM', () => {
  console.log('🛑 Received SIGTERM - Worker terminating...');
  isRunning = false;
  setTimeout(() => process.exit(0), 1000);
});

// Start the worker
console.log('✅ Worker initialized successfully');
console.log('👀 Monitoring for evidence to process...');
console.log('🔧 Press Ctrl+C to stop the worker');

// Start the main loop
workerLoop().catch(error => {
  console.error('💥 Worker crashed:', error);
  process.exit(1);
});

// Initial health check after 2 seconds
setTimeout(logWorkerHealth, 2000);