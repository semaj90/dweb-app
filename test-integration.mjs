/**
 * Test Integration Script - Phase 14 Enhanced Components
 */

console.log('🚀 Testing Phase 14 Integration...\n');

// Test Database Connection
try {
  const { exec } = await import('child_process');
  const { promisify } = await import('util');
  const execAsync = promisify(exec);
  
  const { stdout } = await execAsync('"C:\\Program Files\\PostgreSQL\\17\\bin\\psql.exe" -U legal_admin -d legal_ai_db -h localhost -c "SELECT version();" -t');
  
  if (stdout.includes('PostgreSQL')) {
    console.log('✅ PostgreSQL connection successful');
  }
} catch (error) {
  console.error('❌ Database test failed:', error.message);
}

// Test Component Files
const fs = await import('fs');

const components = [
  'src/lib/components/ai/EnhancedLegalAIChat.svelte',
  'src/lib/components/upload/EnhancedFileUpload.svelte', 
  'src/lib/components/evidence/EvidenceProcessor.svelte',
  'src/lib/stores/ai-chat-store-enhanced.ts'
];

for (const component of components) {
  if (fs.existsSync(component)) {
    console.log(`✅ ${component.split('/').pop()} - File exists`);
  } else {
    console.log(`❌ ${component.split('/').pop()} - Missing`);
  }
}

console.log('\n🎉 Phase 14 Integration Test Complete!');