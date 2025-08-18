// update-package-json.js
// Updates package.json with all required dependencies for PostgreSQL + LangChain + Ollama
// Run: node scripts/update-package-json.js

const fs = require('fs');
const path = require('path');

const packageJsonPath = path.join(process.cwd(), 'package.json');

console.log('📦 Updating package.json with LangChain dependencies...\n');

// Read current package.json
const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'));

// Dependencies to add/update
const newDependencies = {
  // LangChain ecosystem
  "langchain": "^0.1.0",
  "@langchain/core": "^0.1.0",
  "@langchain/community": "^0.1.0",
  
  // Database
  "postgres": "^3.4.0",
  "drizzle-orm": "^0.29.0",
  
  // Caching & Queuing
  "ioredis": "^5.7.0",
  "amqplib": "^0.10.8",
  
  // Existing dependencies (keeping your versions)
  "@xenova/transformers": "^2.17.2",
  "onnxruntime-web": "^1.22.0",
  "pgvector": "^0.1.0",
  "neo4j-driver": "^5.28.1",
  "minio": "^7.1.3",
  "@qdrant/js-client-rest": "^1.7.0"
};

// Add new scripts
const newScripts = {
  ...packageJson.scripts,
  "test:rag": "cd scripts && python test-integrated-rag.py",
  "setup:rag": "tsx quick-setup-legal-rag.ts",
  "db:setup": "tsx scripts/setup-pgvector.ts",
  "langchain:test": "tsx scripts/test-legal-rag.ts",
  "system:status": "curl http://localhost:8095/api/status",
  "gpu:metrics": "curl http://localhost:8095/api/metrics"
};

// Update package.json
packageJson.dependencies = {
  ...packageJson.dependencies,
  ...newDependencies
};

packageJson.scripts = newScripts;

// Write updated package.json
fs.writeFileSync(
  packageJsonPath,
  JSON.stringify(packageJson, null, 2),
  'utf8'
);

console.log('✅ package.json updated successfully!\n');
console.log('📋 Added dependencies:');
Object.keys(newDependencies).forEach(dep => {
  console.log(`   - ${dep}: ${newDependencies[dep]}`);
});

console.log('\n📋 Added scripts:');
console.log('   - npm run test:rag      (Test RAG system)');
console.log('   - npm run setup:rag     (Setup RAG pipeline)');
console.log('   - npm run db:setup      (Setup pgvector)');
console.log('   - npm run langchain:test (Test LangChain)');
console.log('   - npm run system:status (Check system status)');
console.log('   - npm run gpu:metrics   (Check GPU metrics)');

console.log('\n📝 Next steps:');
console.log('1. Run: npm install');
console.log('2. Test: npm run test:rag');
console.log('3. Start: npm run dev');
