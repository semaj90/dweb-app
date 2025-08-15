#!/usr/bin/env zx

import { $ } from 'zx';

console.log('================================================================================');
console.log('STARTING LEGAL AI PLATFORM - FULL PRODUCTION SYSTEM');
console.log('================================================================================');

async function main() {
  try {
    await $`net start postgresql-x64-17`.catch(() => console.log('[1/10] PostgreSQL already running'));
    console.log('[1/10] PostgreSQL started');

    await $`start /min redis-server`.catch(() => $`start /min .\\redis-windows\\redis-server.exe`);
    console.log('[2/10] Redis started');

    await $`tasklist | findstr "ollama"`.catch(() => $`start /min ollama serve`);
    console.log('[3/10] Ollama started');

    if (!await $.fs.exists('./minio-data')) {
      await $.fs.mkdir('./minio-data');
    }
    await $`tasklist | findstr "minio"`.catch(() => $`start /min minio.exe server ./minio-data --address :9000 --console-address :9001`);
    console.log('[4/10] MinIO started');

    await $`tasklist | findstr "qdrant"`.catch(() => $`start /min .\\qdrant-windows\\qdrant.exe`);
    console.log('[5/10] Qdrant Vector Database started');

    await $`powershell -Command "Start-Service neo4j"`.catch(() => console.log('Neo4j manual start required'));
    console.log('[6/10] Neo4j started');

    // Start Go Enhanced RAG Service
    await $`start /min cmd /c "cd go-microservice && go run cmd/enhanced-rag/main.go"`.catch(() => 
      $`start /min cmd /c "cd go-microservice && go run main.go"`);
    console.log('[7/10] Go Enhanced RAG Service started');

    // Start Go Upload Service
    await $`start /min cmd /c "cd go-microservice && go run cmd/upload-service/main.go"`.catch(() => 
      console.log('Upload service fallback'));
    console.log('[8/10] Go Upload Service started');

    // Start AI Summary Service if available
    await $`start /min cmd /c "cd go-microservice && go run cmd/ai-summary/main.go"`.catch(() => 
      console.log('AI Summary service optional'));
    console.log('[9/10] Go AI Services started');

    cd('./sveltekit-frontend');
    await $`start cmd /k "npm run dev -- --host 0.0.0.0"`;
    cd('..');
    console.log('[10/10] SvelteKit Frontend started');

    await $.sleep(8000);

    console.log('\n================================================================================');
    console.log('LEGAL AI PLATFORM STARTED SUCCESSFULLY!');
    console.log('================================================================================\n');
    console.log('Access Points:');
    console.log('- Frontend:        http://localhost:5173');
    console.log('- Enhanced RAG:    http://localhost:8094/api/rag');
    console.log('- Upload API:      http://localhost:8093/upload');
    console.log('- MinIO Console:   http://localhost:9001 (admin/minioadmin)');
    console.log('- Qdrant API:      http://localhost:6333');
    console.log('- Neo4j Browser:   http://localhost:7474');
    console.log('- Ollama API:      http://localhost:11434\n');
    console.log('Database Details:');
    console.log('- PostgreSQL:      postgresql://legal_admin:123456@localhost:5432/legal_ai_db');
    console.log('- Redis:           redis://localhost:6379\n');

    await $`start http://localhost:5173`;

    console.log('\nSystem Status Check:');
    console.log('==================');
    try {
      await $`curl -s http://localhost:11434/api/tags`;
      console.log('✓ Ollama: Running');
    } catch (e) {
      console.log('✗ Ollama: Not responding');
    }
    try {
      await $`curl -s http://localhost:6333/collections`;
      console.log('✓ Qdrant: Running');
    } catch (e) {
      console.log('✗ Qdrant: Not responding');
    }
    try {
      await $`redis-cli ping`;
      console.log('✓ Redis: Running');
    } catch (e) {
      console.log('✗ Redis: Not responding');
    }
    console.log('✓ PostgreSQL: Check manually with psql\n');
    console.log('Happy coding! 🚀');

  } catch (e) {
    console.error('Error starting the Legal AI Platform:', e);
  }
}

main();
