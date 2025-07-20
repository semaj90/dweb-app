@echo off
echo Fixing specific TypeScript errors...

cd /d "C:\Users\james\Desktop\web-app\sveltekit-frontend"

echo Fixing cache type issues...
call node -e "
const fs = require('fs');

// Fix vector-search.ts cache type issue
try {
  const vectorSearchPath = 'src/lib/server/search/vector-search.ts';
  if (fs.existsSync(vectorSearchPath)) {
    let content = fs.readFileSync(vectorSearchPath, 'utf8');
    // Remove type parameters from cache.get calls
    content = content.replace(/cache\.get<[^>]+>/g, 'cache.get');
    content = content.replace(/\} \|\| \{\}/g, '} ?? {}');
    fs.writeFileSync(vectorSearchPath, content);
    console.log('✅ Fixed cache type issues in vector-search.ts');
  }
} catch (error) {
  console.log('⚠️ Could not fix vector-search.ts:', error.message);
}

// Fix embedding-service.ts error handling
try {
  const embeddingServicePath = 'src/lib/server/services/embedding-service.ts';
  if (fs.existsSync(embeddingServicePath)) {
    let content = fs.readFileSync(embeddingServicePath, 'utf8');
    // Fix error.message usage
    content = content.replace(/error\.message/g, '(error as Error)?.message || String(error)');
    fs.writeFileSync(embeddingServicePath, content);
    console.log('✅ Fixed error handling in embedding-service.ts');
  }
} catch (error) {
  console.log('⚠️ Could not fix embedding-service.ts:', error.message);
}

console.log('TypeScript error fixes completed');
"

echo Running svelte-check to verify fixes...
call npx svelte-check --tsconfig ./tsconfig.json > ../logs/typescript-errors-fixed.log 2>&1

echo TypeScript fixes complete. Check ../logs/typescript-errors-fixed.log for results.
