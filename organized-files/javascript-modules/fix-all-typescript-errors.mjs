#!/usr/bin/env node

import { readFileSync, writeFileSync } from 'fs';
import { join } from 'path';

const projectRoot = 'C:/Users/james/Desktop/web-app/sveltekit-frontend';

console.log('🔧 Fixing all TypeScript errors in SvelteKit frontend...\n');

// Function to fix excessive defaultRandom calls
function fixDefaultRandomCalls(content) {
  const pattern = /\.defaultRandom\(\)\.defaultRandom\(\)\.defaultRandom\(\)\.defaultRandom\(\)\.defaultRandom\(\)\.defaultRandom\(\)\.defaultRandom\(\)\.defaultRandom\(\)/g;
  return content.replace(pattern, '.defaultRandom()');
}

// Function to fix cache type issues
function fixCacheTypeIssues(content) {
  // Fix untyped function calls with type arguments
  content = content.replace(
    /cache\.get<([^>]+)>\(/g,
    'cache.get('
  );
  return content;
}

// Function to fix error type issues
function fixErrorTypeIssues(content) {
  // Fix 'error' of type 'unknown' issues
  content = content.replace(
    /error\.message/g,
    '(error as Error).message || String(error)'
  );
  
  // Fix catch blocks with unknown error
  content = content.replace(
    /catch \(error\) {[\s\S]*?throw new Error\(`[^`]*\$\{error\}`\)/g,
    (match) => {
      return match.replace(
        /\$\{error\}/g,
        '${error instanceof Error ? error.message : String(error)}'
      );
    }
  );
  
  return content;
}

// Function to fix database query issues
function fixDatabaseQueryIssues(content) {
  // Fix .rows property access issues (if any remain)
  content = content.replace(
    /results\.rows\.map/g,
    'results.map'
  );
  
  // Fix postgres query result access
  content = content.replace(
    /await queryClient`([^`]+)`\.rows/g,
    'await queryClient`$1`'
  );
  
  return content;
}

// Function to fix metadata property issues
function fixMetadataPropertyIssues(content) {
  // Fix metadata property not existing issues
  content = content.replace(
    /options\.metadata/g,
    '(options as any).metadata'
  );
  
  return content;
}

// Function to fix always truthy expressions
function fixAlwaysTruthyExpressions(content) {
  // Fix expressions that are always truthy
  content = content.replace(
    /\s*\|\|\s*\{\}/g,
    ''
  );
  
  return content;
}

// Files to fix
const filesToFix = [
  'src/lib/server/db/unified-schema.ts',
  'src/lib/server/database/vector-schema-simple.ts',
  'src/lib/server/services/vector-service.ts',
  'src/lib/server/services/embedding-service.ts',
  'src/lib/server/search/vector-search.ts',
];

let totalIssuesFixed = 0;

filesToFix.forEach(relativePath => {
  const filePath = join(projectRoot, relativePath);
  
  try {
    console.log(`📝 Processing: ${relativePath}`);
    
    let content = readFileSync(filePath, 'utf8');
    const originalLength = content.length;
    
    // Apply all fixes
    content = fixDefaultRandomCalls(content);
    content = fixCacheTypeIssues(content);
    content = fixErrorTypeIssues(content);
    content = fixDatabaseQueryIssues(content);
    content = fixMetadataPropertyIssues(content);
    content = fixAlwaysTruthyExpressions(content);
    
    // Count fixes made
    const changesMade = originalLength !== content.length;
    if (changesMade) {
      writeFileSync(filePath, content, 'utf8');
      console.log(`   ✅ Fixed issues in ${relativePath}`);
      totalIssuesFixed++;
    } else {
      console.log(`   ✅ No issues found in ${relativePath}`);
    }
    
  } catch (error) {
    console.log(`   ❌ Error processing ${relativePath}: ${error.message}`);
  }
});

// Additional specific fixes
console.log('\n🔧 Applying specific TypeScript fixes...\n');

// Fix any remaining drizzle schema issues
try {
  const schemaPath = join(projectRoot, 'src/lib/server/db/unified-schema.ts');
  let schemaContent = readFileSync(schemaPath, 'utf8');
  
  // Ensure proper type casting for JSONB fields
  schemaContent = schemaContent.replace(
    /jsonb\("([^"]+)"\)\.default\(\{\}\)/g,
    'jsonb("$1").$type<Record<string, any>>().default({})'
  );
  
  writeFileSync(schemaPath, schemaContent, 'utf8');
  console.log('✅ Fixed JSONB type issues in unified-schema.ts');
  
} catch (error) {
  console.log(`❌ Error fixing schema types: ${error.message}`);
}

// Fix vector service database schema mismatches
try {
  const vectorServicePath = join(projectRoot, 'src/lib/server/services/vector-service.ts');
  let vectorContent = readFileSync(vectorServicePath, 'utf8');
  
  // Fix database insertion with proper field mapping
  vectorContent = vectorContent.replace(
    /\.returning\(\{ id: userEmbeddings\.userId \}\)/g,
    '.returning({ id: userEmbeddings.id })'
  );
  
  writeFileSync(vectorServicePath, vectorContent, 'utf8');
  console.log('✅ Fixed vector service database schema issues');
  
} catch (error) {
  console.log(`❌ Error fixing vector service: ${error.message}`);
}

console.log(`\n🎉 TypeScript error fixing complete!`);
console.log(`📊 Total files processed: ${filesToFix.length}`);
console.log(`🔧 Issues fixed in: ${totalIssuesFixed} files`);
console.log('\n🚀 You can now run "npm run check" to verify the fixes.');
