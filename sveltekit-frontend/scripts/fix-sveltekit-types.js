#!/usr/bin/env node

/**
 * SvelteKit Type Generation Fix Script
 * Prevents and resolves proxy+layout.ts type generation errors
 * 
 * Usage: node scripts/fix-sveltekit-types.js
 * Or add as npm script: "fix:types": "node scripts/fix-sveltekit-types.js"
 */

import { execSync } from 'child_process';
import { existsSync, readFileSync, readdirSync, statSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const projectRoot = join(__dirname, '..');

console.log('🔧 SvelteKit Type Generation Fix Script');
console.log('=====================================\n');

// Step 1: Clean existing types
console.log('1️⃣ Cleaning existing .svelte-kit types...');
try {
  execSync('rm -rf .svelte-kit/types', { cwd: projectRoot, stdio: 'ignore' });
  console.log('   ✅ Removed corrupted types');
} catch (error) {
  console.log('   ℹ️  No existing types to clean');
}

// Step 2: Verify route file integrity
console.log('\n2️⃣ Verifying route file integrity...');

function checkRouteFiles(dir, basePath = '') {
  const items = readdirSync(dir);
  let issues = [];
  
  for (const item of items) {
    const fullPath = join(dir, item);
    const relativePath = join(basePath, item);
    
    if (statSync(fullPath).isDirectory()) {
      // Recursively check subdirectories
      issues = issues.concat(checkRouteFiles(fullPath, relativePath));
    } else if (item.match(/\+page\.ts$|\+layout\.ts$|\+page\.server\.ts$|\+layout\.server\.ts$/)) {
      // Check for invalid imports
      try {
        const content = readFileSync(fullPath, 'utf8');
        const proxyImports = content.match(/import.*proxy\+layout|from\s+['""][^'"]*proxy\+layout/g);
        
        if (proxyImports) {
          issues.push({
            file: relativePath,
            imports: proxyImports
          });
        }
      } catch (error) {
        issues.push({
          file: relativePath,
          error: `Failed to read file: ${error.message}`
        });
      }
    }
  }
  
  return issues;
}

const routesDir = join(projectRoot, 'src', 'routes');
if (existsSync(routesDir)) {
  const issues = checkRouteFiles(routesDir);
  
  if (issues.length > 0) {
    console.log('   ❌ Found issues in route files:');
    issues.forEach(issue => {
      console.log(`      - ${issue.file}: ${issue.imports ? issue.imports.join(', ') : issue.error}`);
    });
    console.log('\n   💡 Please fix these imports before continuing');
    process.exit(1);
  } else {
    console.log('   ✅ All route files look good');
  }
} else {
  console.log('   ❌ Routes directory not found');
  process.exit(1);
}

// Step 3: Check package.json versions
console.log('\n3️⃣ Checking package versions...');
const packageJson = JSON.parse(readFileSync(join(projectRoot, 'package.json'), 'utf8'));

const svelteKitVersion = packageJson.devDependencies['@sveltejs/kit'];
const typescriptVersion = packageJson.devDependencies['typescript'];

console.log(`   📦 @sveltejs/kit: ${svelteKitVersion}`);
console.log(`   📦 typescript: ${typescriptVersion}`);

// Validate versions
const isValidSvelteKit = svelteKitVersion && svelteKitVersion.match(/\^2\./);
const isValidTypeScript = typescriptVersion && typescriptVersion.match(/\^5\./);

if (!isValidSvelteKit) {
  console.log('   ⚠️  Consider updating @sveltejs/kit to ^2.0.0');
}
if (!isValidTypeScript) {
  console.log('   ⚠️  Consider updating typescript to ^5.0.0');
}

if (isValidSvelteKit && isValidTypeScript) {
  console.log('   ✅ Package versions are compatible');
}

// Step 4: Regenerate SvelteKit types
console.log('\n4️⃣ Regenerating SvelteKit types...');
try {
  execSync('npx svelte-kit sync', { 
    cwd: projectRoot, 
    stdio: ['ignore', 'pipe', 'pipe']
  });
  console.log('   ✅ Types regenerated successfully');
} catch (error) {
  console.log('   ❌ Failed to regenerate types:');
  console.log(`   ${error.message}`);
  process.exit(1);
}

// Step 5: Verify generated types
console.log('\n5️⃣ Verifying generated types...');
const typesDir = join(projectRoot, '.svelte-kit', 'types');
if (existsSync(typesDir)) {
  const proxyFiles = execSync('find .svelte-kit/types -name "*proxy*" -type f 2>/dev/null || echo ""', {
    cwd: projectRoot,
    encoding: 'utf8'
  }).trim();
  
  if (proxyFiles) {
    console.log('   ✅ Proxy types generated (this is normal)');
    console.log(`   📂 Generated ${proxyFiles.split('\n').length} proxy type files`);
  } else {
    console.log('   ✅ No proxy files found');
  }
  console.log('   ✅ Type generation completed successfully');
} else {
  console.log('   ❌ Types directory not created - there may be an issue');
  process.exit(1);
}

// Success message
console.log('\n✨ SvelteKit types fixed successfully!');
console.log('💡 You can now restart your dev server:');
console.log('   npm run dev\n');