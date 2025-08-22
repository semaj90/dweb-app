#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

console.log('🔍 SvelteKit Hydration Error Diagnostic\n');

// Check critical files
const criticalFiles = [
  'src/app.html',
  'src/routes/+layout.svelte',
  'src/routes/+page.svelte',
  'src/hooks.client.ts',
  'src/hooks.server.ts',
  'vite.config.js',
  'svelte.config.js'
];

console.log('📁 Checking critical files:');
criticalFiles.forEach(file => {
  if (fs.existsSync(file)) {
    const stats = fs.statSync(file);
    console.log(`✅ ${file} (${stats.size} bytes)`);
  } else {
    console.log(`❌ ${file} - Missing!`);
  }
});

// Check for common hydration issues
console.log('\n🔧 Checking for common hydration issues:');

// Check if there are any undefined variables in key files
function checkFileForUndefined(filePath) {
  if (!fs.existsSync(filePath)) return;
  
  try {
    const content = fs.readFileSync(filePath, 'utf8');
    
    // Check for common patterns that cause hydration issues
    const issues = [];
    
    // Check for accessing undefined properties
    if (content.includes('.source') && !content.includes('source:')) {
      issues.push('Potential undefined .source access');
    }
    
    // Check for unhandled promises
    if (content.includes('await') && !content.includes('try')) {
      issues.push('Potential unhandled async operations');
    }
    
    // Check for missing error boundaries
    if (content.includes('$state') && !content.includes('try')) {
      issues.push('Svelte 5 runes without error handling');
    }
    
    if (issues.length > 0) {
      console.log(`⚠️  ${filePath}:`);
      issues.forEach(issue => console.log(`   - ${issue}`));
    }
  } catch (error) {
    console.log(`❌ Error reading ${filePath}: ${error.message}`);
  }
}

// Check layout and main page
checkFileForUndefined('src/routes/+layout.svelte');
checkFileForUndefined('src/routes/+page.svelte');
checkFileForUndefined('src/hooks.client.ts');

// Check package.json for dependency issues
console.log('\n📦 Checking package.json:');
try {
  const pkg = JSON.parse(fs.readFileSync('package.json', 'utf8'));
  
  const criticalDeps = [
    '@sveltejs/kit',
    'svelte',
    'vite',
    'sveltekit-superforms',
    'zod'
  ];
  
  criticalDeps.forEach(dep => {
    if (pkg.dependencies?.[dep] || pkg.devDependencies?.[dep]) {
      const version = pkg.dependencies?.[dep] || pkg.devDependencies?.[dep];
      console.log(`✅ ${dep}: ${version}`);
    } else {
      console.log(`❌ ${dep} - Missing!`);
    }
  });
  
} catch (error) {
  console.log('❌ Error reading package.json:', error.message);
}

// Check for TypeScript configuration
console.log('\n🔧 TypeScript configuration:');
const tsconfigs = ['tsconfig.json', '.svelte-kit/tsconfig.json'];
tsconfigs.forEach(file => {
  if (fs.existsSync(file)) {
    console.log(`✅ ${file} exists`);
  } else {
    console.log(`❌ ${file} missing`);
  }
});

// Recommendations
console.log('\n💡 Recommendations to fix hydration error:');
console.log('1. Ensure all critical dependencies are installed');
console.log('2. Check for undefined variable access (especially .source)');
console.log('3. Verify Svelte 5 runes syntax is correct');
console.log('4. Add error boundaries around async operations');
console.log('5. Temporarily disable complex components to isolate the issue');

console.log('\n🚀 Quick fix commands:');
console.log('npm install @sveltejs/kit@latest svelte@latest');
console.log('npm run dev:minimal  # Use minimal configuration');
console.log('npm run check:ultra-fast  # Check for TypeScript errors');

console.log('\n✨ Diagnostic complete!');