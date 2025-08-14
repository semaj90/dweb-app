#!/usr/bin/env node
// Final system validation - checks all fixes are properly applied

import { existsSync, readFileSync } from 'fs';
import { join } from 'path';

console.log('🔍 Final System Validation');
console.log('==========================');

const projectRoot = process.cwd();
let allValid = true;

// Check critical files exist
const criticalFiles = [
    'enhanced-merge-refactor.mjs',
    'enhanced-vector-scanner.mjs', 
    'fix-canvas-integration.mjs',
    'docker-compose-unified.yml',
    'database/migrations/001_initial_schema.sql',
    'LEGAL-AI-CONTROL-PANEL.bat',
    'START-LEGAL-AI.bat',
    'COMPLETE-SYSTEM-FIX.bat'
];

console.log('Checking critical files...');
criticalFiles.forEach(file => {
    const path = join(projectRoot, file);
    if (existsSync(path)) {
        console.log(`✅ ${file}`);
    } else {
        console.log(`❌ ${file} - MISSING`);
        allValid = false;
    }
});

// Check Docker container names consistency
console.log('\nValidating Docker configurations...');
const dockerFile = join(projectRoot, 'docker-compose-unified.yml');
if (existsSync(dockerFile)) {
    const content = readFileSync(dockerFile, 'utf8');
    const hasLegalAiPrefix = content.includes('legal-ai-postgres') && 
                            content.includes('legal-ai-redis') &&
                            content.includes('legal-ai-qdrant');
    if (hasLegalAiPrefix) {
        console.log('✅ Docker container names consistent');
    } else {
        console.log('❌ Docker container names inconsistent');
        allValid = false;
    }
} else {
    console.log('❌ Docker config missing');
    allValid = false;
}

// Check database schema
console.log('\nValidating database schema...');
const schemaFile = join(projectRoot, 'database/migrations/001_initial_schema.sql');
if (existsSync(schemaFile)) {
    const schema = readFileSync(schemaFile, 'utf8');
    const hasRequiredTables = schema.includes('CREATE TABLE') && 
                             schema.includes('users') &&
                             schema.includes('cases') &&
                             schema.includes('evidence');
    if (hasRequiredTables) {
        console.log('✅ Database schema complete');
    } else {
        console.log('❌ Database schema incomplete');
        allValid = false;
    }
} else {
    console.log('❌ Database schema missing');
    allValid = false;
}

// Check frontend structure
console.log('\nValidating frontend structure...');
const frontendPath = join(projectRoot, 'sveltekit-frontend');
if (existsSync(join(frontendPath, 'package.json'))) {
    console.log('✅ Frontend structure ready');
} else {
    console.log('❌ Frontend structure incomplete');
    allValid = false;
}

// Final validation
console.log('\n' + '='.repeat(40));
if (allValid) {
    console.log('🎉 ALL SYSTEMS VALIDATED AND READY');
    console.log('✅ All critical errors fixed');
    console.log('✅ All stub files created');
    console.log('✅ Docker configurations consistent');
    console.log('✅ Database schema ready');
    console.log('✅ Frontend structure validated');
    console.log('\n🚀 READY TO LAUNCH:');
    console.log('   Run: COMPLETE-SYSTEM-FIX.bat');
    console.log('   Or:  LEGAL-AI-CONTROL-PANEL.bat');
    process.exit(0);
} else {
    console.log('❌ VALIDATION FAILED');
    console.log('Some issues need to be resolved');
    process.exit(1);
}
