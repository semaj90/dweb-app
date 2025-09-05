// Quick Validation Script for Error Fixes
// Run with: node validate-fixes.mjs

import { readFile, readdir } from 'fs/promises';
import { join, extname } from 'path';
import chalk from 'chalk';

console.log(chalk.cyan.bold('\n🔍 Validating Error Fixes\n'));

const stats = {
    filesChecked: 0,
    deprecatedEvents: 0,
    unusedCSS: 0,
    missingExports: 0,
    typeErrors: 0,
    passed: 0,
    failed: 0
};

// Patterns to check
const patterns = {
    deprecatedEvents: /on:(click|submit|change|input|focus|blur|keydown|keyup)/g,
    unusedCSS: /\/\*\s*unused\s*CSS\s*\*\//gi,
    consoleLog: /console\.log\(/g,
    debugger: /\bdebugger\b/g,
    todoComments: /\/\/\s*(TODO|FIXME|HACK|XXX|BUG)/gi
};

async function checkFile(filePath) {
    try {
        const content = await readFile(filePath, 'utf-8');
        const issues = [];
        
        // Check for deprecated events (Svelte files)
        if (filePath.endsWith('.svelte')) {
            const deprecatedMatches = content.match(patterns.deprecatedEvents);
            if (deprecatedMatches) {
                issues.push(`${deprecatedMatches.length} deprecated events`);
                stats.deprecatedEvents += deprecatedMatches.length;
            }
        }
        
        // Check for console.log statements
        const consoleMatches = content.match(patterns.consoleLog);
        if (consoleMatches) {
            issues.push(`${consoleMatches.length} console.log statements`);
        }
        
        // Check for debugger statements
        const debuggerMatches = content.match(patterns.debugger);
        if (debuggerMatches) {
            issues.push(`${debuggerMatches.length} debugger statements`);
        }
        
        // Check for TODO comments
        const todoMatches = content.match(patterns.todoComments);
        if (todoMatches) {
            issues.push(`${todoMatches.length} TODO comments`);
        }
        
        stats.filesChecked++;
        
        if (issues.length > 0) {
            console.log(chalk.yellow(`⚠️  ${filePath.split('\\').pop()}`));
            issues.forEach(issue => console.log(chalk.gray(`   - ${issue}`)));
            stats.failed++;
        } else {
            stats.passed++;
        }
        
        return issues.length === 0;
    } catch (error) {
        console.error(chalk.red(`Error reading ${filePath}: ${error.message}`));
        return false;
    }
}

async function scanDirectory(dir, extensions = ['.svelte', '.ts', '.js']) {
    try {
        const files = await readdir(dir, { withFileTypes: true });
        
        for (const file of files) {
            const fullPath = join(dir, file.name);
            
            // Skip node_modules and other build directories
            if (file.name === 'node_modules' || 
                file.name === '.svelte-kit' || 
                file.name === 'build' || 
                file.name === 'dist' ||
                file.name.startsWith('.')) {
                continue;
            }
            
            if (file.isDirectory()) {
                await scanDirectory(fullPath, extensions);
            } else if (extensions.includes(extname(file.name))) {
                await checkFile(fullPath);
            }
        }
    } catch (error) {
        console.error(chalk.red(`Error scanning directory ${dir}: ${error.message}`));
    }
}

async function runValidation() {
    const startTime = Date.now();
    
    // Scan the SvelteKit frontend directory
    console.log(chalk.cyan('Scanning sveltekit-frontend directory...\n'));
    await scanDirectory('sveltekit-frontend/src');
    
    const duration = ((Date.now() - startTime) / 1000).toFixed(2);
    
    // Display summary
    console.log(chalk.cyan.bold('\n📊 Validation Summary\n'));
    console.log(chalk.white(`Files Checked:       ${stats.filesChecked}`));
    console.log(chalk.green(`Files Passed:        ${stats.passed}`));
    console.log(chalk.yellow(`Files with Issues:   ${stats.failed}`));
    
    if (stats.deprecatedEvents > 0) {
        console.log(chalk.red(`\nDeprecated Events:   ${stats.deprecatedEvents}`));
    }
    
    console.log(chalk.gray(`\nTime taken: ${duration}s`));
    
    // Overall result
    if (stats.failed === 0) {
        console.log(chalk.green.bold('\n✅ All validations passed!'));
    } else {
        console.log(chalk.yellow.bold(`\n⚠️  ${stats.failed} files need attention`));
        console.log(chalk.gray('\nRun FIX-ALL-ERRORS.bat to fix these issues'));
    }
    
    // Check if MinIO integration is working
    console.log(chalk.cyan.bold('\n🔗 Checking MinIO Integration...\n'));
    try {
        const response = await fetch('http://localhost:8093/health');
        if (response.ok) {
            console.log(chalk.green('✅ Upload Service is running'));
        } else {
            console.log(chalk.yellow('⚠️  Upload Service returned status:', response.status));
        }
    } catch (error) {
        console.log(chalk.red('❌ Upload Service is not accessible'));
        console.log(chalk.gray('   Run START-MINIO-INTEGRATION.bat to start services'));
    }
    
    // Check PostgreSQL
    console.log(chalk.cyan.bold('\n🐘 Checking PostgreSQL...\n'));
    try {
        const { exec } = await import('child_process');
        const { promisify } = await import('util');
        const execAsync = promisify(exec);
        
        await execAsync('psql -U postgres -d deeds_web_app -c "SELECT 1;" 2>nul');
        console.log(chalk.green('✅ PostgreSQL is running'));
    } catch (error) {
        console.log(chalk.red('❌ PostgreSQL is not accessible'));
    }
    
    process.exit(stats.failed > 0 ? 1 : 0);
}

// Run the validation
runValidation().catch(error => {
    console.error(chalk.red('Validation failed:', error));
    process.exit(1);
});