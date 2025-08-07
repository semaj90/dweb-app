#!/usr/bin/env node

/**
 * Automated TypeScript Error Fix Pipeline
 * Integrates filesystem indexing, error analysis, and AI-powered fixes
 */

import { exec } from 'child_process';
import { promisify } from 'util';
import fs from 'fs/promises';
import path from 'path';
import fetch from 'node-fetch';

const execAsync = promisify(exec);

class TypeScriptFixPipeline {
    constructor(config = {}) {
        this.config = {
            microserviceUrl: config.microserviceUrl || 'http://localhost:8081',
            ollamaUrl: config.ollamaUrl || 'http://localhost:11434',
            projectRoot: config.projectRoot || './sveltekit-frontend',
            llmModel: config.llmModel || 'codellama:13b',
            batchSize: config.batchSize || 10,
            maxRetries: config.maxRetries || 3,
            ...config
        };
        
        this.errorPatterns = new Map();
        this.fixes = [];
        this.stats = {
            totalErrors: 0,
            fixedErrors: 0,
            failedFixes: 0,
            filesModified: new Set()
        };
    }

    async run() {
        console.log('🚀 Starting TypeScript Fix Pipeline');
        
        try {
            // Phase 1: Index the codebase
            await this.indexCodebase();
            
            // Phase 2: Collect and categorize errors
            const errors = await this.collectErrors();
            
            // Phase 3: Analyze error patterns
            const analysis = await this.analyzeErrors(errors);
            
            // Phase 4: Generate fix strategy
            const strategy = await this.generateFixStrategy(analysis);
            
            // Phase 5: Apply fixes
            await this.applyFixes(strategy);
            
            // Phase 6: Verify fixes
            await this.verifyFixes();
            
            this.printSummary();
            
        } catch (error) {
            console.error('❌ Pipeline failed:', error);
            throw error;
        }
    }

    async indexCodebase() {
        console.log('📚 Indexing codebase...');
        
        const response = await fetch(`${this.config.microserviceUrl}/index`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                rootPath: this.config.projectRoot,
                patterns: ['.ts', '.tsx', '.svelte', '.js'],
                exclude: ['node_modules', '.svelte-kit', 'dist', 'build']
            })
        });
        
        if (!response.ok) {
            throw new Error(`Indexing failed: ${response.statusText}`);
        }
        
        // Wait for indexing to complete
        await this.waitForIndexing();
        console.log('✅ Indexing complete');
    }

    async waitForIndexing(maxWait = 60000) {
        const startTime = Date.now();
        
        while (Date.now() - startTime < maxWait) {
            await new Promise(resolve => setTimeout(resolve, 2000));
            
            const status = await fetch(`${this.config.microserviceUrl}/index/status`);
            if (status.ok) {
                const data = await status.json();
                if (data.status === 'complete') {
                    return;
                }
            }
        }
        
        console.warn('⚠️  Indexing timeout, continuing anyway');
    }

    async collectErrors() {
        console.log('🔍 Collecting TypeScript errors...');
        
        try {
            const { stdout, stderr } = await execAsync('npm run check', {
                cwd: this.config.projectRoot
            });
            
            const errorLines = stderr.split('\n').filter(line => 
                line.includes('error TS') || line.includes('Error:')
            );
            
            const errors = this.parseTypeScriptErrors(errorLines);
            this.stats.totalErrors = errors.length;
            
            console.log(`📊 Found ${errors.length} errors`);
            return errors;
            
        } catch (error) {
            // npm run check returns non-zero on errors, which is expected
            const errorLines = error.stderr.split('\n').filter(line => 
                line.includes('error TS') || line.includes('Error:')
            );
            
            const errors = this.parseTypeScriptErrors(errorLines);
            this.stats.totalErrors = errors.length;
            
            console.log(`📊 Found ${errors.length} errors`);
            return errors;
        }
    }

    parseTypeScriptErrors(errorLines) {
        const errors = [];
        
        for (const line of errorLines) {
            const match = line.match(/(.+?)\((\d+),(\d+)\): error (TS\d+): (.+)/);
            if (match) {
                const [, file, line, column, code, message] = match;
                errors.push({
                    file: path.relative(this.config.projectRoot, file),
                    line: parseInt(line),
                    column: parseInt(column),
                    code,
                    message
                });
                
                // Track error patterns
                if (!this.errorPatterns.has(code)) {
                    this.errorPatterns.set(code, []);
                }
                this.errorPatterns.get(code).push({ file, line, column, message });
            }
        }
        
        return errors;
    }

    async analyzeErrors(errors) {
        console.log('🧠 Analyzing error patterns...');
        
        const response = await fetch(`${this.config.microserviceUrl}/analyze-errors`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                errors: errors.map(e => `${e.code}: ${e.message}`)
            })
        });
        
        const analysis = await response.json();
        
        // Group errors by type for batch fixing
        const groupedErrors = {
            typeErrors: errors.filter(e => e.code.startsWith('TS23')),
            importErrors: errors.filter(e => e.code === 'TS2305' || e.code === 'TS2307'),
            propertyErrors: errors.filter(e => e.code === 'TS2339'),
            otherErrors: errors.filter(e => 
                !e.code.startsWith('TS23') && 
                e.code !== 'TS2305' && 
                e.code !== 'TS2307' && 
                e.code !== 'TS2339'
            )
        };
        
        return { ...analysis, groupedErrors };
    }

    async generateFixStrategy(analysis) {
        console.log('📝 Generating fix strategy...');
        
        const strategy = {
            phases: []
        };
        
        // Phase 1: Fix missing types/interfaces
        if (analysis.groupedErrors.typeErrors.length > 0) {
            strategy.phases.push({
                name: 'Fix Type Definitions',
                priority: 1,
                fixes: await this.generateTypeFixes(analysis.groupedErrors.typeErrors)
            });
        }
        
        // Phase 2: Fix import/export issues
        if (analysis.groupedErrors.importErrors.length > 0) {
            strategy.phases.push({
                name: 'Fix Imports',
                priority: 2,
                fixes: await this.generateImportFixes(analysis.groupedErrors.importErrors)
            });
        }
        
        // Phase 3: Fix property access errors
        if (analysis.groupedErrors.propertyErrors.length > 0) {
            strategy.phases.push({
                name: 'Fix Property Access',
                priority: 3,
                fixes: await this.generatePropertyFixes(analysis.groupedErrors.propertyErrors)
            });
        }
        
        return strategy;
    }

    async generateTypeFixes(typeErrors) {
        const fixes = [];
        
        // Group by file for batch processing
        const errorsByFile = new Map();
        for (const error of typeErrors) {
            if (!errorsByFile.has(error.file)) {
                errorsByFile.set(error.file, []);
            }
            errorsByFile.get(error.file).push(error);
        }
        
        for (const [file, errors] of errorsByFile) {
            const fileContent = await fs.readFile(
                path.join(this.config.projectRoot, file),
                'utf-8'
            );
            
            // Use AI to generate fixes
            const prompt = `
                Fix these TypeScript type errors in ${file}:
                ${errors.map(e => `Line ${e.line}: ${e.message}`).join('\n')}
                
                Current code:
                ${fileContent}
                
                Generate only the fixed TypeScript code, no explanations.
            `;
            
            const aiResponse = await this.callLLM(prompt);
            
            if (aiResponse) {
                fixes.push({
                    file,
                    type: 'replace',
                    content: aiResponse,
                    errors: errors.map(e => e.code)
                });
            }
        }
        
        return fixes;
    }

    async generateImportFixes(importErrors) {
        const fixes = [];
        
        for (const error of importErrors) {
            // Query the index for the correct import
            const response = await fetch(
                `${this.config.microserviceUrl}/find-export?name=${encodeURIComponent(error.message)}`
            );
            
            if (response.ok) {
                const data = await response.json();
                if (data.found) {
                    fixes.push({
                        file: error.file,
                        type: 'patch',
                        line: error.line,
                        replacement: `import { ${data.exportName} } from '${data.module}';`,
                        errors: [error.code]
                    });
                }
            }
        }
        
        return fixes;
    }

    async generatePropertyFixes(propertyErrors) {
        const fixes = [];
        
        // Group similar errors
        const propertyGroups = new Map();
        
        for (const error of propertyErrors) {
            const match = error.message.match(/Property '(.+)' does not exist on type '(.+)'/);
            if (match) {
                const [, property, type] = match;
                const key = `${type}:${property}`;
                
                if (!propertyGroups.has(key)) {
                    propertyGroups.set(key, []);
                }
                propertyGroups.get(key).push(error);
            }
        }
        
        // Generate fixes for each group
        for (const [key, errors] of propertyGroups) {
            const [type, property] = key.split(':');
            
            // Find type definition
            const response = await fetch(
                `${this.config.microserviceUrl}/types?name=${encodeURIComponent(type)}`
            );
            
            if (response.ok) {
                const data = await response.json();
                if (data.definition) {
                    // Add property to type
                    const updatedType = this.addPropertyToType(data.definition, property);
                    
                    fixes.push({
                        file: data.file,
                        type: 'patch',
                        find: data.definition,
                        replace: updatedType,
                        errors: errors.map(e => e.code)
                    });
                }
            }
        }
        
        return fixes;
    }

    addPropertyToType(typeDefinition, property) {
        // Simple property addition - in production, use proper AST manipulation
        if (typeDefinition.includes('}')) {
            return typeDefinition.replace(
                '}',
                `  ${property}?: any; // Auto-added by fix pipeline\n}`
            );
        }
        return typeDefinition;
    }

    async applyFixes(strategy) {
        console.log('🔧 Applying fixes...');
        
        for (const phase of strategy.phases.sort((a, b) => a.priority - b.priority)) {
            console.log(`  Phase: ${phase.name}`);
            
            for (const fix of phase.fixes) {
                try {
                    await this.applyFix(fix);
                    this.stats.fixedErrors += fix.errors.length;
                    this.stats.filesModified.add(fix.file);
                } catch (error) {
                    console.error(`    ❌ Failed to apply fix to ${fix.file}:`, error.message);
                    this.stats.failedFixes++;
                }
            }
        }
    }

    async applyFix(fix) {
        const filePath = path.join(this.config.projectRoot, fix.file);
        
        // Backup original file
        const backup = await fs.readFile(filePath, 'utf-8');
        await fs.writeFile(`${filePath}.backup`, backup);
        
        try {
            switch (fix.type) {
                case 'replace':
                    await fs.writeFile(filePath, fix.content);
                    break;
                    
                case 'patch':
                    const content = await fs.readFile(filePath, 'utf-8');
                    const lines = content.split('\n');
                    
                    if (fix.line && fix.replacement) {
                        lines[fix.line - 1] = fix.replacement;
                    } else if (fix.find && fix.replace) {
                        const updated = content.replace(fix.find, fix.replace);
                        await fs.writeFile(filePath, updated);
                        return;
                    }
                    
                    await fs.writeFile(filePath, lines.join('\n'));
                    break;
            }
            
            console.log(`    ✅ Fixed ${fix.file}`);
            
        } catch (error) {
            // Restore backup on failure
            await fs.writeFile(filePath, backup);
            throw error;
        }
    }

    async verifyFixes() {
        console.log('✔️  Verifying fixes...');
        
        try {
            const { stderr } = await execAsync('npm run check', {
                cwd: this.config.projectRoot
            });
            
            const remainingErrors = stderr.split('\n').filter(line => 
                line.includes('error TS')
            ).length;
            
            console.log(`📊 Remaining errors: ${remainingErrors}`);
            
        } catch (error) {
            const remainingErrors = error.stderr.split('\n').filter(line => 
                line.includes('error TS')
            ).length;
            
            console.log(`📊 Remaining errors: ${remainingErrors}`);
        }
    }

    async callLLM(prompt) {
        try {
            const response = await fetch(`${this.config.ollamaUrl}/api/generate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model: this.config.llmModel,
                    prompt,
                    stream: false,
                    options: {
                        temperature: 0.2,
                        top_p: 0.9
                    }
                })
            });
            
            if (response.ok) {
                const data = await response.json();
                return data.response;
            }
        } catch (error) {
            console.error('LLM call failed:', error);
        }
        
        return null;
    }

    printSummary() {
        console.log('\n📈 Fix Pipeline Summary:');
        console.log('========================');
        console.log(`Total Errors Found: ${this.stats.totalErrors}`);
        console.log(`Errors Fixed: ${this.stats.fixedErrors}`);
        console.log(`Failed Fixes: ${this.stats.failedFixes}`);
        console.log(`Files Modified: ${this.stats.filesModified.size}`);
        console.log(`Success Rate: ${((this.stats.fixedErrors / this.stats.totalErrors) * 100).toFixed(1)}%`);
        
        if (this.stats.filesModified.size > 0) {
            console.log('\n📁 Modified Files:');
            for (const file of this.stats.filesModified) {
                console.log(`  - ${file}`);
            }
        }
    }
}

// Run if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
    const pipeline = new TypeScriptFixPipeline({
        projectRoot: process.argv[2] || './sveltekit-frontend'
    });
    
    pipeline.run().catch(console.error);
}

export default TypeScriptFixPipeline;
