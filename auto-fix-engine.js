#!/usr/bin/env node

/**
 * 🔧 Automated TypeScript Fix Engine
 * Implements common fixes based on AI analysis recommendations
 */

const fs = require('fs');
const path = require('path');
const { exec } = require('child_process');

class AutoFixEngine {
    constructor() {
        this.fixesApplied = [];
        this.skippedFixes = [];
    }

    /**
     * 🎯 Apply automatic fixes based on error analysis
     */
    async applyAutomaticFixes(analysisFile) {
        console.log('🔧 Starting automatic fix engine...');
        
        try {
            // Load analysis results
            const analysis = JSON.parse(fs.readFileSync(analysisFile, 'utf8'));
            
            // Apply fixes by category
            await this.fixImportErrors(analysis);
            await this.fixTypeAnnotations(analysis);
            await this.fixPropertyAccess(analysis);
            await this.fixErrorHandling(analysis);
            
            // Generate summary
            this.generateFixSummary();
            
            console.log(`✅ Applied ${this.fixesApplied.length} automatic fixes`);
            console.log(`⚠️ Skipped ${this.skippedFixes.length} complex fixes`);
            
        } catch (error) {
            console.error('❌ Auto-fix failed:', error.message);
        }
    }

    /**
     * 📦 Fix common import/dependency issues
     */
    async fixImportErrors(analysis) {
        console.log('📦 Fixing import errors...');
        
        const missingPackages = new Set();
        
        analysis.analysis?.forEach(item => {
            if (item.category === 'import_error') {
                // Extract package names from common patterns
                const error = item.original_error;
                
                if (error.includes("Cannot find module '@playwright/test'")) {
                    missingPackages.add('@playwright/test');
                }
                if (error.includes("Cannot find name 'process'") || error.includes('node:')) {
                    missingPackages.add('@types/node');
                }
                if (error.includes("Cannot find module 'vitest'")) {
                    missingPackages.add('vitest');
                }
            }
        });

        // Install missing packages
        if (missingPackages.size > 0) {
            const packages = Array.from(missingPackages).join(' ');
            console.log(`📦 Installing: ${packages}`);
            
            await this.runCommand(`npm install --save-dev ${packages}`);
            this.fixesApplied.push(`Installed packages: ${packages}`);
        }
    }

    /**
     * 🏷️ Add basic type annotations
     */
    async fixTypeAnnotations(analysis) {
        console.log('🏷️ Adding type annotations...');
        
        const filesToFix = new Map();
        
        analysis.analysis?.forEach(item => {
            if (item.auto_fixable && item.category === 'type_error') {
                const match = item.original_error.match(/Parameter '(\\w+)' implicitly has an 'any' type/);
                if (match) {
                    const paramName = match[1];
                    // This would be expanded with actual file modification logic
                    this.skippedFixes.push(`Type annotation for parameter '${paramName}' (requires manual review)`);
                }
            }
        });
    }

    /**
     * 🔍 Fix property access patterns
     */
    async fixPropertyAccess(analysis) {
        console.log('🔍 Fixing property access...');
        
        analysis.analysis?.forEach(item => {
            if (item.category === 'property_error') {
                // Common Firebase auth fixes
                if (item.original_error.includes('onAuthStateChanged')) {
                    this.skippedFixes.push('Firebase auth import - requires manual verification');
                }
            }
        });
    }

    /**
     * ⚠️ Improve error handling patterns  
     */
    async fixErrorHandling(analysis) {
        console.log('⚠️ Improving error handling...');
        
        analysis.analysis?.forEach(item => {
            if (item.original_error.includes("'error' is of type 'unknown'")) {
                this.skippedFixes.push('Error type assertion - requires context analysis');
            }
        });
    }

    /**
     * ⚡ Run shell command with promise
     */
    async runCommand(command) {
        return new Promise((resolve, reject) => {
            exec(command, (error, stdout, stderr) => {
                if (error) {
                    reject(error);
                } else {
                    resolve(stdout);
                }
            });
        });
    }

    /**
     * 📊 Generate fix summary report
     */
    generateFixSummary() {
        const timestamp = new Date().toISOString();
        const report = {
            timestamp,
            fixesApplied: this.fixesApplied,
            skippedFixes: this.skippedFixes,
            summary: {
                totalFixes: this.fixesApplied.length,
                skippedCount: this.skippedFixes.length,
                categories: {
                    imports: this.fixesApplied.filter(f => f.includes('Install')).length,
                    types: this.fixesApplied.filter(f => f.includes('Type')).length,
                    properties: this.fixesApplied.filter(f => f.includes('Property')).length
                }
            }
        };

        fs.writeFileSync('./auto-fix-report.json', JSON.stringify(report, null, 2));
        console.log('📄 Fix report saved: auto-fix-report.json');
    }
}

// CLI execution
if (require.main === module) {
    const engine = new AutoFixEngine();
    const analysisFile = process.argv[2] || './error-reports/latest-analysis.json';
    
    engine.applyAutomaticFixes(analysisFile).then(() => {
        console.log('🎉 Auto-fix engine completed!');
    }).catch(error => {
        console.error('❌ Auto-fix engine failed:', error);
        process.exit(1);
    });
}

module.exports = AutoFixEngine;