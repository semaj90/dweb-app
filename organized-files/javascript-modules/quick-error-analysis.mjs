#!/usr/bin/env node

/**
 * 🔍 Quick TypeScript Error Analysis
 * Simplified version for immediate testing
 */

import fs from 'fs';
import { exec } from 'child_process';
import fetch from 'node-fetch';

class QuickAnalyzer {
    constructor() {
        this.apiUrl = 'http://localhost:8080/analyze';
    }

    // Run npm check and parse basic errors
    async collectAndAnalyze() {
        console.log('🔍 Running TypeScript check and AI analysis...');
        
        return new Promise((resolve, reject) => {
            exec('npm run check', async (error, stdout, stderr) => {
                const output = stderr || stdout;
                console.log('📊 TypeScript check completed, parsing errors...');
                
                // Extract error count
                const errorMatch = output.match(/found (\d+) errors/);
                const fileMatch = output.match(/in (\d+) files/);
                const totalErrors = errorMatch ? parseInt(errorMatch[1]) : 0;
                const totalFiles = fileMatch ? parseInt(fileMatch[1]) : 0;
                
                console.log(`📈 Found ${totalErrors} errors in ${totalFiles} files`);
                
                // Create sample errors for analysis
                const sampleErrors = [
                    { error: "TS2304: Cannot find module 'node:worker_threads'" },
                    { error: "TS7006: Parameter 'task' implicitly has an 'any' type" },
                    { error: "TS2322: Type 'any[]' is not assignable to type 'never[]'" },
                    { error: "TS2339: Property 'onAuthStateChanged' does not exist" },
                    { error: "TS2304: Cannot find name 'process'" }
                ];
                
                console.log('🤖 Sending errors to AI analyzer...');
                
                try {
                    const response = await fetch(this.apiUrl, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(sampleErrors)
                    });
                    
                    if (response.ok) {
                        const analysis = await response.json();
                        
                        console.log('✅ AI Analysis Complete!');
                        console.log('=' .repeat(50));
                        console.log(`📊 Total Errors: ${totalErrors}`);
                        console.log(`📁 Files Affected: ${totalFiles}`);
                        console.log(`🔬 Sample Analyzed: ${analysis.total_errors}`);
                        
                        // Count by severity
                        const high = analysis.analysis?.filter(a => a.severity === 'high').length || 0;
                        const medium = analysis.analysis?.filter(a => a.severity === 'medium').length || 0;
                        const autoFix = analysis.analysis?.filter(a => a.auto_fixable).length || 0;
                        
                        console.log(`🚨 High Priority: ${high}`);
                        console.log(`⚠️  Medium Priority: ${medium}`);
                        console.log(`🔧 Auto-fixable: ${autoFix}`);
                        
                        // Show top recommendations
                        console.log('\\n💡 Top Recommendations:');
                        analysis.analysis?.slice(0, 3).forEach((item, i) => {
                            console.log(`   ${i+1}. [${item.severity}] ${item.suggestion}`);
                        });
                        
                        // Save results
                        const timestamp = new Date().toISOString().replace(/:/g, '-').split('.')[0];
                        const report = {
                            timestamp,
                            totalErrors,
                            totalFiles,
                            analysis,
                            improvement: totalErrors < 1181 ? 'Improving' : 'Needs Attention'
                        };
                        
                        fs.writeFileSync(`./quick-analysis-${timestamp}.json`, JSON.stringify(report, null, 2));
                        console.log(`\\n📄 Report saved: quick-analysis-${timestamp}.json`);
                        
                        resolve(report);
                    } else {
                        throw new Error(`API responded with ${response.status}`);
                    }
                } catch (apiError) {
                    console.error('❌ AI Analysis failed:', apiError.message);
                    console.log('💡 Make sure Go service is running on port 8080');
                    reject(apiError);
                }
            });
        });
    }
}

// Run the analysis
const analyzer = new QuickAnalyzer();
analyzer.collectAndAnalyze()
    .then(() => {
        console.log('\\n🎉 Quick analysis completed successfully!');
        process.exit(0);
    })
    .catch((error) => {
        console.error('❌ Analysis failed:', error.message);
        process.exit(1);
    });