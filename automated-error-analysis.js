#!/usr/bin/env node

/**
 * 🔍 Automated TypeScript Error Analysis & Improvement System
 * Integrates with the Legal AI /analyze endpoint for intelligent insights
 */

const fs = require('fs');
const path = require('path');
const { exec } = require('child_process');

class TypeScriptErrorAnalyzer {
    constructor() {
        this.apiUrl = 'http://localhost:8080/analyze';
        this.logDir = './logs';
        this.reportsDir = './error-reports';
        this.historyFile = './error-history.json';
        
        // Ensure directories exist
        [this.logDir, this.reportsDir].forEach(dir => {
            if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
        });
    }

    /**
     * 📊 Collect TypeScript errors from npm run check
     */
    async collectErrors() {
        console.log('🔍 Collecting TypeScript errors...');
        
        return new Promise((resolve, reject) => {
            exec('npm run check', (error, stdout, stderr) => {
                const output = stderr || stdout;
                const errors = this.parseTypeScriptErrors(output);
                
                console.log(`📈 Found ${errors.length} errors to analyze`);
                resolve({
                    totalErrors: this.extractTotalErrorCount(output),
                    totalFiles: this.extractTotalFileCount(output),
                    sampleErrors: errors.slice(0, 50), // Analyze top 50 for performance
                    timestamp: new Date().toISOString(),
                    rawOutput: output
                });
            });
        });
    }

    /**
     * 🧠 Parse TypeScript errors from check output
     */
    parseTypeScriptErrors(output) {
        const errors = [];
        const lines = output.split('\n');
        
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            
            // Match TypeScript error pattern
            const errorMatch = line.match(/^(.+?):(\d+):(\d+)\s*\n?\[31mError\[39m:\s*(.+)/);
            if (errorMatch) {
                errors.push({
                    error: errorMatch[4].trim(),
                    file: errorMatch[1],
                    line: parseInt(errorMatch[2]),
                    column: parseInt(errorMatch[3])
                });
            }
            
            // Alternative pattern for simpler errors
            const simpleMatch = line.match(/^\[31mError\[39m:\s*(.+)/);
            if (simpleMatch && i > 0) {
                const prevLine = lines[i-1];
                const fileMatch = prevLine.match(/^(.+?):(\d+):(\d+)/);
                
                if (fileMatch) {
                    errors.push({
                        error: simpleMatch[1].trim(),
                        file: fileMatch[1],
                        line: parseInt(fileMatch[2]),
                        column: parseInt(fileMatch[3])
                    });
                }
            }
        }
        
        return errors;
    }

    /**
     * 📊 Extract total error count from output
     */
    extractTotalErrorCount(output) {
        const match = output.match(/found (\d+) errors/);
        return match ? parseInt(match[1]) : 0;
    }

    /**
     * 📁 Extract total file count from output
     */
    extractTotalFileCount(output) {
        const match = output.match(/in (\d+) files/);
        return match ? parseInt(match[1]) : 0;
    }

    /**
     * 🔬 Send errors to AI analysis endpoint
     */
    async analyzeErrors(errors) {
        console.log(`🤖 Sending ${errors.length} errors to AI analyzer...`);
        
        try {
            const fetch = (await import('node-fetch')).default;
            
            const response = await fetch(this.apiUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(errors)
            });
            
            if (!response.ok) {
                throw new Error(`Analysis API returned ${response.status}`);
            }
            
            const analysis = await response.json();
            console.log(`✅ Analysis completed: ${analysis.total_errors} errors categorized`);
            
            return analysis;
        } catch (error) {
            console.error('❌ Error analysis failed:', error.message);
            return null;
        }
    }

    /**
     * 📈 Generate prioritized fix recommendations
     */
    generateRecommendations(analysis) {
        if (!analysis || !analysis.analysis) return [];
        
        const recommendations = {
            critical: [],
            high: [],
            medium: [],
            low: [],
            autoFixable: []
        };
        
        analysis.analysis.forEach((item, index) => {
            const rec = {
                priority: item.severity,
                category: item.category,
                error: item.original_error,
                suggestion: item.suggestion,
                autoFixable: item.auto_fixable,
                index: index
            };
            
            // Group by severity
            if (item.severity === 'high') recommendations.high.push(rec);
            else if (item.severity === 'medium') recommendations.medium.push(rec);
            else recommendations.low.push(rec);
            
            // Track auto-fixable separately
            if (item.auto_fixable) recommendations.autoFixable.push(rec);
        });
        
        return recommendations;
    }

    /**
     * 📊 Track improvement over time
     */
    async trackImprovement(currentStats) {
        let history = [];
        
        // Load existing history
        if (fs.existsSync(this.historyFile)) {
            try {
                history = JSON.parse(fs.readFileSync(this.historyFile, 'utf8'));
            } catch (error) {
                console.warn('⚠️ Could not load error history');
            }
        }
        
        // Add current stats
        history.push({
            timestamp: currentStats.timestamp,
            totalErrors: currentStats.totalErrors,
            totalFiles: currentStats.totalFiles,
            analyzedErrors: currentStats.sampleErrors.length
        });
        
        // Keep only last 30 entries
        if (history.length > 30) {
            history = history.slice(-30);
        }
        
        // Calculate trends
        const trends = this.calculateTrends(history);
        
        // Save updated history
        fs.writeFileSync(this.historyFile, JSON.stringify(history, null, 2));
        
        return { history, trends };
    }

    /**
     * 📈 Calculate improvement trends
     */
    calculateTrends(history) {
        if (history.length < 2) {
            return { trend: 'insufficient_data', change: 0 };
        }
        
        const latest = history[history.length - 1];
        const previous = history[history.length - 2];
        
        const errorChange = latest.totalErrors - previous.totalErrors;
        const fileChange = latest.totalFiles - previous.totalFiles;
        const percentChange = previous.totalErrors > 0 
            ? ((errorChange / previous.totalErrors) * 100).toFixed(1)
            : 0;
        
        let trend = 'stable';
        if (errorChange < -10) trend = 'improving';
        else if (errorChange > 10) trend = 'degrading';
        
        return {
            trend,
            errorChange,
            fileChange,
            percentChange: parseFloat(percentChange)
        };
    }

    /**
     * 📄 Generate comprehensive report
     */
    async generateReport(stats, analysis, recommendations, tracking) {
        const timestamp = new Date().toISOString().replace(/:/g, '-').split('.')[0];
        const reportFile = path.join(this.reportsDir, `error-analysis-${timestamp}.md`);
        
        const report = `# 🔍 TypeScript Error Analysis Report
**Generated:** ${stats.timestamp}
**Analyzer:** Legal AI System v2.0.0

## 📊 Current Status
- **Total Errors:** ${stats.totalErrors}
- **Total Files:** ${stats.totalFiles}  
- **Analyzed Sample:** ${stats.sampleErrors.length} errors

## 📈 Improvement Tracking
- **Trend:** ${tracking.trends.trend} (${tracking.trends.percentChange}%)
- **Error Change:** ${tracking.trends.errorChange}
- **File Change:** ${tracking.trends.fileChange}

## 🎯 Priority Recommendations

### 🚨 HIGH PRIORITY (${recommendations.high.length} issues)
${recommendations.high.map(r => `- **${r.category}**: ${r.suggestion}`).join('\n')}

### ⚠️ MEDIUM PRIORITY (${recommendations.medium.length} issues)  
${recommendations.medium.map(r => `- **${r.category}**: ${r.suggestion}`).join('\n')}

### 💡 LOW PRIORITY (${recommendations.low.length} issues)
${recommendations.low.map(r => `- **${r.category}**: ${r.suggestion}`).join('\n')}

## 🔧 Auto-Fixable Issues (${recommendations.autoFixable.length} issues)
${recommendations.autoFixable.map(r => `- ${r.error.substring(0, 80)}...`).join('\n')}

## 📋 Next Actions
1. Address ${recommendations.high.length} high-priority issues first
2. Implement fixes for ${recommendations.autoFixable.length} auto-fixable issues
3. Schedule review for ${recommendations.medium.length} medium-priority items
4. Monitor trends over next 3 analysis cycles

---
*Generated by Legal AI Auto-Indexer System*`;

        fs.writeFileSync(reportFile, report);
        console.log(`📄 Report saved: ${reportFile}`);
        
        return reportFile;
    }

    /**
     * 🚀 Run complete analysis workflow
     */
    async runAnalysis() {
        try {
            console.log('🚀 Starting automated TypeScript error analysis...');
            console.log('=' .repeat(60));
            
            // Step 1: Collect errors
            const stats = await this.collectErrors();
            
            // Step 2: Analyze with AI
            const analysis = await this.analyzeErrors(stats.sampleErrors);
            
            // Step 3: Generate recommendations
            const recommendations = this.generateRecommendations(analysis);
            
            // Step 4: Track improvement
            const tracking = await this.trackImprovement(stats);
            
            // Step 5: Generate report
            const reportFile = await this.generateReport(stats, analysis, recommendations, tracking);
            
            // Step 6: Summary
            console.log('\\n🎉 Analysis Complete!');
            console.log('=' .repeat(60));
            console.log(`📊 Total Errors: ${stats.totalErrors}`);
            console.log(`🎯 High Priority: ${recommendations.high.length}`);
            console.log(`🔧 Auto-fixable: ${recommendations.autoFixable.length}`);
            console.log(`📈 Trend: ${tracking.trends.trend} (${tracking.trends.percentChange}%)`);
            console.log(`📄 Report: ${reportFile}`);
            
            return {
                success: true,
                stats,
                analysis,
                recommendations,
                tracking,
                reportFile
            };
            
        } catch (error) {
            console.error('❌ Analysis workflow failed:', error);
            return { success: false, error: error.message };
        }
    }
}

// CLI execution
if (require.main === module) {
    const analyzer = new TypeScriptErrorAnalyzer();
    analyzer.runAnalysis().then(result => {
        process.exit(result.success ? 0 : 1);
    });
}

module.exports = TypeScriptErrorAnalyzer;