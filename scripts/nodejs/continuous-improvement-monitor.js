#!/usr/bin/env node

/**
 * 📈 Continuous TypeScript Improvement Monitor
 * Tracks codebase health over time and provides actionable insights
 */

const fs = require('fs');
const path = require('path');

class ContinuousMonitor {
    constructor() {
        this.historyFile = './error-history.json';
        this.alertsFile = './improvement-alerts.json';
        this.metricsFile = './quality-metrics.json';
    }

    /**
     * 📊 Generate quality metrics dashboard
     */
    generateQualityDashboard() {
        console.log('📊 Generating Quality Dashboard...');
        
        const history = this.loadHistory();
        if (history.length === 0) {
            console.log('⚠️ No historical data available');
            return;
        }

        const latest = history[history.length - 1];
        const metrics = this.calculateQualityMetrics(history);
        
        const dashboard = {
            timestamp: new Date().toISOString(),
            current: {
                totalErrors: latest.totalErrors,
                totalFiles: latest.totalFiles,
                errorsPerFile: (latest.totalErrors / latest.totalFiles).toFixed(2)
            },
            trends: metrics.trends,
            goals: this.calculateGoals(metrics),
            recommendations: this.generateRecommendations(metrics),
            alerts: this.checkAlerts(metrics)
        };

        fs.writeFileSync(this.metricsFile, JSON.stringify(dashboard, null, 2));
        
        this.displayDashboard(dashboard);
        
        return dashboard;
    }

    /**
     * 📈 Calculate comprehensive quality metrics
     */
    calculateQualityMetrics(history) {
        const metrics = {
            trends: {},
            velocity: {},
            quality: {}
        };

        if (history.length >= 2) {
            const latest = history[history.length - 1];
            const previous = history[history.length - 2];
            
            // Trend calculations
            metrics.trends.errorChange = latest.totalErrors - previous.totalErrors;
            metrics.trends.fileChange = latest.totalFiles - previous.totalFiles;
            metrics.trends.errorRate = latest.totalErrors / latest.totalFiles;
            metrics.trends.previousErrorRate = previous.totalErrors / previous.totalFiles;
            metrics.trends.rateImprovement = metrics.trends.previousErrorRate - metrics.trends.errorRate;
        }

        if (history.length >= 7) {
            // 7-day velocity
            const weekAgo = history[history.length - 7];
            const latest = history[history.length - 1];
            
            metrics.velocity.weeklyErrorReduction = weekAgo.totalErrors - latest.totalErrors;
            metrics.velocity.weeklyFileChange = latest.totalFiles - weekAgo.totalFiles;
            metrics.velocity.weeklyImprovementRate = 
                ((weekAgo.totalErrors - latest.totalErrors) / weekAgo.totalErrors * 100).toFixed(1);
        }

        // Quality scoring (0-100)
        const latest = history[history.length - 1];
        metrics.quality.errorDensity = latest.totalErrors / latest.totalFiles;
        metrics.quality.score = Math.max(0, 100 - (metrics.quality.errorDensity * 2));
        
        return metrics;
    }

    /**
     * 🎯 Calculate improvement goals
     */
    calculateGoals(metrics) {
        const goals = [];
        
        if (metrics.quality?.score < 70) {
            goals.push({
                type: 'quality',
                target: 'Achieve quality score > 70',
                current: metrics.quality.score.toFixed(1),
                action: 'Focus on high-impact error reduction'
            });
        }

        if (metrics.trends?.errorChange > 0) {
            goals.push({
                type: 'stability',
                target: 'Reduce error growth to 0',
                current: `+${metrics.trends.errorChange} errors`,
                action: 'Implement stricter TypeScript checks'
            });
        }

        if (!metrics.velocity?.weeklyErrorReduction || metrics.velocity.weeklyErrorReduction < 10) {
            goals.push({
                type: 'velocity',
                target: 'Reduce 10+ errors per week',
                current: metrics.velocity?.weeklyErrorReduction || 'N/A',
                action: 'Increase automated fix frequency'
            });
        }

        return goals;
    }

    /**
     * 💡 Generate contextual recommendations
     */
    generateRecommendations(metrics) {
        const recommendations = [];

        // High error density
        if (metrics.quality?.errorDensity > 10) {
            recommendations.push({
                priority: 'high',
                category: 'Architecture',
                suggestion: 'Consider TypeScript strict mode or incremental adoption',
                impact: 'High - Will prevent future type errors'
            });
        }

        // Trending upward
        if (metrics.trends?.errorChange > 20) {
            recommendations.push({
                priority: 'urgent',
                category: 'Process',
                suggestion: 'Implement pre-commit TypeScript checks',
                impact: 'Critical - Prevents error accumulation'
            });
        }

        // Low improvement velocity
        if (metrics.velocity?.weeklyErrorReduction < 5) {
            recommendations.push({
                priority: 'medium',
                category: 'Automation',
                suggestion: 'Schedule daily automated fix runs',
                impact: 'Medium - Accelerates improvement pace'
            });
        }

        return recommendations;
    }

    /**
     * 🚨 Check for quality alerts
     */
    checkAlerts(metrics) {
        const alerts = [];

        if (metrics.trends?.errorChange > 50) {
            alerts.push({
                level: 'critical',
                message: 'Error count increased by 50+ in last run',
                action: 'Investigate recent changes'
            });
        }

        if (metrics.quality?.score < 50) {
            alerts.push({
                level: 'warning',
                message: 'Code quality score below 50',
                action: 'Focus on systematic error reduction'
            });
        }

        if (metrics.velocity?.weeklyImprovementRate < 0) {
            alerts.push({
                level: 'notice',
                message: 'Quality trend is negative',
                action: 'Review development practices'
            });
        }

        return alerts;
    }

    /**
     * 🖥️ Display formatted dashboard
     */
    displayDashboard(dashboard) {
        console.log('\\n📊 TYPESCRIPT QUALITY DASHBOARD');
        console.log('=' .repeat(50));
        
        // Current status
        console.log(`📈 Current Status:`);
        console.log(`   Errors: ${dashboard.current.totalErrors}`);
        console.log(`   Files: ${dashboard.current.totalFiles}`);
        console.log(`   Error Rate: ${dashboard.current.errorsPerFile}/file`);
        console.log(`   Quality Score: ${dashboard.trends?.rateImprovement ? dashboard.trends.rateImprovement.toFixed(2) : 'N/A'}`);

        // Alerts
        if (dashboard.alerts.length > 0) {
            console.log('\\n🚨 Alerts:');
            dashboard.alerts.forEach(alert => {
                console.log(`   ${alert.level.toUpperCase()}: ${alert.message}`);
            });
        }

        // Goals
        if (dashboard.goals.length > 0) {
            console.log('\\n🎯 Goals:');
            dashboard.goals.forEach(goal => {
                console.log(`   ${goal.type}: ${goal.target} (${goal.current})`);
            });
        }

        // Top recommendations
        console.log('\\n💡 Top Recommendations:');
        dashboard.recommendations.slice(0, 3).forEach((rec, i) => {
            console.log(`   ${i+1}. [${rec.priority}] ${rec.suggestion}`);
        });

        console.log('\\n📄 Detailed metrics saved to:', this.metricsFile);
    }

    /**
     * 📚 Load error history
     */
    loadHistory() {
        if (!fs.existsSync(this.historyFile)) {
            return [];
        }

        try {
            return JSON.parse(fs.readFileSync(this.historyFile, 'utf8'));
        } catch (error) {
            console.warn('⚠️ Could not load error history');
            return [];
        }
    }

    /**
     * 🔄 Schedule continuous monitoring
     */
    startContinuousMonitoring(intervalMinutes = 30) {
        console.log(`🔄 Starting continuous monitoring (every ${intervalMinutes} minutes)...`);
        
        const runMonitoring = () => {
            this.generateQualityDashboard();
        };

        // Run immediately
        runMonitoring();

        // Schedule recurring runs
        setInterval(runMonitoring, intervalMinutes * 60 * 1000);
    }
}

// CLI execution
if (require.main === module) {
    const monitor = new ContinuousMonitor();
    
    if (process.argv.includes('--continuous')) {
        monitor.startContinuousMonitoring(30); // Every 30 minutes
    } else {
        monitor.generateQualityDashboard();
    }
}

module.exports = ContinuousMonitor;