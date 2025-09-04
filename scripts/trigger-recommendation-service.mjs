#!/usr/bin/env node

import { readdir, readFile } from 'fs/promises';
import { join } from 'path';

async function triggerRecommendationService() {
    try {
        console.log('🔍 Finding latest error log...');
        
        // Find the latest timestamped error log
        const errorLogsDir = 'error-logs';
        const files = await readdir(errorLogsDir);
        const logFiles = files.filter(f => f.startsWith('npm-check-') && f.endsWith('.log'));
        
        if (logFiles.length === 0) {
            console.log('⚠️ No error logs found');
            return;
        }
        
        // Sort by timestamp (latest first)
        logFiles.sort().reverse();
        const latestLog = logFiles[0];
        
        console.log(`📄 Latest log: ${latestLog}`);
        
        // Read the error log
        const logPath = join(errorLogsDir, latestLog);
        const logContent = await readFile(logPath, 'utf8');
        
        // Send to Go microservice for JSON conversion and recommendation
        const response = await fetch('http://localhost:8096/api/process-error-log', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                logFile: latestLog,
                content: logContent,
                timestamp: new Date().toISOString()
            })
        });
        
        if (response.ok) {
            const result = await response.json();
            console.log('✅ Error processing triggered:', result.message);
            console.log('🔧 Recommendations generated:', result.recommendationCount);
        } else {
            console.log('❌ Failed to process errors:', response.statusText);
        }
        
    } catch (error) {
        console.error('❌ Error in recommendation service:', error.message);
    }
}

triggerRecommendationService();