#!/usr/bin/env node
/**
 * Vite Error Analysis Script
 * Analyzes the vite-errors.json file for error patterns and statistics
 */

const fs = require('fs');
const path = require('path');

function analyzeViteErrors() {
  console.log('📊 Vite Error Analysis\n');
  
  const errorFilePath = path.join(__dirname, '../.vscode/vite-errors.json');
  
  try {
    // Check if error file exists
    if (!fs.existsSync(errorFilePath)) {
      console.log('✅ No error log found - this is good news!');
      return;
    }
    
    // Read and parse error file
    const data = JSON.parse(fs.readFileSync(errorFilePath, 'utf8'));
    const errors = data.errors || [];
    
    // Basic statistics
    console.log(`📈 Total entries: ${errors.length}`);
    console.log(`🕐 Last updated: ${data.metadata?.lastUpdated || 'Unknown'}\n`);
    
    // Error level breakdown
    const errorLevels = errors.reduce((acc, e) => {
      acc[e.level] = (acc[e.level] || 0) + 1;
      return acc;
    }, {});
    
    console.log('📋 Entry Levels:');
    Object.entries(errorLevels).forEach(([level, count]) => {
      const emoji = level === 'error' ? '❌' : level === 'warn' ? '⚠️' : '📝';
      console.log(`   ${emoji} ${level}: ${count}`);
    });
    
    // File-based error analysis
    const fileErrors = errors.filter(e => e.file).reduce((acc, e) => {
      acc[e.file] = (acc[e.file] || 0) + 1;
      return acc;
    }, {});
    
    if (Object.keys(fileErrors).length > 0) {
      console.log('\n📁 Most Problematic Files:');
      Object.entries(fileErrors)
        .sort(([,a], [,b]) => b - a)
        .slice(0, 5)
        .forEach(([file, count]) => {
          console.log(`   📄 ${file}: ${count} issues`);
        });
    }
    
    // Recent errors (last 10)
    const recentErrors = errors.slice(-10).filter(e => e.level === 'error');
    if (recentErrors.length > 0) {
      console.log('\n🔍 Recent Errors:');
      recentErrors.forEach(error => {
        console.log(`   ❌ ${error.message} ${error.file ? `(${error.file})` : ''}`);
      });
    } else {
      console.log('\n✅ No recent errors found!');
    }
    
    // Build phase analysis
    const buildPhases = errors.reduce((acc, e) => {
      if (e.buildPhase) {
        acc[e.buildPhase] = (acc[e.buildPhase] || 0) + 1;
      }
      return acc;
    }, {});
    
    if (Object.keys(buildPhases).length > 0) {
      console.log('\n🏗️ Build Phases:');
      Object.entries(buildPhases).forEach(([phase, count]) => {
        console.log(`   🔧 ${phase}: ${count} events`);
      });
    }
    
    // Health assessment
    const errorCount = errorLevels.error || 0;
    const warningCount = errorLevels.warn || 0;
    
    console.log('\n🩺 Health Assessment:');
    if (errorCount === 0 && warningCount === 0) {
      console.log('   ✅ System is healthy - no errors or warnings!');
    } else if (errorCount === 0) {
      console.log(`   ⚠️ ${warningCount} warnings present, but no errors`);
    } else {
      console.log(`   ❌ ${errorCount} errors and ${warningCount} warnings need attention`);
    }
    
  } catch (error) {
    console.error('❌ Error analyzing vite-errors.json:', error.message);
  }
}

// Run the analysis
analyzeViteErrors();