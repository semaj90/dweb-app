#!/usr/bin/env node

import { exec } from 'child_process';
import { promises as fs } from 'fs';
import path from 'path';

async function runTypeScriptCheck() {
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  const logFile = `typescript-check-${timestamp}.log`;
  
  console.log('🔍 Running comprehensive TypeScript check...');
  console.log(`📝 Logging to: ${logFile}`);
  
  return new Promise((resolve, reject) => {
    // Run svelte-check with verbose output
    const command = 'npx svelte-kit sync && npx svelte-check --tsconfig ./tsconfig.json --output human-verbose';
    
    console.log(`🚀 Executing: ${command}`);
    
    const child = exec(command, {
      cwd: process.cwd(),
      maxBuffer: 1024 * 1024 * 10 // 10MB buffer
    });

    let stdout = '';
    let stderr = '';

    child.stdout?.on('data', (data) => {
      const output = data.toString();
      stdout += output;
      console.log(output);
    });

    child.stderr?.on('data', (data) => {
      const output = data.toString();
      stderr += output;
      console.error(output);
    });

    child.on('close', async (code) => {
      const results = {
        timestamp: new Date().toISOString(),
        exitCode: code,
        stdout,
        stderr,
        summary: ''
      };

      // Parse errors from output
      const errors = parseTypeScriptErrors(stdout + stderr);
      results.summary = generateSummary(errors, code);

      const logContent = `
=== TypeScript Check Results ===
Timestamp: ${results.timestamp}
Exit Code: ${code}
Command: ${command}

=== SUMMARY ===
${results.summary}

=== FULL STDOUT ===
${stdout}

=== FULL STDERR ===
${stderr}

=== ERROR ANALYSIS ===
${JSON.stringify(errors, null, 2)}
`;

      try {
        await fs.writeFile(logFile, logContent);
        console.log(`\n✅ Results saved to: ${logFile}`);
      } catch (err) {
        console.error(`❌ Failed to save log: ${err.message}`);
      }

      resolve(results);
    });

    child.on('error', (err) => {
      console.error(`❌ Command failed: ${err.message}`);
      reject(err);
    });

    // Timeout after 5 minutes
    setTimeout(() => {
      child.kill();
      reject(new Error('TypeScript check timed out after 5 minutes'));
    }, 5 * 60 * 1000);
  });
}

function parseTypeScriptErrors(output) {
  const errors = [];
  const lines = output.split('\n');
  
  let currentError = null;
  
  for (const line of lines) {
    // Match error lines like: src/routes/api/cases/+server.ts:65:7 Error: Type '...' is not assignable...
    const errorMatch = line.match(/^(.+):(\d+):(\d+)\s+(Error|Warning):\s*(.+)$/);
    
    if (errorMatch) {
      if (currentError) {
        errors.push(currentError);
      }
      
      currentError = {
        file: errorMatch[1],
        line: parseInt(errorMatch[2]),
        column: parseInt(errorMatch[3]),
        type: errorMatch[4],
        message: errorMatch[5],
        fullMessage: line
      };
    } else if (currentError && line.trim() && !line.includes('✖')) {
      // Continue multi-line error message
      currentError.message += ' ' + line.trim();
      currentError.fullMessage += '\n' + line;
    }
  }
  
  if (currentError) {
    errors.push(currentError);
  }
  
  return errors;
}

function generateSummary(errors, exitCode) {
  const errorCount = errors.filter(e => e.type === 'Error').length;
  const warningCount = errors.filter(e => e.type === 'Warning').length;
  
  const categories = {};
  errors.forEach(error => {
    const category = categorizeError(error.message);
    categories[category] = (categories[category] || 0) + 1;
  });

  let summary = `
📊 TYPESCRIPT CHECK SUMMARY
Exit Code: ${exitCode}
Total Errors: ${errorCount}
Total Warnings: ${warningCount}
Total Issues: ${errors.length}

📋 ERROR CATEGORIES:
`;

  Object.entries(categories).forEach(([category, count]) => {
    summary += `  ${category}: ${count}\n`;
  });

  if (errorCount === 0) {
    summary += '\n🎉 NO TYPESCRIPT ERRORS! System is ready for production.\n';
  } else if (errorCount < 10) {
    summary += '\n✅ Low error count - System is nearly production ready.\n';
  } else if (errorCount < 25) {
    summary += '\n⚠️  Moderate error count - Focus on critical fixes.\n';
  } else {
    summary += '\n❌ High error count - Systematic fixes needed.\n';
  }

  return summary;
}

function categorizeError(message) {
  if (message.includes('Drizzle') || message.includes('PgSelectBase')) return 'Drizzle ORM Types';
  if (message.includes('does not exist on type')) return 'Missing Properties';
  if (message.includes('is not assignable to type')) return 'Type Mismatch';
  if (message.includes('Cannot find module') || message.includes('Module not found')) return 'Import Errors';
  if (message.includes('WebGPU') || message.includes('GPU')) return 'WebGPU Types';
  if (message.includes('fabric') || message.includes('Filter')) return 'Fabric.js Types';
  if (message.includes('subscribe') || message.includes('store')) return 'Svelte Store Types';
  if (message.includes('Unused CSS selector')) return 'CSS Warnings';
  if (message.includes('vllm') || message.includes('ollama')) return 'AI Service Types';
  if (message.includes('clustering') || message.includes('kmeans')) return 'Clustering Types';
  return 'Other';
}

// Run the check
runTypeScriptCheck()
  .then((results) => {
    console.log('\n🏁 TypeScript check completed');
    console.log('📊 Results available in log file');
    process.exit(results.exitCode);
  })
  .catch((error) => {
    console.error('💥 TypeScript check failed:', error.message);
    process.exit(1);
  });
