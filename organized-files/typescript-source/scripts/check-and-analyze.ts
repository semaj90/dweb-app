import { spawn } from 'child_process';
import fs from 'fs/promises';
import path from 'path';

interface ErrorEntry {
  file: string;
  line: number;
  column: number;
  severity: 'error' | 'warning';
  message: string;
  code?: string;
}

interface ClaudePayload {
  task: string;
  prompt: string;
  context: {
    errors: ErrorEntry[];
    summary: {
      total_errors: number;
      total_warnings: number;
      files_affected: string[];
    };
  };
  instructions: string;
}

async function runCheck(): Promise<string> {
  return new Promise((resolve, reject) => {
    console.log('🔍 Running npm run check...');
    const check = spawn('npm', ['run', 'check'], { shell: true });
    
    let output = '';
    let errorOutput = '';
    
    check.stdout.on('data', (data) => {
      const chunk = data.toString();
      output += chunk;
      process.stdout.write(chunk);
    });
    
    check.stderr.on('data', (data) => {
      const chunk = data.toString();
      errorOutput += chunk;
      process.stderr.write(chunk);
    });
    
    check.on('close', (code) => {
      if (code !== 0) {
        resolve(output + errorOutput);
      } else {
        console.log('✅ No errors found!');
        resolve('');
      }
    });
    
    check.on('error', reject);
  });
}

function parseTypeScriptErrors(output: string): ErrorEntry[] {
  const errors: ErrorEntry[] = [];
  const lines = output.split('\n');
  
  // Pattern for TypeScript errors: file.ts(line,col): error TS####: message
  const tsErrorPattern = /^(.+\.(?:ts|tsx|svelte))\((\d+),(\d+)\):\s+(error|warning)\s+(TS\d+)?:?\s*(.+)$/;
  
  for (const line of lines) {
    const match = line.match(tsErrorPattern);
    if (match) {
      errors.push({
        file: match[1],
        line: parseInt(match[2]),
        column: parseInt(match[3]),
        severity: match[4] as 'error' | 'warning',
        code: match[5],
        message: match[6].trim()
      });
    }
  }
  
  return errors;
}

async function createClaudePayload(output: string): Promise<ClaudePayload> {
  const errors = parseTypeScriptErrors(output);
  const filesAffected = [...new Set(errors.map(e => e.file))];
  
  return {
    task: "analyze_typescript_errors",
    prompt: "Analyze these TypeScript/Svelte errors and provide specific fixes",
    context: {
      errors: errors.slice(0, 50), // Limit to first 50 errors
      summary: {
        total_errors: errors.filter(e => e.severity === 'error').length,
        total_warnings: errors.filter(e => e.severity === 'warning').length,
        files_affected: filesAffected
      }
    },
    instructions: "For each error pattern, provide: 1) Root cause explanation, 2) Specific code fix, 3) Prevention tips"
  };
}

async function saveResults(output: string, payload: ClaudePayload) {
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  const logDir = path.join(process.cwd(), '.check-logs');
  
  await fs.mkdir(logDir, { recursive: true });
  
  // Save raw output
  await fs.writeFile(
    path.join(logDir, `check-output-${timestamp}.log`),
    output
  );
  
  // Save Claude payload
  await fs.writeFile(
    path.join(logDir, `claude-payload-${timestamp}.json`),
    JSON.stringify(payload, null, 2)
  );
  
  console.log(`\n📁 Logs saved to: ${logDir}/`);
  
  return { logDir, timestamp };
}

async function main() {
  try {
    const output = await runCheck();
    
    if (!output) {
      console.log('No errors to analyze!');
      return;
    }
    
    console.log('\n🤖 Preparing Claude analysis...');
    const payload = await createClaudePayload(output);
    
    console.log(`\n📊 Error Summary:`);
    console.log(`- Total Errors: ${payload.context.summary.total_errors}`);
    console.log(`- Total Warnings: ${payload.context.summary.total_warnings}`);
    console.log(`- Files Affected: ${payload.context.summary.files_affected.length}`);
    
    const { logDir, timestamp } = await saveResults(output, payload);
    
    console.log('\n📋 Claude-ready JSON payload created!');
    console.log('You can now:');
    console.log('1. Copy the payload from the log directory');
    console.log('2. Paste it into Claude Code for analysis');
    console.log('3. Or use the VS Code task to auto-send to Claude');
    
    // Option to auto-copy to clipboard
    if (process.platform === 'win32') {
      const { exec } = await import('child_process');
      const payloadPath = path.join(logDir, `claude-payload-${timestamp}.json`);
      exec(`type "${payloadPath}" | clip`, (err) => {
        if (!err) {
          console.log('\n📋 Payload copied to clipboard!');
        }
      });
    }
    
  } catch (error) {
    console.error('❌ Error:', error);
    process.exit(1);
  }
}

main();