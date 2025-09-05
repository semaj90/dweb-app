// scripts/multi-agent-check-fix.ts

import { spawn, execSync } from 'child_process';
import fs from 'fs';
import path from 'path';
import fetch from 'node-fetch';

const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
const logsDir = `logs_${timestamp}`;
const todoDir = `todolist_${timestamp}`;
const logFile = path.join(logsDir, 'npm_check.log');
const summaryFile = path.join(todoDir, 'summary.md');
const outputJson = path.join(todoDir, 'claude_suggestions.json');

if (!fs.existsSync(logsDir)) fs.mkdirSync(logsDir, { recursive: true });
if (!fs.existsSync(todoDir)) fs.mkdirSync(todoDir, { recursive: true });

console.log(`📁 Created log/todo folders: ${logsDir}, ${todoDir}`);

function isCudaAvailable() {
  try {
    const output = execSync('nvidia-smi').toString();
    return output.includes('CUDA');
  } catch (err) {
    return false;
  }
}

function runCheck(): Promise<string> {
  return new Promise((resolve) => {
    const check = spawn('powershell', ['-Command', 'npm run check'], { shell: true });
    let output = '';

    check.stdout.on('data', (data) => (output += data.toString()));
    check.stderr.on('data', (data) => (output += data.toString()));

    check.on('close', () => {
      fs.writeFileSync(logFile, output);
      resolve(output);
    });
  });
}

async function askClaude(errorLog: string) {
  const prompt = {
    task: 'multi_agent_fix',
    prompt:
      'These are build/type errors. Generate TODOs and organize them for Autogen or CrewAI agent planning.',
    error_log: errorLog.slice(0, 8000),
  };

  const res = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${process.env.CLAUDE_API_KEY}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: 'claude-3-opus-20240229',
      messages: [{ role: 'user', content: JSON.stringify(prompt) }],
      temperature: 0.3,
    }),
  });

  const json = await res.json();
  const result = json?.content?.trim() ?? 'No response.';
  fs.writeFileSync(summaryFile, result);
  
  let parsed;
  try {
    parsed = JSON.parse(result);
  } catch (e) {
    parsed = { error: 'Failed to parse JSON', raw: result };
  }
  
  fs.writeFileSync(outputJson, JSON.stringify(parsed, null, 2));

  // Simple memory cache
  const memoryFile = path.join(todoDir, 'agent_memory.json');
  const memoryData = { timestamp, summary: result, suggestions: parsed };
  fs.writeFileSync(memoryFile, JSON.stringify(memoryData, null, 2));

  console.log(`✅ Claude suggestions saved:\n- ${summaryFile}\n- ${outputJson}`);
}

(async () => {
  const log = await runCheck();
  
  // Skip Claude API if no key available
  if (process.env.CLAUDE_API_KEY) {
    await askClaude(log);
  } else {
    console.log('⚠️ CLAUDE_API_KEY not found, skipping Claude analysis');
    fs.writeFileSync(summaryFile, `# Build Log Analysis\n\n${log}`);
  }

  console.log('🚀 Ready for Autogen / CrewAI loop.');
  console.log(`You can now launch:\n\n1. Claude Code (manual)\n2. VS Code terminal\n3. Autogen fix loop reading from ${outputJson}`);

  // Optional: Launch VS Code
  try {
    spawn('code', [summaryFile], { stdio: 'inherit' });
  } catch (e) {
    console.log('VS Code not found, open manually:', summaryFile);
  }

  // Optional: Trigger embed pipeline (nomic-embed-text)
  try {
    const embed = await fetch('http://localhost:11434/api/embeddings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: 'nomic-embed-text', prompt: log.slice(0, 2048) })
    });
    const embedJson = await embed.json();
    console.log('🧠 Embedding complete:', embedJson?.embedding ? 'Success' : 'Failed');
  } catch (err) {
    console.warn('⚠️ Embedding skipped (Ollama not running?)');
  }
})();