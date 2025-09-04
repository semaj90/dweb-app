#!/usr/bin/env node
import { spawn } from 'node:child_process';
import { mkdirSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';

const projectRoot = resolve(process.cwd());
const tsconfigPath = resolve(projectRoot, 'sveltekit-frontend', 'tsconfig.json');

function runTSC() {
  return new Promise((resolvePromise) => {
    // Resolve local tsc path explicitly; avoid spawn EINVAL issues on Windows by using cmd.exe when needed
    const frontendDir = resolve(projectRoot, 'sveltekit-frontend');
    const localTsc = process.platform === 'win32'
      ? resolve(frontendDir, 'node_modules', '.bin', 'tsc.cmd')
      : resolve(frontendDir, 'node_modules', '.bin', 'tsc');
    const useNpx = !!process.env.USE_NPX_TSC;

    /** @type {string} */
    let command;
    /** @type {string[]} */
    let args;

    if (useNpx) {
      command = process.platform === 'win32' ? 'npx.cmd' : 'npx';
      args = ['tsc', '-p', tsconfigPath, '--noEmit', '--skipLibCheck'];
    } else if (process.platform === 'win32') {
      // Use cmd.exe /c to execute the .cmd shim reliably
      command = 'cmd.exe';
      args = ['/c', localTsc, '-p', tsconfigPath, '--noEmit', '--skipLibCheck'];
    } else {
      command = localTsc;
      args = ['-p', tsconfigPath, '--noEmit', '--skipLibCheck'];
    }

    const opts = { cwd: frontendDir, shell: false, env: process.env };

    if (!command) {
      resolvePromise({ code: -1, output: 'No command resolved for tsc execution' });
      return;
    }

    console.log(`Detected architecture: ${process.arch}`);
    console.log(`▶️ Running TSC: command="${command}" args=${args.join(' ')} cwd=${frontendDir}`);

    let out = '';
    let err = '';
    try {
      const proc = spawn(command, args, opts);
      proc.stdout.on('data', d => { out += d.toString(); });
      proc.stderr.on('data', d => { err += d.toString(); });
      proc.on('error', (e) => {
        resolvePromise({ code: -1, output: `Failed to spawn tsc: ${e.message}` });
      });
      proc.on('close', (code) => {
        resolvePromise({ code: code ?? -1, output: out + err });
      });
    } catch (e) {
      resolvePromise({ code: -1, output: `Exception launching tsc: ${e instanceof Error ? e.message : String(e)}` });
    }
  });
}

function classify(lines) {
  const categories = {
    TS1138: [], // Parameter declaration expected
    TS7006: [], // Parameter implicitly has 'any'
    TS2307: [], // Cannot find module
    TS2322: [], // Type 'X' not assignable to type 'Y'
    TS2551: [], // Property does not exist
    TS2741: [],
    TS2742: [],
    OTHER: []
  };
  for (const line of lines) {
    let matched = false;
    for (const key of Object.keys(categories)) {
      if (key !== 'OTHER' && line.includes(key)) { categories[key].push(line); matched = true; break; }
    }
    if (!matched) categories.OTHER.push(line);
  }
  return categories;
}

function summarize(categories) {
  const summary = {};
  let total = 0;
  for (const [k,v] of Object.entries(categories)) { summary[k] = v.length; total += v.length; }
  summary.total = total;
  return summary;
}

function topN(arr, n=10) { return arr.slice(0, n); }

(async () => {
  const { output } = await runTSC();
  const lines = output.split(/\r?\n/).filter(l => /error TS\d+:/.test(l));
  const categories = classify(lines);
  // Per-file aggregation
  /** @type {Record<string,{total:number,codes:Record<string,number>}>>} */
  const perFile = {};
  const errorLineRe = /^(.*?\.[tj]sx?|.*?\.svelte)\((\d+),(\d+)\): error TS(\d+): (.*)$/;
  const parseCodes = new Set(['1005','1109','1128','1136','1137','1138','1011','1131','1434']);
  let parseErrorTotal = 0;
  for (const line of lines) {
    const m = line.match(errorLineRe);
    if (m) {
      const file = m[1];
      const code = 'TS' + m[4];
      perFile[file] = perFile[file] || { total: 0, codes: {} };
      perFile[file].total += 1;
      perFile[file].codes[code] = (perFile[file].codes[code] || 0) + 1;
      if (parseCodes.has(m[4])) parseErrorTotal++;
    }
  }
  const perFileTop = Object.entries(perFile)
    .sort((a,b) => b[1].total - a[1].total)
    .slice(0, 30)
    .map(([file,data]) => ({ file, total: data.total, topCodes: Object.entries(data.codes).sort((a,b)=>b[1]-a[1]).slice(0,5).map(([c,count])=>({code:c,count})) }));
  const parseErrorShare = lines.length ? +( (parseErrorTotal / lines.length) * 100 ).toFixed(2) : 0;
  // Build full distribution of TS error codes
  const distribution = {};
  for (const line of lines) {
    const m = line.match(/error TS(\d+):/);
    if (m) {
      const code = `TS${m[1]}`;
      distribution[code] = (distribution[code] || 0) + 1;
    }
  }
  const topCodes = Object.entries(distribution)
    .sort((a,b) => b[1]-a[1])
    .slice(0, 25)
    .map(([code,count]) => ({ code, count }));
  const summary = summarize(categories);
  const timestamp = new Date().toISOString().replace(/[:.]/g,'-');
  const progressDir = resolve(projectRoot, 'progress');
  mkdirSync(progressDir, { recursive: true });
  const jsonPath = resolve(progressDir, `ts-metrics-${timestamp}.json`);
  writeFileSync(jsonPath, JSON.stringify({ timestamp, summary, topCodes, parseErrors: { total: parseErrorTotal, percent: parseErrorShare, codes: Array.from(parseCodes).map(c=>'TS'+c) }, perFileTop, categories: Object.fromEntries(Object.entries(categories).map(([k,v])=>[k, topN(v,25)])) }, null, 2));
  // Also write latest pointer
  writeFileSync(resolve(progressDir, 'ts-metrics-latest.json'), JSON.stringify({ timestamp, summary, topCodes, parseErrors: { total: parseErrorTotal, percent: parseErrorShare } }, null, 2));
  console.log(`✅ TypeScript metrics snapshot written: ${jsonPath}`);
  console.log('Summary:', summary);
  console.log('Top Codes:', topCodes.slice(0,10));
  console.log('Top Files:', perFileTop.slice(0,5));
})();
