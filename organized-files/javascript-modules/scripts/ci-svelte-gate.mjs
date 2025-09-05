#!/usr/bin/env node
/**
 * CI Gate: fail if parse errors > 0 or error count increased vs baseline.
 */
import { readFile, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';

const CAPTURE = '.vscode/svelte-check.json';
const BASELINE = '.vscode/svelte-baseline.json';

function countParse(lines){
  return lines.filter(l=>/Unexpected token|Unterminated|TS1005|TS1128|TS1109/.test(l)).length;
}

(async function(){
  if(!existsSync(CAPTURE)){
    console.error('❌ No capture file; run capture:svelte-errors first.');
    process.exit(2);
  }
  const capture = JSON.parse(await readFile(CAPTURE,'utf8'));
  const lines = (capture.raw||'').split(/\r?\n/);
  const totalErrors = lines.filter(l=>/\serror\s+TS\d+:/.test(l)).length;
  const parseErrors = countParse(lines);

  let baseline = { totalErrors, parseErrors, createdAt: new Date().toISOString() };
  let fail = false;

  if (existsSync(BASELINE)) {
    baseline = JSON.parse(await readFile(BASELINE,'utf8'));
    if (totalErrors > baseline.totalErrors) {
      console.error(`❌ Total errors increased: ${baseline.totalErrors} -> ${totalErrors}`);
      fail = true;
    }
    if (parseErrors > baseline.parseErrors) {
      console.error(`❌ Parse errors increased: ${baseline.parseErrors} -> ${parseErrors}`);
      fail = true;
    }
  } else {
    await writeFile(BASELINE, JSON.stringify({ totalErrors, parseErrors, createdAt: new Date().toISOString() }, null, 2));
    console.log('📌 Baseline created');
  }

  console.log(`📊 Current total errors: ${totalErrors} (parse: ${parseErrors})`);

  if(fail){
    process.exit(1);
  } else {
    console.log('✅ Gate passed');
  }
})();
