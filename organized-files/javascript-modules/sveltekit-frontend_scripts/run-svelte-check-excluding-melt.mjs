#!/usr/bin/env node
import { spawnSync } from 'child_process';
import { glob } from 'glob';
import fs from 'fs';
import path from 'path';

const cwd = process.cwd();

async function main(){
  const listMode = process.argv.includes('--list');
  const stripMelt = process.argv.includes('--strip-melt');
  const batchArgIndex = process.argv.findIndex(a=>a==='--batch');
  const batchSize = batchArgIndex !== -1 ? parseInt(process.argv[batchArgIndex+1]||'150',10) : null;
  const sentinel = process.argv.includes('--sentinel-error');
  const listMelt = process.argv.includes('--list-melt');
  const listBatch = process.argv.includes('--list-batch');
  const jsonOutput = process.argv.includes('--json-output');
  const files = await glob('src/**/*.svelte', { cwd });
  const safeFiles = [];
  const skippedMelt = [];
  for (const f of files){
    try {
      const full = path.join(cwd,f);
      const content = fs.readFileSync(full,'utf8');
      if(content.includes('use:melt')) { skippedMelt.push(f); continue; }
      safeFiles.push(full);
    } catch(e){ /* ignore */ }
  }
  if(safeFiles.length===0){
    console.log('No safe .svelte files found (all contain use:melt). Nothing to run.');
    process.exit(0);
  }

  console.log(`Excluding-melt runner: ${safeFiles.length} safe files selected. (${skippedMelt.length} skipped for use:melt)`);
  if(listMelt){
    console.log('Melt files (all skipped):');
    skippedMelt.forEach(f=>console.log(' -', f));
    process.exit(0);
  }
  if(skippedMelt.length && process.argv.includes('--verbose')){
    console.log('Skipped melt files:');
    skippedMelt.slice(0,50).forEach(s=>console.log(' -', s));
    if(skippedMelt.length>50) console.log(` ... (${skippedMelt.length-50} more)`);
  }
  if(listMode){
    safeFiles.forEach(f=>console.log(' -', path.relative(cwd,f)));
    process.exit(0);
  }

  let filesForCheck = safeFiles;
  if(batchSize){
    // Select subset based on optional --batch-index (0-based)
    const idxArg = process.argv.findIndex(a=>a==='--batch-index');
    const batchIndex = idxArg !== -1 ? parseInt(process.argv[idxArg+1]||'0',10) : 0;
    const start = batchIndex * batchSize;
    const end = start + batchSize;
    filesForCheck = safeFiles.slice(start,end);
    console.log(`Batch mode: batchSize=${batchSize} batchIndex=${batchIndex} (files ${start}..${end-1}) -> ${filesForCheck.length} files`);
    if(listBatch){
      filesForCheck.forEach(f=>console.log(' -', path.relative(cwd,f)));
      process.exit(0);
    }
  }
  let workDir = cwd;
  if(stripMelt){
    workDir = path.join(cwd, '.svelte-check-tmp');
    const tmpSrc = path.join(workDir,'src');
    if(fs.existsSync(workDir)){
      try { fs.rmSync(workDir, { recursive: true, force: true }); } catch(e){}
    }
    fs.mkdirSync(tmpSrc, { recursive: true });
    const meltRegex = /use:melt(\s*=\s*\{[^}]*\})?/g;
    // Copy entire src tree so imported components also have melt stripped
    const allSrcFiles = await glob('src/**/*.*', { cwd });
    for(const rel of allSrcFiles){
      const sourceAbs = path.join(cwd, rel);
      const destAbs = path.join(workDir, rel);
      const destDir = path.dirname(destAbs);
      if(!fs.existsSync(destDir)) fs.mkdirSync(destDir, { recursive: true });
      try {
        if(rel.endsWith('.svelte')){
          let content = fs.readFileSync(sourceAbs,'utf8');
            content = content.replace(meltRegex, 'data-melt-stripped');
          fs.writeFileSync(destAbs, content, 'utf8');
        } else {
          fs.copyFileSync(sourceAbs, destAbs);
        }
      } catch(e){ /* ignore per-file */ }
    }
    filesForCheck = filesForCheck.map(f => path.join(workDir, path.relative(cwd,f)));
    console.log(`Strip-melt mode: copied ${allSrcFiles.length} src files (melt stripped) with ${filesForCheck.length} target include files.`);
  }

  // Optional sentinel error injection into first file to verify diagnostics path
  if(sentinel && filesForCheck.length){
    const target = filesForCheck[0];
    try {
      const original = fs.readFileSync(target,'utf8');
      if(!original.includes('__SENTINEL_ERROR__')){
        fs.writeFileSync(target, original + '\n<script lang=\"ts\">const __SENTINEL_ERROR__: number = "x" as any;</script>\n', 'utf8');
        console.log('Injected sentinel type error into', target);
      } else {
        console.log('Sentinel already present in', target);
      }
    } catch(e){ console.log('Failed to inject sentinel:', e.message); }
  }

  // Build a temporary tsconfig that only includes the chosen files.
  const tmpTsconfigPath = path.join(cwd, '.svelte-check-temp-tsconfig-excluding-melt.json');
  const relativeIncludes = filesForCheck.map(f => path.relative(cwd, f).replace(/\\/g,'/'));
  const tmpTsconfig = {
    extends: './tsconfig.json',
    include: relativeIncludes
  };
  fs.writeFileSync(tmpTsconfigPath, JSON.stringify(tmpTsconfig, null, 2), 'utf8');
  console.log(`Include list count: ${relativeIncludes.length}`);
  console.log('Include sample:', relativeIncludes.slice(0,5));

  const outputMode = jsonOutput ? 'json' : 'human';
  const args = ['--tsconfig', path.resolve(tmpTsconfigPath), '--threshold', 'warning', '--output', outputMode];
  console.log('Running: node node_modules/svelte-check/bin/svelte-check', args.join(' '));

  // Route output to logs for later inspection.
  const logsDir = path.join(cwd, 'logs');
  if(!fs.existsSync(logsDir)) fs.mkdirSync(logsDir);
  const batchLabel = batchSize ? `-batch${process.argv.includes('--batch-index') ? process.argv[process.argv.findIndex(a=>a==='--batch-index')+1]||'0':'0'}` : '';
  const variant = stripMelt ? 'strip' : 'raw';
  const ext = jsonOutput ? 'json' : 'stdout';
  const outPath = path.join(logsDir,`svelte-check-excluding-melt${batchLabel}-${variant}.${ext}`);
  const errPath = path.join(logsDir,`svelte-check-excluding-melt${batchLabel}-${variant}.stderr`);
  const out = fs.openSync(outPath,'w');
  const err = fs.openSync(errPath,'w');
  let r = spawnSync(process.execPath, [path.join(cwd,'node_modules','svelte-check','bin','svelte-check'), ...args], { cwd, stdio: ['ignore', out, err], env: { ...process.env, FORCE_COLOR: '0', SVELTE_CHECK_WORKDIR: workDir } });
  if(r.error){
    console.log('Primary spawn failed:', r.error.message, 'attempting fallback using node command');
    r = spawnSync('node', [path.join(cwd,'node_modules','svelte-check','bin','svelte-check'), ...args], { cwd, stdio: ['ignore', out, err] });
  }
  // If human output was requested alongside json mode we still want json diagnostics separately.
  if(jsonOutput){
    const jsonPath = outPath; // currently contains human lines; write actual JSON to a sibling file.
    const realJsonPath = jsonPath.replace(/\.json$/, '.actual.json');
    const rj = spawnSync(process.execPath, [path.join(cwd,'node_modules','svelte-check','bin','svelte-check'), '--tsconfig', path.resolve(tmpTsconfigPath), '--threshold','warning','--output','json'], { cwd, encoding:'utf8', env: { ...process.env, FORCE_COLOR:'0' } });
    if(rj.stdout){
      fs.writeFileSync(realJsonPath, rj.stdout, 'utf8');
      console.log('Wrote JSON diagnostics to', realJsonPath, '(bytes:', rj.stdout.length, ')');
    } else {
      console.log('JSON diagnostics spawn produced no stdout');
    }
  }
  console.log('Spawn exit status:', r.status, 'signal:', r.signal);
  fs.closeSync(out); fs.closeSync(err);

  try {
    const outSize = fs.statSync(outPath).size;
    const errSize = fs.statSync(errPath).size;
    if(outSize === 0 && errSize === 0){
    console.log('svelte-check: no diagnostics emitted for non-melt files (exit code', r.status + ')');
    if(stripMelt) console.log('Note: strip-melt mode active; verify regex did not over-strip content.');
    } else {
      console.log(`svelte-check: diagnostics captured -> stdout(${outSize} bytes) stderr(${errSize} bytes)`);
    }
  } catch(e){ console.log('Could not stat log files'); }

  try { fs.unlinkSync(tmpTsconfigPath); } catch(e){}
  process.exit(r.status);
}

main().catch(err=>{ console.error(err); process.exit(2); });
