#!/usr/bin/env node
import { spawnSync } from 'child_process';
import { glob } from 'glob';
import fs from 'fs';
import path from 'path';

const cwd = process.cwd();
const tsconfig = './tsconfig.frontend.json';

async function main(){
  const listMode = process.argv.includes('--list');
  // patterns for editor-related files
  const patterns = [
    'src/lib/components/editor/**/*.svelte',
    'src/lib/components/ui/**/*RichTextEditor.svelte',
    'src/lib/components/ui/**/NierRichTextEditor.svelte',
    'src/lib/components/**/Tiptap*.svelte'
  ];

  let files = [];
  for(const p of patterns){
    const matched = await glob(p, { cwd });
    files = files.concat(matched);
  }

  files = Array.from(new Set(files)).map(f=>path.join(cwd,f));

  const safeFiles = [];
  for(const f of files){
    try{
      const content = fs.readFileSync(f,'utf8');
      if(content.includes('use:melt')) continue;
      safeFiles.push(f);
    }catch(e){ /* ignore unreadable */ }
  }

  if(safeFiles.length===0){
    console.log('No safe editor .svelte files found (all contain use:melt). Nothing to run.');
    process.exit(0);
  }

  console.log(`Editor runner: ${safeFiles.length} safe files selected.`);
  if(listMode){
    safeFiles.forEach(f=>console.log(' -', path.relative(cwd,f)));
    process.exit(0);
  }

  // Create a temporary tsconfig that only includes the editor files so svelte-check
  // diagnoses just these files (the CLI does not accept per-file flags).
  const tmpTsconfigPath = path.join(cwd, '.svelte-check-temp-tsconfig-editor.json');
  const relativeIncludes = safeFiles.map(f => path.relative(cwd, f).replace(/\\/g, '/'));
  const tmpTsconfig = {
    extends: './tsconfig.frontend.json',
    include: relativeIncludes
  };
  fs.writeFileSync(tmpTsconfigPath, JSON.stringify(tmpTsconfig, null, 2), 'utf8');

  const args = ['svelte-check','--tsconfig', path.resolve(tmpTsconfigPath), '--threshold', 'error', '--output', 'human'];
  console.log('Running:', 'npx', ...args);

  // Extra debug: show first 3 included files and confirm tsconfig exists
  try {
    console.log('Debug includes sample:', relativeIncludes.slice(0,3));
    const stat = fs.statSync(tmpTsconfigPath);
    console.log('Temp tsconfig size bytes:', stat.size);
  } catch(e){ console.log('Could not stat temp tsconfig', e.message); }

  // ensure logs dir
  const logsDir = path.join(cwd, 'logs');
  try { if(!fs.existsSync(logsDir)) fs.mkdirSync(logsDir); } catch(e){}
  const outPath = path.join(logsDir, 'svelte-check-editor.stdout');
  const errPath = path.join(logsDir, 'svelte-check-editor.stderr');
  const out = fs.openSync(outPath, 'w');
  const err = fs.openSync(errPath, 'w');
  // Windows spawn quirk: directly invoking 'npx' may fail (ENOENT); try explicit .cmd fallback.
  // Prefer invoking via node directly to avoid Windows shell issues
  const nodeArgs = ['node', path.join(cwd,'node_modules','svelte-check','bin','svelte-check'), '--tsconfig', path.resolve(tmpTsconfigPath), '--threshold','error','--output','human'];
  let r = spawnSync(process.execPath, nodeArgs.slice(1), { cwd, stdio: ['ignore', out, err], env: { ...process.env, FORCE_COLOR: '0' } });
  // If that somehow failed, attempt legacy fallbacks
  if(r.error){
    console.log('Primary direct node invocation failed:', r.error.message);
    r = spawnSync('node', nodeArgs.slice(1), { cwd, stdio: ['ignore', out, err], env: { ...process.env } });
  }
  if(r.error){
    console.log('Spawn error:', r.error.message);
  }
  console.log('Spawn exit status:', r.status, 'signal:', r.signal);
  fs.closeSync(out); fs.closeSync(err);

  try {
    const outSize = fs.statSync(outPath).size;
    const errSize = fs.statSync(errPath).size;
    if(outSize === 0 && errSize === 0){
      console.log('svelte-check: no diagnostics emitted for selected editor files (exit code', r.status + ')');
    } else {
      console.log(`svelte-check: diagnostics captured -> stdout(${outSize} bytes) stderr(${errSize} bytes). See logs/ files.`);
    }
  } catch(e){ console.log('Could not stat log files'); }

  try { fs.unlinkSync(tmpTsconfigPath); } catch(e){}
  process.exit(r.status);
}

main().catch(err=>{ console.error(err); process.exit(2); });
