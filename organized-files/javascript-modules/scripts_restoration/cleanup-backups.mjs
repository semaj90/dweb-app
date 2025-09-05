#!/usr/bin/env node
/*
 * Backup Restoration Cleanup Utility
 * Scans for *.backup / numeric suffixed restored files and normalizes directory.
 * Phases:
 *  1. Discovery: enumerate candidate backup/restored files.
 *  2. Classification: active, duplicate, orphan, superseded.
 *  3. Plan generation (JSON + table) with recommended action.
 *  4. Apply phase (when --apply) executing atomic moves / deletions with safety.
 *  5. Autosolve hook (optional) – emit summary for autosolve event loop consumption.
 */
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';

const cwd = process.cwd();
const args = process.argv.slice(2);
const APPLY = args.includes('--apply');
const DRY_RUN = !APPLY;

const OUTPUT_DIR = path.join('.vscode');
const REPORT_JSON = path.join(OUTPUT_DIR, 'backup-cleanup-report.json');
const REPORT_MD = path.join(OUTPUT_DIR, 'backup-cleanup-report.md');

const BACKUP_PATTERNS = [
  /\.backup$/i,
  /\.bak$/i,
  /~$/,
  /\.old$/i,
  /copy \(\d+\)\./i,
  /\.(\d{8,})$/ // timestamp suffix
];

const HASH_CACHE = new Map();

function hashFile(fp){
  try {
    const buf = fs.readFileSync(fp);
    const h = crypto.createHash('sha1').update(buf).digest('hex');
    HASH_CACHE.set(fp, h);
    return h;
  } catch { return null; }
}

function walk(dir, out){
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  for(const e of entries){
    if(e.name === 'node_modules' || e.name === '.git' || e.name.startsWith('.turbo')) continue;
    const full = path.join(dir, e.name);
    if(e.isDirectory()) walk(full, out); else out.push(full);
  }
  return out;
}

function classify(files){
  const candidates = files.filter(f => BACKUP_PATTERNS.some(rx => rx.test(f)));
  const groups = {};
  for(const file of candidates){
    const base = file.replace(/(\.backup|\.bak|\.old|~| copy \(\d+\)|\.[0-9]{8,})$/i,'');
    (groups[base] ||= []).push(file);
  }
  const plan = [];
  for(const [base, versions] of Object.entries(groups)){
    const baseExists = fs.existsSync(base);
    const baseHash = baseExists ? hashFile(base) : null;
    const versionMeta = versions.map(v => ({ file: v, hash: hashFile(v), size: fs.existsSync(v)?fs.statSync(v).size:0, mtime: fs.existsSync(v)?fs.statSync(v).mtimeMs:0 }));
    // Determine keep vs remove: keep the newest differing hash if base missing.
    versionMeta.sort((a,b)=>b.mtime - a.mtime);
    const actions = [];
    if(!baseExists){
      // Promote most recent version to base path
      const promote = versionMeta[0];
      actions.push({ action: 'PROMOTE', from: promote.file, to: base });
      for(const rest of versionMeta.slice(1)) actions.push({ action: 'DELETE_DUP', file: rest.file });
    } else {
      for(const vm of versionMeta){
        if(vm.hash === baseHash){
          actions.push({ action: 'DELETE_REDUNDANT', file: vm.file });
        } else {
          // Different content – archive unique variant
            const archiveName = vm.file + '.archived';
            actions.push({ action: 'ARCHIVE_UNIQUE', from: vm.file, to: archiveName });
        }
      }
    }
    plan.push({ base, baseExists, versions: versionMeta, actions });
  }
  return { candidates: candidates.length, plan };
}

function apply(plan){
  for(const entry of plan){
    for(const act of entry.actions){
      try {
        switch(act.action){
          case 'PROMOTE':
            if(DRY_RUN) break;
            fs.renameSync(act.from, act.to);
            break;
          case 'DELETE_DUP':
          case 'DELETE_REDUNDANT':
            if(DRY_RUN) break;
            fs.unlinkSync(act.file);
            break;
          case 'ARCHIVE_UNIQUE':
            if(DRY_RUN) break;
            fs.renameSync(act.from, act.to);
            break;
        }
      } catch (err){
        console.error('Action failed', act, err.message);
      }
    }
  }
}

(function main(){
  if(!fs.existsSync(OUTPUT_DIR)) fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  const all = walk(cwd, []);
  const { candidates, plan } = classify(all);
  apply(plan);
  // Aggregate action counts
  const counters = { PROMOTE:0, DELETE_DUP:0, DELETE_REDUNDANT:0, ARCHIVE_UNIQUE:0 };
  for(const p of plan){
    for(const a of p.actions){ if(counters[a.action] !== undefined) counters[a.action]++; }
  }
  // Optional second pass (post-promotion) to hash-compare archived variants and purge redundant copies
  // Strategy:
  //  1. Scan for *.archived files (these were produced in first pass via ARCHIVE_UNIQUE).
  //  2. For each archived file, derive its base path (strip trailing .archived chain).
  //  3. Compute hashes: if archived hash === base hash -> delete (redundant after promotion changes).
  //  4. Within each base group, if multiple archived variants share identical hash, keep newest (by mtime) and delete older duplicates.
  //  5. Retain genuinely unique historical variants (hash differs from base and from other kept archives).
  let secondPass = null;
  if(APPLY){
    const allAfterApply = walk(cwd, []);
    const archivedFiles = allAfterApply.filter(f=>/\.archived$/i.test(f));
    const baseGroups = new Map();
    for(const af of archivedFiles){
      // Support multiple .archived suffixes; strip them all to find original base candidate
      const baseCandidate = af.replace(/(\.archived)+$/i,'');
      const meta = fs.statSync(af);
      const group = baseGroups.get(baseCandidate) || { base: baseCandidate, baseExists: fs.existsSync(baseCandidate), baseHash: null, archives: [] };
      if(group.baseExists && group.baseHash === null){
        group.baseHash = hashFile(baseCandidate);
      }
      group.archives.push({ file: af, hash: hashFile(af), size: meta.size, mtime: meta.mtimeMs });
      baseGroups.set(baseCandidate, group);
    }
    const deleteSet = new Set();
    const keptSet = new Set();
    for(const { base, baseExists, baseHash, archives } of baseGroups.values()){
      // Sort archives newest first for duplicate retention logic
      archives.sort((a,b)=> b.mtime - a.mtime);
      const seenHashes = new Set();
      for(const arch of archives){
        const sameAsBase = baseExists && arch.hash && baseHash && arch.hash === baseHash;
        if(sameAsBase){
          deleteSet.add(arch.file); // redundant copy now identical to base
          continue;
        }
        if(arch.hash && seenHashes.has(arch.hash)){
          // Older duplicate of an already kept unique hash
            deleteSet.add(arch.file);
            continue;
        }
        keptSet.add(arch.file);
        if(arch.hash) seenHashes.add(arch.hash);
      }
    }
    // Execute deletions
    let deletedCount = 0;
    for(const f of deleteSet){
      try { fs.unlinkSync(f); deletedCount++; } catch {}
    }
    secondPass = {
      archived_scanned: archivedFiles.length,
      base_groups: baseGroups.size,
      deleted_archived_duplicates: deletedCount,
      kept_unique_archives: keptSet.size,
      purged_percentage: archivedFiles.length ? +(deletedCount/archivedFiles.length*100).toFixed(2) : 0
    };
  }
  const summary = {
    timestamp: new Date().toISOString(),
    mode: DRY_RUN ? 'dry-run' : 'apply',
    root: cwd,
    candidates,
    totals: plan.reduce((acc,p)=>{acc.actions += p.actions.length; return acc;},{actions:0}),
    actions: counters,
    restored_files: counters.PROMOTE,
    archived_files: counters.ARCHIVE_UNIQUE,
    deleted_duplicates: counters.DELETE_DUP + counters.DELETE_REDUNDANT,
    second_pass: secondPass
  };
  fs.writeFileSync(REPORT_JSON, JSON.stringify({ summary, plan }, null, 2));
  const md = [
    `# Backup Cleanup Report`,
    `\n**Mode:** ${summary.mode}  `,
    `**Timestamp:** ${summary.timestamp}  `,
    `**Candidates:** ${candidates}  `,
    `**Total Actions:** ${summary.totals.actions}  `,
    `\n| Base | Base Exists | Versions | Planned Actions |`,
    `|------|------------|----------|----------------|`,
    ...plan.slice(0,200).map(p=>`| ${p.base} | ${p.baseExists} | ${p.versions.length} | ${p.actions.map(a=>a.action).join('<br>')} |`),
    plan.length>200 ? `\n_Truncated: showing first 200 groups of ${plan.length}_` : ''
  ].join('\n');
  fs.writeFileSync(REPORT_MD, md);
  console.log(`✅ Backup cleanup ${DRY_RUN? 'plan generated':'applied'}: ${REPORT_JSON}`);
})();
