const fs = require('fs');
const glob = require('glob');

const path = require('path');

function isIgnored(p) {
  return p.split(path.sep).includes('node_modules') || p.split(path.sep).includes('.svelte-kit');
}

function findFiles(dir, results = []) {
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  for (const e of entries) {
    const full = path.join(dir, e.name);
    if (isIgnored(full)) continue;
    if (e.isDirectory()) {
      findFiles(full, results);
    } else if (e.isFile() && e.name === '+server.ts') {
      results.push(full);
    }
  }
  return results;
}

function main() {
  const root = process.cwd();
  const files = findFiles(root);
  const hits = [];
  for (const f of files) {
    try {
      const s = fs.readFileSync(f, 'utf8');
      const count = (s.match(/RequestHandler/g) || []).length;
      if (count > 1) hits.push({ file: path.relative(root, f), count });
    } catch (e) {
      // ignore read errors
    }
  }
  hits.sort((a, b) => b.count - a.count);
  if (hits.length === 0) {
    console.log('No +server.ts files found with more than one RequestHandler occurrence.');
    return;
  }
  console.log('Files with >1 RequestHandler (count):');
  for (const h of hits) console.log(`${h.file} -> ${h.count}`);
  console.log('\nTotal files reported:', hits.length);
}

main();
