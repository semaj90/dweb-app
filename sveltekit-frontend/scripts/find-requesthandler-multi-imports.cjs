const fs = require('fs');
const path = require('path');

function find(dir, arr = []) {
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name);
    if (p.includes('node_modules') || p.includes('.svelte-kit')) continue;
    if (e.isDirectory()) find(p, arr);
    else if (e.isFile() && e.name === '+server.ts') arr.push(p);
  }
  return arr;
}

const files = find(process.cwd());
const results = [];
for (const f of files) {
  const s = fs.readFileSync(f, 'utf8');
  const imports = Array.from(s.matchAll(/import[^\n]*RequestHandler/g));
  if (imports.length > 1) results.push({ file: path.relative(process.cwd(), f), count: imports.length });
}
results.sort((a, b) => b.count - a.count);
if (results.length === 0) console.log('No files found with multiple import lines referencing RequestHandler.');
else {
  console.log('Files with multiple RequestHandler import lines (count):');
  for (const r of results) console.log(`${r.file} -> ${r.count}`);
  console.log('\nTotal:', results.length);
}
