import fs from 'fs';
import path from 'path';

const root = path.resolve(process.cwd(), 'node_modules', 'bits-ui', 'dist');
if (!fs.existsSync(root)) { console.error('root missing'); process.exit(1); }

function walk(dir) {
  const results = [];
  for (const name of fs.readdirSync(dir)) {
    const full = path.join(dir, name);
    if (fs.statSync(full).isDirectory()) results.push(...walk(full));
    else results.push(full);
  }
  return results;
}

const files = walk(root).filter(f => /\.(js|mjs|svelte|ts|cjs)$/.test(f));
const hits = [];
for (const f of files) {
  const s = fs.readFileSync(f, 'utf8');
  if (s.includes('$props()') || s.includes('$derived(') || s.includes('$props.')) hits.push(f);
}
console.log('Hits:', hits.length);
for (const h of hits) console.log(h);
