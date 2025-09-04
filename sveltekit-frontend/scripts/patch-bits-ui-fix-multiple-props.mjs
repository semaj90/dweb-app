import fs from 'fs';
import path from 'path';

function walk(dir) {
  const files = [];
  for (const name of fs.readdirSync(dir)) {
    const p = path.join(dir, name);
    const stat = fs.statSync(p);
    if (stat.isDirectory()) files.push(...walk(p));
    else if (stat.isFile() && (/\.svelte$/.test(name) || /\.js$/.test(name))) files.push(p);
  }
  return files;
}

const dist = path.resolve('./node_modules/bits-ui/dist');
if (!fs.existsSync(dist)) {
  console.error('bits-ui dist not found at', dist);
  process.exit(1);
}

const files = walk(dist);
let changed = 0;

for (const file of files) {
  let src = fs.readFileSync(file, 'utf8');
  const propsVarMatch = /const\s+(__svelte_props_[0-9a-f]{6})\s*=\s*\$props\(\)\s*;/m.exec(src);
  if (!propsVarMatch) continue;
  const propsVar = propsVarMatch[1];

  let out = src;
  // replace patterns like "= $props();" and ": Type = $props();" etc.
  out = out.replace(/=\s*\$props\(\)\s*;/g, `= ${propsVar};`);
  out = out.replace(/=\s*\$props\(\)\s*\n/g, `= ${propsVar}\n`);
  // also replace "= $props()" followed by comma or )
  out = out.replace(/=\s*\$props\(\)\s*(,|\))/g, `= ${propsVar} $1`);

  // If any $props() still present (standalone), replace them with propsVar
  out = out.replace(/\$props\(\)\s*([^\.\w]|$)/g, `${propsVar}$1`);

  if (out !== src) {
    const bak = file + '.orig.props2.bak';
    if (!fs.existsSync(bak)) fs.copyFileSync(file, bak);
    fs.writeFileSync(file, out, 'utf8');
    console.log('Patched', file);
    changed++;
  }
}

console.log('Done. Files changed:', changed);
