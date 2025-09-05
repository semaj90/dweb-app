import fs from 'fs';
import path from 'path';
import crypto from 'crypto';

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
  // find occurrences of $props().<ident>
  const propAccessRegex = /\$props\(\)\.([A-Za-z_$][A-Za-z0-9_$]*)/g;
  let m;
  const matches = [];
  while ((m = propAccessRegex.exec(src))) {
    matches.push(m[1]);
  }
  if (matches.length === 0) continue;

  // generate a unique props var name
  const uid = crypto.createHash('md5').update(file).digest('hex').slice(0, 6);
  const propsVar = `__svelte_props_${uid}`;

  let out = src;

  if (/\.svelte$/.test(file)) {
    // find <script ...> tag
    const scriptOpen = /<script([\s\S]*?)>/i.exec(out);
    if (!scriptOpen) continue; // skip if no script tag
    const scriptStartIdx = scriptOpen.index + scriptOpen[0].length;
    // insert const props var after script open
    out = out.slice(0, scriptStartIdx) + `\n\tconst ${propsVar} = $props();\n` + out.slice(scriptStartIdx);
  } else {
    // .js file: insert after last import statement or at top
    const importRegex = /(?:^|\n)(import[\s\S]+?from\s+['"][^'"]+['"];?)/g;
    let lastImport;
    let lastIdx = 0;
    while ((m = importRegex.exec(out))) {
      lastImport = m[0];
      lastIdx = importRegex.lastIndex;
    }
    if (lastImport) {
      out = out.slice(0, lastIdx) + `\nconst ${propsVar} = $props();\n` + out.slice(lastIdx);
    } else {
      out = `const ${propsVar} = $props();\n` + out;
    }
  }

  // replace all $props().name with propsVar.name
  out = out.replace(propAccessRegex, `${propsVar}.$1`);

  if (out !== src) {
    const bak = file + '.orig.props.bak';
    if (!fs.existsSync(bak)) fs.copyFileSync(file, bak);
    fs.writeFileSync(file, out, 'utf8');
    console.log('Patched', file);
    changed++;
  }
}

console.log('Done. Files changed:', changed);
