import fs from 'fs';
import path from 'path';

function walk(dir) {
  const files = [];
  for (const name of fs.readdirSync(dir)) {
    const p = path.join(dir, name);
    const stat = fs.statSync(p);
    if (stat.isDirectory()) files.push(...walk(p));
    else if (stat.isFile() && /\.js$/.test(name)) files.push(p);
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
  // find patterns: assignment in constructor: "this.#enabled = $derived(...);"
  const assignRegex = /this\.(#?[A-Za-z0-9_$]+)\s*=\s*\$derived\s*\(([^;\n]+)\);/g;
  let m;
  const assigns = [];
  while ((m = assignRegex.exec(src))) {
    assigns.push({ full: m[0], field: m[1].replace(/^#/, ''), expr: m[2].trim() });
  }
  if (assigns.length === 0) continue;

  let out = src;
  for (const a of assigns) {
    // find a private field declaration "#field;" or "#field = ...;"
    const fieldDeclRegex = new RegExp('(#' + a.field.replace(/[-\\/\\^$*+?.()|[\]{}]/g,'\\$&') + '\\s*;)', 'm');
    const matchDecl = fieldDeclRegex.exec(out);
    const derivedExpr = `$derived(${a.expr})`;
    if (matchDecl) {
      // replace the declaration with an initializer
      out = out.replace(matchDecl[1], `#${a.field} = ${derivedExpr};`);
      // remove the assignment in constructor
      out = out.replace(a.full, '');
      changed++;
    } else {
      // try to insert a class-field initializer after the class header
      const classHeader = /export\s+class\s+[A-Za-z0-9_$]+\s*\{/m;
      const ch = classHeader.exec(out);
      if (ch) {
        const insertPos = ch.index + ch[0].length;
        out = out.slice(0, insertPos) + `\n    #${a.field} = ${derivedExpr};` + out.slice(insertPos);
        out = out.replace(a.full, '');
        changed++;
      }
    }
  }

  if (out !== src) {
    // backup
    const bak = file + '.orig.derived.bak';
    if (!fs.existsSync(bak)) fs.copyFileSync(file, bak);
    fs.writeFileSync(file, out, 'utf8');
    console.log('Patched', file);
  }
}

console.log('Done. Files changed:', changed);
