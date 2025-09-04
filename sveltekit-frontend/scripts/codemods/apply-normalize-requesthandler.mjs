#!/usr/bin/env node
import fs from 'fs/promises';
import path from 'path';

const root = process.cwd();
const srcDir = path.join(root, 'src');

async function walk(dir) {
  const entries = await fs.readdir(dir, { withFileTypes: true });
  const files = [];
  for (const e of entries) {
    const full = path.join(dir, e.name);
    if (e.isDirectory()) {
      files.push(...await walk(full));
    } else {
      files.push(full);
    }
  }
  return files;
}

function removeRequestHandlerImports(content) {
  // Remove any import lines that mention RequestHandler (from any module)
  // Keep other imports on the same line if they do not reference RequestHandler
  // We match lines containing RequestHandler and 'import' and 'from'
  const lines = content.split(/\r?\n/);
  const filtered = lines.filter(line => {
    const trimmed = line.trim();
    if (!trimmed.startsWith('import')) return true;
    // If line contains RequestHandler token, drop the line
    if (/RequestHandler/.test(line)) return false;
    return true;
  });
  // Remove consecutive blank lines at top introduced by removal
  return filtered.join('\n');
}

function ensureTypesImport(content) {
  const typesImport = "import type { RequestHandler } from './$types';";
  if (content.includes("from './$types'") || content.includes('from "./$types"')) return content;

  const lines = content.split(/\r?\n/);
  // find last import line index
  let insertAt = 0;
  for (let i = 0; i < lines.length; i++) {
    if (lines[i].startsWith('import')) insertAt = i + 1;
    else if (lines[i].trim() === '') continue;
    else if (insertAt === 0) insertAt = i; // no imports, insert at top
    if (lines[i] && !lines[i].startsWith('import')) break;
  }
  // prevent duplicate if already exists elsewhere
  if (content.includes(typesImport)) return content;

  lines.splice(insertAt, 0, typesImport, '');
  return lines.join('\n');
}

(async () => {
  try {
    console.log('Scanning for +server.ts files under', srcDir);
    const all = await walk(srcDir);
    const target = all.filter(f => f.endsWith('+server.ts'));
    console.log('Found', target.length, '+server.ts files');

    const modified = [];
    for (const file of target) {
      let txt = await fs.readFile(file, 'utf8');
      const before = txt;
      txt = removeRequestHandlerImports(txt);
      txt = ensureTypesImport(txt);
      if (txt !== before) {
        await fs.writeFile(file, txt, 'utf8');
        modified.push(path.relative(root, file));
        console.log('Modified:', file);
      }
    }

    console.log('Done. Files modified:', modified.length);
    if (modified.length > 0) {
      console.log(modified.join('\n'));
    }
  } catch (err) {
    console.error('Error applying codemod:', err);
    process.exit(2);
  }
})();
