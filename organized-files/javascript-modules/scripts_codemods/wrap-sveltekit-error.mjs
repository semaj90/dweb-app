import fs from 'fs';
import path from 'path';
import { sync as globSync } from 'glob';

function processFile(filePath) {
  const src = fs.readFileSync(filePath, 'utf8');
  let out = src;

  // Simple regex: error(status, { ... }) or error(status, { ... }, ...)
  // Replace second arg object literal with ensureError({ ... })
  // This is conservative and won't handle nested complex expressions reliably.

  const regex = /\berror\(\s*([0-9]{3})\s*,\s*({[\s\S]*?})\s*\)/g;

  let changed = false;
  out = out.replace(regex, (match, status, obj) => {
    changed = true;
    return `error(${status}, ensureError(${obj}))`;
  });

  if (changed) {
    // Ensure import for ensureError exists
    if (!/ensureError/.test(out)) return; // shouldn't happen

  const importRegex = /import\s+\{?\s*ensureError\s*\}?\s+from\s+['"][^'\"]*(?:\$lib\/utils\/ensure-error|src\/lib\/utils\/ensure-error)['"]/;
    if (!importRegex.test(out)) {
      // Find first import block and add after
      const firstImport = out.match(/(^import[\s\S]*?;\s*)/m);
      if (firstImport) {
        const insertAt = firstImport.index + firstImport[0].length;
        out = out.slice(0, insertAt) + `\nimport { ensureError } from '$lib/utils/ensure-error';\n` + out.slice(insertAt);
      } else {
        out = `import { ensureError } from '$lib/utils/ensure-error';\n` + out;
      }
    }

    fs.writeFileSync(filePath, out, 'utf8');
    console.log(`Patched: ${filePath}`);
  }
}

function run() {
  const pattern = 'sveltekit-frontend/src/routes/**/*.ts';
  const files = globSync(pattern, { nodir: true });
  for (const f of files) {
    try { processFile(f); } catch (e) { console.error('Failed', f, e); }
  }
}

run();
