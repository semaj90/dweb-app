import fs from 'fs';
import { sync as globSync } from 'glob';
import path from 'path';

const pattern = 'sveltekit-frontend/src/**/*.ts';
const files = globSync(pattern, { nodir: true });

for (const file of files) {
  try {
    const p = path.resolve(file);
    const src = fs.readFileSync(p, 'utf8');
    let out = src;
    // conservative textual replacements
    out = out.replace(/\bas\s+unknown\b/g, 'as any');
    out = out.replace(/\bas\s+unknown\[\]\b/g, 'as any[]');
    out = out.replace(/<unknown\[\]>/g, '<any[]>');
    out = out.replace(/<unknown\b/g, '<any');

    if (out !== src) {
      fs.writeFileSync(p, out, 'utf8');
      console.log('Patched:', file);
    }
  } catch (e) {
    console.error('Failed:', file, e.message);
  }
}
