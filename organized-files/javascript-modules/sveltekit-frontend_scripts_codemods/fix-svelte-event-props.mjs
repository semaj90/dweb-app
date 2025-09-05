import fs from 'fs';
import path from 'path';

const root = process.cwd();
const startDir = path.join(root, 'src');

function replaceAll(text) {
  // 1) Replace className => class
  text = text.replace(/\bclassName\b/g, 'class');

  // 2) Replace attributes like onclick={...} or onclick="..." to on:click=...
  // Avoid converting existing on:click
  text = text.replace(/\b(on[a-z][A-Za-z0-9_]*)\s*=\s*(\{[^}]*\}|"[^"]*"|'[^']*')/g, (m, ev, val) => {
    if (ev.startsWith('on:')) return m;
    return `on:${ev.slice(2)}=${val}`.replace('on::', 'on:');
  });

  // 3) Also handle known event-like attributes without braces
  text = text.replace(/\b(ontoggle|onresponse|onupload|onclose|onopen|onchange)\s*=\s*([^\s>]+)/g, (m, ev, val) => {
    if (ev.startsWith('on:')) return m;
    return `on:${ev.slice(2)}=${val}`.replace('on::', 'on:');
  });

  return text;
}

let modified = 0;

function walk(dir) {
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  for (const e of entries) {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) walk(p);
    else if (e.isFile() && p.endsWith('.svelte')) {
      try {
        const src = fs.readFileSync(p, 'utf8');
        const out = replaceAll(src);
        if (out !== src) {
          fs.writeFileSync(p, out, 'utf8');
          modified++;
        }
      } catch (err) {
        console.error('ERROR', p, err.message);
      }
    }
  }
}

walk(startDir);
console.log(`Codemod complete. Modified ${modified} files.`);
