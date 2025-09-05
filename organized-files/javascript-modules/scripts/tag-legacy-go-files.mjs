#!/usr/bin/env node
import { promises as fs } from 'node:fs';
import path from 'node:path';

const projectRoot = path.resolve(process.cwd());
const goRoot = path.join(projectRoot, 'go-microservice');

const SKIP_DIRS = new Set(['service', 'cmd']);
// Optional allowlist mode for extra safety. Provide comma-separated dirs relative to go-microservice
// via env ALLOWLIST_DIRS or --allowlist=dir1,dir2
const argAllow = process.argv.find(a => a.startsWith('--allowlist='));
const envAllow = process.env.ALLOWLIST_DIRS;
const ALLOWLIST = (argAllow?.split('=')[1] || envAllow || '')
  .split(',')
  .map(s => s.trim())
  .filter(Boolean);
const allowMode = ALLOWLIST.length > 0;

/**
 * Walk directory recursively and return all .go files.
 */
async function* walk(dir) {
  const entries = await fs.readdir(dir, { withFileTypes: true });
  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      // Skip vendor and specified dirs
      if (entry.name === 'vendor' || SKIP_DIRS.has(entry.name)) continue;
      // If allowlist mode, only descend into explicitly allowed top-level dirs under goRoot
      if (allowMode) {
        const relFromGoRoot = path.relative(goRoot, fullPath).split(path.sep)[0];
        if (relFromGoRoot && !ALLOWLIST.includes(relFromGoRoot)) continue;
      }
      yield* walk(fullPath);
    } else if (entry.isFile()) {
      if (fullPath.endsWith('.go')) {
        yield fullPath;
      }
    }
  }
}

function detectNewline(text) {
  const idx = text.indexOf('\n');
  if (idx === -1) return '\n';
  return text[idx - 1] === '\r' ? '\r\n' : '\n';
}

function hasLegacyTag(text) {
  return /^\s*\/\/\s*go:build\s+legacy/m.test(text) || /^\s*\/\/\s*\+build\s+legacy/m.test(text);
}

function hasGoBuildTag(text) {
  return /^\s*\/\/\s*go:build\b/m.test(text);
}

function hasPlusBuildTag(text) {
  return /^\s*\/\/\s*\+build\b/m.test(text);
}

function replaceExistingTags(text, nl) {
  // Replace first go:build expr with legacy
  text = text.replace(/^(\s*\/\/\s*go:build)\b.*$/m, '$1 legacy');
  // If there is a +build line, replace it too
  if (hasPlusBuildTag(text)) {
    text = text.replace(/^(\s*\/\/\s*\+build)\b.*$/m, '$1 legacy');
  }
  return text;
}

function prependLegacyTags(text, nl) {
  // Build constraints must be at the very top, before package and before any non-build comments
  // So we insert at start of file
  return `//go:build legacy${nl}// +build legacy${nl}${nl}` + text;
}

async function main() {
  const tagged = [];
  const skipped = [];

  try {
    // Ensure go-microservice exists
    await fs.access(goRoot);
  } catch (e) {
    console.error(`go-microservice directory not found at ${goRoot}`);
    process.exit(1);
  }

  for await (const file of walk(goRoot)) {
    try {
      const rel = path.relative(projectRoot, file);
      const relFromGoRoot = path.relative(goRoot, file);
      // If allowlist mode, ensure file is inside an allowed dir
      if (allowMode) {
        const top = relFromGoRoot.split(path.sep)[0];
        if (!ALLOWLIST.includes(top)) { continue; }
      }
      // Skip tests and go: module files
      if (file.endsWith('_test.go')) { skipped.push({ rel, reason: 'test file' }); continue; }

      let txt = await fs.readFile(file, 'utf8');
      const nl = detectNewline(txt);

      // Keep files inside go-microservice/service and go-microservice/cmd untouched (already skipped)
      // For everything else, enforce legacy tag

      if (hasLegacyTag(txt)) {
        skipped.push({ rel, reason: 'already legacy' });
        continue;
      }

      if (hasGoBuildTag(txt) || hasPlusBuildTag(txt)) {
        // Replace existing build constraints with legacy
        const updated = replaceExistingTags(txt, nl);
        if (updated !== txt) {
          await fs.writeFile(file, updated, 'utf8');
          tagged.push(rel);
        } else {
          skipped.push({ rel, reason: 'no change needed' });
        }
      } else {
        // Insert tags at file start
        const updated = prependLegacyTags(txt, nl);
        await fs.writeFile(file, updated, 'utf8');
        tagged.push(rel);
      }
    } catch (err) {
      console.error(`Failed tagging ${file}:`, err?.message || err);
    }
  }

  console.log('✅ Tagging complete');
  console.log(`Tagged files: ${tagged.length}`);
  if (tagged.length) tagged.forEach(f => console.log('  +', f));
  console.log(`Skipped files: ${skipped.length}`);
  if (skipped.length) skipped.forEach(s => console.log(`  - ${s.rel} (${s.reason})`));
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
