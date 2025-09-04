#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
const glob = require('glob');

const projectRoot = process.cwd();

const EVENTS = ['click','change','submit','input','keydown','keyup'];
const eventPattern = new RegExp(`on:(${EVENTS.join('|')})(\\b|:)`, 'g');

function convertTagAttributes(tagName, attrs) {
  // only convert for lowercase html-like tags
  if (!/^[a-z]/.test(tagName)) return attrs;
  return attrs.replace(eventPattern, (_, ev, suffix) => `${ev}${suffix}`);
}

function processFile(filePath) {
  let txt = fs.readFileSync(filePath, 'utf8');
  let modified = false;

  // iterate over opening tags and convert attributes for lowercase tags only
  txt = txt.replace(/<([a-zA-Z0-9-_:]+)([^>]*)>/g, (full, tag, attrs) => {
    const newAttrs = convertTagAttributes(tag, attrs);
    if (newAttrs !== attrs) modified = true;
    return `<${tag}${newAttrs}>`;
  });

  if (modified) {
    fs.writeFileSync(filePath, txt, 'utf8');
    console.log(`Patched: ${filePath}`);
  }
}

async function run() {
  const files = glob.sync('src/**/*.svelte', { cwd: projectRoot, absolute: true });
  let touched = 0;
  for (const f of files) {
    try {
      const before = fs.readFileSync(f, 'utf8');
      processFile(f);
      const after = fs.readFileSync(f, 'utf8');
      if (before !== after) touched++;
    } catch (e) {
      console.error('Failed', f, e.message);
    }
  }
  console.log(`Converted element-level events in ${touched} files.`);
}

run();
