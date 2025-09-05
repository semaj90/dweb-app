#!/usr/bin/env node
// Parse svelte-check output, group errors, and attach suggestions.
import { spawn } from 'child_process';

const patterns = [
  { regex: /Cannot use 'bind:' with this property/, suggestion: "In Svelte 5 runes, mark the prop bindable: let { value = $bindable() } = $props()" },
  { regex: /Object literal may only specify known properties/, suggestion: 'Remove unknown props or update component Props types to include them.' },
  { regex: /Expected token >/i, suggestion: 'Check HTML syntax and ensure Svelte tags are properly closed.' },
  { regex: /is not assignable to type/, suggestion: 'Fix the type mismatch or adjust the component prop types/interfaces.' },
  { regex: /has no exported member/, suggestion: 'Verify import path and component/library version; update named exports accordingly.' },
  { regex: /Identifier .* has already been declared/, suggestion: 'Rename the variable or remove duplicate declarations.' },
  { regex: /`\$:` is not allowed in runes mode/, suggestion: 'Replace $: with $derived or $effect according to Svelte 5 runes.' }
];

function suggest(msg) {
  for (const p of patterns) if (p.regex.test(msg)) return p.suggestion;
  return 'Review the Svelte 5 migration guide for this pattern.';
}

function run(cmd, args) {
  return new Promise((resolve) => {
    const child = spawn(cmd, args, { shell: true });
    let out = '';
    child.stdout.on('data', (d) => (out += d.toString()));
    child.stderr.on('data', (d) => (out += d.toString()));
    child.on('close', () => resolve(out));
  });
}

const main = async () => {
  const output = await run('npx', ['svelte-check', '--output', 'human']);
  const lines = output.split(/\r?\n/);
  const entries = [];
  let current = null;

  for (const line of lines) {
    const loc = line.match(/^(.*):(\d+):(\d+)/);
    if (loc) {
      if (current) entries.push(current);
      current = { file: loc[1], line: Number(loc[2]), column: Number(loc[3]), messages: [] };
    } else if (current && line.trim()) {
      current.messages.push(line.trim());
    }
  }
  if (current) entries.push(current);

  const annotated = entries.map((e) => ({
    ...e,
    suggestion: suggest(e.messages.join(' '))
  }));

  console.log(JSON.stringify({ count: annotated.length, items: annotated.slice(0, 200) }, null, 2));
};

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
