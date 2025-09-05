#!/usr/bin/env node
'use strict';

import fs from 'fs/promises';
import fsSync from 'fs';
import path from 'path';

const HANDLER_REGEX = /export\s+(?:const|let|var)\s+([A-Z]+)\s*=|export\s+(?:async\s+)?function\s+([A-Z]+)\s*\(|exports\.([A-Z]+)\s*=/g;
const CANDIDATE_EXTS = new Set(['.js', '.ts', '.mjs', '.cjs']);

async function fileExists(p) {
  try {
	await fs.access(p);
	return true;
  } catch {
	return false;
  }
}

async function findRoutesRoot(root) {
  const candidates = [
	path.join(root, 'src', 'routes'),
	path.join(root, 'routes'),
	path.join(root, 'src'),
  ];
  for (const c of candidates) {
	if (await fileExists(c)) return c;
  }
  return root;
}

async function walk(dir, cb) {
  const entries = await fs.readdir(dir, { withFileTypes: true });
  for (const e of entries) {
	const full = path.join(dir, e.name);
	if (e.isDirectory()) {
	  await walk(full, cb);
	} else if (e.isFile()) {
	  await cb(full);
	}
  }
}

function normalizeRoute(routesRoot, file) {
  const dirname = path.dirname(file);
  let rel = path.relative(routesRoot, dirname);
  rel = rel.split(path.sep).map(seg => seg === 'index' ? '' : seg).filter(Boolean).join('/');
  return '/' + rel;
}

async function main() {
  const root = process.cwd();
  const routesRoot = await findRoutesRoot(root);

  const serverFiles = [];
  await walk(root, async (file) => {
	const base = path.basename(file);
	const ext = path.extname(file).toLowerCase();
	if (!CANDIDATE_EXTS.has(ext)) return;
	if (base.startsWith('+server')) {
	  serverFiles.push(file);
	  return;
	}
	// Also include +page.server.* which can contain handlers in some setups
	if (base.startsWith('+page.server')) {
	  serverFiles.push(file);
	}
  });

  if (serverFiles.length === 0) {
	console.log('No +server or +page.server files found.');
	return;
  }

  const issues = [];
  const routeMap = new Map(); // route -> [file,...]

  for (const file of serverFiles) {
	let content;
	try {
	  content = await fs.readFile(file, 'utf8');
	} catch (err) {
	  issues.push({ type: 'read-error', file, message: String(err) });
	  continue;
	}

	// collect handler occurrences in this file
	const counts = new Map();
	let m;
	HANDLER_REGEX.lastIndex = 0;
	while ((m = HANDLER_REGEX.exec(content)) !== null) {
	  const name = (m[1] || m[2] || m[3] || '').trim();
	  if (!name) continue;
	  counts.set(name, (counts.get(name) || 0) + 1);
	}

	const dupHandlers = [...counts.entries()].filter(([,c]) => c > 1).map(([h,c]) => ({ handler: h, count: c }));
	if (dupHandlers.length) {
	  issues.push({ type: 'duplicate-handler-in-file', file, details: dupHandlers });
	}

	// map file to route for cross-file duplicate checks
	let route;
	try {
	  route = normalizeRoute(routesRoot, file);
	} catch {
	  route = path.relative(root, path.dirname(file)).split(path.sep).join('/');
	}
	routeMap.set(route, (routeMap.get(route) || []).concat(file));
  }

  // detect multiple files mapped to same route
  for (const [route, files] of routeMap.entries()) {
	if (files.length > 1) {
	  issues.push({ type: 'multiple-server-files-for-route', route, files });
	}
  }

  // output results
  if (issues.length === 0) {
	console.log('No duplicate RequestHandler exports or route collisions detected.');
	return;
  }

  console.error('Found issues:');
  for (const it of issues) {
	if (it.type === 'duplicate-handler-in-file') {
	  console.error(`- Duplicate handler exports in file: ${it.file}`);
	  for (const d of it.details) {
		console.error(`    ${d.handler} -> ${d.count} occurrences`);
	  }
	} else if (it.type === 'multiple-server-files-for-route') {
	  console.error(`- Multiple +server files map to the same route "${it.route}":`);
	  for (const f of it.files) console.error(`    ${f}`);
	} else if (it.type === 'read-error') {
	  console.error(`- Could not read file ${it.file}: ${it.message}`);
	} else {
	  console.error(`- ${JSON.stringify(it)}`);
	}
  }

  process.exit(1);
}

main().catch(err => {
  console.error('Unexpected error:', err);
  process.exit(2);
});
