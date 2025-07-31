#!/usr/bin/env node
/**
 * Automated Stub/TODO Detection Script for deeds-web-app
 *
 * Scans the codebase for:
 *   - throw new Error('Not implemented')
 *   - TODO, FIXME, stub, mock, placeholder comments
 *   - Known stubbed files and methods (from todostubstimestamp.md)
 *
 * Outputs a report to [Automated Stub/TODO Detection] in todostubstimestamp.md and/or a separate file.
 */

import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const ROOT = path.resolve(__dirname, "..");
const REPORT_FILE = path.join(ROOT, "stub_todo_report.md");
const TARGETS = [
  "throw new Error('Not implemented')",
  "TODO",
  "FIXME",
  "stub",
  "mock",
  "placeholder",
];
const KNOWN_STUB_FILES = [
  "src/lib/server/services/vector-service.ts",
  "src/lib/server/services/vector.service.ts",
  "src/routes/api/qdrant/+server.ts",
  "src/routes/api/enhanced-rag/+server.ts",
  "mcp-helpers.ts",
  "ENHANCED_FEATURES_TODO.md",
  "BATCH_FIX_PROGRESS_SESSION_7.md",
  "error_summary_report.md",
  "TODO_SIMPLIFIED_FILES_FOR_REENHANCEMENT.md",
  "phase10nextsteps.md",
];

function walk(dir, extFilter = [".ts", ".js", ".svelte", ".md"]) {
  let results = [];
  fs.readdirSync(dir).forEach((file) => {
    const filePath = path.join(dir, file);
    const stat = fs.statSync(filePath);
    if (stat && stat.isDirectory()) {
      results = results.concat(walk(filePath, extFilter));
    } else if (extFilter.includes(path.extname(file))) {
      results.push(filePath);
    }
  });
  return results;
}

function scanFile(filePath, patterns) {
  const content = fs.readFileSync(filePath, "utf8");
  const lines = content.split("\n");
  const matches = [];
  lines.forEach((line, idx) => {
    for (const pat of patterns) {
      if (line.match(new RegExp(pat, "i"))) {
        matches.push({ line: idx + 1, text: line.trim(), pattern: pat });
      }
    }
  });
  return matches;
}

function runDetectStubsTodos() {
  const allFiles = walk(ROOT);
  const report = [];
  report.push("# Automated Stub/TODO Detection Report\n");
  report.push(`_Generated: ${new Date().toISOString()}_\n`);

  // Scan for patterns
  for (const file of allFiles) {
    const rel = path.relative(ROOT, file);
    const matches = scanFile(file, TARGETS);
    if (matches.length > 0) {
      report.push(`## ${rel}`);
      matches.forEach((m) => {
        report.push(`- [${m.pattern}] Line ${m.line}: ${m.text}`);
      });
      report.push("");
    }
  }

  // List known stub files
  report.push("---\n");
  report.push("## Known Stubbed Files (from documentation)\n");
  KNOWN_STUB_FILES.forEach((f) => report.push(`- ${f}`));

  fs.writeFileSync(REPORT_FILE, report.join("\n"));
  console.log(`Stub/TODO report written to ${REPORT_FILE}`);
}

// Run if this is the main module (ES module equivalent)
if (import.meta.url === `file://${process.argv[1]}`) {
  runDetectStubsTodos();
}
