#!/usr/bin/env node
/**
 * list-changes-since.mjs
 * Usage: node scripts/list-changes-since.mjs "1 day ago"
 *        node scripts/list-changes-since.mjs "2025-08-09"
 */
import { spawn } from "node:child_process";
import { writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";

function runGit(args, cwd) {
  return new Promise((resolve, reject) => {
    const p = spawn("git", args, { cwd, stdio: ["ignore", "pipe", "pipe"] });
    let out = "";
    let err = "";
    p.stdout.on("data", (d) => (out += d.toString()));
    p.stderr.on("data", (d) => (err += d.toString()));
    p.on("exit", (code) => {
      if (code === 0) resolve(out);
      else reject(new Error(err || `git exited with code ${code}`));
    });
  });
}

function parseNameStatus(output) {
  // Lines like: "M\tpath/to/file" or "A\tpath" or bare file when pretty=format: separators
  const files = new Set();
  const lines = output.split(/\r?\n/);
  for (const line of lines) {
    const s = line.trim();
    if (!s) continue;
    // try to parse name-status
    const tabIdx = s.indexOf("\t");
    if (tabIdx > 0) {
      files.add(s.slice(tabIdx + 1).trim());
    } else if (!/^[A-Z]$/.test(s)) {
      // when using --name-only, you get plain paths
      files.add(s);
    }
  }
  return Array.from(files);
}

function summarizeByDir(files) {
  const map = new Map();
  for (const f of files) {
    const d = dirname(f) || ".";
    map.set(d, (map.get(d) || 0) + 1);
  }
  return Array.from(map.entries()).sort((a, b) => b[1] - a[1]);
}

function timestamp() {
  const d = new Date();
  const pad = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}-${pad(
    d.getHours()
  )}${pad(d.getMinutes())}${pad(d.getSeconds())}`;
}

async function main() {
  const since = process.argv[2] || "1 day ago";
  const cwd = process.cwd();

  // Prefer name-status for change type context, but include pretty=format: to suppress commit headers
  const raw = await runGit(
    ["log", `--since=${since}`, "--name-status", "--pretty=format:"],
    cwd
  );
  const files = parseNameStatus(raw);
  const total = files.length;
  const byDir = summarizeByDir(files);

  const lines = [];
  lines.push(`Since: ${since}`);
  lines.push(`Total files changed: ${total}`);
  lines.push("Top directories:");
  for (const [dir, count] of byDir.slice(0, 20)) {
    lines.push(`  ${dir}  (${count})`);
  }
  lines.push("");
  lines.push("Files (first 200):");
  for (const f of files.slice(0, 200)) lines.push(`  ${f}`);

  const reportPath = join(cwd, `CHANGES-since-${timestamp()}.txt`);
  await writeFile(reportPath, lines.join("\n"));

  console.log(`TOTAL=${total}`);
  console.log(lines.slice(0, 40).join("\n"));
  console.log(`\nReport saved: ${reportPath}`);
}

main().catch((e) => {
  console.error("Failed to list changes:", e.message || e);
  process.exit(1);
});
