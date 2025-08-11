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
import { promisify } from 'util';
import path from 'path';

const execAsync = promisify(exec);

// Get command line argument for date/time
const dateArg = process.argv[2];

if (!dateArg) {
  console.error('Usage: node scripts/list-changes-since.mjs "1 day ago" or "2025-08-09"');
  process.exit(1);
}

async function listChangesSince(since) {
  try {
    // Get all commits since the specified date
    const { stdout: commitOutput } = await execAsync(
      `git log --since="${since}" --pretty=format:"%h|%an|%ad|%s" --date=short`
    );

    if (!commitOutput.trim()) {
      console.log(`No commits found since ${since}`);
      return;
    }

    console.log(`\n📝 CHANGES SINCE ${since.toUpperCase()}\n`);
    console.log('=' * 60);

    const commits = commitOutput.trim().split('\n');
    
    for (const commit of commits) {
      const [hash, author, date, message] = commit.split('|');
      console.log(`\n🔸 ${hash} - ${author} (${date})`);
      console.log(`   ${message}`);
      
      // Get files changed in this commit
      try {
        const { stdout: filesOutput } = await execAsync(
          `git show --name-status ${hash}`
        );
        
        const lines = filesOutput.split('\n');
        const fileChanges = lines.filter(line => line.match(/^[AMDRC]\s+/));
        
        if (fileChanges.length > 0) {
          console.log('   Files changed:');
          fileChanges.forEach(change => {
            const [status, file] = change.split('\t');
            const statusMap = {
              'A': '➕ Added',
              'M': '✏️  Modified', 
              'D': '❌ Deleted',
              'R': '↩️  Renamed',
              'C': '📋 Copied'
            };
            console.log(`     ${statusMap[status] || status}: ${file}`);
          });
        }
      } catch (error) {
        console.log('     (Could not get file details)');
      }
    }

    // Get current working directory status
    console.log('\n\n📊 CURRENT WORKING DIRECTORY STATUS\n');
    console.log('=' * 60);
    
    try {
      const { stdout: statusOutput } = await execAsync('git status --porcelain');
      
      if (statusOutput.trim()) {
        console.log('\n🔄 Uncommitted changes:');
        const statusLines = statusOutput.trim().split('\n');
        statusLines.forEach(line => {
          const status = line.substring(0, 2);
          const file = line.substring(3);
          const statusMap = {
            '??': '❓ Untracked',
            ' M': '✏️  Modified',
            ' D': '❌ Deleted',
            'A ': '➕ Added (staged)',
            'M ': '✏️  Modified (staged)',
            'D ': '❌ Deleted (staged)',
            'MM': '✏️  Modified (staged & unstaged)',
            'AM': '➕ Added (staged) + modified'
          };
          console.log(`   ${statusMap[status] || status}: ${file}`);
        });
      } else {
        console.log('\n✅ Working directory is clean');
      }
    } catch (error) {
      console.log('Could not get working directory status');
    }

    // Summary statistics
    console.log('\n\n📈 SUMMARY STATISTICS\n');
    console.log('=' * 60);
    console.log(`Total commits: ${commits.length}`);
    
    // Count file changes across all commits
    let totalFiles = 0;
    let fileTypes = {};
    
    for (const commit of commits) {
      const [hash] = commit.split('|');
      try {
        const { stdout: diffOutput } = await execAsync(
          `git show --name-only --format="" ${hash}`
        );
        const files = diffOutput.trim().split('\n').filter(f => f);
        totalFiles += files.length;
        
        files.forEach(file => {
          const ext = path.extname(file) || 'no extension';
          fileTypes[ext] = (fileTypes[ext] || 0) + 1;
        });
      } catch (error) {
        // Skip if error
      }
    }
    
    console.log(`Total files changed: ${totalFiles}`);
    console.log('\nMost changed file types:');
    Object.entries(fileTypes)
      .sort(([,a], [,b]) => b - a)
      .slice(0, 5)
      .forEach(([ext, count]) => {
        console.log(`  ${ext}: ${count} changes`);
      });

  } catch (error) {
    console.error('Error:', error.message);
    
    if (error.message.includes('not a git repository')) {
      console.log('\n❌ This directory is not a git repository');
      console.log('Make sure you are in the root of your git project');
    } else if (error.message.includes('git log')) {
      console.log('\n❌ Invalid date format or git error');
      console.log('Try formats like: "1 day ago", "2025-08-09", "2 weeks ago"');
    }
    
    process.exit(1);
  }
}

// Run the function
listChangesSince(dateArg);
>>>>>>> lightweight-updates
