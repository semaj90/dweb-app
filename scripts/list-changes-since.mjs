#!/usr/bin/env node
import { exec } from 'child_process';
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