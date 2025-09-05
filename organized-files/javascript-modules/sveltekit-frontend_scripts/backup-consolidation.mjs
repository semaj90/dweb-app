#!/usr/bin/env node
import { readFileSync, writeFileSync, existsSync, mkdirSync, renameSync, unlinkSync } from 'fs';
import { join, dirname, basename } from 'path';
import { glob } from 'glob';

console.log('🗂️ BACKUP FILE CONSOLIDATION & CLEANUP');
console.log('=====================================');

// Create a consolidation directory
const consolidatedBackupsDir = 'archives/component-backups/consolidated';
const inferiorMeltUIDir = 'archives/component-backups/inferior-melt-ui';
const superiorSvelte5Dir = 'archives/component-backups/superior-svelte5';

// Ensure directories exist
[consolidatedBackupsDir, inferiorMeltUIDir, superiorSvelte5Dir].forEach(dir => {
  mkdirSync(dir, { recursive: true });
  console.log(`📁 Created directory: ${dir}`);
});

async function analyzeAndConsolidate() {
  console.log('\n🔍 Analyzing backup files...');

  // Find all backup files
  const backupFiles = await glob.glob('src/**/*.backup*');
  const backupAnalysis = new Map();

  for (const backupFile of backupFiles) {
    try {
      const content = readFileSync(backupFile, 'utf8');
      const originalFile = backupFile.replace(/\.backup.*$/, '');

      // Determine characteristics
      const isMeltUI = content.includes('@melt-ui/svelte') ||
                      content.includes('createDialog') ||
                      content.includes('createButton') ||
                      content.includes('createSelect') ||
                      content.includes('melt(');

      const isSvelte5 = content.includes('$props()') ||
                       content.includes('$bindable()') ||
                       content.includes('$state(') ||
                       content.includes('$derived(');

      const hasBitsUI = content.includes('bits-ui') || content.includes('@bits-ui');
      const hasValidation = content.includes('zod') ||
                          content.includes('validate') ||
                          content.includes('schema') ||
                          content.includes('z.');

      const hasTodos = content.includes('TODO') || content.includes('FIXME');

      if (!backupAnalysis.has(originalFile)) {
        backupAnalysis.set(originalFile, {
          original: originalFile,
          backups: []
        });
      }

      backupAnalysis.get(originalFile).backups.push({
        path: backupFile,
        isMeltUI,
        isSvelte5,
        hasBitsUI,
        hasValidation,
        hasTodos,
        size: content.length,
        timestamp: getFileTimestamp(backupFile)
      });

    } catch (err) {
      console.error(`❌ Error analyzing ${backupFile}:`, err.message);
    }
  }

  console.log(`\n📊 Found ${backupFiles.length} backup files for ${backupAnalysis.size} original files`);

  // Analyze current files too
  console.log('\n🔍 Analyzing current files...');
  for (const [originalFile, data] of backupAnalysis) {
    if (existsSync(originalFile)) {
      try {
        const content = readFileSync(originalFile, 'utf8');

        data.current = {
          path: originalFile,
          isMeltUI: content.includes('@melt-ui/svelte') ||
                   content.includes('createDialog') ||
                   content.includes('createButton'),
          isSvelte5: content.includes('$props()') ||
                    content.includes('$bindable()') ||
                    content.includes('$state(') ||
                    content.includes('$derived('),
          hasBitsUI: content.includes('bits-ui') || content.includes('@bits-ui'),
          hasValidation: content.includes('zod') ||
                        content.includes('validate') ||
                        content.includes('schema'),
          hasTodos: content.includes('TODO') || content.includes('FIXME'),
          size: content.length
        };
      } catch (err) {
        console.error(`❌ Error analyzing current ${originalFile}:`, err.message);
      }
    }
  }

  // Consolidate and categorize
  console.log('\n🗂️ Consolidating files...');
  let inferiorMeltCount = 0;
  let superiorSvelte5Count = 0;
  let duplicateCount = 0;

  for (const [originalFile, data] of backupAnalysis) {
    const componentName = basename(originalFile, '.svelte');
    console.log(`\n📁 Processing: ${componentName}`);

    // Sort backups by quality (Svelte 5 > Bits UI > Validation > Recent timestamp)
    const sortedBackups = data.backups.sort((a, b) => {
      if (a.isSvelte5 !== b.isSvelte5) return b.isSvelte5 - a.isSvelte5;
      if (a.hasBitsUI !== b.hasBitsUI) return b.hasBitsUI - a.hasBitsUI;
      if (a.hasValidation !== b.hasValidation) return b.hasValidation - a.hasValidation;
      return b.timestamp - a.timestamp;
    });

    // Identify the best backup
    const bestBackup = sortedBackups[0];
    const currentFile = data.current;

    console.log(`  📊 Analysis:`);
    console.log(`    Current: Svelte5=${currentFile?.isSvelte5}, BitsUI=${currentFile?.hasBitsUI}, MeltUI=${currentFile?.isMeltUI}`);
    console.log(`    Best Backup: Svelte5=${bestBackup?.isSvelte5}, BitsUI=${bestBackup?.hasBitsUI}, MeltUI=${bestBackup?.isMeltUI}`);

    // Move inferior melt-ui files
    for (const backup of sortedBackups) {
      if (backup.isMeltUI && !backup.isSvelte5) {
        const destPath = join(inferiorMeltUIDir, basename(backup.path));
        try {
          renameSync(backup.path, destPath);
          inferiorMeltCount++;
          console.log(`  🗑️ Moved inferior melt-ui file: ${basename(backup.path)}`);
        } catch (err) {
          console.error(`  ❌ Failed to move ${backup.path}:`, err.message);
        }
      }
    }

    // Move superior Svelte 5 files
    const superiorBackups = sortedBackups.filter(b =>
      (b.isSvelte5 && b.hasBitsUI) ||
      (b.hasValidation && b.isSvelte5) ||
      (b.hasTodos && b.isSvelte5)
    );

    for (const backup of superiorBackups) {
      if (backup.isMeltUI && !backup.isSvelte5) continue; // Already moved

      const destPath = join(superiorSvelte5Dir, `${componentName}-${backup.timestamp}.svelte`);
      try {
        if (existsSync(backup.path)) {
          renameSync(backup.path, destPath);
          superiorSvelte5Count++;
          console.log(`  ✅ Archived superior Svelte5 file: ${basename(destPath)}`);
        }
      } catch (err) {
        console.error(`  ❌ Failed to move ${backup.path}:`, err.message);
      }
    }

    // Remove duplicate backups (same size and similar content)
    const remainingBackups = sortedBackups.filter(b => existsSync(b.path));
    for (let i = 1; i < remainingBackups.length; i++) {
      const current = remainingBackups[i];
      const previous = remainingBackups[i-1];

      if (Math.abs(current.size - previous.size) < 100) {
        try {
          if (existsSync(current.path)) {
            unlinkSync(current.path);
            duplicateCount++;
            console.log(`  🗂️ Removed duplicate: ${basename(current.path)}`);
          }
        } catch (err) {
          console.error(`  ❌ Failed to remove ${current.path}:`, err.message);
        }
      }
    }
  }

  // Generate summary report
  const summaryReport = `# Backup Consolidation Report
Generated: ${new Date().toISOString()}

## Summary
- 📁 Total backup files processed: ${backupFiles.length}
- 🗑️ Inferior melt-ui files moved: ${inferiorMeltCount}
- ✅ Superior Svelte5 files archived: ${superiorSvelte5Count}
- 🗂️ Duplicate files removed: ${duplicateCount}

## Directories
- Inferior melt-ui files: \`${inferiorMeltUIDir}\`
- Superior Svelte5 files: \`${superiorSvelte5Dir}\`

## Component Analysis
${Array.from(backupAnalysis.entries()).map(([file, data]) => {
  const name = basename(file, '.svelte');
  const current = data.current || {};
  const best = data.backups.sort((a, b) => {
    if (a.isSvelte5 !== b.isSvelte5) return b.isSvelte5 - a.isSvelte5;
    if (a.hasBitsUI !== b.hasBitsUI) return b.hasBitsUI - a.hasBitsUI;
    return b.timestamp - a.timestamp;
  })[0];

  return `
### ${name}
- **Current**: Svelte5: ${current.isSvelte5 ? '✅' : '❌'}, BitsUI: ${current.hasBitsUI ? '✅' : '❌'}, MeltUI: ${current.isMeltUI ? '⚠️' : '✅'}
- **Backups**: ${data.backups.length} files
- **Best**: Svelte5: ${best?.isSvelte5 ? '✅' : '❌'}, BitsUI: ${best?.hasBitsUI ? '✅' : '❌'}, Validation: ${best?.hasValidation ? '✅' : '❌'}
`;
}).join('')}
`;

  writeFileSync(join(consolidatedBackupsDir, 'CONSOLIDATION_REPORT.md'), summaryReport);

  console.log(`\n✅ CONSOLIDATION COMPLETE!`);
  console.log(`📊 Summary:`);
  console.log(`  🗑️ Moved ${inferiorMeltCount} inferior melt-ui files`);
  console.log(`  ✅ Archived ${superiorSvelte5Count} superior Svelte5 files`);
  console.log(`  🗂️ Removed ${duplicateCount} duplicate files`);
  console.log(`  📋 Report saved to: ${join(consolidatedBackupsDir, 'CONSOLIDATION_REPORT.md')}`);
}

function getFileTimestamp(filename) {
  const match = filename.match(/\.backup\.(\d+)$/);
  return match ? parseInt(match[1]) : Date.now();
}

analyzeAndConsolidate().catch(console.error);
