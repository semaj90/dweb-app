#!/usr/bin/env node

/**
 * Smart Targeted Cleanup - Remove only safe placeholder/stub files
 * Preserve ALL files with TODOs, Svelte 5 patterns, and valuable content
 */

import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, '..');

// Load the comprehensive scan results
async function loadScanResults() {
    try {
        const reportPath = path.join(projectRoot, 'COMPREHENSIVE_BACKUP_SCAN_REPORT.json');
        const data = await fs.readFile(reportPath, 'utf-8');
        return JSON.parse(data);
    } catch (error) {
        console.error('❌ Could not load scan results. Please run comprehensive-backup-scan.mjs first');
        process.exit(1);
    }
}

// Categorize files for safe removal
function categorizeFiles(scanResults) {
    const categories = {
        preserve: [],
        safeToRemove: [],
        duplicates: [],
        analysis: {
            totalFiles: 0,
            withTODOs: 0,
            withSvelte5: 0,
            placeholders: 0,
            duplicateGroups: 0
        }
    };

    // Group files by base name to identify duplicates
    const fileGroups = {};

    for (const [directory, files] of Object.entries(scanResults.byDirectory)) {
        for (const file of files) {
            categories.analysis.totalFiles++;

            const baseName = file.name.replace(/\.backup.*$/, '').replace(/\.replaced$/, '');
            if (!fileGroups[baseName]) fileGroups[baseName] = [];
            fileGroups[baseName].push({ ...file, directory });

            // PRESERVE criteria (very conservative)
            const shouldPreserve =
                file.patterns?.hasTODOs ||                    // Has TODOs
                file.patterns?.hasSvelte5 ||                  // Has Svelte 5 patterns
                file.patterns?.hasValidation ||               // Has validation
                file.patterns?.isLarge ||                     // Large files likely have content
                (file.lineCount && file.lineCount > 500) ||  // Substantial content
                file.name.includes('.replaced') ||           // Replaced files are often newer
                directory === 'archives\\component-backups\\superior-svelte5' || // Already identified as superior
                !file.patterns?.hasPlaceholders;             // No placeholder patterns

            if (shouldPreserve) {
                categories.preserve.push({ ...file, directory, reason: 'Has valuable content' });

                if (file.patterns?.hasTODOs) categories.analysis.withTODOs++;
                if (file.patterns?.hasSvelte5) categories.analysis.withSvelte5++;
            } else {
                // SAFE TO REMOVE criteria (very strict)
                const isSafeToRemove =
                    file.patterns?.hasPlaceholders &&         // Has placeholder content
                    !file.patterns?.hasTODOs &&              // No TODOs
                    !file.patterns?.hasSvelte5 &&            // No Svelte 5
                    !file.patterns?.hasValidation &&         // No validation
                    !file.patterns?.isLarge &&               // Not large
                    (file.lineCount && file.lineCount < 200); // Small files

                if (isSafeToRemove) {
                    categories.safeToRemove.push({
                        ...file,
                        directory,
                        reason: 'Placeholder content without TODOs'
                    });
                    categories.analysis.placeholders++;
                } else {
                    // When in doubt, preserve
                    categories.preserve.push({ ...file, directory, reason: 'Uncertain - preserving' });
                }
            }
        }
    }

    // Identify duplicate groups
    for (const [baseName, files] of Object.entries(fileGroups)) {
        if (files.length > 1) {
            categories.analysis.duplicateGroups++;

            // Sort by recency and quality
            files.sort((a, b) => {
                // Prefer files with TODOs
                if (a.patterns?.hasTODOs && !b.patterns?.hasTODOs) return -1;
                if (!a.patterns?.hasTODOs && b.patterns?.hasTODOs) return 1;

                // Prefer Svelte 5
                if (a.patterns?.hasSvelte5 && !b.patterns?.hasSvelte5) return -1;
                if (!a.patterns?.hasSvelte5 && b.patterns?.hasSvelte5) return 1;

                // Prefer larger files
                return (b.lineCount || 0) - (a.lineCount || 0);
            });

            // Keep the best one, mark others as duplicates
            const keep = files[0];
            const duplicates = files.slice(1);

            categories.duplicates.push({
                baseName,
                keep: `${keep.directory}/${keep.name}`,
                remove: duplicates.map(f => `${f.directory}/${f.name}`)
            });
        }
    }

    return categories;
}

async function executeCleanup(categories, dryRun = true) {
    console.log(`${dryRun ? '🔍 DRY RUN -' : '🗑️'} Executing cleanup...`);

    const results = {
        removed: 0,
        errors: 0,
        preserved: categories.preserve.length,
        skipped: 0
    };

    // Only remove files that are clearly safe
    const filesToRemove = [
        ...categories.safeToRemove.filter(f =>
            f.patterns?.hasPlaceholders &&
            !f.patterns?.hasTODOs &&
            f.lineCount < 100 // Extra safety check
        )
    ];

    console.log(`\n📋 Cleanup Plan:`);
    console.log(`  ✅ Files to preserve: ${categories.preserve.length}`);
    console.log(`  🗑️ Files to remove: ${filesToRemove.length}`);
    console.log(`  📦 Duplicate groups: ${categories.analysis.duplicateGroups}`);

    if (!dryRun && filesToRemove.length > 0) {
        console.log('\n🗑️ Removing safe placeholder files...');

        for (const file of filesToRemove) {
            try {
                const fullPath = path.resolve(projectRoot, file.directory, file.name);
                await fs.unlink(fullPath);
                console.log(`  ✓ Removed: ${file.directory}/${file.name} (${file.reason})`);
                results.removed++;
            } catch (error) {
                console.error(`  ❌ Error removing ${file.name}: ${error.message}`);
                results.errors++;
            }
        }
    } else {
        console.log('\n🔍 Files that would be removed:');
        filesToRemove.slice(0, 10).forEach(file => {
            console.log(`  - ${file.directory}/${file.name} (${file.lineCount} lines, ${file.reason})`);
        });
        if (filesToRemove.length > 10) {
            console.log(`  ... and ${filesToRemove.length - 10} more files`);
        }
    }

    return results;
}

async function main() {
    console.log('🎯 Smart Targeted Cleanup');
    console.log('Criteria: Remove ONLY placeholder/stub files without TODOs or valuable content');
    console.log('Preserve: ALL files with TODOs, Svelte 5 patterns, validation, or substantial content\n');

    const scanResults = await loadScanResults();
    console.log(`📊 Loaded scan results: ${scanResults.total} backup files`);

    const categories = categorizeFiles(scanResults);

    console.log('\n🔍 Analysis Results:');
    console.log(`  📁 Total files analyzed: ${categories.analysis.totalFiles}`);
    console.log(`  📝 Files with TODOs: ${categories.analysis.withTODOs} (PRESERVE)`);
    console.log(`  ⚡ Files with Svelte 5: ${categories.analysis.withSvelte5} (PRESERVE)`);
    console.log(`  🚧 Placeholder files: ${categories.analysis.placeholders} (potential removal)`);
    console.log(`  📦 Duplicate groups: ${categories.analysis.duplicateGroups}`);

    // Show some examples of what will be preserved
    console.log('\n✅ Examples of files being PRESERVED:');
    const preserveExamples = categories.preserve
        .filter(f => f.patterns?.hasTODOs || f.patterns?.hasSvelte5)
        .slice(0, 5);

    for (const file of preserveExamples) {
        const tags = [];
        if (file.patterns?.hasTODOs) tags.push('TODO');
        if (file.patterns?.hasSvelte5) tags.push('Svelte5');
        if (file.patterns?.hasValidation) tags.push('Validation');
        console.log(`  - ${file.directory}/${file.name} [${tags.join(', ')}]`);
    }

    // Execute dry run first
    console.log('\n🔍 DRY RUN - No files will be deleted yet');
    await executeCleanup(categories, true);

    // Ask for confirmation for actual cleanup
    console.log('\n❓ Proceed with actual cleanup? (This will only remove small placeholder files)');
    console.log('   All files with TODOs, Svelte 5 patterns, or substantial content will be preserved.');

    // For now, just show what would happen - manual confirmation needed
    console.log('\n⚠️  Manual confirmation needed. Run with --execute flag when ready.');

    // Save cleanup plan
    const planPath = path.join(projectRoot, 'CLEANUP_PLAN.json');
    await fs.writeFile(planPath, JSON.stringify(categories, null, 2));
    console.log(`📋 Cleanup plan saved to: ${planPath}`);
}

main().catch(console.error);
