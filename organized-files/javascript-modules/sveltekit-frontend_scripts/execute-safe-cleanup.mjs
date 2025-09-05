#!/usr/bin/env node

/**
 * Execute Safe Cleanup - Remove only the 14 identified placeholder files
 * Preserves all 755 files with TODOs, Svelte 5, or valuable content
 */

import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, '..');

async function loadCleanupPlan() {
    try {
        const planPath = path.join(projectRoot, 'CLEANUP_PLAN.json');
        const data = await fs.readFile(planPath, 'utf-8');
        return JSON.parse(data);
    } catch (error) {
        console.error('❌ Could not load cleanup plan. Please run smart-targeted-cleanup.mjs first');
        process.exit(1);
    }
}

async function executeTargetedCleanup() {
    console.log('🎯 Executing Safe Targeted Cleanup');
    console.log('Removing ONLY small placeholder files without TODOs or valuable content\n');

    const cleanupPlan = await loadCleanupPlan();

    // Get only the safest files to remove
    const safeToRemove = cleanupPlan.safeToRemove.filter(file =>
        file.patterns?.hasPlaceholders &&
        !file.patterns?.hasTODOs &&
        !file.patterns?.hasSvelte5 &&
        !file.patterns?.hasValidation &&
        file.lineCount < 100 && // Extra conservative
        file.reason === 'Placeholder content without TODOs'
    );

    console.log('📊 Final Safety Check:');
    console.log(`  ✅ Files being preserved: ${cleanupPlan.preserve.length}`);
    console.log(`  🗑️ Files to remove: ${safeToRemove.length}`);
    console.log(`  📝 Files with TODOs preserved: ${cleanupPlan.preserve.filter(f => f.patterns?.hasTODOs).length}`);
    console.log(`  ⚡ Files with Svelte 5 preserved: ${cleanupPlan.preserve.filter(f => f.patterns?.hasSvelte5).length}`);

    if (safeToRemove.length === 0) {
        console.log('\n✅ No files meet the strict removal criteria. All files preserved!');
        return;
    }

    console.log('\n🗑️ Removing placeholder files:');

    let removed = 0;
    let errors = 0;

    for (const file of safeToRemove) {
        try {
            const fullPath = path.resolve(projectRoot, file.directory, file.name);

            // Double-check file exists and is small
            const stats = await fs.stat(fullPath);
            if (stats.size < 5000) { // Extra safety check - only small files
                await fs.unlink(fullPath);
                console.log(`  ✓ Removed: ${file.name} (${file.lineCount} lines, ${stats.size} bytes)`);
                removed++;
            } else {
                console.log(`  ⚠️ Skipped: ${file.name} (too large: ${stats.size} bytes)`);
            }
        } catch (error) {
            console.error(`  ❌ Error removing ${file.name}: ${error.message}`);
            errors++;
        }
    }

    console.log('\n✅ SAFE CLEANUP COMPLETE!');
    console.log('📊 Results:');
    console.log(`  🗑️ Files safely removed: ${removed}`);
    console.log(`  ❌ Errors: ${errors}`);
    console.log(`  ✅ Files preserved: ${cleanupPlan.preserve.length}`);
    console.log(`  📝 TODOs preserved: ${cleanupPlan.preserve.filter(f => f.patterns?.hasTODOs).length}`);
    console.log(`  ⚡ Svelte 5 files preserved: ${cleanupPlan.preserve.filter(f => f.patterns?.hasSvelte5).length}`);

    console.log('\n🎯 Summary:');
    console.log('  • All files with TODOs have been preserved');
    console.log('  • All Svelte 5 files have been preserved');
    console.log('  • All files with validation have been preserved');
    console.log('  • Only small placeholder files without value were removed');
    console.log('  • Conservative approach maintained - when in doubt, preserve!');
}

executeTargetedCleanup().catch(console.error);
