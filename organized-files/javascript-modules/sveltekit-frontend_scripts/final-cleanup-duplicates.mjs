#!/usr/bin/env node

/**
 * Final Cleanup: Remove only duplicate neutral backup files
 * Keep: Superior Svelte 5 files (untouched, as requested)
 * Remove: Duplicate .backup files that are neutral/old versions
 */

import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, '..');

async function findDuplicateBackupFiles() {
    const duplicateFiles = [];

    // Find files with multiple .backup versions (neutral duplicates only)
    async function findDuplicatesInDir(dir) {
        try {
            const entries = await fs.readdir(dir, { withFileTypes: true });

            // Group files by base name
            const fileGroups = {};

            for (const entry of entries) {
                if (entry.isFile() && entry.name.includes('.backup')) {
                    // Extract base name (remove .backup.timestamp or just .backup)
                    let baseName = entry.name.replace(/\.backup.*$/, '');
                    if (!fileGroups[baseName]) {
                        fileGroups[baseName] = [];
                    }
                    fileGroups[baseName].push({
                        name: entry.name,
                        fullPath: path.join(dir, entry.name)
                    });
                }
            }

            // Find groups with multiple backup files
            for (const [baseName, files] of Object.entries(fileGroups)) {
                if (files.length > 1) {
                    // Keep the most recent (by timestamp), remove the rest
                    files.sort((a, b) => {
                        // Extract timestamps or use modification time
                        const timestampA = a.name.match(/\.backup\.(\d+)/)?.[1] || '0';
                        const timestampB = b.name.match(/\.backup\.(\d+)/)?.[1] || '0';
                        return parseInt(timestampB) - parseInt(timestampA);
                    });

                    // Add all but the first (most recent) to duplicates list
                    for (let i = 1; i < files.length; i++) {
                        duplicateFiles.push(files[i].fullPath);
                    }
                }
            }

            // Recursively check subdirectories
            for (const entry of entries) {
                if (entry.isDirectory()) {
                    await findDuplicatesInDir(path.join(dir, entry.name));
                }
            }

        } catch (error) {
            // Skip inaccessible directories
        }
    }

    await findDuplicatesInDir(projectRoot);
    return duplicateFiles;
}

async function main() {
    console.log('🧹 Final Cleanup: Removing duplicate neutral backup files...');
    console.log('✅ Keeping: All Superior Svelte 5 files (untouched as requested)');
    console.log('🗑️ Removing: Only duplicate/old backup versions');

    const duplicateFiles = await findDuplicateBackupFiles();
    console.log(`\n📁 Found ${duplicateFiles.length} duplicate backup files to remove`);

    let removed = 0;
    let errors = 0;

    for (const filePath of duplicateFiles) {
        try {
            await fs.unlink(filePath);
            console.log(`🗑️ Removed: ${path.basename(filePath)}`);
            removed++;
        } catch (error) {
            console.error(`❌ Error removing ${path.basename(filePath)}: ${error.message}`);
            errors++;
        }
    }

    console.log('\n✅ FINAL CLEANUP COMPLETE!');
    console.log('📊 Summary:');
    console.log(`  🗑️ Duplicate files removed: ${removed}`);
    console.log(`  ❌ Errors: ${errors}`);
    console.log('  ✅ Superior Svelte 5 files: Kept in place (as requested)');
    console.log('  🎯 Workspace: Now clean and organized');
}

main().catch(console.error);
