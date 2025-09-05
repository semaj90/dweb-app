#!/usr/bin/env node

/**
 * Safe Removal Script - Execute cleanup of identified placeholder/stub files
 * Only removes files that were confidently identified as inferior with no important TODOs
 */

import fs from 'fs/promises';
import path from 'path';

// Files identified for safe removal (90%+ confidence, no important TODOs)
const REMOVAL_CANDIDATES = [
    // Legacy backups without important TODOs (80% confidence)
    'src/lib/components/ui/bits-ui-demo/BitsUnoDemo.svelte.backup.1754931625258',
    'src/lib/components/ui/drawer/DialogRoot.svelte.backup.1754931625294',
    'src/lib/components/ui/bits-ui-demo/HeadlessDemo.svelte.backup.1754931625147',
    'src/lib/components/canvas/LegalDocumentEditor.svelte.backup.1754931625079',
    'src/lib/components/login/LoginModal.svelte.backup.1754931625011',
    'src/lib/components/ui/NoteViewerModal.svelte.backup.1754931625336',
    'src/lib/components/canvas/WysiwygEditor.svelte.backup.1754931625091',

    // Placeholder content without TODOs (90% confidence)
    'src/routes/cases/+page.svelte.backup',
    'src/routes/search/+page.server.ts.backup'
];

async function safeRemoveFile(filePath) {
    try {
        // Create backup in archive before removing
        const fileName = path.basename(filePath);
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const archivePath = path.join('archives', 'removed-placeholders', `${fileName}_${timestamp}`);

        // Ensure archive directory exists
        await fs.mkdir(path.dirname(archivePath), { recursive: true });

        // Copy to archive first
        await fs.copyFile(filePath, archivePath);
        console.log(`📦 Archived: ${fileName} → ${archivePath}`);

        // Now safely remove original
        await fs.unlink(filePath);
        console.log(`🗑️ Removed: ${filePath}`);

        return { success: true, archived: archivePath };

    } catch (error) {
        console.error(`❌ Failed to remove ${filePath}: ${error.message}`);
        return { success: false, error: error.message };
    }
}

async function main() {
    console.log('🚀 SAFE REMOVAL EXECUTION');
    console.log('═══════════════════════════');
    console.log(`📂 Removing ${REMOVAL_CANDIDATES.length} identified placeholder/stub files`);
    console.log('📦 All files will be archived before removal for safety\n');

    const results = {
        total: REMOVAL_CANDIDATES.length,
        removed: 0,
        archived: 0,
        failed: 0,
        errors: []
    };

    for (const filePath of REMOVAL_CANDIDATES) {
        console.log(`\n🔍 Processing: ${path.basename(filePath)}`);

        try {
            // Check if file exists first
            await fs.access(filePath);

            const result = await safeRemoveFile(filePath);
            if (result.success) {
                results.removed++;
                results.archived++;
                console.log(`✅ Successfully removed and archived`);
            } else {
                results.failed++;
                results.errors.push({ file: filePath, error: result.error });
                console.log(`❌ Failed: ${result.error}`);
            }

        } catch (error) {
            if (error.code === 'ENOENT') {
                console.log(`⚠️ File not found (already removed): ${filePath}`);
            } else {
                results.failed++;
                results.errors.push({ file: filePath, error: error.message });
                console.log(`❌ Error accessing file: ${error.message}`);
            }
        }
    }

    // Generate removal summary
    console.log('\n📊 REMOVAL SUMMARY');
    console.log('═══════════════════');
    console.log(`📂 Total candidates: ${results.total}`);
    console.log(`🗑️ Successfully removed: ${results.removed}`);
    console.log(`📦 Files archived: ${results.archived}`);
    console.log(`❌ Failed operations: ${results.failed}`);

    if (results.errors.length > 0) {
        console.log('\n❌ ERRORS:');
        results.errors.forEach((error, index) => {
            console.log(`${index + 1}. ${path.basename(error.file)}: ${error.error}`);
        });
    }

    // Save removal log
    const removalLog = {
        timestamp: new Date().toISOString(),
        operation: 'safe-removal',
        candidates: REMOVAL_CANDIDATES,
        results: results,
        note: 'All files were identified as placeholder/stub content without important TODOs'
    };

    const logPath = path.join('archives', 'removal-logs', `removal-${Date.now()}.json`);
    await fs.mkdir(path.dirname(logPath), { recursive: true });
    await fs.writeFile(logPath, JSON.stringify(removalLog, null, 2));

    console.log(`\n📋 Removal log saved: ${logPath}`);

    if (results.removed > 0) {
        console.log('\n✅ CLEANUP SUCCESSFUL!');
        console.log(`   Removed ${results.removed} placeholder/stub files`);
        console.log('   All files safely archived before removal');
        console.log('   TODOs preserved in EXTRACTED_TODOS.md');
    } else {
        console.log('\n⚠️ No files were removed');
        console.log('   Check if files already exist or paths are correct');
    }
}

main().catch(console.error);
