#!/usr/bin/env node

/**
 * Comprehensive Backup File Scanner
 * Scan ALL components and directories to see what backup files actually exist
 */

import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, '..');

// Directories to scan
const scanDirectories = [
    'src',
    '_consolidated-backups',
    'archived-backups',
    'archives',
    'scripts',
    '.'
];

async function scanForBackups() {
    const backupFiles = {
        byDirectory: {},
        byType: {},
        total: 0,
        patterns: {
            hasBackupExtension: 0,
            hasTimestamp: 0,
            hasReplaced: 0,
            hasTODOs: 0,
            hasPlaceholders: 0,
            hasSvelte5: 0,
            hasMeltUI: 0
        }
    };

    async function scanDirectory(dir, relativePath = '') {
        try {
            const entries = await fs.readdir(dir, { withFileTypes: true });

            for (const entry of entries) {
                const fullPath = path.join(dir, entry.name);
                const relativeEntryPath = path.join(relativePath, entry.name);

                if (entry.isDirectory()) {
                    // Skip node_modules and other large directories
                    if (!['node_modules', '.git', '.svelte-kit', 'build', 'dist'].includes(entry.name)) {
                        await scanDirectory(fullPath, relativeEntryPath);
                    }
                } else if (entry.isFile()) {
                    // Check if it's a backup file
                    const isBackup = entry.name.includes('.backup') ||
                                   entry.name.includes('.replaced') ||
                                   entry.name.includes('.old') ||
                                   entry.name.includes('.bak');

                    if (isBackup) {
                        const dirKey = relativePath || 'root';
                        if (!backupFiles.byDirectory[dirKey]) {
                            backupFiles.byDirectory[dirKey] = [];
                        }

                        const fileInfo = {
                            name: entry.name,
                            path: relativeEntryPath,
                            fullPath: fullPath,
                            size: 0,
                            type: path.extname(entry.name),
                            hasTimestamp: /\.\d+$/.test(entry.name),
                            isReplaced: entry.name.includes('.replaced')
                        };

                        try {
                            const stats = await fs.stat(fullPath);
                            fileInfo.size = stats.size;
                            fileInfo.modified = stats.mtime;
                        } catch (e) {
                            // Skip if can't stat
                        }

                        // Analyze content for patterns
                        try {
                            const content = await fs.readFile(fullPath, 'utf-8');
                            fileInfo.lineCount = content.split('\n').length;

                            // Pattern detection
                            fileInfo.patterns = {
                                hasTODOs: /TODO|FIXME|NOTE:|@todo/gi.test(content),
                                hasPlaceholders: /placeholder|stub|mock|TODO:|FIXME:|not implemented/gi.test(content),
                                hasSvelte5: /\$props\(\)|\$state\(\)|\$bindable\(\)|\$derived\(\)/g.test(content),
                                hasMeltUI: /createDialog|createButton|melt-ui|createAccordion/g.test(content),
                                hasBitsUI: /bits-ui|Dialog\.Root|Button\.Root/g.test(content),
                                hasValidation: /z\.|schema\.|validate/gi.test(content),
                                isEmpty: content.trim().length < 50,
                                isLarge: content.length > 5000
                            };

                            // Update global counters
                            if (fileInfo.patterns.hasTODOs) backupFiles.patterns.hasTODOs++;
                            if (fileInfo.patterns.hasPlaceholders) backupFiles.patterns.hasPlaceholders++;
                            if (fileInfo.patterns.hasSvelte5) backupFiles.patterns.hasSvelte5++;
                            if (fileInfo.patterns.hasMeltUI) backupFiles.patterns.hasMeltUI++;

                        } catch (e) {
                            fileInfo.contentError = e.message;
                        }

                        backupFiles.byDirectory[dirKey].push(fileInfo);

                        // Track by type
                        const ext = fileInfo.type || 'no-ext';
                        if (!backupFiles.byType[ext]) backupFiles.byType[ext] = 0;
                        backupFiles.byType[ext]++;

                        backupFiles.total++;

                        // Update pattern counters
                        if (fileInfo.hasTimestamp) backupFiles.patterns.hasTimestamp++;
                        if (fileInfo.isReplaced) backupFiles.patterns.hasReplaced++;
                        if (entry.name.includes('.backup')) backupFiles.patterns.hasBackupExtension++;
                    }
                }
            }

        } catch (error) {
            console.warn(`⚠️ Skipped directory ${dir}: ${error.message}`);
        }
    }

    // Scan all specified directories
    for (const dirName of scanDirectories) {
        const dirPath = path.join(projectRoot, dirName);
        try {
            await fs.access(dirPath);
            console.log(`🔍 Scanning: ${dirName}/`);
            await scanDirectory(dirPath, dirName === '.' ? '' : dirName);
        } catch (error) {
            console.log(`⚪ Skipping: ${dirName}/ (doesn't exist)`);
        }
    }

    return backupFiles;
}

function generateReport(backupFiles) {
    console.log('\n📊 COMPREHENSIVE BACKUP SCAN RESULTS');
    console.log('=====================================\n');

    console.log(`📁 Total backup files found: ${backupFiles.total}\n`);

    // By directory
    console.log('📂 By Directory:');
    for (const [dir, files] of Object.entries(backupFiles.byDirectory)) {
        console.log(`  ${dir}/: ${files.length} files`);

        // Show sample files
        const sample = files.slice(0, 3);
        for (const file of sample) {
            const patterns = file.patterns || {};
            const indicators = [];
            if (patterns.hasTODOs) indicators.push('TODO');
            if (patterns.hasSvelte5) indicators.push('Svelte5');
            if (patterns.hasMeltUI) indicators.push('MeltUI');
            if (patterns.hasPlaceholders) indicators.push('Placeholder');

            console.log(`    - ${file.name} (${file.lineCount || '?'} lines) [${indicators.join(', ') || 'Basic'}]`);
        }

        if (files.length > 3) {
            console.log(`    ... and ${files.length - 3} more files`);
        }
        console.log('');
    }

    // By type
    console.log('📄 By File Type:');
    for (const [ext, count] of Object.entries(backupFiles.byType)) {
        console.log(`  ${ext}: ${count} files`);
    }
    console.log('');

    // Pattern analysis
    console.log('🔍 Pattern Analysis:');
    console.log(`  📝 Files with TODOs: ${backupFiles.patterns.hasTODOs}`);
    console.log(`  🚧 Files with placeholders: ${backupFiles.patterns.hasPlaceholders}`);
    console.log(`  ⚡ Files with Svelte 5 patterns: ${backupFiles.patterns.hasSvelte5}`);
    console.log(`  🏗️ Files with Melt-UI patterns: ${backupFiles.patterns.hasMeltUI}`);
    console.log(`  🕐 Files with timestamps: ${backupFiles.patterns.hasTimestamp}`);
    console.log(`  🔄 Replaced files: ${backupFiles.patterns.hasReplaced}`);
    console.log('');

    // Recommendations
    console.log('💡 Recommendations:');

    const todoFiles = [];
    const placeholderFiles = [];
    const largeFiles = [];
    const duplicateGroups = {};

    for (const [dir, files] of Object.entries(backupFiles.byDirectory)) {
        for (const file of files) {
            if (file.patterns?.hasTODOs) todoFiles.push(`${dir}/${file.name}`);
            if (file.patterns?.hasPlaceholders && !file.patterns?.hasTODOs) {
                placeholderFiles.push(`${dir}/${file.name}`);
            }
            if (file.patterns?.isLarge) largeFiles.push(`${dir}/${file.name}`);

            // Group potential duplicates
            const baseName = file.name.replace(/\.backup.*$/, '').replace(/\.replaced$/, '');
            if (!duplicateGroups[baseName]) duplicateGroups[baseName] = [];
            duplicateGroups[baseName].push(`${dir}/${file.name}`);
        }
    }

    if (todoFiles.length > 0) {
        console.log(`  ✅ Preserve ${todoFiles.length} files with TODOs`);
        todoFiles.slice(0, 5).forEach(file => console.log(`    - ${file}`));
        if (todoFiles.length > 5) console.log(`    ... and ${todoFiles.length - 5} more`);
    }

    if (placeholderFiles.length > 0) {
        console.log(`  🗑️ Consider removing ${placeholderFiles.length} placeholder files`);
        placeholderFiles.slice(0, 5).forEach(file => console.log(`    - ${file}`));
        if (placeholderFiles.length > 5) console.log(`    ... and ${placeholderFiles.length - 5} more`);
    }

    const duplicates = Object.entries(duplicateGroups).filter(([name, files]) => files.length > 1);
    if (duplicates.length > 0) {
        console.log(`  📦 Found ${duplicates.length} potential duplicate groups`);
        duplicates.slice(0, 3).forEach(([name, files]) => {
            console.log(`    - ${name}: ${files.length} versions`);
        });
    }

    console.log('');
}

async function main() {
    console.log('🔍 Starting comprehensive backup file scan...');
    console.log('Analyzing patterns: TODOs, placeholders, stubs, mocks, Svelte 5, Melt-UI\n');

    const backupFiles = await scanForBackups();
    generateReport(backupFiles);

    // Save detailed report
    const reportPath = path.join(projectRoot, 'COMPREHENSIVE_BACKUP_SCAN_REPORT.json');
    await fs.writeFile(reportPath, JSON.stringify(backupFiles, null, 2));

    console.log(`📋 Detailed report saved to: ${reportPath}`);
    console.log('✅ Comprehensive scan complete!');
}

main().catch(console.error);
