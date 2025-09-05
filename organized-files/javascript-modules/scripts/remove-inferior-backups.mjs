#!/usr/bin/env node

/**
 * REMOVE INFERIOR BACKUP FILES
 * Removes melt-ui, legacy, and other inferior backup files since we're using bits-ui v2 + Svelte 5
 */

import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, '..');

// Criteria for identifying inferior backup files
const INFERIOR_CRITERIA = {
    // Inferior UI library patterns (melt-ui is inferior to bits-ui v2)
    inferiorLibraries: [
        /from ['"]@melt-ui\/svelte['"]/,
        /import.*melt-ui/,
        /melt.*ui/i,
        /createDialog/,  // melt-ui pattern
        /createTooltip/, // melt-ui pattern
        /createSelect/,  // melt-ui pattern
    ],

    // Inferior file patterns
    inferiorPatterns: [
        /deprecated/i,
        /obsolete/i,
        /old.*version/i,
        /legacy.*backup/i,
        /inferior/i,
        /placeholder.*stub/i,
        /todo.*replace.*this/i,
    ],

    // Inferior directory indicators
    inferiorDirectories: [
        'inferior-melt-ui',
        'deprecated',
        'old-components',
        'legacy-backups',
        'melt-ui-backups',
    ],

    // Keep these patterns (modern Svelte 5 + bits-ui v2)
    modernPatterns: [
        /from ['"]bits-ui/,
        /\$props\(\)/,
        /\$state\(\)/,
        /\$derived\(\)/,
        /\$effect\(\)/,
        /runes.*true/i,
        /phase.*context/i, // Important roadmaps
        /phase.*integration/i,
    ]
};

async function findAllBackupFiles() {
    const backupFiles = [];

    async function scanDirectory(dir) {
        try {
            const entries = await fs.readdir(dir, { withFileTypes: true });

            for (const entry of entries) {
                const fullPath = path.join(dir, entry.name);

                if (entry.isDirectory()) {
                    if (entry.name === 'node_modules' || entry.name.startsWith('.git')) continue;
                    await scanDirectory(fullPath);
                } else if (entry.isFile() && 
                          (entry.name.includes('.backup') || 
                           entry.name.includes('.replaced'))) {
                    backupFiles.push(fullPath);
                }
            }
        } catch (error) {
            console.warn(`⚠️ Skipping ${dir}: ${error.message}`);
        }
    }

    await scanDirectory(projectRoot);
    return backupFiles;
}

async function analyzeBackupFile(filePath) {
    try {
        const content = await fs.readFile(filePath, 'utf-8');
        const fileName = path.basename(filePath);
        const relativePath = path.relative(projectRoot, filePath);
        const dirPath = path.dirname(relativePath);

        const analysis = {
            path: filePath,
            relativePath,
            fileName,
            dirPath,
            size: content.length,
            isInferior: false,
            isModern: false,
            inferiorReasons: [],
            action: 'keep'
        };

        // Check for inferior directory
        const isInferiorDirectory = INFERIOR_CRITERIA.inferiorDirectories.some(dir => 
            relativePath.toLowerCase().includes(dir.toLowerCase())
        );
        
        if (isInferiorDirectory) {
            analysis.isInferior = true;
            analysis.inferiorReasons.push('Located in inferior directory');
        }

        // Check content for inferior patterns
        const inferiorLibMatches = INFERIOR_CRITERIA.inferiorLibraries.filter(pattern => 
            pattern.test(content)
        );
        if (inferiorLibMatches.length > 0) {
            analysis.isInferior = true;
            analysis.inferiorReasons.push('Contains inferior UI library patterns (melt-ui)');
        }

        const inferiorPatMatches = INFERIOR_CRITERIA.inferiorPatterns.filter(pattern => 
            pattern.test(content)
        );
        if (inferiorPatMatMatches.length > 0) {
            analysis.isInferior = true;
            analysis.inferiorReasons.push('Contains inferior/deprecated patterns');
        }

        // Check for modern patterns (preserve these)
        const modernMatches = INFERIOR_CRITERIA.modernPatterns.filter(pattern => 
            pattern.test(content)
        );
        analysis.isModern = modernMatches.length > 0;

        // Decision logic
        if (analysis.isInferior && !analysis.isModern) {
            analysis.action = 'remove';
        } else if (analysis.isInferior && analysis.isModern) {
            analysis.action = 'keep_important'; // Has modern patterns, keep despite inferior elements
        } else {
            analysis.action = 'keep';
        }

        return analysis;

    } catch (error) {
        console.warn(`❌ Could not analyze ${filePath}: ${error.message}`);
        return null;
    }
}

async function main() {
    console.log('🎯 REMOVING INFERIOR BACKUP FILES');
    console.log('📋 Target: melt-ui, deprecated, and other inferior patterns');
    console.log('✅ Preserving: bits-ui v2, Svelte 5, and modern patterns');
    
    const startTime = Date.now();
    const backupFiles = await findAllBackupFiles();
    console.log(`📁 Found ${backupFiles.length} backup files to analyze`);

    const results = {
        analyzed: 0,
        toRemove: [],
        toKeep: [],
        modernKept: [],
        removed: 0,
        errors: 0,
        totalSizeRemoved: 0
    };

    // Analyze all files
    console.log('\n🔍 Analyzing backup files...');
    for (const filePath of backupFiles) {
        const analysis = await analyzeBackupFile(filePath);
        if (!analysis) {
            results.errors++;
            continue;
        }

        results.analyzed++;

        switch (analysis.action) {
            case 'remove':
                results.toRemove.push(analysis);
                break;
            case 'keep_important':
                results.modernKept.push(analysis);
                break;
            case 'keep':
                results.toKeep.push(analysis);
                break;
        }

        // Progress indicator
        if (results.analyzed % 50 === 0) {
            console.log(`📊 Progress: ${results.analyzed}/${backupFiles.length} analyzed`);
        }
    }

    // Show what will be removed
    console.log(`\n🗑️ Found ${results.toRemove.length} inferior backup files to remove:`);
    
    if (results.toRemove.length > 0) {
        // Group by reason
        const byReason = {};
        results.toRemove.forEach(analysis => {
            analysis.inferiorReasons.forEach(reason => {
                if (!byReason[reason]) byReason[reason] = [];
                byReason[reason].push(analysis);
            });
        });

        Object.entries(byReason).forEach(([reason, analyses]) => {
            console.log(`\n📂 ${reason} (${analyses.length} files):`);
            analyses.slice(0, 3).forEach(analysis => {
                console.log(`   • ${analysis.fileName} (${analysis.size} chars)`);
            });
            if (analyses.length > 3) {
                console.log(`   ... and ${analyses.length - 3} more files`);
            }
        });

        // Remove the files
        console.log(`\n🗑️ Removing ${results.toRemove.length} inferior backup files...`);
        for (const analysis of results.toRemove) {
            try {
                await fs.unlink(analysis.path);
                results.removed++;
                results.totalSizeRemoved += analysis.size;
                console.log(`✅ Removed: ${analysis.fileName}`);
            } catch (error) {
                console.error(`❌ Failed to remove ${analysis.fileName}: ${error.message}`);
                results.errors++;
            }
        }
    }

    const duration = ((Date.now() - startTime) / 1000).toFixed(1);
    const sizeMB = (results.totalSizeRemoved / (1024 * 1024)).toFixed(2);

    console.log('\n📊 CLEANUP COMPLETE!');
    console.log('═════════════════════════════════');
    console.log(`📁 Files analyzed: ${results.analyzed}`);
    console.log(`🗑️ Files removed: ${results.removed}`);
    console.log(`✅ Modern files kept: ${results.modernKept.length}`);
    console.log(`🔒 Other files kept: ${results.toKeep.length}`);
    console.log(`💾 Space freed: ${sizeMB} MB`);
    console.log(`⏱️ Duration: ${duration}s`);
    console.log(`❌ Errors: ${results.errors}`);

    if (results.modernKept.length > 0) {
        console.log(`\n📝 Kept ${results.modernKept.length} files with modern patterns despite inferior elements`);
    }

    // Generate final report
    const reportPath = path.join(projectRoot, 'archives', 'component-backups', 'INFERIOR_CLEANUP_REPORT.md');
    await generateReport(results, reportPath);
    console.log(`\n📋 Detailed report saved: ${reportPath}`);
}

async function generateReport(results, reportPath) {
    let report = `# Inferior Backup Files Cleanup Report\n`;
    report += `Generated: ${new Date().toISOString()}\n\n`;
    
    report += `## Summary\n`;
    report += `- Files analyzed: ${results.analyzed}\n`;
    report += `- Files removed: ${results.removed}\n`;
    report += `- Modern files kept: ${results.modernKept.length}\n`;
    report += `- Other files kept: ${results.toKeep.length}\n`;
    report += `- Space freed: ${(results.totalSizeRemoved / (1024 * 1024)).toFixed(2)} MB\n`;
    report += `- Errors: ${results.errors}\n\n`;

    if (results.toRemove.length > 0) {
        report += `## Files Removed\n\n`;
        results.toRemove.forEach((analysis, index) => {
            report += `${index + 1}. **${analysis.fileName}**\n`;
            report += `   - Path: \`${analysis.relativePath}\`\n`;
            report += `   - Reasons: ${analysis.inferiorReasons.join(', ')}\n`;
            report += `   - Size: ${analysis.size} chars\n\n`;
        });
    }

    if (results.modernKept.length > 0) {
        report += `## Modern Files Kept (Despite Inferior Elements)\n\n`;
        results.modernKept.forEach((analysis, index) => {
            report += `${index + 1}. **${analysis.fileName}**\n`;
            report += `   - Path: \`${analysis.relativePath}\`\n`;
            report += `   - Inferior reasons: ${analysis.inferiorReasons.join(', ')}\n`;
            report += `   - Kept because: Contains modern patterns\n\n`;
        });
    }

    await fs.mkdir(path.dirname(reportPath), { recursive: true });
    await fs.writeFile(reportPath, report);
}

main().catch(console.error);