#!/usr/bin/env node

/**
 * Organize Actual Backup Files - Remove inferior melt-ui, keep superior Svelte 5
 * Addresses user request: "compare back up files in sveltekit components and move to backs up,
 * no melt-ui the older inferior one, newer one has todo, supeior svelte 5 validations"
 */

import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, '..');

// Archive directories
const archiveBase = path.join(projectRoot, 'archives', 'component-backups');
const inferiorDir = path.join(archiveBase, 'inferior-melt-ui');
const superiorDir = path.join(archiveBase, 'superior-svelte5');
const duplicateDir = path.join(archiveBase, 'duplicates-removed');

// Create directories
async function ensureDirectories() {
    for (const dir of [archiveBase, inferiorDir, superiorDir, duplicateDir]) {
        await fs.mkdir(dir, { recursive: true });
    }
}

// Analyze file content for patterns
async function analyzeFile(filePath) {
    try {
        const content = await fs.readFile(filePath, 'utf-8');

        const analysis = {
            path: filePath,
            hasMeltUI: false,
            hasSvelte5Runes: false,
            hasBitsUI: false,
            hasValidation: false,
            hasTODO: false,
            isTypeScript: filePath.endsWith('.ts'),
            lineCount: content.split('\n').length
        };

        // Check for melt-ui patterns (inferior)
        const meltUIPatterns = [
            /createDialog/g,
            /createButton/g,
            /createAccordion/g,
            /createTooltip/g,
            /createDropdownMenu/g,
            /from ['"]@melt-ui/g,
            /melt-ui/g,
            /\.createDropdown\(/g,
            /\.createMenu\(/g
        ];

        // Check for Svelte 5 patterns (superior)
        const svelte5Patterns = [
            /\$props\(\)/g,
            /\$bindable\(\)/g,
            /\$state\(\)/g,
            /\$derived\(\)/g,
            /let \{ .+ \} = \$props\(\)/g,
            /export let .+ = \$bindable\(/g
        ];

        // Check for bits-ui patterns (superior)
        const bitsUIPatterns = [
            /from ['"]bits-ui/g,
            /bits-ui/g,
            /<Dialog\.Root/g,
            /<Button\.Root/g,
            /<Accordion\.Root/g
        ];

        // Check validation patterns
        const validationPatterns = [
            /z\./g,  // Zod validation
            /schema\./g,
            /validate/gi,
            /\.required\(\)/g,
            /\.min\(/g,
            /\.max\(/g
        ];

        // Count pattern matches
        analysis.hasMeltUI = meltUIPatterns.some(pattern => pattern.test(content));
        analysis.hasSvelte5Runes = svelte5Patterns.some(pattern => pattern.test(content));
        analysis.hasBitsUI = bitsUIPatterns.some(pattern => pattern.test(content));
        analysis.hasValidation = validationPatterns.some(pattern => pattern.test(content));
        analysis.hasTODO = /TODO|FIXME|NOTE:/gi.test(content);

        // Calculate quality score
        let qualityScore = 0;
        if (analysis.hasSvelte5Runes) qualityScore += 3;  // Highest weight for Svelte 5
        if (analysis.hasBitsUI) qualityScore += 2;
        if (analysis.hasValidation) qualityScore += 2;
        if (analysis.hasTODO) qualityScore += 1;
        if (analysis.hasMeltUI) qualityScore -= 3;  // Penalty for melt-ui

        analysis.qualityScore = qualityScore;

        // Determine category
        if (analysis.hasMeltUI && !analysis.hasSvelte5Runes) {
            analysis.category = 'inferior-melt-ui';
        } else if (analysis.hasSvelte5Runes || analysis.hasBitsUI || analysis.hasValidation) {
            analysis.category = 'superior-svelte5';
        } else {
            analysis.category = 'neutral';
        }

        return analysis;

    } catch (error) {
        console.warn(`❌ Could not analyze ${filePath}: ${error.message}`);
        return null;
    }
}

// Find all backup files
async function findBackupFiles() {
    const backupFiles = [];

    // Check _consolidated-backups directory
    const consolidatedDir = path.join(projectRoot, '_consolidated-backups');
    try {
        const files = await fs.readdir(consolidatedDir);
        for (const file of files) {
            if (file.includes('backup') || file.includes('replaced')) {
                backupFiles.push(path.join(consolidatedDir, file));
            }
        }
    } catch (error) {
        console.log('📁 No _consolidated-backups directory found');
    }

    // Check for scattered .backup files
    async function findBackupsInDir(dir) {
        try {
            const entries = await fs.readdir(dir, { withFileTypes: true });

            for (const entry of entries) {
                const fullPath = path.join(dir, entry.name);

                if (entry.isDirectory()) {
                    await findBackupsInDir(fullPath);
                } else if (entry.name.includes('.backup')) {
                    backupFiles.push(fullPath);
                }
            }
        } catch (error) {
            // Skip inaccessible directories
        }
    }

    await findBackupsInDir(path.join(projectRoot, 'src'));

    return backupFiles;
}

// Main execution
async function main() {
    console.log('🚀 Starting Backup Organization...');
    console.log('📋 User Request: Remove ONLY inferior melt-ui, KEEP superior Svelte 5 files in place');
    console.log('⚠️  Modified: Superior Svelte 5 files will NOT be moved - they stay where they are!');

    await ensureDirectories();

    const backupFiles = await findBackupFiles();
    console.log(`📁 Found ${backupFiles.length} backup files`);

    const results = {
        analyzed: 0,
        inferiorMoved: 0,
        superiorArchived: 0,
        neutralSkipped: 0,
        errors: 0
    };

    const detailedAnalysis = [];

    // Analyze each backup file
    for (const filePath of backupFiles) {
        console.log(`\n📊 Analyzing: ${path.basename(filePath)}`);

        const analysis = await analyzeFile(filePath);
        if (!analysis) {
            results.errors++;
            continue;
        }

        results.analyzed++;
        detailedAnalysis.push(analysis);

        console.log(`   🔍 Pattern Analysis:`);
        console.log(`     Melt-UI: ${analysis.hasMeltUI ? '❌' : '✅'}`);
        console.log(`     Svelte 5 Runes: ${analysis.hasSvelte5Runes ? '✅' : '❌'}`);
        console.log(`     Bits-UI: ${analysis.hasBitsUI ? '✅' : '❌'}`);
        console.log(`     Validation: ${analysis.hasValidation ? '✅' : '❌'}`);
        console.log(`     TODOs: ${analysis.hasTODO ? '✅' : '❌'}`);
        console.log(`     Quality Score: ${analysis.qualityScore}`);
        console.log(`     Category: ${analysis.category}`);

        // Move ONLY inferior melt-ui files (keep superior Svelte 5 files in place)
        const fileName = path.basename(filePath);

        try {
            if (analysis.category === 'inferior-melt-ui') {
                const destPath = path.join(inferiorDir, fileName);
                await fs.rename(filePath, destPath);
                console.log(`   🗑️ Moved inferior melt-ui: ${fileName}`);
                results.inferiorMoved++;

            } else if (analysis.category === 'superior-svelte5') {
                // DO NOT MOVE - Keep superior Svelte 5 files in place since they're needed
                console.log(`   ✅ Kept superior Svelte 5 (not moved): ${fileName}`);
                results.superiorArchived++;

            } else {
                console.log(`   ⚪ Skipped neutral: ${fileName}`);
                results.neutralSkipped++;
            }
        } catch (moveError) {
            console.error(`   ❌ Error moving ${fileName}: ${moveError.message}`);
            results.errors++;
        }
    }

    // Generate report
    const reportPath = path.join(archiveBase, 'BACKUP_ORGANIZATION_REPORT.md');
    const report = generateReport(results, detailedAnalysis);

    await fs.writeFile(reportPath, report);

    console.log('\n✅ BACKUP ORGANIZATION COMPLETE!');
    console.log('📊 Summary:');
    console.log(`  📁 Files analyzed: ${results.analyzed}`);
    console.log(`  🗑️ Inferior melt-ui moved: ${results.inferiorMoved}`);
    console.log(`  ✅ Superior Svelte 5 kept in place: ${results.superiorArchived}`);
    console.log(`  ⚪ Neutral files skipped: ${results.neutralSkipped}`);
    console.log(`  ❌ Errors: ${results.errors}`);
    console.log(`  📋 Report saved to: ${reportPath}`);
}

function generateReport(results, detailedAnalysis) {
    const timestamp = new Date().toISOString();

    let report = `# Backup Organization Report\nGenerated: ${timestamp}\n\n`;
    report += `## User Request\n"don't move the svelte 5 files we need those ones" - Keep superior Svelte 5 files in place, only remove inferior melt-ui\n\n`;

    report += `## Summary\n`;
    report += `- 📁 Files analyzed: ${results.analyzed}\n`;
    report += `- 🗑️ Inferior melt-ui files moved: ${results.inferiorMoved}\n`;
    report += `- ✅ Superior Svelte 5 files kept in place: ${results.superiorArchived}\n`;
    report += `- ⚪ Neutral files skipped: ${results.neutralSkipped}\n`;
    report += `- ❌ Errors: ${results.errors}\n\n`;

    report += `## Directory Structure\n`;
    report += `- 🗑️ Inferior files: \`archives/component-backups/inferior-melt-ui/\`\n`;
    report += `- ✅ Superior files: \`archives/component-backups/superior-svelte5/\`\n`;
    report += `- 🗂️ Duplicates: \`archives/component-backups/duplicates-removed/\`\n\n`;

    report += `## Detailed Analysis\n\n`;

    // Group by category
    const categories = ['inferior-melt-ui', 'superior-svelte5', 'neutral'];

    for (const category of categories) {
        const categoryFiles = detailedAnalysis.filter(a => a.category === category);
        if (categoryFiles.length === 0) continue;

        report += `### ${category.toUpperCase()} (${categoryFiles.length} files)\n\n`;

        for (const analysis of categoryFiles) {
            const fileName = path.basename(analysis.path);
            report += `#### ${fileName}\n`;
            report += `- **Quality Score**: ${analysis.qualityScore}\n`;
            report += `- **Patterns**: Melt-UI: ${analysis.hasMeltUI ? '❌' : '✅'} | Svelte 5: ${analysis.hasSvelte5Runes ? '✅' : '❌'} | Bits-UI: ${analysis.hasBitsUI ? '✅' : '❌'} | Validation: ${analysis.hasValidation ? '✅' : '❌'} | TODOs: ${analysis.hasTODO ? '✅' : '❌'}\n`;
            report += `- **Lines**: ${analysis.lineCount}\n\n`;
        }
    }

    return report;
}

main().catch(console.error);
