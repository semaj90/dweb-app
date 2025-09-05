#!/usr/bin/env node

/**
 * COMPREHENSIVE BACKUP CLEANUP - All 383 backup files across deeds-web-app
 * Handles ALL backup directories: sveltekit-frontend, archived-backups, _consolidated-backups, etc.
 */

import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, '..');

// Enhanced removal criteria for comprehensive cleanup
const REMOVAL_CRITERIA = {
    definiteRemoval: [
        /this is a placeholder/i,
        /placeholder.*component/i,
        /todo.*implement.*placeholder/i,
        /coming soon/i,
        /not implemented.*yet/i,
        /replace this with/i,
        /stub.*implementation/i,
        /mock.*data/i,
        /fake.*content/i,
        /dummy.*implementation/i,
        /example.*placeholder/i,
        /deprecated/i,
        /legacy.*backup/i
    ],

    preservation: [
        // Important TODO roadmaps to preserve
        /phase.*\d+.*context/i,
        /phase.*integration/i,
        /roadmap/i,
        /milestone/i,
        /phase.*\d+/i,
        /v\d+\.\d+.*features/i,
        /feature.*plan/i,
        /implementation.*plan/i,
        /context7.*evidence.*actions/i,
        /merge.*reason/i,
        /conflicts:/i
    ],

    modernPatterns: [
        // Keep files with these modern patterns
        /\$props\(\)/,
        /\$bindable\(\)/,
        /\$state\(\)/,
        /\$derived\(\)/,
        /\$effect\(\)/,
        /from ['"]bits-ui/,
        /from ['"]@melt-ui\/svelte/,
        /interface \w+/,
        /type \w+.*=/,
        /export.*interface/,
        /runes.*true/i
    ]
};

async function findAllBackupFiles() {
    const backupFiles = [];

    async function scanDirectory(dir, skipNodeModules = true) {
        try {
            const entries = await fs.readdir(dir, { withFileTypes: true });

            for (const entry of entries) {
                const fullPath = path.join(dir, entry.name);

                if (entry.isDirectory()) {
                    if (skipNodeModules && entry.name === 'node_modules') continue;
                    if (entry.name.startsWith('.git')) continue;
                    await scanDirectory(fullPath, skipNodeModules);
                } else if (entry.isFile() && 
                          (entry.name.includes('.backup') || 
                           entry.name.includes('.replaced'))) {
                    backupFiles.push(fullPath);
                }
            }
        } catch (error) {
            // Skip inaccessible directories
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

        const analysis = {
            path: filePath,
            relativePath,
            fileName,
            size: content.length,
            lines: content.split('\n').length,

            // Analysis flags
            isPlaceholder: false,
            isStub: false,
            hasImportantTodos: false,
            hasModernPatterns: false,
            isLegacyBackup: false,
            hasPhaseContext: false,

            // Action to take
            action: 'keep', // keep, remove, preserve_todos
            reason: '',
            confidence: 0,
            directoryCategory: getDirectoryCategory(relativePath)
        };

        // Check for definite removal criteria
        const removalMatches = REMOVAL_CRITERIA.definiteRemoval.filter(pattern => pattern.test(content));
        analysis.isPlaceholder = removalMatches.length > 0;

        // Check for preservation criteria  
        const preservationMatches = REMOVAL_CRITERIA.preservation.filter(pattern => pattern.test(content));
        analysis.hasImportantTodos = preservationMatches.length > 0;
        analysis.hasPhaseContext = /phase.*\d+.*context/i.test(content);

        // Check for modern patterns
        const modernMatches = REMOVAL_CRITERIA.modernPatterns.filter(pattern => pattern.test(content));
        analysis.hasModernPatterns = modernMatches.length > 0;

        // Legacy backup detection (more sophisticated)
        analysis.isLegacyBackup = (
            fileName.includes('.backup') && 
            !analysis.hasImportantTodos &&
            !analysis.hasModernPatterns &&
            (content.includes('createDialog') || 
             content.includes('melt-ui') || 
             analysis.size < 10000)
        );

        // Determine action with enhanced logic
        if (analysis.isPlaceholder && !analysis.hasImportantTodos && analysis.size < 5000) {
            analysis.action = 'remove';
            analysis.reason = 'Contains placeholder content without important TODOs';
            analysis.confidence = 0.9;
        } else if (analysis.hasPhaseContext || analysis.hasImportantTodos) {
            analysis.action = 'preserve_todos';
            analysis.reason = 'Contains important phase context or roadmap TODOs';
            analysis.confidence = 0.95;
        } else if (analysis.hasModernPatterns) {
            analysis.action = 'keep';
            analysis.reason = 'Contains modern Svelte 5 patterns';
            analysis.confidence = 0.95;
        } else if (analysis.isLegacyBackup && analysis.directoryCategory === 'archived') {
            analysis.action = 'remove';
            analysis.reason = 'Legacy backup in archived directory without modern patterns';
            analysis.confidence = 0.85;
        } else if (analysis.size < 1000 && !analysis.hasImportantTodos) {
            analysis.action = 'remove';
            analysis.reason = 'Very small file without important content';
            analysis.confidence = 0.8;
        } else {
            analysis.action = 'keep';
            analysis.reason = 'Contains substantial content - keeping safe';
            analysis.confidence = 0.6;
        }

        return analysis;

    } catch (error) {
        console.warn(`❌ Could not analyze ${filePath}: ${error.message}`);
        return null;
    }
}

function getDirectoryCategory(relativePath) {
    if (relativePath.includes('archived-backups')) return 'archived';
    if (relativePath.includes('_consolidated-backups')) return 'consolidated';
    if (relativePath.includes('archives/component-backups')) return 'component-archives';
    if (relativePath.includes('phase2-backups')) return 'phase-backups';
    if (relativePath.includes('src/lib')) return 'active-lib';
    if (relativePath.includes('src/routes')) return 'active-routes';
    return 'other';
}

async function extractTodos(filePath, content) {
    const todos = [];
    const lines = content.split('\n');

    for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        if (/TODO:|FIXME:|NOTE:|PHASE.*CONTEXT|PHASE.*INTEGRATION|roadmap|milestone|phase.*\d+|CONFLICTS:|MERGE.*REASON/i.test(line)) {
            todos.push({
                line: i + 1,
                content: line.trim(),
                context: lines.slice(Math.max(0, i-2), i+3).join('\n')
            });
        }
    }

    return todos;
}

async function main() {
    console.log('🎯 COMPREHENSIVE BACKUP CLEANUP - All 383 files across deeds-web-app');
    console.log('📋 Scanning sveltekit-frontend, archived-backups, _consolidated-backups, and all subdirectories');
    
    const startTime = Date.now();
    const backupFiles = await findAllBackupFiles();
    console.log(`📁 Found ${backupFiles.length} backup files across entire project`);

    const results = {
        analyzed: 0,
        toRemove: [],
        toPreserveTodos: [],
        toKeep: [],
        todosExtracted: [],
        categoryBreakdown: {},
        errors: 0
    };

    // Process files in batches for better performance
    const batchSize = 50;
    for (let i = 0; i < backupFiles.length; i += batchSize) {
        const batch = backupFiles.slice(i, i + batchSize);
        console.log(`\n🔍 Processing batch ${Math.floor(i/batchSize) + 1}/${Math.ceil(backupFiles.length/batchSize)}: ${batch.length} files`);

        for (const filePath of batch) {
            const analysis = await analyzeBackupFile(filePath);
            if (!analysis) {
                results.errors++;
                continue;
            }

            results.analyzed++;

            // Track by directory category
            const category = analysis.directoryCategory;
            if (!results.categoryBreakdown[category]) {
                results.categoryBreakdown[category] = { total: 0, remove: 0, keep: 0, preserve: 0 };
            }
            results.categoryBreakdown[category].total++;

            // Show progress every 25 files
            if (results.analyzed % 25 === 0) {
                console.log(`📊 Progress: ${results.analyzed}/${backupFiles.length} files analyzed`);
            }

            // Categorize by action
            switch (analysis.action) {
                case 'remove':
                    results.toRemove.push(analysis);
                    results.categoryBreakdown[category].remove++;
                    break;
                case 'preserve_todos':
                    results.toPreserveTodos.push(analysis);
                    results.categoryBreakdown[category].preserve++;
                    break;
                case 'keep':
                    results.toKeep.push(analysis);
                    results.categoryBreakdown[category].keep++;
                    break;
            }
        }
    }

    // Extract TODOs from preservation targets
    console.log('\n📝 Extracting TODOs from preservation targets...');
    for (const analysis of results.toPreserveTodos) {
        try {
            const content = await fs.readFile(analysis.path, 'utf-8');
            const todos = await extractTodos(analysis.path, content);

            if (todos.length > 0) {
                results.todosExtracted.push({
                    fileName: analysis.fileName,
                    relativePath: analysis.relativePath,
                    path: analysis.path,
                    todos: todos
                });
                console.log(`📝 Extracted ${todos.length} TODOs from ${analysis.fileName}`);
            }
        } catch (error) {
            console.error(`❌ Could not extract TODOs from ${analysis.fileName}`);
        }
    }

    const duration = ((Date.now() - startTime) / 1000).toFixed(1);

    // Generate comprehensive report
    console.log('\n📊 COMPREHENSIVE CLEANUP ANALYSIS COMPLETE!');
    console.log('═══════════════════════════════════════════════');
    console.log(`📁 Files analyzed: ${results.analyzed} (in ${duration}s)`);
    console.log(`🗑️ Files to remove: ${results.toRemove.length}`);
    console.log(`📝 TODO preservation targets: ${results.toPreserveTodos.length}`);
    console.log(`✅ Files to keep: ${results.toKeep.length}`);
    console.log(`📋 TODOs extracted: ${results.todosExtracted.length} files`);
    console.log(`❌ Errors: ${results.errors}`);

    // Category breakdown
    console.log('\n📂 BREAKDOWN BY DIRECTORY:');
    Object.entries(results.categoryBreakdown).forEach(([category, stats]) => {
        console.log(`  ${category}: ${stats.total} files (${stats.remove} remove, ${stats.keep} keep, ${stats.preserve} preserve)`);
    });

    // Show removal candidates by category
    if (results.toRemove.length > 0) {
        console.log('\n🗑️ REMOVAL CANDIDATES BY CATEGORY:');
        const removalsByCategory = {};
        results.toRemove.forEach(analysis => {
            const cat = analysis.directoryCategory;
            if (!removalsByCategory[cat]) removalsByCategory[cat] = [];
            removalsByCategory[cat].push(analysis);
        });

        Object.entries(removalsByCategory).forEach(([category, analyses]) => {
            console.log(`\n  📂 ${category.toUpperCase()} (${analyses.length} files):`);
            analyses.slice(0, 5).forEach((analysis, index) => {
                console.log(`    ${index + 1}. ${analysis.fileName} - ${analysis.reason} (${(analysis.confidence * 100).toFixed(0)}%)`);
            });
            if (analyses.length > 5) {
                console.log(`    ... and ${analyses.length - 5} more files`);
            }
        });
    }

    // Save comprehensive reports
    await generateReports(results);

    console.log('\n✅ Comprehensive analysis complete!');
    console.log(`📋 Reports saved in archives/component-backups/`);
    console.log('   Next: Review reports and execute selective cleanup');
}

async function generateReports(results) {
    const timestamp = new Date().toISOString();
    const reportsDir = path.join(projectRoot, 'archives', 'component-backups');
    
    // Ensure reports directory exists
    await fs.mkdir(reportsDir, { recursive: true });

    // Comprehensive cleanup report
    const reportPath = path.join(reportsDir, 'COMPREHENSIVE_CLEANUP_ANALYSIS.md');
    let report = `# Comprehensive Backup Cleanup Analysis\n`;
    report += `Generated: ${timestamp}\n`;
    report += `Scope: All ${results.analyzed} backup files across deeds-web-app\n\n`;

    report += `## Executive Summary\n`;
    report += `- Files analyzed: ${results.analyzed}\n`;
    report += `- Files to remove: ${results.toRemove.length}\n`;
    report += `- TODO preservation targets: ${results.toPreserveTodos.length}\n`;
    report += `- Files to keep: ${results.toKeep.length}\n`;
    report += `- TODOs extracted: ${results.todosExtracted.length} files\n\n`;

    report += `## Directory Breakdown\n`;
    Object.entries(results.categoryBreakdown).forEach(([category, stats]) => {
        report += `- **${category}**: ${stats.total} files (${stats.remove} remove, ${stats.keep} keep, ${stats.preserve} preserve)\n`;
    });

    report += `\n## Detailed Removal Candidates\n\n`;
    const removalsByCategory = {};
    results.toRemove.forEach(analysis => {
        const cat = analysis.directoryCategory;
        if (!removalsByCategory[cat]) removalsByCategory[cat] = [];
        removalsByCategory[cat].push(analysis);
    });

    Object.entries(removalsByCategory).forEach(([category, analyses]) => {
        report += `### ${category.toUpperCase()} Directory (${analyses.length} files)\n\n`;
        analyses.forEach((analysis, index) => {
            report += `${index + 1}. **${analysis.fileName}**\n`;
            report += `   - Path: \`${analysis.relativePath}\`\n`;
            report += `   - Reason: ${analysis.reason}\n`;
            report += `   - Confidence: ${(analysis.confidence * 100).toFixed(0)}%\n`;
            report += `   - Size: ${analysis.size} chars\n\n`;
        });
    });

    await fs.writeFile(reportPath, report);

    // TODOs report
    if (results.todosExtracted.length > 0) {
        const todoReportPath = path.join(reportsDir, 'COMPREHENSIVE_EXTRACTED_TODOS.md');
        let todoReport = `# Comprehensive Extracted TODOs and Roadmaps\n`;
        todoReport += `Generated: ${timestamp}\n\n`;
        todoReport += `## Important TODOs preserved from ${results.todosExtracted.length} backup files\n\n`;

        results.todosExtracted.forEach(file => {
            todoReport += `### ${file.fileName}\n`;
            todoReport += `Source: \`${file.relativePath}\`\n\n`;

            file.todos.forEach(todo => {
                todoReport += `**Line ${todo.line}:**\n`;
                todoReport += `\`\`\`\n${todo.context}\n\`\`\`\n\n`;
            });

            todoReport += `---\n\n`;
        });

        await fs.writeFile(todoReportPath, todoReport);
        console.log(`\n📋 Comprehensive TODOs report saved: ${todoReportPath}`);
    }

    console.log(`\n📋 Comprehensive cleanup report saved: ${reportPath}`);
}

main().catch(console.error);