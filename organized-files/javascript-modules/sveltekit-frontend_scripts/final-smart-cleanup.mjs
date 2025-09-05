#!/usr/bin/env node

/**
 * Final Smart Cleanup - Remove placeholders/stubs while preserving TODOs
 * Based on comprehensive analysis: Main files are superior, many backups are placeholders
 */

import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, '..');

// Comprehensive cleanup patterns based on analysis
const REMOVAL_CRITERIA = {
    definiteRemoval: [
        // Clear placeholder indicators
        /this is a placeholder/i,
        /placeholder.*component/i,
        /todo.*implement.*placeholder/i,
        /coming soon/i,
        /not implemented.*yet/i,
        /replace this with/i,
        /stub.*implementation/i,

        // Mock/fake content
        /mock.*data/i,
        /fake.*content/i,
        /dummy.*implementation/i,
        /example.*placeholder/i
    ],

    preservation: [
        // Important TODO roadmaps to preserve
        /roadmap/i,
        /milestone/i,
        /phase.*\d+/i,
        /v\d+\.\d+.*features/i,
        /feature.*plan/i,
        /implementation.*plan/i
    ],

    modernPatterns: [
        // Keep files with these modern patterns
        /\$props\(\)/,
        /\$bindable\(\)/,
        /\$state\(\)/,
        /\$derived\(\)/,
        /from ['"]bits-ui/,
        /interface \w+/,
        /type \w+/
    ]
};

async function analyzeAndClean(filePath) {
    try {
        const content = await fs.readFile(filePath, 'utf-8');
        const fileName = path.basename(filePath);

        const analysis = {
            path: filePath,
            fileName,
            size: content.length,
            lines: content.split('\n').length,

            // Analysis flags
            isPlaceholder: false,
            isStub: false,
            hasImportantTodos: false,
            hasModernPatterns: false,
            isLegacyBackup: false,

            // Action to take
            action: 'keep', // keep, remove, preserve_todos
            reason: '',
            confidence: 0
        };

        // Check for definite removal criteria
        const removalMatches = REMOVAL_CRITERIA.definiteRemoval.filter(pattern => pattern.test(content));
        analysis.isPlaceholder = removalMatches.length > 0;

        // Check for preservation criteria
        const preservationMatches = REMOVAL_CRITERIA.preservation.filter(pattern => pattern.test(content));
        analysis.hasImportantTodos = preservationMatches.length > 0;

        // Check for modern patterns
        const modernMatches = REMOVAL_CRITERIA.modernPatterns.filter(pattern => pattern.test(content));
        analysis.hasModernPatterns = modernMatches.length > 0;

        // Check if it's a legacy backup
        analysis.isLegacyBackup = fileName.includes('.backup') && content.includes('createDialog');

        // Determine action
        if (analysis.isPlaceholder && !analysis.hasImportantTodos && analysis.size < 5000) {
            analysis.action = 'remove';
            analysis.reason = 'Contains placeholder content without important TODOs';
            analysis.confidence = 0.9;
        } else if (analysis.isLegacyBackup && !analysis.hasImportantTodos) {
            analysis.action = 'remove';
            analysis.reason = 'Legacy backup without important TODOs';
            analysis.confidence = 0.8;
        } else if (analysis.hasImportantTodos && !analysis.hasModernPatterns) {
            analysis.action = 'preserve_todos';
            analysis.reason = 'Extract important TODOs before removal';
            analysis.confidence = 0.85;
        } else if (analysis.hasModernPatterns) {
            analysis.action = 'keep';
            analysis.reason = 'Contains modern patterns - keep in place';
            analysis.confidence = 0.95;
        } else {
            analysis.action = 'keep';
            analysis.reason = 'Neutral content - keep safe';
            analysis.confidence = 0.5;
        }

        return analysis;

    } catch (error) {
        console.warn(`❌ Could not analyze ${filePath}: ${error.message}`);
        return null;
    }
}

async function extractTodos(filePath, content) {
    const todos = [];
    const lines = content.split('\n');

    for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        if (/TODO:|FIXME:|NOTE:|roadmap|milestone|phase/i.test(line)) {
            todos.push({
                line: i + 1,
                content: line.trim(),
                context: lines.slice(Math.max(0, i-1), i+2).join('\n')
            });
        }
    }

    return todos;
}

async function findTargetFiles() {
    const targetFiles = [];

    async function scanDirectory(dir) {
        try {
            const entries = await fs.readdir(dir, { withFileTypes: true });

            for (const entry of entries) {
                const fullPath = path.join(dir, entry.name);

                if (entry.isDirectory() && !entry.name.startsWith('.') && entry.name !== 'node_modules') {
                    await scanDirectory(fullPath);
                } else if (entry.isFile() &&
                          (entry.name.includes('.backup') ||
                           entry.name.includes('.replaced') ||
                           (entry.name.endsWith('.svelte') && entry.name.includes('placeholder')))) {
                    targetFiles.push(fullPath);
                }
            }
        } catch (error) {
            // Skip inaccessible directories
        }
    }

    await scanDirectory(projectRoot);
    return targetFiles;
}

async function main() {
    console.log('🎯 FINAL SMART CLEANUP - Remove placeholders/stubs, preserve TODOs');
    console.log('📋 Based on analysis: Main files superior, backup placeholders removable');

    const targetFiles = await findTargetFiles();
    console.log(`📁 Found ${targetFiles.length} potential cleanup targets`);

    const results = {
        analyzed: 0,
        toRemove: [],
        toPreserveTodos: [],
        toKeep: [],
        todosExtracted: [],
        errors: 0
    };

    // Analyze files in batches
    const batchSize = 25;
    for (let i = 0; i < targetFiles.length; i += batchSize) {
        const batch = targetFiles.slice(i, i + batchSize);
        console.log(`\n🔍 Processing batch ${Math.floor(i/batchSize) + 1}: ${batch.length} files`);

        for (const filePath of batch) {
            const analysis = await analyzeAndClean(filePath);
            if (!analysis) {
                results.errors++;
                continue;
            }

            results.analyzed++;

            console.log(`📊 ${analysis.fileName}`);
            console.log(`   Action: ${analysis.action} (${(analysis.confidence * 100).toFixed(0)}% confidence)`);
            console.log(`   Reason: ${analysis.reason}`);
            console.log(`   Size: ${analysis.size} chars, ${analysis.lines} lines`);

            // Categorize by action
            switch (analysis.action) {
                case 'remove':
                    results.toRemove.push(analysis);
                    console.log(`   🗑️ Marked for removal`);
                    break;
                case 'preserve_todos':
                    results.toPreserveTodos.push(analysis);
                    console.log(`   📝 TODO extraction needed`);
                    break;
                case 'keep':
                    results.toKeep.push(analysis);
                    console.log(`   ✅ Keeping in place`);
                    break;
            }
        }
    }

    // Extract TODOs from files marked for TODO preservation
    console.log('\n📝 Extracting TODOs from preservation targets...');
    for (const analysis of results.toPreserveTodos) {
        try {
            const content = await fs.readFile(analysis.path, 'utf-8');
            const todos = await extractTodos(analysis.path, content);

            if (todos.length > 0) {
                results.todosExtracted.push({
                    fileName: analysis.fileName,
                    path: analysis.path,
                    todos: todos
                });
                console.log(`📝 Extracted ${todos.length} TODOs from ${analysis.fileName}`);
            }
        } catch (error) {
            console.error(`❌ Could not extract TODOs from ${analysis.fileName}`);
        }
    }

    // Generate summary report
    console.log('\n📊 FINAL CLEANUP ANALYSIS COMPLETE!');
    console.log('═══════════════════════════════════════════');
    console.log(`📁 Files analyzed: ${results.analyzed}`);
    console.log(`🗑️ Files to remove: ${results.toRemove.length}`);
    console.log(`📝 TODO extraction targets: ${results.toPreserveTodos.length}`);
    console.log(`✅ Files to keep: ${results.toKeep.length}`);
    console.log(`📋 TODOs extracted: ${results.todosExtracted.length} files`);
    console.log(`❌ Errors: ${results.errors}`);

    // Show removal candidates
    if (results.toRemove.length > 0) {
        console.log('\n🗑️ REMOVAL CANDIDATES:');
        results.toRemove.forEach((analysis, index) => {
            console.log(`${index + 1}. ${analysis.fileName} - ${analysis.reason}`);
        });
    }

    // Save extracted TODOs
    if (results.todosExtracted.length > 0) {
        const todoReportPath = path.join(projectRoot, 'archives', 'component-backups', 'EXTRACTED_TODOS.md');
        const todoReport = generateTodoReport(results.todosExtracted);
        await fs.writeFile(todoReportPath, todoReport);
        console.log(`\n📋 Extracted TODOs saved: ${todoReportPath}`);
    }

    // Generate final cleanup report
    const reportPath = path.join(projectRoot, 'archives', 'component-backups', 'FINAL_CLEANUP_ANALYSIS.md');
    const report = generateCleanupReport(results);
    await fs.writeFile(reportPath, report);

    console.log(`\n📋 Final analysis saved: ${reportPath}`);
    console.log('\n✅ Ready for final cleanup execution!');
    console.log('   Next step: Review removal candidates and execute cleanup');
}

function generateTodoReport(todosExtracted) {
    let report = `# Extracted TODOs and Roadmaps\n`;
    report += `Generated: ${new Date().toISOString()}\n\n`;
    report += `## Important TODOs preserved from backup files\n\n`;

    todosExtracted.forEach(file => {
        report += `### ${file.fileName}\n`;
        report += `Source: \`${file.path}\`\n\n`;

        file.todos.forEach(todo => {
            report += `**Line ${todo.line}:**\n`;
            report += `\`\`\`\n${todo.context}\n\`\`\`\n\n`;
        });

        report += `---\n\n`;
    });

    return report;
}

function generateCleanupReport(results) {
    const timestamp = new Date().toISOString();

    let report = `# Final Smart Cleanup Analysis\n`;
    report += `Generated: ${timestamp}\n\n`;

    report += `## Summary\n`;
    report += `- Files analyzed: ${results.analyzed}\n`;
    report += `- Files to remove: ${results.toRemove.length}\n`;
    report += `- TODO extraction targets: ${results.toPreserveTodos.length}\n`;
    report += `- Files to keep: ${results.toKeep.length}\n`;
    report += `- TODOs extracted: ${results.todosExtracted.length} files\n\n`;

    if (results.toRemove.length > 0) {
        report += `## 🗑️ Files Marked for Removal\n\n`;
        results.toRemove.forEach((analysis, index) => {
            report += `${index + 1}. **${analysis.fileName}**\n`;
            report += `   - Reason: ${analysis.reason}\n`;
            report += `   - Confidence: ${(analysis.confidence * 100).toFixed(0)}%\n`;
            report += `   - Size: ${analysis.size} chars\n\n`;
        });
    }

    return report;
}

main().catch(console.error);
