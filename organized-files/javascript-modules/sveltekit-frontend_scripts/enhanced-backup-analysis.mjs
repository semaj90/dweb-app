#!/usr/bin/env node

/**
 * Enhanced Backup Analysis - Focus on Placeholders, Stubs, Mocks, and TODO preservation
 * Addresses: "when you review also look for 'placeholder, stubs or mocks' when making determination"
 * Ensures: "make sure the to do is done in the superior one"
 */

import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, '..');

// Enhanced analysis patterns
const QUALITY_PATTERNS = {
    // High quality indicators (modern, complete)
    superior: {
        svelte5: [/\$props\(\)/, /\$bindable\(\)/, /\$state\(\)/, /\$derived\(\)/, /let \{ .+ \} = \$props\(\)/],
        bitsUI: [/from ['"]bits-ui/, /<Dialog\.Root/, /<Button\.Root/, /<Accordion\.Root/],
        validation: [/z\./, /schema\./, /validate/i, /\.required\(\)/, /\.min\(/, /\.max\(/],
        implementation: [/export function/, /export class/, /interface \w+/, /type \w+/],
        comprehensive: [/try\s*\{/, /catch\s*\(/, /async\s+function/, /await\s+/]
    },

    // Low quality indicators (placeholders, stubs, mocks)
    inferior: {
        placeholders: [/placeholder/i, /todo.*implement/i, /coming soon/i, /not implemented/i],
        stubs: [/stub/i, /throw new Error.*not implemented/i, /return null.*placeholder/i],
        mocks: [/mock/i, /fake/i, /dummy/i, /test.*data/i],
        incomplete: [/\/\/ TODO.*REMOVE/i, /\/\/ FIXME.*STUB/i, /\/\/ HACK/i],
        legacy: [/createDialog/, /createButton/, /from ['"]@melt-ui/]
    },

    // Content quality indicators
    content: {
        todos: [/TODO:/gi, /FIXME:/gi, /NOTE:/gi, /HACK:/gi],
        roadmaps: [/roadmap/i, /milestone/i, /phase/i, /v\d+\.\d+/],
        documentation: [/\/\*\*[\s\S]*?\*\//, /\/\*[\s\S]*?\*\//, /\/\/.*@/]
    }
};

async function analyzeFileContent(filePath) {
    try {
        const content = await fs.readFile(filePath, 'utf-8');
        const stats = await fs.stat(filePath);

        const analysis = {
            path: filePath,
            fileName: path.basename(filePath),
            size: content.length,
            lines: content.split('\n').length,
            modified: stats.mtime,

            // Quality scores
            superiorScore: 0,
            inferiorScore: 0,

            // Pattern matches
            patterns: {
                svelte5: 0,
                bitsUI: 0,
                validation: 0,
                implementation: 0,
                comprehensive: 0,
                placeholders: 0,
                stubs: 0,
                mocks: 0,
                incomplete: 0,
                legacy: 0,
                todos: [],
                roadmaps: [],
                documentation: 0
            },

            // Content analysis
            hasSubstantialContent: content.length > 500,
            hasTypes: /interface|type/.test(content),
            hasComments: /\/\/|\/\*/.test(content),

            // Overall assessment
            category: 'neutral',
            confidence: 0
        };

        // Count superior patterns
        for (const [category, patterns] of Object.entries(QUALITY_PATTERNS.superior)) {
            const matches = patterns.filter(pattern => pattern.test(content)).length;
            analysis.patterns[category] = matches;
            analysis.superiorScore += matches * 2; // Higher weight for superior patterns
        }

        // Count inferior patterns
        for (const [category, patterns] of Object.entries(QUALITY_PATTERNS.inferior)) {
            const matches = patterns.filter(pattern => pattern.test(content)).length;
            analysis.patterns[category] = matches;
            analysis.inferiorScore += matches * 3; // Higher penalty for inferior patterns
        }

        // Extract TODOs and roadmaps
        analysis.patterns.todos = content.match(/TODO:.*$/gim) || [];
        analysis.patterns.roadmaps = content.match(/roadmap.*$/gim) || [];
        analysis.patterns.documentation = (content.match(/\/\*\*[\s\S]*?\*\//g) || []).length;

        // Determine category and confidence
        const netScore = analysis.superiorScore - analysis.inferiorScore;

        if (analysis.patterns.placeholders > 0 || analysis.patterns.stubs > 0 || analysis.patterns.mocks > 2) {
            analysis.category = 'inferior-stub';
            analysis.confidence = 0.9;
        } else if (netScore > 5 && analysis.hasSubstantialContent) {
            analysis.category = 'superior';
            analysis.confidence = Math.min(0.95, netScore / 10);
        } else if (netScore < -3 || analysis.patterns.legacy > 2) {
            analysis.category = 'inferior-legacy';
            analysis.confidence = 0.8;
        } else if (analysis.patterns.todos.length > 5 && analysis.hasSubstantialContent) {
            analysis.category = 'superior-with-todos';
            analysis.confidence = 0.85;
        } else {
            analysis.category = 'neutral';
            analysis.confidence = 0.5;
        }

        return analysis;

    } catch (error) {
        console.warn(`❌ Could not analyze ${filePath}: ${error.message}`);
        return null;
    }
}

async function findBackupPairs() {
    const allFiles = [];

    // Find all backup files and their corresponding main files
    async function findFilesInDir(dir) {
        try {
            const entries = await fs.readdir(dir, { withFileTypes: true });

            for (const entry of entries) {
                const fullPath = path.join(dir, entry.name);

                if (entry.isDirectory()) {
                    await findFilesInDir(fullPath);
                } else if (entry.name.includes('.backup') || entry.name.includes('.replaced')) {
                    allFiles.push(fullPath);

                    // Also check for corresponding main file
                    const mainFileName = entry.name.replace(/\.(backup|replaced).*$/, '');
                    const mainFilePath = path.join(dir, mainFileName);

                    try {
                        await fs.access(mainFilePath);
                        if (!allFiles.includes(mainFilePath)) {
                            allFiles.push(mainFilePath);
                        }
                    } catch {
                        // Main file doesn't exist
                    }
                }
            }
        } catch (error) {
            // Skip inaccessible directories
        }
    }

    await findFilesInDir(projectRoot);
    return allFiles;
}

async function main() {
    console.log('🔍 Enhanced Backup Analysis - Placeholders, Stubs, Mocks & TODOs');
    console.log('📋 Focus: Identify placeholders/stubs/mocks + preserve TODOs in superior files');

    const allFiles = await findBackupPairs();
    console.log(`📁 Found ${allFiles.length} files to analyze`);

    const analyses = [];
    const results = {
        superiorFiles: [],
        inferiorStubs: [],
        inferiorLegacy: [],
        superiorWithTodos: [],
        todoPreservationNeeded: [],
        processed: 0,
        errors: 0
    };

    // Analyze each file
    for (const filePath of allFiles.slice(0, 50)) { // Process in batches
        console.log(`\n🔍 Analyzing: ${path.basename(filePath)}`);

        const analysis = await analyzeFileContent(filePath);
        if (!analysis) {
            results.errors++;
            continue;
        }

        analyses.push(analysis);
        results.processed++;

        // Log analysis details
        console.log(`   📊 Category: ${analysis.category} (confidence: ${(analysis.confidence * 100).toFixed(0)}%)`);
        console.log(`   📏 Size: ${analysis.size} chars, ${analysis.lines} lines`);
        console.log(`   🎯 Patterns: Svelte5:${analysis.patterns.svelte5} BitsUI:${analysis.patterns.bitsUI} Stubs:${analysis.patterns.stubs} Placeholders:${analysis.patterns.placeholders}`);

        if (analysis.patterns.todos.length > 0) {
            console.log(`   📝 TODOs: ${analysis.patterns.todos.length} found`);
        }

        if (analysis.patterns.roadmaps.length > 0) {
            console.log(`   🗺️ Roadmaps: ${analysis.patterns.roadmaps.length} found`);
        }

        // Categorize results
        switch (analysis.category) {
            case 'superior':
                results.superiorFiles.push(analysis);
                break;
            case 'superior-with-todos':
                results.superiorWithTodos.push(analysis);
                break;
            case 'inferior-stub':
                results.inferiorStubs.push(analysis);
                break;
            case 'inferior-legacy':
                results.inferiorLegacy.push(analysis);
                break;
        }

        // Check if TODOs need to be preserved from backup to main
        if (analysis.fileName.includes('.backup') && analysis.patterns.todos.length > 3) {
            results.todoPreservationNeeded.push(analysis);
        }
    }

    // Generate summary report
    console.log('\n📊 ENHANCED ANALYSIS COMPLETE!');
    console.log('═══════════════════════════════════════════');
    console.log(`📁 Files processed: ${results.processed}`);
    console.log(`✅ Superior files: ${results.superiorFiles.length}`);
    console.log(`📝 Superior with TODOs: ${results.superiorWithTodos.length}`);
    console.log(`🔲 Inferior stubs/placeholders: ${results.inferiorStubs.length}`);
    console.log(`🗑️ Inferior legacy: ${results.inferiorLegacy.length}`);
    console.log(`⚠️ TODO preservation needed: ${results.todoPreservationNeeded.length}`);
    console.log(`❌ Errors: ${results.errors}`);

    // Detailed breakdown
    if (results.inferiorStubs.length > 0) {
        console.log('\n🔲 INFERIOR STUBS/PLACEHOLDERS FOUND:');
        results.inferiorStubs.forEach(file => {
            console.log(`  • ${file.fileName} - ${file.patterns.placeholders} placeholders, ${file.patterns.stubs} stubs`);
        });
    }

    if (results.todoPreservationNeeded.length > 0) {
        console.log('\n⚠️ TODO PRESERVATION NEEDED:');
        results.todoPreservationNeeded.forEach(file => {
            console.log(`  • ${file.fileName} - ${file.patterns.todos.length} TODOs to preserve`);
            file.patterns.todos.slice(0, 3).forEach(todo => {
                console.log(`    - ${todo.trim()}`);
            });
        });
    }

    // Save detailed report
    const reportPath = path.join(projectRoot, 'archives', 'component-backups', 'ENHANCED_ANALYSIS_REPORT.md');
    const report = generateEnhancedReport(results, analyses);
    await fs.writeFile(reportPath, report);
    console.log(`\n📋 Detailed report saved: ${reportPath}`);
}

function generateEnhancedReport(results, analyses) {
    const timestamp = new Date().toISOString();

    let report = `# Enhanced Backup Analysis Report\n`;
    report += `Generated: ${timestamp}\n\n`;
    report += `## Focus Areas\n`;
    report += `- 🔍 Identify placeholders, stubs, and mocks\n`;
    report += `- 📝 Ensure TODOs are preserved in superior versions\n`;
    report += `- 🎯 Modern Svelte 5 patterns vs legacy implementations\n\n`;

    report += `## Summary\n`;
    report += `- Files processed: ${results.processed}\n`;
    report += `- Superior files: ${results.superiorFiles.length}\n`;
    report += `- Superior with TODOs: ${results.superiorWithTodos.length}\n`;
    report += `- Inferior stubs/placeholders: ${results.inferiorStubs.length}\n`;
    report += `- Inferior legacy: ${results.inferiorLegacy.length}\n`;
    report += `- TODO preservation needed: ${results.todoPreservationNeeded.length}\n\n`;

    // Add detailed sections for each category
    if (results.inferiorStubs.length > 0) {
        report += `## 🔲 Inferior Stubs/Placeholders (${results.inferiorStubs.length})\n\n`;
        results.inferiorStubs.forEach(file => {
            report += `### ${file.fileName}\n`;
            report += `- **Size**: ${file.size} chars, ${file.lines} lines\n`;
            report += `- **Issues**: ${file.patterns.placeholders} placeholders, ${file.patterns.stubs} stubs, ${file.patterns.mocks} mocks\n`;
            report += `- **Confidence**: ${(file.confidence * 100).toFixed(0)}%\n\n`;
        });
    }

    if (results.todoPreservationNeeded.length > 0) {
        report += `## ⚠️ TODO Preservation Needed (${results.todoPreservationNeeded.length})\n\n`;
        results.todoPreservationNeeded.forEach(file => {
            report += `### ${file.fileName}\n`;
            report += `- **TODOs found**: ${file.patterns.todos.length}\n`;
            report += `- **Roadmaps**: ${file.patterns.roadmaps.length}\n`;
            file.patterns.todos.slice(0, 5).forEach(todo => {
                report += `  - ${todo.trim()}\n`;
            });
            report += `\n`;
        });
    }

    return report;
}

main().catch(console.error);
