#!/usr/bin/env node

/**
 * Batch 2: Continue Enhanced Analysis with TODO Preservation
 * Focus: Next 50 files + specific TODO roadmap preservation
 */

import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, '..');

// Enhanced analysis patterns (refined based on batch 1 results)
const QUALITY_PATTERNS = {
    superior: {
        svelte5: [/\$props\(\)/, /\$bindable\(\)/, /\$state\(\)/, /\$derived\(\)/, /let \{ .+ \} = \$props\(\)/],
        bitsUI: [/from ['"]bits-ui/, /<Dialog\.Root/, /<Button\.Root/, /<Accordion\.Root/],
        validation: [/z\./, /schema\./, /validate/i, /\.required\(\)/, /\.min\(/, /\.max\(/],
        implementation: [/export function/, /export class/, /interface \w+/, /type \w+/],
        comprehensive: [/try\s*\{/, /catch\s*\(/, /async\s+function/, /await\s+/],
        documentation: [/\/\*\*[\s\S]*?\*\//, /\/\/ @param/, /\/\/ @returns/]
    },

    inferior: {
        placeholders: [
            /placeholder/i,
            /todo.*implement/i,
            /coming soon/i,
            /not implemented/i,
            /this is a placeholder/i,
            /replace this/i,
            /example.*placeholder/i
        ],
        stubs: [
            /stub/i,
            /throw new Error.*not implemented/i,
            /return null.*placeholder/i,
            /function.*stub/i,
            /\.\.\..*stub/i
        ],
        mocks: [/mock/i, /fake/i, /dummy/i, /test.*data/i],
        incomplete: [/\/\/ TODO.*REMOVE/i, /\/\/ FIXME.*STUB/i, /\/\/ HACK/i],
        legacy: [/createDialog/, /createButton/, /from ['"]@melt-ui/]
    },

    content: {
        todos: [/TODO:/gi, /FIXME:/gi, /NOTE:/gi, /HACK:/gi],
        roadmaps: [/roadmap/i, /milestone/i, /phase/i, /v\d+\.\d+/, /feature.*plan/i],
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

            superiorScore: 0,
            inferiorScore: 0,

            patterns: {
                svelte5: 0, bitsUI: 0, validation: 0, implementation: 0, comprehensive: 0,
                placeholders: 0, stubs: 0, mocks: 0, incomplete: 0, legacy: 0,
                todos: [], roadmaps: [], documentation: 0
            },

            hasSubstantialContent: content.length > 500,
            hasTypes: /interface|type/.test(content),
            hasComments: /\/\/|\/\*/.test(content),

            category: 'neutral',
            confidence: 0,

            // Enhanced categorization
            isPlaceholder: false,
            isStub: false,
            hasTodoRoadmap: false,
            preservationNeeded: false
        };

        // Count superior patterns
        for (const [category, patterns] of Object.entries(QUALITY_PATTERNS.superior)) {
            const matches = patterns.filter(pattern => pattern.test(content)).length;
            analysis.patterns[category] = matches;
            analysis.superiorScore += matches * 2;
        }

        // Count inferior patterns
        for (const [category, patterns] of Object.entries(QUALITY_PATTERNS.inferior)) {
            const matches = patterns.filter(pattern => pattern.test(content)).length;
            analysis.patterns[category] = matches;
            analysis.inferiorScore += matches * 3;
        }

        // Extract TODOs and roadmaps with context
        const todoMatches = content.match(/TODO:.*$/gim) || [];
        const roadmapMatches = content.match(/(?:roadmap|milestone|phase|feature.*plan).*$/gim) || [];

        analysis.patterns.todos = todoMatches;
        analysis.patterns.roadmaps = roadmapMatches;
        analysis.patterns.documentation = (content.match(/\/\*\*[\s\S]*?\*\//g) || []).length;

        // Enhanced categorization logic
        analysis.isPlaceholder = analysis.patterns.placeholders > 0;
        analysis.isStub = analysis.patterns.stubs > 0 || content.includes('throw new Error');
        analysis.hasTodoRoadmap = todoMatches.length > 3 || roadmapMatches.length > 0;

        const netScore = analysis.superiorScore - analysis.inferiorScore;

        if (analysis.isPlaceholder || analysis.isStub) {
            analysis.category = 'inferior-stub';
            analysis.confidence = 0.95;
        } else if (netScore > 5 && analysis.hasSubstantialContent) {
            if (analysis.hasTodoRoadmap) {
                analysis.category = 'superior-with-todos';
                analysis.preservationNeeded = true;
            } else {
                analysis.category = 'superior';
            }
            analysis.confidence = Math.min(0.95, netScore / 10);
        } else if (netScore < -3 || analysis.patterns.legacy > 2) {
            analysis.category = 'inferior-legacy';
            analysis.confidence = 0.8;
        } else if (analysis.hasTodoRoadmap && analysis.hasSubstantialContent) {
            analysis.category = 'neutral-with-todos';
            analysis.preservationNeeded = true;
            analysis.confidence = 0.7;
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

async function findRemainingBackupFiles() {
    const allFiles = [];

    async function findFilesInDir(dir) {
        try {
            const entries = await fs.readdir(dir, { withFileTypes: true });

            for (const entry of entries) {
                const fullPath = path.join(dir, entry.name);

                if (entry.isDirectory()) {
                    await findFilesInDir(fullPath);
                } else if (entry.name.includes('.backup') || entry.name.includes('.replaced') || entry.name.endsWith('.svelte')) {
                    allFiles.push(fullPath);
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
    console.log('🔄 BATCH 2: Enhanced Backup Analysis - Continue with TODO Preservation');
    console.log('📋 Focus: Files 51-100 + comprehensive TODO roadmap preservation');

    const allFiles = await findRemainingBackupFiles();
    console.log(`📁 Found ${allFiles.length} total files`);

    // Process files 51-100 (batch 2)
    const batch2Files = allFiles.slice(50, 100);
    console.log(`🎯 Processing batch 2: ${batch2Files.length} files`);

    const results = {
        superiorFiles: [],
        inferiorStubs: [],
        inferiorLegacy: [],
        superiorWithTodos: [],
        neutralWithTodos: [],
        todoPreservationNeeded: [],
        placeholderFiles: [],
        processed: 0,
        errors: 0
    };

    for (const filePath of batch2Files) {
        console.log(`\n🔍 Analyzing: ${path.basename(filePath)}`);

        const analysis = await analyzeFileContent(filePath);
        if (!analysis) {
            results.errors++;
            continue;
        }

        results.processed++;

        // Enhanced logging
        console.log(`   📊 Category: ${analysis.category} (confidence: ${(analysis.confidence * 100).toFixed(0)}%)`);
        console.log(`   📏 Size: ${analysis.size} chars, ${analysis.lines} lines`);
        console.log(`   🎯 Quality: Superior:${analysis.superiorScore} Inferior:${analysis.inferiorScore}`);
        console.log(`   🔍 Flags: Placeholder:${analysis.isPlaceholder} Stub:${analysis.isStub} TodoRoadmap:${analysis.hasTodoRoadmap}`);

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
            case 'neutral-with-todos':
                results.neutralWithTodos.push(analysis);
                break;
            case 'inferior-stub':
                results.inferiorStubs.push(analysis);
                if (analysis.isPlaceholder) results.placeholderFiles.push(analysis);
                break;
            case 'inferior-legacy':
                results.inferiorLegacy.push(analysis);
                break;
        }

        // Check preservation needs
        if (analysis.preservationNeeded) {
            results.todoPreservationNeeded.push(analysis);
        }
    }

    // Generate batch 2 summary
    console.log('\n📊 BATCH 2 ANALYSIS COMPLETE!');
    console.log('═══════════════════════════════════════════');
    console.log(`📁 Files processed: ${results.processed}`);
    console.log(`✅ Superior files: ${results.superiorFiles.length}`);
    console.log(`📝 Superior with TODOs: ${results.superiorWithTodos.length}`);
    console.log(`📋 Neutral with TODOs: ${results.neutralWithTodos.length}`);
    console.log(`🔲 Inferior stubs/placeholders: ${results.inferiorStubs.length}`);
    console.log(`🗑️ Inferior legacy: ${results.inferiorLegacy.length}`);
    console.log(`⚠️ TODO preservation needed: ${results.todoPreservationNeeded.length}`);
    console.log(`🔍 Pure placeholders: ${results.placeholderFiles.length}`);

    // Action recommendations
    console.log('\n🎯 RECOMMENDED ACTIONS:');

    if (results.inferiorStubs.length > 0) {
        console.log(`🗑️ REMOVE ${results.inferiorStubs.length} inferior stubs/placeholders`);
        results.inferiorStubs.slice(0, 5).forEach(file => {
            console.log(`  • ${file.fileName} - ${file.isPlaceholder ? 'PLACEHOLDER' : 'STUB'}`);
        });
    }

    if (results.todoPreservationNeeded.length > 0) {
        console.log(`⚠️ PRESERVE TODOs from ${results.todoPreservationNeeded.length} files`);
        results.todoPreservationNeeded.forEach(file => {
            console.log(`  • ${file.fileName} - ${file.patterns.todos.length} TODOs, ${file.patterns.roadmaps.length} roadmaps`);
            if (file.patterns.roadmaps.length > 0) {
                file.patterns.roadmaps.slice(0, 2).forEach(roadmap => {
                    console.log(`    📍 ${roadmap.trim()}`);
                });
            }
        });
    }

    // Save batch 2 report
    const reportPath = path.join(projectRoot, 'archives', 'component-backups', 'BATCH_2_ANALYSIS_REPORT.md');
    const report = generateBatch2Report(results);
    await fs.writeFile(reportPath, report);
    console.log(`\n📋 Batch 2 report saved: ${reportPath}`);

    // Continue signal
    console.log('\n🔄 READY FOR BATCH 3? Pattern is clear:');
    console.log('  ✅ Main files: Superior with modern patterns');
    console.log('  🗑️ Backup files: Often placeholders/stubs to remove');
    console.log('  📝 Preserve: TODO roadmaps in superior versions');
}

function generateBatch2Report(results) {
    const timestamp = new Date().toISOString();

    let report = `# Batch 2: Enhanced Analysis Report\n`;
    report += `Generated: ${timestamp}\n\n`;

    report += `## Summary - Files 51-100\n`;
    report += `- Files processed: ${results.processed}\n`;
    report += `- Superior files: ${results.superiorFiles.length}\n`;
    report += `- Superior with TODOs: ${results.superiorWithTodos.length}\n`;
    report += `- Neutral with TODOs: ${results.neutralWithTodos.length}\n`;
    report += `- Inferior stubs/placeholders: ${results.inferiorStubs.length}\n`;
    report += `- Inferior legacy: ${results.inferiorLegacy.length}\n`;
    report += `- TODO preservation needed: ${results.todoPreservationNeeded.length}\n\n`;

    if (results.todoPreservationNeeded.length > 0) {
        report += `## ⚠️ TODO Preservation Required\n\n`;
        results.todoPreservationNeeded.forEach(file => {
            report += `### ${file.fileName}\n`;
            report += `- **TODOs**: ${file.patterns.todos.length}\n`;
            report += `- **Roadmaps**: ${file.patterns.roadmaps.length}\n`;
            report += `- **Size**: ${file.size} chars\n`;
            if (file.patterns.roadmaps.length > 0) {
                report += `- **Roadmap Items**:\n`;
                file.patterns.roadmaps.forEach(roadmap => {
                    report += `  - ${roadmap.trim()}\n`;
                });
            }
            report += `\n`;
        });
    }

    return report;
}

main().catch(console.error);
