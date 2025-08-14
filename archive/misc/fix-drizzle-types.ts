#!/usr/bin/env node

/**
 * Drizzle ORM Type Assertion Fix Script
 * 
 * This script systematically finds and fixes Drizzle ORM type compatibility issues
 * by adding "as any" type assertions to query chains that cause TypeScript errors.
 * 
 * Patterns fixed:
 * 1. db.select().from().where() chains
 * 2. queryBuilder patterns with .orderBy(), .limit(), .offset()
 * 3. Complex query builder patterns with multiple chaining methods
 */

import * as fs from 'fs';
import * as path from 'path';
import { glob } from 'glob';

interface DrizzlePattern {
  pattern: RegExp;
  replacement: string;
  description: string;
}

// Define patterns that need type assertion fixes
const drizzlePatterns: DrizzlePattern[] = [
  // Pattern 1: .where() chains that need type assertion
  {
    pattern: /(\w+(?:Query|queryBuilder|query)?\s*=\s*\w+\.select\(\)\.from\([^)]+\)(?:\s*\.[\w()]+)*)\s*\.where\(([^)]+)\)(?!\s*as\s+any)/g,
    replacement: '$1.where($2) as any',
    description: 'Fix .where() chain type assertions'
  },
  
  // Pattern 2: .orderBy() chains that need type assertion
  {
    pattern: /(\w+(?:Query|queryBuilder|query)?\s*=\s*[^;]+)\.orderBy\(([^)]+)\)(?!\s*as\s+any)/g,
    replacement: '$1.orderBy($2) as any',
    description: 'Fix .orderBy() chain type assertions'
  },
  
  // Pattern 3: .limit() chains that need type assertion
  {
    pattern: /(\w+(?:Query|queryBuilder|query)?\s*=\s*[^;]+)\.limit\(([^)]+)\)(?!\s*as\s+any)/g,
    replacement: '$1.limit($2) as any',
    description: 'Fix .limit() chain type assertions'
  },
  
  // Pattern 4: .offset() chains that need type assertion
  {
    pattern: /(\w+(?:Query|queryBuilder|query)?\s*=\s*[^;]+)\.offset\(([^)]+)\)(?!\s*as\s+any)/g,
    replacement: '$1.offset($2) as any',
    description: 'Fix .offset() chain type assertions'
  },
  
  // Pattern 5: Complex query builder reassignments
  {
    pattern: /(queryBuilder\s*=\s*queryBuilder\.(?:where|orderBy|limit|offset)\([^)]+\))(?!\s*as\s+any)/g,
    replacement: '$1 as any',
    description: 'Fix queryBuilder reassignment type assertions'
  },
  
  // Pattern 6: caseQuery specific patterns
  {
    pattern: /(caseQuery\s*=\s*caseQuery\.(?:where|orderBy|limit|offset)\([^)]+\))(?!\s*as\s+any)/g,
    replacement: '$1 as any',
    description: 'Fix caseQuery reassignment type assertions'
  },
  
  // Pattern 7: evidenceQuery specific patterns
  {
    pattern: /(evidenceQuery\s*=\s*evidenceQuery\.(?:where|orderBy|limit|offset)\([^)]+\))(?!\s*as\s+any)/g,
    replacement: '$1 as any',
    description: 'Fix evidenceQuery reassignment type assertions'
  },
  
  // Pattern 8: countQuery patterns
  {
    pattern: /(countQuery\s*=\s*countQuery\.where\([^)]+\))(?!\s*as\s+any)/g,
    replacement: '$1 as any',
    description: 'Fix countQuery type assertions'
  },

  // Pattern 9: Direct select chains without variable assignment
  {
    pattern: /(await\s+db\.select\(\)\.from\([^)]+\)(?:\s*\.[\w()]+)*)\s*\.where\(([^)]+)\)(?!\s*as\s+any)/g,
    replacement: '$1.where($2) as any',
    description: 'Fix direct select chain .where() type assertions'
  }
];

// Additional specific fixes for known problematic patterns
const specificFixes: DrizzlePattern[] = [
  // Fix for totalCountResult pattern
  {
    pattern: /(const totalCountResult = await db\s*\.select\([^)]+\)\s*\.from\([^)]+\))(?!\s*as\s+any)/g,
    replacement: '$1 as any',
    description: 'Fix totalCountResult type assertion'
  }
];

// Target file patterns
const targetPatterns = [
  '**/api/**/*.ts',
  '**/routes/**/*.ts',
  '**/lib/**/*.ts'
];

interface FileAnalysis {
  filePath: string;
  originalContent: string;
  modifiedContent: string;
  changesCount: number;
  appliedFixes: string[];
}

class DrizzleTypeFixer {
  private baseDir: string;
  private results: FileAnalysis[] = [];
  private dryRun: boolean;

  constructor(baseDir: string, dryRun: boolean = false) {
    this.baseDir = baseDir;
    this.dryRun = dryRun;
  }

  async findTargetFiles(): Promise<string[]> {
    const allFiles: string[] = [];
    
    for (const pattern of targetPatterns) {
      const files = await glob(pattern, { 
        cwd: this.baseDir,
        absolute: true,
        ignore: ['**/node_modules/**', '**/dist/**', '**/.svelte-kit/**']
      });
      allFiles.push(...files);
    }
    
    // Remove duplicates and filter for .ts files
    return [...new Set(allFiles)].filter(file => file.endsWith('.ts'));
  }

  async analyzeFile(filePath: string): Promise<FileAnalysis | null> {
    try {
      const originalContent = await fs.promises.readFile(filePath, 'utf-8');
      
      // Skip files that don't contain Drizzle patterns
      if (!this.containsDrizzlePattern(originalContent)) {
        return null;
      }

      let modifiedContent = originalContent;
      const appliedFixes: string[] = [];
      let changesCount = 0;

      // Apply all pattern fixes
      for (const fix of [...drizzlePatterns, ...specificFixes]) {
        const beforeCount = (modifiedContent.match(fix.pattern) || []).length;
        modifiedContent = modifiedContent.replace(fix.pattern, fix.replacement);
        const afterCount = (modifiedContent.match(fix.pattern) || []).length;
        
        const changes = beforeCount - afterCount;
        if (changes > 0) {
          appliedFixes.push(`${fix.description} (${changes} instances)`);
          changesCount += changes;
        }
      }

      if (changesCount === 0) {
        return null;
      }

      return {
        filePath,
        originalContent,
        modifiedContent,
        changesCount,
        appliedFixes
      };
    } catch (error) {
      console.error(`Error analyzing file ${filePath}:`, error);
      return null;
    }
  }

  private containsDrizzlePattern(content: string): boolean {
    // Quick check for Drizzle-related patterns
    const drizzleIndicators = [
      '.select().from(',
      'drizzle-orm',
      'queryBuilder',
      'caseQuery',
      'evidenceQuery',
      'countQuery'
    ];
    
    return drizzleIndicators.some(indicator => content.includes(indicator));
  }

  async applyFix(analysis: FileAnalysis): Promise<void> {
    if (this.dryRun) {
      console.log(`[DRY RUN] Would fix ${analysis.filePath}`);
      return;
    }

    try {
      await fs.promises.writeFile(analysis.filePath, analysis.modifiedContent, 'utf-8');
      console.log(`✅ Fixed ${analysis.filePath}`);
    } catch (error) {
      console.error(`❌ Failed to write ${analysis.filePath}:`, error);
    }
  }

  async run(): Promise<void> {
    console.log(`🔍 Finding TypeScript files in ${this.baseDir}...`);
    const targetFiles = await this.findTargetFiles();
    console.log(`📁 Found ${targetFiles.length} TypeScript files to analyze`);

    console.log('\n🔧 Analyzing files for Drizzle type issues...');
    
    for (const filePath of targetFiles) {
      const analysis = await this.analyzeFile(filePath);
      if (analysis) {
        this.results.push(analysis);
        console.log(`📄 ${path.relative(this.baseDir, filePath)}: ${analysis.changesCount} fixes needed`);
      }
    }

    if (this.results.length === 0) {
      console.log('✨ No Drizzle type assertion issues found!');
      return;
    }

    console.log(`\n🎯 Found ${this.results.length} files with Drizzle type issues`);
    
    // Show detailed analysis
    for (const result of this.results) {
      console.log(`\n📄 ${path.relative(this.baseDir, result.filePath)}:`);
      for (const fix of result.appliedFixes) {
        console.log(`   - ${fix}`);
      }
    }

    if (this.dryRun) {
      console.log('\n🚀 To apply fixes, run without --dry-run flag');
      return;
    }

    // Apply fixes
    console.log('\n🔨 Applying fixes...');
    for (const result of this.results) {
      await this.applyFix(result);
    }

    console.log(`\n✅ Successfully fixed ${this.results.length} files!`);
    console.log('\n📊 Summary:');
    console.log(`   - Files processed: ${targetFiles.length}`);
    console.log(`   - Files with issues: ${this.results.length}`);
    console.log(`   - Total fixes applied: ${this.results.reduce((sum, r) => sum + r.changesCount, 0)}`);
  }

  // Generate a detailed report
  generateReport(): string {
    let report = '# Drizzle ORM Type Assertion Fix Report\n\n';
    report += `Generated: ${new Date().toISOString()}\n\n`;
    
    if (this.results.length === 0) {
      report += 'No issues found.\n';
      return report;
    }

    report += `## Summary\n`;
    report += `- Files with issues: ${this.results.length}\n`;
    report += `- Total fixes: ${this.results.reduce((sum, r) => sum + r.changesCount, 0)}\n\n`;

    report += `## Files Fixed\n\n`;
    for (const result of this.results) {
      report += `### ${path.relative(this.baseDir, result.filePath)}\n`;
      report += `Fixes applied: ${result.changesCount}\n\n`;
      for (const fix of result.appliedFixes) {
        report += `- ${fix}\n`;
      }
      report += '\n';
    }

    return report;
  }
}

// CLI Interface
async function main() {
  const args = process.argv.slice(2);
  const dryRun = args.includes('--dry-run');
  const generateReport = args.includes('--report');
  
  // Get the base directory (default to current working directory)
  const baseDir = args.find(arg => !arg.startsWith('--')) || process.cwd();

  console.log('🚀 Drizzle ORM Type Assertion Fix Script');
  console.log(`📁 Base directory: ${baseDir}`);
  if (dryRun) {
    console.log('🔍 Running in DRY RUN mode - no files will be modified');
  }

  const fixer = new DrizzleTypeFixer(baseDir, dryRun);
  
  try {
    await fixer.run();
    
    if (generateReport) {
      const report = fixer.generateReport();
      const reportPath = path.join(baseDir, 'drizzle-fix-report.md');
      await fs.promises.writeFile(reportPath, report);
      console.log(`📄 Report generated: ${reportPath}`);
    }
    
  } catch (error) {
    console.error('❌ Script failed:', error);
    process.exit(1);
  }
}

// Run if called directly
if (require.main === module) {
  main().catch(console.error);
}

export { DrizzleTypeFixer };