#!/usr/bin/env node

/**
 * Repository Organization Script
 * 
 * This script organizes the cluttered root directory by moving files into
 * appropriate subdirectories to resolve the "directory truncated to 1,000 files" issue.
 */

import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Configuration for file organization
const ORGANIZATION_RULES = {
  // Scripts and automation files
  'scripts/automation': {
    patterns: [/^(FIX|START|STOP|RUN|SETUP|INSTALL|BUILD|DEPLOY|TEST|CHECK|VALIDATE)-.*\.(bat|ps1|mjs|js)$/i],
    description: 'Automation and setup scripts'
  },
  'scripts/batch': {
    patterns: [/^.*\.bat$/i],
    exclude: [/^(package|README|CHANGELOG)/i],
    description: 'Windows batch files'
  },
  'scripts/powershell': {
    patterns: [/^.*\.ps1$/i],
    exclude: [/^(package|README|CHANGELOG)/i],
    description: 'PowerShell scripts'
  },
  'scripts/nodejs': {
    patterns: [/^.*\.(mjs|js)$/i],
    exclude: [
      /^(package|README|CHANGELOG)/i,
      /^(vite\.config|svelte\.config|playwright\.config|uno\.config|drizzle\.config)/i,
      /^ecosystem\.(config|dev\.config|prod\.config)/i
    ],
    description: 'Node.js scripts'
  },
  
  // Documentation files
  'docs/guides': {
    patterns: [/^.*-GUIDE\.md$/i, /^.*-README\.md$/i, /^GUIDE-.*\.md$/i],
    description: 'User guides and documentation'
  },
  'docs/status': {
    patterns: [/^.*-STATUS\.md$/i, /^STATUS-.*\.md$/i, /^.*_STATUS.*\.md$/i],
    description: 'Status and progress documentation'
  },
  'docs/summaries': {
    patterns: [/^.*-SUMMARY\.md$/i, /^SUMMARY-.*\.md$/i, /^.*_SUMMARY.*\.md$/i],
    description: 'Summary and report documentation'
  },
  'docs/todo': {
    patterns: [/^TODO.*\.md$/i, /^.*-TODO\.md$/i, /^.*_TODO.*\.md$/i],
    description: 'TODO lists and planning documents'
  },
  
  // Log and temporary files
  'logs/errors': {
    patterns: [/^.*error.*\.(txt|log|md)$/i, /^ERROR-.*\.(txt|log|md)$/i],
    description: 'Error logs and reports'
  },
  'logs/phase': {
    patterns: [/^phase.*\.(txt|log|md)$/i, /^.*phase.*\.(txt|log|md)$/i],
    description: 'Phase-related logs'
  },
  'logs/general': {
    patterns: [/^.*\.(txt|log)$/i],
    exclude: [/^(README|CHANGELOG|LICENSE)/i],
    description: 'General log and text files'
  },
  
  // Configuration files  
  'config/models': {
    patterns: [/^Modelfile/i],
    description: 'Model configuration files'
  },
  'config/environments': {
    patterns: [/^\.env/i],
    description: 'Environment configuration files'
  },
  
  // Archives and deprecated items
  'archive/deprecated': {
    patterns: [/^.*deprecated.*$/i, /^.*old.*$/i, /^.*backup.*$/i],
    description: 'Deprecated and backup files'
  }
};

// Files to keep in root (critical project files)
const KEEP_IN_ROOT = [
  'package.json',
  'package-lock.json', 
  'README.md',
  'CHANGELOG.md',
  'LICENSE',
  'LICENSE.md',
  '.gitignore',
  '.gitattributes',
  'docker-compose.yml',
  'docker-compose.override.yml',
  'Dockerfile',
  'vite.config.js',
  'svelte.config.js',
  'playwright.config.ts',
  'tsconfig.json',
  'uno.config.ts',
  'unocss.config.ts',
  'drizzle.config.ts',
  'ecosystem.config.js',
  'codegen.yml',
  'graphql.config.js',
  'common.config.ts',
  'pg.config.ts'
];

class RepositoryOrganizer {
  constructor() {
    this.dryRun = process.argv.includes('--dry-run');
    this.verbose = process.argv.includes('--verbose');
    this.stats = {
      totalFiles: 0,
      movedFiles: 0,
      skippedFiles: 0,
      createdDirs: 0
    };
  }

  async run() {
    console.log('🗂️  Repository Organization Tool');
    console.log(`📍 Working directory: ${process.cwd()}`);
    console.log(`🔧 Mode: ${this.dryRun ? 'DRY RUN' : 'LIVE'}`);
    console.log('');

    try {
      // Get all files in root directory
      const rootItems = await fs.readdir('.', { withFileTypes: true });
      const files = rootItems
        .filter(item => item.isFile())
        .map(item => item.name);

      this.stats.totalFiles = files.length;
      console.log(`📊 Found ${files.length} files in root directory`);
      
      // Create organization plan
      const organizationPlan = this.createOrganizationPlan(files);
      
      // Show plan summary
      this.showOrganizationPlan(organizationPlan);
      
      if (!this.dryRun) {
        console.log('\n🚀 Executing organization plan...\n');
        await this.executeOrganizationPlan(organizationPlan);
      }
      
      this.showStats();
      
    } catch (error) {
      console.error('❌ Error during organization:', error.message);
      process.exit(1);
    }
  }

  createOrganizationPlan(files) {
    const plan = new Map();
    
    for (const file of files) {
      // Skip files that should stay in root
      if (KEEP_IN_ROOT.includes(file)) {
        this.stats.skippedFiles++;
        if (this.verbose) {
          console.log(`⏭️  Keeping in root: ${file}`);
        }
        continue;
      }

      // Find matching organization rule
      let targetDir = null;
      let ruleDescription = '';

      for (const [dir, rule] of Object.entries(ORGANIZATION_RULES)) {
        const matches = rule.patterns.some(pattern => pattern.test(file));
        const excluded = rule.exclude?.some(pattern => pattern.test(file)) || false;
        
        if (matches && !excluded) {
          targetDir = dir;
          ruleDescription = rule.description;
          break;
        }
      }

      if (targetDir) {
        if (!plan.has(targetDir)) {
          plan.set(targetDir, { files: [], description: ruleDescription });
        }
        plan.get(targetDir).files.push(file);
      } else {
        // Fallback: move to archive/misc for unmatched files
        const fallbackDir = 'archive/misc';
        if (!plan.has(fallbackDir)) {
          plan.set(fallbackDir, { files: [], description: 'Miscellaneous files' });
        }
        plan.get(fallbackDir).files.push(file);
      }
    }

    return plan;
  }

  showOrganizationPlan(plan) {
    console.log('\n📋 Organization Plan:');
    console.log('='.repeat(60));
    
    for (const [targetDir, { files, description }] of plan) {
      console.log(`\n📁 ${targetDir}/ (${files.length} files)`);
      console.log(`   ${description}`);
      
      if (this.verbose && files.length <= 10) {
        files.forEach(file => console.log(`   - ${file}`));
      } else if (files.length > 10) {
        console.log(`   - ${files.slice(0, 3).join(', ')}, ... and ${files.length - 3} more`);
      }
    }
    
    console.log('\n' + '='.repeat(60));
  }

  async executeOrganizationPlan(plan) {
    for (const [targetDir, { files }] of plan) {
      // Create target directory
      await this.ensureDirectory(targetDir);
      
      // Move files
      for (const file of files) {
        await this.moveFile(file, path.join(targetDir, file));
      }
    }
  }

  async ensureDirectory(dirPath) {
    try {
      await fs.access(dirPath);
    } catch {
      if (this.verbose) {
        console.log(`📁 Creating directory: ${dirPath}`);
      }
      await fs.mkdir(dirPath, { recursive: true });
      this.stats.createdDirs++;
    }
  }

  async moveFile(sourceFile, targetPath) {
    try {
      if (this.verbose) {
        console.log(`📄 Moving: ${sourceFile} → ${targetPath}`);
      }
      await fs.rename(sourceFile, targetPath);
      this.stats.movedFiles++;
    } catch (error) {
      console.warn(`⚠️  Failed to move ${sourceFile}: ${error.message}`);
    }
  }

  showStats() {
    console.log('\n📊 Organization Statistics:');
    console.log('='.repeat(40));
    console.log(`Total files found:     ${this.stats.totalFiles}`);
    console.log(`Files moved:           ${this.stats.movedFiles}`);
    console.log(`Files kept in root:    ${this.stats.skippedFiles}`);
    console.log(`Directories created:   ${this.stats.createdDirs}`);
    console.log('='.repeat(40));
    
    if (this.dryRun) {
      console.log('\n💡 This was a dry run. Use without --dry-run to execute.');
    } else {
      console.log('\n✅ Repository organization complete!');
      console.log(`📁 Root directory now contains ${this.stats.skippedFiles} essential files.`);
    }
  }
}

// Run the organizer
const organizer = new RepositoryOrganizer();
await organizer.run();