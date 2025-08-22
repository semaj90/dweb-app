#!/usr/bin/env node

/**
 * TypeScript Error Analysis and Categorization Tool
 * Analyzes 742+ TypeScript errors and creates strategic fixing plan
 */

const fs = require('fs');
const path = require('path');

function analyzeTypeScriptErrors() {
  console.log('🔍 Analyzing TypeScript Errors...\n');
  
  const errorLog = fs.readFileSync('typescript-errors.log', 'utf8');
  const lines = errorLog.split('\n').filter(line => line.trim());
  
  const errorCategories = {
    'syntax': {
      patterns: [/TS1003/, /TS1005/, /TS1128/, /TS1131/, /TS1136/, /TS1434/, /TS1109/, /TS1011/],
      description: 'Syntax errors - missing semicolons, identifiers, declarations',
      priority: 'HIGH',
      errors: []
    },
    'imports': {
      patterns: [/TS2307/, /TS2306/, /TS2304/, /TS2305/, /TS2339/],
      description: 'Import and module resolution errors',
      priority: 'HIGH',
      errors: []
    },
    'types': {
      patterns: [/TS2457/, /TS2322/, /TS2345/, /TS2339/, /TS2571/, /TS2493/],
      description: 'Type definition and assignment errors',
      priority: 'MEDIUM',
      errors: []
    },
    'declarations': {
      patterns: [/TS2451/, /TS2450/, /TS2448/, /TS2447/],
      description: 'Variable and function declaration issues',
      priority: 'MEDIUM',
      errors: []
    },
    'accessibility': {
      patterns: [/TS6133/, /TS6138/, /TS6196/],
      description: 'Unused variables and accessibility issues',
      priority: 'LOW',
      errors: []
    }
  };

  const fileErrorCount = {};
  const errorCodeCount = {};
  
  let matchedLines = 0;
  lines.forEach((line, index) => {
    // Debug: Show first few lines
    if (index < 3) {
      console.log(`Debug line ${index}: "${line}"`);
    }
    
    // Extract file path and error code
    const match = line.match(/^(.*?)\((\d+),(\d+)\): error (TS\d+): (.*)$/);
    if (match) {
      matchedLines++;
      if (index < 3) {
        console.log(`Debug match ${index}:`, match[1], match[4]);
      }
      const [, filePath, lineNum, colNum, errorCode, message] = match;
      
      // Count errors per file
      if (!fileErrorCount[filePath]) fileErrorCount[filePath] = 0;
      fileErrorCount[filePath]++;
      
      // Count error codes
      if (!errorCodeCount[errorCode]) errorCodeCount[errorCode] = 0;
      errorCodeCount[errorCode]++;
      
      // Categorize errors
      let categorized = false;
      for (const [category, config] of Object.entries(errorCategories)) {
        if (config.patterns.some(pattern => pattern.test(errorCode))) {
          config.errors.push({
            file: filePath,
            line: lineNum,
            column: colNum,
            code: errorCode,
            message: message.trim()
          });
          categorized = true;
          break;
        }
      }
      
      if (!categorized) {
        if (!errorCategories.other) {
          errorCategories.other = {
            description: 'Other/Uncategorized errors',
            priority: 'REVIEW',
            errors: []
          };
        }
        errorCategories.other.errors.push({
          file: filePath,
          line: lineNum,
          column: colNum,
          code: errorCode,
          message: message.trim()
        });
      }
    }
  });

  console.log(`Debug: Matched ${matchedLines} out of ${lines.length} lines`);

  // Generate analysis report
  const report = {
    timestamp: new Date().toISOString(),
    totalErrors: lines.length,
    categories: {},
    topErrorFiles: Object.entries(fileErrorCount)
      .sort(([,a], [,b]) => b - a)
      .slice(0, 15),
    topErrorCodes: Object.entries(errorCodeCount)
      .sort(([,a], [,b]) => b - a)
      .slice(0, 15),
    fixingStrategy: []
  };

  console.log('📊 ERROR ANALYSIS SUMMARY');
  console.log('========================\n');
  console.log(`Total Errors: ${report.totalErrors}`);
  console.log(`Unique Files: ${Object.keys(fileErrorCount).length}`);
  console.log(`Unique Error Codes: ${Object.keys(errorCodeCount).length}\n`);

  // Categorize and prioritize
  Object.entries(errorCategories).forEach(([category, config]) => {
    if (config.errors.length > 0) {
      report.categories[category] = {
        count: config.errors.length,
        priority: config.priority,
        description: config.description,
        errors: config.errors
      };
      
      console.log(`${getPriorityIcon(config.priority)} ${category.toUpperCase()}: ${config.errors.length} errors`);
      console.log(`   ${config.description}`);
      console.log(`   Priority: ${config.priority}\n`);
    }
  });

  console.log('🔥 TOP ERROR FILES:');
  report.topErrorFiles.forEach(([file, count], index) => {
    console.log(`${(index + 1).toString().padStart(2)}. ${path.basename(file)} (${count} errors)`);
  });

  console.log('\n🚨 TOP ERROR CODES:');
  report.topErrorCodes.forEach(([code, count], index) => {
    console.log(`${(index + 1).toString().padStart(2)}. ${code} (${count} occurrences)`);
  });

  // Generate fixing strategy
  const strategy = generateFixingStrategy(report.categories);
  report.fixingStrategy = strategy;

  console.log('\n🛠️  STRATEGIC FIXING PLAN:');
  strategy.forEach((phase, index) => {
    console.log(`\nPhase ${index + 1}: ${phase.name}`);
    console.log(`Priority: ${phase.priority}`);
    console.log(`Files: ${phase.files.length}`);
    console.log(`Expected Impact: ${phase.impact}`);
    if (phase.automatable) {
      console.log('✅ Can be automated');
    }
  });

  // Save detailed report
  fs.writeFileSync('typescript-error-analysis.json', JSON.stringify(report, null, 2));
  
  // Generate actionable TODO files
  generateTodoFiles(report);
  
  console.log('\n📝 Generated Files:');
  console.log('- typescript-error-analysis.json (detailed report)');
  console.log('- fix-high-priority-errors.md (action plan)');
  console.log('- error-categories.md (categorized list)');
  
  return report;
}

function getPriorityIcon(priority) {
  switch (priority) {
    case 'HIGH': return '🔴';
    case 'MEDIUM': return '🟡';
    case 'LOW': return '🟢';
    default: return '🔵';
  }
}

function generateFixingStrategy(categories) {
  const strategy = [];
  
  // Phase 1: Critical syntax errors
  if (categories.syntax) {
    strategy.push({
      name: 'Fix Critical Syntax Errors',
      priority: 'HIGH',
      files: [...new Set(categories.syntax.errors.map(e => e.file))],
      impact: 'Enables compilation and basic functionality',
      automatable: true,
      description: 'Fix missing semicolons, identifiers, and basic syntax issues'
    });
  }

  // Phase 2: Import/module resolution
  if (categories.imports) {
    strategy.push({
      name: 'Resolve Import Issues',
      priority: 'HIGH', 
      files: [...new Set(categories.imports.errors.map(e => e.file))],
      impact: 'Enables module loading and dependencies',
      automatable: false,
      description: 'Fix import paths and module resolution'
    });
  }

  // Phase 3: Type definitions
  if (categories.types) {
    strategy.push({
      name: 'Fix Type Definitions',
      priority: 'MEDIUM',
      files: [...new Set(categories.types.errors.map(e => e.file))],
      impact: 'Improves type safety and IntelliSense',
      automatable: false,
      description: 'Add missing types and fix type mismatches'
    });
  }

  // Phase 4: Cleanup and optimization
  if (categories.accessibility) {
    strategy.push({
      name: 'Code Cleanup',
      priority: 'LOW',
      files: [...new Set(categories.accessibility.errors.map(e => e.file))],
      impact: 'Code quality and maintainability',
      automatable: true,
      description: 'Remove unused imports and variables'
    });
  }

  return strategy;
}

function generateTodoFiles(report) {
  // Generate high-priority action plan
  const highPriorityPlan = `# High-Priority TypeScript Error Fixes

Generated: ${new Date().toLocaleString()}
Total Errors: ${report.totalErrors}

## Phase 1: Critical Syntax Errors (Immediate Action Required)

${report.categories.syntax ? `
### Syntax Errors (${report.categories.syntax.count} errors)
Priority: 🔴 HIGH - These prevent compilation

**Top affected files:**
${getTopFilesForCategory(report.categories.syntax, 10).map(([file, count]) => 
  `- \`${file}\` (${count} errors)`
).join('\n')}

**Common error patterns:**
- TS1003: Missing identifiers  
- TS1005: Missing semicolons
- TS1128: Missing declarations
- TS1131: Property/signature issues

**Quick Fix Strategy:**
1. ✅ Use automated tools where possible
2. ✅ Fix files with highest error counts first
3. ✅ Focus on syntax completion before logic

` : ''}

${report.categories.imports ? `
### Import Resolution (${report.categories.imports.count} errors)  
Priority: 🔴 HIGH - Critical for functionality

**Action Items:**
- Review import paths and module resolution
- Check for missing dependencies
- Verify barrel export configurations

` : ''}

## Phase 2: Type Safety (Medium Priority)

${report.categories.types ? `
### Type Definitions (${report.categories.types.count} errors)
Priority: 🟡 MEDIUM

**Strategy:**
- Add missing type annotations
- Fix type mismatches  
- Review generic constraints

` : ''}

## Implementation Plan

### Week 1: Syntax & Imports
- [ ] Fix syntax errors in top 10 files
- [ ] Resolve critical import issues
- [ ] Enable basic compilation

### Week 2: Type Safety  
- [ ] Add missing type definitions
- [ ] Fix type assignment errors
- [ ] Improve generic types

### Week 3: Cleanup
- [ ] Remove unused imports/variables
- [ ] Code quality improvements
- [ ] Documentation updates

## Monitoring Progress

Track progress using:
\`\`\`bash
npm run check:typescript | grep -c "error TS"
\`\`\`

Target: Reduce from ${report.totalErrors} to <50 errors
`;

  fs.writeFileSync('fix-high-priority-errors.md', highPriorityPlan);

  // Generate categorized error list
  const categorizedErrors = `# TypeScript Error Categories

${Object.entries(report.categories).map(([category, data]) => `
## ${category.toUpperCase()} (${data.count} errors)

**Priority:** ${data.priority}  
**Description:** ${data.description}

### Top Files:
${getTopFilesForCategory(data, 5).map(([file, count]) => 
  `- \`${file}\` (${count} errors)`
).join('\n')}

### Sample Errors:
${data.errors.slice(0, 3).map(err => 
  `- **${err.code}:** Line ${err.line} - ${err.message}`
).join('\n')}

`).join('\n')}`;

  fs.writeFileSync('error-categories.md', categorizedErrors);
}

function getTopFilesForCategory(category, limit = 5) {
  const fileCounts = {};
  category.errors.forEach(error => {
    if (!fileCounts[error.file]) fileCounts[error.file] = 0;
    fileCounts[error.file]++;
  });
  
  return Object.entries(fileCounts)
    .sort(([,a], [,b]) => b - a)
    .slice(0, limit);
}

// Run the analysis
if (require.main === module) {
  try {
    analyzeTypeScriptErrors();
  } catch (error) {
    console.error('❌ Error during analysis:', error.message);
    process.exit(1);
  }
}

module.exports = { analyzeTypeScriptErrors };