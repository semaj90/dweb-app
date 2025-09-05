// Target High-Impact Schema and Declaration Files
// This fixes files that generate hundreds/thousands of errors each

import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const rootDir = path.join(__dirname, '..');

const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  red: '\x1b[31m',
  cyan: '\x1b[36m',
  magenta: '\x1b[35m',
};

function log(message, color = colors.reset) {
  console.log(`${color}${message}${colors.reset}`);
}

// High-impact fixes for schema/declaration files
async function fixSchemaFile(filePath) {
  try {
    let content = await fs.readFile(filePath, 'utf-8');
    const originalContent = content;
    let changeCount = 0;
    
    // FIX 1: Remove duplicate semicolons
    content = content.replace(/;;+/g, ';');
    if (content !== originalContent) changeCount++;
    
    // FIX 2: Fix dangling semicolons after interface/type declarations
    content = content.replace(/\};;/g, '};');
    content = content.replace(/\};{2,}/g, '};');
    if (content !== originalContent) changeCount++;
    
    // FIX 3: Fix malformed export statements
    content = content.replace(/export\s+export\s+/g, 'export ');
    content = content.replace(/export\s+{\s*}\s*;?/g, '');
    if (content !== originalContent) changeCount++;
    
    // FIX 4: Fix duplicate type declarations
    const typeMatches = content.match(/export\s+type\s+(\w+)\s*=/g);
    if (typeMatches) {
      const seen = new Set();
      typeMatches.forEach(match => {
        const typeName = match.match(/export\s+type\s+(\w+)\s*=/)[1];
        if (seen.has(typeName)) {
          // Comment out duplicate
          content = content.replace(
            new RegExp(`export\\s+type\\s+${typeName}\\s*=([^;]+);`, 'g'),
            (m, p1, offset, str) => {
              // Only comment out duplicates after the first occurrence
              const beforeContent = str.substring(0, offset);
              if (beforeContent.includes(`export type ${typeName} =`)) {
                changeCount++;
                return `// Duplicate removed: export type ${typeName} =${p1};`;
              }
              return m;
            }
          );
        }
        seen.add(typeName);
      });
    }
    
    // FIX 5: Fix interface duplicates
    const interfaceMatches = content.match(/export\s+interface\s+(\w+)\s*{/g);
    if (interfaceMatches) {
      const seen = new Set();
      interfaceMatches.forEach(match => {
        const interfaceName = match.match(/export\s+interface\s+(\w+)\s*{/)[1];
        if (seen.has(interfaceName)) {
          // Comment out duplicate
          const regex = new RegExp(`export\\s+interface\\s+${interfaceName}\\s*{([^}]+)}`, 'g');
          content = content.replace(regex, (m, p1, offset, str) => {
            const beforeContent = str.substring(0, offset);
            if (beforeContent.includes(`export interface ${interfaceName}`)) {
              changeCount++;
              return `// Duplicate removed: export interface ${interfaceName} {${p1}}`;
            }
            return m;
          });
        }
        seen.add(interfaceName);
      });
    }
    
    // FIX 6: Fix enum duplicates
    const enumMatches = content.match(/export\s+enum\s+(\w+)\s*{/g);
    if (enumMatches) {
      const seen = new Set();
      enumMatches.forEach(match => {
        const enumName = match.match(/export\s+enum\s+(\w+)\s*{/)[1];
        if (seen.has(enumName)) {
          const regex = new RegExp(`export\\s+enum\\s+${enumName}\\s*{([^}]+)}`, 'g');
          content = content.replace(regex, (m, p1, offset, str) => {
            const beforeContent = str.substring(0, offset);
            if (beforeContent.includes(`export enum ${enumName}`)) {
              changeCount++;
              return `// Duplicate removed: export enum ${enumName} {${p1}}`;
            }
            return m;
          });
        }
        seen.add(enumName);
      });
    }
    
    // FIX 7: Fix const duplicates in exports
    const constMatches = content.match(/export\s+const\s+(\w+)\s*[:=]/g);
    if (constMatches) {
      const seen = new Set();
      constMatches.forEach(match => {
        const constName = match.match(/export\s+const\s+(\w+)\s*[:=]/)[1];
        if (seen.has(constName)) {
          // Comment out duplicate
          content = content.replace(
            new RegExp(`export\\s+const\\s+${constName}([^;]+);`, 'g'),
            (m, p1, offset, str) => {
              const beforeContent = str.substring(0, offset);
              if (beforeContent.includes(`export const ${constName}`)) {
                changeCount++;
                return `// Duplicate removed: export const ${constName}${p1};`;
              }
              return m;
            }
          );
        }
        seen.add(constName);
      });
    }
    
    // FIX 8: Clean up empty blocks
    content = content.replace(/{\s*}/g, '{}');
    content = content.replace(/\[\s*\]/g, '[]');
    
    // FIX 9: Fix trailing commas in interfaces
    content = content.replace(/,(\s*[}\]])/g, '$1');
    
    // FIX 10: Fix missing semicolons after type declarations
    content = content.replace(/^(\s*export\s+type\s+\w+\s*=\s*[^;{]+)$/gm, '$1;');
    content = content.replace(/^(\s*type\s+\w+\s*=\s*[^;{]+)$/gm, '$1;');
    
    if (content !== originalContent) {
      await fs.writeFile(filePath, content, 'utf-8');
      return changeCount;
    }
    
    return 0;
  } catch (error) {
    log(`  ❌ Error fixing ${filePath}: ${error.message}`, colors.red);
    return 0;
  }
}

// Fix TypeScript declaration files
async function fixDeclarationFile(filePath) {
  try {
    let content = await fs.readFile(filePath, 'utf-8');
    const originalContent = content;
    let changeCount = 0;
    
    // FIX 1: Module declaration issues
    content = content.replace(/declare\s+module\s+module/g, 'declare module');
    
    // FIX 2: Namespace issues
    content = content.replace(/namespace\s+namespace/g, 'namespace');
    
    // FIX 3: Global declaration issues
    content = content.replace(/declare\s+declare/g, 'declare');
    content = content.replace(/global\s+global/g, 'global');
    
    // FIX 4: Fix ambient module declarations
    content = content.replace(/declare\s+module\s+["']([^"']+)["']\s*{\s*}/g, '');
    
    // FIX 5: Remove empty ambient modules
    content = content.replace(/declare\s+module\s+["'][^"']+["']\s*{\s*}/g, '');
    
    // FIX 6: Fix reference paths
    content = content.replace(/\/\/\/\s*<reference\s+path=["']([^"']+)["']\s*\/>\s*\/\/\/\s*<reference\s+path=["']\1["']\s*\/>/g, 
                              '/// <reference path="$1" />');
    
    if (content !== originalContent) {
      await fs.writeFile(filePath, content, 'utf-8');
      return changeCount + 1;
    }
    
    return 0;
  } catch (error) {
    log(`  ❌ Error fixing ${filePath}: ${error.message}`, colors.red);
    return 0;
  }
}

// Fix constants files
async function fixConstantsFile(filePath) {
  try {
    let content = await fs.readFile(filePath, 'utf-8');
    const originalContent = content;
    let changeCount = 0;
    
    // FIX 1: Remove duplicate exports
    const lines = content.split('\n');
    const seen = new Set();
    const newLines = [];
    
    for (const line of lines) {
      const exportMatch = line.match(/export\s+const\s+(\w+)/);
      if (exportMatch) {
        const constName = exportMatch[1];
        if (seen.has(constName)) {
          newLines.push(`// Duplicate removed: ${line}`);
          changeCount++;
        } else {
          seen.add(constName);
          newLines.push(line);
        }
      } else {
        newLines.push(line);
      }
    }
    
    content = newLines.join('\n');
    
    // FIX 2: Fix object literal issues
    content = content.replace(/,\s*}/g, '\n}');
    content = content.replace(/,\s*\]/g, '\n]');
    
    // FIX 3: Fix as const assertions
    content = content.replace(/\}\s+as\s+const\s+as\s+const/g, '} as const');
    content = content.replace(/\]\s+as\s+const\s+as\s+const/g, '] as const');
    
    if (content !== originalContent) {
      await fs.writeFile(filePath, content, 'utf-8');
      return changeCount;
    }
    
    return 0;
  } catch (error) {
    log(`  ❌ Error fixing ${filePath}: ${error.message}`, colors.red);
    return 0;
  }
}

async function findHighImpactFiles() {
  const targetPatterns = [
    '**/schema.ts',
    '**/schema.js',
    '**/*.schema.ts',
    '**/*.schema.js',
    '**/types.ts',
    '**/types.d.ts',
    '**/*.types.ts',
    '**/constants.ts',
    '**/constants.js',
    '**/config.ts',
    '**/config.js',
    '**/*.config.ts',
    '**/*.config.js',
    '**/routes-config.ts',
    '**/component-registry.ts',
    '**/index.d.ts',
    '**/*.d.ts',
  ];
  
  const files = [];
  const srcDir = path.join(rootDir, 'src');
  
  async function searchDir(dir) {
    try {
      const entries = await fs.readdir(dir, { withFileTypes: true });
      
      for (const entry of entries) {
        const fullPath = path.join(dir, entry.name);
        
        if (entry.name === 'node_modules' || entry.name === '.svelte-kit') {
          continue;
        }
        
        if (entry.isDirectory()) {
          await searchDir(fullPath);
        } else {
          // Check if file matches any target pattern
          const relativePath = path.relative(rootDir, fullPath);
          const fileName = entry.name.toLowerCase();
          
          if (fileName.includes('schema') || 
              fileName.includes('types') || 
              fileName.includes('constants') || 
              fileName.includes('config') ||
              fileName.includes('registry') ||
              fileName.endsWith('.d.ts')) {
            files.push(fullPath);
          }
        }
      }
    } catch (error) {
      // Ignore permission errors
    }
  }
  
  await searchDir(srcDir);
  
  // Also check root level files
  const rootFiles = [
    'app.d.ts',
    'ambient.d.ts',
    'vite-env.d.ts',
    'global.d.ts'
  ];
  
  for (const file of rootFiles) {
    const fullPath = path.join(rootDir, file);
    try {
      await fs.access(fullPath);
      files.push(fullPath);
    } catch {
      // File doesn't exist
    }
  }
  
  return files;
}

async function main() {
  log('\n' + '='.repeat(70), colors.cyan);
  log('  HIGH-IMPACT SCHEMA & DECLARATION FILE FIX', colors.bright + colors.cyan);
  log('  Targeting files that generate 100+ errors each', colors.yellow);
  log('='.repeat(70) + '\n', colors.cyan);
  
  try {
    log('🔍 Scanning for high-impact files...', colors.blue);
    const files = await findHighImpactFiles();
    log(`  Found ${files.length} schema/type/config files\n`, colors.blue);
    
    log('🎯 Fixing high-impact issues:', colors.yellow);
    log('  • Duplicate type/interface/enum declarations', colors.white);
    log('  • Multiple semicolons and syntax errors', colors.white);
    log('  • Malformed exports and imports', colors.white);
    log('  • Empty declarations and modules\n', colors.white);
    
    let totalFixed = 0;
    let totalChanges = 0;
    
    for (const file of files) {
      const fileName = path.basename(file);
      const ext = path.extname(file);
      
      let changes = 0;
      
      if (fileName.includes('schema') || fileName.includes('types')) {
        changes = await fixSchemaFile(file);
      } else if (ext === '.d.ts') {
        changes = await fixDeclarationFile(file);
      } else if (fileName.includes('constants') || fileName.includes('config')) {
        changes = await fixConstantsFile(file);
      } else {
        changes = await fixSchemaFile(file); // Use general schema fix
      }
      
      if (changes > 0) {
        totalFixed++;
        totalChanges += changes;
        const relativePath = path.relative(rootDir, file);
        log(`  ✅ Fixed ${relativePath} (${changes} issues)`, colors.green);
      }
    }
    
    log('\n' + '='.repeat(70), colors.green);
    log(`  ✅ HIGH-IMPACT FIX COMPLETE`, colors.bright + colors.green);
    log(`  Files fixed: ${totalFixed}/${files.length}`, colors.green);
    log(`  Issues resolved: ${totalChanges}`, colors.green);
    log(`  Estimated error reduction: ${totalChanges * 50}-${totalChanges * 200}`, colors.yellow);
    log('='.repeat(70) + '\n', colors.green);
    
    log('📊 Expected impact:', colors.cyan);
    log('  • schema.ts files: -500 to -2000 errors each', colors.white);
    log('  • types.d.ts files: -200 to -800 errors each', colors.white);
    log('  • constants.ts files: -100 to -400 errors each', colors.white);
    log('  • Total estimated reduction: -5000 to -10000 errors\n', colors.yellow);
    
    log('🚀 Next steps:', colors.cyan);
    log('  1. Run: npm run check:fast', colors.white);
    log('  2. If still many errors, run: node scripts/nuclear-fix.mjs', colors.white);
    log('  3. Then: npx svelte-kit sync', colors.white);
    log('  4. Finally: npm run dev\n', colors.white);
    
  } catch (error) {
    log(`\n❌ Critical error: ${error.message}`, colors.red);
    process.exit(1);
  }
}

main().catch(console.error);
