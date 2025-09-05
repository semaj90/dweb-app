#!/usr/bin/env node
/**
 * Phase 9: Automated Component Migration Script
 * Based on Context7 MCP Best Practices for SvelteKit 2 + Svelte 5
 */

import { readFileSync, writeFileSync, readdirSync, statSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const srcDir = join(__dirname, '../src');

console.log('🚀 Phase 9: Automated Svelte 5 Component Migration');
console.log('=====================================');

let migrationStats = {
  filesScanned: 0,
  filesModified: 0,
  eventDispatcherFixed: 0,
  exportLetFixed: 0,
  eventHandlerFixed: 0,
  cssClassFixed: 0,
  errors: []
};

// Migration patterns
const MIGRATION_PATTERNS = [
  // 1. createEventDispatcher to callback props
  {
    name: 'EventDispatcher Migration',
    detect: /import.*createEventDispatcher.*from\s+['"]svelte['"];?\n.*const\s+dispatch\s*=\s*createEventDispatcher\(\)/s,
    apply: (content, filename) => {
      if (!content.includes('createEventDispatcher')) return content;
      
      // Extract event types from dispatch calls
      const dispatchCalls = content.match(/dispatch\(['"](\w+)['"][^)]*\)/g) || [];
      const eventTypes = [...new Set(dispatchCalls.map(call => call.match(/['"](\w+)['"]/)?.[1]).filter(Boolean))];
      
      let modified = content;
      
      // Remove createEventDispatcher import and usage
      modified = modified.replace(/import.*createEventDispatcher.*from\s+['"]svelte['"];?\n/g, '');
      modified = modified.replace(/const\s+dispatch\s*=\s*createEventDispatcher\(\);\n?/g, '');
      
      // Add callback props to interface or create one
      const interfaceMatch = modified.match(/interface\s+Props\s*{([^}]*)}/s);
      if (interfaceMatch) {
        const interfaceContent = interfaceMatch[1];
        const callbackProps = eventTypes.map(event => 
          `    on${event}?: (event?: any) => void;`
        ).join('\n');
        
        modified = modified.replace(
          /interface\s+Props\s*{([^}]*)}/s,
          `interface Props {${interfaceContent}${callbackProps}\n  }`
        );
      } else {
        // Create interface if none exists
        const callbackProps = eventTypes.map(event => 
          `    on${event}?: (event?: any) => void;`
        ).join('\n');
        
        const interfaceDef = `  interface Props {\n${callbackProps}\n  }\n\n`;
        modified = modified.replace(/(<script[^>]*>)/i, `$1\n${interfaceDef}`);
      }
      
      // Update props destructuring
      eventTypes.forEach(event => {
        const propName = `on${event}`;
        if (!modified.includes(`${propName},`) && !modified.includes(`${propName} }`)) {
          modified = modified.replace(
            /(let\s*{\s*[^}]*?)(}\s*:\s*Props\s*=\s*\$props\(\))/,
            `$1,\n    ${propName}$2`
          );
        }
      });
      
      // Replace dispatch calls with callback calls
      eventTypes.forEach(event => {
        const dispatchRegex = new RegExp(`dispatch\\(['"]${event}['"][^)]*\\)`, 'g');
        modified = modified.replace(dispatchRegex, `on${event}?.()`);
      });
      
      migrationStats.eventDispatcherFixed++;
      return modified;
    }
  },
  
  // 2. export let to $props()
  {
    name: 'Export Let Migration',
    detect: /export\s+let\s+\w+/,
    apply: (content, filename) => {
      const exportLets = content.match(/export\s+let\s+(\w+)(?:\s*:\s*[^=\n]+)?(?:\s*=\s*[^;\n]+)?;?/g);
      if (!exportLets) return content;
      
      let modified = content;
      const props = [];
      
      exportLets.forEach(exportLet => {
        const match = exportLet.match(/export\s+let\s+(\w+)(?:\s*:\s*([^=\n]+))?(?:\s*=\s*([^;\n]+))?/);
        if (match) {
          const [, name, type, defaultValue] = match;
          props.push({ name, type, defaultValue });
          
          // Remove the export let
          modified = modified.replace(exportLet + '\n', '');
        }
      });
      
      if (props.length > 0) {
        // Check if Props interface exists
        const hasPropsInterface = /interface\s+Props\s*{/.test(modified);
        
        if (!hasPropsInterface) {
          // Create Props interface
          const interfaceDef = `  interface Props {\n${props.map(p => 
            `    ${p.name}${p.type ? `: ${p.type}` : '?: any'}${p.defaultValue ? '' : ''};`
          ).join('\n')}\n  }\n\n`;
          modified = modified.replace(/(<script[^>]*>)/i, `$1\n${interfaceDef}`);
        }
        
        // Add props destructuring
        const propsDestructure = `  let {\n${props.map(p => 
          `    ${p.name}${p.defaultValue ? ` = ${p.defaultValue}` : ''}`
        ).join(',\n')}\n  }: Props = $props();\n\n`;
        
        const scriptMatch = modified.match(/(<script[^>]*>[\s\S]*?interface Props[^}]*}[\s\S]*?\n)/);
        if (scriptMatch) {
          modified = modified.replace(scriptMatch[1], scriptMatch[1] + propsDestructure);
        }
        
        migrationStats.exportLetFixed++;
      }
      
      return modified;
    }
  },
  
  // 3. Event handler type safety
  {
    name: 'Event Handler Type Safety',
    detect: /on\w+={(.*?e\.target.*?)}/g,
    apply: (content, filename) => {
      let modified = content;
      
      // Fix common event handler patterns
      const patterns = [
        {
          from: /oninput={(.*?)e\.target\.value(.*?)}/g,
          to: 'oninput={$1(e.target as HTMLInputElement).value$2}'
        },
        {
          from: /onchange={(.*?)e\.target\.value(.*?)}/g,
          to: 'onchange={$1(e.target as HTMLSelectElement).value$2}'
        },
        {
          from: /onchange={(.*?)e\.target\.checked(.*?)}/g,
          to: 'onchange={$1(e.target as HTMLInputElement).checked$2}'
        },
        {
          from: /onclick={(.*?)e\.target(.*?)}/g,
          to: 'onclick={$1(e.target as HTMLButtonElement)$2}'
        }
      ];
      
      patterns.forEach(pattern => {
        if (pattern.from.test(content)) {
          modified = modified.replace(pattern.from, pattern.to);
          migrationStats.eventHandlerFixed++;
        }
      });
      
      return modified;
    }
  },
  
  // 4. CSS placeholder class fixes
  {
    name: 'CSS Placeholder Fix',
    detect: /class="container mx-auto px-4"/g,
    apply: (content, filename) => {
      if (!content.includes('class="container mx-auto px-4"')) return content;
      
      let modified = content;
      
      // Common replacements based on context
      const replacements = [
        { from: 'class="container mx-auto px-4"', to: 'class="space-y-4"' },
      ];
      
      replacements.forEach(replacement => {
        modified = modified.replace(new RegExp(replacement.from, 'g'), replacement.to);
      });
      
      migrationStats.cssClassFixed++;
      return modified;
    }
  }
];

function findSvelteFiles(dir) {
  let files = [];
  
  try {
    const items = readdirSync(dir);
    
    for (const item of items) {
      const fullPath = join(dir, item);
      const stat = statSync(fullPath);
      
      if (stat.isDirectory() && !item.startsWith('.') && item !== 'node_modules') {
        files = files.concat(findSvelteFiles(fullPath));
      } else if (stat.isFile() && item.endsWith('.svelte')) {
        files.push(fullPath);
      }
    }
  } catch (error) {
    migrationStats.errors.push(`Error reading directory ${dir}: ${error.message}`);
  }
  
  return files;
}

function migrateComponent(filePath) {
  try {
    const content = readFileSync(filePath, 'utf8');
    let modified = content;
    let hasChanges = false;
    
    migrationStats.filesScanned++;
    
    for (const pattern of MIGRATION_PATTERNS) {
      if (pattern.detect.test(content)) {
        console.log(`  🔧 Applying ${pattern.name} to ${filePath.split('/').pop()}`);
        const newContent = pattern.apply(modified, filePath);
        if (newContent !== modified) {
          modified = newContent;
          hasChanges = true;
        }
      }
    }
    
    if (hasChanges) {
      // Create backup
      const backupPath = filePath + `.backup.${Date.now()}`;
      writeFileSync(backupPath, content);
      
      // Write modified content
      writeFileSync(filePath, modified);
      migrationStats.filesModified++;
      
      console.log(`  ✅ Migrated ${filePath.split('/').pop()} (backup: ${backupPath.split('/').pop()})`);
    }
    
  } catch (error) {
    migrationStats.errors.push(`Error processing ${filePath}: ${error.message}`);
  }
}

async function main() {
  console.log(`📁 Scanning for Svelte components in: ${srcDir}`);
  
  const svelteFiles = findSvelteFiles(srcDir);
  console.log(`📊 Found ${svelteFiles.length} Svelte components`);
  
  console.log('\n🔄 Starting migration...\n');
  
  for (const file of svelteFiles) {
    migrateComponent(file);
  }
  
  // Print results
  console.log('\n📋 Migration Summary:');
  console.log('====================');
  console.log(`📄 Files scanned: ${migrationStats.filesScanned}`);
  console.log(`✅ Files modified: ${migrationStats.filesModified}`);
  console.log(`🔀 Event dispatchers fixed: ${migrationStats.eventDispatcherFixed}`);
  console.log(`📤 Export lets fixed: ${migrationStats.exportLetFixed}`);
  console.log(`🎯 Event handlers fixed: ${migrationStats.eventHandlerFixed}`);
  console.log(`🎨 CSS classes fixed: ${migrationStats.cssClassFixed}`);
  
  if (migrationStats.errors.length > 0) {
    console.log(`\n❌ Errors encountered: ${migrationStats.errors.length}`);
    migrationStats.errors.forEach(error => console.log(`   ${error}`));
  }
  
  console.log('\n🎉 Phase 9 migration completed!');
  console.log('\nNext steps:');
  console.log('1. Run: npm run check');
  console.log('2. Run: npm run dev');
  console.log('3. Test functionality');
  console.log('4. Commit changes');
}

main().catch(console.error);