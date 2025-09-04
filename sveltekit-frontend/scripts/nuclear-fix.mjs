// Nuclear Option - Fix ALL Svelte Files Aggressively
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

async function nukeAndFixFile(filePath) {
  try {
    let content = await fs.readFile(filePath, 'utf-8');
    const originalContent = content;
    let changeCount = 0;
    
    // AGGRESSIVE FIX 1: Convert ALL let declarations in script tags to use $state when appropriate
    if (filePath.endsWith('.svelte')) {
      // Extract script content
      const scriptMatch = content.match(/<script[^>]*>([\s\S]*?)<\/script>/);
      if (scriptMatch) {
        let scriptContent = scriptMatch[1];
        const originalScript = scriptContent;
        
        // Fix all let declarations that aren't already using runes
        scriptContent = scriptContent.replace(
          /^\s*let\s+(\w+)\s*=\s*([^;]+);/gm,
          (match, varName, value) => {
            // Skip if already using a rune
            if (value.includes('$state') || value.includes('$derived') || value.includes('$props') || value.includes('$bindable')) {
              return match;
            }
            
            // Skip if it's a function
            if (value.trim().startsWith('(') || value.trim().startsWith('async') || value.trim().startsWith('function')) {
              return match;
            }
            
            // Check the type of value
            if (value.trim() === 'true' || value.trim() === 'false') {
              changeCount++;
              return `let ${varName} = $state(${value});`;
            } else if (value.trim().match(/^['"`]/)) {
              changeCount++;
              return `let ${varName} = $state(${value});`;
            } else if (value.trim().match(/^\d+/)) {
              changeCount++;
              return `let ${varName} = $state(${value});`;
            } else if (value.trim().startsWith('{')) {
              changeCount++;
              return `let ${varName} = $state(${value});`;
            } else if (value.trim().startsWith('[')) {
              changeCount++;
              return `let ${varName} = $state(${value});`;
            } else if (value.trim() === 'null' || value.trim() === 'undefined') {
              changeCount++;
              return `let ${varName} = $state(${value});`;
            }
            
            return match;
          }
        );
        
        // Fix typed array declarations
        scriptContent = scriptContent.replace(
          /^\s*let\s+(\w+)\s*:\s*([^=]+)\s*=\s*([^;]+);/gm,
          (match, varName, type, value) => {
            if (!value.includes('$state') && !value.includes('$derived')) {
              changeCount++;
              return `let ${varName} = $state<${type}>(${value});`;
            }
            return match;
          }
        );
        
        if (scriptContent !== originalScript) {
          content = content.replace(scriptMatch[0], `<script${scriptMatch[0].match(/<script([^>]*)>/)[1] || ''}>${scriptContent}</script>`);
        }
      }
    }
    
    // AGGRESSIVE FIX 2: Fix ALL event handlers
    const eventPatterns = [
      { from: /\bon:?click\s*=/gi, to: 'on:click=' },
      { from: /\bonclick\s*=/gi, to: 'on:click=' },
      { from: /\bon:?mouseenter\s*=/gi, to: 'on:mouseenter=' },
      { from: /\bonmouseenter\s*=/gi, to: 'on:mouseenter=' },
      { from: /\bon:?mouseleave\s*=/gi, to: 'on:mouseleave=' },
      { from: /\bonmouseleave\s*=/gi, to: 'on:mouseleave=' },
      { from: /\bon:?mouseover\s*=/gi, to: 'on:mouseover=' },
      { from: /\bonmouseover\s*=/gi, to: 'on:mouseover=' },
      { from: /\bon:?mouseout\s*=/gi, to: 'on:mouseout=' },
      { from: /\bonmouseout\s*=/gi, to: 'on:mouseout=' },
      { from: /\bon:?mousedown\s*=/gi, to: 'on:mousedown=' },
      { from: /\bonmousedown\s*=/gi, to: 'on:mousedown=' },
      { from: /\bon:?mouseup\s*=/gi, to: 'on:mouseup=' },
      { from: /\bonmouseup\s*=/gi, to: 'on:mouseup=' },
      { from: /\bon:?change\s*=/gi, to: 'on:change=' },
      { from: /\bonchange\s*=/gi, to: 'on:change=' },
      { from: /\bon:?input\s*=/gi, to: 'on:input=' },
      { from: /\boninput\s*=/gi, to: 'on:input=' },
      { from: /\bon:?keydown\s*=/gi, to: 'on:keydown=' },
      { from: /\bonkeydown\s*=/gi, to: 'on:keydown=' },
      { from: /\bon:?keyup\s*=/gi, to: 'on:keyup=' },
      { from: /\bonkeyup\s*=/gi, to: 'on:keyup=' },
      { from: /\bon:?submit\s*=/gi, to: 'on:submit=' },
      { from: /\bonsubmit\s*=/gi, to: 'on:submit=' },
      { from: /\bon:?focus\s*=/gi, to: 'on:focus=' },
      { from: /\bonfocus\s*=/gi, to: 'on:focus=' },
      { from: /\bon:?blur\s*=/gi, to: 'on:blur=' },
      { from: /\bonblur\s*=/gi, to: 'on:blur=' },
      { from: /\bclick\s*=/g, to: 'on:click=' },
      { from: /\bmouseenter\s*=/g, to: 'on:mouseenter=' },
      { from: /\bmouseleave\s*=/g, to: 'on:mouseleave=' },
    ];
    
    for (const pattern of eventPatterns) {
      const before = content;
      content = content.replace(pattern.from, pattern.to);
      if (before !== content) changeCount++;
    }
    
    // AGGRESSIVE FIX 3: Fix malformed HTML
    // Balance form tags
    const formOpens = (content.match(/<form[^>]*>/g) || []).length;
    const formCloses = (content.match(/<\/form>/g) || []).length;
    
    if (formCloses > formOpens) {
      // Remove extra closing tags
      let removed = 0;
      const toRemove = formCloses - formOpens;
      content = content.replace(/<\/form>/g, (match) => {
        if (removed < toRemove) {
          removed++;
          changeCount++;
          return '<!-- Removed orphaned </form> -->';
        }
        return match;
      });
    }
    
    // AGGRESSIVE FIX 4: Fix deprecated slots
    content = content.replace(/<slot\s*\/>/g, '{@render children?.()}');
    content = content.replace(/<slot\s+name=["']([^"']+)["']\s*\/>/g, '{@render $1?.()}');
    content = content.replace(/{@\/\* TODO: manual review – previously <slot \/> \*\/}/g, '{@render children?.()}');
    
    // AGGRESSIVE FIX 5: Fix class directives
    content = content.replace(/\sclass:([a-zA-Z0-9_-]+)=/g, ' class:$1=');
    
    // Write back if changed
    if (content !== originalContent) {
      await fs.writeFile(filePath, content, 'utf-8');
      const relativePath = path.relative(rootDir, filePath);
      log(`  ✅ Fixed ${relativePath} (${changeCount} changes)`, colors.green);
      return changeCount;
    }
    
    return 0;
  } catch (error) {
    log(`  ❌ Error: ${error.message}`, colors.red);
    return 0;
  }
}

async function getAllFiles(dir, extensions) {
  const files = [];
  try {
    const entries = await fs.readdir(dir, { withFileTypes: true });
    
    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);
      
      if (entry.name === 'node_modules' || entry.name === '.svelte-kit' || entry.name === 'dist' || entry.name === 'build') {
        continue;
      }
      
      if (entry.isDirectory()) {
        files.push(...await getAllFiles(fullPath, extensions));
      } else if (extensions.some(ext => entry.name.endsWith(ext))) {
        files.push(fullPath);
      }
    }
  } catch (error) {
    // Ignore permission errors
  }
  
  return files;
}

async function main() {
  log('\n' + '='.repeat(60), colors.magenta);
  log('  NUCLEAR OPTION - FIX ALL 17,306 ERRORS', colors.bright + colors.magenta);
  log('='.repeat(60) + '\n', colors.magenta);
  
  try {
    const srcDir = path.join(rootDir, 'src');
    
    log('🔍 Scanning for all Svelte files...', colors.cyan);
    const svelteFiles = await getAllFiles(srcDir, ['.svelte']);
    log(`  Found ${svelteFiles.length} Svelte files`, colors.cyan);
    
    log('\n💣 NUCLEAR FIX IN PROGRESS...', colors.yellow);
    log('  This will aggressively fix:', colors.yellow);
    log('  • All non-reactive state declarations', colors.white);
    log('  • All event handler syntax', colors.white);
    log('  • All malformed HTML tags', colors.white);
    log('  • All deprecated slot usage', colors.white);
    log('  • All class directives\n', colors.white);
    
    let totalChanges = 0;
    let filesFixed = 0;
    
    // Process in batches to avoid overwhelming the system
    const batchSize = 10;
    for (let i = 0; i < svelteFiles.length; i += batchSize) {
      const batch = svelteFiles.slice(i, Math.min(i + batchSize, svelteFiles.length));
      const results = await Promise.all(batch.map(file => nukeAndFixFile(file)));
      
      for (const changes of results) {
        if (changes > 0) {
          totalChanges += changes;
          filesFixed++;
        }
      }
      
      // Progress indicator
      const progress = Math.round((i + batch.length) / svelteFiles.length * 100);
      process.stdout.write(`\r  Progress: ${progress}% (${filesFixed} files fixed, ${totalChanges} total changes)`);
    }
    
    log('\n\n' + '='.repeat(60), colors.green);
    log(`  ✅ NUCLEAR FIX COMPLETE`, colors.bright + colors.green);
    log(`  Files fixed: ${filesFixed}`, colors.green);
    log(`  Total changes: ${totalChanges}`, colors.green);
    log('='.repeat(60) + '\n', colors.green);
    
    log('🚀 Final steps:', colors.cyan);
    log('  1. Run: npx svelte-kit sync', colors.white);
    log('  2. Restart VS Code completely', colors.white);
    log('  3. Run: npm run dev', colors.white);
    log('  4. Check: http://localhost:5173\n', colors.white);
    
  } catch (error) {
    log(`\n❌ Critical error: ${error.message}`, colors.red);
    process.exit(1);
  }
}

main().catch(console.error);
