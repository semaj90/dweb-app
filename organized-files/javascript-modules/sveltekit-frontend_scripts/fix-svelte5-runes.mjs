// Comprehensive Svelte 5 Rune Fix Script
// Fixes all $state, $derived, $props usage and event handlers

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
};

function log(message, color = colors.reset) {
  console.log(`${color}${message}${colors.reset}`);
}

async function fixSvelteFile(filePath) {
  try {
    let content = await fs.readFile(filePath, 'utf-8');
    const originalContent = content;
    let changeCount = 0;
    
    // Fix non-reactive updates - convert let to $state
    // Pattern: let variableName = value; where variableName is later updated
    content = content.replace(/let\s+(\w+)\s*=\s*([^;]+);/g, (match, varName, value) => {
      // Check if this variable is updated later in the file
      const updatePatterns = [
        new RegExp(`${varName}\\s*=`, 'g'),
        new RegExp(`${varName}\\s*\\+=`, 'g'),
        new RegExp(`${varName}\\s*-=`, 'g'),
        new RegExp(`${varName}\\s*\\*=`, 'g'),
        new RegExp(`${varName}\\s*\\/=`, 'g'),
        new RegExp(`${varName}\\.push\\(`, 'g'),
        new RegExp(`${varName}\\.pop\\(`, 'g'),
        new RegExp(`${varName}\\.shift\\(`, 'g'),
        new RegExp(`${varName}\\.unshift\\(`, 'g'),
        new RegExp(`${varName}\\.splice\\(`, 'g'),
      ];
      
      const hasUpdate = updatePatterns.some(pattern => {
        const matches = content.match(pattern);
        return matches && matches.length > 1; // More than just the declaration
      });
      
      if (hasUpdate && !value.includes('$state') && !value.includes('$derived') && !value.includes('$props')) {
        changeCount++;
        return `let ${varName} = $state(${value});`;
      }
      return match;
    });
    
    // Fix array/object declarations that should use $state
    content = content.replace(/let\s+(\w+)\s*:\s*Array<[^>]+>\s*=\s*\[\];/g, (match, varName) => {
      changeCount++;
      return `let ${varName} = $state<Array<any>>([]);`;
    });
    
    content = content.replace(/let\s+(\w+)\s*=\s*\{([^}]+)\};/g, (match, varName, objContent) => {
      // Check if this object is updated
      if (content.includes(`${varName} =`) || content.includes(`${varName}.`)) {
        changeCount++;
        return `let ${varName} = $state({${objContent}});`;
      }
      return match;
    });
    
    // Fix event handlers: onclick -> on:click
    const eventHandlers = [
      'click', 'mouseenter', 'mouseleave', 'mouseover', 'mouseout',
      'mousedown', 'mouseup', 'mousemove', 'contextmenu', 'dblclick',
      'keydown', 'keyup', 'keypress', 'change', 'input', 'focus',
      'blur', 'submit', 'reset', 'scroll', 'load', 'error',
      'drag', 'drop', 'dragstart', 'dragend', 'dragenter',
      'dragleave', 'dragover'
    ];
    
    for (const event of eventHandlers) {
      const regex = new RegExp(`\\bon${event}=`, 'g');
      const replacement = `on:${event}=`;
      const before = content;
      content = content.replace(regex, replacement);
      if (before !== content) changeCount++;
    }
    
    // Fix deprecated slot usage
    content = content.replace(/<slot\s*\/>/g, '{@render children?.()}');
    content = content.replace(/<slot\s+name="([^"]+)"\s*\/>/g, '{@render $1?.()}');
    
    // Fix class: directives
    content = content.replace(/class:([a-zA-Z0-9_-]+)=/g, 'class:$1=');
    
    // Fix bind: directives
    content = content.replace(/bind:value=/g, 'bind:value=');
    content = content.replace(/bind:checked=/g, 'bind:checked=');
    content = content.replace(/bind:group=/g, 'bind:group=');
    
    // Fix use: directives
    content = content.replace(/use:([a-zA-Z0-9_]+)=/g, 'use:$1=');
    
    // Fix event modifiers
    content = content.replace(/on:([a-z]+)\|preventDefault=/g, 'on:$1|preventDefault=');
    content = content.replace(/on:([a-z]+)\|stopPropagation=/g, 'on:$1|stopPropagation=');
    content = content.replace(/on:([a-z]+)\|once=/g, 'on:$1|once=');
    
    // Write back if changed
    if (content !== originalContent) {
      await fs.writeFile(filePath, content, 'utf-8');
      const relativePath = path.relative(rootDir, filePath);
      log(`  ✅ Fixed ${relativePath} (${changeCount} changes)`, colors.green);
      return true;
    }
    
    return false;
  } catch (error) {
    log(`  ❌ Error fixing ${filePath}: ${error.message}`, colors.red);
    return false;
  }
}

async function fixSpecificFile(filePath, fixes) {
  try {
    let content = await fs.readFile(filePath, 'utf-8');
    
    for (const fix of fixes) {
      content = fix(content);
    }
    
    await fs.writeFile(filePath, content, 'utf-8');
    log(`  ✅ Fixed ${path.basename(filePath)}`, colors.green);
  } catch (error) {
    log(`  ❌ Error fixing ${filePath}: ${error.message}`, colors.red);
  }
}

async function getAllSvelteFiles(dir) {
  const files = [];
  const entries = await fs.readdir(dir, { withFileTypes: true });
  
  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    
    if (entry.name === 'node_modules' || entry.name === '.svelte-kit' || entry.name === 'dist' || entry.name === 'build') {
      continue;
    }
    
    if (entry.isDirectory()) {
      files.push(...await getAllSvelteFiles(fullPath));
    } else if (entry.name.endsWith('.svelte')) {
      files.push(fullPath);
    }
  }
  
  return files;
}

async function main() {
  log('\n========================================', colors.cyan);
  log('  Svelte 5 Comprehensive Rune Fix', colors.bright + colors.cyan);
  log('========================================\n', colors.cyan);
  
  try {
    // Fix specific known issues first
    log('🔧 Fixing specific known issues...', colors.yellow);
    
    // Fix admin page
    const adminPagePath = path.join(rootDir, 'src', 'routes', 'admin', '+page.svelte');
    await fixSpecificFile(adminPagePath, [
      (content) => {
        // Fix non-reactive state declarations
        content = content.replace('let systemMetrics = {', 'let systemMetrics = $state({');
        content = content.replace('let recentActivity: Array<{', 'let recentActivity = $state<Array<{');
        content = content.replace('}> = [];', '}>([]);');
        content = content.replace('let isLoadingMetrics = true;', 'let isLoadingMetrics = $state(true);');
        // Fix all click handlers
        content = content.replace(/\bclick=/g, 'on:click=');
        return content;
      }
    ]);
    
    // Fix admin/users page - find and fix malformed form tag
    const adminUsersPath = path.join(rootDir, 'src', 'routes', 'admin', 'users', '+page.svelte');
    await fixSpecificFile(adminUsersPath, [
      (content) => {
        // Count opening and closing form tags
        const openForms = (content.match(/<form/g) || []).length;
        const closeForms = (content.match(/<\/form>/g) || []).length;
        
        if (closeForms > openForms) {
          // Remove extra closing form tag around line 421
          const lines = content.split('\n');
          for (let i = 415; i < 425 && i < lines.length; i++) {
            if (lines[i].includes('</form>')) {
              // Check if there's no corresponding opening form nearby
              const nearbyContent = lines.slice(Math.max(0, i - 50), i).join('\n');
              if (!nearbyContent.includes('<form')) {
                lines[i] = lines[i].replace('</form>', '');
                log('    Removed orphaned </form> tag at line ' + (i + 1), colors.yellow);
                break;
              }
            }
          }
          content = lines.join('\n');
        }
        return content;
      }
    ]);
    
    // Fix showcase layout
    const showcaseLayoutPath = path.join(rootDir, 'src', 'routes', 'showcase', '+layout.svelte');
    await fixSpecificFile(showcaseLayoutPath, [
      (content) => {
        // Fix deprecated slot usage
        content = content.replace('{@/* TODO: manual review – previously <slot /> */}', '{@render children?.()}');
        return content;
      }
    ]);
    
    // Now fix all Svelte files
    log('\n📁 Scanning for all Svelte files...', colors.blue);
    const srcDir = path.join(rootDir, 'src');
    const files = await getAllSvelteFiles(srcDir);
    log(`  Found ${files.length} Svelte files\n`, colors.blue);
    
    log('🔧 Applying Svelte 5 fixes...', colors.yellow);
    let fixedCount = 0;
    
    for (const file of files) {
      const fixed = await fixSvelteFile(file);
      if (fixed) fixedCount++;
    }
    
    log('\n========================================', colors.cyan);
    log(`  ✅ Fixed ${fixedCount} files`, colors.green);
    log('========================================\n', colors.cyan);
    
    log('📝 Additional recommendations:', colors.yellow);
    log('  1. Run "npx svelte-kit sync" to regenerate types', colors.white);
    log('  2. Run "npm run dev" to test the application', colors.white);
    log('  3. Check remaining errors with "npm run check"', colors.white);
    
  } catch (error) {
    log(`\n❌ Error: ${error.message}`, colors.red);
    process.exit(1);
  }
}

main().catch(console.error);
