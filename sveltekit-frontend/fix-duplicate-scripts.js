import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Function to remove duplicate script tags containing CommonProps interface
function removeDuplicateScripts(filePath) {
  try {
    const content = fs.readFileSync(filePath, 'utf8');
    
    // Pattern to match the duplicate script tag with CommonProps
    const duplicatePattern = /<script lang="ts">\s*import type \{ CommonProps \} from '\$lib\/types\/common-props';\s*interface Props extends CommonProps \{\}\s*<\/script>/g;
    
    // Remove all instances of the duplicate pattern
    const cleaned = content.replace(duplicatePattern, '');
    
    // Only write if content changed
    if (cleaned !== content) {
      fs.writeFileSync(filePath, cleaned, 'utf8');
      console.log(`Fixed: ${filePath}`);
      return true;
    }
    return false;
  } catch (error) {
    console.error(`Error processing ${filePath}:`, error.message);
    return false;
  }
}

// Recursively find all .svelte files
function findSvelteFiles(dir) {
  let results = [];
  const list = fs.readdirSync(dir);
  
  list.forEach(file => {
    const filePath = path.join(dir, file);
    const stat = fs.statSync(filePath);
    
    if (stat && stat.isDirectory()) {
      results = results.concat(findSvelteFiles(filePath));
    } else if (file.endsWith('.svelte')) {
      results.push(filePath);
    }
  });
  
  return results;
}

// Main execution
const srcDir = path.join(__dirname, 'src');
const svelteFiles = findSvelteFiles(srcDir);

console.log(`Found ${svelteFiles.length} Svelte files`);

let fixedCount = 0;
svelteFiles.forEach(file => {
  if (removeDuplicateScripts(file)) {
    fixedCount++;
  }
});

console.log(`Fixed ${fixedCount} files with duplicate script tags`);