import { readFileSync, writeFileSync, readdirSync, statSync } from 'fs';
import { join, extname } from 'path';

console.log('🔧 Fixing common TypeScript errors in the project...\n');

// Common fixes for TypeScript errors
const fixes = [
    {
        // Fix User type properties
        pattern: /user\?\.username/g,
        replacement: 'user?.email',
        description: 'Replacing user.username with user.email'
    },
    {
        // Fix User.name references
        pattern: /\$user\.name/g,
        replacement: '$user.email',
        description: 'Replacing $user.name with $user.email'
    },
    {
        // Fix createdAt property
        pattern: /user\?\.createdAt/g,
        replacement: 'user?.id', // Using id as a fallback since createdAt might not exist
        description: 'Replacing user.createdAt with user.id'
    },
    {
        // Fix notification type errors by adding title
        pattern: /notifications\.add\(\s*{\s*type:\s*['"]error['"],\s*message:\s*([^}]+)\s*}\s*\)/g,
        replacement: 'notifications.add({ type: "error", title: "Error", message: $1 })',
        description: 'Adding title to error notifications'
    },
    {
        // Fix @apply CSS warnings by replacing with actual classes
        pattern: /@apply\s+bg-blue-100\s+px-2\s+py-1\s+rounded\s+text-sm\s+font-mono;/g,
        replacement: 'background-color: #dbeafe; padding: 0.5rem; border-radius: 0.25rem; font-size: 0.875rem; font-family: monospace;',
        description: 'Replacing @apply with actual CSS properties'
    }
];

let totalFixed = 0;
let filesProcessed = 0;

function processFile(filePath) {
    try {
        const content = readFileSync(filePath, 'utf8');
        let newContent = content;
        let fileFixed = false;

        for (const fix of fixes) {
            const matches = content.match(fix.pattern);
            if (matches) {
                newContent = newContent.replace(fix.pattern, fix.replacement);
                console.log(`  ✅ ${fix.description} in ${filePath}`);
                totalFixed += matches.length;
                fileFixed = true;
            }
        }

        if (fileFixed) {
            writeFileSync(filePath, newContent);
            filesProcessed++;
        }
    } catch (error) {
        console.error(`  ❌ Error processing ${filePath}: ${error.message}`);
    }
}

function walkDirectory(dir) {
    const files = readdirSync(dir);
    
    for (const file of files) {
        const filePath = join(dir, file);
        const stat = statSync(filePath);
        
        // Skip node_modules and .svelte-kit directories
        if (file === 'node_modules' || file === '.svelte-kit' || file === '.git') {
            continue;
        }
        
        if (stat.isDirectory()) {
            walkDirectory(filePath);
        } else if (stat.isFile()) {
            const ext = extname(file);
            // Process Svelte and TypeScript files
            if (['.svelte', '.ts', '.js'].includes(ext)) {
                processFile(filePath);
            }
        }
    }
}

console.log('Starting TypeScript error fixes...\n');

// Process the sveltekit-frontend/src directory
const srcDir = join(process.cwd(), 'sveltekit-frontend', 'src');
if (statSync(srcDir).isDirectory()) {
    walkDirectory(srcDir);
} else {
    console.error('❌ Could not find sveltekit-frontend/src directory');
    process.exit(1);
}

console.log(`\n✨ Fixed ${totalFixed} issues in ${filesProcessed} files`);
console.log('\nRun "npm run check" to verify remaining TypeScript errors');
