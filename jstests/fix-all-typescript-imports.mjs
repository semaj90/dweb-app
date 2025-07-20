import { readFileSync, writeFileSync, readdirSync, statSync, existsSync } from 'fs';
import { join, dirname, extname } from 'path';

console.log('🔧 Comprehensive TypeScript and Import Fix Script\n');

// Comprehensive fixes for all common issues
const fixes = [
    // User type property fixes
    {
        pattern: /user\?\.username/g,
        replacement: 'user?.email',
        description: 'Fix User.username to User.email'
    },
    {
        pattern: /\$user\.name\b/g,
        replacement: '$user.email',
        description: 'Fix $user.name to $user.email'
    },
    {
        pattern: /user\?\.createdAt/g,
        replacement: 'new Date()',
        description: 'Fix user.createdAt reference'
    },
    // Notification fixes
    {
        pattern: /notifications\.add\(\s*{\s*type:\s*['"](\w+)['"],\s*message:\s*([^}]+)\s*}\s*\)/g,
        replacement: 'notifications.add({ type: "$1", title: "$1", message: $2 })',
        description: 'Add title to notifications'
    },
    // Import path fixes for UI components
    {
        pattern: /from\s+['"]\.\.\/\$lib\/components\/ui\//g,
        replacement: 'from "$lib/components/ui/',
        description: 'Fix relative UI component imports'
    },
    {
        pattern: /from\s+['"]\$lib\/components\/ui\/([^'"]+)\.svelte['"]/g,
        replacement: 'from "$lib/components/ui/$1/index.js"',
        description: 'Fix UI component import extensions'
    },
    // Database import fixes
    {
        pattern: /from\s+['"]\$lib\/server\/db\.js['"]/g,
        replacement: 'from "$lib/server/db/index.js"',
        description: 'Fix database import paths'
    },
    {
        pattern: /from\s+['"]\$lib\/server\/db\/schema\.js['"]/g,
        replacement: 'from "$lib/server/db/unified-schema.js"',
        description: 'Fix schema import to unified-schema'
    },
    // Fix CSS @apply directives
    {
        pattern: /@apply\s+([^;]+);/g,
        replacement: (match, classes) => {
            const cssMap = {
                'bg-blue-100': 'background-color: #dbeafe',
                'px-2': 'padding-left: 0.5rem; padding-right: 0.5rem',
                'py-1': 'padding-top: 0.25rem; padding-bottom: 0.25rem',
                'rounded': 'border-radius: 0.25rem',
                'text-sm': 'font-size: 0.875rem',
                'font-mono': 'font-family: monospace',
            };
            
            const cssRules = classes.split(/\s+/)
                .map(cls => cssMap[cls] || `/* ${cls} */`)
                .filter(Boolean)
                .join('; ');
            
            return cssRules + ';';
        },
        description: 'Convert @apply to CSS properties'
    },
    // Fix line-clamp CSS
    {
        pattern: /-webkit-line-clamp:\s*(\d+);/g,
        replacement: '-webkit-line-clamp: $1; line-clamp: $1;',
        description: 'Add standard line-clamp property'
    }
];

// Additional import fixes for common patterns
const importFixes = [
    {
        // Fix Button imports
        from: /import\s+{\s*Button\s*}\s+from\s+['"]\.\.\/ui\/button['"]/g,
        to: 'import { Button } from "$lib/components/ui/button"'
    },
    {
        // Fix Card imports
        from: /import\s+{\s*Card[^}]*}\s+from\s+['"][^'"]*ui\/card['"]/g,
        to: 'import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "$lib/components/ui/card"'
    },
    {
        // Fix store imports
        from: /import\s+{\s*(\w+)\s*}\s+from\s+['"]\$lib\/stores\.js['"]/g,
        to: 'import { $1 } from "$lib/stores"'
    }
];

let totalFixed = 0;
let filesProcessed = 0;
const errors = [];

function ensureDirectoryExists(filePath) {
    const dir = dirname(filePath);
    if (!existsSync(dir)) {
        console.log(`Creating directory: ${dir}`);
        require('fs').mkdirSync(dir, { recursive: true });
    }
}

function processFile(filePath) {
    try {
        const content = readFileSync(filePath, 'utf8');
        let newContent = content;
        let fileFixed = false;

        // Apply general fixes
        for (const fix of fixes) {
            const matches = content.match(fix.pattern);
            if (matches) {
                newContent = newContent.replace(fix.pattern, fix.replacement);
                console.log(`  ✅ ${fix.description} in ${filePath}`);
                totalFixed += matches.length;
                fileFixed = true;
            }
        }

        // Apply import-specific fixes
        for (const fix of importFixes) {
            if (fix.from.test(newContent)) {
                newContent = newContent.replace(fix.from, fix.to);
                console.log(`  ✅ Fixed imports in ${filePath}`);
                fileFixed = true;
                totalFixed++;
            }
        }

        // Fix missing imports for common components
        if (filePath.endsWith('.svelte')) {
            // Check if Button is used but not imported
            if (/&lt;Button\b/.test(newContent) && !/import.*Button.*from/.test(newContent)) {
                const scriptMatch = newContent.match(/&lt;script[^>]*>/);
                if (scriptMatch) {
                    const insertPos = scriptMatch.index + scriptMatch[0].length;
                    newContent = newContent.slice(0, insertPos) + 
                        '\nimport { Button } from "$lib/components/ui/button";\n' +
                        newContent.slice(insertPos);
                    console.log(`  ✅ Added missing Button import in ${filePath}`);
                    fileFixed = true;
                    totalFixed++;
                }
            }
        }

        if (fileFixed) {
            writeFileSync(filePath, newContent);
            filesProcessed++;
        }
    } catch (error) {
        errors.push({ file: filePath, error: error.message });
        console.error(`  ❌ Error processing ${filePath}: ${error.message}`);
    }
}

function walkDirectory(dir) {
    try {
        const files = readdirSync(dir);
        
        for (const file of files) {
            const filePath = join(dir, file);
            const stat = statSync(filePath);
            
            // Skip directories we shouldn't process
            if (file === 'node_modules' || file === '.svelte-kit' || file === '.git' || file === 'build') {
                continue;
            }
            
            if (stat.isDirectory()) {
                walkDirectory(filePath);
            } else if (stat.isFile()) {
                const ext = extname(file);
                // Process Svelte, TypeScript, and JavaScript files
                if (['.svelte', '.ts', '.js'].includes(ext) && !file.endsWith('.d.ts')) {
                    processFile(filePath);
                }
            }
        }
    } catch (error) {
        console.error(`Error walking directory ${dir}: ${error.message}`);
    }
}

// Create missing UI component exports if needed
function createMissingUIExports() {
    const uiComponentsPath = join(process.cwd(), 'sveltekit-frontend', 'src', 'lib', 'components', 'ui');
    const components = ['button', 'card', 'input', 'label', 'textarea', 'select', 'checkbox', 'badge'];
    
    for (const component of components) {
        const componentPath = join(uiComponentsPath, component);
        const indexPath = join(componentPath, 'index.js');
        const svelteFilePath = join(componentPath, `${component}.svelte`);
        
        if (existsSync(svelteFilePath) && !existsSync(indexPath)) {
            ensureDirectoryExists(indexPath);
            const exportContent = `export { default as ${component.charAt(0).toUpperCase() + component.slice(1)} } from './${component}.svelte';\n`;
            writeFileSync(indexPath, exportContent);
            console.log(`  ✅ Created index.js for ${component} component`);
        }
    }
}

console.log('🚀 Starting comprehensive TypeScript and import fixes...\n');

// First, create any missing UI component exports
createMissingUIExports();

// Process the sveltekit-frontend/src directory
const srcDir = join(process.cwd(), 'sveltekit-frontend', 'src');
if (existsSync(srcDir)) {
    walkDirectory(srcDir);
} else {
    console.error('❌ Could not find sveltekit-frontend/src directory');
    process.exit(1);
}

console.log(`\n✨ Fixed ${totalFixed} issues in ${filesProcessed} files`);

if (errors.length > 0) {
    console.log('\n⚠️  Errors encountered:');
    errors.forEach(({ file, error }) => {
        console.log(`  - ${file}: ${error}`);
    });
}

console.log('\n📝 Next steps:');
console.log('1. Run "cd sveltekit-frontend && npm run check" to verify TypeScript errors');
console.log('2. Run "npm run dev" to start the development server');
console.log('3. Check for any remaining import errors in the browser console');
