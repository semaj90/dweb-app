// Comprehensive AutoSolve script for Svelte 5 and bits-ui compatibility
// Fixes TypeScript errors and upgrades components to modern patterns

import { promises as fs } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { glob } from 'glob';
import chalk from 'chalk';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Patterns to fix for Svelte 5 compatibility
const SVELTE_5_FIXES = {
  // Convert old props to $props rune
  propsPattern: {
    regex: /export\s+let\s+(\w+)(?:\s*:\s*([^=;]+))?(?:\s*=\s*([^;]+))?;?/g,
    replacement: (match, name, type, defaultValue) => {
      if (defaultValue) {
        return `// Converted to $props\n\t// ${name}${type ? `: ${type}` : ''} = ${defaultValue}`;
      }
      return `// Converted to $props\n\t// ${name}${type ? `: ${type}` : ''}`;
    }
  },

  // Fix $props() syntax
  propsInterface: {
    regex: /let\s*{\s*([^}]+)\s*}\s*=\s*\$props\(\)/g,
    replacement: (match, props) => {
      // Parse props and create proper interface
      const propsList = props.split(',').map(p => p.trim());
      const propsWithTypes = propsList.map(prop => {
        const [name, defaultValue] = prop.split('=').map(s => s.trim());
        return `${name}${defaultValue ? ` = ${defaultValue}` : ''}`;
      }).join(',\n\t\t');

      return `let {\n\t\t${propsWithTypes}\n\t}: Props = $props()`;
    }
  },

  // Convert reactive statements to $derived
  derivedPattern: {
    regex: /\$:\s*(.+)/g,
    replacement: (match, statement) => {
      // Check if it's a simple assignment
      if (statement.includes('=')) {
        const [variable, expression] = statement.split('=').map(s => s.trim());
        return `let ${variable} = $derived(${expression})`;
      }
      return `$effect(() => { ${statement} })`;
    }
  },

  // Fix $state usage
  statePattern: {
    regex: /let\s+(\w+)\s*=\s*\$state<([^>]+)>\((.*?)\)/g,
    replacement: 'let $1 = $state<$2>($3)'
  },

  // Fix bits-ui imports
  bitsUIImports: {
    regex: /import\s*{\s*([^}]+)\s*}\s*from\s*['"]bits-ui['"]/g,
    replacement: (match, imports) => {
      // Ensure proper component imports
      const componentImports = imports.split(',')
        .map(imp => imp.trim())
        .filter(imp => imp)
        .join(', ');
      return `import { ${componentImports} } from 'bits-ui'`;
    }
  }
};

// TypeScript error fixes
const TYPESCRIPT_FIXES = {
  // Fix missing type imports
  missingTypes: {
    pattern: /Cannot find name '(\w+)'/,
    fix: (match, typeName) => {
      const typeImports = {
        'Props': "interface Props {}",
        'Document': "import type { Document } from '$lib/types'",
        'RequestHandler': "import type { RequestHandler } from './$types'"
      };
      return typeImports[typeName] || '';
    }
  },

  // Fix async function types
  asyncFunctions: {
    pattern: /async\s+function\s+(\w+)(?:<[^>]+>)?\s*\([^)]*\)(?:\s*:\s*Promise<[^>]+>)?/g,
    fix: (match) => {
      if (!match.includes('Promise<')) {
        return match.replace(/\)/, '): Promise<void>');
      }
      return match;
    }
  },

  // Fix optional chaining
  optionalChaining: {
    pattern: /(\w+)\.(\w+)(?!\?)/g,
    fix: (match, obj, prop) => {
      const safeObjects = ['console', 'window', 'document', 'Math', 'JSON'];
      if (safeObjects.includes(obj)) return match;
      return `${obj}?.${prop}`;
    }
  }
};

class Svelte5AutoSolver {
  constructor() {
    /** @type {number} */ this.fixedCount = 0;
    /** @type {number} */ this.errorCount = 0;
    /** @type {Set<string>} */ this.processedFiles = new Set();
  }

  async run() {
    console.log(chalk.cyan('\n🚀 Starting Svelte 5 & bits-ui AutoSolve...\n'));

    // Find all Svelte and TypeScript files
    const svelteFiles = await glob('src/**/*.svelte', {
      cwd: path.join(__dirname, '..'),
      absolute: true
    });

    const tsFiles = await glob('src/**/*.ts', {
      cwd: path.join(__dirname, '..'),
      absolute: true
    });

    console.log(chalk.gray(`Found ${svelteFiles.length} Svelte files and ${tsFiles.length} TypeScript files\n`));

    // Process Svelte files
    for (const file of svelteFiles) {
      await this.processSvelteFile(file);
    }

    // Process TypeScript files
    for (const file of tsFiles) {
      await this.processTypeScriptFile(file);
    }

    // Fix specific component issues
    await this.fixSpecificComponents();

    // Update bits-ui components
    await this.updateBitsUIComponents();

    console.log(chalk.green(`\n✅ AutoSolve Complete!`));
    console.log(chalk.cyan(`   Fixed: ${this.fixedCount} issues`));
    console.log(chalk.yellow(`   Remaining: ${this.errorCount} issues`));
  }

  async processSvelteFile(filePath) {
    try {
      let content = await fs.readFile(filePath, 'utf8');
      const originalContent = content;
      let changed = false;

      // Apply Svelte 5 fixes
      for (const [fixName, fix] of Object.entries(SVELTE_5_FIXES)) {
        const regex = fix.regex;
        if (regex.test(content)) {
          content = content.replace(regex, fix.replacement as any);
          changed = true;
          this.fixedCount++;
        }
      }

      // Fix script tag issues
      content = this.fixScriptTags(content);

      // Fix $props() usage
      content = this.fixPropsUsage(content);

      // Fix $derived usage
      content = this.fixDerivedUsage(content);

      if (changed) {
        await fs.writeFile(filePath, content, 'utf8');
        console.log(chalk.green(`✓ Fixed ${path.basename(filePath)}`));
      }

      this.processedFiles.add(filePath);
    } catch (error) {
      console.error(chalk.red(`✗ Error processing ${filePath}:`), error);
      this.errorCount++;
    }
  }

  async processTypeScriptFile(filePath) {
    try {
      let content = await fs.readFile(filePath, 'utf8');
      const originalContent = content;
      let changed = false;

      // Apply TypeScript fixes
      for (const [fixName, fix] of Object.entries(TYPESCRIPT_FIXES)) {
        const regex = new RegExp(fix.pattern);
        if (regex.test(content)) {
          content = content.replace(regex, fix.fix as any);
          changed = true;
          this.fixedCount++;
        }
      }

      // Add missing imports
      content = this.addMissingImports(content);

      // Fix type annotations
      content = this.fixTypeAnnotations(content);

      if (changed) {
        await fs.writeFile(filePath, content, 'utf8');
        console.log(chalk.green(`✓ Fixed ${path.basename(filePath)}`));
      }

      this.processedFiles.add(filePath);
    } catch (error) {
      console.error(chalk.red(`✗ Error processing ${filePath}:`), error);
      this.errorCount++;
    }
  }

  fixScriptTags(content) {
    // Fix malformed script tags
    content = content.replace(/<script lang="ts">\s*>;/g, '<script lang="ts">');
    content = content.replace(/^\s*}\s*$/gm, (match, offset) => {
      // Check if this is a dangling closing brace
      const before = content.substring(0, offset);
      const openBraces = (before.match(/{/g) || []).length;
      const closeBraces = (before.match(/}/g) || []).length;
      if (closeBraces >= openBraces) {
        return ''; // Remove dangling brace
      }
      return match;
    });
    return content;
  }

  fixPropsUsage(content) {
    // Fix $props() syntax with proper interface
    const propsMatch = content.match(/let\s*{([^}]+)}\s*=\s*\$props\(\);?/);
    if (propsMatch) {
      const props = propsMatch[1];
      const propsInterface = this.generatePropsInterface(props);

      // Add interface if not present
      if (!content.includes('interface Props')) {
        const scriptIndex = content.indexOf('<script');
        const scriptEndIndex = content.indexOf('>', scriptIndex);
        content = content.slice(0, scriptEndIndex + 1) +
                 '\n\t' + propsInterface + '\n' +
                 content.slice(scriptEndIndex + 1);
      }

      // Fix props destructuring
      content = content.replace(
        /let\s*{([^}]+)}\s*=\s*\$props\(\);?/,
        'let { $1 }: Props = $props()'
      );
    }
    return content;
  }

  fixDerivedUsage(content) {
    // Fix $derived syntax
    content = content.replace(
      /let\s+(\w+)\s*=\s*\$derived\(([^;]+)\);{2,}/g,
      'let $1 = $derived($2);'
    );

    // Fix broken $derived with object literals
    content = content.replace(
      /let\s+(\w+)\s*=\s*\$derived\({}\);/g,
      'let $1 = $derived({});'
    );

    return content;
  }

  generatePropsInterface(propsString) {
    const props = propsString.split(',').map(p => p.trim());
    const propDefinitions = props.map(prop => {
      const [name, defaultValue] = prop.split('=').map(s => s.trim());
      let type = 'any'; // Default type

      // Infer type from default value
      if (defaultValue) {
        if (defaultValue === 'true' || defaultValue === 'false') type = 'boolean';
        else if (defaultValue.startsWith('"') || defaultValue.startsWith("'")) type = 'string';
        else if (!isNaN(Number(defaultValue))) type = 'number';
        else if (defaultValue === '[]') type = 'any[]';
        else if (defaultValue === '{}') type = 'Record<string, any>';
        else if (defaultValue.includes('=>')) type = 'Function';
      }

      return `\t\t${name}${defaultValue ? '?' : ''}: ${type};`;
    }).join('\n');

    return `interface Props {\n${propDefinitions}\n\t}`;
  }

  addMissingImports(content) {
    // Add common missing imports
    const imports = {
      'drizzle-orm': "import { eq, sql, and, or } from 'drizzle-orm';",
      '@sveltejs/kit': "import { json, error } from '@sveltejs/kit';",
      'xstate': "import { createMachine, interpret } from 'xstate';",
      'bits-ui': "import { Button, Dialog, Select } from 'bits-ui';",
      'lucide-svelte': "import { Upload, FileText, Loader2, Check, X } from 'lucide-svelte';"
    };

    for (const [pkg, importStatement] of Object.entries(imports)) {
      if (content.includes(pkg.replace('-', '')) && !content.includes(`from '${pkg}'`)) {
        const firstImport = content.indexOf('import');
        if (firstImport !== -1) {
          content = importStatement + '\n' + content;
        }
      }
    }

    return content;
  }

  fixTypeAnnotations(content) {
    // Fix function type annotations
    content = content.replace(
      /function\s+(\w+)\s*\(([^)]*)\)\s*{/g,
      (match, name, params) => {
        if (!match.includes(':')) {
          return `function ${name}(${params}): void {`;
        }
        return match;
      }
    );

    // Fix arrow function types
    content = content.replace(
      /const\s+(\w+)\s*=\s*\(([^)]*)\)\s*=>/g,
      (match, name, params) => {
        if (!params.includes(':')) {
          return `const ${name} = (${params}: any) =>`;
        }
        return match;
      }
    );

    return content;
  }

  async fixSpecificComponents() {
    // Fix specific known problematic components
    const componentsToFix = [
      'src/lib/components/NierNavigation.svelte',
      'src/lib/components/ai/AIDropdown.svelte',
      'src/lib/components/canvas/EnhancedCanvasEditor.svelte',
      'src/lib/components/analysis/MultiAgentAnalysisCard.svelte'
    ];

    for (const componentPath of componentsToFix) {
      const fullPath = path.join(__dirname, '..', componentPath);
      try {
        let content = await fs.readFile(fullPath, 'utf8');

        // Apply specific fixes for each component
        if (componentPath.includes('NierNavigation')) {
          content = this.fixNierNavigation(content);
        } else if (componentPath.includes('AIDropdown')) {
          content = this.fixAIDropdown(content);
        } else if (componentPath.includes('EnhancedCanvasEditor')) {
          content = this.fixEnhancedCanvasEditor(content);
        } else if (componentPath.includes('MultiAgentAnalysisCard')) {
          content = this.fixMultiAgentAnalysisCard(content);
        }

        await fs.writeFile(fullPath, content, 'utf8');
        console.log(chalk.green(`✓ Fixed specific component: ${path.basename(fullPath)}`));
        this.fixedCount++;
      } catch (error) {
        console.log(chalk.yellow(`⚠ Component not found: ${componentPath}`));
      }
    }
  }

  fixNierNavigation(content) {
    // Fix the malformed script tag
    content = content.replace(/<script lang="ts">\s*>;[\s\S]*?}/, '<script lang="ts">');

    // Add proper imports and props
    const properScript = `<script lang="ts">
\timport { page } from '$app/state';
\timport { Button } from 'bits-ui';
\t
\tinterface Props {
\t\tbrand?: string;
\t\tversion?: string;
\t\tlinks?: Array<{
\t\t\thref: string;
\t\t\tlabel: string;
\t\t\ticon?: any;
\t\t}>;
\t}
\t
\tlet {
\t\tbrand = 'Legal AI',
\t\tversion = 'v4.0',
\t\tlinks = []
\t}: Props = $props();
</script>`;

    // Replace the entire script section
    const scriptStart = content.indexOf('<script');
    const scriptEnd = content.indexOf('</script>') + 9;
    if (scriptStart !== -1 && scriptEnd > scriptStart) {
      content = content.slice(0, scriptStart) + properScript + content.slice(scriptEnd);
    }

    return content;
  }

  fixAIDropdown(content) {
    // Fix malformed function types
    content = content.replace(
      /on(\w+)\s*=\s*>\s*void\s*=\s*\(\)\s*=>\s*{}/g,
      'on$1?: () => void'
    );

    return content;
  }

  fixEnhancedCanvasEditor(content) {
    // Fix $derived syntax with extra semicolons
    content = content.replace(
      /\$derived\(([^)]+)\);\);/g,
      '$derived($1)'
    );

    return content;
  }

  fixMultiAgentAnalysisCard(content) {
    // Fix broken object literal in $derived
    content = content.replace(
      /\$derived\({}\);[\s\S]*?}[^}]*;/g,
      (match) => {
        // Extract the object content
        const objectMatch = match.match(/{[\s\S]*?}/);
        if (objectMatch) {
          return `$derived(${objectMatch[0]})`;
        }
        return match;
      }
    );

    return content;
  }

  async updateBitsUIComponents() {
    console.log(chalk.cyan('\n📦 Updating bits-ui components...'));

    // Create a bits-ui wrapper for better integration
    const bitsUIWrapper = `// bits-ui wrapper for Svelte 5 compatibility
export { Button, Dialog, Select, Tabs, Accordion, AlertDialog, Checkbox, Collapsible, ContextMenu, DropdownMenu, Label, Menubar, NavigationMenu, Popover, Progress, RadioGroup, ScrollArea, Separator, Slider, Switch, Toggle, ToggleGroup, Tooltip } from 'bits-ui';

// Re-export with type safety
export type { ButtonProps, DialogProps, SelectProps } from 'bits-ui';
`;

    const wrapperPath = path.join(__dirname, '..', 'src/lib/components/ui/bits-ui-exports.ts');
    await fs.mkdir(path.dirname(wrapperPath), { recursive: true });
    await fs.writeFile(wrapperPath, bitsUIWrapper, 'utf8');
    console.log(chalk.green('✓ Created bits-ui wrapper'));
  }
}

// Run the AutoSolver
const solver = new Svelte5AutoSolver();
solver.run().catch(console.error);
