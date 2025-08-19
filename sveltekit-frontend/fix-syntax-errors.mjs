#!/usr/bin/env node

import { readFile, writeFile } from 'fs/promises';

// Files with interface syntax errors that need fixing
const interfaceErrors = [
  'src/lib/components/DocumentUploadForm.svelte',
  'src/lib/components/EvidenceAnalysisForm.svelte', 
  'src/lib/components/CaseInfoForm.svelte',
  'src/lib/components/ReviewSubmitForm.svelte',
  'src/lib/components/subcomponents/ProgressIndicator.svelte'
];

// Files with duplicate property declarations
const duplicateProps = [
  'src/lib/components/LoadingSpinner.svelte',
  'src/lib/components/subcomponents/LoadingSpinner.svelte',
  'src/lib/components/EvidencePanel.svelte'
];

// Files with end-of-file issues
const eofIssues = [
  'src/lib/components/EnhancedRAGInterface.svelte',
  'src/lib/components/AIChat.svelte'
];

async function fixInterfaceErrors() {
  console.log('🔧 Fixing interface syntax errors...');
  
  for (const file of interfaceErrors) {
    try {
      let content = await readFile(file, 'utf-8');
      
      // Fix malformed interface syntax: formData: {\r; -> formData: {
      content = content.replace(/(\w+):\s*\{\s*\\r\s*;/g, '$1: {');
      
      // Fix standalone semicolons on lines
      content = content.replace(/^\s*;\s*$/gm, '');
      
      // Fix trailing \r; patterns
      content = content.replace(/\\r\s*;/g, '');
      
      await writeFile(file, content);
      console.log(`✅ Fixed interface errors in ${file}`);
    } catch (error) {
      console.log(`⚠️ Failed to fix ${file}: ${error.message}`);
    }
  }
}

async function fixDuplicateProps() {
  console.log('🔧 Fixing duplicate property declarations...');
  
  for (const file of duplicateProps) {
    try {
      let content = await readFile(file, 'utf-8');
      
      // Remove duplicate export let declarations (keep first occurrence)
      const lines = content.split('\n');
      const seenProps = new Set();
      const fixedLines = [];
      
      for (let line of lines) {
        // Check for property declarations
        const propMatch = line.match(/\s*let\s*\{\s*(\w+)\s*=/);
        if (propMatch) {
          const propName = propMatch[1];
          if (seenProps.has(propName)) {
            // Skip duplicate
            console.log(`  Removing duplicate prop: ${propName}`);
            continue;
          }
          seenProps.add(propName);
        }
        fixedLines.push(line);
      }
      
      content = fixedLines.join('\n');
      
      await writeFile(file, content);
      console.log(`✅ Fixed duplicate props in ${file}`);
    } catch (error) {
      console.log(`⚠️ Failed to fix ${file}: ${error.message}`);
    }
  }
}

async function fixEofIssues() {
  console.log('🔧 Fixing end-of-file issues...');
  
  for (const file of eofIssues) {
    try {
      let content = await readFile(file, 'utf-8');
      
      // Ensure proper closing tags
      if (!content.includes('</script>') && content.includes('<script')) {
        content += '\n</script>\n';
      }
      
      // Add missing closing markup if needed
      const scriptEnd = content.lastIndexOf('</script>');
      if (scriptEnd !== -1) {
        const afterScript = content.substring(scriptEnd + 9).trim();
        if (!afterScript) {
          content += '\n<div><!-- Component content goes here --></div>\n';
        }
      }
      
      await writeFile(file, content);
      console.log(`✅ Fixed EOF issues in ${file}`);
    } catch (error) {
      console.log(`⚠️ Failed to fix ${file}: ${error.message}`);
    }
  }
}

async function fixSpecificErrors() {
  console.log('🔧 Fixing specific syntax errors...');
  
  // Fix Toolbar.svelte specific error
  try {
    const toolbarFile = 'src/lib/components/Toolbar.svelte';
    let content = await readFile(toolbarFile, 'utf-8');
    
    // Fix the malformed line: onformatToggled?.()[formatType] });
    content = content.replace(/onformatToggled\?\.\(\)\[formatType\]\s*\}\);/, 'onformatToggled?.(formatType);');
    
    await writeFile(toolbarFile, content);
    console.log(`✅ Fixed Toolbar.svelte syntax error`);
  } catch (error) {
    console.log(`⚠️ Failed to fix Toolbar.svelte: ${error.message}`);
  }
  
  // Fix ContextMenuContent.svelte specific error
  try {
    const contextFile = 'src/lib/components/ui/ContextMenuContent.svelte';
    let content = await readFile(contextFile, 'utf-8');
    
    // Fix script tag closure issue - ensure proper script block
    if (content.includes('const contextMenu') && !content.includes('</script>')) {
      const lines = content.split('\n');
      let inScript = false;
      let scriptStartIndex = -1;
      
      for (let i = 0; i < lines.length; i++) {
        if (lines[i].includes('<script')) {
          inScript = true;
          scriptStartIndex = i;
        }
        if (inScript && lines[i].trim() === '' && i > scriptStartIndex + 5) {
          // Insert </script> tag
          lines.splice(i, 0, '</script>');
          break;
        }
      }
      
      content = lines.join('\n');
      await writeFile(contextFile, content);
      console.log(`✅ Fixed ContextMenuContent.svelte script closure`);
    }
  } catch (error) {
    console.log(`⚠️ Failed to fix ContextMenuContent.svelte: ${error.message}`);
  }
}

async function fixAllSyntaxErrors() {
  console.log('🚀 Starting comprehensive syntax error fixes...\n');
  
  await fixInterfaceErrors();
  console.log();
  
  await fixDuplicateProps();
  console.log();
  
  await fixEofIssues();
  console.log();
  
  await fixSpecificErrors();
  
  console.log('\n🎉 Syntax error fixes complete!');
}

fixAllSyntaxErrors().catch(console.error);