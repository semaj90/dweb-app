#!/usr/bin/env node

import fs from 'fs';
import { glob } from 'glob';

// Find and fix malformed script tag imports
async function fixMalformedImports() {
  console.log('🔧 Fixing malformed script imports...\n');
  
  const files = await glob('src/**/*.svelte');
  console.log(`Found ${files.length} Svelte files to check\n`);
  
  let fixedCount = 0;
  
  for (const file of files) {
    try {
      const content = fs.readFileSync(file, 'utf-8');
      
      // Check for malformed imports like: <script lang="ts">import
      if (content.includes('<script lang="ts">import')) {
        // Fix by adding proper line breaks
        const fixed = content
          .replace(/<script lang="ts">import/g, '<script lang="ts">\n  import')
          .replace(/from "svelte";import/g, 'from "svelte";\n  import')
          .replace(/from 'svelte';import/g, "from 'svelte';\n  import");
        
        if (fixed !== content) {
          fs.writeFileSync(file, fixed);
          console.log(`✅ ${file}: Fixed malformed imports`);
          fixedCount++;
        }
      }
    } catch (error) {
      console.error(`❌ ${file}: ${error.message}`);
    }
  }
  
  console.log(`\n✨ Summary: Fixed ${fixedCount} files with malformed imports`);
}

fixMalformedImports().catch(console.error);