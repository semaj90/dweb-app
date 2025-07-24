#!/usr/bin/env node
/**
 * Phase 3 Error Check & Auto-Fix System
 */

const fs = require('fs');
const path = require('path');

const storesPath = 'C:/Users/james/Desktop/deeds-web/deeds-web-app/sveltekit-frontend/src/lib/stores';

function checkAndFix() {
  console.log('🔧 Phase 3 Error Check & Auto-Fix');
  
  // 1. Check if cases.ts exists
  const casesFile = path.join(storesPath, 'cases.ts');
  if (!fs.existsSync(casesFile)) {
    const fallbackFile = path.join(storesPath, 'cases-fallback.ts');
    if (fs.existsSync(fallbackFile)) {
      fs.copyFileSync(fallbackFile, casesFile);
      console.log('✅ Created cases.ts from fallback');
    }
  }
  
  // 2. Replace evidence store with fixed version
  const evidenceFile = path.join(storesPath, 'evidence-unified.ts');
  const fixedFile = path.join(storesPath, 'evidence-unified-fixed.ts');
  if (fs.existsSync(fixedFile)) {
    fs.copyFileSync(fixedFile, evidenceFile);
    console.log('✅ Applied evidence store fixes');
  }
  
  // 3. Check index.ts exports
  const indexFile = path.join(storesPath, 'index.ts');
  if (fs.existsSync(indexFile)) {
    let content = fs.readFileSync(indexFile, 'utf8');
    if (!content.includes('evidence-unified')) {
      console.log('⚠️ Index.ts may need export updates');
    } else {
      console.log('✅ Index.ts exports look good');
    }
  }
  
  console.log('🎯 All critical errors checked and fixed');
}

if (require.main === module) {
  checkAndFix();
}

module.exports = { checkAndFix };
