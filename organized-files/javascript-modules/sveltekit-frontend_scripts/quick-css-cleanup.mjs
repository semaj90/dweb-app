#!/usr/bin/env zx

/**
 * Quick CSS Cleanup - Remove common unused selectors
 */

import 'zx/globals'

// Common unused selectors based on previous error patterns
const UNUSED_SELECTORS = [
  'profile-content',
  'stats-grid', 
  'stat-card',
  'form-section',
  'toolbar-button',
  'dialog-content',
  'canvas-container',
  'evidence-grid',
  'chat-container',
  'loading-spinner',
  'error-message',
  'success-message',
  'navigation-item',
  'sidebar-content',
  'modal-overlay'
]

async function main() {
  console.log(chalk.cyan('🧹 Quick CSS Cleanup'))
  
  // Get all Svelte files
  const files = await glob('src/**/*.svelte')
  
  let totalRemoved = 0
  let filesModified = 0
  
  for (const file of files) {
    try {
      let content = await fs.readFile(file, 'utf8')
      let fileChanged = false
      
      // Check for unused selectors
      for (const selector of UNUSED_SELECTORS) {
        // Find CSS rule
        const cssRuleRegex = new RegExp(`\\s*\\.${selector}\\s*\\{[^}]*\\}\\s*`, 'g')
        const matches = content.match(cssRuleRegex)
        
        if (matches) {
          // Check if selector is actually used in template
          const usageRegex = new RegExp(`class=["'][^"']*${selector}[^"']*["']|class:${selector}`, 'g')
          
          if (!usageRegex.test(content)) {
            // Remove unused CSS rule
            content = content.replace(cssRuleRegex, '\n')
            totalRemoved += matches.length
            fileChanged = true
          }
        }
      }
      
      // Clean up extra whitespace
      if (fileChanged) {
        content = content.replace(/\n\s*\n\s*\n/g, '\n\n')
        await fs.writeFile(file, content, 'utf8')
        filesModified++
        console.log(chalk.green(`✨ Cleaned: ${file.replace(process.cwd(), '.')}`))
      }
      
    } catch (error) {
      console.error(chalk.red(`💥 Error: ${file}:`), error.message)
    }
  }
  
  console.log(chalk.cyan(`\n🎉 Removed ${totalRemoved} unused selectors from ${filesModified} files`))
  
  // Verify
  console.log(chalk.blue('🔍 Quick verification...'))
  await $`npm run check:ultra-fast`
}

main().catch(console.error)