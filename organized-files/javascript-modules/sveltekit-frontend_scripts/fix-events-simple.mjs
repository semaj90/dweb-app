#!/usr/bin/env zx

/**
 * Simple Event Directive Fixes
 * Fast replacement of on:click -> onclick patterns
 */

import 'zx/globals'

const EVENT_MAPPINGS = {
  'on:click=': 'onclick=',
  'on:submit=': 'onsubmit=',
  'on:change=': 'onchange=',
  'on:input=': 'oninput=',
  'on:focus=': 'onfocus=',
  'on:blur=': 'onblur=',
  'on:keydown=': 'onkeydown=',
  'on:keyup=': 'onkeyup=',
  'on:mouseenter=': 'onmouseenter=',
  'on:mouseleave=': 'onmouseleave='
}

async function main() {
  console.log(chalk.cyan('🔧 Simple Event Directive Fixes'))
  
  // Get Svelte files with deprecated events
  const result = await $`find src -name "*.svelte" -exec grep -l "on:" {} \\;`.quiet()
  const files = result.stdout.trim().split('\n').filter(f => f)
  
  console.log(chalk.yellow(`📋 Processing ${files.length} files with deprecated events`))
  
  let totalFixed = 0
  let totalReplacements = 0
  
  for (const file of files) {
    try {
      let content = await fs.readFile(file, 'utf8')
      let fileChanged = false
      
      // Apply replacements
      for (const [oldEvent, newEvent] of Object.entries(EVENT_MAPPINGS)) {
        const before = content
        content = content.replaceAll(oldEvent, newEvent)
        
        if (content !== before) {
          const matches = (before.match(new RegExp(oldEvent.replace(':', '\\:'), 'g')) || []).length
          totalReplacements += matches
          fileChanged = true
        }
      }
      
      if (fileChanged) {
        await fs.writeFile(file, content, 'utf8')
        totalFixed++
        console.log(chalk.green(`✨ Fixed: ${file.replace(process.cwd(), '.')}`))
      }
      
    } catch (error) {
      console.error(chalk.red(`💥 Error: ${file}:`), error.message)
    }
  }
  
  console.log(chalk.cyan(`\n🎉 Fixed ${totalFixed} files with ${totalReplacements} replacements`))
  
  // Quick verification
  console.log(chalk.blue('🔍 Quick verification...'))
  await $`npm run check:ultra-fast`
}

main().catch(console.error)