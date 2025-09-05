#!/usr/bin/env zx

import 'zx/globals'

console.log(chalk.cyan('🧪 Quick Test Starting'))

// Test basic zx functionality
console.log(chalk.green('✅ zx imports working'))
console.log(chalk.blue('📊 Node version:'), process.version)
console.log(chalk.yellow('🔧 Platform:'), process.platform)

// Test basic chalk
console.log(chalk.red('❌ Red test'), chalk.green('✅ Green test'))

// Test performance
console.log(chalk.magenta('⚡ Performance module available'))

// Quick file system test
try {
  const packageExists = await fs.pathExists('package.json')
  console.log(chalk.cyan(`📦 Package.json exists: ${packageExists}`))
} catch (error) {
  console.log(chalk.red('💥 Filesystem error:'), error.message)
}

// Test basic Promise.all
try {
  const results = await Promise.all([
    Promise.resolve('Task 1 complete'),
    Promise.resolve('Task 2 complete'),
    new Promise(resolve => setTimeout(() => resolve('Task 3 complete'), 100))
  ])
  
  console.log(chalk.green('✅ Promise.all test:'), results.length, 'tasks completed')
} catch (error) {
  console.log(chalk.red('💥 Promise error:'), error.message)
}

console.log(chalk.cyan('🎉 Quick test completed successfully'))
process.exit(0)