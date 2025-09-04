#!/usr/bin/env zx

/**
 * Simple Task Testing - Quick verification of all concurrent tasks
 */

import 'zx/globals'
import { performance } from 'perf_hooks'

console.log(chalk.cyan('🧪 Starting Simple Task Tests'))

const testResults = {
  passed: 0,
  failed: 0,
  tests: []
}

// Test 1: Check if scripts exist
async function testScriptExists() {
  console.log(chalk.blue('\n🔍 Test 1: Checking script files'))
  
  const scripts = [
    'scripts/gpu-cluster-concurrent-executor.mjs',
    'scripts/cluster-multicore-manager.mjs',
    'scripts/test-all-tasks.mjs'
  ]
  
  for (const script of scripts) {
    try {
      const exists = await fs.pathExists(script)
      if (exists) {
        console.log(chalk.green(`✅ ${script} exists`))
        testResults.passed++
      } else {
        console.log(chalk.red(`❌ ${script} missing`))
        testResults.failed++
      }
      testResults.tests.push({ name: `Script exists: ${script}`, passed: exists })
    } catch (error) {
      console.log(chalk.red(`❌ ${script} error: ${error.message}`))
      testResults.failed++
      testResults.tests.push({ name: `Script exists: ${script}`, passed: false, error: error.message })
    }
  }
}

// Test 2: Check npm scripts
async function testNpmScripts() {
  console.log(chalk.blue('\n🔍 Test 2: Checking npm scripts'))
  
  const scripts = [
    'dev:full',
    'check:autosolve',
    'gpu:cluster:execute',
    'multicore:full',
    'concurrent:simd'
  ]
  
  try {
    const packageJson = await fs.readJSON('package.json')
    
    for (const script of scripts) {
      const exists = packageJson.scripts && packageJson.scripts[script]
      if (exists) {
        console.log(chalk.green(`✅ npm script '${script}' defined`))
        testResults.passed++
      } else {
        console.log(chalk.red(`❌ npm script '${script}' missing`))
        testResults.failed++
      }
      testResults.tests.push({ name: `NPM script: ${script}`, passed: !!exists })
    }
  } catch (error) {
    console.log(chalk.red(`❌ Package.json read error: ${error.message}`))
    testResults.failed++
  }
}

// Test 3: Check VS Code tasks
async function testVSCodeTasks() {
  console.log(chalk.blue('\n🔍 Test 3: Checking VS Code tasks'))
  
  try {
    const tasksFile = '.vscode/tasks.json'
    const exists = await fs.pathExists(tasksFile)
    
    if (exists) {
      const tasksJson = await fs.readJSON(tasksFile)
      const taskLabels = tasksJson.tasks.map(t => t.label)
      
      const expectedTasks = [
        'GPU Cluster Concurrent Executor',
        'SIMD + WebGPU Acceleration', 
        'WebGPU SOM Cache Processing',
        'Multicore Performance Analysis'
      ]
      
      expectedTasks.forEach(expectedTask => {
        const found = taskLabels.some(label => label.includes(expectedTask))
        if (found) {
          console.log(chalk.green(`✅ VS Code task: ${expectedTask}`))
          testResults.passed++
        } else {
          console.log(chalk.red(`❌ VS Code task missing: ${expectedTask}`))
          testResults.failed++
        }
        testResults.tests.push({ name: `VS Code task: ${expectedTask}`, passed: found })
      })
    } else {
      console.log(chalk.red(`❌ .vscode/tasks.json missing`))
      testResults.failed++
    }
  } catch (error) {
    console.log(chalk.red(`❌ VS Code tasks error: ${error.message}`))
    testResults.failed++
  }
}

// Test 4: Test basic script loading (syntax check)
async function testScriptSyntax() {
  console.log(chalk.blue('\n🔍 Test 4: Testing script syntax'))
  
  const scripts = [
    'scripts/gpu-cluster-concurrent-executor.mjs',
    'scripts/cluster-multicore-manager.mjs'
  ]
  
  for (const script of scripts) {
    try {
      // Try to parse the script without executing
      await $`node --check ${script}`
      console.log(chalk.green(`✅ ${script} syntax OK`))
      testResults.passed++
      testResults.tests.push({ name: `Syntax check: ${script}`, passed: true })
    } catch (error) {
      console.log(chalk.red(`❌ ${script} syntax error: ${error.message}`))
      testResults.failed++
      testResults.tests.push({ name: `Syntax check: ${script}`, passed: false, error: error.message })
    }
  }
}

// Test 5: Test integration readiness
async function testIntegrationReadiness() {
  console.log(chalk.blue('\n🔍 Test 5: Testing integration readiness'))
  
  // Check if required directories exist
  const dirs = ['scripts', '.vscode', 'src/lib']
  
  for (const dir of dirs) {
    try {
      const exists = await fs.pathExists(dir)
      if (exists) {
        console.log(chalk.green(`✅ Directory exists: ${dir}`))
        testResults.passed++
      } else {
        console.log(chalk.red(`❌ Directory missing: ${dir}`))
        testResults.failed++
      }
      testResults.tests.push({ name: `Directory: ${dir}`, passed: exists })
    } catch (error) {
      testResults.failed++
      testResults.tests.push({ name: `Directory: ${dir}`, passed: false, error: error.message })
    }
  }
}

// Run all tests
async function runAllTests() {
  const startTime = performance.now()
  
  await testScriptExists()
  await testNpmScripts()  
  await testVSCodeTasks()
  await testScriptSyntax()
  await testIntegrationReadiness()
  
  const totalTime = performance.now() - startTime
  
  // Generate report
  console.log(chalk.cyan('\n📊 Test Results Summary'))
  console.log('='.repeat(50))
  
  const total = testResults.passed + testResults.failed
  const successRate = total > 0 ? (testResults.passed / total) * 100 : 0
  
  console.log(chalk.green(`✅ Passed: ${testResults.passed}/${total} (${successRate.toFixed(1)}%)`))
  console.log(chalk.red(`❌ Failed: ${testResults.failed}/${total}`))
  console.log(chalk.blue(`⏱️ Total Time: ${totalTime.toFixed(2)}ms`))
  
  // Detailed results
  console.log(chalk.yellow('\n📈 Detailed Results:'))
  testResults.tests.forEach((test, index) => {
    const status = test.passed ? chalk.green('✅ PASS') : chalk.red('❌ FAIL')
    console.log(`  ${index + 1}. ${status} ${test.name}`)
    if (test.error) {
      console.log(`     Error: ${test.error}`)
    }
  })
  
  // Integration status
  console.log(chalk.cyan('\n🔗 Integration Status:'))
  if (successRate >= 90) {
    console.log(chalk.green('🎉 System ready for dev:full and autosolve integration'))
  } else if (successRate >= 70) {
    console.log(chalk.yellow('⚠️ Most components ready, minor fixes needed'))
  } else {
    console.log(chalk.red('💥 Major issues detected, fixes required before integration'))
  }
  
  // Save report
  const report = {
    timestamp: new Date().toISOString(),
    summary: {
      total,
      passed: testResults.passed,
      failed: testResults.failed,
      successRate,
      totalTime
    },
    tests: testResults.tests
  }
  
  await fs.writeFile('simple-task-test-report.json', JSON.stringify(report, null, 2))
  console.log(chalk.green('💾 Report saved to simple-task-test-report.json'))
  
  return successRate >= 70
}

// Execute if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
  const success = await runAllTests()
  process.exit(success ? 0 : 1)
}

export { runAllTests }