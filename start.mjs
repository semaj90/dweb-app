#!/usr/bin/env zx

// Start script for YoRHa Interface using zx and pm2
import { $, chalk } from 'zx'

console.log(chalk.cyan('🤖 Starting YoRHa Interface System...'))

try {
  // Kill any existing pm2 processes
  console.log(chalk.yellow('🔄 Cleaning up existing processes...'))
  await $`pm2 delete all || echo "No existing processes"`
  
  // Start the main SvelteKit application
  console.log(chalk.green('🚀 Starting SvelteKit server...'))
  await $`pm2 start npm --name "yorha-web" -- run dev`
  
  // Start the queue worker if the script exists
  console.log(chalk.blue('📨 Starting queue worker...'))
  try {
    await $`pm2 start tsx --name "yorha-queue" -- scripts/queue-worker.ts`
  } catch (error) {
    console.log(chalk.yellow('⚠️ Queue worker script not found, skipping...'))
  }
  
  // Show pm2 status
  console.log(chalk.cyan('📊 Current process status:'))
  await $`pm2 list`
  
  console.log(chalk.green('✅ YoRHa Interface started successfully!'))
  console.log(chalk.cyan('🌐 Web interface: http://localhost:5173'))
  console.log(chalk.cyan('📊 PM2 monitoring: pm2 monit'))
  
} catch (error) {
  console.error(chalk.red('❌ Failed to start YoRHa Interface:'), error)
  process.exit(1)
}