#!/usr/bin/env node

import chalk from 'chalk';
import ora from 'ora';
import 'zx/globals';

async function main() {
  console.log(chalk.cyan.bold('🚀 YoRHa Legal AI — Deploy'));

  const spinner = ora('Building frontend...').start();
  try {
    await $`cd sveltekit-frontend && npm ci && npm run build`;
    spinner.succeed('Frontend built');
  } catch (e) {
    spinner.fail('Frontend build failed');
    process.exit(1);
  }

  const goSpinner = ora('Building Go microservice...').start();
  try {
    await $`cd go-microservice && go build -o enhanced-legal-ai.exe enhanced-grpc-legal-server.go`;
    goSpinner.succeed('Go microservice built');
  } catch (e) {
    goSpinner.fail('Go build failed');
    process.exit(1);
  }

  const goOllamaSpinner = ora('Building Go Ollama SIMD service...').start();
  try {
  await $`cd go-microservice && go build -o go-ollama-simd.exe ./cmd/go-ollama-simd`;
    goOllamaSpinner.succeed('Go Ollama SIMD service built');
  } catch (e) {
    goOllamaSpinner.fail('Go Ollama SIMD build failed');
    process.exit(1);
  }

  console.log(chalk.green('✅ Deploy artifacts ready. Use npm run start -m to launch.'));
}

main().catch(e => { console.error(e); process.exit(1); });
