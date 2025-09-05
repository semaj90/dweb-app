#!/usr/bin/env node
/** Start core evidence microservices (OCR + standalone uploader) */
import { $, echo } from 'zx';
import chalk from 'chalk';
import boxen from 'boxen';
import { createSpinner } from 'nanospinner';

async function spawnDetached(cmd){
  const spinner = createSpinner(`Starting ${cmd.join(' ')}`).start();
  const p = $`${cmd}`.nothrow();
  // fire and forget; can't truly detach via zx easily, assume user uses separate terminals
  setTimeout(()=>spinner.success({ text: `Spawned ${cmd[1]||cmd[0]}` }),700);
  return p;
}

async function main(){
  console.log(boxen(chalk.cyan.bold('Evidence System - Start Services'), { padding: 1, borderColor: 'cyan' }));
  await spawnDetached(['node','tools/ocr-service.mjs']);
  await spawnDetached(['node','tools/standalone-upload-server.mjs']);
  console.log('\n'+chalk.green('Services launching...'));
  console.log(chalk.gray('Use evidence:detect to verify health.'));
}

main().catch(e=>{ console.error(chalk.red('Failed to start services'), e); process.exit(1); });
