#!/usr/bin/env node
/** Smart setup orchestrator: ensures directories, env, and minimal DB migrations */
import fs from 'fs';
import path from 'path';
import boxen from 'boxen';
import chalk from 'chalk';
import { $ } from 'zx';
import { createSpinner } from 'nanospinner';

const root = path.resolve(process.cwd(), '..');

async function ensureDir(p){ if(!fs.existsSync(p)) fs.mkdirSync(p,{recursive:true}); }

async function run(){
  console.log(boxen(chalk.magenta.bold('Evidence System - Smart Setup'), { padding: 1, borderColor: 'magenta' }));
  const spinner = createSpinner('Preparing environment').start();
  await ensureDir(path.join(root,'uploads'));
  await ensureDir(path.join(root,'temp'));
  spinner.success({ text: 'Directories ready' });

  const dbSpin = createSpinner('Applying migrations (if any)').start();
  const mig = await $`npm run db:migrate`.nothrow();
  if(mig.exitCode===0) dbSpin.success({ text: 'Migrations applied' }); else dbSpin.warn({ text: 'Migration step reported issues' });

  const seedSpin = createSpinner('Seeding baseline data').start();
  const seed = await $`npm run db:seed`.nothrow();
  if(seed.exitCode===0) seedSpin.success({ text: 'Seed complete' }); else seedSpin.warn({ text: 'Seed skipped or failed' });

  console.log(chalk.green('\nSetup complete.'));
}

run().catch(e=>{ console.error(chalk.red('Setup failed'), e); process.exit(1); });
