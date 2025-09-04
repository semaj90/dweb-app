#!/usr/bin/env node
/** Database setup helper: migrate + optional seed */
import chalk from 'chalk';
import boxen from 'boxen';
import { $ } from 'zx';
import { createSpinner } from 'nanospinner';

async function main(){
  console.log(boxen(chalk.blue.bold('Evidence System - Database Setup'), { padding:1, borderColor:'blue' }));
  const mig = createSpinner('Running migrations').start();
  const m = await $`npm run db:migrate`.nothrow();
  m.exitCode===0?mig.success({text:'Migrations complete'}):mig.error({text:'Migration errors'});
  const seed = createSpinner('Seeding').start();
  const s = await $`npm run db:seed`.nothrow();
  s.exitCode===0?seed.success({text:'Seed complete'}):seed.warn({text:'Seed skipped'});
}

main().catch(e=>{ console.error(chalk.red('DB setup failed'), e); process.exit(1); });
