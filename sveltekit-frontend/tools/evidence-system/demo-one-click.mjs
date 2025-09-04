#!/usr/bin/env node
/** One-click demo: smart setup -> start services -> test */
import { $ } from 'zx';
import chalk from 'chalk';
import boxen from 'boxen';
import { createSpinner } from 'nanospinner';

async function main(){
  console.log(boxen(chalk.bold('Evidence System - One Click Demo'), { padding:1, borderColor:'white' }));
  const steps = [
    { name: 'Smart Setup', cmd: 'node tools/evidence-system/setup-smart.mjs' },
    { name: 'Start Services', cmd: 'node tools/evidence-system/start-services.mjs' },
    { name: 'System Test', cmd: 'node tools/evidence-system/test-system.mjs' }
  ];
  for(const s of steps){
    const spin = createSpinner(s.name).start();
    const r = await $`${s.cmd}`.nothrow();
    r.exitCode===0?spin.success({text: s.name+' complete'}):spin.warn({text: s.name+' reported issues'});
  }
  console.log('\n'+chalk.green('Demo sequence finished.'));
}

main().catch(e=>{ console.error(chalk.red('Demo failed'), e); process.exit(1); });
