#!/usr/bin/env node
/** Naive stop: find processes by filename and kill */
import chalk from 'chalk';
import boxen from 'boxen';
import { $ } from 'zx';

const TARGETS = ['ocr-service.mjs','standalone-upload-server.mjs'];

async function main(){
  console.log(boxen(chalk.red.bold('Evidence System - Stop Services'), { padding:1, borderColor:'red' }));
  for(const t of TARGETS){
    const grep = process.platform==='win32'
      ? await $`powershell -Command "Get-Process node -ErrorAction SilentlyContinue | Where-Object { $_.Path -like '*node*' } | Select-Object Id, Path"`.nothrow()
      : await $`ps -ef | grep node`.nothrow();
    const lines = (grep.stdout||'').split(/\r?\n/).filter(l=>l.includes(t));
    if(!lines.length){
      console.log(chalk.yellow(`No process found for ${t}`));
      continue;
    }
    for(const line of lines){
      const pidMatch = line.match(/\b(\d{2,6})\b/);
      if(pidMatch){
        const pid = pidMatch[1];
        try { process.kill(pid, 'SIGTERM'); console.log(chalk.green(`Stopped ${t} (pid ${pid})`)); }
        catch(e){ console.log(chalk.red(`Failed to stop ${t} (pid ${pid}): ${e.message}`)); }
      }
    }
  }
}

main().catch(e=>{ console.error(chalk.red('Stop failed'), e); process.exit(1); });
