#!/usr/bin/env node
import { execSync } from 'child_process';

function tailFile(file) {
  const isWindows = process.platform === 'win32';
  
  if (isWindows) {
    execSync(`powershell Get-Content ${file} -Wait`, { stdio: 'inherit' });
  } else {
    execSync(`tail -f ${file}`, { stdio: 'inherit' });
  }
}

const file = process.argv[2];
if (file) tailFile(file);
