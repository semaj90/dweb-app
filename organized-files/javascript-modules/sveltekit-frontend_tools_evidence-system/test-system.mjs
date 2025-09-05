#!/usr/bin/env node
/** System test: run detection then a sample OCR + upload */
import chalk from 'chalk';
import boxen from 'boxen';
import { createSpinner } from 'nanospinner';
import { $ } from 'zx';
import fs from 'fs';
import path from 'path';

async function simpleOCRText(){
  const samplePath = path.join(process.cwd(),'temp','sample.txt');
  fs.mkdirSync(path.dirname(samplePath), { recursive: true });
  fs.writeFileSync(samplePath, 'This is a sample evidence text file for OCR passthrough.');
  const curl = await $`curl -s -X POST -F file=@${samplePath} http://localhost:8601/ocr`.nothrow();
  return curl.stdout || curl.stderr || '';
}

async function main(){
  console.log(boxen(chalk.green.bold('Evidence System - Test Suite'), { padding:1, borderColor:'green' }));
  const detectSpin = createSpinner('Detecting services').start();
  const detect = await $`node tools/evidence-system/detect-services.mjs`.nothrow();
  detect.exitCode===0?detectSpin.success({text:'All mandatory services detected or optional missing'}):detectSpin.warn({text:'Some services missing'});

  const ocrSpin = createSpinner('Testing OCR service (text passthrough)').start();
  let ocrResult = '';
  try { ocrResult = await simpleOCRText(); ocrSpin.success({ text: 'OCR path OK' }); } catch(e){ ocrSpin.error({ text: 'OCR test failed' }); }
  console.log(chalk.gray(ocrResult.slice(0,120)+'...'));
}

main().catch(e=>{ console.error(chalk.red('System test failed'), e); process.exit(1); });
