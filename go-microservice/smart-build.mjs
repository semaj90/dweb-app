#!/usr/bin/env node

/**
 * Smart Go Build Script with Cache-Aware Dependency Management
 * Only downloads legacy dependencies when needed, uses cache efficiently
 */

import { spawn, exec } from 'child_process';
import { promises as fs } from 'fs';
import { join } from 'path';

const LEGACY_DEPS = [
  'github.com/prometheus/client_golang/prometheus',
  'github.com/prometheus/client_golang/prometheus/promhttp',
  'github.com/streadway/amqp',
  'google.golang.org/grpc',
  'gorgonia.org/gorgonia', 
  'gorgonia.org/tensor'
];

const runCommand = (command, args = []) => {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      stdio: 'pipe',
      shell: true,
      cwd: process.cwd()
    });

    let stdout = '';
    let stderr = '';

    child.stdout?.on('data', (data) => stdout += data.toString());
    child.stderr?.on('data', (data) => stderr += data.toString());

    child.on('close', (code) => {
      resolve({ code, stdout, stderr });
    });

    child.on('error', reject);
  });
};

const getGoPath = async () => {
  const result = await runCommand('go', ['env', 'GOPATH']);
  return result.stdout.trim();
};

const getCacheSize = async () => {
  try {
    const goPath = await getGoPath();
    const result = await runCommand('powershell', [
      `Get-ChildItem '${goPath}\\pkg\\mod' -Recurse | Measure-Object -Property Length -Sum | Select-Object -ExpandProperty Sum`
    ]);
    const bytes = parseInt(result.stdout.trim());
    return Math.round(bytes / (1024 * 1024)); // MB
  } catch (error) {
    return 'Unknown';
  }
};

const checkDependencyInCache = async (dep) => {
  try {
    const goPath = await getGoPath();
    const depPath = dep.replace('/', '\\');
    const result = await runCommand('powershell', [
      `Test-Path "${goPath}\\pkg\\mod\\${depPath}*"`
    ]);
    return result.stdout.trim() === 'True';
  } catch (error) {
    return false;
  }
};

const checkAllDepsInCache = async () => {
  console.log('🔍 Checking cached dependencies...');
  const results = await Promise.all(
    LEGACY_DEPS.map(async dep => ({
      dep,
      cached: await checkDependencyInCache(dep)
    }))
  );
  
  const missing = results.filter(r => !r.cached);
  const cached = results.filter(r => r.cached);
  
  console.log(`✅ Cached: ${cached.length}/${LEGACY_DEPS.length} dependencies`);
  if (missing.length > 0) {
    console.log(`❌ Missing: ${missing.map(m => m.dep).join(', ')}`);
  }
  
  return missing;
};

const downloadMissingDeps = async (missingDeps) => {
  console.log(`📥 Downloading ${missingDeps.length} missing dependencies...`);
  
  for (const { dep } of missingDeps) {
    console.log(`  ⬇️  ${dep}`);
    const result = await runCommand('go', ['get', dep]);
    if (result.code !== 0) {
      console.log(`    ❌ Failed: ${result.stderr}`);
    } else {
      console.log(`    ✅ Cached`);
    }
  }
  
  // Clean up go.mod
  console.log('🧹 Running go mod tidy...');
  await runCommand('go', ['mod', 'tidy']);
};

const buildWithTags = async (tags = []) => {
  const tagArgs = tags.length > 0 ? ['-tags', tags.join(',')] : [];
  const buildCmd = ['build', ...tagArgs, './...'];
  
  console.log(`🔨 Building: go ${buildCmd.join(' ')}`);
  const result = await runCommand('go', buildCmd);
  
  if (result.code === 0) {
    console.log('✅ Build successful');
    return true;
  } else {
    console.log('❌ Build failed:');
    console.log(result.stderr);
    return false;
  }
};

const main = async () => {
  console.log('🚀 Smart Go Build with Cache-Aware Dependencies\n');
  
  const cacheSize = await getCacheSize();
  console.log(`📊 Current cache size: ${cacheSize} MB`);
  
  const buildType = process.argv[2] || 'normal';
  
  switch (buildType) {
    case 'normal':
      console.log('\n🎯 Building without legacy tags...');
      await buildWithTags();
      break;
      
    case 'legacy':
      console.log('\n🎯 Building with legacy tags...');
      
      // Check if legacy deps are cached
      const missingDeps = await checkAllDepsInCache();
      
      if (missingDeps.length > 0) {
        console.log(`\n❓ ${missingDeps.length} dependencies need to be downloaded.`);
        console.log('This will add them to your Go module cache (no re-download needed later).');
        
        // Auto-download in CI or if --auto flag passed
        if (process.argv.includes('--auto') || process.env.CI) {
          await downloadMissingDeps(missingDeps);
        } else {
          console.log('\nOptions:');
          console.log('  1. Download now (recommended): node smart-build.mjs legacy --auto');
          console.log('  2. Skip legacy build: node smart-build.mjs normal');
          console.log('  3. Manual install: go get <dependency>');
          process.exit(1);
        }
      } else {
        console.log('✅ All legacy dependencies already cached!');
      }
      
      await buildWithTags(['legacy']);
      break;
      
    case 'cache-info':
      console.log('\n📊 Cache Information:');
      await checkAllDepsInCache();
      break;
      
    case 'clean-cache':
      console.log('\n🧹 Cleaning module cache...');
      const result = await runCommand('go', ['clean', '-modcache']);
      if (result.code === 0) {
        console.log('✅ Cache cleaned');
      } else {
        console.log('❌ Failed to clean cache');
      }
      break;
      
    default:
      console.log('Usage:');
      console.log('  node smart-build.mjs normal              # Build without legacy');
      console.log('  node smart-build.mjs legacy              # Build with legacy (check deps)'); 
      console.log('  node smart-build.mjs legacy --auto       # Build with legacy (auto-download)');
      console.log('  node smart-build.mjs cache-info          # Show cache status');
      console.log('  node smart-build.mjs clean-cache         # Clear entire cache');
      break;
  }
  
  const finalCacheSize = await getCacheSize();
  console.log(`\n📊 Final cache size: ${finalCacheSize} MB`);
};

main().catch(console.error);