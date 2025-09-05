import { exec } from 'child_process';
import { promisify } from 'util';
import path from 'path';

const execAsync = promisify(exec);

const projectPath = 'C:\\Users\\james\\Desktop\\web-app\\sveltekit-frontend';

async function runSvelteCheck() {
  console.log('🔍 Running svelte-check to identify current errors...\n');
  
  try {
    // Change to project directory and run check
    const { stdout, stderr } = await execAsync('npm run check', {
      cwd: projectPath,
      maxBuffer: 1024 * 1024 * 10 // 10MB buffer for large output
    });
    
    console.log('=== SVELTE-CHECK OUTPUT ===');
    if (stdout) {
      console.log(stdout);
    }
    
    if (stderr) {
      console.log('=== STDERR ===');
      console.log(stderr);
    }
    
    console.log('✅ Check completed successfully - No errors found!');
    
  } catch (error) {
    console.log('=== ERRORS FOUND ===');
    console.log(error.stdout || '');
    console.log(error.stderr || '');
    
    // Count error types
    const output = (error.stdout || '') + (error.stderr || '');
    const errorCount = (output.match(/Error:/g) || []).length;
    const warningCount = (output.match(/Warning:/g) || []).length;
    
    console.log(`\n📊 Summary:`);
    console.log(`   Errors: ${errorCount}`);
    console.log(`   Warnings: ${warningCount}`);
    
    if (errorCount === 0) {
      console.log('✅ No TypeScript errors found!');
    } else {
      console.log('❌ TypeScript errors need to be fixed');
    }
  }
}

// Run the check
runSvelteCheck().catch(console.error);
