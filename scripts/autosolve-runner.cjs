// autosolve-runner.cjs - TypeScript Error Auto-Fixer with RAG Integration
const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');
const fetch = require('node-fetch');

// Configuration
const CONFIG = {
    maxIterations: 10,
    backupDir: 'backups',
    logFile: 'logs/autosolve-history.jsonl',
    declarationsFile: 'src/auto-decls.d.ts',
    ragEndpoint: 'http://localhost:8097/api/recommendations/generate',
    ollamaEndpoint: 'http://localhost:11434/api/generate',
    aggregateEndpoint: 'http://localhost:8123/aggregate'
};

// Ensure directories exist
function ensureDirectories() {
    ['logs', 'backups', 'src'].forEach(dir => {
        if (!fs.existsSync(dir)) {
            fs.mkdirSync(dir, { recursive: true });
        }
    });
}

// Log action to history
function logAction(action, result) {
    const entry = {
        timestamp: new Date().toISOString(),
        action,
        result,
        pid: process.pid
    };
    
    fs.appendFileSync(CONFIG.logFile, JSON.stringify(entry) + '\n');
}

// Create backup of file
function createBackup(filePath) {
    if (!fs.existsSync(filePath)) return null;
    
    const timestamp = Date.now();
    const backupPath = path.join(CONFIG.backupDir, `${path.basename(filePath)}.bak.${timestamp}`);
    fs.copyFileSync(filePath, backupPath);
    return backupPath;
}

// Parse TypeScript errors
function parseTscErrors(output) {
    const errors = [];
    const lines = output.split('\n');
    const errorPattern = /^(.+?)\((\d+),(\d+)\):\s+error\s+(\w+):\s+(.+)$/;
    
    for (const line of lines) {
        const match = line.match(errorPattern);
        if (match) {
            errors.push({
                file: match[1],
                line: parseInt(match[2]),
                column: parseInt(match[3]),
                code: match[4],
                message: match[5]
            });
        }
    }
    
    return errors;
}

// Get AI recommendation for fix
async function getAIRecommendation(error, context) {
    try {
        // Try RAG first
        const ragResponse = await fetch(CONFIG.ragEndpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                user_id: 'autosolve',
                context: {
                    error: error,
                    file_context: context,
                    type: 'typescript_error'
                }
            })
        });
        
        if (ragResponse.ok) {
            const data = await ragResponse.json();
            if (data.recommendations && data.recommendations.length > 0) {
                return data.recommendations[0].solution;
            }
        }
        
        // Fallback to Ollama
        const ollamaResponse = await fetch(CONFIG.ollamaEndpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model: 'gemma3:latest',
                prompt: `Fix this TypeScript error:\n${error.message}\nFile: ${error.file}\nLine: ${error.line}\n\nSuggest a fix:`,
                stream: false
            })
        });
        
        if (ollamaResponse.ok) {
            const data = await ollamaResponse.json();
            return data.response;
        }
    } catch (err) {
        console.error('AI recommendation failed:', err);
    }
    
    return null;
}

// Apply fix to file
function applyFix(error, fix) {
    const filePath = error.file;
    
    if (!fs.existsSync(filePath)) {
        console.error(`File not found: ${filePath}`);
        return false;
    }
    
    // Create backup
    const backupPath = createBackup(filePath);
    logAction('backup', { file: filePath, backup: backupPath });
    
    try {
        let content = fs.readFileSync(filePath, 'utf8');
        const lines = content.split('\n');
        
        // Apply different fixes based on error code
        switch (error.code) {
            case 'TS2304': // Cannot find name
                // Add declaration
                const symbolName = error.message.match(/Cannot find name '([^']+)'/)?.[1];
                if (symbolName) {
                    appendDeclaration(symbolName);
                    logAction('add_declaration', { symbol: symbolName });
                    return true;
                }
                break;
                
            case 'TS7006': // Parameter implicitly has 'any' type
                // Add type annotation
                const lineIndex = error.line - 1;
                if (lines[lineIndex]) {
                    lines[lineIndex] = lines[lineIndex].replace(
                        /(\w+)(?=\s*[,)])/,
                        '$1: any'
                    );
                    content = lines.join('\n');
                    fs.writeFileSync(filePath, content);
                    logAction('add_any_type', { file: filePath, line: error.line });
                    return true;
                }
                break;
                
            case 'TS6133': // Variable is declared but never used
                // Comment out unused variable
                const varLineIndex = error.line - 1;
                if (lines[varLineIndex] && !lines[varLineIndex].trim().startsWith('//')) {
                    lines[varLineIndex] = '// ' + lines[varLineIndex];
                    content = lines.join('\n');
                    fs.writeFileSync(filePath, content);
                    logAction('comment_unused', { file: filePath, line: error.line });
                    return true;
                }
                break;
                
            case 'TS2345': // Argument type mismatch
            case 'TS2322': // Type not assignable
                // Use AI recommendation
                if (fix) {
                    // Apply AI-suggested fix
                    console.log(`Applying AI fix for ${error.code}: ${fix}`);
                    // This would need more sophisticated AST manipulation
                    logAction('ai_fix', { error: error.code, fix });
                    return false; // Not implemented yet
                }
                break;
                
            default:
                console.log(`Unhandled error code: ${error.code}`);
                return false;
        }
    } catch (err) {
        console.error(`Failed to apply fix: ${err}`);
        return false;
    }
    
    return false;
}

// Append declaration to auto-decls.d.ts
function appendDeclaration(symbolName) {
    const declaration = `declare const ${symbolName}: any;\n`;
    
    if (!fs.existsSync(CONFIG.declarationsFile)) {
        fs.writeFileSync(CONFIG.declarationsFile, '// Auto-generated declarations\n');
    }
    
    const existing = fs.readFileSync(CONFIG.declarationsFile, 'utf8');
    if (!existing.includes(`declare const ${symbolName}`)) {
        fs.appendFileSync(CONFIG.declarationsFile, declaration);
    }
}

// Run TypeScript compiler
function runTsc() {
    try {
        const output = execSync('npx tsc --noEmit', { 
            encoding: 'utf8',
            maxBuffer: 10 * 1024 * 1024 // 10MB buffer
        });
        return { success: true, output };
    } catch (err) {
        return { 
            success: false, 
            output: err.stdout || err.message,
            stderr: err.stderr
        };
    }
}

// Get aggregate summary
async function getAggregateSummary() {
    try {
        const response = await fetch(CONFIG.aggregateEndpoint);
        if (response.ok) {
            return await response.json();
        }
    } catch (err) {
        console.error('Failed to get aggregate summary:', err);
    }
    return null;
}

// Main autosolve function
async function autosolve() {
    console.log('🔧 Starting TypeScript Auto-Solver...\n');
    ensureDirectories();
    
    let iteration = 0;
    let fixedCount = 0;
    let previousErrorCount = Infinity;
    
    while (iteration < CONFIG.maxIterations) {
        iteration++;
        console.log(`\n📍 Iteration ${iteration}/${CONFIG.maxIterations}`);
        
        // Run TypeScript compiler
        const tscResult = runTsc();
        
        if (tscResult.success) {
            console.log('✅ No TypeScript errors found!');
            logAction('complete', { iterations: iteration, fixed: fixedCount });
            
            // Get and display summary
            const summary = await getAggregateSummary();
            if (summary) {
                console.log('\n📊 Summary:', summary);
            }
            
            return 0; // Success
        }
        
        // Parse errors
        const errors = parseTscErrors(tscResult.output);
        console.log(`Found ${errors.length} errors`);
        
        if (errors.length === 0) {
            console.log('⚠️  TypeScript failed but no errors parsed');
            return 1;
        }
        
        if (errors.length >= previousErrorCount) {
            console.log('⚠️  Error count not decreasing, stopping');
            return 2; // Needs manual intervention
        }
        
        previousErrorCount = errors.length;
        
        // Group errors by type
        const errorsByCode = {};
        errors.forEach(err => {
            if (!errorsByCode[err.code]) {
                errorsByCode[err.code] = [];
            }
            errorsByCode[err.code].push(err);
        });
        
        console.log('\nError summary:');
        Object.entries(errorsByCode).forEach(([code, errs]) => {
            console.log(`  ${code}: ${errs.length} errors`);
        });
        
        // Fix errors
        let fixedInIteration = 0;
        for (const error of errors.slice(0, 10)) { // Fix up to 10 errors per iteration
            console.log(`\nFixing: ${error.file}:${error.line} - ${error.code}`);
            
            // Get AI recommendation for complex errors
            let aiRecommendation = null;
            if (['TS2345', 'TS2322'].includes(error.code)) {
                const context = fs.existsSync(error.file) 
                    ? fs.readFileSync(error.file, 'utf8').split('\n').slice(Math.max(0, error.line - 5), error.line + 5).join('\n')
                    : '';
                aiRecommendation = await getAIRecommendation(error, context);
            }
            
            if (applyFix(error, aiRecommendation)) {
                fixedInIteration++;
                fixedCount++;
            }
        }
        
        console.log(`Fixed ${fixedInIteration} errors in this iteration`);
        
        if (fixedInIteration === 0) {
            console.log('⚠️  No fixes applied, stopping');
            return 2;
        }
    }
    
    console.log(`\n⚠️  Reached maximum iterations (${CONFIG.maxIterations})`);
    return 2;
}

// Run if executed directly
if (require.main === module) {
    autosolve().then(exitCode => {
        process.exit(exitCode);
    }).catch(err => {
        console.error('Fatal error:', err);
        process.exit(1);
    });
}

module.exports = { autosolve, parseTscErrors, applyFix };
