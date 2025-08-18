// autosolve-loop.cjs - Continuous Improvement Loop
const { autosolve } = require('./autosolve-runner.cjs');
const { execSync } = require('child_process');
const fs = require('fs');

const CONFIG = {
    maxCycles: 5,
    delayBetweenCycles: 5000, // 5 seconds
    convergenceThreshold: 0.95, // 95% reduction in errors
    logFile: 'logs/autosolve-loop.log'
};

// Log to file and console
function log(message) {
    const timestamp = new Date().toISOString();
    const logMessage = `[${timestamp}] ${message}`;
    console.log(logMessage);
    fs.appendFileSync(CONFIG.logFile, logMessage + '\n');
}

// Run recommendation aggregation
async function runAggregation() {
    try {
        log('Running recommendation aggregation...');
        execSync('npm run recommend:aggregate', { stdio: 'inherit' });
        return true;
    } catch (err) {
        log(`Aggregation failed: ${err.message}`);
        return false;
    }
}

// Get current error count
function getErrorCount() {
    try {
        execSync('npx tsc --noEmit', { stdio: 'pipe' });
        return 0;
    } catch (err) {
        const output = err.stdout?.toString() || '';
        const matches = output.match(/Found (\d+) error/);
        return matches ? parseInt(matches[1]) : -1;
    }
}

// Main loop
async function continuousLoop() {
    log('🔄 Starting Continuous Autosolve Loop');

    let cycle = 0;
    let initialErrorCount = getErrorCount();
    let previousErrorCount = initialErrorCount;

    if (initialErrorCount === 0) {
        log('✅ No errors found, nothing to fix!');
        return 0;
    }

    log(`Initial error count: ${initialErrorCount}`);

    while (cycle < CONFIG.maxCycles) {
        cycle++;
        log(`\n═══════════════════════════════════════`);
        log(`Cycle ${cycle}/${CONFIG.maxCycles}`);
        log(`═══════════════════════════════════════`);

        // Run aggregation
        await runAggregation();

        // Run autosolve
        log('Running autosolve...');
        const exitCode = await autosolve();

        // Check current error count
        const currentErrorCount = getErrorCount();
        log(`Current error count: ${currentErrorCount}`);

        // Check for convergence
        if (currentErrorCount === 0) {
            log('✅ All errors fixed! Loop complete.');
            return 0;
        }

        if (exitCode === 0) {
            log('✅ Autosolve completed successfully');
            return 0;
        }

        if (currentErrorCount >= previousErrorCount) {
            log('⚠️  No improvement detected, stopping loop');
            return 2;
        }

        // Calculate improvement
        const improvement = (initialErrorCount - currentErrorCount) / initialErrorCount;
        log(`Improvement: ${(improvement * 100).toFixed(1)}%`);

        if (improvement >= CONFIG.convergenceThreshold) {
            log(`✅ Reached convergence threshold (${CONFIG.convergenceThreshold * 100}%)`);
            return 0;
        }

        previousErrorCount = currentErrorCount;

        // Wait before next cycle
        if (cycle < CONFIG.maxCycles) {
            log(`Waiting ${CONFIG.delayBetweenCycles / 1000} seconds before next cycle...`);
            await new Promise(resolve => setTimeout(resolve, CONFIG.delayBetweenCycles));
        }
    }

    log(`⚠️  Reached maximum cycles (${CONFIG.maxCycles})`);
    log(`Final error count: ${getErrorCount()}`);
    log(`Total improvement: ${((initialErrorCount - getErrorCount()) / initialErrorCount * 100).toFixed(1)}%`);

    return 1;
}

// Run if executed directly
if (require.main === module) {
    // Ensure log directory exists
    if (!fs.existsSync('logs')) {
        fs.mkdirSync('logs', { recursive: true });
    }

    continuousLoop().then(exitCode => {
        log(`Loop completed with exit code: ${exitCode}`);
        process.exit(exitCode);
    }).catch(err => {
        log(`Fatal error: ${err}`);
        process.exit(1);
    });
}

module.exports = { continuousLoop };
