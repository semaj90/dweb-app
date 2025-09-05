// workers/start-workers.js
const { spawn } = require('child_process');
const path = require('path');

const workers = [
  'document-processor.worker.js',
  'analysis.worker.js',
  'notification.worker.js'
];

console.log('Starting Legal AI Workers...');

workers.forEach(worker => {
  const workerPath = path.join(__dirname, worker);
  
  // Check if worker file exists
  if (!require('fs').existsSync(workerPath)) {
    console.error(`Worker file not found: ${workerPath}`);
    return;
  }
  
  const proc = spawn('node', [workerPath], {
    stdio: 'inherit',
    env: { ...process.env }
  });

  proc.on('error', (err) => {
    console.error(`Failed to start ${worker}:`, err);
  });

  console.log(`Started ${worker} with PID ${proc.pid}`);
});

process.on('SIGINT', () => {
  console.log('Shutting down workers...');
  process.exit(0);
});
