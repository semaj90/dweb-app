// Native HTTP fallback when Express is not available
let express, http;
let useNativeHttp = false;

try {
  express = require('express');
} catch (e) {
  console.log('📦 Express not found, using native HTTP server');
  http = require('http');
  useNativeHttp = true;
}

const fs = require('fs').promises;

// Optional multi-core clustering (set WORKER_CLUSTER=1, WORKER_WORKERS=n)
const os = require('os');
const cluster = require('cluster');

// Optional Go HTTP server (set USE_GO_SERVER=1, GO_SERVER_PATH to file or GO_SERVER_BIN to binary)
const cp = require('child_process');
const pathMod = require('path');

const enableCluster = process.env.WORKER_CLUSTER === '1';
const desiredWorkers = parseInt(process.env.WORKER_WORKERS || os.cpus().length, 10);

const useGoServer = process.env.USE_GO_SERVER === '1';
const goServerFile = process.env.GO_SERVER_PATH || pathMod.join(__dirname, '../go-microservice/server.go');
const goServerBin = process.env.GO_SERVER_BIN; // pre-built binary path (faster)

// If using Go server we suppress the Node HTTP auto-start block below
if (useGoServer) {
  // Prevent the bottom `if (require.main === module)` from running
  require.main = null;
  try {
    const args = goServerBin
      ? [goServerBin]
      : ['go', 'run', goServerFile];
    const cmd = args.shift();
    console.log(`🚀 Starting Go HTTP server (${goServerBin ? 'binary' : 'go run'})`);
    const goProc = cp.spawn(cmd, args, { stdio: 'inherit' });
    process.on('exit', () => goProc.kill());
    goProc.on('exit', (code) => {
      console.log(`⚠️ Go server exited with code ${code}`);
      process.exit(code || 1);
    });
  } catch (e) {
    console.error('Failed to start Go server, falling back to Node implementation:', e.message);
    if (!enableCluster) {
      require.main = module;
    }
  }
}

// Cluster primary: fork workers, skip starting server in primary
if (enableCluster && cluster.isPrimary) {
  console.log(`🧵 Cluster mode enabled: launching ${desiredWorkers} workers`);
  for (let i = 0; i < desiredWorkers; i++) cluster.fork();

  cluster.on('exit', (worker, code, signal) => {
    console.log(`🔁 Worker ${worker.process.pid} died (code=${code} signal=${signal}). Restarting.`);
    cluster.fork();
  });

  // Prevent primary from running bottom auto-start block
  require.main = null;
}

// PM2 hint (informational)
if (process.env.PM2_HOME) {
  console.log('🟢 Detected PM2 environment. (e.g. pm2 start recommendation-worker.cjs -i max --name rec-worker)');
}

// Minimal zx-like helper for shell commands: const { sh } = require('./...'); await sh('echo hi');
async function sh(cmd, opts = {}) {
  return new Promise((resolve, reject) => {
    const p = cp.spawn(cmd, { shell: true, stdio: 'pipe', ...opts });
    let stdout = '', stderr = '';
    p.stdout.on('data', d => stdout += d);
    p.stderr.on('data', d => stderr += d);
    p.on('close', code => {
      if (code === 0) resolve({ stdout, stderr, code });
      else reject(Object.assign(new Error(`Command failed (${code}): ${cmd}\n${stderr}`), { stdout, stderr, code }));
    });
  });
}

module.exports = module.exports || {};
module.exports.sh = sh;
const path = require('path');

class RecommendationWorker {
  constructor(options = {}) {
    this.port = options.port || process.env.npm_config_port || 4100;
    this.logDir = options.logDir || 'logs';
    this.useNativeHttp = useNativeHttp;

    if (this.useNativeHttp) {
      this.setupNativeServer();
    } else {
      this.app = express();
      this.setupMiddleware();
      this.setupRoutes();
    }

    this.recommendations = [];
    this.actionHistory = [];
  }

  setupNativeServer() {
    this.server = http.createServer((req, res) => {
      // Enable CORS
      res.setHeader('Access-Control-Allow-Origin', '*');
      res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
      res.setHeader('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept, Authorization');

      if (req.method === 'OPTIONS') {
        res.writeHead(200);
        res.end();
        return;
      }

      // Parse URL
      const url = new URL(req.url, `http://${req.headers.host}`);

      if (req.method === 'GET' && url.pathname === '/health') {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({
          status: 'healthy',
          service: 'recommendation-worker',
          port: this.port,
          uptime: process.uptime(),
          timestamp: new Date().toISOString(),
          actionsProcessed: this.actionHistory.length,
          mode: 'native-http'
        }));
        return;
      }

      if (req.method === 'POST' && url.pathname === '/api/recommendations') {
        let body = '';
        req.on('data', chunk => {
          body += chunk.toString();
        });

        req.on('end', async () => {
          try {
            const data = JSON.parse(body);
            const result = await this.handleRecommendations(data);
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify(result));
          } catch (error) {
            console.error('Error processing recommendations:', error);
            res.writeHead(500, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ error: 'Internal server error', message: error.message }));
          }
        });
        return;
      }

      // 404 for all other routes
      res.writeHead(404, { 'Content-Type': 'text/plain' });
      res.end('Not Found');
    });
  }

  async handleRecommendations(data) {
    const { iteration, timestamp, automatedRecommendations } = data;
    console.log(`🔄 Received ${automatedRecommendations.length} automated recommendations for iteration ${iteration}`);

    const processedActions = [];

    for (const rec of automatedRecommendations) {
      const actionResult = await this.processRecommendation(rec, iteration);
      processedActions.push(actionResult);

      this.actionHistory.push({
        iteration,
        timestamp,
        recommendation: rec,
        result: actionResult,
        processedAt: new Date().toISOString()
      });
    }

    // Save action history
    await this.saveActionHistory();

    return {
      status: 'processed',
      iteration,
      actionsProcessed: processedActions.length,
      results: processedActions
    };
  }

  setupMiddleware() {
    this.app.use(express.json());
    this.app.use(express.urlencoded({ extended: true }));

    // CORS for local development
    this.app.use((req, res, next) => {
      res.header('Access-Control-Allow-Origin', '*');
      res.header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
      res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept, Authorization');
      if (req.method === 'OPTIONS') {
        res.sendStatus(200);
      } else {
        next();
      }
    });
  }

  setupRoutes() {
    // Health check
    this.app.get('/health', (req, res) => {
      res.json({
        status: 'healthy',
        service: 'recommendation-worker',
        port: this.port,
        uptime: process.uptime(),
        timestamp: new Date().toISOString(),
        actionsProcessed: this.actionHistory.length
      });
    });

    // Receive recommendations from PowerShell script
    this.app.post('/api/recommendations', async (req, res) => {
      try {
        const { iteration, timestamp, automatedRecommendations } = req.body;

        console.log(`🔄 Received ${automatedRecommendations.length} automated recommendations for iteration ${iteration}`);

        const processedActions = [];

        for (const rec of automatedRecommendations) {
          const actionResult = await this.processRecommendation(rec, iteration);
          processedActions.push(actionResult);

          this.actionHistory.push({
            iteration,
            timestamp,
            recommendation: rec,
            result: actionResult,
            processedAt: new Date().toISOString()
          });
        }

        // Save action history
        await this.saveActionHistory();

        res.json({
          status: 'processed',
          iteration,
          actionsProcessed: processedActions.length,
          results: processedActions
        });

      } catch (error) {
        console.error('Error processing recommendations:', error);
        res.status(500).json({
          status: 'error',
          message: error.message
        });
      }
    });

    // Get recommendation status
    this.app.get('/api/recommendations/status', async (req, res) => {
      try {
        const recentActions = this.actionHistory.slice(-10);
        const successRate = this.actionHistory.length > 0
          ? (this.actionHistory.filter(a => a.result.success).length / this.actionHistory.length) * 100
          : 0;

        res.json({
          totalActions: this.actionHistory.length,
          recentActions,
          successRate: Math.round(successRate),
          lastProcessed: this.actionHistory.length > 0
            ? this.actionHistory[this.actionHistory.length - 1].processedAt
            : null
        });
      } catch (error) {
        res.status(500).json({
          status: 'error',
          message: error.message
        });
      }
    });

    // Manual trigger for specific action types
    this.app.post('/api/recommendations/trigger/:actionType', async (req, res) => {
      try {
        const { actionType } = req.params;
        const { priority = 'medium', description = 'Manually triggered action' } = req.body;

        const mockRecommendation = {
          priority,
          type: 'manual',
          action: actionType,
          description,
          automated: true,
          suggestedChanges: req.body.changes || []
        };

        const result = await this.processRecommendation(mockRecommendation, 'manual');

        res.json({
          status: 'executed',
          action: actionType,
          result
        });
      } catch (error) {
        res.status(500).json({
          status: 'error',
          message: error.message
        });
      }
    });
  }

  async processRecommendation(recommendation, iteration) {
    console.log(`⚡ Processing: ${recommendation.action} (${recommendation.priority})`);

    const actionResult = {
      action: recommendation.action,
      priority: recommendation.priority,
      success: false,
      message: '',
      changesApplied: [],
      duration: 0
    };

    const startTime = Date.now();

    try {
      switch (recommendation.action) {
        case 'optimize_typescript_config':
          actionResult.success = await this.optimizeTypeScriptConfig(recommendation);
          actionResult.message = 'TypeScript configuration optimized';
          break;

        case 'optimize_go_binary':
          actionResult.success = await this.optimizeGoBinary(recommendation);
          actionResult.message = 'Go build optimization flags updated';
          break;

        case 'fix_typescript_errors':
          actionResult.success = await this.fixTypeScriptErrors(recommendation);
          actionResult.message = 'Attempted TypeScript error fixes';
          break;

        case 'fix_go_build':
          actionResult.success = await this.fixGoBuild(recommendation);
          actionResult.message = 'Go build dependencies updated';
          break;

        case 'escalate_typescript_investigation':
          actionResult.success = await this.escalateInvestigation(recommendation);
          actionResult.message = 'Investigation escalated to development team';
          break;

        default:
          actionResult.success = false;
          actionResult.message = `Unknown action: ${recommendation.action}`;
      }

    } catch (error) {
      actionResult.success = false;
      actionResult.message = `Error: ${error.message}`;
      console.error(`❌ Failed to process ${recommendation.action}:`, error);
    }

    actionResult.duration = Date.now() - startTime;

    console.log(`${actionResult.success ? '✅' : '❌'} ${recommendation.action}: ${actionResult.message} (${actionResult.duration}ms)`);

    return actionResult;
  }

  async optimizeTypeScriptConfig(recommendation) {
    // Read and update tsconfig.json with performance optimizations
    const tsconfigPath = path.join(__dirname, '../sveltekit-frontend/tsconfig.json');

    try {
      const configContent = await fs.readFile(tsconfigPath, 'utf8');
      const config = JSON.parse(configContent);

      // Apply optimizations
      if (!config.compilerOptions.incremental) {
        config.compilerOptions.incremental = true;
        config.compilerOptions.tsBuildInfoFile = '.tsbuildinfo';
      }

      if (!config.compilerOptions.skipLibCheck) {
        config.compilerOptions.skipLibCheck = true;
      }

      await fs.writeFile(tsconfigPath, JSON.stringify(config, null, 2));
      return true;
    } catch (error) {
      console.error('Failed to optimize TypeScript config:', error);
      return false;
    }
  }

  async optimizeGoBinary(recommendation) {
    // Update Go build scripts with optimization flags
    const buildScript = `go build -ldflags="-w -s" -tags legacy -o ./bin/simd-parser.exe ./simd_parser.go`;

    // Write optimized build command to a script file
    const scriptPath = path.join(__dirname, '../go-microservice/build-optimized.sh');
    await fs.writeFile(scriptPath, buildScript);

    return true;
  }

  async fixTypeScriptErrors(recommendation) {
    // Attempt basic TypeScript fixes
    console.log('Running TypeScript error analysis...');

    // Could integrate with existing fix scripts
    // For now, just log the errors for manual review
    if (recommendation.errors) {
      await this.logIssue('typescript-errors', recommendation.errors);
    }

    return true;
  }

  async fixGoBuild(recommendation) {
    // Run go mod tidy and update dependencies
    const { exec } = require('child_process');
    const util = require('util');
    const execAsync = util.promisify(exec);

    try {
      const goDir = path.join(__dirname, '../go-microservice');
      await execAsync('go mod tidy', { cwd: goDir });
      return true;
    } catch (error) {
      console.error('Go mod tidy failed:', error);
      return false;
    }
  }

  async escalateInvestigation(recommendation) {
    // Create detailed issue report for manual review
    const issueReport = {
      type: 'escalated_investigation',
      pattern: recommendation.pattern,
      description: recommendation.description,
      timestamp: new Date().toISOString(),
      requiresManualReview: true
    };

    await this.logIssue('escalated-investigations', issueReport);
    return true;
  }

  async logIssue(category, data) {
    const logPath = path.join(this.logDir, `${category}.json`);

    let logs = [];
    try {
      const existing = await fs.readFile(logPath, 'utf8');
      logs = JSON.parse(existing);
    } catch (error) {
      // File doesn't exist yet, start fresh
    }

    logs.push({
      timestamp: new Date().toISOString(),
      data
    });

    // Keep only last 50 entries
    if (logs.length > 50) {
      logs = logs.slice(-50);
    }

    await fs.writeFile(logPath, JSON.stringify(logs, null, 2));
  }

  async saveActionHistory() {
    const historyPath = path.join(this.logDir, 'action-history.json');

    // Keep only last 100 actions
    const recentHistory = this.actionHistory.slice(-100);

    await fs.writeFile(historyPath, JSON.stringify(recentHistory, null, 2));
  }

  async start() {
    // Ensure logs directory exists
    try {
      await fs.access(this.logDir);
    } catch {
      await fs.mkdir(this.logDir, { recursive: true });
    }

    // Load existing action history
    try {
      const historyPath = path.join(this.logDir, 'action-history.json');
      const historyContent = await fs.readFile(historyPath, 'utf8');
      this.actionHistory = JSON.parse(historyContent);
      console.log(`📚 Loaded ${this.actionHistory.length} previous actions`);
    } catch (error) {
      console.log('📚 Starting with fresh action history');
    }

    return new Promise((resolve, reject) => {
      if (this.useNativeHttp) {
        this.server.listen(this.port, (error) => {
          if (error) {
            reject(error);
          } else {
            console.log(`🤖 Recommendation Worker running on port ${this.port} (native HTTP)`);
            console.log(`💡 Ready to process automated recommendations`);
            resolve();
          }
        });
      } else {
        this.server = this.app.listen(this.port, (error) => {
          if (error) {
            reject(error);
          } else {
            console.log(`🤖 Recommendation Worker running on port ${this.port} (Express)`);
            console.log(`💡 Ready to process automated recommendations`);
            resolve();
          }
        });
      }
    });
  }

  stop() {
    if (this.server) {
      this.server.close();
    }
  }
}

// Start the worker if run directly
if (require.main === module) {
  const worker = new RecommendationWorker({
    port: process.env.WORKER_PORT || 4100,
    logDir: process.env.LOG_DIR || 'logs'
  });

  worker.start().catch(console.error);

  // Graceful shutdown
  process.on('SIGINT', () => {
    console.log('\n🛑 Shutting down Recommendation Worker...');
    worker.stop();
    process.exit(0);
  });
}

module.exports = RecommendationWorker;
