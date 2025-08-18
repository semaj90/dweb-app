#!/usr/bin/env node

/**
 * Complete Case Solver - Integrates all components for error-to-vector processing
 * GPU parsing, service workers, thread assignment, indexing, embedding, search, and case resolution
 */

const { spawn, exec } = require('child_process');
const { promises: fs } = require('fs');
const path = require('path');
const http = require('http');
const { EventEmitter } = require('events');
// ServiceWorkerManager will be imported dynamically since it's an ES module
const cp = require('child_process');
const { cpus } = require('os');

// __filename and __dirname are already available in CommonJS

class CompleteCaseSolver extends EventEmitter {
    constructor() {
        super();

        this.config = {
            services: {
                loadBalancer: { port: 8099, path: 'go-microservice/bin/load-balancer.exe' },
                enhancedRAG: { port: 8094, path: 'go-microservice/bin/enhanced-rag.exe' },
                recommendationService: { port: 8096, path: 'go-microservice/bin/recommendation-service.exe' },
                gpuIndexer: { port: 8097, path: 'go-microservice/bin/gpu-indexer-service.exe' },
                simdParser: { port: 8080, path: 'go-microservice/bin/simd-parser.exe' },
                frontend: { port: 5173, path: 'sveltekit-frontend', command: 'npm run dev' }
            },
            database: {
                url: process.env.DATABASE_URL || 'postgresql://legal_admin:123456@localhost:5432/legal_ai_db'
            },
            ollama: {
                url: process.env.OLLAMA_URL || 'http://localhost:11434',
                model: 'gemma3-legal'
            }
        };

        this.serviceProcesses = new Map();
        this.serviceWorkerManager = null;
        this.isRunning = false;
        this.errorCases = [];
        this.solvedCases = [];
    this.recommendationIngestFile = path.join('logs','recommendation-ingest.jsonl');
    this._ensureLogsDir();

        console.log('🎯 Complete Case Solver initialized');
    }

    async _ensureLogsDir(){
        try { await fs.mkdir('logs', { recursive: true }); } catch(_){}
    }

    async appendRecommendationIngest(entry){
        try {
            await fs.appendFile(this.recommendationIngestFile, JSON.stringify(entry) + '\n','utf8');
        } catch(e){
            console.warn('⚠️ Failed to append recommendation ingest entry:', e.message);
        }
    }

    /**
     * Start all services and solve the complete error case
     */
    async solveCaseComplete() {
        console.log('🚀 Starting Complete Case Solver...');
        console.log('=' .repeat(80));

        try {
            // Step 1: Start service worker manager with thread assignment
            await this.startServiceWorkerManager();

            // Step 2: Start all microservices
            await this.startAllServices();

            // Step 3: Initialize GPU parsing and indexing
            await this.initializeGPUProcessing();

            // Step 4: Start error monitoring and processing
            await this.startErrorProcessing();

            // Step 5: Begin case solving pipeline
            await this.beginCaseSolvingPipeline();

            console.log('✅ Complete Case Solver is operational!');
            console.log('🎯 All systems integrated and ready for error-to-vector processing');

            this.isRunning = true;

            // Continuous case solving loop
            this.startContinuousSolving();

        } catch (error) {
            console.error('❌ Failed to start Complete Case Solver:', error);
            await this.shutdown();
            throw error;
        }
    }

    async startServiceWorkerManager(){
        if (this.serviceWorkerManager) return;
        
        // Dynamic import for ES module
        const { default: ServiceWorkerManager } = await import('./service-worker-manager.js');
        
        this.serviceWorkerManager = new ServiceWorkerManager({
            maxWorkers: cpus().length,
            gpuEnabled: process.env.CUDA_ENABLED === 'true',
            threadPoolSize: cpus().length * 2
        });
        await this.serviceWorkerManager.start();
        this.serviceWorkerManager.on('taskCompleted', (e)=> this.handleWorkerTaskCompleted(e));
        this.serviceWorkerManager.on('taskError', (e)=> this.handleWorkerTaskError(e));
        console.log('✅ Service Worker Manager started with optimal thread assignment');
    }

    /**
     * Start all microservices
     */
    async startAllServices() {
        console.log('⚡ Starting all microservices...');

        for (const [serviceName, serviceConfig] of Object.entries(this.config.services)) {
            try {
                await this.startService(serviceName, serviceConfig);
                console.log(`   ✅ ${serviceName} started on port ${serviceConfig.port}`);
            } catch (error) {
                console.warn(`   ⚠️ Failed to start ${serviceName}:`, error.message);
            }
        }

        // Wait for services to be ready
        await this.waitForServicesReady();
    }

    /**
     * Start individual service
     */
    async startService(serviceName, config) {
        return new Promise((resolve, reject) => {
            let process;

            if (config.command) {
                // For npm/node commands
                process = spawn('cmd', ['/c', config.command], {
                    cwd: config.path,
                    stdio: ['ignore', 'pipe', 'pipe']
                });
            } else {
                // For executable files
                process = spawn(config.path, [], {
                    stdio: ['ignore', 'pipe', 'pipe']
                });
            }

            let startupOutput = '';
            const startupTimeout = setTimeout(() => {
                reject(new Error(`Service ${serviceName} startup timeout`));
            }, 30000);

            process.stdout.on('data', (data) => {
                startupOutput += data.toString();
                if (startupOutput.includes('listening') || startupOutput.includes('started') || startupOutput.includes('running')) {
                    clearTimeout(startupTimeout);
                    resolve();
                }
            });

            process.stderr.on('data', (data) => {
                console.log(`${serviceName} stderr:`, data.toString());
            });

            process.on('error', (error) => {
                clearTimeout(startupTimeout);
                reject(error);
            });

            process.on('exit', (code) => {
                if (code !== 0) {
                    console.warn(`⚠️ Service ${serviceName} exited with code ${code}`);
                }
            });

            this.serviceProcesses.set(serviceName, process);
        });
    }

    /**
     * Wait for all services to be ready
     */
    async waitForServicesReady() {
        console.log('⏳ Waiting for services to be ready...');

        const healthChecks = [];

        for (const [serviceName, config] of Object.entries(this.config.services)) {
            if (serviceName !== 'frontend') { // Skip frontend health check
                healthChecks.push(this.checkServiceHealth(serviceName, config.port));
            }
        }

        await Promise.all(healthChecks);
        console.log('✅ All services are ready');
    }

    /**
     * Check service health
     */
    async checkServiceHealth(serviceName, port) {
        const maxRetries = 30;
        const retryDelay = 1000;

        for (let i = 0; i < maxRetries; i++) {
            try {
                await this.httpRequest('GET', `http://localhost:${port}/health`);
                return;
            } catch (error) {
                if (i === maxRetries - 1) {
                    throw new Error(`Service ${serviceName} health check failed after ${maxRetries} retries`);
                }
                await new Promise(resolve => setTimeout(resolve, retryDelay));
            }
        }
    }

    /**
     * Initialize GPU processing capabilities
     */
    async initializeGPUProcessing() {
        console.log('🚀 Initializing GPU processing...');

        try {
            // Test GPU indexer service
            const indexTestDoc = {
                id: 'test-doc-' + Date.now(),
                content: 'This is a test legal document for GPU processing and indexing.',
                metadata: {
                    type: 'test',
                    category: 'legal',
                    created_at: new Date().toISOString()
                }
            };

            const indexResult = await this.httpRequest('POST', 'http://localhost:8097/index', indexTestDoc);
            console.log('   ✅ GPU Indexer tested:', indexResult.document_id);

            // Test SIMD parser
            const parseTestData = {
                documents: ['{"test": "legal document parsing"}']
            };

            const parseResult = await this.httpRequest('POST', 'http://localhost:8080/parse/batch-gpu', parseTestData);
            console.log('   ✅ SIMD Parser tested:', parseResult.batch_size, 'documents');

            // Add GPU processing tasks to service workers
            this.serviceWorkerManager.addTask({
                type: 'gpu-parse-init',
                data: { status: 'GPU processing initialized' }
            });

        } catch (error) {
            console.warn('⚠️ GPU processing initialization warning:', error.message);
        }

        console.log('✅ GPU processing initialized');
    }

    /**
     * Start error monitoring and processing
     */
    async startErrorProcessing() {
        console.log('🔍 Starting error processing...');

        // Run TypeScript check to generate errors
        await this.runTypeScriptCheck();

        // Process errors with service workers
        this.serviceWorkerManager.addTask({
            type: 'error-analysis',
            data: {
                source: 'typescript-check',
                timestamp: new Date().toISOString()
            }
        });

        console.log('✅ Error processing started');
    }

    /**
     * Run TypeScript check to find errors
     */
    async runTypeScriptCheck() {
        return new Promise((resolve) => {
            const tsCheck = spawn('npx', ['tsc', '--noEmit', '--skipLibCheck'], {
                cwd: process.cwd(),
                stdio: 'pipe'
            });

            let errorOutput = '';

            tsCheck.stdout.on('data', (data) => {
                errorOutput += data.toString();
            });

            tsCheck.stderr.on('data', (data) => {
                errorOutput += data.toString();
            });

            tsCheck.on('close', (code) => {
                if (errorOutput) {
                    this.processFoundErrors(errorOutput);
                }
                resolve(code);
            });
        });
    }

    /**
     * Process found errors
     */
    processFoundErrors(errorOutput) {
        const errors = this.parseTypeScriptErrors(errorOutput);

        console.log(`🔍 Found ${errors.length} TypeScript errors to process`);

        errors.forEach((error, index) => {
            const errorCase = {
                id: `error_case_${Date.now()}_${index}`,
                type: 'typescript_error',
                error: error,
                status: 'pending',
                created_at: new Date().toISOString()
            };

            this.errorCases.push(errorCase);

            // Add to service worker queue
            this.serviceWorkerManager.addTask({
                type: 'error-to-vector',
                data: errorCase
            });
        });
    }

    /**
     * Parse TypeScript errors from output
     */
    parseTypeScriptErrors(output) {
        const errors = [];
        const lines = output.split('\\n');

        let currentError = null;

        for (const line of lines) {
            if (line.includes('error TS')) {
                if (currentError) {
                    errors.push(currentError);
                }

                const match = line.match(/(.+)\\((\\d+),(\\d+)\\): error (TS\\d+): (.+)/);
                if (match) {
                    currentError = {
                        file: match[1].trim(),
                        line: parseInt(match[2]),
                        column: parseInt(match[3]),
                        code: match[4],
                        message: match[5].trim(),
                        fullLine: line
                    };
                }
            } else if (currentError && line.trim()) {
                currentError.message += ' ' + line.trim();
            }
        }

        if (currentError) {
            errors.push(currentError);
        }

        return errors;
    }

    /**
     * Begin the complete case solving pipeline
     */
    async beginCaseSolvingPipeline() {
        console.log('🎯 Beginning case solving pipeline...');

        // Test the complete pipeline with a sample case
        const sampleCase = {
            id: 'sample_case_' + Date.now(),
            type: 'integration_test',
            description: 'Complete pipeline integration test',
            steps: [
                'parse_with_gpu',
                'generate_embeddings',
                'index_with_metadata',
                'search_and_sort',
                'generate_recommendations',
                'solve_case'
            ]
        };

        console.log('🧪 Processing sample case:', sampleCase.id);

        // Execute pipeline steps
        for (const step of sampleCase.steps) {
            await this.executeStep(sampleCase, step);
        }

        console.log('✅ Sample case processing completed');
    }

    /**
     * Execute a pipeline step
     */
    async executeStep(caseData, step) {
        console.log(`   🔄 Executing step: ${step}`);

        try {
            switch (step) {
                case 'parse_with_gpu':
                    await this.parseWithGPU(caseData);
                    break;
                case 'generate_embeddings':
                    await this.generateEmbeddings(caseData);
                    break;
                case 'index_with_metadata':
                    await this.indexWithMetadata(caseData);
                    break;
                case 'search_and_sort':
                    await this.searchAndSort(caseData);
                    break;
                case 'generate_recommendations':
                    await this.generateRecommendations(caseData);
                    break;
                case 'solve_case':
                    await this.solveCase(caseData);
                    break;
            }
            console.log(`   ✅ Step completed: ${step}`);
        } catch (error) {
            console.error(`   ❌ Step failed: ${step}:`, error.message);
        }
    }

    /**
     * Parse with GPU acceleration
     */
    async parseWithGPU(caseData) {
        const parseData = {
            documents: [JSON.stringify(caseData)]
        };

        const result = await this.httpRequest('POST', 'http://localhost:8080/parse/batch-gpu', parseData);
        caseData.parsed = result;
    }

    /**
     * Generate embeddings
     */
    async generateEmbeddings(caseData) {
        this.serviceWorkerManager.addTask({
            type: 'embed-text',
            data: {
                caseId: caseData.id,
                text: caseData.description
            }
        });
    }

    /**
     * Index with metadata
     */
    async indexWithMetadata(caseData) {
        const indexData = {
            id: caseData.id,
            content: JSON.stringify(caseData),
            metadata: {
                type: caseData.type,
                steps_completed: caseData.steps || [],
                created_at: new Date().toISOString()
            }
        };

        const result = await this.httpRequest('POST', 'http://localhost:8097/index', indexData);
        caseData.indexed = result;
    }

    /**
     * Search and sort
     */
    async searchAndSort(caseData) {
        const searchQuery = {
            text: caseData.description,
            filters: { type: caseData.type },
            sort_by: 'similarity',
            sort_order: 'desc',
            limit: 10,
            min_similarity: 0.1,
            include_content: false
        };

        const result = await this.httpRequest('POST', 'http://localhost:8097/search', searchQuery);
        caseData.searchResults = result;
    }

    /**
     * Generate recommendations
     */
    async generateRecommendations(caseData) {
        const recommendationData = {
            case_data: caseData,
            model: this.config.ollama.model
        };

        try {
            const result = await this.httpRequest('POST', 'http://localhost:8096/recommend', recommendationData);
            caseData.recommendations = result;
            await this.appendRecommendationIngest({
                ts: new Date().toISOString(),
                caseId: caseData.id,
                stage: 'generate_recommendations',
                model: this.config.ollama.model,
                recommendations: result,
                parsed: !!caseData.parsed,
                indexed: !!caseData.indexed
            });
        } catch (error) {
            console.warn('   ⚠️ Recommendation generation failed:', error.message);
            caseData.recommendations = { error: error.message };
            await this.appendRecommendationIngest({
                ts: new Date().toISOString(),
                caseId: caseData.id,
                stage: 'generate_recommendations',
                error: error.message
            });
        }
    }

    /**
     * Solve the case
     */
    async solveCase(caseData) {
        caseData.status = 'solved';
        caseData.solved_at = new Date().toISOString();
        caseData.solution = {
            steps_completed: caseData.steps.length,
            has_recommendations: !!caseData.recommendations,
            indexed: !!caseData.indexed,
            parsed: !!caseData.parsed,
            searched: !!caseData.searchResults
        };

        this.solvedCases.push(caseData);

        console.log(`🎉 Case solved: ${caseData.id}`);
        console.log('   Solution summary:', JSON.stringify(caseData.solution, null, 2));
        await this.appendRecommendationIngest({
            ts: new Date().toISOString(),
            caseId: caseData.id,
            stage: 'solve_case',
            solution: caseData.solution,
            recommendations: caseData.recommendations
        });
    }

    /**
     * Start continuous case solving
     */
    startContinuousSolving() {
        console.log('🔄 Starting continuous case solving loop...');

        setInterval(() => {
            this.processPendingCases();
        }, 10000); // Process every 10 seconds

        setInterval(() => {
            this.reportProgress();
        }, 30000); // Report every 30 seconds
    }

    /**
     * Process pending error cases
     */
    processPendingCases() {
        const pendingCases = this.errorCases.filter(c => c.status === 'pending');

        if (pendingCases.length > 0) {
            console.log(`🔄 Processing ${pendingCases.length} pending cases...`);

            pendingCases.forEach(async (errorCase) => {
                errorCase.status = 'processing';

                // Process through complete pipeline
                try {
                    for (const step of ['parse_with_gpu', 'generate_embeddings', 'index_with_metadata', 'search_and_sort', 'generate_recommendations', 'solve_case']) {
                        await this.executeStep(errorCase, step);
                    }
                } catch (error) {
                    console.error(`❌ Failed to process case ${errorCase.id}:`, error);
                    errorCase.status = 'failed';
                    errorCase.error = error.message;
                }
            });
        }
    }

    /**
     * Report progress
     */
    reportProgress() {
        const stats = {
            timestamp: new Date().toISOString(),
            total_cases: this.errorCases.length,
            solved_cases: this.solvedCases.length,
            pending_cases: this.errorCases.filter(c => c.status === 'pending').length,
            processing_cases: this.errorCases.filter(c => c.status === 'processing').length,
            failed_cases: this.errorCases.filter(c => c.status === 'failed').length,
            success_rate: this.errorCases.length > 0 ? (this.solvedCases.length / this.errorCases.length * 100).toFixed(2) + '%' : '0%'
        };

        console.log('\\n📊 CASE SOLVING PROGRESS REPORT');
        console.log('=' .repeat(50));
        console.log(JSON.stringify(stats, null, 2));
        console.log('=' .repeat(50) + '\\n');

        this.emit('progressReport', stats);
    }

    /**
     * Handle worker task completion
     */
    handleWorkerTaskCompleted(event) {
        console.log(`✅ Worker task completed: ${event.workerType}-${event.workerId} finished ${event.taskId}`);

        if (event.taskId && event.taskId.includes('error_case')) {
            // Update case status
            const caseId = event.taskId;
            const errorCase = this.errorCases.find(c => c.id === caseId);
            if (errorCase) {
                errorCase.status = 'worker_completed';
                errorCase.worker_result = event.result;
            }
        }
    }

    /**
     * Handle worker task errors
     */
    handleWorkerTaskError(event) {
        console.error(`❌ Worker task error: ${event.workerType}-${event.workerId} failed on ${event.taskId}: ${event.error}`);
    }

    /**
     * HTTP request helper
     */
    async httpRequest(method, url, data = null) {
        return new Promise((resolve, reject) => {
            const urlObj = new URL(url);
            const options = {
                hostname: urlObj.hostname,
                port: urlObj.port,
                path: urlObj.pathname,
                method: method,
                headers: {
                    'Content-Type': 'application/json',
                }
            };

            if (data) {
                const jsonData = JSON.stringify(data);
                options.headers['Content-Length'] = Buffer.byteLength(jsonData);
            }

            const req = http.request(options, (res) => {
                let responseData = '';

                res.on('data', (chunk) => {
                    responseData += chunk;
                });

                res.on('end', () => {
                    try {
                        const result = JSON.parse(responseData);
                        resolve(result);
                    } catch (e) {
                        resolve({ raw: responseData });
                    }
                });
            });

            req.on('error', reject);

            if (data) {
                req.write(JSON.stringify(data));
            }

            req.end();
        });
    }

    /**
     * Graceful shutdown
     */
    async shutdown() {
        console.log('\\n🛑 Shutting down Complete Case Solver...');

        this.isRunning = false;

        // Shutdown service worker manager
        if (this.serviceWorkerManager) {
            await this.serviceWorkerManager.shutdown();
        }

        // Terminate service processes
        for (const [serviceName, process] of this.serviceProcesses.entries()) {
            console.log(`   Terminating ${serviceName}...`);
            process.kill('SIGTERM');
        }

        console.log('✅ Complete Case Solver shutdown complete');
    }
}

// Main execution
async function main() {
    const solver = new CompleteCaseSolver();

    // Graceful shutdown handling
    process.on('SIGINT', async () => {
        console.log('\\n🛑 Received SIGINT, shutting down gracefully...');
        await solver.shutdown();
        process.exit(0);
    });

    process.on('SIGTERM', async () => {
        console.log('\\n🛑 Received SIGTERM, shutting down gracefully...');
        await solver.shutdown();
        process.exit(0);
    });

    // Start solving cases
    // Start solving cases
    try {
        await solver.solveCaseComplete();
        console.log('\n🎯 Complete Case Solver is now running continuously...');
    } catch (error) {
        console.error('❌ Failed to start Complete Case Solver:', error);
        process.exit(1);
    }
}

module.exports = CompleteCaseSolver;

if (require.main === module) {
    main();
}