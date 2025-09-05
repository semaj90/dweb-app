/**
 * Multi-Node Cluster Architecture for High-Performance PDF Processing
 * Supports concurrent multi-core processing with intelligent load balancing
 */

const cluster = require('cluster');
const os = require('os');
const express = require('express');
const Redis = require('ioredis');
const { Worker } = require('worker_threads');
const fs = require('fs').promises;
const path = require('path');

const CLUSTER_CONFIG = {
    maxWorkers: os.cpus().length,
    redisUrl: process.env.REDIS_URL || 'redis://localhost:6379',
    queueName: 'pdf_processing_queue',
    concurrentJobs: 4,
    gpuEnabled: process.env.GPU_ENABLED === 'true'
};

class ClusterManager {
    constructor() {
        this.redis = new Redis(CLUSTER_CONFIG.redisUrl);
        this.workers = new Map();
        this.jobQueue = [];
        this.processing = new Set();
    }

    async initialize() {
        if (cluster.isMaster) {
            console.log(`🚀 Master ${process.pid} starting ${CLUSTER_CONFIG.maxWorkers} workers`);
            
            // Create worker processes
            for (let i = 0; i < CLUSTER_CONFIG.maxWorkers; i++) {
                this.createWorker();
            }

            // Handle worker lifecycle
            cluster.on('exit', (worker, code, signal) => {
                console.log(`⚠️ Worker ${worker.process.pid} died. Restarting...`);
                this.createWorker();
            });

            // Start job dispatcher
            this.startJobDispatcher();
            
        } else {
            // Worker process
            this.startWorkerProcess();
        }
    }

    createWorker() {
        const worker = cluster.fork();
        this.workers.set(worker.id, {
            instance: worker,
            busy: false,
            jobs: 0,
            created: Date.now()
        });

        worker.on('message', (msg) => {
            if (msg.type === 'job_complete') {
                this.handleJobComplete(worker.id, msg.result);
            }
        });
    }

    async startJobDispatcher() {
        const app = express();
        app.use(express.json({ limit: '50mb' }));

        // Multi-PDF processing endpoint
        app.post('/api/process-pdfs', async (req, res) => {
            try {
                const { files, options = {} } = req.body;
                const jobId = `job_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

                console.log(`📄 Received ${files.length} PDFs for processing (Job: ${jobId})`);

                // Queue jobs for parallel processing
                const jobs = files.map((file, index) => ({
                    id: `${jobId}_${index}`,
                    type: 'pdf_ocr',
                    file,
                    options: {
                        ...options,
                        ocrEngine: options.ocrEngine || 'tesseract',
                        enhanceRag: options.enhanceRag || true,
                        legalAnalysis: options.legalAnalysis || true
                    }
                }));

                // Distribute jobs across available workers
                const results = await this.distributeJobs(jobs);

                res.json({
                    success: true,
                    jobId,
                    processed: results.length,
                    results: results.map(r => ({
                        id: r.id,
                        filename: r.filename,
                        summary: r.summary,
                        entities: r.entities,
                        prosecutionScore: r.prosecutionScore,
                        processingTime: r.processingTime
                    }))
                });

            } catch (error) {
                console.error('❌ PDF processing failed:', error);
                res.status(500).json({
                    success: false,
                    error: error.message
                });
            }
        });

        // Cluster status endpoint
        app.get('/api/cluster/status', (req, res) => {
            const workers = Array.from(this.workers.values()).map(w => ({
                id: w.instance.id,
                pid: w.instance.process.pid,
                busy: w.busy,
                jobs: w.jobs,
                uptime: Date.now() - w.created
            }));

            res.json({
                master: process.pid,
                workers,
                totalWorkers: workers.length,
                busyWorkers: workers.filter(w => w.busy).length,
                queueLength: this.jobQueue.length,
                processing: this.processing.size
            });
        });

        const PORT = process.env.CLUSTER_PORT || 3001;
        app.listen(PORT, () => {
            console.log(`🎯 Cluster master listening on port ${PORT}`);
        });
    }

    async distributeJobs(jobs) {
        return new Promise((resolve, reject) => {
            let completed = 0;
            const results = [];
            const startTime = Date.now();

            jobs.forEach((job, index) => {
                // Find least busy worker
                const availableWorker = this.findAvailableWorker();
                
                if (availableWorker) {
                    this.assignJob(availableWorker, job, (result) => {
                        results[index] = result;
                        completed++;
                        
                        if (completed === jobs.length) {
                            console.log(`✅ All ${jobs.length} jobs completed in ${Date.now() - startTime}ms`);
                            resolve(results.filter(Boolean));
                        }
                    });
                } else {
                    // Queue job if no workers available
                    this.jobQueue.push({ job, callback: (result) => {
                        results[index] = result;
                        completed++;
                        
                        if (completed === jobs.length) {
                            resolve(results.filter(Boolean));
                        }
                    }});
                }
            });
        });
    }

    findAvailableWorker() {
        for (const [id, worker] of this.workers) {
            if (!worker.busy) {
                return worker;
            }
        }
        return null;
    }

    assignJob(worker, job, callback) {
        worker.busy = true;
        worker.jobs++;
        this.processing.add(job.id);

        worker.instance.send({
            type: 'process_job',
            job
        });

        // Store callback for result handling
        worker.callback = callback;
    }

    handleJobComplete(workerId, result) {
        const worker = this.workers.get(workerId);
        if (worker) {
            worker.busy = false;
            this.processing.delete(result.id);
            
            if (worker.callback) {
                worker.callback(result);
                delete worker.callback;
            }

            // Process queued jobs
            if (this.jobQueue.length > 0) {
                const { job, callback } = this.jobQueue.shift();
                this.assignJob(worker, job, callback);
            }
        }
    }

    startWorkerProcess() {
        console.log(`👷 Worker ${process.pid} started`);

        process.on('message', async (msg) => {
            if (msg.type === 'process_job') {
                try {
                    const result = await this.processJob(msg.job);
                    process.send({
                        type: 'job_complete',
                        result
                    });
                } catch (error) {
                    process.send({
                        type: 'job_complete',
                        result: {
                            id: msg.job.id,
                            error: error.message,
                            success: false
                        }
                    });
                }
            }
        });
    }

    async processJob(job) {
        const startTime = Date.now();
        console.log(`🔄 Worker ${process.pid} processing job ${job.id}`);

        try {
            // Create dedicated worker thread for intensive OCR processing
            const result = await new Promise((resolve, reject) => {
                const worker = new Worker(path.join(__dirname, 'pdf-worker.js'), {
                    workerData: job
                });

                worker.on('message', resolve);
                worker.on('error', reject);
                worker.on('exit', (code) => {
                    if (code !== 0) {
                        reject(new Error(`Worker stopped with exit code ${code}`));
                    }
                });
            });

            return {
                ...result,
                id: job.id,
                processingTime: Date.now() - startTime,
                worker: process.pid,
                success: true
            };

        } catch (error) {
            console.error(`❌ Job ${job.id} failed:`, error);
            return {
                id: job.id,
                error: error.message,
                processingTime: Date.now() - startTime,
                worker: process.pid,
                success: false
            };
        }
    }
}

// PDF Worker Thread Handler
const pdfWorkerCode = `
const { parentPort, workerData } = require('worker_threads');
const fs = require('fs');
const path = require('path');

// Import OCR and AI processing libraries
const Tesseract = require('tesseract.js');
const pdf2pic = require('pdf2pic');

async function processPDF(job) {
    const { file, options } = job;
    
    try {
        // Convert PDF to images for OCR
        const convert = pdf2pic.fromBuffer(Buffer.from(file.data, 'base64'), {
            density: 200,
            saveDir: '/tmp',
            saveName: \`pdf_\${job.id}\`,
            format: 'png',
            width: 2000,
            height: 2000
        });

        const pages = await convert.bulk(-1);
        
        // OCR each page in parallel
        const ocrPromises = pages.map(async (page, index) => {
            const { data: { text } } = await Tesseract.recognize(page.path, 'eng', {
                logger: m => console.log(\`Page \${index + 1}: \${m.status}\`)
            });
            
            // Clean up temp file
            fs.unlinkSync(page.path);
            
            return text;
        });

        const extractedTexts = await Promise.all(ocrPromises);
        const fullText = extractedTexts.join('\\n\\n');

        // Extract legal entities
        const entities = extractLegalEntities(fullText);
        
        // Generate summary using AI
        const summary = await generateSummary(fullText, options);
        
        // Calculate prosecution relevance score
        const prosecutionScore = calculateProsecutionScore(fullText, entities);

        return {
            filename: file.name,
            extractedText: fullText,
            summary,
            entities,
            prosecutionScore,
            pageCount: pages.length,
            wordCount: fullText.split(/\\s+/).length
        };

    } catch (error) {
        throw new Error(\`PDF processing failed: \${error.message}\`);
    }
}

function extractLegalEntities(text) {
    const patterns = {
        parties: /(?:plaintiff|defendant|appellant|appellee)\\s+([A-Z][a-z]+(?:\\s+[A-Z][a-z]+)*)/gi,
        cases: /([A-Z][a-z]+(?:\\s+[A-Z][a-z]+)*)\\s+v\\.\\s+([A-Z][a-z]+(?:\\s+[A-Z][a-z]+)*)/gi,
        statutes: /\\b\\d+\\s+U\\.?S\\.?C\\.?\\s+§?\\s*\\d+/gi,
        courts: /\\b(?:Supreme Court|Court of Appeals|District Court|Circuit Court)\\b/gi
    };

    const entities = {};
    for (const [type, pattern] of Object.entries(patterns)) {
        entities[type] = [...text.matchAll(pattern)].map(match => match[0]);
    }

    return entities;
}

async function generateSummary(text, options) {
    // Simulate AI summary generation
    // In production, this would call your Gemma3 legal model
    const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 20);
    const topSentences = sentences.slice(0, 3).join('. ');
    
    return {
        brief: topSentences,
        keyPoints: extractKeyPoints(text),
        legalIssues: identifyLegalIssues(text)
    };
}

function calculateProsecutionScore(text, entities) {
    let score = 0.5; // Base score
    
    // Boost for criminal law indicators
    const criminalIndicators = ['criminal', 'prosecution', 'guilty', 'innocent', 'sentence', 'conviction'];
    const matches = criminalIndicators.filter(term => 
        text.toLowerCase().includes(term)
    ).length;
    
    score += (matches / criminalIndicators.length) * 0.3;
    
    // Boost for legal entity complexity
    const totalEntities = Object.values(entities).flat().length;
    score += Math.min(totalEntities / 20, 0.2);
    
    return Math.min(0.95, score);
}

function extractKeyPoints(text) {
    // Extract sentences containing legal keywords
    const legalKeywords = ['court', 'judge', 'law', 'statute', 'ruling', 'decision'];
    const sentences = text.split(/[.!?]+/);
    
    return sentences
        .filter(sentence => 
            legalKeywords.some(keyword => 
                sentence.toLowerCase().includes(keyword)
            )
        )
        .slice(0, 5)
        .map(s => s.trim());
}

function identifyLegalIssues(text) {
    const issuePatterns = [
        'breach of contract',
        'negligence',
        'constitutional violation',
        'trademark infringement',
        'patent dispute',
        'employment discrimination'
    ];

    return issuePatterns.filter(issue => 
        text.toLowerCase().includes(issue)
    );
}

// Main processing
processPDF(workerData)
    .then(result => parentPort.postMessage(result))
    .catch(error => parentPort.postMessage({ error: error.message }));
`;

// Write the PDF worker file
if (cluster.isMaster) {
    fs.writeFileSync(path.join(__dirname, 'pdf-worker.js'), pdfWorkerCode);
}

// Initialize cluster
const clusterManager = new ClusterManager();
clusterManager.initialize();

module.exports = ClusterManager;