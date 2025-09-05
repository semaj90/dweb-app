/**
 * Enhanced RAG System - Agent Orchestrator
 * Multi-Agent Coordination for Legal AI with LLM Optimization Patterns
 */

import { EventEmitter } from 'events';
import ClaudeAgent from './agents/claude.js';
import CrewAIAgent from './agents/crewai.js';
import GemmaAgent from './agents/gemma.js';
import OllamaAgent from './agents/ollama.js';
import ConfigLoader from './config-loader.js';

export class AgentOrchestrator extends EventEmitter {
    constructor(config = {}) {
        super();
        this.config = config;
        this.agents = new Map();
        this.isInitialized = false;
        this.activeJobs = new Map();
        this.jobCounter = 0;
        
        // Initialize configuration loader
        this.configLoader = new ConfigLoader(config.configPath);
        
        // Initialize agents based on configuration
        this.initializeAgents();
    }
    
    initializeAgents() {
        // Claude Agent
        if (this.config.claude?.enabled !== false) {
            this.agents.set('claude', new ClaudeAgent(this.config.claude || {}));
        }
        
        // CrewAI Agent
        if (this.config.crewai?.enabled !== false) {
            this.agents.set('crewai', new CrewAIAgent(this.config.crewai || {}));
        }
        
        // Gemma Agent
        if (this.config.gemma?.enabled !== false) {
            this.agents.set('gemma', new GemmaAgent(this.config.gemma || {}));
        }
        
        // Ollama Agent
        if (this.config.ollama?.enabled !== false) {
            this.agents.set('ollama', new OllamaAgent(this.config.ollama || {}));
        }
        
        // Set up event forwarding
        for (const [name, agent] of this.agents) {
            agent.on('error', (error) => this.emit('agent-error', { agent: name, ...error }));
            agent.on('initialized', (data) => this.emit('agent-initialized', { agent: name, ...data }));
        }
    }
    
    async initialize() {
        // Load configuration with LLM optimizations
        try {
            await this.configLoader.loadConfig();
            console.log('📋 Configuration loaded with LLM optimizations');
        } catch (error) {
            console.warn('⚠️ Failed to load advanced config, using default:', error.message);
        }
        
        const initPromises = [];
        
        for (const [name, agent] of this.agents) {
            initPromises.push(
                agent.initialize().catch(error => {
                    console.warn(`Failed to initialize ${name} agent:`, error.message);
                    return { agent: name, error: error.message };
                })
            );
        }
        
        const results = await Promise.allSettled(initPromises);
        this.isInitialized = true;
        
        const initializedAgents = [];
        const failedAgents = [];
        
        results.forEach((result, index) => {
            const agentName = Array.from(this.agents.keys())[index];
            if (result.status === 'fulfilled' && !result.value.error) {
                initializedAgents.push(agentName);
            } else {
                failedAgents.push(agentName);
                // Remove failed agents
                this.agents.delete(agentName);
            }
        });
        
        this.emit('orchestrator-initialized', {
            initializedAgents,
            failedAgents,
            totalAgents: initializedAgents.length,
            optimizations: this.configLoader.getSummary().optimizations
        });
        
        return {
            initialized: initializedAgents,
            failed: failedAgents,
            configSummary: this.configLoader.getSummary()
        };
    }
    
    async analyzeLegalDocument(document, options = {}) {
        if (!this.isInitialized) {
            await this.initialize();
        }
        
        const jobId = ++this.jobCounter;
        const job = {
            id: jobId,
            type: 'legal-document-analysis',
            document: document,
            options: options,
            results: {},
            startTime: Date.now(),
            status: 'running'
        };
        
        this.activeJobs.set(jobId, job);
        this.emit('job-started', { jobId, type: job.type });
        
        try {
            const selectedAgents = options.agents || Array.from(this.agents.keys());
            const analysisPromises = [];
            
            for (const agentName of selectedAgents) {
                if (this.agents.has(agentName)) {
                    const agent = this.agents.get(agentName);
                    
                    if (agent.analyzeLegalDocument) {
                        analysisPromises.push(
                            agent.analyzeLegalDocument(document, options)
                                .then(result => ({ agent: agentName, result }))
                                .catch(error => ({ agent: agentName, error: error.message }))
                        );
                    }
                }
            }
            
            const results = await Promise.allSettled(analysisPromises);
            
            results.forEach(result => {
                if (result.status === 'fulfilled') {
                    job.results[result.value.agent] = result.value.result || result.value.error;
                }
            });
            
            job.status = 'completed';
            job.endTime = Date.now();
            job.duration = job.endTime - job.startTime;
            
            // Synthesize results
            const synthesis = await this.synthesizeResults(job.results, 'legal-analysis');
            job.synthesis = synthesis;
            
            this.emit('job-completed', { jobId, results: job.results, synthesis });
            
            return {
                jobId,
                results: job.results,
                synthesis,
                duration: job.duration
            };
        } catch (error) {
            job.status = 'failed';
            job.error = error.message;
            this.emit('job-failed', { jobId, error: error.message });
            throw error;
        }
    }
    
    async synthesizeResults(results, type = 'general') {
        const validResults = Object.values(results).filter(result => 
            result && typeof result === 'object' && !result.error
        );
        
        if (validResults.length === 0) {
            return {
                type: 'synthesis',
                summary: 'No valid results to synthesize',
                confidence: 0,
                recommendations: []
            };
        }
        
        // Simple synthesis based on result type
        switch (type) {
            case 'legal-analysis':
                return this.synthesizeLegalAnalysis(validResults);
            default:
                return this.synthesizeGeneral(validResults);
        }
    }
    
    synthesizeLegalAnalysis(results) {
        const synthesis = {
            type: 'legal-analysis-synthesis',
            agentCount: results.length,
            commonFindings: [],
            recommendations: [],
            confidence: 0,
            summary: ''
        };
        
        // Extract common themes
        const findings = results.map(r => r.content || r.response || '').join(' ');
        
        // Simple keyword extraction for legal concepts
        const legalKeywords = [
            'contract', 'liability', 'compliance', 'breach', 'damages',
            'jurisdiction', 'statute', 'regulation', 'precedent', 'clause'
        ];
        
        synthesis.commonFindings = legalKeywords.filter(keyword => 
            findings.toLowerCase().includes(keyword)
        );
        
        synthesis.confidence = Math.min(results.length * 0.2, 1.0);
        synthesis.summary = `Analysis completed by ${results.length} agents. ` +
            `Key legal concepts identified: ${synthesis.commonFindings.join(', ')}.`;
        
        synthesis.recommendations = [
            'Review findings from multiple agents for consistency',
            'Validate legal conclusions with qualified attorney',
            'Consider jurisdiction-specific requirements'
        ];
        
        return synthesis;
    }
    
    synthesizeGeneral(results) {
        return {
            type: 'general-synthesis',
            agentCount: results.length,
            confidence: Math.min(results.length * 0.15, 1.0),
            summary: `Analysis completed by ${results.length} agents with varying perspectives.`,
            recommendations: ['Review individual agent results', 'Consider expert validation']
        };
    }
    
    getAgentStatus() {
        const status = {};
        for (const [name, agent] of this.agents) {
            status[name] = agent.getStatus ? agent.getStatus() : { agent: name, status: 'unknown' };
        }
        return status;
    }
    
    getJobStatus(jobId) {
        return this.activeJobs.get(jobId);
    }
    
    getAllJobs() {
        return Array.from(this.activeJobs.values());
    }
    
    /**
     * Process streaming tokens with optimization patterns
     */
    async processStreamingTokens(tokens, options = {}) {
        if (!this.configLoader.isLoaded) {
            // Fallback to basic processing
            return tokens.map(token => ({ token, timestamp: Date.now() }));
        }
        
        try {
            // Use worker thread for optimized token processing
            const result = await this.configLoader.processTokensWithWorker(tokens, 'processTokens');
            
            this.emit('tokens-processed', {
                count: tokens.length,
                compressed: result.length < tokens.length,
                worker: true
            });
            
            return result;
        } catch (error) {
            console.warn('⚠️ Worker processing failed, using fallback:', error.message);
            return tokens.map(token => ({ token, timestamp: Date.now() }));
        }
    }
    
    /**
     * Compress token payload for transmission
     */
    async compressTokens(tokens) {
        if (!this.configLoader.isLoaded) {
            return { compressed: false, data: tokens };
        }
        
        try {
            const result = await this.configLoader.processTokensWithWorker(tokens, 'compactTokens');
            
            this.emit('tokens-compressed', {
                originalSize: JSON.stringify(tokens).length,
                compressedSize: result.compact.length,
                ratio: result.ratio
            });
            
            return {
                compressed: true,
                data: result,
                savings: `${Math.round((1 - result.ratio) * 100)}%`
            };
        } catch (error) {
            console.warn('⚠️ Token compression failed:', error.message);
            return { compressed: false, data: tokens };
        }
    }
    
    /**
     * Execute workflow with optimization patterns
     */
    async executeWorkflow(workflowName, input, options = {}) {
        const workflow = this.configLoader.getWorkflow(workflowName);
        if (!workflow) {
            throw new Error(`Workflow not found: ${workflowName}`);
        }
        
        const jobId = ++this.jobCounter;
        const job = {
            id: jobId,
            type: 'workflow-execution',
            workflow: workflowName,
            input,
            options,
            results: {},
            startTime: Date.now(),
            status: 'running'
        };
        
        this.activeJobs.set(jobId, job);
        this.emit('workflow-started', { jobId, workflow: workflowName });
        
        try {
            const results = {};
            
            // Execute workflow steps
            for (const step of workflow.steps) {
                if (step.type === 'aggregation') {
                    // Synthesis step
                    results[step.name] = await this.synthesizeResults(results, step.method);
                } else {
                    // Agent step
                    const agent = this.agents.get(step.agent);
                    if (agent && agent[step.tasks[0]]) {
                        results[step.name] = await agent[step.tasks[0]](input, options);
                    }
                }
                
                // Emit progress
                this.emit('workflow-progress', {
                    jobId,
                    step: step.name,
                    completed: Object.keys(results).length,
                    total: workflow.steps.length
                });
            }
            
            job.status = 'completed';
            job.endTime = Date.now();
            job.duration = job.endTime - job.startTime;
            job.results = results;
            
            this.emit('workflow-completed', { jobId, results, duration: job.duration });
            
            return {
                jobId,
                results,
                duration: job.duration,
                workflow: workflowName
            };
        } catch (error) {
            job.status = 'failed';
            job.error = error.message;
            this.emit('workflow-failed', { jobId, error: error.message });
            throw error;
        }
    }
    
    /**
     * Get optimization metrics and bottleneck analysis
     */
    getOptimizationMetrics() {
        const bottlenecks = this.configLoader.getBottleneckAnalysis();
        const summary = this.configLoader.getSummary();
        
        return {
            configSummary: summary,
            bottleneckAnalysis: bottlenecks,
            activeJobs: this.activeJobs.size,
            agentStatus: this.getAgentStatus(),
            workerStats: this.configLoader.workers.size
        };
    }
    
    /**
     * Get real-time performance dashboard data
     */
    async getPerformanceDashboard() {
        const metrics = this.getOptimizationMetrics();
        
        // Get worker statistics if available
        let workerStats = null;
        try {
            workerStats = await this.configLoader.processTokensWithWorker([], 'getStats');
        } catch (error) {
            // Worker stats not available
        }
        
        return {
            timestamp: Date.now(),
            orchestrator: {
                agents: metrics.agentStatus,
                jobs: {
                    active: this.activeJobs.size,
                    completed: this.jobCounter - this.activeJobs.size
                }
            },
            optimization: {
                workers: metrics.workerStats,
                workerStats,
                bottlenecks: metrics.bottleneckAnalysis
            },
            config: metrics.configSummary
        };
    }
    
    async cleanup() {
        console.log('🧹 Cleaning up orchestrator...');
        
        // Cleanup agents
        for (const [name, agent] of this.agents) {
            if (agent.cleanup) {
                try {
                    await agent.cleanup();
                } catch (error) {
                    console.warn(`Cleanup failed for ${name}:`, error.message);
                }
            }
        }
        
        // Cleanup configuration loader (worker threads, etc.)
        if (this.configLoader) {
            await this.configLoader.cleanup();
        }
        
        this.activeJobs.clear();
        console.log('✅ Orchestrator cleanup complete');
    }
}

export default AgentOrchestrator;
