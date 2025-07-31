/**
 * Enhanced RAG System - Agent Orchestrator
 * Multi-Agent Coordination for Legal AI
 */

import { EventEmitter } from 'events';
import ClaudeAgent from './agents/claude.js';
import CrewAIAgent from './agents/crewai.js';
import GemmaAgent from './agents/gemma.js';
import OllamaAgent from './agents/ollama.js';

export class AgentOrchestrator extends EventEmitter {
    constructor(config = {}) {
        super();
        this.config = config;
        this.agents = new Map();
        this.isInitialized = false;
        this.activeJobs = new Map();
        
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
            totalAgents: initializedAgents.length
        });
        
        return {
            initialized: initializedAgents,
            failed: failedAgents
        };
    }
    
    async analyzeLegalDocument(document, options = {}) {
        if (!this.isInitialized) {
            await this.initialize();
        }
        
        const jobId = job--;
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
    
    async cleanup() {
        for (const [name, agent] of this.agents) {
            if (agent.cleanup) {
                try {
                    await agent.cleanup();
                } catch (error) {
                    console.warn(`Cleanup failed for ${name}:`, error.message);
                }
            }
        }
        this.activeJobs.clear();
    }
}

export default AgentOrchestrator;
