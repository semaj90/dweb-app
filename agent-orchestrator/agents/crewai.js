/**
 * CrewAI Multi-Agent System for Legal Workflows
 * Enhanced RAG System Integration
 */

import { EventEmitter } from 'events';
import fetch from 'node-fetch';

export class CrewAIAgent extends EventEmitter {
    constructor(config = {}) {
        super();
        this.config = {
            endpoint: config.endpoint || process.env.CREWAI_ENDPOINT || 'http://localhost:8001',
            timeout: config.timeout || 30000,
            maxRetries: config.maxRetries || 3,
            ...config
        };
        
        this.isInitialized = false;
        this.crews = new Map();
        this.capabilities = [
            'multi-agent-collaboration',
            'legal-research',
            'document-drafting',
            'case-preparation',
            'workflow-orchestration'
        ];
    }
    
    async initialize() {
        try {
            // Test CrewAI endpoint
            const response = await fetch(`${this.config.endpoint}/health`, {
                method: 'GET',
                timeout: this.config.timeout
            });
            
            if (!response.ok) {
                throw new Error(`CrewAI service unavailable: ${response.status}`);
            }
            
            this.isInitialized = true;
            this.emit('initialized', { agent: 'crewai', status: 'ready' });
            return true;
        } catch (error) {
            this.emit('error', { agent: 'crewai', error: error.message });
            throw error;
        }
    }
    
    async createLegalResearchCrew(caseData) {
        if (!this.isInitialized) {
            await this.initialize();
        }
        
        const crewConfig = {
            name: `legal-research-${Date.now()}`,
            agents: [
                {
                    role: 'Senior Legal Researcher',
                    goal: 'Find relevant case law and statutes',
                    backstory: 'Expert in legal research with 20+ years experience'
                },
                {
                    role: 'Case Analyst',
                    goal: 'Analyze case similarities and differences',
                    backstory: 'Specialized in comparative case analysis'
                },
                {
                    role: 'Legal Writer',
                    goal: 'Synthesize research into comprehensive reports',
                    backstory: 'Expert legal writer and communicator'
                }
            ],
            tasks: [
                {
                    description: `Research case law related to: ${caseData.subject || 'unknown'}`,
                    agent: 'Senior Legal Researcher',
                    expected_output: 'List of relevant cases with citations'
                },
                {
                    description: 'Analyze similarities between found cases and current case',
                    agent: 'Case Analyst',
                    expected_output: 'Comparative analysis report'
                },
                {
                    description: 'Create comprehensive legal research report',
                    agent: 'Legal Writer',
                    expected_output: 'Professional legal research report'
                }
            ]
        };
        
        try {
            const response = await fetch(`${this.config.endpoint}/crews`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(crewConfig),
                timeout: this.config.timeout
            });
            
            if (!response.ok) {
                throw new Error(`Failed to create crew: ${createResponse.status}`);
            }
            
            const crew = await response.json();
            this.crews.set(crew.id, crew);
            
            this.emit('crew-created', { agent: 'crewai', crewId: crew.id });
            return crew;
        } catch (error) {
            this.emit('error', { agent: 'crewai', error: error.message });
            throw error;
        }
    }
    
    async executeCrew(crewId, input = {}) {
        if (!this.crews.has(crewId)) {
            throw new Error(`Crew ${crewId} not found`);
        }
        
        try {
            const response = await fetch(`${this.config.endpoint}/crews/${crewId}/execute`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(input),
                timeout: this.config.timeout
            });
            
            if (!response.ok) {
                throw new Error(`Crew execution failed: ${response.status}`);
            }
            
            const result = await response.json();
            
            this.emit('crew-executed', { 
                agent: 'crewai', 
                crewId: crewId, 
                result: result 
            });
            
            return {
                agent: 'crewai',
                crewId: crewId,
                result: result,
                timestamp: new Date().toISOString()
            };
        } catch (error) {
            this.emit('error', { agent: 'crewai', crewId: crewId, error: error.message });
            throw error;
        }
    }
    
    async analyzeDocument(document, analysisType = 'comprehensive') {
        const caseData = {
            document: document,
            legal_issues: analysisType,
            analysis_type: analysisType
        };
        
        const crew = await this.createLegalResearchCrew(caseData);
        return await this.executeCrew(crew.id, { document: document });
    }
    
    getCapabilities() {
        return this.capabilities;
    }
    
    getStatus() {
        return {
            agent: 'crewai',
            initialized: this.isInitialized,
            endpoint: this.config.endpoint,
            activeCrews: this.crews.size,
            capabilities: this.capabilities
        };
    }
    
    async cleanup() {
        // Clean up active crews
        for (const [crewId, crew] of this.crews) {
            try {
                await fetch(`${this.config.endpoint}/crews/${this.currentCrewId}`, {
                    method: 'DELETE'
                });
            } catch (error) {
                console.warn(`Failed to cleanup crew ${this.currentCrewId}:`, error.message);
            }
        }
        this.crews.clear();
    }
}

export default CrewAIAgent;
