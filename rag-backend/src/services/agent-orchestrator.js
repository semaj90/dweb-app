/**
 * Agent Orchestrator Service
 * Manages multi-agent workflows and coordinated AI responses
 */

export class AgentOrchestrator {
  constructor(services) {
    this.ollama = services.ollama;
    this.database = services.database;
    this.cache = services.cache;
    
    // Agent configurations
    this.agents = {
      analyzer: {
        name: 'Document Analyzer',
        model: 'gemma2:9b',
        systemPrompt: 'You are a legal document analyzer. Provide detailed analysis of legal documents, identifying key clauses, risks, and important information.',
        temperature: 0.1,
        maxTokens: 1024
      },
      summarizer: {
        name: 'Content Summarizer',
        model: 'gemma2:9b',
        systemPrompt: 'You are a professional summarizer. Create concise, accurate summaries that capture the most important information.',
        temperature: 0.2,
        maxTokens: 512
      },
      researcher: {
        name: 'Legal Researcher',
        model: 'gemma2:9b',
        systemPrompt: 'You are a legal research specialist. Find relevant case law, statutes, and legal precedents related to the given query.',
        temperature: 0.3,
        maxTokens: 1024
      },
      reviewer: {
        name: 'Content Reviewer',
        model: 'gemma2:9b',
        systemPrompt: 'You are a quality assurance reviewer. Evaluate the accuracy, completeness, and relevance of legal content.',
        temperature: 0.1,
        maxTokens: 512
      },
      strategist: {
        name: 'Legal Strategist',
        model: 'gemma2:9b',
        systemPrompt: 'You are a legal strategy advisor. Provide strategic recommendations and identify potential legal approaches.',
        temperature: 0.4,
        maxTokens: 1024
      }
    };
  }

  /**
   * Orchestrate multi-agent workflow
   */
  async orchestrateWorkflow(workflowType, input, options = {}) {
    try {
      console.log(`🎭 Starting ${workflowType} workflow`);
      
      // Check cache for recent similar workflows
      const cacheKey = `workflow:${workflowType}:${this.hashInput(input)}`;
      const cached = await this.cache.getCachedAgentResults(workflowType, input);
      
      if (cached && !options.skipCache) {
        console.log('🔄 Using cached workflow result');
        return cached;
      }

      let result;
      
      switch (workflowType) {
        case 'document_analysis':
          result = await this.documentAnalysisWorkflow(input, options);
          break;
        case 'legal_research':
          result = await this.legalResearchWorkflow(input, options);
          break;
        case 'case_preparation':
          result = await this.casePreparationWorkflow(input, options);
          break;
        case 'contract_review':
          result = await this.contractReviewWorkflow(input, options);
          break;
        case 'evidence_analysis':
          result = await this.evidenceAnalysisWorkflow(input, options);
          break;
        default:
          throw new Error(`Unknown workflow type: ${workflowType}`);
      }

      // Cache the result
      await this.cache.cacheAgentResults(workflowType, input, result, 3600);

      console.log(`✅ ${workflowType} workflow completed`);
      return result;

    } catch (error) {
      console.error('Workflow orchestration failed:', error);
      throw error;
    }
  }

  /**
   * Document Analysis Workflow
   */
  async documentAnalysisWorkflow(document, options = {}) {
    const startTime = Date.now();
    const agents = ['analyzer', 'summarizer', 'reviewer'];
    const results = {};

    try {
      // Phase 1: Initial Analysis
      console.log('📊 Phase 1: Document Analysis');
      results.analysis = await this.runAgent('analyzer', document.content, {
        context: `Analyzing ${document.documentType || 'legal'} document: ${document.title || 'Untitled'}`,
        ...options
      });

      // Phase 2: Summary Generation
      console.log('📝 Phase 2: Summary Generation');
      results.summary = await this.runAgent('summarizer', document.content, {
        context: 'Create a comprehensive summary of this document',
        ...options
      });

      // Phase 3: Quality Review
      console.log('🔍 Phase 3: Quality Review');
      const reviewInput = `
Analysis: ${results.analysis.content}
Summary: ${results.summary.content}
Original Document: ${document.content.substring(0, 2000)}...
`;
      
      results.review = await this.runAgent('reviewer', reviewInput, {
        context: 'Review the analysis and summary for accuracy and completeness',
        ...options
      });

      // Phase 4: Synthesis
      console.log('🔄 Phase 4: Synthesis');
      const synthesis = await this.synthesizeResults(results, 'document_analysis');

      return {
        workflowType: 'document_analysis',
        document: {
          title: document.title,
          type: document.documentType,
          length: document.content.length
        },
        phases: {
          analysis: results.analysis,
          summary: results.summary,
          review: results.review
        },
        synthesis,
        metadata: {
          processingTime: Date.now() - startTime,
          agentsUsed: agents,
          timestamp: new Date().toISOString()
        }
      };

    } catch (error) {
      console.error('Document analysis workflow failed:', error);
      throw error;
    }
  }

  /**
   * Legal Research Workflow
   */
  async legalResearchWorkflow(query, options = {}) {
    const startTime = Date.now();
    const results = {};

    try {
      // Phase 1: Research Planning
      console.log('🔍 Phase 1: Research Planning');
      const planningPrompt = `
Research Query: ${query.text || query}
Context: ${query.context || 'General legal research'}

Plan a comprehensive legal research strategy. Identify:
1. Key legal concepts to investigate
2. Relevant jurisdictions
3. Types of sources to search
4. Potential case law and statutes
`;

      results.plan = await this.runAgent('researcher', planningPrompt, {
        context: 'Legal research planning',
        ...options
      });

      // Phase 2: Research Execution
      console.log('📚 Phase 2: Research Execution');
      const researchPrompt = `
Based on this research plan: ${results.plan.content}

Conduct detailed legal research on: ${query.text || query}

Provide:
1. Relevant case law
2. Applicable statutes
3. Legal precedents
4. Jurisdictional considerations
`;

      results.research = await this.runAgent('researcher', researchPrompt, {
        context: 'Legal research execution',
        temperature: 0.3,
        ...options
      });

      // Phase 3: Strategy Development
      console.log('⚖️ Phase 3: Strategy Development');
      const strategyPrompt = `
Research Results: ${results.research.content}
Original Query: ${query.text || query}

Based on the research findings, develop strategic recommendations:
1. Legal arguments and theories
2. Potential challenges and risks
3. Recommended approaches
4. Next steps for investigation
`;

      results.strategy = await this.runAgent('strategist', strategyPrompt, {
        context: 'Legal strategy development',
        ...options
      });

      // Phase 4: Synthesis
      const synthesis = await this.synthesizeResults(results, 'legal_research');

      return {
        workflowType: 'legal_research',
        query: query.text || query,
        phases: {
          plan: results.plan,
          research: results.research,
          strategy: results.strategy
        },
        synthesis,
        metadata: {
          processingTime: Date.now() - startTime,
          agentsUsed: ['researcher', 'strategist'],
          timestamp: new Date().toISOString()
        }
      };

    } catch (error) {
      console.error('Legal research workflow failed:', error);
      throw error;
    }
  }

  /**
   * Case Preparation Workflow
   */
  async casePreparationWorkflow(caseData, options = {}) {
    const startTime = Date.now();
    const results = {};

    try {
      // Phase 1: Case Analysis
      console.log('📋 Phase 1: Case Analysis');
      const analysisPrompt = `
Case Information:
Title: ${caseData.title}
Description: ${caseData.description}
Evidence Count: ${caseData.evidenceCount || 0}
Case Type: ${caseData.caseType || 'General'}

Analyze this case and identify:
1. Key legal issues
2. Evidence requirements
3. Potential witnesses
4. Legal theories to pursue
5. Challenges and risks
`;

      results.analysis = await this.runAgent('analyzer', analysisPrompt, {
        context: 'Case preparation analysis',
        ...options
      });

      // Phase 2: Strategy Development
      console.log('🎯 Phase 2: Strategy Development');
      const strategyPrompt = `
Case Analysis: ${results.analysis.content}

Develop a comprehensive case strategy:
1. Primary and alternative legal theories
2. Evidence collection priorities
3. Discovery strategy
4. Timeline and milestones
5. Resource requirements
`;

      results.strategy = await this.runAgent('strategist', strategyPrompt, {
        context: 'Case strategy development',
        ...options
      });

      // Phase 3: Preparation Checklist
      console.log('✅ Phase 3: Preparation Checklist');
      const checklistPrompt = `
Case Analysis: ${results.analysis.content}
Case Strategy: ${results.strategy.content}

Create a detailed case preparation checklist:
1. Immediate action items
2. Research tasks
3. Evidence collection steps
4. Filing requirements
5. Deadline tracking
`;

      results.checklist = await this.runAgent('analyzer', checklistPrompt, {
        context: 'Case preparation checklist',
        temperature: 0.1,
        ...options
      });

      const synthesis = await this.synthesizeResults(results, 'case_preparation');

      return {
        workflowType: 'case_preparation',
        case: {
          title: caseData.title,
          type: caseData.caseType,
          evidenceCount: caseData.evidenceCount
        },
        phases: {
          analysis: results.analysis,
          strategy: results.strategy,
          checklist: results.checklist
        },
        synthesis,
        metadata: {
          processingTime: Date.now() - startTime,
          agentsUsed: ['analyzer', 'strategist'],
          timestamp: new Date().toISOString()
        }
      };

    } catch (error) {
      console.error('Case preparation workflow failed:', error);
      throw error;
    }
  }

  /**
   * Contract Review Workflow
   */
  async contractReviewWorkflow(contract, options = {}) {
    const startTime = Date.now();
    const results = {};

    try {
      // Phase 1: Contract Analysis
      console.log('📄 Phase 1: Contract Analysis');
      results.analysis = await this.runAgent('analyzer', contract.content, {
        context: `Contract review: ${contract.title || 'Untitled Contract'}`,
        systemPrompt: 'You are a contract analysis expert. Identify key terms, obligations, risks, and important clauses in this contract.',
        ...options
      });

      // Phase 2: Risk Assessment
      console.log('⚠️ Phase 2: Risk Assessment');
      const riskPrompt = `
Contract: ${contract.content}
Initial Analysis: ${results.analysis.content}

Conduct a comprehensive risk assessment:
1. Identify potential legal risks
2. Assess financial implications
3. Evaluate compliance requirements
4. Flag problematic clauses
5. Suggest risk mitigation strategies
`;

      results.riskAssessment = await this.runAgent('reviewer', riskPrompt, {
        context: 'Contract risk assessment',
        ...options
      });

      // Phase 3: Recommendations
      console.log('💡 Phase 3: Recommendations');
      const recommendationPrompt = `
Contract Analysis: ${results.analysis.content}
Risk Assessment: ${results.riskAssessment.content}

Provide specific recommendations:
1. Suggested contract modifications
2. Additional clauses to include
3. Terms to negotiate
4. Legal protections to add
5. Approval recommendations
`;

      results.recommendations = await this.runAgent('strategist', recommendationPrompt, {
        context: 'Contract recommendations',
        ...options
      });

      const synthesis = await this.synthesizeResults(results, 'contract_review');

      return {
        workflowType: 'contract_review',
        contract: {
          title: contract.title,
          length: contract.content.length
        },
        phases: {
          analysis: results.analysis,
          riskAssessment: results.riskAssessment,
          recommendations: results.recommendations
        },
        synthesis,
        metadata: {
          processingTime: Date.now() - startTime,
          agentsUsed: ['analyzer', 'reviewer', 'strategist'],
          timestamp: new Date().toISOString()
        }
      };

    } catch (error) {
      console.error('Contract review workflow failed:', error);
      throw error;
    }
  }

  /**
   * Evidence Analysis Workflow
   */
  async evidenceAnalysisWorkflow(evidence, options = {}) {
    const startTime = Date.now();
    const results = {};

    try {
      // Phase 1: Evidence Examination
      console.log('🔍 Phase 1: Evidence Examination');
      const examinationPrompt = `
Evidence Type: ${evidence.type}
Evidence Description: ${evidence.description}
Content: ${evidence.content || 'See attached files'}

Examine this evidence and identify:
1. Relevant facts and details
2. Legal significance
3. Potential admissibility issues
4. Connections to case theories
5. Additional investigation needs
`;

      results.examination = await this.runAgent('analyzer', examinationPrompt, {
        context: 'Evidence examination',
        systemPrompt: 'You are a forensic evidence analyst. Examine evidence for legal relevance and investigative value.',
        ...options
      });

      // Phase 2: Authenticity Assessment
      console.log('🔐 Phase 2: Authenticity Assessment');
      const authenticityPrompt = `
Evidence Details: ${results.examination.content}

Assess evidence authenticity and admissibility:
1. Chain of custody considerations
2. Authentication requirements
3. Potential challenges to admissibility
4. Foundation requirements
5. Expert testimony needs
`;

      results.authenticity = await this.runAgent('reviewer', authenticityPrompt, {
        context: 'Evidence authenticity assessment',
        ...options
      });

      // Phase 3: Strategic Analysis
      console.log('⚖️ Phase 3: Strategic Analysis');
      const strategicPrompt = `
Evidence Examination: ${results.examination.content}
Authenticity Assessment: ${results.authenticity.content}

Provide strategic analysis:
1. How this evidence supports case theories
2. Potential counterarguments
3. Best presentation methods
4. Corroborating evidence needed
5. Strategic timing for disclosure
`;

      results.strategic = await this.runAgent('strategist', strategicPrompt, {
        context: 'Evidence strategic analysis',
        ...options
      });

      const synthesis = await this.synthesizeResults(results, 'evidence_analysis');

      return {
        workflowType: 'evidence_analysis',
        evidence: {
          type: evidence.type,
          description: evidence.description
        },
        phases: {
          examination: results.examination,
          authenticity: results.authenticity,
          strategic: results.strategic
        },
        synthesis,
        metadata: {
          processingTime: Date.now() - startTime,
          agentsUsed: ['analyzer', 'reviewer', 'strategist'],
          timestamp: new Date().toISOString()
        }
      };

    } catch (error) {
      console.error('Evidence analysis workflow failed:', error);
      throw error;
    }
  }

  /**
   * Run individual agent
   */
  async runAgent(agentType, input, options = {}) {
    const agent = this.agents[agentType];
    if (!agent) {
      throw new Error(`Unknown agent type: ${agentType}`);
    }

    try {
      const prompt = options.systemPrompt ? 
        `${options.systemPrompt}\n\n${input}` : 
        `${agent.systemPrompt}\n\n${input}`;

      const result = await this.ollama.generateCompletion(prompt, {
        model: agent.model,
        temperature: options.temperature || agent.temperature,
        maxTokens: options.maxTokens || agent.maxTokens
      });

      return {
        agent: agent.name,
        content: result.content,
        model: result.model,
        processingTime: result.total_duration,
        context: options.context || 'General'
      };

    } catch (error) {
      console.error(`Agent ${agentType} failed:`, error);
      throw error;
    }
  }

  /**
   * Synthesize results from multiple agents
   */
  async synthesizeResults(results, workflowType) {
    try {
      const synthesisPrompt = `
Workflow Type: ${workflowType}

Agent Results:
${Object.entries(results).map(([phase, result]) => 
  `${phase.toUpperCase()}: ${result.content}`
).join('\n\n')}

Synthesize these results into a comprehensive, actionable summary. Provide:
1. Key findings and insights
2. Consolidated recommendations
3. Priority actions
4. Potential issues or risks
5. Next steps

Focus on practical, actionable guidance for legal professionals.
`;

      const synthesis = await this.ollama.generateCompletion(synthesisPrompt, {
        model: 'gemma2:9b',
        temperature: 0.2,
        maxTokens: 1024
      });

      return {
        content: synthesis.content,
        confidence: this.calculateSynthesisConfidence(results),
        agentCount: Object.keys(results).length,
        generatedAt: new Date().toISOString()
      };

    } catch (error) {
      console.error('Results synthesis failed:', error);
      return {
        content: 'Failed to synthesize results',
        confidence: 0,
        error: error.message,
        generatedAt: new Date().toISOString()
      };
    }
  }

  /**
   * Calculate synthesis confidence based on agent results
   */
  calculateSynthesisConfidence(results) {
    const agentCount = Object.keys(results).length;
    let confidence = 0.3; // Base confidence

    // More agents = higher confidence
    confidence += Math.min(agentCount * 0.15, 0.4);

    // Check for consistent results (simple heuristic)
    const contentLengths = Object.values(results).map(r => r.content.length);
    const avgLength = contentLengths.reduce((a, b) => a + b, 0) / contentLengths.length;
    
    if (avgLength > 200) confidence += 0.2; // Substantial responses
    if (agentCount >= 3) confidence += 0.1; // Multi-agent validation

    return Math.min(confidence, 1.0);
  }

  /**
   * Hash input for caching
   */
  hashInput(input) {
    const str = typeof input === 'object' ? JSON.stringify(input) : String(input);
    return Buffer.from(str.substring(0, 1000)).toString('base64').substring(0, 16);
  }

  /**
   * Get workflow statistics
   */
  async getWorkflowStats() {
    try {
      return {
        availableWorkflows: [
          'document_analysis',
          'legal_research', 
          'case_preparation',
          'contract_review',
          'evidence_analysis'
        ],
        availableAgents: Object.keys(this.agents),
        agentDetails: Object.fromEntries(
          Object.entries(this.agents).map(([key, agent]) => [
            key, 
            {
              name: agent.name,
              model: agent.model,
              temperature: agent.temperature,
              maxTokens: agent.maxTokens
            }
          ])
        ),
        ollamaHealthy: this.ollama.isHealthy,
        cacheEnabled: this.cache.connected
      };
    } catch (error) {
      console.error('Failed to get workflow stats:', error);
      return {
        error: error.message
      };
    }
  }
}