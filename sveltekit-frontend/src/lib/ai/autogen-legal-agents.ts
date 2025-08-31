
// Autogen Multi-Agent Framework for Legal AI
// Ultra-efficient legal analysis using specialized AI agents

export interface AgentConfig {
  name: string;
  role: string;
  systemMessage: string;
  maxTokens: number;
  temperature: number;
  model: string;
}

export interface ConversationMessage {
  role: "user" | "assistant" | "system";
  content: string;
  agent?: string;
  timestamp: Date;
  metadata?: Record<string, any>;
}

export interface LegalAnalysisRequest {
  caseId?: string;
  evidenceIds?: string[];
  query: string;
  analysisType:
    | "case_review"
    | "evidence_analysis"
    | "legal_research"
    | "prosecution_strategy";
  priority: "low" | "medium" | "high" | "urgent";
}

export interface LegalAnalysisResult {
  sessionId: string;
  agentConversations: ConversationMessage[];
  finalAnalysis: string;
  confidence: number;
  recommendations: string[];
  sources: Array<{
    type: "case" | "evidence" | "precedent" | "statute";
    id: string;
    relevance: number;
    excerpt: string;
  }>;
  processingTime: number;
}

class AutogenLegalTeam {
  private agents: Map<string, AgentConfig>;
  private ollamaEndpoint: string;

  constructor(
    config: {
      ollamaEndpoint?: string;
      useGPU?: boolean;
    } = {},
  ) {
    this.ollamaEndpoint = config.ollamaEndpoint || "http://localhost:11434";
    this.agents = new Map();
    this.initializeLegalAgents();
  }

  private initializeLegalAgents() {
    // Chief Legal Analyst - Coordinates the analysis
    this.agents.set("chief_analyst", {
      name: "Chief Legal Analyst",
      role: "coordinator",
      systemMessage: `You are the Chief Legal Analyst coordinating a prosecution team. 
Your role is to:
1. Break down complex legal queries into specific tasks for specialized agents
2. Synthesize findings from multiple agents into coherent analysis
3. Ensure all legal requirements and procedures are followed
4. Provide strategic recommendations for prosecution

Always maintain objectivity and focus on evidence-based conclusions.`,
      maxTokens: 2048,
      temperature: 0.3,
      model: "gemma3-legal",
    });

    // Evidence Specialist - Analyzes evidence admissibility and strength
    this.agents.set("evidence_specialist", {
      name: "Evidence Specialist",
      role: "evidence_analysis",
      systemMessage: `You are an Evidence Analysis Specialist with expertise in:
1. Evidence admissibility under Federal Rules of Evidence
2. Chain of custody requirements
3. Digital forensics and electronic evidence
4. Expert witness testimony standards
5. Evidence authentication and foundation requirements

Focus on practical admissibility issues and evidentiary strength assessment.`,
      maxTokens: 1536,
      temperature: 0.2,
      model: "gemma3-legal",
    });

    // Legal Research Specialist - Finds relevant case law and statutes
    this.agents.set("research_specialist", {
      name: "Legal Research Specialist",
      role: "legal_research",
      systemMessage: `You are a Legal Research Specialist specializing in:
1. Case law research and precedent analysis
2. Statutory interpretation and application
3. Constitutional law considerations
4. Jurisdiction-specific legal requirements
5. Recent legal developments and trends

Provide thorough research with proper citations and relevance scoring.`,
      maxTokens: 1536,
      temperature: 0.25,
      model: "gemma3-legal",
    });

    // Strategy Advisor - Develops prosecution strategies
    this.agents.set("strategy_advisor", {
      name: "Prosecution Strategy Advisor",
      role: "strategy_development",
      systemMessage: `You are a Prosecution Strategy Advisor focused on:
1. Trial strategy development and case theory
2. Witness preparation and presentation order
3. Anticipating defense arguments and counter-strategies
4. Plea negotiation considerations
5. Resource allocation and case prioritization

Always consider ethical obligations and prosecutorial discretion.`,
      maxTokens: 1536,
      temperature: 0.4,
      model: "gemma3-legal",
    });

    // Compliance Officer - Ensures procedural compliance
    this.agents.set("compliance_officer", {
      name: "Legal Compliance Officer",
      role: "compliance_review",
      systemMessage: `You are a Legal Compliance Officer responsible for:
1. Due process requirements and constitutional protections
2. Discovery obligations and Brady material
3. Statute of limitations and procedural deadlines
4. Ethical considerations and professional responsibility
5. Appeal-proofing strategies and error prevention

Ensure all recommendations comply with legal and ethical standards.`,
      maxTokens: 1024,
      temperature: 0.1,
      model: "gemma3-legal",
    });
  }

  async analyzeCase(
    request: LegalAnalysisRequest,
  ): Promise<LegalAnalysisResult> {
    const startTime = Date.now();
    const sessionId = `autogen_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const conversations: ConversationMessage[] = [];

    try {
      // Phase 1: Chief Analyst breaks down the task
      const taskBreakdown = await this.queryAgent(
        "chief_analyst",
        `Analyze this legal query and break it down into specific tasks for our specialist team:
        
Query: ${request.query}
Case ID: ${request.caseId || "N/A"}
Evidence Count: ${request.evidenceIds?.length || 0}
Analysis Type: ${request.analysisType}
Priority: ${request.priority}

Please provide:
1. Specific tasks for each specialist agent
2. Key questions that need answers
3. Expected deliverables from each agent
4. Analysis coordination strategy`,
        conversations,
      );

      // Phase 2: Parallel specialist analysis
      const specialistTasks = await Promise.all([
        this.runEvidenceAnalysis(request, conversations),
        this.runLegalResearch(request, conversations),
        this.runComplianceReview(request, conversations),
      ]);

      // Phase 3: Strategy development based on specialist findings
      const strategyAnalysis = await this.queryAgent(
        "strategy_advisor",
        `Based on the following specialist analyses, develop a comprehensive prosecution strategy:

Evidence Analysis: ${specialistTasks[0]}
Legal Research: ${specialistTasks[1]}
Compliance Review: ${specialistTasks[2]}

Original Query: ${request.query}

Provide specific strategic recommendations and identify potential challenges.`,
        conversations,
      );

      // Phase 4: Chief Analyst synthesizes final analysis
      const finalAnalysis = await this.queryAgent(
        "chief_analyst",
        `Synthesize the following specialist analyses into a comprehensive legal analysis:

Task Breakdown: ${taskBreakdown}
Evidence Analysis: ${specialistTasks[0]}
Legal Research: ${specialistTasks[1]}
Compliance Review: ${specialistTasks[2]}
Strategy Recommendations: ${strategyAnalysis}

Provide:
1. Executive summary of findings
2. Key legal issues and recommendations
3. Risk assessment and mitigation strategies
4. Next steps and action items
5. Overall confidence level (0.0-1.0)`,
        conversations,
      );

      // Extract recommendations and confidence from final analysis
      const recommendations = this.extractRecommendations(finalAnalysis);
      const confidence = this.extractConfidence(finalAnalysis);

      return {
        sessionId,
        agentConversations: conversations,
        finalAnalysis,
        confidence,
        recommendations,
        sources: [], // TODO: Implement source extraction
        processingTime: Date.now() - startTime,
      };
    } catch (error: any) {
      console.error("Autogen analysis error:", error);
      throw new Error(`Multi-agent analysis failed: ${error}`);
    }
  }

  private async runEvidenceAnalysis(
    request: LegalAnalysisRequest,
    conversations: ConversationMessage[],
  ): Promise<string> {
    return await this.queryAgent(
      "evidence_specialist",
      `Analyze the evidence aspects of this case:

Query: ${request.query}
Case ID: ${request.caseId || "N/A"}
Evidence Items: ${request.evidenceIds?.length || 0} pieces of evidence

Focus on:
1. Evidence admissibility issues
2. Chain of custody concerns
3. Authentication requirements
4. Potential evidentiary challenges
5. Strength of evidence assessment

Provide specific recommendations for evidence handling and presentation.`,
      conversations,
    );
  }

  private async runLegalResearch(
    request: LegalAnalysisRequest,
    conversations: ConversationMessage[],
  ): Promise<string> {
    return await this.queryAgent(
      "research_specialist",
      `Conduct legal research for this query:

Query: ${request.query}
Analysis Type: ${request.analysisType}

Research focus areas:
1. Relevant case law and precedents
2. Applicable statutes and regulations
3. Constitutional considerations
4. Jurisdiction-specific requirements
5. Recent legal developments

Provide citations and relevance analysis for each source.`,
      conversations,
    );
  }

  private async runComplianceReview(
    request: LegalAnalysisRequest,
    conversations: ConversationMessage[],
  ): Promise<string> {
    return await this.queryAgent(
      "compliance_officer",
      `Review this case for legal and ethical compliance:

Query: ${request.query}
Priority: ${request.priority}

Compliance checklist:
1. Due process requirements
2. Discovery obligations
3. Ethical considerations
4. Procedural deadlines
5. Constitutional protections

Identify any compliance issues and provide mitigation strategies.`,
      conversations,
    );
  }

  private async queryAgent(
    agentName: string,
    prompt: string,
    conversations: ConversationMessage[],
  ): Promise<string> {
    const agent = this.agents.get(agentName);
    if (!agent) {
      throw new Error(`Agent ${agentName} not found`);
    }

    const message: ConversationMessage = {
      role: "user",
      content: prompt,
      agent: agentName,
      timestamp: new Date(),
    };
    conversations.push(message);

    try {
      const response = await this.queryOllama(agent, prompt);

      const responseMessage: ConversationMessage = {
        role: "assistant",
        content: response,
        agent: agentName,
        timestamp: new Date(),
      };
      conversations.push(responseMessage);

      return response;
    } catch (error: any) {
      console.error(`Agent ${agentName} query failed:`, error);
      throw error;
    }
  }

  private async queryOllama(
    agent: AgentConfig,
    prompt: string,
  ): Promise<string> {
    const response = await fetch(`${this.ollamaEndpoint}/api/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: agent.model,
        prompt: `${agent.systemMessage}\n\nUser: ${prompt}\n\nAssistant:`,
        stream: false,
        options: {
          temperature: agent.temperature,
          num_predict: agent.maxTokens,
          gpu_layers: -1, // Use all GPU layers for acceleration
          num_ctx: 8192,
          repeat_penalty: 1.1,
          top_k: 40,
          top_p: 0.9,
        },
      }),
    });

    if (!response.ok) {
      throw new Error(`Ollama request failed: ${response.statusText}`);
    }

    const data = await response.json();
    return data.response;
  }

  

  private extractRecommendations(analysis: string): string[] {
    // Extract recommendations from analysis using regex patterns
    const recommendationPatterns = [
      /(?:recommend|suggestion|should|consider):\s*(.+?)(?:\n|$)/gi,
      /(?:\d+\.\s*)(.+?)(?:\n|$)/g,
      /(?:•\s*)(.+?)(?:\n|$)/g,
    ];

    const recommendations: string[] = [];
    for (const pattern of recommendationPatterns) {
      const matches = Array.from(analysis.matchAll(pattern));
      for (const match of matches) {
        if (match[1]?.trim()) {
          recommendations.push(match[1].trim());
        }
      }
    }

    return recommendations.slice(0, 10); // Limit to top 10 recommendations
  }

  private extractConfidence(analysis: string): number {
    // Extract confidence score from analysis
    const confidencePattern = /confidence[:\s]+(\d*\.?\d+)/i;
    const match = analysis.match(confidencePattern);

    if (match && match[1]) {
      const confidence = parseFloat(match[1]);
      return confidence > 1 ? confidence / 100 : confidence; // Normalize to 0-1 range
    }

    // Default confidence based on analysis length and completeness
    return analysis.length > 1000 ? 0.85 : 0.75;
  }

  // Get available agents for debugging/monitoring
  getAgents(): AgentConfig[] {
    return Array.from(this.agents.values());
  }

  // Update agent configuration
  updateAgent(name: string, config: Partial<AgentConfig>): void {
    const existing = this.agents.get(name);
    if (existing) {
      this.agents.set(name, { ...existing, ...config });
    }
  }
}

export {
  AutogenLegalTeam,
};
