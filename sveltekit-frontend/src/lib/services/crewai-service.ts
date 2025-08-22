
/**
 * CrewAI Multi-Agent Service
 * Handles role-based agent crews for specialized legal workflows
 */

import crypto from "crypto";
import type {
  AgentDefinition,
  AgentWorkflow,
  WorkflowStep,
  AITask,
  AIResponse,
} from "$lib/types/ai-worker.js";

export interface CrewAIAgent {
  id: string;
  role: string;
  goal: string;
  backstory: string;
  tools: string[];
  llmConfig: {
    model: string;
    temperature: number;
    maxTokens: number;
    apiBase?: string;
  };
  maxExecution: number;
  memory: boolean;
  verbose: boolean;
  allowDelegation: boolean;
}

export interface CrewAITask {
  id: string;
  description: string;
  expectedOutput: string;
  agent?: string;
  tools?: string[];
  context?: string[];
  dependencies?: string[];
}

export interface CrewAICrew {
  id: string;
  name: string;
  description: string;
  agents: CrewAIAgent[];
  tasks: CrewAITask[];
  process: "sequential" | "hierarchical" | "consensus";
  manager?: string;
  verbose: boolean;
  memoryEnabled: boolean;
}

export interface CrewExecution {
  id: string;
  crewId: string;
  status: "pending" | "running" | "completed" | "failed" | "cancelled";
  startTime: number;
  endTime?: number;
  results: CrewTaskResult[];
  finalOutput?: string;
  metrics: {
    totalTime: number;
    tasksCompleted: number;
    agentInteractions: number;
    tokensUsed: number;
  };
}

export interface CrewTaskResult {
  taskId: string;
  agentId: string;
  output: string;
  executionTime: number;
  status: "completed" | "failed" | "delegated";
  metadata?: Record<string, any>;
}

export class CrewAIService {
  private baseUrl: string;
  private apiKey?: string;
  private defaultTimeout: number = 60000;

  constructor(baseUrl: string = "http://localhost:8002", apiKey?: string) {
    this.baseUrl = baseUrl;
    this.apiKey = apiKey;
  }

  /**
   * Create a specialized legal investigation crew
   */
  createLegalInvestigationCrew(): CrewAICrew {
    const agents: CrewAIAgent[] = [
      {
        id: "case-investigator",
        role: "Lead Case Investigator",
        goal: "Conduct comprehensive legal case investigation and evidence analysis",
        backstory: `You are an experienced criminal investigator with 15 years in law enforcement. 
        You specialize in complex cases involving financial crimes, digital evidence, and witness coordination. 
        Your expertise includes evidence collection protocols, interview techniques, and case documentation standards.`,
        tools: [
          "evidence_analyzer",
          "witness_interview_tool",
          "timeline_builder",
          "case_documentation",
        ],
        llmConfig: {
          model: "gemma3-legal",
          temperature: 0.1,
          maxTokens: 1536,
          apiBase: "http://localhost:11434",
        },
        maxExecution: 5,
        memory: true,
        verbose: true,
        allowDelegation: true,
      },
      {
        id: "legal-analyst",
        role: "Senior Legal Analyst",
        goal: "Provide legal analysis, precedent research, and case strategy recommendations",
        backstory: `You are a senior legal analyst with expertise in criminal and civil law. 
        You have worked with major law firms and prosecutor's offices for over 12 years. 
        Your specialties include case law research, legal precedent analysis, and litigation strategy.`,
        tools: [
          "legal_research_tool",
          "precedent_finder",
          "statute_analyzer",
          "case_strategy_builder",
        ],
        llmConfig: {
          model: "llama3:8b-instruct",
          temperature: 0.2,
          maxTokens: 2048,
          apiBase: "http://localhost:11434",
        },
        maxExecution: 4,
        memory: true,
        verbose: true,
        allowDelegation: false,
      },
      {
        id: "evidence-specialist",
        role: "Digital Evidence Specialist",
        goal: "Analyze digital evidence, verify authenticity, and ensure admissibility",
        backstory: `You are a certified digital forensics expert with advanced training in cybersecurity 
        and digital evidence analysis. You have testified as an expert witness in over 100 cases. 
        Your expertise covers mobile forensics, network analysis, and digital chain of custody procedures.`,
        tools: [
          "digital_forensics_tool",
          "metadata_analyzer",
          "authenticity_verifier",
          "chain_custody_tracker",
        ],
        llmConfig: {
          model: "codellama:7b-code",
          temperature: 0.1,
          maxTokens: 1024,
          apiBase: "http://localhost:11434",
        },
        maxExecution: 3,
        memory: true,
        verbose: true,
        allowDelegation: false,
      },
      {
        id: "report-writer",
        role: "Legal Report Writer",
        goal: "Synthesize analysis into comprehensive legal reports and recommendations",
        backstory: `You are a professional legal writer with expertise in creating clear, comprehensive reports 
        for law enforcement and legal proceedings. You have authored hundreds of investigation reports, 
        legal briefs, and expert summaries. Your writing is known for clarity, accuracy, and legal precision.`,
        tools: [
          "report_generator",
          "citation_formatter",
          "legal_writer",
          "document_compiler",
        ],
        llmConfig: {
          model: "gemma3-legal",
          temperature: 0.3,
          maxTokens: 3072,
          apiBase: "http://localhost:11434",
        },
        maxExecution: 2,
        memory: true,
        verbose: true,
        allowDelegation: false,
      },
    ];

    const tasks: CrewAITask[] = [
      {
        id: "initial-investigation",
        description:
          "Conduct initial case investigation and evidence inventory",
        expectedOutput: `Comprehensive investigation report including:
        - Case summary and key facts
        - Evidence inventory with classification
        - Timeline of events
        - Identified witnesses and persons of interest
        - Initial assessment of case strength`,
        agent: "case-investigator",
        tools: ["evidence_analyzer", "timeline_builder", "case_documentation"],
      },
      {
        id: "legal-research",
        description:
          "Research applicable laws, precedents, and legal strategies",
        expectedOutput: `Legal analysis report containing:
        - Applicable statutes and regulations
        - Relevant case precedents with citations
        - Legal theories and potential charges
        - Jurisdictional considerations
        - Recommended legal strategies`,
        agent: "legal-analyst",
        tools: ["legal_research_tool", "precedent_finder", "statute_analyzer"],
        dependencies: ["initial-investigation"],
      },
      {
        id: "evidence-analysis",
        description:
          "Perform detailed analysis of all digital and physical evidence",
        expectedOutput: `Evidence analysis report with:
        - Detailed evidence examination results
        - Authenticity verification status
        - Chain of custody validation
        - Admissibility assessment
        - Technical findings and metadata analysis`,
        agent: "evidence-specialist",
        tools: [
          "digital_forensics_tool",
          "metadata_analyzer",
          "authenticity_verifier",
        ],
        dependencies: ["initial-investigation"],
      },
      {
        id: "final-report",
        description: "Compile comprehensive final report with recommendations",
        expectedOutput: `Final investigation report including:
        - Executive summary
        - Detailed findings from all team members
        - Evidence analysis conclusions
        - Legal recommendations
        - Next steps and action items
        - Professional formatting with proper citations`,
        agent: "report-writer",
        tools: ["report_generator", "citation_formatter", "legal_writer"],
        dependencies: ["legal-research", "evidence-analysis"],
      },
    ];

    return {
      id: "legal-investigation-crew",
      name: "Legal Investigation Crew",
      description:
        "Specialized crew for comprehensive legal case investigation and analysis",
      agents,
      tasks,
      process: "sequential",
      verbose: true,
      memoryEnabled: true,
    };
  }

  /**
   * Create a contract analysis crew
   */
  createContractAnalysisCrew(): CrewAICrew {
    const agents: CrewAIAgent[] = [
      {
        id: "contract-reviewer",
        role: "Senior Contract Reviewer",
        goal: "Analyze contract terms, identify risks, and assess legal compliance",
        backstory: `You are a senior attorney specializing in contract law with 20 years experience 
        in commercial transactions. You have reviewed thousands of contracts across various industries 
        and are expert at identifying potential issues, risks, and non-standard terms.`,
        tools: [
          "contract_analyzer",
          "risk_assessor",
          "compliance_checker",
          "term_extractor",
        ],
        llmConfig: {
          model: "gemma3-legal",
          temperature: 0.1,
          maxTokens: 2048,
          apiBase: "http://localhost:11434",
        },
        maxExecution: 4,
        memory: true,
        verbose: true,
        allowDelegation: true,
      },
      {
        id: "compliance-officer",
        role: "Legal Compliance Officer",
        goal: "Ensure contract compliance with applicable regulations and standards",
        backstory: `You are a legal compliance officer with expertise in regulatory requirements 
        across multiple industries. You specialize in ensuring contracts meet all applicable 
        legal standards, industry regulations, and corporate governance requirements.`,
        tools: [
          "regulatory_checker",
          "standards_validator",
          "governance_analyzer",
          "audit_tool",
        ],
        llmConfig: {
          model: "llama3:8b-instruct",
          temperature: 0.2,
          maxTokens: 1536,
          apiBase: "http://localhost:11434",
        },
        maxExecution: 3,
        memory: true,
        verbose: true,
        allowDelegation: false,
      },
      {
        id: "negotiation-advisor",
        role: "Contract Negotiation Advisor",
        goal: "Provide strategic negotiation recommendations and alternative terms",
        backstory: `You are a skilled contract negotiator with extensive experience in complex 
        commercial deals. You excel at identifying negotiation opportunities, proposing alternative 
        terms, and developing win-win solutions that protect client interests.`,
        tools: [
          "negotiation_analyzer",
          "alternative_drafter",
          "leverage_assessor",
          "strategy_builder",
        ],
        llmConfig: {
          model: "gemma3-legal",
          temperature: 0.3,
          maxTokens: 1536,
          apiBase: "http://localhost:11434",
        },
        maxExecution: 3,
        memory: true,
        verbose: true,
        allowDelegation: false,
      },
    ];

    const tasks: CrewAITask[] = [
      {
        id: "contract-review",
        description: "Perform comprehensive contract review and risk analysis",
        expectedOutput: `Contract review report with:
        - Key terms summary
        - Risk assessment with severity ratings
        - Problematic clauses identification
        - Missing provisions analysis
        - Overall contract strength assessment`,
        agent: "contract-reviewer",
        tools: ["contract_analyzer", "risk_assessor", "term_extractor"],
      },
      {
        id: "compliance-check",
        description:
          "Verify contract compliance with all applicable regulations",
        expectedOutput: `Compliance analysis including:
        - Regulatory requirements assessment
        - Industry standards verification
        - Corporate governance compliance
        - Legal requirement satisfaction
        - Compliance gaps and recommendations`,
        agent: "compliance-officer",
        tools: [
          "regulatory_checker",
          "standards_validator",
          "governance_analyzer",
        ],
        dependencies: ["contract-review"],
      },
      {
        id: "negotiation-strategy",
        description: "Develop negotiation strategy and alternative terms",
        expectedOutput: `Negotiation strategy report with:
        - Key negotiation points
        - Alternative term proposals
        - Leverage analysis
        - Risk mitigation strategies
        - Recommended negotiation approach`,
        agent: "negotiation-advisor",
        tools: [
          "negotiation_analyzer",
          "alternative_drafter",
          "strategy_builder",
        ],
        dependencies: ["contract-review", "compliance-check"],
      },
    ];

    return {
      id: "contract-analysis-crew",
      name: "Contract Analysis Crew",
      description:
        "Specialized crew for comprehensive contract review and negotiation support",
      agents,
      tasks,
      process: "sequential",
      verbose: true,
      memoryEnabled: true,
    };
  }

  /**
   * Execute a crew workflow
   */
  async executeCrew(
    crew: CrewAICrew,
    inputs: Record<string, any> = {},
    options: {
      timeout?: number;
      priority?: "low" | "medium" | "high";
      streamResults?: boolean;
    } = {},
  ): Promise<CrewExecution> {
    const executionId = crypto.randomUUID();

    try {
      const response = await fetch(`${this.baseUrl}/api/crew/execute`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(this.apiKey && { Authorization: `Bearer ${this.apiKey}` }),
        },
        body: JSON.stringify({
          executionId,
          crew,
          inputs,
          options: {
            timeout: options.timeout || this.defaultTimeout,
            priority: options.priority || "medium",
            streamResults: options.streamResults || false,
          },
        }),
        signal: AbortSignal.timeout(options.timeout || this.defaultTimeout),
      });

      if (!response.ok) {
        throw new Error(
          `CrewAI API error: ${response.status} ${response.statusText}`,
        );
      }

      const data = await response.json();
      return data;
    } catch (error) {
      console.error("Failed to execute CrewAI crew:", error);
      throw error;
    }
  }

  /**
   * Get execution status and results
   */
  async getExecution(executionId: string): Promise<CrewExecution> {
    try {
      const response = await fetch(
        `${this.baseUrl}/api/execution/${executionId}`,
        {
          method: "GET",
          headers: {
            ...(this.apiKey && { Authorization: `Bearer ${this.apiKey}` }),
          },
        },
      );

      if (!response.ok) {
        throw new Error(`Failed to get execution: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error("Failed to get execution:", error);
      throw error;
    }
  }

  /**
   * Cancel a running execution
   */
  async cancelExecution(executionId: string): Promise<void> {
    try {
      await fetch(`${this.baseUrl}/api/execution/${executionId}/cancel`, {
        method: "POST",
        headers: {
          ...(this.apiKey && { Authorization: `Bearer ${this.apiKey}` }),
        },
      });
    } catch (error) {
      console.error("Failed to cancel execution:", error);
      throw error;
    }
  }

  /**
   * Health check for CrewAI service
   */
  async healthCheck(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/health`, {
        method: "GET",
        signal: AbortSignal.timeout(5000),
      });
      return response.ok;
    } catch {
      return false;
    }
  }

  /**
   * Get available tools and capabilities
   */
  async getAvailableTools(): Promise<string[]> {
    try {
      const response = await fetch(`${this.baseUrl}/api/tools`, {
        method: "GET",
        headers: {
          ...(this.apiKey && { Authorization: `Bearer ${this.apiKey}` }),
        },
      });

      if (!response.ok) {
        throw new Error("Failed to get tools");
      }

      const data = await response.json();
      return data.tools || [];
    } catch (error) {
      console.error("Failed to get available tools:", error);
      return [
        "evidence_analyzer",
        "legal_research_tool",
        "contract_analyzer",
        "witness_interview_tool",
        "digital_forensics_tool",
        "report_generator",
        "precedent_finder",
        "risk_assessor",
        "compliance_checker",
      ];
    }
  }

  /**
   * Stream execution results in real-time
   */
  async *streamExecution(
    executionId: string,
  ): AsyncGenerator<CrewTaskResult, void, unknown> {
    const eventSource = new EventSource(
      `${this.baseUrl}/api/execution/${executionId}/stream`,
    );

    try {
      while (true) {
        const event = await new Promise<MessageEvent>((resolve, reject) => {
          eventSource.onmessage = resolve;
          eventSource.onerror = reject;
        });

        if (event.data === "DONE") {
          break;
        }

        try {
          const result: CrewTaskResult = JSON.parse(event.data);
          yield result;
        } catch (error) {
          console.error("Failed to parse streaming result:", error);
        }
      }
    } finally {
      eventSource.close();
    }
  }

  /**
   * Create a custom crew with specific configuration
   */
  createCustomCrew(
    name: string,
    description: string,
    agents: CrewAIAgent[],
    tasks: CrewAITask[],
    process: "sequential" | "hierarchical" | "consensus" = "sequential",
  ): CrewAICrew {
    return {
      id: crypto.randomUUID(),
      name,
      description,
      agents,
      tasks,
      process,
      verbose: true,
      memoryEnabled: true,
    };
  }
}

// Singleton instance
export const crewAIService = new CrewAIService();

// Helper functions for common legal workflows
export async function analyzeLegalCaseWithCrew(
  caseDescription: string,
  evidenceFiles: string[] = [],
  jurisdiction: string = "federal",
): Promise<AIResponse> {
  const crew = crewAIService.createLegalInvestigationCrew();

  const inputs = {
    caseDescription,
    evidenceFiles,
    jurisdiction,
    analysisType: "comprehensive",
  };

  try {
    const execution = await crewAIService.executeCrew(crew, inputs, {
      timeout: 120000, // 2 minutes
      priority: "high",
    });

    // Wait for completion
    let status = execution.status;
    let attempts = 0;
    const maxAttempts = 24; // 2 minutes with 5-second intervals

    while (status === "running" && attempts < maxAttempts) {
      await new Promise((resolve) => setTimeout(resolve, 5000));
      const updated = await crewAIService.getExecution(execution.id);
      status = updated.status;
      attempts++;
    }

    const finalExecution = await crewAIService.getExecution(execution.id);

    return {
      id: crypto.randomUUID(),
      content: finalExecution.finalOutput || "Case analysis completed",
      providerId: "crewai",
      model: "crewai-agents",
      tokensUsed: finalExecution.metrics.tokensUsed,
      responseTime: finalExecution.metrics.totalTime,
      metadata: {
        executionId: execution.id,
        tasksCompleted: finalExecution.metrics.tasksCompleted,
        agentInteractions: finalExecution.metrics.agentInteractions,
        crewType: "legal-investigation",
      },
    };
  } catch (error) {
    console.error("Legal case analysis with crew failed:", error);
    throw error;
  }
}

export async function analyzeContractWithCrew(
  contractText: string,
  contractType: string = "commercial",
  industryContext: string = "general",
): Promise<AIResponse> {
  const crew = crewAIService.createContractAnalysisCrew();

  const inputs = {
    contractText,
    contractType,
    industryContext,
    analysisDepth: "comprehensive",
  };

  try {
    const execution = await crewAIService.executeCrew(crew, inputs, {
      timeout: 90000, // 1.5 minutes
      priority: "high",
    });

    // Wait for completion
    let status = execution.status;
    let attempts = 0;
    const maxAttempts = 18; // 1.5 minutes with 5-second intervals

    while (status === "running" && attempts < maxAttempts) {
      await new Promise((resolve) => setTimeout(resolve, 5000));
      const updated = await crewAIService.getExecution(execution.id);
      status = updated.status;
      attempts++;
    }

    const finalExecution = await crewAIService.getExecution(execution.id);

    return {
      id: crypto.randomUUID(),
      content: finalExecution.finalOutput || "Contract analysis completed",
      providerId: "crewai",
      model: "crewai-agents",
      tokensUsed: finalExecution.metrics.tokensUsed,
      responseTime: finalExecution.metrics.totalTime,
      metadata: {
        executionId: execution.id,
        tasksCompleted: finalExecution.metrics.tasksCompleted,
        agentInteractions: finalExecution.metrics.agentInteractions,
        crewType: "contract-analysis",
      },
    };
  } catch (error) {
    console.error("Contract analysis with crew failed:", error);
    throw error;
  }
}
