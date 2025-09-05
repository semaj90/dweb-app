// @ts-nocheck
/**
 * AutoGen Multi-Agent Service
 * Handles conversational AI agents with role-based interactions
 */

import type {
  AgentDefinition,
  AgentWorkflow,
  WorkflowStep,
  AITask,
  AIResponse,
} from "$lib/types/ai-worker.js";

export interface AutoGenAgent {
  name: string;
  systemMessage: string;
  llmConfig: {
    model: string;
    temperature: number;
    maxTokens: number;
    apiBase?: string;
  };
  humanInputMode: "ALWAYS" | "NEVER" | "TERMINATE";
  maxConsecutiveAutoReply: number;
  tools?: string[];
}

export interface AutoGenConversation {
  id: string;
  participants: AutoGenAgent[];
  messages: AutoGenMessage[];
  status: "active" | "completed" | "failed" | "terminated";
  startTime: number;
  endTime?: number;
  metadata: Record<string, any>;
}

export interface AutoGenMessage {
  id: string;
  sender: string;
  recipient: string;
  content: string;
  timestamp: number;
  messageType: "text" | "function_call" | "function_response";
  metadata?: Record<string, any>;
}

export interface LegalAgentTeam {
  prosecutor: AutoGenAgent;
  legalResearcher: AutoGenAgent;
  evidenceAnalyst: AutoGenAgent;
  coordinator: AutoGenAgent;
}

export class AutoGenService {
  private baseUrl: string;
  private apiKey?: string;
  private defaultTimeout: number = 30000;

  constructor(baseUrl: string = "http://localhost:8001", apiKey?: string) {
    this.baseUrl = baseUrl;
    this.apiKey = apiKey;
  }

  /**
   * Create specialized legal AI agents
   */
  createLegalAgentTeam(): LegalAgentTeam {
    const prosecutor: AutoGenAgent = {
      name: "prosecutor",
      systemMessage: `You are an experienced prosecutor with expertise in criminal law, evidence evaluation, and case strategy. 
      Your role is to:
      - Evaluate evidence for prosecutorial merit
      - Identify legal theories and charges
      - Assess case strengths and weaknesses
      - Provide strategic recommendations
      
      Always maintain ethical standards and consider due process requirements.`,
      llmConfig: {
        model: "gemma3-legal",
        temperature: 0.1,
        maxTokens: 1024,
        apiBase: "http://localhost:11434",
      },
      humanInputMode: "NEVER",
      maxConsecutiveAutoReply: 3,
      tools: [
        "legal_database_search",
        "case_precedent_lookup",
        "statute_analysis",
      ],
    };

    const legalResearcher: AutoGenAgent = {
      name: "legal_researcher",
      systemMessage: `You are a skilled legal researcher specializing in case law, statutes, and legal precedents.
      Your role is to:
      - Research relevant case law and statutes
      - Find legal precedents and citations
      - Analyze jurisdictional requirements
      - Provide comprehensive legal background
      
      Focus on accuracy and cite all sources with proper legal citations.`,
      llmConfig: {
        model: "llama3:8b-instruct",
        temperature: 0.2,
        maxTokens: 1536,
        apiBase: "http://localhost:11434",
      },
      humanInputMode: "NEVER",
      maxConsecutiveAutoReply: 2,
      tools: [
        "westlaw_search",
        "lexis_search",
        "statute_lookup",
        "citation_formatter",
      ],
    };

    const evidenceAnalyst: AutoGenAgent = {
      name: "evidence_analyst",
      systemMessage: `You are a forensic evidence analyst with expertise in digital and physical evidence evaluation.
      Your role is to:
      - Analyze evidence authenticity and reliability
      - Identify chain of custody issues
      - Assess evidence admissibility
      - Recommend additional evidence collection
      
      Apply rigorous scientific and legal standards to all analysis.`,
      llmConfig: {
        model: "codellama:7b-code",
        temperature: 0.1,
        maxTokens: 1024,
        apiBase: "http://localhost:11434",
      },
      humanInputMode: "NEVER",
      maxConsecutiveAutoReply: 2,
      tools: [
        "evidence_validator",
        "chain_custody_tracker",
        "admissibility_checker",
      ],
    };

    const coordinator: AutoGenAgent = {
      name: "coordinator",
      systemMessage: `You are a case coordination specialist responsible for orchestrating the legal team's analysis.
      Your role is to:
      - Coordinate between team members
      - Synthesize different perspectives
      - Ensure comprehensive case coverage
      - Provide final recommendations
      
      Facilitate productive collaboration and ensure all aspects are covered.`,
      llmConfig: {
        model: "gemma3-legal",
        temperature: 0.3,
        maxTokens: 2048,
        apiBase: "http://localhost:11434",
      },
      humanInputMode: "NEVER",
      maxConsecutiveAutoReply: 5,
      tools: [
        "case_synthesizer",
        "recommendation_generator",
        "team_coordinator",
      ],
    };

    return { prosecutor, legalResearcher, evidenceAnalyst, coordinator };
  }

  /**
   * Initialize a conversation between agents
   */
  async startConversation(
    agents: AutoGenAgent[],
    initialMessage: string,
    taskContext: Record<string, any> = {},
  ): Promise<AutoGenConversation> {
    const conversationId = crypto.randomUUID();

    try {
      const response = await fetch(`${this.baseUrl}/api/conversation/start`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(this.apiKey && { Authorization: `Bearer ${this.apiKey}` }),
        },
        body: JSON.stringify({
          conversationId,
          agents,
          initialMessage,
          context: taskContext,
          maxRounds: 10,
          terminationCondition: "max_rounds_or_agreement",
        }),
        signal: AbortSignal.timeout(this.defaultTimeout),
      });

      if (!response.ok) {
        throw new Error(
          `AutoGen API error: ${response.status} ${response.statusText}`,
        );
      }

      const data = await response.json();

      return {
        id: conversationId,
        participants: agents,
        messages: [],
        status: "active",
        startTime: Date.now(),
        metadata: data.metadata || {},
      };
    } catch (error) {
      console.error("Failed to start AutoGen conversation:", error);
      throw error;
    }
  }

  /**
   * Get conversation status and messages
   */
  async getConversation(conversationId: string): Promise<AutoGenConversation> {
    try {
      const response = await fetch(
        `${this.baseUrl}/api/conversation/${conversationId}`,
        {
          method: "GET",
          headers: {
            ...(this.apiKey && { Authorization: `Bearer ${this.apiKey}` }),
          },
          signal: AbortSignal.timeout(5000),
        },
      );

      if (!response.ok) {
        throw new Error(`Failed to get conversation: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error("Failed to get conversation:", error);
      throw error;
    }
  }

  /**
   * Send a message to continue the conversation
   */
  async sendMessage(
    conversationId: string,
    message: string,
    sender: string = "user",
  ): Promise<AutoGenMessage[]> {
    try {
      const response = await fetch(
        `${this.baseUrl}/api/conversation/${conversationId}/message`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            ...(this.apiKey && { Authorization: `Bearer ${this.apiKey}` }),
          },
          body: JSON.stringify({
            message,
            sender,
            timestamp: Date.now(),
          }),
          signal: AbortSignal.timeout(this.defaultTimeout),
        },
      );

      if (!response.ok) {
        throw new Error(`Failed to send message: ${response.status}`);
      }

      const data = await response.json();
      return data.messages || [];
    } catch (error) {
      console.error("Failed to send message:", error);
      throw error;
    }
  }

  /**
   * Terminate a conversation
   */
  async terminateConversation(conversationId: string): Promise<void> {
    try {
      await fetch(
        `${this.baseUrl}/api/conversation/${conversationId}/terminate`,
        {
          method: "POST",
          headers: {
            ...(this.apiKey && { Authorization: `Bearer ${this.apiKey}` }),
          },
        },
      );
    } catch (error) {
      console.error("Failed to terminate conversation:", error);
      throw error;
    }
  }

  /**
   * Execute a predefined legal workflow
   */
  async executeLegalWorkflow(
    workflowType: "case_analysis" | "evidence_review" | "legal_research",
    input: string,
    context: Record<string, any> = {},
  ): Promise<AIResponse> {
    const team = this.createLegalAgentTeam();
    let agents: AutoGenAgent[] = [];
    let initialPrompt = "";

    switch (workflowType) {
      case "case_analysis":
        agents = [team.prosecutor, team.legalResearcher, team.coordinator];
        initialPrompt = `Please analyze the following case for prosecutorial merit and legal strategy:\n\n${input}`;
        break;

      case "evidence_review":
        agents = [team.evidenceAnalyst, team.prosecutor, team.coordinator];
        initialPrompt = `Please review and analyze the following evidence:\n\n${input}`;
        break;

      case "legal_research":
        agents = [team.legalResearcher, team.prosecutor, team.coordinator];
        initialPrompt = `Please research legal precedents and applicable law for:\n\n${input}`;
        break;
    }

    try {
      const conversation = await this.startConversation(
        agents,
        initialPrompt,
        context,
      );

      // Wait for conversation completion
      let status = "active";
      let attempts = 0;
      const maxAttempts = 30; // 5 minutes with 10-second intervals

      while (status === "active" && attempts < maxAttempts) {
        await new Promise((resolve) => setTimeout(resolve, 10000)); // Wait 10 seconds

        const updated = await this.getConversation(conversation.id);
        status = updated.status;
        attempts++;
      }

      const finalConversation = await this.getConversation(conversation.id);

      // Extract final response from coordinator
      const coordinatorMessages = finalConversation.messages.filter(
        (m) => m.sender === "coordinator",
      );

      const finalMessage = coordinatorMessages[coordinatorMessages.length - 1];

      return {
        id: crypto.randomUUID(),
        content:
          finalMessage?.content || "Workflow completed without final message",
        providerId: "autogen",
        model: "autogen-agents",
        tokensUsed: finalConversation.messages.length * 100, // Estimate
        responseTime:
          (finalConversation.endTime || Date.now()) -
          finalConversation.startTime,
        metadata: {
          conversationId: conversation.id,
          messagesCount: finalConversation.messages.length,
          participants: agents.map((a) => a.name),
          workflowType,
        },
      };
    } catch (error) {
      console.error("Legal workflow execution failed:", error);
      throw error;
    }
  }

  /**
   * Health check for AutoGen service
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
   * Get available models and capabilities
   */
  async getCapabilities(): Promise<{
    models: string[];
    tools: string[];
    maxAgents: number;
    supportedWorkflows: string[];
  }> {
    try {
      const response = await fetch(`${this.baseUrl}/api/capabilities`, {
        method: "GET",
        headers: {
          ...(this.apiKey && { Authorization: `Bearer ${this.apiKey}` }),
        },
      });

      if (!response.ok) {
        throw new Error("Failed to get capabilities");
      }

      return await response.json();
    } catch (error) {
      console.error("Failed to get capabilities:", error);
      // Return defaults
      return {
        models: ["gemma3-legal", "llama3:8b-instruct", "codellama:7b-code"],
        tools: [
          "legal_database_search",
          "case_precedent_lookup",
          "evidence_validator",
        ],
        maxAgents: 5,
        supportedWorkflows: [
          "case_analysis",
          "evidence_review",
          "legal_research",
        ],
      };
    }
  }

  /**
   * Create a custom agent with specific configuration
   */
  createCustomAgent(
    name: string,
    role: string,
    systemMessage: string,
    model: string = "gemma3-legal",
    tools: string[] = [],
  ): AutoGenAgent {
    return {
      name,
      systemMessage: `${systemMessage}\n\nYour role is: ${role}`,
      llmConfig: {
        model,
        temperature: 0.2,
        maxTokens: 1024,
        apiBase: "http://localhost:11434",
      },
      humanInputMode: "NEVER",
      maxConsecutiveAutoReply: 3,
      tools,
    };
  }

  /**
   * Stream conversation updates
   */
  async *streamConversation(
    conversationId: string,
  ): AsyncGenerator<AutoGenMessage, void, unknown> {
    const eventSource = new EventSource(
      `${this.baseUrl}/api/conversation/${conversationId}/stream`,
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
          const message: AutoGenMessage = JSON.parse(event.data);
          yield message;
        } catch (error) {
          console.error("Failed to parse streaming message:", error);
        }
      }
    } finally {
      eventSource.close();
    }
  }
}

// Singleton instance
export const autoGenService = new AutoGenService();

// Helper functions for common workflows
export async function analyzeCaseWithAgents(
  caseDescription: string,
  evidenceList: string[] = [],
  jurisdiction: string = "federal",
): Promise<AIResponse> {
  const context = {
    evidenceCount: evidenceList.length,
    jurisdiction,
    analysisType: "comprehensive",
  };

  const input = `
Case Description: ${caseDescription}

Evidence Available:
${evidenceList.map((evidence, index) => `${index + 1}. ${evidence}`).join("\n")}

Jurisdiction: ${jurisdiction}

Please provide a comprehensive analysis including legal theories, evidence evaluation, and prosecution recommendations.
  `.trim();

  return autoGenService.executeLegalWorkflow("case_analysis", input, context);
}

export async function reviewEvidenceWithAgents(
  evidenceDescription: string,
  evidenceType: string = "digital",
  chainOfCustody: string[] = [],
): Promise<AIResponse> {
  const context = {
    evidenceType,
    custodySteps: chainOfCustody.length,
    reviewType: "admissibility",
  };

  const input = `
Evidence Description: ${evidenceDescription}
Evidence Type: ${evidenceType}

Chain of Custody:
${chainOfCustody.map((step, index) => `${index + 1}. ${step}`).join("\n")}

Please evaluate this evidence for authenticity, reliability, and admissibility in court.
  `.trim();

  return autoGenService.executeLegalWorkflow("evidence_review", input, context);
}

export async function researchLegalPrecedents(
  legalQuestion: string,
  jurisdiction: string = "federal",
  caseType: string = "criminal",
): Promise<AIResponse> {
  const context = {
    jurisdiction,
    caseType,
    researchDepth: "comprehensive",
  };

  const input = `
Legal Question: ${legalQuestion}
Jurisdiction: ${jurisdiction}
Case Type: ${caseType}

Please research relevant case law, statutes, and legal precedents that apply to this question.
  `.trim();

  return autoGenService.executeLegalWorkflow("legal_research", input, context);
}
