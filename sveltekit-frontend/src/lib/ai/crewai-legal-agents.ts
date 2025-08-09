// @ts-nocheck
// CrewAI Legal Document Review Multi-Agent System
// Integrates Claude Code CLI + Local Gemma3 + Self-Prompting + Auto-Save

import { ChatOllama } from '@langchain/ollama';
import { HumanMessage, SystemMessage } from '@langchain/core/messages';
import { documentUpdateLoop } from '$lib/services/documentUpdateLoop';
import { db } from '$lib/server/database';
import { documents, aiHistory, cases } from '$lib/db/schema';
import { eq } from 'drizzle-orm';

// ============================================================================
// AGENT DEFINITIONS & TYPES
// ============================================================================

export interface LegalAgent {
  id: string;
  name: string;
  role: string;
  expertise: string[];
  model: 'claude' | 'gemma3' | 'gpt-4';
  systemPrompt: string;
  maxTokens: number;
  temperature: number;
}

export interface DocumentReviewTask {
  taskId: string;
  documentId: string;
  documentContent: string;
  reviewType: 'comprehensive' | 'compliance' | 'risk_assessment' | 'quick_scan';
  priority: 'low' | 'medium' | 'high' | 'urgent';
  assignedAgents: string[];
  context?: {
    caseType?: string;
    jurisdiction?: string;
    clientGoals?: string[];
    riskTolerance?: 'low' | 'medium' | 'high';
  };
}

export interface AgentResponse {
  agentId: string;
  taskId: string;
  status: 'completed' | 'failed' | 'in_progress';
  analysis: {
    summary: string;
    keyFindings: string[];
    risks: Array<{
      type: string;
      severity: 'low' | 'medium' | 'high' | 'critical';
      description: string;
      recommendation: string;
    }>;
    recommendations: string[];
    confidence: number;
    processingTime: number;
  };
  inlineEdits?: Array<{
    position: number;
    length: number;
    originalText: string;
    suggestedText: string;
    reasoning: string;
    confidence: number;
  }>;
  metadata: {
    modelUsed: string;
    tokensUsed: number;
    timestamp: string;
  };
}

// ============================================================================
// LEGAL AGENT DEFINITIONS
// ============================================================================

export const LEGAL_AGENTS: Record<string, LegalAgent> = {
  // Legal Compliance Specialist
  compliance_specialist: {
    id: 'compliance_specialist',
    name: 'Legal Compliance Specialist',
    role: 'Compliance Analysis',
    expertise: ['regulatory_compliance', 'legal_standards', 'jurisdiction_analysis'],
    model: 'gemma3',
    systemPrompt: `You are a Legal Compliance Specialist with expertise in regulatory compliance and legal standards analysis.

Your responsibilities:
- Identify potential compliance issues and regulatory violations
- Check adherence to jurisdictional requirements
- Flag missing mandatory clauses or disclosures
- Assess legal language accuracy and precision
- Provide specific compliance recommendations

Analysis approach:
1. Scan for jurisdiction-specific requirements
2. Identify missing or inadequate legal provisions
3. Check for contradictory clauses
4. Assess regulatory compliance risks
5. Provide actionable remediation steps

Response format: Provide structured analysis with specific references to legal standards and regulations.`,
    maxTokens: 2000,
    temperature: 0.3
  },

  // Risk Assessment Analyst
  risk_analyst: {
    id: 'risk_analyst',
    name: 'Legal Risk Analyst',
    role: 'Risk Assessment',
    expertise: ['risk_analysis', 'litigation_risk', 'financial_exposure', 'strategic_risk'],
    model: 'claude',
    systemPrompt: `You are a Legal Risk Analyst specializing in comprehensive risk assessment of legal documents and situations.

Your responsibilities:
- Identify and categorize potential legal risks
- Assess likelihood and impact of various risk scenarios
- Evaluate financial and reputational exposure
- Provide risk mitigation strategies
- Priority-rank risks for management attention

Risk categories to analyze:
- Litigation risk
- Financial exposure
- Regulatory compliance risk
- Reputational risk
- Operational risk
- Strategic risk

For each risk identified:
- Severity: Critical/High/Medium/Low
- Likelihood: Very Likely/Likely/Possible/Unlikely
- Potential impact: Financial, operational, reputational
- Recommended mitigation actions
- Timeline for action

Provide clear, actionable risk assessments with specific recommendations.`,
    maxTokens: 2500,
    temperature: 0.4
  },

  // Contract Review Specialist
  contract_specialist: {
    id: 'contract_specialist',
    name: 'Contract Review Specialist',
    role: 'Contract Analysis',
    expertise: ['contract_law', 'terms_analysis', 'negotiation_strategy', 'liability_assessment'],
    model: 'gemma3',
    systemPrompt: `You are a Contract Review Specialist with deep expertise in contract law and commercial agreements.

Your responsibilities:
- Analyze contract terms and provisions
- Identify favorable and unfavorable clauses
- Assess liability and indemnification provisions
- Review termination and dispute resolution clauses
- Provide negotiation strategy recommendations

Key analysis areas:
1. Core terms (price, scope, timeline, deliverables)
2. Liability and indemnification clauses
3. Termination and renewal provisions
4. Intellectual property provisions
5. Dispute resolution mechanisms
6. Force majeure and risk allocation
7. Confidentiality and non-disclosure terms

For each provision, assess:
- Client favor level (highly favorable, favorable, neutral, unfavorable, highly unfavorable)
- Risk level associated with the provision
- Specific negotiation recommendations
- Alternative language suggestions

Provide specific, actionable contract review insights.`,
    maxTokens: 2200,
    temperature: 0.35
  },

  // Legal Editor & Style Reviewer  
  legal_editor: {
    id: 'legal_editor',
    name: 'Legal Editor & Style Reviewer',
    role: 'Document Editing',
    expertise: ['legal_writing', 'document_structure', 'clarity_improvement', 'style_consistency'],
    model: 'claude',
    systemPrompt: `You are a Legal Editor and Style Reviewer specializing in improving legal document clarity, structure, and effectiveness.

Your responsibilities:
- Improve document structure and organization
- Enhance clarity and readability while maintaining legal precision
- Ensure consistent terminology and style
- Identify and correct ambiguous language
- Suggest inline edits for immediate improvement

Focus areas:
1. Document structure and logical flow
2. Sentence clarity and conciseness
3. Consistent terminology usage
4. Proper legal formatting and citations
5. Elimination of ambiguous or confusing language
6. Reader comprehension optimization

For inline edits, provide:
- Exact position and length of text to change
- Original text and suggested replacement
- Clear reasoning for the change
- Confidence level in the improvement

Maintain legal accuracy while improving accessibility and clarity.`,
    maxTokens: 1800,
    temperature: 0.5
  }
};

// ============================================================================
// CREWAI ORCHESTRATOR
// ============================================================================

export class CrewAILegalOrchestrator {
  private ollamaClient: ChatOllama;
  private claudeClient: any; // Claude Code CLI integration
  private activeReviews: Map<string, DocumentReviewTask> = new Map();

  constructor() {
    this.ollamaClient = new ChatOllama({
      baseUrl: 'http://localhost:11434',
      model: 'gemma3-legal', // Your fine-tuned legal model
    });

    // Initialize Claude Code CLI integration
    this.initializeClaudeIntegration();
  }

  private async initializeClaudeIntegration() {
    // Integration with Claude Code CLI will be implemented
    // For now, we'll use a placeholder
    this.claudeClient = {
      async chat(messages: any[]) {
        // This would integrate with Claude Code CLI
        // For now, fallback to Ollama
        return await this.ollamaClient.invoke(messages);
      }
    };
  }

  // ============================================================================
  // MAIN REVIEW ORCHESTRATION
  // ============================================================================

  async startDocumentReview(task: DocumentReviewTask): Promise<string> {
    console.log(`🚀 Starting CrewAI review for document: ${task.documentId}`);
    
    this.activeReviews.set(task.taskId, task);

    try {
      // Phase 1: Parallel agent analysis
      const agentPromises = task.assignedAgents.map(agentId: any => 
        this.runAgentAnalysis(agentId, task)
      );

      const agentResponses = await Promise.allSettled(agentPromises);
      const successfulResponses = agentResponses
        .filter(result: any => result.status === 'fulfilled')
        .map(result: any => (result as PromiseFulfilledResult<AgentResponse>).value);

      // Phase 2: Synthesize results
      const synthesis = await this.synthesizeAgentResults(task, successfulResponses);

      // Phase 3: Generate final recommendations
      const finalReview = await this.generateFinalReview(task, successfulResponses, synthesis);

      // Phase 4: Store results and trigger updates
      await this.storeReviewResults(task, finalReview);

      // Phase 5: Queue document update if changes suggested
      if (finalReview.inlineEdits && finalReview.inlineEdits.length > 0) {
        await this.queueDocumentUpdates(task, finalReview);
      }

      console.log(`✅ CrewAI review completed for ${task.documentId}`);
      return task.taskId;

    } catch (error) {
      console.error(`❌ CrewAI review failed for ${task.documentId}:`, error);
      throw error;
    } finally {
      this.activeReviews.delete(task.taskId);
    }
  }

  // ============================================================================
  // INDIVIDUAL AGENT EXECUTION
  // ============================================================================

  private async runAgentAnalysis(agentId: string, task: DocumentReviewTask): Promise<AgentResponse> {
    const agent = LEGAL_AGENTS[agentId];
    if (!agent) {
      throw new Error(`Agent ${agentId} not found`);
    }

    const startTime = Date.now();
    console.log(`🤖 Running ${agent.name} analysis...`);

    try {
      // Prepare context-aware prompt
      const contextPrompt = this.buildAgentPrompt(agent, task);
      
      let response;
      let tokensUsed = 0;

      // Route to appropriate model
      if (agent.model === 'claude') {
        response = await this.claudeClient.chat([
          new SystemMessage(agent.systemPrompt),
          new HumanMessage(contextPrompt)
        ]);
      } else {
        response = await this.ollamaClient.invoke([
          new SystemMessage(agent.systemPrompt),
          new HumanMessage(contextPrompt)
        ]);
      }

      // Parse agent response
      const analysis = this.parseAgentResponse(response.content, agent);
      
      const agentResponse: AgentResponse = {
        agentId,
        taskId: task.taskId,
        status: 'completed',
        analysis: {
          ...analysis,
          processingTime: Date.now() - startTime
        },
        inlineEdits: agent.id === 'legal_editor' ? this.extractInlineEdits(response.content) : undefined,
        metadata: {
          modelUsed: agent.model,
          tokensUsed,
          timestamp: new Date().toISOString()
        }
      };

      console.log(`✅ ${agent.name} completed analysis`);
      return agentResponse;

    } catch (error) {
      console.error(`❌ ${agent.name} analysis failed:`, error);
      
      return {
        agentId,
        taskId: task.taskId,
        status: 'failed',
        analysis: {
          summary: `Analysis failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
          keyFindings: [],
          risks: [],
          recommendations: [],
          confidence: 0,
          processingTime: Date.now() - startTime
        },
        metadata: {
          modelUsed: agent.model,
          tokensUsed: 0,
          timestamp: new Date().toISOString()
        }
      };
    }
  }

  private buildAgentPrompt(agent: LegalAgent, task: DocumentReviewTask): string {
    let prompt = `DOCUMENT REVIEW TASK
Review Type: ${task.reviewType}
Priority: ${task.priority}

DOCUMENT CONTENT:
${task.documentContent}

`;

    if (task.context) {
      prompt += `CONTEXT:
Case Type: ${task.context.caseType || 'Not specified'}
Jurisdiction: ${task.context.jurisdiction || 'Not specified'}
Client Goals: ${task.context.clientGoals?.join(', ') || 'Not specified'}
Risk Tolerance: ${task.context.riskTolerance || 'Not specified'}

`;
    }

    prompt += `Please provide your specialized ${agent.role} analysis according to your expertise in: ${agent.expertise.join(', ')}.

Focus on actionable insights and specific recommendations that align with the client's needs.`;

    return prompt;
  }

  private parseAgentResponse(content: string, agent: LegalAgent): any {
    // Basic parsing - in production, you'd want more sophisticated parsing
    return {
      summary: this.extractSection(content, 'SUMMARY') || 'No summary provided',
      keyFindings: this.extractList(content, 'KEY FINDINGS') || [],
      risks: this.extractRisks(content) || [],
      recommendations: this.extractList(content, 'RECOMMENDATIONS') || [],
      confidence: this.extractConfidence(content) || 0.8
    };
  }

  private extractSection(content: string, sectionName: string): string | null {
    const regex = new RegExp(`${sectionName}:?\\s*([\\s\\S]*?)(?=\\n\\n|$)`, 'i');
    const match = content.match(regex);
    return match ? match[1].trim() : null;
  }

  private extractList(content: string, listName: string): string[] {
    const section = this.extractSection(content, listName);
    if (!section) return [];
    
    return section
      .split(/\n/)
      .map(line: any => line.replace(/^[-*•]\s*/, '').trim())
      .filter(line: any => line.length > 0);
  }

  private extractRisks(content: string): Array<any> {
    // Simplified risk extraction - would be more sophisticated in production
    const riskSection = this.extractSection(content, 'RISKS');
    if (!riskSection) return [];

    return [{
      type: 'general',
      severity: 'medium',
      description: riskSection.substring(0, 200) + '...',
      recommendation: 'Review identified risks carefully'
    }];
  }

  private extractConfidence(content: string): number {
    const confidenceMatch = content.match(/confidence:?\s*(\d+(?:\.\d+)?)/i);
    return confidenceMatch ? parseFloat(confidenceMatch[1]) : 0.8;
  }

  private extractInlineEdits(content: string): Array<any> {
    // Extract inline edit suggestions from legal editor
    // This would be more sophisticated in production
    return [];
  }

  // ============================================================================
  // RESULT SYNTHESIS
  // ============================================================================

  private async synthesizeAgentResults(task: DocumentReviewTask, responses: AgentResponse[]): Promise<any> {
    console.log('🔄 Synthesizing agent results...');

    const allRisks = responses.flatMap(r: any => r.analysis.risks);
    const allRecommendations = responses.flatMap(r: any => r.analysis.recommendations);
    const avgConfidence = responses.reduce((sum, r) => sum + r.analysis.confidence, 0) / responses.length;

    return {
      overallAssessment: this.generateOverallAssessment(responses),
      prioritizedRisks: this.prioritizeRisks(allRisks),
      consolidatedRecommendations: this.consolidateRecommendations(allRecommendations),
      confidence: avgConfidence,
      agentConsensus: this.calculateAgentConsensus(responses)
    };
  }

  private generateOverallAssessment(responses: AgentResponse[]): string {
    const summaries = responses.map(r: any => r.analysis.summary);
    return `Comprehensive review completed by ${responses.length} specialized agents. ${summaries.join(' ')}`;
  }

  private prioritizeRisks(risks: any[]): any[] {
    return risks.sort((a, b) => {
      const severityOrder = { critical: 4, high: 3, medium: 2, low: 1 };
      return (severityOrder[b.severity as keyof typeof severityOrder] || 0) - 
             (severityOrder[a.severity as keyof typeof severityOrder] || 0);
    });
  }

  private consolidateRecommendations(recommendations: string[]): string[] {
    // Remove duplicates and prioritize
    return [...new Set(recommendations)].slice(0, 10);
  }

  private calculateAgentConsensus(responses: AgentResponse[]): number {
    // Calculate how much agents agree (simplified)
    return responses.reduce((sum, r) => sum + r.analysis.confidence, 0) / responses.length;
  }

  private async generateFinalReview(task: DocumentReviewTask, responses: AgentResponse[], synthesis: any): Promise<any> {
    console.log('📋 Generating final review report...');

    return {
      taskId: task.taskId,
      documentId: task.documentId,
      reviewType: task.reviewType,
      completedAt: new Date().toISOString(),
      synthesis,
      agentResponses: responses,
      inlineEdits: responses.find(r: any => r.inlineEdits)?.inlineEdits || [],
      nextSteps: this.generateNextSteps(synthesis),
      qualityScore: this.calculateQualityScore(responses, synthesis)
    };
  }

  private generateNextSteps(synthesis: any): string[] {
    return [
      'Review prioritized risks and implement mitigation strategies',
      'Address critical compliance issues identified',
      'Consider agent recommendations for document improvements',
      'Schedule follow-up review after revisions'
    ];
  }

  private calculateQualityScore(responses: AgentResponse[], synthesis: any): number {
    // Simplified quality scoring
    const avgConfidence = synthesis.confidence;
    const riskCoverage = synthesis.prioritizedRisks.length > 0 ? 1 : 0.5;
    const agentCoverage = responses.length / Object.keys(LEGAL_AGENTS).length;
    
    return Math.round((avgConfidence * 0.4 + riskCoverage * 0.3 + agentCoverage * 0.3) * 100);
  }

  // ============================================================================
  // STORAGE & INTEGRATION
  // ============================================================================

  private async storeReviewResults(task: DocumentReviewTask, finalReview: any): Promise<void> {
    try {
      // Store in AI history
      await db.insert(aiHistory).values({
        caseId: (await this.getCaseIdForDocument(task.documentId)) || undefined,
        userId: 'crewai-system', // System user for agent reviews
        prompt: `CrewAI Document Review - ${task.reviewType}`,
        response: JSON.stringify(finalReview),
        model: 'crewai-multi-agent',
        tokensUsed: finalReview.agentResponses.reduce((sum: number, r: any) => sum + r.metadata.tokensUsed, 0),
        metadata: {
          taskId: task.taskId,
          documentId: task.documentId,
          agentsUsed: task.assignedAgents,
          qualityScore: finalReview.qualityScore
        }
      });

      console.log('💾 Review results stored in database');

    } catch (error) {
      console.error('❌ Failed to store review results:', error);
    }
  }

  private async getCaseIdForDocument(documentId: string): Promise<string | null> {
    try {
      const [doc] = await db
        .select({ caseId: documents.caseId })
        .from(documents)
        .where(eq(documents.id, documentId))
        .limit(1);
      
      return doc?.caseId || null;
    } catch {
      return null;
    }
  }

  private async queueDocumentUpdates(task: DocumentReviewTask, finalReview: any): Promise<void> {
    if (!finalReview.inlineEdits || finalReview.inlineEdits.length === 0) {
      return;
    }

    try {
      // Apply suggested inline edits to create updated content
      let updatedContent = task.documentContent;
      
      // Sort edits by position (descending) to avoid position shifts
      const sortedEdits = finalReview.inlineEdits.sort((a: any, b: any) => b.position - a.position);
      
      for (const edit of sortedEdits) {
        if (edit.confidence > 0.8) { // Only apply high-confidence edits
          updatedContent = updatedContent.slice(0, edit.position) + 
                          edit.suggestedText + 
                          updatedContent.slice(edit.position + edit.length);
        }
      }

      // Queue for re-embedding
      await documentUpdateLoop.queueDocumentUpdate(task.documentId, updatedContent);
      
      console.log('🔄 Queued document update with agent suggestions');

    } catch (error) {
      console.error('❌ Failed to queue document updates:', error);
    }
  }

  // ============================================================================
  // PUBLIC API
  // ============================================================================

  async getActiveReviews(): Promise<DocumentReviewTask[]> {
    return Array.from(this.activeReviews.values());
  }

  async cancelReview(taskId: string): Promise<boolean> {
    if (this.activeReviews.has(taskId)) {
      this.activeReviews.delete(taskId);
      return true;
    }
    return false;
  }

  getAvailableAgents(): LegalAgent[] {
    return Object.values(LEGAL_AGENTS);
  }
}

// ============================================================================
// SINGLETON INSTANCE
// ============================================================================

export const crewAIOrchestrator = new CrewAILegalOrchestrator();