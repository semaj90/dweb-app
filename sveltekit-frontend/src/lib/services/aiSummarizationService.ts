import type { EvidenceItem } from "$lib/types/api";
import crypto from "crypto";
// @ts-nocheck
import type { Case } from "$lib/types";
// Orphaned content: // AI Summarization Service for Case Management
// Integrates with local LLM to generate comprehensive case reports

import { dev

interface CaseData {
  id: string;
  title: string;
  description: string | null;
  evidence: EvidenceItem[];
  activities: CaseActivity[];
  metadata: Record<string, any>;
}
interface CaseActivity {
  id: string;
  activityType: string;
  title: string;
  description?: string;
  status: string;
  completedAt?: Date;
}
interface AISummaryReport {
  id: string;
  caseId: string;
  reportType:
    | "case_overview"
    | "evidence_analysis"
    | "timeline_summary"
    | "prosecution_strategy";
  title: string;
  content: string;
  richTextContent?: any; // Tiptap JSON content
  metadata: {
    generatedAt: string;
    modelUsed: string;
    confidence: number;
    keyPoints: string[];
    recommendations: string[];
    riskFactors: string[];
  };
  canvasElements: CanvasElement[];
  createdAt: Date;
  updatedAt: Date;
}
interface CanvasElement {
  id: string;
  type: "text" | "evidence" | "timeline" | "connection" | "note";
  position: { x: number; y: number };
  size: { width: number; height: number };
  content: any;
  style: Record<string, any>;
  connections: string[];
}
class AISummarizationService {
  private baseUrl: string;
  private modelName: string;

  constructor() {
    this.baseUrl = dev
      ? "http://localhost:11434"
      : import.meta.env.OLLAMA_URL || "http://localhost:11434";
    this.modelName = "gemma3-legal";
  }
  /**
   * Generate comprehensive AI summary report for a case
   */
  async generateCaseSummaryReport(
    caseData: CaseData,
  ): Promise<AISummaryReport> {
    try {
      const prompt = this.buildCaseSummaryPrompt(caseData);
      const response = await this.callLLM(prompt, {
        temperature: 0.3,
        max_tokens: 2000,
        system:
          "Generate a comprehensive legal case summary with structured analysis.",
      });

      const report = this.parseAIResponse(
        response,
        "case_overview",
        caseData.id,
      );

      // Generate canvas elements based on the analysis
      report.canvasElements = await this.generateCanvasElements(
        caseData,
        report,
      );

      return report;
    } catch (error) {
      console.error("Error generating case summary:", error);
      throw new Error("Failed to generate AI case summary");
    }
  }
  /**
   * Generate evidence analysis report
   */
  async generateEvidenceAnalysis(
    evidenceItems: EvidenceItem[],
    caseId: string,
  ): Promise<AISummaryReport> {
    try {
      const prompt = this.buildEvidenceAnalysisPrompt(evidenceItems);
      const response = await this.callLLM(prompt, {
        temperature: 0.2,
        max_tokens: 1500,
        system: "Analyze evidence items for legal prosecution strategy.",
      });

      const report = this.parseAIResponse(
        response,
        "evidence_analysis",
        caseId,
      );
      report.canvasElements =
        this.generateEvidenceCanvasElements(evidenceItems);

      return report;
    } catch (error) {
      console.error("Error generating evidence analysis:", error);
      throw new Error("Failed to generate evidence analysis");
    }
  }
  /**
   * Generate prosecution strategy report
   */
  async generateProsecutionStrategy(
    caseData: CaseData,
  ): Promise<AISummaryReport> {
    try {
      const prompt = this.buildProsecutionStrategyPrompt(caseData);
      const response = await this.callLLM(prompt, {
        temperature: 0.4,
        max_tokens: 2500,
        system: "Generate strategic legal prosecution recommendations.",
      });

      const report = this.parseAIResponse(
        response,
        "prosecution_strategy",
        caseData.id,
      );
      report.canvasElements = this.generateStrategyCanvasElements(caseData);

      return report;
    } catch (error) {
      console.error("Error generating prosecution strategy:", error);
      throw new Error("Failed to generate prosecution strategy");
    }
  }
  /**
   * Generate timeline summary
   */
  async generateTimelineSummary(
    activities: CaseActivity[],
    caseId: string,
  ): Promise<AISummaryReport> {
    try {
      const prompt = this.buildTimelinePrompt(activities);
      const response = await this.callLLM(prompt, {
        temperature: 0.3,
        max_tokens: 1200,
        system: "Create chronological case timeline analysis.",
      });

      const report = this.parseAIResponse(response, "timeline_summary", caseId);
      report.canvasElements = this.generateTimelineCanvasElements(activities);

      return report;
    } catch (error) {
      console.error("Error generating timeline summary:", error);
      throw new Error("Failed to generate timeline summary");
    }
  }
  private buildCaseSummaryPrompt(caseData: CaseData): string {
    return `
CASE ANALYSIS REQUEST

Case Title: ${caseData.title}
Case Description: ${caseData.description || "No description provided"}

EVIDENCE SUMMARY:
${caseData.evidence
  .map(
    (e) => `
- ${e.title} (${e.evidenceType})
  Description: ${e.description || "No description"}
  Type: ${e.fileType || "Unknown"}
  AI Analysis: ${JSON.stringify(e.aiAnalysis || {})}
`,
  )
  .join("\n")}

CASE ACTIVITIES:
${caseData.activities
  .map(
    (a) => `
- ${a.title} (${a.activityType})
  Status: ${a.status}
  Description: ${a.description || "No description"}
  Completed: ${a.completedAt ? new Date(a.completedAt).toLocaleDateString() : "Pending"}
`,
  )
  .join("\n")}

Please provide a comprehensive case summary including:

1. EXECUTIVE SUMMARY
   - Key case facts and timeline
   - Primary legal issues
   - Strength of prosecution case (1-10 scale)

2. EVIDENCE ASSESSMENT
   - Critical evidence items
   - Evidence strengths and weaknesses
   - Chain of custody considerations
   - Admissibility concerns

3. LEGAL STRATEGY
   - Recommended prosecution approach
   - Potential defense arguments
   - Key witnesses needed
   - Timeline recommendations

4. RISK ANALYSIS
   - Case strengths (bullet points)
   - Potential weaknesses (bullet points)
   - Mitigation strategies

5. ACTION ITEMS
   - Immediate next steps
   - Investigation priorities
   - Evidence collection needs

Format the response as structured JSON with these sections clearly marked.
`;
  }
  private buildEvidenceAnalysisPrompt(evidenceItems: EvidenceItem[]): string {
    return `
EVIDENCE ANALYSIS REQUEST

Please analyze the following evidence items for a legal prosecution case:

${evidenceItems
  .map(
    (item) => `
EVIDENCE ITEM: ${item.title}
Type: ${item.evidenceType}
File Type: ${item.fileType || "N/A"}
Description: ${item.description || "No description"}
AI Analysis: ${JSON.stringify(item.aiAnalysis || {})}
Summary: ${item.summary || "No summary"}
---
`,
  )
  .join("\n")}

Provide analysis including:

1. EVIDENCE STRENGTH ASSESSMENT
   - Rate each evidence item (1-10)
   - Admissibility likelihood
   - Corroboration needs

2. EVIDENCE RELATIONSHIPS
   - How evidence items connect
   - Timeline correlations
   - Contradictions or gaps

3. PROSECUTION VALUE
   - Most compelling evidence
   - Supporting evidence
   - Evidence order for presentation

4. DEFENSE CHALLENGES
   - Likely objections
   - Chain of custody issues
   - Authentication concerns

5. RECOMMENDATIONS
   - Additional evidence needed
   - Expert witnesses required
   - Testing or analysis needed

Structure the response as detailed JSON analysis.
`;
  }
  private buildProsecutionStrategyPrompt(caseData: CaseData): string {
    return `
PROSECUTION STRATEGY REQUEST

Develop a comprehensive prosecution strategy for this case:

CASE: ${caseData.title}
DESCRIPTION: ${caseData.description}

AVAILABLE EVIDENCE: ${caseData.evidence.length} items
KEY EVIDENCE TYPES: ${Array.from(new Set(caseData.evidence.map((e) => e.evidenceType))).join(", ")}

CASE ACTIVITIES: ${caseData.activities.length} activities
COMPLETED ACTIVITIES: ${caseData.activities.filter((a) => a.status === "completed").length}

Provide strategic recommendations including:

1. CASE THEORY
   - Central narrative
   - Key legal elements to prove
   - Burden of proof strategy

2. EVIDENCE PRESENTATION ORDER
   - Opening evidence sequence
   - Build-up strategy
   - Closing arguments preparation

3. WITNESS STRATEGY
   - Essential witnesses
   - Expert witness needs
   - Witness preparation priorities

4. POTENTIAL DEFENSES
   - Likely defense strategies
   - Counter-arguments preparation
   - Preemptive responses

5. TIMELINE AND MILESTONES
   - Investigation completion targets
   - Filing deadlines
   - Trial preparation schedule

6. RISK MITIGATION
   - Weak points reinforcement
   - Alternative theories
   - Plea negotiation considerations

Format as strategic prosecution plan in JSON structure.
`;
  }
  private buildTimelinePrompt(activities: CaseActivity[]): string {
    return `
TIMELINE ANALYSIS REQUEST

Analyze the following case activities to create a comprehensive timeline:

${activities
  .map(
    (activity) => `
ACTIVITY: ${activity.title}
Type: ${activity.activityType}
Status: ${activity.status}
Description: ${activity.description || "No description"}
Completed: ${activity.completedAt ? new Date(activity.completedAt).toLocaleDateString() : "Pending"}
---
`,
  )
  .join("\n")}

Provide timeline analysis including:

1. CHRONOLOGICAL SEQUENCE
   - Activities in chronological order
   - Key milestones identification
   - Timeline gaps or overlaps

2. CRITICAL PATH ANALYSIS
   - Dependencies between activities
   - Bottlenecks or delays
   - Parallel work opportunities

3. PROGRESS ASSESSMENT
   - Completion percentage
   - On-time performance
   - Resource utilization

4. FUTURE PLANNING
   - Remaining activities
   - Estimated completion times
   - Resource requirements

5. RECOMMENDATIONS
   - Schedule optimizations
   - Priority adjustments
   - Risk mitigation steps

Structure as comprehensive timeline JSON analysis.
`;
  }
  private async callLLM(prompt: string, options: any = {}): Promise<string> {
    try {
      const response = await fetch(`${this.baseUrl}/api/generate`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model: this.modelName,
          prompt: prompt,
          system: options.system || "",
          stream: false,
          options: {
            temperature: options.temperature || 0.3,
            top_p: 0.9,
            top_k: 40,
            num_predict: options.max_tokens || 2000,
          },
        }),
      });

      if (!response.ok) {
        throw new Error(
          `LLM API error: ${response.status} ${response.statusText}`,
        );
      }
      const data = await response.json();
      return data.response || "";
    } catch (error) {
      console.error("LLM API call failed:", error);
      throw error;
    }
  }
  private parseAIResponse(
    response: string,
    reportType: AISummaryReport["reportType"],
    caseId: string,
  ): AISummaryReport {
    // Try to parse JSON response, fallback to structured text parsing
    let parsedContent;
    try {
      parsedContent = JSON.parse(response);
    } catch {
      parsedContent = this.parseStructuredText(response);
    }
    return {
      id: crypto.randomUUID(),
      caseId,
      reportType,
      title: this.getReportTitle(reportType),
      content: response,
      richTextContent: this.convertToTiptapJSON(parsedContent),
      metadata: {
        generatedAt: new Date().toISOString(),
        modelUsed: this.modelName,
        confidence: this.estimateConfidence(response),
        keyPoints: this.extractKeyPoints(parsedContent),
        recommendations: this.extractRecommendations(parsedContent),
        riskFactors: this.extractRiskFactors(parsedContent),
      },
      canvasElements: [], // Will be populated by specific methods
      createdAt: new Date(),
      updatedAt: new Date(),
    };
  }
  private parseStructuredText(text: string): any {
    // Parse structured text into sections
    const sections: Record<string, string> = {};
    const lines = text.split("\n");
    let currentSection = "general";
    let currentContent = "";

    for (const line of lines) {
      if (line.match(/^\d+\.\s+[A-Z\s]+$/)) {
        if (currentContent.trim()) {
          sections[currentSection] = currentContent.trim();
        }
        currentSection = line
          .toLowerCase()
          .replace(/^\d+\.\s+/, "")
          .replace(/\s+/g, "_");
        currentContent = "";
      } else {
        currentContent += line + "\n";
      }
    }
    if (currentContent.trim()) {
      sections[currentSection] = currentContent.trim();
    }
    return sections;
  }
  private convertToTiptapJSON(content: any): any {
    // Convert parsed content to Tiptap-compatible JSON structure
    const doc = {
      type: "doc",
      content: [],
    };

    if (typeof content === "object") {
      for (const [section, text] of Object.entries(content)) {
        // Add heading
        doc.content.push({
          type: "heading",
          attrs: { level: 2 },
          content: [
            { type: "text", text: section.replace(/_/g, " ").toUpperCase() },
          ],
        });

        // Add content
        doc.content.push({
          type: "paragraph",
          content: [{ type: "text", text: String(text) }],
        });
      }
    } else {
      doc.content.push({
        type: "paragraph",
        content: [{ type: "text", text: String(content) }],
      });
    }
    return doc;
  }
  private getReportTitle(reportType: AISummaryReport["reportType"]): string {
    const titles = {
      case_overview: "AI Case Overview & Analysis",
      evidence_analysis: "AI Evidence Assessment",
      timeline_summary: "AI Timeline Analysis",
      prosecution_strategy: "AI Prosecution Strategy",
    };
    return titles[reportType] || "AI Legal Analysis Report";
  }
  private estimateConfidence(response: string): number {
    // Simple confidence estimation based on response characteristics
    const hasNumbers = /\d+/.test(response);
    const hasStructure = response.includes("1.") || response.includes("•");
    const hasLegalTerms =
      /(evidence|prosecution|defense|admissible|testimony)/i.test(response);
    const length = response.length;

    let confidence = 0.5;
    if (hasNumbers) confidence += 0.1;
    if (hasStructure) confidence += 0.15;
    if (hasLegalTerms) confidence += 0.2;
    if (length > 500) confidence += 0.05;

    return Math.min(confidence, 0.95);
  }
  private extractKeyPoints(content: any): string[] {
    const points: string[] = [];
    const text =
      typeof content === "string" ? content : JSON.stringify(content);

    // Extract bullet points and numbered lists
    const matches =
      text.match(/[•\-\*]\s*([^\n]+)/g) ||
      text.match(/\d+\.\s*([^\n]+)/g) ||
      [];

    return matches
      .map((match) => match.replace(/^[•\-\*\d\.]\s*/, ""))
      .slice(0, 10);
  }
  private extractRecommendations(content: any): string[] {
    const text =
      typeof content === "string" ? content : JSON.stringify(content);
    const recommendations: string[] = [];

    // Look for recommendation keywords
    const recMatches = text.match(/recommend[^\.]*\./gi) || [];
    const shouldMatches = text.match(/should[^\.]*\./gi) || [];

    return [...recMatches, ...shouldMatches].slice(0, 8);
  }
  private extractRiskFactors(content: any): string[] {
    const text =
      typeof content === "string" ? content : JSON.stringify(content);
    const risks: string[] = [];

    // Look for risk-related keywords
    const riskMatches =
      text.match(/(risk|weakness|concern|challenge)[^\.]*\./gi) || [];

    return riskMatches.slice(0, 6);
  }
  private async generateCanvasElements(
    caseData: CaseData,
    report: AISummaryReport,
  ): Promise<CanvasElement[]> {
    const elements: CanvasElement[] = [];

    // Add case overview element
    elements.push({
      id: crypto.randomUUID(),
      type: "text",
      position: { x: 100, y: 100 },
      size: { width: 400, height: 300 },
      content: {
        title: "Case Overview",
        text: report.content.substring(0, 500) + "...",
        reportId: report.id,
      },
      style: {
        backgroundColor: "#f0f9ff",
        borderColor: "#0ea5e9",
        borderWidth: 2,
      },
      connections: [],
    });

    // Add evidence elements
    caseData.evidence.forEach((evidence, index) => {
      elements.push({
        id: evidence.id,
        type: "evidence",
        position: {
          x: 600 + (index % 3) * 250,
          y: 150 + Math.floor(index / 3) * 200,
        },
        size: { width: 200, height: 150 },
        content: {
          title: evidence.title,
          type: evidence.evidenceType,
          description: evidence.description,
          fileType: evidence.fileType,
        },
        style: {
          backgroundColor: "#fef3c7",
          borderColor: "#f59e0b",
        },
        connections: [elements[0].id], // Connect to case overview
      });
    });

    return elements;
  }
  private generateEvidenceCanvasElements(
    evidenceItems: EvidenceItem[],
  ): CanvasElement[] {
    return evidenceItems.map((evidence, index) => ({
      id: evidence.id,
      type: "evidence",
      position: {
        x: 100 + (index % 4) * 220,
        y: 100 + Math.floor(index / 4) * 180,
      },
      size: { width: 200, height: 160 },
      content: {
        title: evidence.title,
        type: evidence.evidenceType,
        description: evidence.description,
        strength: Math.floor(Math.random() * 5) + 6, // Mock strength rating
      },
      style: {
        backgroundColor:
          evidence.evidenceType === "physical" ? "#dcfce7" : "#fef3c7",
        borderColor:
          evidence.evidenceType === "physical" ? "#16a34a" : "#f59e0b",
      },
      connections: [],
    }));
  }
  private generateStrategyCanvasElements(caseData: CaseData): CanvasElement[] {
    const elements: CanvasElement[] = [];

    // Central strategy node
    elements.push({
      id: crypto.randomUUID(),
      type: "text",
      position: { x: 400, y: 200 },
      size: { width: 300, height: 150 },
      content: {
        title: "Prosecution Strategy",
        text: "Central case strategy and approach",
      },
      style: {
        backgroundColor: "#fce7f3",
        borderColor: "#ec4899",
        borderWidth: 3,
      },
      connections: [],
    });

    // Evidence strategy branches
    const evidenceTypes = Array.from(
      new Set(caseData.evidence.map((e) => e.evidenceType)),
    );
    evidenceTypes.forEach((type, index) => {
      const angle = (index / evidenceTypes.length) * 2 * Math.PI;
      const radius = 250;
      const x = 400 + Math.cos(angle) * radius;
      const y = 200 + Math.sin(angle) * radius;

      const element = {
        id: crypto.randomUUID(),
        type: "text" as
          | "text"
          | "evidence"
          | "connection"
          | "timeline"
          | "note",
        position: { x: x - 75, y: y - 50 },
        size: { width: 150, height: 100 },
        content: {
          title: `${type} Evidence`,
          text: `Strategy for ${type} evidence presentation`,
        },
        style: {
          backgroundColor: "#ddd6fe",
          borderColor: "#8b5cf6",
        },
        connections: [elements[0].id],
      };

      elements.push(element);
    });

    return elements;
  }
  private generateTimelineCanvasElements(
    activities: CaseActivity[],
  ): CanvasElement[] {
    const elements: CanvasElement[] = [];
    const sortedActivities = activities.sort((a, b) => {
      const dateA = a.completedAt ? new Date(a.completedAt) : new Date();
      const dateB = b.completedAt ? new Date(b.completedAt) : new Date();
      return dateA.getTime() - dateB.getTime();
    });

    sortedActivities.forEach((activity, index) => {
      elements.push({
        id: activity.id,
        type: "timeline",
        position: { x: 100 + index * 200, y: 200 },
        size: { width: 180, height: 120 },
        content: {
          title: activity.title,
          type: activity.activityType,
          status: activity.status,
          date: activity.completedAt,
        },
        style: {
          backgroundColor:
            activity.status === "completed" ? "#dcfce7" : "#fef3c7",
          borderColor: activity.status === "completed" ? "#16a34a" : "#f59e0b",
        },
        connections: index > 0 ? [elements[index - 1].id] : [],
      });
    });

    return elements;
  }
}
export const aiSummarizationService = new AISummarizationService();
