// Web Worker context - no imports needed
declare const self: DedicatedWorkerGlobalScope;
// fetch is available globally in Web Workers

/**
 * Phase 4: Analysis Worker
 * Handles legal case analysis, pattern recognition, and AI-powered insights
 */

// Type definitions
export interface WorkerConfig {
  ollamaUrl: string;
  analysisModel: string;
  maxAnalysisLength: number;
  confidenceThreshold: number;
  timeout: number;
}

export interface WorkerStats {
  analyzed: number;
  patterns: number;
  recommendations: number;
  errors: number;
}

export interface LegalPatterns {
  evidenceTypes: Record<string, string[]>;
  chargePatterns: Record<string, string[]>;
  strengthIndicators: string[];
  weaknessIndicators: string[];
}

export interface Evidence {
  id?: string;
  type?: string;
  description?: string;
  date?: string;
  timestamp?: string;
  location?: string;
  person?: string;
  reliability?: string;
}

export interface Charge {
  statute?: string;
  description?: string;
  severity?: string;
}

export interface Witness {
  reliability?: string;
}

export interface Person {
  id?: string;
  name?: string;
  role?: string;
}

export interface TimelineEvent {
  date?: string;
  timestamp?: string;
  type?: string;
  description?: string;
}

export interface CaseData {
  id?: string;
  title?: string;
  status?: string;
  evidence?: Evidence[];
  charges?: Charge[];
  people?: Person[];
  timeline?: TimelineEvent[];
  witnesses?: Witness[];
  filingDate?: string;
}

export interface TaskMessage {
  taskId: string;
  data: {
    type: string;
    case?: CaseData;
    evidence?: Evidence[];
    items?: any[];
    caseData?: CaseData;
    analysisData?: any;
    query?: string;
  };
  options?: any;
}

export interface AnalysisResult {
  caseId?: string;
  summary: any;
  evidence: any;
  patterns: any;
  strength: any;
  aiInsights: any;
  recommendations: any;
  timeline: any;
  riskFactors: any[];
  processingTime: number;
  workerId: string;
  timestamp: string;
}

export interface StrengthAssessment {
  overallStrength: {
    level: string;
    confidence: number;
    score: number;
  };
  strengths: string[];
  weaknesses: string[];
  recommendations: string[];
  factors: Record<string, any>;
}

export interface EvidenceAnalysis {
  total: number;
  byType: Record<string, number>;
  strengths: string[];
  weaknesses: string[];
  recommendations: string[];
}

export interface PatternAnalysis {
  temporal: any;
  categorical: any;
  relational: any;
  anomalies: any[];
}

export interface ScoreResult {
  score: number;
  maxPossible: number;
  normalized: number;
  details: string;
}

export interface RiskFactor {
  type: string;
  level: string;
  description: string;
}

export interface Recommendation {
  type: string;
  priority: string;
  action: string;
  rationale: string;
}

class AnalysisWorker {
  public workerId: string;
  private config: WorkerConfig;
  private analysisCache: Map<string, any>;
  public legalPatterns: LegalPatterns;
  private stats: WorkerStats;

  constructor() {
    this.workerId = workerData?.workerId || "analysis-worker";
    this.config = {
      ollamaUrl: process.env.OLLAMA_BASE_URL || "http://localhost:11435",
      analysisModel: process.env.OLLAMA_MODEL || "gemma3-legal",
      maxAnalysisLength: 8192,
      confidenceThreshold: 0.7,
      timeout: 60000,
    };

    this.analysisCache = new Map();
    this.legalPatterns = this.initializeLegalPatterns();
    this.stats = {
      analyzed: 0,
      patterns: 0,
      recommendations: 0,
      errors: 0,
    };

    console.log(`⚖️ Analysis Worker ${this.workerId} initialized`);
  }

  /**
   * Initialize legal analysis patterns
   */
  private initializeLegalPatterns(): LegalPatterns {
    return {
      evidenceTypes: {
        physical: ["weapon", "drugs", "fingerprint", "dna", "blood"],
        documentary: ["contract", "receipt", "email", "text", "record"],
        testimonial: ["witness", "victim", "confession", "statement"],
        digital: ["video", "photo", "metadata", "logs", "surveillance"],
      },
      chargePatterns: {
        violent: ["assault", "battery", "murder", "rape", "robbery"],
        property: ["theft", "burglary", "vandalism", "fraud", "embezzlement"],
        drug: ["possession", "distribution", "manufacturing", "trafficking"],
        white_collar: [
          "fraud",
          "insider trading",
          "money laundering",
          "bribery",
        ],
      },
      strengthIndicators: [
        "multiple witnesses",
        "physical evidence",
        "confession",
        "video evidence",
        "dna match",
        "expert testimony",
      ],
      weaknessIndicators: [
        "single witness",
        "circumstantial evidence",
        "recanted statement",
        "chain of custody issues",
        "unreliable witness",
        "contaminated evidence",
      ],
    };
  }

  /**
   * Process incoming messages
   */
  handleMessage(message: TaskMessage): void {
    const { taskId, data, options } = message;

    try {
      let result: any;

      switch (data.type) {
        case "analyze_case":
          result = this.analyzeCase(data.case!, options);
          break;
        case "analyze_evidence":
          result = this.analyzeEvidence(data.evidence!, options);
          break;
        case "pattern_analysis":
          result = this.performPatternAnalysis(data.items!, options);
          break;
        case "strength_assessment":
          result = this.assessCaseStrength(data.caseData!, options);
          break;
        case "generate_recommendations":
          result = this.generateRecommendations(data.analysisData!, options);
          break;
        case "legal_research":
          result = this.performLegalResearch(data.query!, options);
          break;
        default:
          throw new Error(`Unknown analysis task type: ${data.type}`);
      }

      // Handle async results
      if (result instanceof Promise) {
        result
          .then((asyncResult: any) => {
            self.postMessage({
              taskId,
              success: true,
              data: asyncResult,
            });
          })
          .catch((error: Error) => {
            this.stats.errors++;
            self.postMessage({
              taskId,
              success: false,
              error: error.message,
            });
          });
      } else {
        self.postMessage({
          taskId,
          success: true,
          data: result,
        });
      }
    } catch (error: any) {
      console.error(`❌ Analysis error in ${this.workerId}:`, error);
      this.stats.errors++;
      self.postMessage({
        taskId,
        success: false,
        error: error instanceof Error ? error.message : String(error),
      });
    }
  }

  /**
   * Analyze a complete case
   */
  private async analyzeCase(caseData: CaseData, options: any = {}): Promise<AnalysisResult> {
    console.log(`⚖️ Analyzing case: ${caseData.id || "Unknown"}`);

    try {
      const startTime = Date.now();

      // Extract key components
      const evidence = caseData.evidence || [];
      const charges = caseData.charges || [];
      const people = caseData.people || [];
      const timeline = caseData.timeline || [];

      // Perform multiple analysis types
      const [
        evidenceAnalysis,
        patternAnalysis,
        strengthAssessment,
        aiAnalysis,
      ] = await Promise.all([
        this.analyzeEvidenceSet(evidence),
        this.performPatternAnalysis([...evidence, ...charges]),
        this.assessCaseStrength(caseData),
        this.performAIAnalysis(caseData, options),
      ]);

      // Generate comprehensive recommendations
      const recommendations = await this.generateRecommendations({
        evidence: evidenceAnalysis,
        patterns: patternAnalysis,
        strength: strengthAssessment,
        ai: aiAnalysis,
      });

      const analysis: AnalysisResult = {
        caseId: caseData.id,
        summary: this.generateCaseSummary(caseData, strengthAssessment),
        evidence: evidenceAnalysis,
        patterns: patternAnalysis,
        strength: strengthAssessment,
        aiInsights: aiAnalysis,
        recommendations,
        timeline: this.analyzeTimeline(timeline),
        riskFactors: this.identifyRiskFactors(caseData),
        processingTime: Date.now() - startTime,
        workerId: this.workerId,
        timestamp: new Date().toISOString(),
      };

      this.stats.analyzed++;
      return analysis;
    } catch (error: any) {
      throw new Error(`Case analysis failed: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  /**
   * Generate case summary
   */
  private generateCaseSummary(caseData: CaseData, strengthAssessment: StrengthAssessment): any {
    const evidenceCount = caseData.evidence?.length || 0;
    const chargeCount = caseData.charges?.length || 0;
    const strength = strengthAssessment.overallStrength;

    return {
      title: caseData.title || "Untitled Case",
      evidenceCount,
      chargeCount,
      strength: strength.level,
      confidence: strength.confidence,
      keyPoints: this.extractKeyPoints(caseData),
      urgentActions: this.identifyUrgentActions(caseData, strengthAssessment),
    };
  }

  /**
   * Extract key points from case data
   */
  private extractKeyPoints(caseData: CaseData): string[] {
    const points: string[] = [];

    // Evidence-based points
    if (caseData.evidence?.length && caseData.evidence.length > 0) {
      const physicalEvidence = caseData.evidence.filter((e: Evidence) =>
        this.legalPatterns.evidenceTypes.physical.some((type: string) =>
          e.type?.toLowerCase().includes(type)
        )
      );

      if (physicalEvidence.length > 0) {
        points.push(`${physicalEvidence.length} pieces of physical evidence`);
      }
    }

    // Charge-based points
    if (caseData.charges?.length && caseData.charges.length > 0) {
      const chargeTypes = new Set<string>();
      caseData.charges.forEach((charge: Charge) => {
        for (const [category, patterns] of Object.entries(
          this.legalPatterns.chargePatterns
        )) {
          if (
            patterns.some((pattern: string) =>
              charge.description?.toLowerCase().includes(pattern)
            )
          ) {
            chargeTypes.add(category);
          }
        }
      });

      if (chargeTypes.size > 0) {
        points.push(`Charges include: ${Array.from(chargeTypes).join(", ")}`);
      }
    }

    return points;
  }

  /**
   * Identify urgent actions
   */
  private identifyUrgentActions(caseData: CaseData, strengthAssessment: StrengthAssessment): string[] {
    const actions: string[] = [];

    // Based on strength assessment
    if (strengthAssessment.overallStrength.confidence < 0.5) {
      actions.push("Gather additional evidence to strengthen case");
    }

    // Based on evidence issues
    if (
      strengthAssessment.weaknesses.some((w: string) => w.includes("chain of custody"))
    ) {
      actions.push("Review and document evidence chain of custody");
    }

    // Based on timeline
    const today = new Date();
    if (caseData.filingDate) {
      const filingDate = new Date(caseData.filingDate);
      const daysSinceFiling = (today.getTime() - filingDate.getTime()) / (1000 * 60 * 60 * 24);

      if (daysSinceFiling > 30) {
        actions.push(
          "Case approaching statutory deadlines - prioritize completion"
        );
      }
    }

    return actions;
  }

  /**
   * Analyze evidence set
   */
  private async analyzeEvidenceSet(evidence: Evidence[]): Promise<EvidenceAnalysis> {
    const analysis: EvidenceAnalysis = {
      total: evidence.length,
      byType: {},
      strengths: [],
      weaknesses: [],
      recommendations: [],
    };

    // Categorize evidence
    for (const item of evidence) {
      const type = this.classifyEvidence(item);
      analysis.byType[type] = (analysis.byType[type] || 0) + 1;
    }

    // Identify strengths and weaknesses
    for (const item of evidence) {
      this.assessEvidenceStrength(item, analysis);
    }

    return analysis;
  }

  /**
   * Classify evidence type
   */
  private classifyEvidence(evidence: Evidence): string {
    const description = (
      evidence.description ||
      evidence.type ||
      ""
    ).toLowerCase();

    for (const [category, keywords] of Object.entries(
      this.legalPatterns.evidenceTypes
    )) {
      if (keywords.some((keyword: string) => description.includes(keyword))) {
        return category;
      }
    }

    return "other";
  }

  /**
   * Assess individual evidence strength
   */
  private assessEvidenceStrength(evidence: Evidence, analysis: EvidenceAnalysis): void {
    const description = (evidence.description || "").toLowerCase();

    // Check for strength indicators
    for (const indicator of this.legalPatterns.strengthIndicators) {
      if (description.includes(indicator.toLowerCase())) {
        analysis.strengths.push(
          `Strong evidence: ${evidence.type} (${indicator})`
        );
      }
    }

    // Check for weakness indicators
    for (const indicator of this.legalPatterns.weaknessIndicators) {
      if (description.includes(indicator.toLowerCase())) {
        analysis.weaknesses.push(
          `Potential weakness: ${evidence.type} (${indicator})`
        );
      }
    }
  }

  /**
   * Perform pattern analysis
   */
  private performPatternAnalysis(items: any[], options: any = {}): PatternAnalysis {
    const patterns: PatternAnalysis = {
      temporal: this.findTemporalPatterns(items),
      categorical: this.findCategoricalPatterns(items),
      relational: this.findRelationalPatterns(items),
      anomalies: this.findAnomalies(items),
    };

    this.stats.patterns++;
    return patterns;
  }

  /**
   * Find temporal patterns
   */
  private findTemporalPatterns(items: any[]): any {
    const datedItems = items.filter((item: any) => {
      const obj = item as any;
      return obj.date || obj.timestamp;
    });

    if (datedItems.length < 2) {
      return { message: "Insufficient temporal data" };
    }

    // Sort by date
    datedItems.sort(
      (a: any, b: any) => {
        const objA = a as any;
        const objB = b as any;
        return new Date(objA.date || objA.timestamp).getTime() - new Date(objB.date || objB.timestamp).getTime();
      }
    );

    return {
      timeline: datedItems.map((item: any) => {
        const obj = item as any;
        return {
          date: obj.date || obj.timestamp,
          type: obj.type,
          description: obj.description,
        };
      }),
      timespan: {
        start: (datedItems[0] as any).date || (datedItems[0] as any).timestamp,
        end:
          (datedItems[datedItems.length - 1] as any).date ||
          (datedItems[datedItems.length - 1] as any).timestamp,
      },
    };
  }

  /**
   * Find categorical patterns
   */
  private findCategoricalPatterns(items: any[]): any {
    const categories: Record<string, number> = {};

    for (const item of items) {
      const obj = item as any;
      const category = obj.type || obj.category || "uncategorized";
      categories[category] = (categories[category] || 0) + 1;
    }

    return {
      distribution: categories,
      mostCommon: Object.entries(categories).sort((a, b) => b[1] - a[1])[0],
      diversity: Object.keys(categories).length,
    };
  }

  /**
   * Find relational patterns
   */
  private findRelationalPatterns(items: any[]): any {
    const relationships: any[] = [];

    // Look for items that reference each other
    for (let i = 0; i < items.length; i++) {
      for (let j = i + 1; j < items.length; j++) {
        const item1 = items[i];
        const item2 = items[j];

        const relationship = this.findRelationship(item1, item2);
        if (relationship) {
          relationships.push(relationship);
        }
      }
    }

    return {
      relationships,
      count: relationships.length,
      types: Array.from(new Set(relationships.map((r: any) => (r as any).type))),
    };
  }

  /**
   * Find relationship between two items
   */
  private findRelationship(item1: any, item2: any): any | null {
    const obj1 = item1 as any;
    const obj2 = item2 as any;
    
    // Same location
    if (obj1.location && obj2.location && obj1.location === obj2.location) {
      return {
        type: "location",
        items: [obj1.id, obj2.id],
        description: `Both items associated with ${obj1.location}`,
      };
    }

    // Same person
    if (obj1.person && obj2.person && obj1.person === obj2.person) {
      return {
        type: "person",
        items: [obj1.id, obj2.id],
        description: `Both items associated with ${obj1.person}`,
      };
    }

    // Same date
    if (obj1.date && obj2.date && obj1.date === obj2.date) {
      return {
        type: "temporal",
        items: [obj1.id, obj2.id],
        description: `Both items occurred on ${obj1.date}`,
      };
    }

    return null;
  }

  /**
   * Find anomalies in the data
   */
  private findAnomalies(items: any[]): any[] {
    const anomalies: any[] = [];

    // Check for missing critical information
    const missingInfo = items.filter(
      (item: any) => {
        const obj = item as any;
        return !obj.date || !obj.type || !obj.description;
      }
    );
    if (missingInfo.length > 0) {
      anomalies.push({
        type: "missing_information",
        count: missingInfo.length,
        description: "Items with missing critical information",
      });
    }

    // Check for duplicate items
    const seen = new Set<string>();
    const duplicates: any[] = [];
    for (const item of items) {
      const obj = item as any;
      const key = `${obj.type}-${obj.description}`;
      if (seen.has(key)) {
        duplicates.push(item);
      } else {
        seen.add(key);
      }
    }

    if (duplicates.length > 0) {
      anomalies.push({
        type: "duplicates",
        count: duplicates.length,
        description: "Potential duplicate items",
      });
    }

    return anomalies;
  }

  /**
   * Assess case strength
   */
  private assessCaseStrength(caseData: CaseData, options: any = {}): StrengthAssessment {
    const assessment: StrengthAssessment = {
      overallStrength: { level: "unknown", confidence: 0, score: 0 },
      strengths: [],
      weaknesses: [],
      recommendations: [],
      factors: {},
    };

    let totalScore = 0;
    let maxScore = 0;

    // Evidence strength
    const evidenceScore = this.scoreEvidence(caseData.evidence || []);
    assessment.factors.evidence = evidenceScore;
    totalScore += evidenceScore.score;
    maxScore += evidenceScore.maxPossible;

    // Witness reliability
    const witnessScore = this.scoreWitnesses(caseData.witnesses || []);
    assessment.factors.witnesses = witnessScore;
    totalScore += witnessScore.score;
    maxScore += witnessScore.maxPossible;

    // Legal precedent
    const precedentScore = this.scorePrecedent(caseData.charges || []);
    assessment.factors.precedent = precedentScore;
    totalScore += precedentScore.score;
    maxScore += precedentScore.maxPossible;

    // Calculate overall strength
    const normalizedScore = maxScore > 0 ? totalScore / maxScore : 0;
    assessment.overallStrength.score = normalizedScore;
    assessment.overallStrength.confidence = this.calculateConfidence(caseData);

    if (normalizedScore >= 0.8) {
      assessment.overallStrength.level = "strong";
    } else if (normalizedScore >= 0.6) {
      assessment.overallStrength.level = "moderate";
    } else if (normalizedScore >= 0.4) {
      assessment.overallStrength.level = "weak";
    } else {
      assessment.overallStrength.level = "very_weak";
    }

    return assessment;
  }

  /**
   * Score evidence quality
   */
  private scoreEvidence(evidence: Evidence[]): ScoreResult {
    let score = 0;
    let maxPossible = evidence.length * 10;

    for (const item of evidence) {
      const type = this.classifyEvidence(item);

      switch (type) {
        case "physical":
          score += 10; // Strongest evidence
          break;
        case "digital":
          score += 8;
          break;
        case "documentary":
          score += 6;
          break;
        case "testimonial":
          score += 4;
          break;
        default:
          score += 2;
      }
    }

    return {
      score,
      maxPossible,
      normalized: maxPossible > 0 ? score / maxPossible : 0,
      details: `${evidence.length} pieces of evidence analyzed`,
    };
  }

  /**
   * Score witness reliability
   */
  private scoreWitnesses(witnesses: Witness[]): ScoreResult {
    let score = 0;
    let maxPossible = witnesses.length * 10;

    for (const witness of witnesses) {
      const reliability = witness.reliability || "unknown";

      switch (reliability.toLowerCase()) {
        case "high":
        case "reliable":
          score += 10;
          break;
        case "medium":
        case "moderate":
          score += 6;
          break;
        case "low":
        case "unreliable":
          score += 2;
          break;
        default:
          score += 5; // Unknown reliability
      }
    }

    return {
      score,
      maxPossible,
      normalized: maxPossible > 0 ? score / maxPossible : 0,
      details: `${witnesses.length} witnesses evaluated`,
    };
  }

  /**
   * Score legal precedent strength
   */
  private scorePrecedent(charges: Charge[]): ScoreResult {
    let score = 0;
    let maxPossible = charges.length * 10;

    // This would typically involve legal database lookup
    // For now, use simplified scoring
    for (const charge of charges) {
      const severity = charge.severity || "unknown";

      switch (severity.toLowerCase()) {
        case "felony":
        case "high":
          score += 10;
          break;
        case "misdemeanor":
        case "medium":
          score += 6;
          break;
        case "infraction":
        case "low":
          score += 3;
          break;
        default:
          score += 5;
      }
    }

    return {
      score,
      maxPossible,
      normalized: maxPossible > 0 ? score / maxPossible : 0,
      details: `${charges.length} charges analyzed`,
    };
  }

  /**
   * Calculate confidence in the assessment
   */
  private calculateConfidence(caseData: CaseData): number {
    let confidence = 0.5; // Base confidence

    // More evidence increases confidence
    const evidenceCount = (caseData.evidence || []).length;
    confidence += Math.min(evidenceCount * 0.1, 0.3);

    // Complete information increases confidence
    const hasBasicInfo =
      caseData.title && caseData.charges && caseData.evidence;
    if (hasBasicInfo) {
      confidence += 0.2;
    }

    return Math.min(confidence, 1.0);
  }

  /**
   * Analyze timeline
   */
  private analyzeTimeline(timeline: TimelineEvent[]): any {
    if (!timeline || timeline.length === 0) {
      return { message: "No timeline data available" };
    }

    const sortedEvents = timeline.sort(
      (a: TimelineEvent, b: TimelineEvent) =>
        new Date(a.date || a.timestamp!).getTime() - new Date(b.date || b.timestamp!).getTime()
    );

    return {
      eventCount: timeline.length,
      timespan: {
        start: sortedEvents[0].date || sortedEvents[0].timestamp,
        end:
          sortedEvents[sortedEvents.length - 1].date ||
          sortedEvents[sortedEvents.length - 1].timestamp,
      },
      keyEvents: sortedEvents.slice(0, 5), // Top 5 events
      gaps: this.findTimelineGaps(sortedEvents),
    };
  }

  /**
   * Find gaps in timeline
   */
  private findTimelineGaps(events: TimelineEvent[]): any[] {
    const gaps: any[] = [];

    for (let i = 0; i < events.length - 1; i++) {
      const current = new Date(events[i].date || events[i].timestamp!);
      const next = new Date(events[i + 1].date || events[i + 1].timestamp!);

      const daysDiff = (next.getTime() - current.getTime()) / (1000 * 60 * 60 * 24);

      if (daysDiff > 7) {
        // Gap of more than a week
        gaps.push({
          start: events[i].date || events[i].timestamp,
          end: events[i + 1].date || events[i + 1].timestamp,
          duration: `${Math.floor(daysDiff)} days`,
        });
      }
    }

    return gaps;
  }

  /**
   * Identify risk factors
   */
  private identifyRiskFactors(caseData: CaseData): RiskFactor[] {
    const risks: RiskFactor[] = [];

    // Evidence risks
    const evidenceCount = (caseData.evidence || []).length;
    if (evidenceCount < 3) {
      risks.push({
        type: "evidence",
        level: "high",
        description: "Limited evidence may weaken case",
      });
    }

    // Timeline risks
    if (caseData.filingDate) {
      const daysSinceFiling =
        (new Date().getTime() - new Date(caseData.filingDate).getTime()) / (1000 * 60 * 60 * 24);
      if (daysSinceFiling > 180) {
        risks.push({
          type: "timeline",
          level: "medium",
          description:
            "Extended case duration may affect witness memory and evidence integrity",
        });
      }
    }

    // Witness risks
    const witnessCount = (caseData.witnesses || []).length;
    if (witnessCount === 1) {
      risks.push({
        type: "witness",
        level: "medium",
        description: "Single witness testimony creates dependency risk",
      });
    }

    return risks;
  }

  /**
   * Perform AI analysis using LLM
   */
  private async performAIAnalysis(caseData: CaseData, options: any = {}): Promise<any> {
    try {
      const prompt = this.buildAnalysisPrompt(caseData);

      const response = await fetch(`${this.config.ollamaUrl}/api/generate`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model: this.config.analysisModel,
          prompt,
          stream: false,
          options: {
            temperature: 0.3,
            max_tokens: 2048,
          },
        }),
        // Remove timeout property as it's not valid for fetch
      });

      if (!response.ok) {
        throw new Error(`AI analysis failed: ${response.status}`);
      }

      const data = await response.json();

      return {
        analysis: (data as any).response,
        model: this.config.analysisModel,
        prompt_length: prompt.length,
        response_length: ((data as any).response?.length || 0),
        timestamp: new Date().toISOString(),
      };
    } catch (error: any) {
      console.error("AI analysis failed:", error);
      return {
        error: error instanceof Error ? error.message : String(error),
        fallback: "AI analysis unavailable - using rule-based analysis only",
      };
    }
  }

  /**
   * Build analysis prompt for LLM
   */
  private buildAnalysisPrompt(caseData: CaseData): string {
    const evidenceSummary = (caseData.evidence || [])
      .map((e: Evidence) => `- ${e.type}: ${e.description}`)
      .join("\n");

    const chargesSummary = (caseData.charges || [])
      .map((c: Charge) => `- ${c.statute}: ${c.description}`)
      .join("\n");

    return `
As a legal AI assistant for prosecutors, analyze this case:

CASE: ${caseData.title || "Untitled"}
STATUS: ${caseData.status || "Unknown"}

CHARGES:
${chargesSummary || "No charges listed"}

EVIDENCE:
${evidenceSummary || "No evidence listed"}

ANALYSIS REQUEST:
Please provide a comprehensive legal analysis including:

1. Case strength assessment
2. Key evidence evaluation
3. Potential legal challenges
4. Recommended prosecution strategy
5. Areas requiring additional investigation

Focus on practical prosecutorial considerations and provide specific, actionable insights.
    `.trim();
  }

  /**
   * Generate recommendations
   */
  private async generateRecommendations(analysisData: any, options: any = {}): Promise<any> {
    const recommendations: Recommendation[] = [];

    const data = analysisData as any;
    
    // Evidence-based recommendations
    if (data.evidence) {
      if (data.evidence.total < 5) {
        recommendations.push({
          type: "evidence",
          priority: "high",
          action: "Gather additional evidence to strengthen case",
          rationale: `Only ${data.evidence.total} pieces of evidence available`,
        });
      }

      if (data.evidence.weaknesses.length > 0) {
        recommendations.push({
          type: "evidence",
          priority: "medium",
          action: "Address evidence weaknesses",
          rationale: `${data.evidence.weaknesses.length} potential issues identified`,
        });
      }
    }

    // Strength-based recommendations
    if (data.strength) {
      if (data.strength.overallStrength.level === "weak") {
        recommendations.push({
          type: "strategy",
          priority: "high",
          action: "Consider plea negotiation or case dismissal",
          rationale:
            "Case strength assessment indicates low probability of conviction",
        });
      }
    }

    this.stats.recommendations++;

    return {
      recommendations,
      total: recommendations.length,
      highPriority: recommendations.filter((r: Recommendation) => r.priority === "high").length,
      generated: new Date().toISOString(),
    };
  }

  /**
   * Perform legal research
   */
  private async performLegalResearch(query: string, options: any = {}): Promise<any> {
    // This would typically integrate with legal databases
    // For now, provide structured research framework

    return {
      query,
      researchAreas: [
        "Case law precedents",
        "Statutory requirements",
        "Procedural considerations",
        "Evidence admissibility rules",
      ],
      suggestedSources: [
        "Federal court decisions",
        "State court decisions",
        "Legal encyclopedias",
        "Practice guides",
      ],
      nextSteps: [
        "Conduct comprehensive case law search",
        "Review relevant statutes",
        "Consult practice guides",
        "Analyze fact patterns",
      ],
      timestamp: new Date().toISOString(),
    };
  }

  /**
   * Analyze evidence (for external API calls)
   */
  private async analyzeEvidence(evidence: Evidence[], options: any = {}): Promise<any> {
    return this.analyzeEvidenceSet(evidence);
  }

  /**
   * Get worker statistics
   */
  getStats(): any {
    return {
      ...this.stats,
      cacheSize: this.analysisCache.size,
      workerId: this.workerId,
      config: {
        model: this.config.analysisModel,
        maxLength: this.config.maxAnalysisLength,
        confidenceThreshold: this.config.confidenceThreshold,
      },
    };
  }
}

// Initialize worker
const worker = new AnalysisWorker();

// Handle messages from main thread
self.addEventListener("message", (message: TaskMessage) => {
  worker.handleMessage(message);
});

// Send ready signal
self.postMessage({
  type: "ready",
  workerId: worker.workerId,
  patterns: Object.keys(worker.legalPatterns),
});