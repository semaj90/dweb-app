import { parentPort, workerData } from "worker_threads";
import fetch from "node-fetch";

/**
 * Phase 4: Analysis Worker
 * Handles legal case analysis, pattern recognition, and AI-powered insights
 */

class AnalysisWorker {
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
  initializeLegalPatterns() {
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
  handleMessage(message) {
    const { taskId, data, options } = message;

    try {
      let result;

      switch (data.type) {
        case "analyze_case":
          result = this.analyzeCase(data.case, options);
          break;
        case "analyze_evidence":
          result = this.analyzeEvidence(data.evidence, options);
          break;
        case "pattern_analysis":
          result = this.performPatternAnalysis(data.items, options);
          break;
        case "strength_assessment":
          result = this.assessCaseStrength(data.caseData, options);
          break;
        case "generate_recommendations":
          result = this.generateRecommendations(data.analysisData, options);
          break;
        case "legal_research":
          result = this.performLegalResearch(data.query, options);
          break;
        default:
          throw new Error(`Unknown analysis task type: ${data.type}`);
      }

      // Handle async results
      if (result instanceof Promise) {
        result
          .then((asyncResult) => {
            parentPort.postMessage({
              taskId,
              success: true,
              data: asyncResult,
            });
          })
          .catch((error) => {
            this.stats.errors++;
            parentPort.postMessage({
              taskId,
              success: false,
              error: error.message,
            });
          });
      } else {
        parentPort.postMessage({
          taskId,
          success: true,
          data: result,
        });
      }
    } catch (error) {
      console.error(`❌ Analysis error in ${this.workerId}:`, error);
      this.stats.errors++;
      parentPort.postMessage({
        taskId,
        success: false,
        error: error.message,
      });
    }
  }

  /**
   * Analyze a complete case
   */
  async analyzeCase(caseData, options = {}) {
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

      const analysis = {
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
    } catch (error) {
      throw new Error(`Case analysis failed: ${error.message}`);
    }
  }

  /**
   * Generate case summary
   */
  generateCaseSummary(caseData, strengthAssessment) {
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
  extractKeyPoints(caseData) {
    const points = [];

    // Evidence-based points
    if (caseData.evidence?.length > 0) {
      const physicalEvidence = caseData.evidence.filter((e) =>
        this.legalPatterns.evidenceTypes.physical.some((type) =>
          e.type?.toLowerCase().includes(type)
        )
      );

      if (physicalEvidence.length > 0) {
        points.push(`${physicalEvidence.length} pieces of physical evidence`);
      }
    }

    // Charge-based points
    if (caseData.charges?.length > 0) {
      const chargeTypes = new Set();
      caseData.charges.forEach((charge) => {
        for (const [category, patterns] of Object.entries(
          this.legalPatterns.chargePatterns
        )) {
          if (
            patterns.some((pattern) =>
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
  identifyUrgentActions(caseData, strengthAssessment) {
    const actions = [];

    // Based on strength assessment
    if (strengthAssessment.overallStrength.confidence < 0.5) {
      actions.push("Gather additional evidence to strengthen case");
    }

    // Based on evidence issues
    if (
      strengthAssessment.weaknesses.some((w) => w.includes("chain of custody"))
    ) {
      actions.push("Review and document evidence chain of custody");
    }

    // Based on timeline
    const today = new Date();
    if (caseData.filingDate) {
      const filingDate = new Date(caseData.filingDate);
      const daysSinceFiling = (today - filingDate) / (1000 * 60 * 60 * 24);

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
  async analyzeEvidenceSet(evidence) {
    const analysis = {
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
  classifyEvidence(evidence) {
    const description = (
      evidence.description ||
      evidence.type ||
      ""
    ).toLowerCase();

    for (const [category, keywords] of Object.entries(
      this.legalPatterns.evidenceTypes
    )) {
      if (keywords.some((keyword) => description.includes(keyword))) {
        return category;
      }
    }

    return "other";
  }

  /**
   * Assess individual evidence strength
   */
  assessEvidenceStrength(evidence, analysis) {
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
  performPatternAnalysis(items, options = {}) {
    const patterns = {
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
  findTemporalPatterns(items) {
    const datedItems = items.filter((item) => item.date || item.timestamp);

    if (datedItems.length < 2) {
      return { message: "Insufficient temporal data" };
    }

    // Sort by date
    datedItems.sort(
      (a, b) =>
        new Date(a.date || a.timestamp) - new Date(b.date || b.timestamp)
    );

    return {
      timeline: datedItems.map((item) => ({
        date: item.date || item.timestamp,
        type: item.type,
        description: item.description,
      })),
      timespan: {
        start: datedItems[0].date || datedItems[0].timestamp,
        end:
          datedItems[datedItems.length - 1].date ||
          datedItems[datedItems.length - 1].timestamp,
      },
    };
  }

  /**
   * Find categorical patterns
   */
  findCategoricalPatterns(items) {
    const categories = {};

    for (const item of items) {
      const category = item.type || item.category || "uncategorized";
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
  findRelationalPatterns(items) {
    const relationships = [];

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
      types: [...new Set(relationships.map((r) => r.type))],
    };
  }

  /**
   * Find relationship between two items
   */
  findRelationship(item1, item2) {
    // Same location
    if (item1.location && item2.location && item1.location === item2.location) {
      return {
        type: "location",
        items: [item1.id, item2.id],
        description: `Both items associated with ${item1.location}`,
      };
    }

    // Same person
    if (item1.person && item2.person && item1.person === item2.person) {
      return {
        type: "person",
        items: [item1.id, item2.id],
        description: `Both items associated with ${item1.person}`,
      };
    }

    // Same date
    if (item1.date && item2.date && item1.date === item2.date) {
      return {
        type: "temporal",
        items: [item1.id, item2.id],
        description: `Both items occurred on ${item1.date}`,
      };
    }

    return null;
  }

  /**
   * Find anomalies in the data
   */
  findAnomalies(items) {
    const anomalies = [];

    // Check for missing critical information
    const missingInfo = items.filter(
      (item) => !item.date || !item.type || !item.description
    );
    if (missingInfo.length > 0) {
      anomalies.push({
        type: "missing_information",
        count: missingInfo.length,
        description: "Items with missing critical information",
      });
    }

    // Check for duplicate items
    const seen = new Set();
    const duplicates = [];
    for (const item of items) {
      const key = `${item.type}-${item.description}`;
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
  assessCaseStrength(caseData, options = {}) {
    const assessment = {
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
  scoreEvidence(evidence) {
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
  scoreWitnesses(witnesses) {
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
  scorePrecedent(charges) {
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
  calculateConfidence(caseData) {
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
  analyzeTimeline(timeline) {
    if (!timeline || timeline.length === 0) {
      return { message: "No timeline data available" };
    }

    const sortedEvents = timeline.sort(
      (a, b) =>
        new Date(a.date || a.timestamp) - new Date(b.date || b.timestamp)
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
  findTimelineGaps(events) {
    const gaps = [];

    for (let i = 0; i < events.length - 1; i++) {
      const current = new Date(events[i].date || events[i].timestamp);
      const next = new Date(events[i + 1].date || events[i + 1].timestamp);

      const daysDiff = (next - current) / (1000 * 60 * 60 * 24);

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
  identifyRiskFactors(caseData) {
    const risks = [];

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
        (new Date() - new Date(caseData.filingDate)) / (1000 * 60 * 60 * 24);
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
  async performAIAnalysis(caseData, options = {}) {
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
        timeout: this.config.timeout,
      });

      if (!response.ok) {
        throw new Error(`AI analysis failed: ${response.status}`);
      }

      const data = await response.json();

      return {
        analysis: data.response,
        model: this.config.analysisModel,
        prompt_length: prompt.length,
        response_length: data.response?.length || 0,
        timestamp: new Date().toISOString(),
      };
    } catch (error) {
      console.error("AI analysis failed:", error);
      return {
        error: error.message,
        fallback: "AI analysis unavailable - using rule-based analysis only",
      };
    }
  }

  /**
   * Build analysis prompt for LLM
   */
  buildAnalysisPrompt(caseData) {
    const evidenceSummary = (caseData.evidence || [])
      .map((e) => `- ${e.type}: ${e.description}`)
      .join("\n");

    const chargesSummary = (caseData.charges || [])
      .map((c) => `- ${c.statute}: ${c.description}`)
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
  async generateRecommendations(analysisData, options = {}) {
    const recommendations = [];

    // Evidence-based recommendations
    if (analysisData.evidence) {
      if (analysisData.evidence.total < 5) {
        recommendations.push({
          type: "evidence",
          priority: "high",
          action: "Gather additional evidence to strengthen case",
          rationale: `Only ${analysisData.evidence.total} pieces of evidence available`,
        });
      }

      if (analysisData.evidence.weaknesses.length > 0) {
        recommendations.push({
          type: "evidence",
          priority: "medium",
          action: "Address evidence weaknesses",
          rationale: `${analysisData.evidence.weaknesses.length} potential issues identified`,
        });
      }
    }

    // Strength-based recommendations
    if (analysisData.strength) {
      if (analysisData.strength.overallStrength.level === "weak") {
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
      highPriority: recommendations.filter((r) => r.priority === "high").length,
      generated: new Date().toISOString(),
    };
  }

  /**
   * Perform legal research
   */
  async performLegalResearch(query, options = {}) {
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
   * Get worker statistics
   */
  getStats() {
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
parentPort.on("message", (message) => {
  worker.handleMessage(message);
});

// Send ready signal
parentPort.postMessage({
  type: "ready",
  workerId: worker.workerId,
  patterns: Object.keys(worker.legalPatterns),
});
