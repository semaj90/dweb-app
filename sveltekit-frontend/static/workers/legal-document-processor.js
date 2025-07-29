// Legal Document Processing Web Worker
// This worker handles heavy document analysis without blocking the UI

const LEGAL_ENTITY_PATTERNS = {
  CASE_NUMBER:
    /(?:Case\s+(?:No\.?|Number)\s*:?\s*)?([A-Z]{1,4}[-\s]?\d{2,4}[-\s]?\d{2,6})/gi,
  COURT_NAME:
    /(?:in\s+the\s+)?((?:United States|U\.S\.)\s+(?:District|Supreme|Appeals?)\s+Court|(?:State|Superior|Municipal|Family)\s+Court)/gi,
  JUDGE_NAME: /(?:(?:Judge|Justice|Hon\.)\s+)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)/g,
  ATTORNEY: /(?:Attorney|Counsel|Esq\.)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)/g,
  LEGAL_CITATION: /\d+\s+[A-Z][a-z\.]+\s+\d+(?:\s*\(\d{4}\))?/g,
  DATE: /\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b/g,
  DOLLAR_AMOUNT: /\$[\d,]+(?:\.\d{2})?/g,
  STATUTE_REFERENCE: /\b\d+\s+U\.S\.C\.?\s+§?\s*\d+(?:\([a-z]\))?/gi,
};

const COMPLIANCE_RULES = [
  {
    id: "CONFIDENTIALITY_MARKING",
    pattern: /\b(?:CONFIDENTIAL|ATTORNEY-CLIENT PRIVILEGE|WORK PRODUCT)\b/i,
    description: "Document contains proper confidentiality markings",
    required: true,
  },
  {
    id: "DATE_REQUIREMENT",
    pattern:
      /\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b/,
    description: "Document contains required date information",
    required: true,
  },
  {
    id: "SIGNATURE_BLOCK",
    pattern: /(?:Respectfully submitted|Signed|Signature|Attorney for)/i,
    description: "Document contains proper signature block",
    required: true,
  },
  {
    id: "COURT_HEADER",
    pattern: /(?:IN THE|BEFORE THE).+(?:COURT|COMMISSION)/i,
    description: "Document contains proper court header",
    required: false,
  },
];

const RISK_KEYWORDS = {
  HIGH: ["criminal", "felony", "fraud", "malpractice", "sanctions", "contempt"],
  MEDIUM: ["breach", "violation", "damages", "liability", "negligence"],
  LOW: ["consultation", "review", "advisory", "guidance"],
};

class LegalDocumentProcessor {
  extractEntities(content) {
    const entities = [];

    for (const [entityType, pattern] of Object.entries(LEGAL_ENTITY_PATTERNS)) {
      let match;
      const regex = new RegExp(pattern.source, pattern.flags);

      while ((match = regex.exec(content)) !== null) {
        entities.push({
          type: entityType,
          value: match[0].trim(),
          confidence: this.calculateEntityConfidence(entityType, match[0]),
          position: {
            start: match.index,
            end: match.index + match[0].length,
          },
        });
      }
    }

    return entities.sort((a, b) => a.position.start - b.position.start);
  }

  performComplianceCheck(content) {
    return COMPLIANCE_RULES.map((rule) => {
      const match = rule.pattern.test(content);

      return {
        rule: rule.id,
        passed: match,
        confidence: match ? 0.9 : 0.1,
        description: rule.description,
      };
    });
  }

  assessRiskFactors(content) {
    const riskFactors = [];
    const lowerContent = content.toLowerCase();

    for (const keyword of RISK_KEYWORDS.HIGH) {
      if (lowerContent.includes(keyword)) {
        riskFactors.push({
          type: "HIGH_RISK_CONTENT",
          severity: "HIGH",
          description: `Document contains high-risk keyword: "${keyword}"`,
        });
      }
    }

    for (const keyword of RISK_KEYWORDS.MEDIUM) {
      if (lowerContent.includes(keyword)) {
        riskFactors.push({
          type: "MEDIUM_RISK_CONTENT",
          severity: "MEDIUM",
          description: `Document contains medium-risk keyword: "${keyword}"`,
        });
      }
    }

    if (!LEGAL_ENTITY_PATTERNS.DATE.test(content)) {
      riskFactors.push({
        type: "MISSING_DATE",
        severity: "MEDIUM",
        description: "Document appears to be missing date information",
      });
    }

    if (
      !LEGAL_ENTITY_PATTERNS.CASE_NUMBER.test(content) &&
      content.length > 500
    ) {
      riskFactors.push({
        type: "MISSING_CASE_NUMBER",
        severity: "LOW",
        description: "Document may be missing case number reference",
      });
    }

    return riskFactors;
  }

  calculateEntityConfidence(entityType, value) {
    switch (entityType) {
      case "CASE_NUMBER":
        return value.match(/[A-Z]{2,4}[-\s]?\d{4}[-\s]?\d{4,6}/) ? 0.95 : 0.7;
      case "LEGAL_CITATION":
        return value.includes("(") ? 0.9 : 0.75;
      case "DOLLAR_AMOUNT":
        return 0.95;
      case "DATE":
        return 0.9;
      default:
        return 0.8;
    }
  }

  extractKeyPhrases(content) {
    const sentences = content.split(/[.!?]+/);
    const keyPhrases = [];

    for (const sentence of sentences) {
      const trimmed = sentence.trim();
      if (trimmed.length > 20 && trimmed.length < 200) {
        if (
          /\b(?:hereby|whereas|therefore|notwithstanding|pursuant to)\b/i.test(
            trimmed,
          )
        ) {
          keyPhrases.push(trimmed);
        }
      }
    }

    return keyPhrases.slice(0, 5);
  }

  calculateSentimentScore(content) {
    const positiveWords = [
      "agree",
      "consent",
      "approve",
      "accept",
      "comply",
      "cooperate",
    ];
    const negativeWords = [
      "deny",
      "reject",
      "dispute",
      "violation",
      "breach",
      "fail",
      "refuse",
    ];

    const words = content.toLowerCase().split(/\W+/);
    let positiveCount = 0;
    let negativeCount = 0;

    for (const word of words) {
      if (positiveWords.includes(word)) positiveCount++;
      if (negativeWords.includes(word)) negativeCount++;
    }

    const total = positiveCount + negativeCount;
    if (total === 0) return 0;

    return (positiveCount - negativeCount) / total;
  }

  async processDocument(document, analysisType, options = {}) {
    const startTime = performance.now();

    try {
      const results = {};

      if (options.extractEntities || analysisType === "FULL_COMPLIANCE_CHECK") {
        results.entities = this.extractEntities(document.content);
      }

      if (
        options.performCompliance ||
        analysisType === "FULL_COMPLIANCE_CHECK"
      ) {
        results.complianceChecks = this.performComplianceCheck(
          document.content,
        );
      }

      if (
        options.includeRiskAssessment ||
        analysisType === "FULL_COMPLIANCE_CHECK"
      ) {
        results.riskFactors = this.assessRiskFactors(document.content);
      }

      results.keyPhrases = this.extractKeyPhrases(document.content);
      results.sentimentScore = this.calculateSentimentScore(document.content);
      results.summary = this.generateSummary(results, document);

      const processingTime = performance.now() - startTime;

      return {
        documentId: document.id,
        analysisType,
        results,
        processingTime: Math.round(processingTime),
      };
    } catch (error) {
      return {
        documentId: document.id,
        analysisType,
        results: {},
        processingTime: performance.now() - startTime,
        error: error.message,
      };
    }
  }

  generateSummary(results, document) {
    const summaryParts = [];

    summaryParts.push(
      `Analyzed ${document.type || "legal document"} (${Math.round(document.content.length / 1000)}k characters)`,
    );

    if (results.entities) {
      const entityTypes = [...new Set(results.entities.map((e) => e.type))];
      summaryParts.push(
        `Found ${results.entities.length} legal entities of ${entityTypes.length} types`,
      );
    }

    if (results.complianceChecks) {
      const passedChecks = results.complianceChecks.filter(
        (c) => c.passed,
      ).length;
      const totalChecks = results.complianceChecks.length;
      summaryParts.push(
        `Compliance: ${passedChecks}/${totalChecks} checks passed`,
      );
    }

    if (results.riskFactors) {
      const highRisk = results.riskFactors.filter(
        (r) => r.severity === "HIGH",
      ).length;
      const mediumRisk = results.riskFactors.filter(
        (r) => r.severity === "MEDIUM",
      ).length;

      if (highRisk > 0) {
        summaryParts.push(`⚠️ ${highRisk} high-risk factors identified`);
      } else if (mediumRisk > 0) {
        summaryParts.push(`⚠️ ${mediumRisk} medium-risk factors identified`);
      } else {
        summaryParts.push(`✅ No significant risk factors identified`);
      }
    }

    return summaryParts.join(". ");
  }
}

const processor = new LegalDocumentProcessor();

self.onmessage = async function (e) {
  const { type, document, analysisType, options } = e.data;

  try {
    switch (type) {
      case "ANALYZE_LEGAL_DOCUMENT":
        const result = await processor.processDocument(
          document,
          analysisType,
          options,
        );
        self.postMessage({
          type: "ANALYSIS_COMPLETE",
          result,
        });
        break;

      case "EXTRACT_ENTITIES":
        const entities = processor.extractEntities(document.content);
        self.postMessage({
          type: "ENTITIES_EXTRACTED",
          result: {
            documentId: document.id,
            entities,
            processingTime: 0,
          },
        });
        break;

      case "COMPLIANCE_CHECK":
        const complianceChecks = processor.performComplianceCheck(
          document.content,
        );
        self.postMessage({
          type: "COMPLIANCE_COMPLETE",
          result: {
            documentId: document.id,
            complianceChecks,
            processingTime: 0,
          },
        });
        break;

      default:
        self.postMessage({
          type: "ERROR",
          error: `Unknown message type: ${type}`,
        });
    }
  } catch (error) {
    self.postMessage({
      type: "ERROR",
      error: error.message,
      documentId: document?.id,
    });
  }
};

self.postMessage({
  type: "WORKER_READY",
  capabilities: [
    "ANALYZE_LEGAL_DOCUMENT",
    "EXTRACT_ENTITIES",
    "COMPLIANCE_CHECK",
  ],
});
