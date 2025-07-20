#!/usr/bin/env node

/**
 * Document Processing Script for Legal AI
 * Processes documents with AI tagging and classification
 */

import fs from "fs";
import path from "path";
import { JSDOM } from "jsdom";

console.log("📄 Processing documents for Legal AI...");
console.log("======================================");

const rawDir = "./docs/raw";
const outDir = "./docs/processed";

// Ensure output directory exists
if (!fs.existsSync(outDir)) {
  fs.mkdirSync(outDir, { recursive: true });
}

// Enhanced tag mapping for legal documents
const enhancedTagMap = {
  javascript: ["vanilla-js", "frontend", "programming", "web-dev"],
  webassembly: ["wasm", "performance", "low-level", "optimization"],
  bits: ["ui", "accessibility", "svelte", "components", "design-system"],
  melt: ["ui", "accessibility", "headless", "svelte", "components"],
  tailwind: ["css", "styling", "design-system", "responsive"],
  drizzle: ["orm", "database", "typescript", "sql", "backend"],
  postgresql: ["database", "sql", "backend", "persistence"],
  llama: ["llm", "ai", "inference", "cpp", "machine-learning"],

  // Legal-specific tags
  criminal: ["criminal-law", "procedure", "prosecution", "defense"],
  civil: ["civil-law", "litigation", "contracts", "torts"],
  evidence: ["evidence-law", "procedure", "admissibility", "chain-of-custody"],
  constitutional: [
    "constitutional-law",
    "rights",
    "amendments",
    "supreme-court",
  ],
  procedure: ["legal-procedure", "rules", "motions", "hearings"],
  statute: ["statutes", "legislation", "codes", "regulations"],
  case: ["case-law", "precedent", "opinions", "rulings"],
  witness: ["witness-law", "testimony", "examination", "credibility"],
  court: ["court-procedure", "judicial", "hearings", "orders"],
  discovery: ["discovery-law", "depositions", "interrogatories", "requests"],
};

// Legal document classification patterns
const legalClassification = {
  "criminal-law": [
    "crime",
    "criminal",
    "defendant",
    "prosecution",
    "jury",
    "felony",
    "misdemeanor",
  ],
  "civil-law": [
    "plaintiff",
    "civil",
    "damages",
    "contract",
    "tort",
    "liability",
  ],
  "evidence-law": [
    "evidence",
    "witness",
    "testimony",
    "admissible",
    "hearsay",
    "authentication",
  ],
  "constitutional-law": [
    "constitutional",
    "amendment",
    "rights",
    "due process",
    "equal protection",
  ],
  procedure: ["procedure", "rule", "motion", "hearing", "discovery", "trial"],
  "contract-law": [
    "contract",
    "agreement",
    "breach",
    "consideration",
    "performance",
  ],
  "property-law": ["property", "real estate", "ownership", "deed", "title"],
  "family-law": ["family", "divorce", "custody", "adoption", "marriage"],
  "corporate-law": [
    "corporate",
    "business",
    "corporation",
    "partnership",
    "securities",
  ],
  "intellectual-property": [
    "patent",
    "trademark",
    "copyright",
    "trade secret",
    "ip",
  ],
};

// Legal complexity indicators
const complexityIndicators = {
  high: [
    "constitutional",
    "appellate",
    "supreme court",
    "class action",
    "federal",
  ],
  medium: ["motion", "discovery", "deposition", "expert witness", "jury trial"],
  low: ["summary", "simple", "uncontested", "stipulated", "agreed"],
};

console.log("🔄 Processing documents with enhanced AI tagging...");

// Check if raw directory exists
if (!fs.existsSync(rawDir)) {
  console.log("⚠️  Raw documents directory not found:", rawDir);
  console.log(
    "💡 Run fetch-docs script first or create the directory manually",
  );
  process.exit(0);
}

const files = fs.readdirSync(rawDir);
if (files.length === 0) {
  console.log("⚠️  No documents found in:", rawDir);
  console.log("💡 Run fetch-docs script first to download documents");
  process.exit(0);
}

let processedCount = 0;
let errorCount = 0;

files.forEach((file) => {
  try {
    const filePath = path.join(rawDir, file);
    const stats = fs.statSync(filePath);

    if (!stats.isFile()) {
      return; // Skip directories
    }

    console.log(`📄 Processing: ${file}`);

    const fileContent = fs.readFileSync(filePath, "utf-8");
    let text = "";
    let title = file;

    // Handle different file types
    if (file.endsWith(".html")) {
      try {
        const dom = new JSDOM(fileContent);
        text = dom.window.document.body?.textContent?.trim() || "";
        title = dom.window.document.querySelector("title")?.textContent || file;
      } catch (domError) {
        console.log(`⚠️  HTML parsing failed for ${file}, using raw text`);
        text = fileContent;
      }
    } else {
      // For non-HTML files, use content as-is
      text = fileContent;
    }

    // Enhanced tagging system
    let tags = [];
    let legalType = "general";
    let practiceAreas = [];

    // Technical tags
    for (const [key, keyTags] of Object.entries(enhancedTagMap)) {
      if (file.toLowerCase().includes(key)) {
        tags = tags.concat(keyTags);
      }
    }

    // Legal classification
    for (const [type, keywords] of Object.entries(legalClassification)) {
      if (keywords.some((keyword) => text.toLowerCase().includes(keyword))) {
        legalType = type;
        practiceAreas.push(type);
        tags.push(`legal-${type}`);
      }
    }

    // Content analysis for additional tags
    const lowerText = text.toLowerCase();

    // Chain of custody indicators
    if (
      lowerText.includes("chain of custody") ||
      lowerText.includes("evidence handling")
    ) {
      tags.push("evidence-handling", "chain-of-custody");
    }

    // Constitutional law indicators
    if (
      lowerText.includes("constitutional") ||
      lowerText.includes("amendment")
    ) {
      tags.push("constitutional-law");
    }

    // Search and seizure
    if (
      lowerText.includes("search warrant") ||
      lowerText.includes("fourth amendment")
    ) {
      tags.push("fourth-amendment", "search-seizure");
    }

    // Miranda rights
    if (
      lowerText.includes("miranda") ||
      lowerText.includes("right to remain silent")
    ) {
      tags.push("fifth-amendment", "miranda-rights");
    }

    // Due process
    if (
      lowerText.includes("due process") ||
      lowerText.includes("procedural rights")
    ) {
      tags.push("due-process", "procedural-rights");
    }

    // Remove duplicates
    tags = Array.from(new Set(tags));

    // Assess document complexity
    const complexity = assessComplexity(text);

    // Extract key legal terms
    const keyTerms = extractKeyLegalTerms(text);

    // Detect legal citations
    const citations = extractLegalCitations(text);

    // Enhanced output structure for AI processing
    const output = {
      id: file.replace(/\.(html|txt|md)$/, ""),
      file,
      title,
      text,
      tags,
      legalType,
      practiceAreas,
      wordCount: text.split(/\s+/).length,
      characterCount: text.length,
      extractedAt: new Date().toISOString(),
      summary: generateSummary(text),

      // Enhanced AI context for thinking style
      thinkingContext: {
        documentType: legalType,
        keyTerms,
        complexity,
        legalCitations: citations,
        practiceAreas,

        // Analysis hints for AI
        analysisHints: {
          requiresDetailedAnalysis:
            complexity === "high" ||
            practiceAreas.includes("constitutional-law"),
          evidenceRelated:
            tags.includes("evidence-handling") ||
            tags.includes("chain-of-custody"),
          proceduralCompliance:
            tags.includes("procedure") || tags.includes("due-process"),
          constitutionalIssues:
            tags.includes("constitutional-law") ||
            tags.includes("fourth-amendment"),
        },

        // Suggested analysis types
        suggestedAnalysis: getSuggestedAnalysisTypes(tags, legalType),

        // Quality indicators
        qualityMetrics: {
          textLength: text.length,
          hasStructure: text.includes("\n") && text.split("\n").length > 5,
          hasCitations: citations.length > 0,
          hasLegalTerms: keyTerms.length > 3,
        },
      },
    };

    const outputPath = path.join(
      outDir,
      file.replace(/\.(html|txt|md)$/, ".json"),
    );
    fs.writeFileSync(outputPath, JSON.stringify(output, null, 2));

    processedCount++;
    console.log(
      `✅ Processed: ${file} (${tags.length} tags, ${practiceAreas.length} practice areas)`,
    );
  } catch (error) {
    console.error(`❌ Error processing ${file}:`, error.message);
    errorCount++;
  }
});

// Function implementations
function assessComplexity(text) {
  const sentences = text.split(/[.!?]+/).filter((s) => s.trim().length > 0);
  const avgWordsPerSentence = text.split(/\s+/).length / sentences.length;

  // Check for complexity indicators
  const hasHighComplexity = complexityIndicators.high.some((indicator) =>
    text.toLowerCase().includes(indicator),
  );

  const hasMediumComplexity = complexityIndicators.medium.some((indicator) =>
    text.toLowerCase().includes(indicator),
  );

  if (hasHighComplexity || avgWordsPerSentence > 25) return "high";
  if (hasMediumComplexity || avgWordsPerSentence > 15) return "medium";
  return "low";
}

function extractKeyLegalTerms(text) {
  const legalTerms = [
    // Evidence and procedure
    "evidence",
    "witness",
    "testimony",
    "hearsay",
    "admissible",
    "chain of custody",
    "authentication",
    "foundation",
    "relevance",
    "prejudicial",

    // Criminal law
    "defendant",
    "prosecution",
    "beyond reasonable doubt",
    "plea",
    "sentence",
    "miranda rights",
    "search warrant",
    "probable cause",
    "reasonable suspicion",

    // Civil law
    "plaintiff",
    "damages",
    "liability",
    "negligence",
    "breach of contract",
    "injunction",
    "summary judgment",
    "discovery",
    "deposition",

    // Constitutional law
    "due process",
    "equal protection",
    "first amendment",
    "fourth amendment",
    "fifth amendment",
    "constitutional",
    "supreme court",
    "judicial review",

    // Procedure
    "motion",
    "objection",
    "sustained",
    "overruled",
    "voir dire",
    "cross-examination",
    "direct examination",
    "opening statement",
    "closing argument",
  ];

  const foundTerms = legalTerms.filter((term) =>
    text.toLowerCase().includes(term.toLowerCase()),
  );

  return [...new Set(foundTerms)]; // Remove duplicates
}

function extractLegalCitations(text) {
  // Simple legal citation detection patterns
  const citationPatterns = [
    /\d+\s+[A-Z][a-zA-Z\s\.]+\d+/g, // Basic case citation
    /\d+\s+U\.?S\.?\s+\d+/g, // US Supreme Court
    /\d+\s+F\.\d*d?\s+\d+/g, // Federal courts
    /\d+\s+S\.?Ct\.?\s+\d+/g, // Supreme Court Reporter
    /\d+\s+[A-Z]+\s+\d+/g, // State citations
  ];

  const citations = [];
  citationPatterns.forEach((pattern) => {
    const matches = text.match(pattern) || [];
    citations.push(...matches);
  });

  return [...new Set(citations)].slice(0, 10); // Limit to 10 unique citations
}

function generateSummary(text) {
  const sentences = text.split(/[.!?]+/).filter((s) => s.trim().length > 10);
  if (sentences.length === 0) return "No content summary available.";

  // Take first sentence and a middle sentence for summary
  const firstSentence = sentences[0].trim() + ".";
  const middleSentence =
    sentences.length > 3
      ? sentences[Math.floor(sentences.length / 2)].trim() + "."
      : "";

  const summary = middleSentence
    ? `${firstSentence} ${middleSentence}`
    : firstSentence;

  return summary.length > 300 ? summary.substring(0, 297) + "..." : summary;
}

function getSuggestedAnalysisTypes(tags, legalType) {
  const suggestions = [];

  if (tags.includes("evidence-handling")) {
    suggestions.push("chain_of_custody", "evidence_quality");
  }

  if (tags.includes("constitutional-law")) {
    suggestions.push("constitutional_analysis", "rights_assessment");
  }

  if (legalType === "criminal-law") {
    suggestions.push("criminal_analysis", "prosecution_strategy");
  }

  if (legalType === "civil-law") {
    suggestions.push("liability_assessment", "damages_analysis");
  }

  if (tags.includes("procedure")) {
    suggestions.push("procedural_compliance", "rule_analysis");
  }

  // Default suggestions
  if (suggestions.length === 0) {
    suggestions.push("classification", "content_analysis");
  }

  return [...new Set(suggestions)];
}

console.log("\n📊 Processing Summary");
console.log("====================");
console.log(`✅ Successfully processed: ${processedCount} documents`);
console.log(`❌ Errors encountered: ${errorCount} documents`);
console.log(`📁 Output directory: ${outDir}`);

if (processedCount > 0) {
  console.log("\n📁 Processed documents are ready for AI analysis");
  console.log(
    "💡 Use these in your /api/analyze endpoint for enhanced legal reasoning",
  );

  // Create index file for processed documents
  const indexData = {
    totalDocuments: processedCount,
    processedAt: new Date().toISOString(),
    outputDirectory: outDir,
    documentTypes: ["legal", "technical", "mixed"],
    aiReady: true,
  };

  fs.writeFileSync(
    path.join(outDir, "index.json"),
    JSON.stringify(indexData, null, 2),
  );
  console.log("✅ Document index created");
}

console.log("\n🎉 Document processing complete!");
