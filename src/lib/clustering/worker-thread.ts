import { parentPort, workerData } from 'node:worker_threads';
import { performance } from 'node:perf_hooks';

/**
 * Legal AI Worker Thread Implementation
 * Handles CPU-intensive tasks: document analysis, vector calculations, AI inference
 */

// Type definitions
interface WorkerTask {
  id: string;
  type: keyof typeof taskProcessors;
  payload: unknown;
}

interface WorkerResult {
  taskId: string;
  success: boolean;
  result?: unknown;
  error?: string;
  processingTime: number;
  workerId: string;
  memoryUsage: number;
}

interface DocumentAnalysisPayload {
  content: string;
  documentType: string;
  analysisType: string;
  regulations?: string[];
}

interface VectorSearchPayload {
  queryVector: number[];
  targetVectors: Array<{
    id: string;
    vector: number[];
    metadata: Record<string, any>;
  }>;
  threshold?: number;
}

interface AIInferencePayload {
  prompt: string;
  modelType: string;
  parameters: Record<string, any>;
}

interface DataProcessingPayload {
  operation: string;
  data: unknown;
}

interface LegalEntity {
  persons: string[];
  organizations: string[];
  locations: string[];
  dates: string[];
  monetary_amounts: string[];
  case_numbers: string[];
  citations: string[];
}

interface ClassificationFeatures {
  hasContractLanguage: boolean;
  hasSignatureBlocks: boolean;
  hasMotionLanguage: boolean;
  hasProcedural: boolean;
  hasEvidenceLanguage: boolean;
  hasLetterFormat: boolean;
  hasBriefLanguage: boolean;
  hasCitations: boolean;
  hasStatutes: boolean;
  hasLegalTerms: number;
}

interface DocumentClassification {
  predictedType: string;
  confidence: number;
  features: ClassificationFeatures;
  alternativeTypes: Array<{ type: string; confidence: number }>;
}

interface RiskItem {
  type: string;
  severity: 'low' | 'medium' | 'high';
  description: string;
  keywords: string[];
}

interface RiskAssessment {
  totalRisks: number;
  highRiskCount: number;
  risks: RiskItem[];
}

interface ComplianceCheck {
  overallCompliance: 'unknown' | 'compliant' | 'partial' | 'non-compliant';
  checksPassed: number;
  checksTotal: number;
  violations: string[];
  recommendations: string[];
}

interface SentimentAnalysis {
  sentiment: number;
  label: 'positive' | 'negative' | 'neutral';
  confidence: number;
  positiveCount: number;
  negativeCount: number;
}

interface TextClassification {
  class: string;
  confidence: number;
  allScores: Record<string, number>;
}

interface Citation {
  fullCitation: string;
  plaintiff: string;
  defendant: string;
  volume: string;
  reporter: string;
  page: string;
}

interface DocumentSummary {
  summary: string;
  keyPoints: string[];
  wordCount: number;
  entities: LegalEntity;
}

interface MergedDocuments {
  mergedContent: string;
  documentCount: number;
  totalLength: number;
  mergedAt: string;
}

interface FullAnalysisResult {
  entities: LegalEntity;
  classification: DocumentClassification;
  risks: RiskAssessment;
  compliance: ComplianceCheck;
  metadata: {
    wordCount: number;
    characterCount: number;
    paragraphs: number;
    processedAt: string;
  };
}

const { workerId } = workerData;

// Task processors for different task types
const taskProcessors = {
  document_analysis: processDocumentAnalysis,
  vector_search: processVectorSearch,
  ai_inference: processAIInference,
  data_processing: processDataProcessing
} as const;

// Listen for tasks from the main thread
parentPort?.on('message', async (task: WorkerTask) => {
  const startTime = performance.now();
  const startMemory = process.memoryUsage();

  try {
    const processor = taskProcessors[task.type];
    if (!processor) {
      throw new Error(`Unknown task type: ${task.type}`);
    }

    const result = await processor(task.payload);
    const endTime = performance.now();
    const endMemory = process.memoryUsage();

    // Send result back to main thread
    const response: WorkerResult = {
      taskId: task.id,
      success: true,
      result,
      processingTime: endTime - startTime,
      workerId,
      memoryUsage: endMemory.heapUsed
    };

    parentPort?.postMessage(response);

  } catch (error) {
    const endTime = performance.now();
    const endMemory = process.memoryUsage();

    const response: WorkerResult = {
      taskId: task.id,
      success: false,
      error: (error instanceof Error) ? error.message : String(error),
      processingTime: endTime - startTime,
      workerId,
      memoryUsage: endMemory.heapUsed
    };

    parentPort?.postMessage(response);
  }
});

/**
 * Document Analysis Processor
 * Performs intensive text analysis, entity extraction, and classification
 */
async function processDocumentAnalysis(payload: DocumentAnalysisPayload): Promise<any> {
  const { content, documentType, analysisType } = payload;

  switch (analysisType) {
    case 'entity_extraction':
      return extractLegalEntities(content);
    
    case 'classification':
      return classifyDocument(content, documentType);
    
    case 'risk_assessment':
      return assessDocumentRisks(content);
    
    case 'compliance_check':
      return checkCompliance(content, payload.regulations || []);
    
    default:
      return performFullAnalysis(content, documentType);
  }
}

/**
 * Vector Search Processor
 * Handles vector similarity calculations and search operations
 */
async function processVectorSearch(payload: VectorSearchPayload): Promise<Array<{id: string; similarity: number; metadata: Record<string, any>}>> {
  const { queryVector, targetVectors, threshold = 0.7 } = payload;

  const results: Array<{id: string; similarity: number; metadata: Record<string, any>}> = [];

  for (let i = 0; i < targetVectors.length; i++) {
    const similarity = cosineSimilarity(queryVector, targetVectors[i].vector);
    
    if (similarity >= threshold) {
      results.push({
        id: targetVectors[i].id,
        similarity,
        metadata: targetVectors[i].metadata
      });
    }
  }

  // Sort by similarity descending
  return results.sort((a, b) => b.similarity - a.similarity);
}

/**
 * AI Inference Processor
 * Handles local AI model inference and processing
 */
async function processAIInference(payload: AIInferencePayload): Promise<any> {
  const { prompt, modelType, parameters } = payload;

  switch (modelType) {
    case 'sentiment_analysis':
      return analyzeSentiment(prompt);
    
    case 'text_classification':
      return classifyText(prompt, parameters.classes);
    
    case 'summarization':
      return summarizeText(prompt, parameters.maxLength);
    
    case 'embedding_generation':
      return generateEmbedding(prompt);
    
    default:
      throw new Error(`Unsupported model type: ${modelType}`);
  }
}

/**
 * Data Processing Processor
 * Handles data transformation, cleaning, and preparation
 */
async function processDataProcessing(payload: DataProcessingPayload): Promise<any> {
  const { operation, data } = payload;

  switch (operation) {
    case 'clean_text':
      return cleanLegalText(data);
    
    case 'extract_citations':
      return extractCitations(data);
    
    case 'normalize_dates':
      return normalizeDates(data);
    
    case 'merge_documents':
      return mergeDocuments(data.documents);
    
    case 'generate_summary':
      return generateDocumentSummary(data);
    
    default:
      throw new Error(`Unsupported operation: ${operation}`);
  }
}

/**
 * Legal Entity Extraction Implementation
 */
function extractLegalEntities(text: string): LegalEntity {
  const entities: LegalEntity = {
    persons: [],
    organizations: [],
    locations: [],
    dates: [],
    monetary_amounts: [],
    case_numbers: [],
    citations: []
  };

  // Person names (improved pattern)
  const personPattern = /\b([A-Z][a-z]+(?:\s+[A-Z][a-z]*\.?)*\s+[A-Z][a-z]+(?:\s+(?:Jr\.|Sr\.|II|III|IV))?)\b/g;
  entities.persons = Array.from(new Set((text.match(personPattern) || []).map((match: string) => match.trim())));

  // Organizations
  const orgPattern = /\b([A-Z][a-zA-Z\s&.,-]+(?:Inc\.?|Corp\.?|Corporation|LLC\.?|Ltd\.?|LP\.?|LLP\.?|Co\.?|Company|Associates|Partners|Group|Holdings|Enterprises))\b/g;
  entities.organizations = Array.from(new Set((text.match(orgPattern) || []).map((match: string) => match.trim())));

  // Locations
  const locationPattern = /\b([A-Z][a-zA-Z\s]+ (?:City|County|State|Province|District|Court|Courthouse))\b/g;
  entities.locations = Array.from(new Set((text.match(locationPattern) || []).map((match: string) => match.trim())));

  // Dates
  const datePattern = /\b(?:\d{1,2}\/\d{1,2}\/\d{4}|\d{4}-\d{2}-\d{2}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b/g;
  entities.dates = Array.from(new Set(text.match(datePattern) || []));

  // Monetary amounts
  const moneyPattern = /\$\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?(?:\s*(?:million|billion|thousand|M|B|K))?/g;
  entities.monetary_amounts = Array.from(new Set(text.match(moneyPattern) || []));

  // Case numbers
  const casePattern = /\b(?:Case\s+No\.?|Docket\s+No\.?|Civil\s+No\.?)\s*:?\s*(\d{1,2}:\d{2}-[A-Z]{2,4}-\d{4,6}(?:-[A-Z]{1,3})?)/gi;
  entities.case_numbers = Array.from(new Set((text.match(casePattern) || []).map((match: string) => match.trim())));

  // Legal citations
  const citationPattern = /\b([A-Z][a-zA-Z\s.,'&-]+)\s+v\.?\s+([A-Z][a-zA-Z\s.,'&-]+),?\s+(\d+)\s+([A-Z][a-z.]*)\s+(\d+)/g;
  entities.citations = Array.from(new Set((text.match(citationPattern) || []).map((match: string) => match.trim())));

  return entities;
}

// Document Classification Implementation
function classifyDocument(content: string, documentType: string): DocumentClassification {
  const features = extractClassificationFeatures(content);
  const confidence = calculateClassificationConfidence(features, documentType);

  return {
    predictedType: documentType,
    confidence,
    features,
    alternativeTypes: suggestAlternativeTypes(features)
  };
}

function extractClassificationFeatures(content: string): ClassificationFeatures {
  const lowerContent = content.toLowerCase();
  
  return {
    // Contract indicators
    hasContractLanguage: /\b(agreement|contract|covenant|consideration|party|parties)\b/.test(lowerContent),
    hasSignatureBlocks: /\b(signature|signed|executed|witness)\b/.test(lowerContent),
    
    // Motion indicators
    hasMotionLanguage: /\b(motion|petition|request|court|honor|respectfully)\b/.test(lowerContent),
    hasProcedural: /\b(hearing|discovery|summary judgment|dismiss)\b/.test(lowerContent),
    
    // Evidence indicators
    hasEvidenceLanguage: /\b(exhibit|evidence|proof|document|record|attachment)\b/.test(lowerContent),
    
    // Correspondence indicators
    hasLetterFormat: /\b(dear|sincerely|regards|letter|memo|email)\b/.test(lowerContent),
    
    // Brief indicators
    hasBriefLanguage: /\b(brief|argument|analysis|conclusion|precedent|cite|holding)\b/.test(lowerContent),
    
    // General legal indicators
    hasCitations: /\b\d+\s+[A-Z][a-z.]*\s+\d+\b/.test(content),
    hasStatutes: /\b\d+\s+U\.?S\.?C\.?\s+§?\s*\d+/.test(content),
    hasLegalTerms: countLegalTerms(lowerContent)
  };
}

function countLegalTerms(content: string): number {
  const legalTerms = [
    'jurisdiction', 'plaintiff', 'defendant', 'liability', 'damages',
    'breach', 'warranty', 'indemnity', 'arbitration', 'venue',
    'statute', 'regulation', 'compliance', 'precedent', 'holding'
  ];
  
  return legalTerms.filter((term: string) => content.includes(term)).length;
}

function calculateClassificationConfidence(features: ClassificationFeatures, documentType: string): number {
  // Simple confidence calculation based on feature presence
  let score = 0;
  let maxScore = 0;

  switch (documentType) {
    case 'contract':
      score += features.hasContractLanguage ? 30 : 0;
      score += features.hasSignatureBlocks ? 20 : 0;
      maxScore = 50;
      break;
    
    case 'motion':
      score += features.hasMotionLanguage ? 25 : 0;
      score += features.hasProcedural ? 15 : 0;
      maxScore = 40;
      break;
    
    case 'evidence':
      score += features.hasEvidenceLanguage ? 20 : 0;
      maxScore = 20;
      break;
    
    case 'correspondence':
      score += features.hasLetterFormat ? 25 : 0;
      maxScore = 25;
      break;
    
    case 'brief':
      score += features.hasBriefLanguage ? 20 : 0;
      score += features.hasCitations ? 15 : 0;
      maxScore = 35;
      break;
  }

  // Add general legal indicators
  score += Math.min(features.hasLegalTerms * 2, 20);
  maxScore += 20;

  return Math.min(score / maxScore, 1.0);
}

function suggestAlternativeTypes(features: ClassificationFeatures): Array<{ type: string; confidence: number }> {
  const scores: Record<string, number> = {};
  
  if (features.hasContractLanguage) scores.contract = 0.8;
  if (features.hasMotionLanguage) scores.motion = 0.7;
  if (features.hasEvidenceLanguage) scores.evidence = 0.6;
  if (features.hasLetterFormat) scores.correspondence = 0.7;
  if (features.hasBriefLanguage) scores.brief = 0.8;

  return Object.entries(scores)
    .sort(([,a], [,b]) => b - a)
    .slice(0, 3)
    .map(([type, confidence]) => ({ type, confidence }));
}

// Risk Assessment Implementation
function assessDocumentRisks(content: string): RiskAssessment {
  const risks: RiskItem[] = [];
  const lowerContent = content.toLowerCase();

  // Financial risks
  if (/\b(penalty|fine|damages|liquidated damages)\b/.test(lowerContent)) {
    risks.push({
      type: 'financial',
      severity: 'high',
      description: 'Document contains penalty or damage clauses',
      keywords: content.match(/\b(penalty|fine|damages|liquidated damages)\b/gi) || []
    });
  }

  // Compliance risks
  if (/\b(regulation|compliance|violation|non-compliance)\b/.test(lowerContent)) {
    risks.push({
      type: 'compliance',
      severity: 'medium',
      description: 'Document references regulatory compliance requirements',
      keywords: content.match(/\b(regulation|compliance|violation|non-compliance)\b/gi) || []
    });
  }

  // Liability risks
  if (/\b(liability|indemnity|hold harmless|defend)\b/.test(lowerContent)) {
    risks.push({
      type: 'liability',
      severity: 'high',
      description: 'Document contains liability or indemnification provisions',
      keywords: content.match(/\b(liability|indemnity|hold harmless|defend)\b/gi) || []
    });
  }

  // Termination risks
  if (/\b(terminate|termination|breach|default)\b/.test(lowerContent)) {
    risks.push({
      type: 'termination',
      severity: 'medium',
      description: 'Document contains termination or breach provisions',
      keywords: content.match(/\b(terminate|termination|breach|default)\b/gi) || []
    });
  }

  return {
    totalRisks: risks.length,
    highRiskCount: risks.filter((r: RiskItem) => r.severity === 'high').length,
    risks: risks
  };
}

// Compliance Check Implementation
function checkCompliance(content: string, regulations: string[] = []): ComplianceCheck {
  const complianceResults: ComplianceCheck = {
    overallCompliance: 'unknown',
    checksPassed: 0,
    checksTotal: 0,
    violations: [],
    recommendations: []
  };

  // GDPR compliance checks
  if (regulations.includes('GDPR')) {
    complianceResults.checksTotal += 3;
    
    if (/\b(personal data|data subject|consent)\b/i.test(content)) {
      complianceResults.checksPassed += 1;
    } else {
      complianceResults.violations.push('GDPR: No clear data protection provisions found');
    }
    
    if (/\b(privacy policy|data processing|lawful basis)\b/i.test(content)) {
      complianceResults.checksPassed += 1;
    } else {
      complianceResults.violations.push('GDPR: Missing privacy policy references');
    }
    
    if (/\b(data retention|deletion|right to be forgotten)\b/i.test(content)) {
      complianceResults.checksPassed += 1;
    } else {
      complianceResults.violations.push('GDPR: No data retention provisions');
    }
  }

  // SOX compliance checks
  if (regulations.includes('SOX')) {
    complianceResults.checksTotal += 2;
    
    if (/\b(financial reporting|internal controls|audit)\b/i.test(content)) {
      complianceResults.checksPassed += 1;
    }
    
    if (/\b(disclosure|certification|whistleblower)\b/i.test(content)) {
      complianceResults.checksPassed += 1;
    }
  }

  // Calculate overall compliance
  if (complianceResults.checksTotal > 0) {
    const complianceRate = complianceResults.checksPassed / complianceResults.checksTotal;
    complianceResults.overallCompliance = complianceRate > 0.8 ? 'compliant' : 
                                         complianceRate > 0.5 ? 'partial' : 'non-compliant';
  }

  return complianceResults;
}

// Vector similarity calculation
function cosineSimilarity(vecA: number[], vecB: number[]): number {
  if (vecA.length !== vecB.length) {
    throw new Error('Vectors must have the same length');
  }

  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }

  normA = Math.sqrt(normA);
  normB = Math.sqrt(normB);

  if (normA === 0 || normB === 0) {
    return 0;
  }

  return dotProduct / (normA * normB);
}

// Full document analysis
function performFullAnalysis(content: string, documentType: string): FullAnalysisResult {
  return {
    entities: extractLegalEntities(content),
    classification: classifyDocument(content, documentType),
    risks: assessDocumentRisks(content),
    compliance: checkCompliance(content),
    metadata: {
      wordCount: content.split(/\s+/).length,
      characterCount: content.length,
      paragraphs: content.split(/\n\s*\n/).length,
      processedAt: new Date().toISOString()
    }
  };
}

// Sentiment analysis
function analyzeSentiment(text: string): SentimentAnalysis {
  const positiveWords = ['benefit', 'advantage', 'profit', 'gain', 'favorable', 'positive', 'agree', 'accept'];
  const negativeWords = ['loss', 'damage', 'penalty', 'breach', 'violation', 'risk', 'liability', 'dispute'];
  
  const words = text.toLowerCase().split(/\s+/);
  let positiveCount = 0;
  let negativeCount = 0;
  
  words.forEach((word: string) => {
    if (positiveWords.some((pos: string) => word.includes(pos))) positiveCount++;
    if (negativeWords.some((neg: string) => word.includes(neg))) negativeCount++;
  });
  
  const total = positiveCount + negativeCount;
  const sentiment = total === 0 ? 0 : (positiveCount - negativeCount) / total;
  
  return {
    sentiment: sentiment,
    label: sentiment > 0.1 ? 'positive' : sentiment < -0.1 ? 'negative' : 'neutral',
    confidence: Math.abs(sentiment),
    positiveCount,
    negativeCount
  };
}

// Text classification
function classifyText(text: string, classes: string[]): TextClassification {
  // Simple keyword-based classification
  const scores: Record<string, number> = {};
  
  classes.forEach((cls: string) => {
    scores[cls] = 0;
    // This would be replaced with actual ML model inference
    if (text.toLowerCase().includes(cls.toLowerCase())) {
      scores[cls] = 0.8;
    }
  });
  
  const maxClass = Object.keys(scores).reduce((a: string, b: string) => scores[a] > scores[b] ? a : b);
  
  return {
    class: maxClass,
    confidence: scores[maxClass],
    allScores: scores
  };
}

// Text summarization
function summarizeText(text: string, maxLength: number = 200): string {
  const sentences = text.split(/[.!?]+/).filter((s: string) => s.trim().length > 10);
  
  if (sentences.length <= 3) {
    return text.substring(0, maxLength);
  }
  
  // Simple extractive summarization - take first, middle, and last sentences
  const firstSentence = sentences[0].trim();
  const middleSentence = sentences[Math.floor(sentences.length / 2)].trim();
  const lastSentence = sentences[sentences.length - 1].trim();
  
  const summary = `${firstSentence}. ${middleSentence}. ${lastSentence}.`;
  
  return summary.length > maxLength ? 
    summary.substring(0, maxLength) + '...' : 
    summary;
}

// Mock embedding generation
function generateEmbedding(text: string): number[] {
  // This would be replaced with actual embedding model inference
  const dimension = 384;
  const embedding = new Array(dimension);
  
  // Simple hash-based mock embedding
  let hash = 0;
  for (let i = 0; i < text.length; i++) {
    hash = ((hash << 5) - hash + text.charCodeAt(i)) & 0xffffffff;
  }
  
  for (let i = 0; i < dimension; i++) {
    hash = ((hash << 5) - hash + i) & 0xffffffff;
    embedding[i] = (hash / 0xffffffff) * 2 - 1; // Normalize to [-1, 1]
  }
  
  return embedding;
}

// Data processing utilities
function cleanLegalText(text: string): string {
  return text
    .replace(/\s+/g, ' ') // Normalize whitespace
    .replace(/[^\w\s\.\,\;\:\!\?\-\(\)]/g, '') // Remove special characters
    .trim();
}

function extractCitations(text: string): Citation[] {
  const citationPattern = /\b([A-Z][a-zA-Z\s.,'&-]+)\s+v\.?\s+([A-Z][a-zA-Z\s.,'&-]+),?\s+(\d+)\s+([A-Z][a-z.]*)\s+(\d+)/g;
  return [...text.matchAll(citationPattern)].map((match: RegExpMatchArray) => ({
    fullCitation: match[0],
    plaintiff: match[1].trim(),
    defendant: match[2].trim(),
    volume: match[3],
    reporter: match[4],
    page: match[5]
  }));
}

function normalizeDates(text: string): string {
  const datePattern = /\b(?:(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})|(\d{1,2})\/(\d{1,2})\/(\d{4})|(\d{4})-(\d{2})-(\d{2}))\b/g;
  
  return text.replace(datePattern, (match: string) => {
    const date = new Date(match);
    return date.toLocaleDateString('en-US');
  });
}

function mergeDocuments(documents: Array<{ content: string }>): MergedDocuments {
  return {
    mergedContent: documents.map((doc: { content: string }) => doc.content).join('\n\n---\n\n'),
    documentCount: documents.length,
    totalLength: documents.reduce((sum: number, doc: { content: string }) => sum + doc.content.length, 0),
    mergedAt: new Date().toISOString()
  };
}

function generateDocumentSummary(data: { content: string; maxLength?: number }): DocumentSummary {
  const { content, maxLength = 300 } = data;
  
  return {
    summary: summarizeText(content, maxLength),
    keyPoints: extractKeyPoints(content),
    wordCount: content.split(/\s+/).length,
    entities: extractLegalEntities(content)
  };
}

function extractKeyPoints(text: string): string[] {
  // Extract sentences that likely contain key information
  const sentences = text.split(/[.!?]+/);
  const keyPoints: string[] = [];
  
  const importantPatterns = [
    /\b(shall|must|will|agree|covenant|warrant)\b/i,
    /\b(penalty|damages|liability|indemnity)\b/i,
    /\b(terminate|termination|breach|default)\b/i,
    /\$[\d,]+/,
    /\b\d{1,2}\/\d{1,2}\/\d{4}\b/
  ];
  
  sentences.forEach((sentence: string) => {
    const trimmed = sentence.trim();
    if (trimmed.length > 20 && importantPatterns.some((pattern: RegExp) => pattern.test(trimmed))) {
      keyPoints.push(trimmed);
    }
  });
  
  return keyPoints.slice(0, 5); // Return top 5 key points
}

console.log(`✅ Worker thread ${workerId} initialized and ready for tasks`);

// Health check message
setInterval(() => {
  const memoryUsage = process.memoryUsage();
  parentPort?.postMessage({
    type: 'health_check',
    workerId,
    memoryUsage: memoryUsage.heapUsed,
    timestamp: Date.now()
  });
}, 60000); // Every minute