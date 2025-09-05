// scripts/enhanced-rag-phase-integration.ts
// Enhanced RAG Phase with ML, caching, ranking, and VS Code integration

import { spawn } from 'child_process';
import fs from 'fs';
import path from 'path';
import fetch from 'node-fetch';
import os from 'os';
import crypto from 'crypto';
import loki from 'lokijs';
import Fuse from 'fuse.js';

// Enhanced RAG Configuration
const config = {
  timestamp: new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19),
  ollamaHost: process.env.OLLAMA_HOST || 'http://localhost:11434',
  embedModel: 'nomic-embed-text',
  legalModel: 'gemma3-legal',
  vectorDimension: 768,
  maxCacheSize: 10000,
  rankingThreshold: 0.75,
  mlEnabled: true,
  viteIntegration: true
};

// Directory setup
const ragDir = `enhanced-rag-${config.timestamp}`;
const cacheDir = path.join(ragDir, 'cache');
const embeddingsDir = path.join(ragDir, 'embeddings');
const logsDir = path.join(ragDir, 'logs');
const rankingsDir = path.join(ragDir, 'rankings');

[ragDir, cacheDir, embeddingsDir, logsDir, rankingsDir].forEach(dir => {
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
});

console.log(`🚀 Enhanced RAG Phase Integration Started
📁 Directories: ${ragDir}
🧠 Models: ${config.embedModel} + ${config.legalModel}
⚡ Cache: LokiJS + Fuse.js enabled
🔧 VS Code Integration: Active`);

// Enhanced document interface with ML features
interface EnhancedDocument {
  id: string;
  content: string;
  summary: string;
  embedding: number[];
  label: DocumentLabel;
  score: number;
  confidence: number;
  source: DocumentSource;
  metadata: DocumentMetadata;
  rankingFeatures: RankingFeatures;
  timestamp: Date;
  hash: string;
}

enum DocumentLabel {
  Contract = 'contract',
  Tort = 'tort',
  Criminal = 'criminal',
  Evidence = 'evidence',
  Precedent = 'precedent',
  Motion = 'motion',
  Brief = 'brief',
  Unknown = 'unknown'
}

enum DocumentSource {
  Upload = 'upload',
  Generated = 'generated',
  Retrieved = 'retrieved',
  Vite = 'vite-error',
  Build = 'build-log'
}

interface DocumentMetadata {
  fileType: string;
  wordCount: number;
  complexity: number;
  legalTerms: string[];
  entities: string[];
  citations: string[];
}

interface RankingFeatures {
  clarity: number;
  relevance: number;
  completeness: number;
  authority: number;
  recency: number;
  usage: number;
}

// Enhanced Cache Manager with LokiJS
class EnhancedCacheManager {
  private db: loki;
  private documents: any;
  private embeddings: any;
  private rankings: any;
  private searchIndex: Fuse<EnhancedDocument>;

  constructor() {
    this.initializeCache();
  }

  private initializeCache() {
    const dbFile = path.join(cacheDir, 'enhanced-cache.db');
    
    try {
      // Initialize LokiJS database
      this.db = new loki(dbFile, {
        autosave: true,
        autosaveInterval: 10000
      });
      
      // Load collections immediately for synchronous initialization
      this.loadCollections();
      
      console.log(`📦 LokiJS cache initialized: ${dbFile}`);
    } catch (error) {
      console.error('❌ Cache initialization failed:', error);
      this.db = new loki();
      this.loadCollections();
    }
  }

  private loadCollections() {
    // Get or create collections
    this.documents = this.db.getCollection('documents') || 
                    this.db.addCollection('documents', { indices: ['hash', 'label', 'score'] });
    
    this.embeddings = this.db.getCollection('embeddings') || 
                     this.db.addCollection('embeddings', { indices: ['documentId'] });
    
    this.rankings = this.db.getCollection('rankings') || 
                   this.db.addCollection('rankings', { indices: ['documentId', 'score'] });

    // Initialize Fuse.js search index
    this.initializeFuseSearch();
    
    console.log(`📦 Collections loaded: ${this.documents.count()} documents`);
  }

  private initializeFuseSearch() {
    const allDocs = this.documents.find({});
    
    this.searchIndex = new Fuse(allDocs, {
      keys: [
        { name: 'content', weight: 0.4 },
        { name: 'summary', weight: 0.3 },
        { name: 'label', weight: 0.2 },
        { name: 'metadata.legalTerms', weight: 0.1 }
      ],
      threshold: 0.3,
      includeScore: true
    });
    
    console.log(`🔍 Fuse.js search index initialized with ${allDocs.length} documents`);
  }

  async getDocument(hash: string): Promise<EnhancedDocument | null> {
    const result = this.documents.findOne({ hash });
    return result;
  }

  async storeDocument(doc: EnhancedDocument): Promise<void> {
    const existing = await this.getDocument(doc.hash);
    
    if (existing) {
      // Update existing document
      this.documents.update({ ...existing, ...doc });
    } else {
      // Add new document
      this.documents.insert(doc);
      
      // Maintain cache size limit
      if (this.documents.count() > config.maxCacheSize) {
        const excess = this.documents.chain()
          .simplesort('score')
          .limit(this.documents.count() - config.maxCacheSize)
          .data();
        
        excess.forEach(doc => this.documents.remove(doc));
      }
    }
    
    // Update Fuse search index
    this.initializeFuseSearch();
    
    // Auto-save handled by LokiJS
  }

  async searchSimilar(query: string, limit: number = 10): Promise<EnhancedDocument[]> {
    const results = this.searchIndex.search(query);
    return results
      .map(result => result.item)
      .slice(0, limit);
  }

  async getTopDocuments(limit: number = 10): Promise<EnhancedDocument[]> {
    return this.documents.chain()
      .simplesort('score', true)
      .limit(limit)
      .data();
  }

  async getDocumentsByLabel(label: DocumentLabel): Promise<EnhancedDocument[]> {
    return this.documents.find({ label });
  }
}

// Enhanced Embedding Service with nomic-embed
class EnhancedEmbeddingService {
  private cacheManager: EnhancedCacheManager;

  constructor(cacheManager: EnhancedCacheManager) {
    this.cacheManager = cacheManager;
  }

  async generateEmbedding(text: string): Promise<number[] | null> {
    const hash = this.hashText(text);
    
    // Check cache first
    const cached = await this.cacheManager.getDocument(hash);
    if (cached?.embedding) {
      console.log(`📦 Cache hit for embedding: ${hash.slice(0, 8)}...`);
      return cached.embedding;
    }

    try {
      console.log(`🧠 Generating embedding with ${config.embedModel}...`);
      
      const response = await fetch(`${config.ollamaHost}/api/embeddings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: config.embedModel,
          prompt: text
        })
      });

      if (!response.ok) {
        throw new Error(`Embedding API error: ${response.status}`);
      }

      const result = await response.json();
      console.log(`✅ Generated ${result.embedding?.length || 0}-dim embedding`);
      
      return result.embedding;
    } catch (error) {
      console.error('❌ Embedding generation failed:', error);
      return null;
    }
  }

  async batchGenerateEmbeddings(texts: string[]): Promise<Array<number[] | null>> {
    console.log(`🔄 Batch generating ${texts.length} embeddings...`);
    
    const embeddings = await Promise.all(
      texts.map(text => this.generateEmbedding(text))
    );
    
    const successful = embeddings.filter(e => e !== null).length;
    console.log(`✅ Batch complete: ${successful}/${texts.length} successful`);
    
    return embeddings;
  }

  private hashText(text: string): string {
    return crypto.createHash('sha256').update(text).digest('hex');
  }
}

// ML-powered Document Classifier and Scorer
class MLDocumentAnalyzer {
  private cacheManager: EnhancedCacheManager;

  constructor(cacheManager: EnhancedCacheManager) {
    this.cacheManager = cacheManager;
  }

  async analyzeDocument(content: string): Promise<Partial<EnhancedDocument>> {
    const hash = crypto.createHash('sha256').update(content).digest('hex');
    
    // Check cache
    const cached = await this.cacheManager.getDocument(hash);
    if (cached) {
      console.log(`📦 Using cached analysis: ${hash.slice(0, 8)}...`);
      return cached;
    }

    console.log(`🤖 Analyzing document with ${config.legalModel}...`);

    // Generate summary and classification
    const [summary, label, metadata, ranking] = await Promise.all([
      this.generateSummary(content),
      this.classifyDocument(content),
      this.extractMetadata(content),
      this.calculateRanking(content)
    ]);

    const analysis = {
      id: hash,
      content,
      summary,
      label,
      metadata,
      rankingFeatures: ranking,
      score: this.calculateScore(ranking),
      confidence: this.calculateConfidence(content, label),
      hash,
      timestamp: new Date()
    };

    console.log(`✅ Analysis complete: ${label} (score: ${analysis.score.toFixed(2)})`);
    return analysis;
  }

  private async generateSummary(content: string): Promise<string> {
    try {
      const response = await fetch(`${config.ollamaHost}/api/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: config.legalModel,
          prompt: `Summarize this legal document in 2-3 sentences:\n\n${content.slice(0, 1000)}`,
          temperature: 0.3,
          stream: false
        })
      });

      const result = await response.json();
      return result.response || 'Summary generation failed';
    } catch (error) {
      console.error('❌ Summary generation failed:', error);
      return content.slice(0, 200) + '...';
    }
  }

  private async classifyDocument(content: string): Promise<DocumentLabel> {
    const lowerContent = content.toLowerCase();
    
    // Simple rule-based classification (can be enhanced with ML)
    if (lowerContent.includes('contract') || lowerContent.includes('agreement')) {
      return DocumentLabel.Contract;
    } else if (lowerContent.includes('tort') || lowerContent.includes('negligence')) {
      return DocumentLabel.Tort;
    } else if (lowerContent.includes('criminal') || lowerContent.includes('defendant')) {
      return DocumentLabel.Criminal;
    } else if (lowerContent.includes('evidence') || lowerContent.includes('exhibit')) {
      return DocumentLabel.Evidence;
    } else if (lowerContent.includes('precedent') || lowerContent.includes('case law')) {
      return DocumentLabel.Precedent;
    } else if (lowerContent.includes('motion') || lowerContent.includes('petition')) {
      return DocumentLabel.Motion;
    } else if (lowerContent.includes('brief') || lowerContent.includes('memorandum')) {
      return DocumentLabel.Brief;
    }
    
    return DocumentLabel.Unknown;
  }

  private async extractMetadata(content: string): Promise<DocumentMetadata> {
    const words = content.split(/\s+/);
    const legalTerms = this.extractLegalTerms(content);
    const entities = this.extractEntities(content);
    const citations = this.extractCitations(content);
    
    return {
      fileType: 'text',
      wordCount: words.length,
      complexity: this.calculateComplexity(content),
      legalTerms,
      entities,
      citations
    };
  }

  private calculateRanking(content: string): RankingFeatures {
    const words = content.split(/\s+/);
    const sentences = content.split(/[.!?]+/);
    
    return {
      clarity: this.calculateClarity(content),
      relevance: this.calculateRelevance(content),
      completeness: Math.min(words.length / 1000, 1.0),
      authority: this.calculateAuthority(content),
      recency: 1.0, // Assume recent for new documents
      usage: 0.0    // Will be updated based on access patterns
    };
  }

  private calculateScore(ranking: RankingFeatures): number {
    const weights = {
      clarity: 0.2,
      relevance: 0.25,
      completeness: 0.15,
      authority: 0.2,
      recency: 0.1,
      usage: 0.1
    };
    
    return Object.entries(ranking).reduce((score, [key, value]) => {
      return score + (weights[key as keyof RankingFeatures] || 0) * value;
    }, 0);
  }

  private calculateConfidence(content: string, label: DocumentLabel): number {
    // Simple confidence calculation based on content length and label certainty
    const baseConfidence = Math.min(content.length / 5000, 1.0);
    const labelConfidence = label === DocumentLabel.Unknown ? 0.3 : 0.8;
    return baseConfidence * labelConfidence;
  }

  private extractLegalTerms(content: string): string[] {
    const legalTerms = [
      'plaintiff', 'defendant', 'contract', 'tort', 'negligence', 'liability',
      'damages', 'evidence', 'precedent', 'statute', 'regulation', 'motion',
      'petition', 'brief', 'memorandum', 'jurisdiction', 'venue', 'discovery'
    ];
    
    return legalTerms.filter(term => 
      content.toLowerCase().includes(term)
    );
  }

  private extractEntities(content: string): string[] {
    // Simple entity extraction (can be enhanced with NLP)
    const entityPatterns = [
      /([A-Z][a-z]+ v\. [A-Z][a-z]+)/g, // Case names
      /(\d{4} WL \d+)/g, // Westlaw citations
      /(\d+ F\.\d+d \d+)/g // Federal citations
    ];
    
    const entities: string[] = [];
    entityPatterns.forEach(pattern => {
      const matches = content.match(pattern);
      if (matches) entities.push(...matches);
    });
    
    return entities;
  }

  private extractCitations(content: string): string[] {
    const citationPattern = /\d+\s+[A-Z][a-z]+\s+\d+/g;
    return content.match(citationPattern) || [];
  }

  private calculateClarity(content: string): number {
    const words = content.split(/\s+/);
    const avgWordLength = words.reduce((sum, word) => sum + word.length, 0) / words.length;
    const sentences = content.split(/[.!?]+/);
    const avgSentenceLength = words.length / sentences.length;
    
    // Score based on readability (shorter words/sentences = higher clarity)
    return Math.max(0, 1 - (avgWordLength - 5) / 10 - (avgSentenceLength - 15) / 30);
  }

  private calculateRelevance(content: string): number {
    // Score based on legal term density
    const legalTerms = this.extractLegalTerms(content);
    const words = content.split(/\s+/);
    return Math.min(legalTerms.length / words.length * 100, 1.0);
  }

  private calculateAuthority(content: string): number {
    const citations = this.extractCitations(content);
    const entities = this.extractEntities(content);
    
    // Score based on citations and legal entities
    return Math.min((citations.length + entities.length) / 10, 1.0);
  }
}

// Vite Integration for Error Logging
class ViteIntegrationManager {
  private logFile: string;

  constructor() {
    this.logFile = path.join(logsDir, 'vite-errors.log');
  }

  async setupViteLogging(): Promise<void> {
    console.log('🔧 Setting up Vite error logging integration...');
    
    // Create Vite plugin configuration
    const vitePluginContent = `
// Enhanced RAG Vite Plugin
import type { Plugin } from 'vite';

export const enhancedRagPlugin = (): Plugin => ({
  name: 'enhanced-rag-logging',
  configureServer(server) {
    server.middlewares.use((req, res, next) => {
      console.log(\`[VITE RAG] \${req.method} \${req.url}\`);
      next();
    });

    server.watcher.on('error', (err) => {
      const errorData = {
        timestamp: new Date().toISOString(),
        type: 'vite-watch-error',
        message: err.message,
        stack: err.stack,
        source: 'vite-watcher'
      };
      
      console.error('[VITE RAG ERROR]', errorData);
      
      // Write to enhanced RAG log
      require('fs').appendFileSync('${this.logFile}', 
        JSON.stringify(errorData) + '\\n'
      );
    });
  },
  
  handleHotUpdate(ctx) {
    console.log(\`[VITE RAG HMR] \${ctx.file}\`);
  }
});`;

    const pluginPath = path.join(process.cwd(), 'vite-enhanced-rag.plugin.ts');
    fs.writeFileSync(pluginPath, vitePluginContent);
    
    console.log(`✅ Vite plugin created: ${pluginPath}`);
  }

  async processViteLogs(): Promise<EnhancedDocument[]> {
    if (!fs.existsSync(this.logFile)) {
      return [];
    }

    const logs = fs.readFileSync(this.logFile, 'utf-8')
      .split('\n')
      .filter(line => line.trim())
      .map(line => {
        try {
          return JSON.parse(line);
        } catch {
          return null;
        }
      })
      .filter(Boolean);

    console.log(`📊 Processing ${logs.length} Vite error logs...`);

    const documents: EnhancedDocument[] = logs.map(log => ({
      id: crypto.createHash('sha256').update(JSON.stringify(log)).digest('hex'),
      content: `${log.type}: ${log.message}\n${log.stack || ''}`,
      summary: `Vite ${log.type}: ${log.message.slice(0, 100)}...`,
      embedding: [], // Will be generated later
      label: DocumentLabel.Unknown,
      score: 0.5,
      confidence: 0.8,
      source: DocumentSource.Vite,
      metadata: {
        fileType: 'error-log',
        wordCount: log.message.split(' ').length,
        complexity: 0.3,
        legalTerms: [],
        entities: [],
        citations: []
      },
      rankingFeatures: {
        clarity: 0.9,
        relevance: 0.7,
        completeness: 0.6,
        authority: 0.1,
        recency: 1.0,
        usage: 0.0
      },
      timestamp: new Date(log.timestamp),
      hash: crypto.createHash('sha256').update(JSON.stringify(log)).digest('hex')
    }));

    return documents;
  }
}

// VS Code Integration Manager
class VSCodeIntegrationManager {
  private summaryFile: string;
  private diagnosticsFile: string;

  constructor() {
    this.summaryFile = path.join(process.cwd(), '.vscode', 'rag-summary.md');
    this.diagnosticsFile = path.join(process.cwd(), '.vscode', 'rag-diagnostics.json');
  }

  async generateVSCodeSummary(documents: EnhancedDocument[]): Promise<void> {
    const topDocs = documents
      .sort((a, b) => b.score - a.score)
      .slice(0, 10);

    const summary = `# Enhanced RAG Analysis Summary

## 📊 Document Analytics
- **Total Documents**: ${documents.length}
- **Average Score**: ${(documents.reduce((sum, doc) => sum + doc.score, 0) / documents.length).toFixed(2)}
- **High-Confidence Docs**: ${documents.filter(doc => doc.confidence > 0.8).length}

## 🏆 Top-Ranked Documents

${topDocs.map((doc, index) => `
### ${index + 1}. ${doc.label} (Score: ${doc.score.toFixed(2)})
**Summary**: ${doc.summary}
**Source**: ${doc.source}
**Confidence**: ${(doc.confidence * 100).toFixed(1)}%
**Features**: Clarity ${(doc.rankingFeatures.clarity * 100).toFixed(0)}% | Relevance ${(doc.rankingFeatures.relevance * 100).toFixed(0)}%

`).join('')}

## 🔍 Label Distribution
${Object.values(DocumentLabel).map(label => {
  const count = documents.filter(doc => doc.label === label).length;
  const percentage = ((count / documents.length) * 100).toFixed(1);
  return `- **${label}**: ${count} documents (${percentage}%)`;
}).join('\n')}

## 🚀 Recommendations
1. Focus on high-scoring ${topDocs[0]?.label || 'documents'} for legal analysis
2. Review low-confidence documents for accuracy
3. Use vector similarity search for related content discovery

---
*Generated by Enhanced RAG Phase Integration*
*Timestamp: ${new Date().toISOString()}*
`;

    fs.writeFileSync(this.summaryFile, summary);
    console.log(`📄 VS Code summary generated: ${this.summaryFile}`);
  }

  async generateDiagnostics(documents: EnhancedDocument[]): Promise<void> {
    const diagnostics = {
      timestamp: new Date().toISOString(),
      totalDocuments: documents.length,
      averageScore: documents.reduce((sum, doc) => sum + doc.score, 0) / documents.length,
      labelDistribution: this.calculateLabelDistribution(documents),
      topDocuments: documents
        .sort((a, b) => b.score - a.score)
        .slice(0, 5)
        .map(doc => ({
          id: doc.id.slice(0, 8),
          label: doc.label,
          score: doc.score,
          confidence: doc.confidence,
          summary: doc.summary.slice(0, 100)
        })),
      recommendations: this.generateRecommendations(documents)
    };

    fs.writeFileSync(this.diagnosticsFile, JSON.stringify(diagnostics, null, 2));
    console.log(`🔍 VS Code diagnostics generated: ${this.diagnosticsFile}`);
  }

  private calculateLabelDistribution(documents: EnhancedDocument[]): Record<string, number> {
    return documents.reduce((dist, doc) => {
      dist[doc.label] = (dist[doc.label] || 0) + 1;
      return dist;
    }, {} as Record<string, number>);
  }

  private generateRecommendations(documents: EnhancedDocument[]): string[] {
    const recommendations = [];
    
    const highScoreDocs = documents.filter(doc => doc.score > 0.8);
    if (highScoreDocs.length > 0) {
      recommendations.push(`Focus on ${highScoreDocs.length} high-scoring documents for primary analysis`);
    }
    
    const lowConfidenceDocs = documents.filter(doc => doc.confidence < 0.5);
    if (lowConfidenceDocs.length > 0) {
      recommendations.push(`Review ${lowConfidenceDocs.length} low-confidence documents for accuracy`);
    }
    
    const contractDocs = documents.filter(doc => doc.label === DocumentLabel.Contract);
    if (contractDocs.length > 0) {
      recommendations.push(`${contractDocs.length} contract documents available for comparative analysis`);
    }
    
    return recommendations;
  }
}

// Main Enhanced RAG Integration
async function runEnhancedRAGPhase() {
  const startTime = Date.now();
  console.log('🚀 Enhanced RAG Phase Integration Starting...');

  // Initialize managers
  const cacheManager = new EnhancedCacheManager();
  const embeddingService = new EnhancedEmbeddingService(cacheManager);
  const mlAnalyzer = new MLDocumentAnalyzer(cacheManager);
  const viteManager = new ViteIntegrationManager();
  const vscodeManager = new VSCodeIntegrationManager();

  // Setup Vite integration
  if (config.viteIntegration) {
    await viteManager.setupViteLogging();
  }

  // Process sample documents (in production, these would come from your legal database)
  const sampleDocuments = [
    "This contract establishes terms between plaintiff and defendant regarding breach of contract claims...",
    "Evidence submitted shows negligence in tort liability case with damages exceeding $100,000...",
    "Criminal defendant pleads not guilty to charges. Motion to suppress evidence filed..."
  ];

  console.log(`📚 Processing ${sampleDocuments.length} sample documents...`);

  // Analyze documents with ML
  const analyzedDocs: EnhancedDocument[] = [];
  for (const content of sampleDocuments) {
    const analysis = await mlAnalyzer.analyzeDocument(content);
    
    // Generate embedding
    const embedding = await embeddingService.generateEmbedding(content);
    if (embedding) {
      analysis.embedding = embedding;
    }
    
    // Store in cache
    await cacheManager.storeDocument(analysis as EnhancedDocument);
    analyzedDocs.push(analysis as EnhancedDocument);
  }

  // Process Vite logs if available
  const viteDocs = await viteManager.processViteLogs();
  analyzedDocs.push(...viteDocs);

  // Generate VS Code integrations
  await vscodeManager.generateVSCodeSummary(analyzedDocs);
  await vscodeManager.generateDiagnostics(analyzedDocs);

  // Performance metrics
  const duration = Date.now() - startTime;
  console.log(`
✅ Enhanced RAG Phase Integration Complete!

📊 Results:
- Documents Processed: ${analyzedDocs.length}
- Embeddings Generated: ${analyzedDocs.filter(doc => doc.embedding?.length > 0).length}
- Average Score: ${(analyzedDocs.reduce((sum, doc) => sum + doc.score, 0) / analyzedDocs.length).toFixed(2)}
- Processing Time: ${(duration / 1000).toFixed(2)}s

📁 Output Files:
- Cache: ${path.join(cacheDir, 'enhanced-cache.json')}
- VS Code Summary: ${path.join(process.cwd(), '.vscode', 'rag-summary.md')}
- Diagnostics: ${path.join(process.cwd(), '.vscode', 'rag-diagnostics.json')}
- Vite Plugin: vite-enhanced-rag.plugin.ts

🚀 Next Steps:
1. Add Vite plugin to your vite.config.ts
2. Install LokiJS and Fuse.js: npm install lokijs fuse.js
3. Check VS Code files for analysis results
4. Use cache for fast retrieval and ranking
`);

  return analyzedDocs;
}

// Export for use in other modules
export {
  runEnhancedRAGPhase,
  EnhancedCacheManager,
  EnhancedEmbeddingService,
  MLDocumentAnalyzer,
  ViteIntegrationManager,
  VSCodeIntegrationManager,
  type EnhancedDocument,
  DocumentLabel,
  DocumentSource
};

// Run if called directly
runEnhancedRAGPhase().catch(console.error);