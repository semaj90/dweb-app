// Advanced Semantic Analysis Pipeline
// Integrates LangChain, Transformers.js ONNX, and Legal-BERT for comprehensive legal document analysis

import { pipeline, env, AutoTokenizer } from '@xenova/transformers';
import { Document } from '@langchain/core/documents';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { PromptTemplate } from '@langchain/core/prompts';

// Set up environment for ONNX models
env.allowRemoteModels = false;
env.allowLocalModels = true;
env.localModelPath = '/models/';

interface SemanticAnalysisConfig {
  models: {
    gemma2b: string;
    legalBert: string;
    ner: string;
    sentiment: string;
    classification: string;
  };
  thresholds: {
    confidenceMin: number;
    entityMin: number;
    sentimentNeutral: number;
  };
  processing: {
    chunkSize: number;
    chunkOverlap: number;
    maxConcurrent: number;
  };
}

interface LegalEntity {
  text: string;
  label: string;
  confidence: number;
  start: number;
  end: number;
  context: string;
  legalCategory: 'person' | 'organization' | 'location' | 'case_law' | 'statute' | 'contract_term' | 'date' | 'money';
}

interface SemanticAnalysisResult {
  documentId: string;
  analysis: {
    summary: string;
    keyTopics: string[];
    legalCategories: string[];
    complexity: 'low' | 'medium' | 'high';
    readabilityScore: number;
  };
  entities: LegalEntity[];
  sentiment: {
    overall: number;
    confidence: number;
    distribution: {
      positive: number;
      neutral: number;
      negative: number;
    };
    legalTone: 'formal' | 'informal' | 'technical' | 'persuasive';
  };
  classification: {
    primaryCategory: string;
    confidence: number;
    secondaryCategories: Array<{ category: string; confidence: number }>;
    practiceAreas: string[];
  };
  riskAssessment: {
    overallRisk: 'low' | 'medium' | 'high';
    riskFactors: Array<{ factor: string; severity: number; description: string }>;
    complianceIssues: string[];
    recommendations: string[];
  };
  keyPhrases: Array<{
    phrase: string;
    importance: number;
    category: string;
    context: string;
  }>;
  structuralAnalysis: {
    sections: Array<{ title: string; content: string; importance: number }>;
    citations: Array<{ text: string; type: 'case' | 'statute' | 'regulation'; verified: boolean }>;
    definitions: Array<{ term: string; definition: string; source: string }>;
  };
  userIntent: {
    primaryIntent: string;
    confidence: number;
    intentCategories: string[];
    suggestedActions: string[];
    urgency: 'low' | 'medium' | 'high';
  };
  processingMetadata: {
    processingTime: number;
    modelsUsed: string[];
    chunks: number;
    confidence: number;
    warnings: string[];
  };
}

export class SemanticAnalysisPipeline {
  private config: SemanticAnalysisConfig;
  private models: Map<string, any> = new Map();
  private tokenizers: Map<string, any> = new Map();
  private initialized = false;
  private textSplitter: RecursiveCharacterTextSplitter;
  
  // LangChain prompt templates
  private summaryTemplate: PromptTemplate;
  private classificationTemplate: PromptTemplate;
  private riskAssessmentTemplate: PromptTemplate;

  constructor(config?: Partial<SemanticAnalysisConfig>) {
    this.config = {
      models: {
        gemma2b: 'Xenova/gemma-2b',
        legalBert: 'Xenova/legal-bert-base-uncased',
        ner: 'Xenova/bert-base-NER',
        sentiment: 'Xenova/distilbert-base-uncased-finetuned-sst-2-english',
        classification: 'Xenova/distilbert-base-uncased',
        ...config?.models
      },
      thresholds: {
        confidenceMin: 0.7,
        entityMin: 0.6,
        sentimentNeutral: 0.1,
        ...config?.thresholds
      },
      processing: {
        chunkSize: 1000,
        chunkOverlap: 200,
        maxConcurrent: 4,
        ...config?.processing
      }
    };

    this.textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: this.config.processing.chunkSize,
      chunkOverlap: this.config.processing.chunkOverlap,
      separators: ['\n\n', '\n', '. ', '! ', '? ', ' ', '']
    });

    this.initializePromptTemplates();
  }

  async initialize(): Promise<void> {
    if (this.initialized) return;

    try {
      console.log('🧠 Loading semantic analysis models...');
      
      // Load models in parallel with limited concurrency
      const modelLoadPromises = [
        this.loadModel('gemma2b', 'text2text-generation'),
        this.loadModel('legalBert', 'feature-extraction'),
        this.loadModel('ner', 'token-classification'),
        this.loadModel('sentiment', 'sentiment-analysis'),
        this.loadModel('classification', 'zero-shot-classification')
      ];

      await Promise.all(modelLoadPromises);
      
      this.initialized = true;
      console.log('✅ Semantic Analysis Pipeline initialized');
      
    } catch (error) {
      console.error('❌ Failed to initialize Semantic Analysis Pipeline:', error);
      throw error;
    }
  }

  private async loadModel(name: string, task: string): Promise<void> {
    try {
      const modelPath = this.config.models[name];
      const model = await pipeline(task as any, modelPath, {
        quantized: true // Use quantized models for better performance
      });
      
      this.models.set(name, model);
      
      // Load tokenizer if needed
      if (['gemma2b', 'legalBert'].includes(name)) {
        const tokenizer = await AutoTokenizer.from_pretrained(modelPath);
        this.tokenizers.set(name, tokenizer);
      }
      
      console.log(`✅ Loaded ${name} model`);
    } catch (error) {
      console.error(`❌ Failed to load ${name} model:`, error);
      throw error;
    }
  }

  private initializePromptTemplates(): void {
    this.summaryTemplate = PromptTemplate.fromTemplate(`
      Analyze the following legal document and provide a comprehensive summary:
      
      Document: {document}
      
      Provide:
      1. Executive summary (2-3 sentences)
      2. Key legal topics
      3. Main parties involved
      4. Critical dates and deadlines
      5. Legal implications
      
      Summary:
    `);

    this.classificationTemplate = PromptTemplate.fromTemplate(`
      Classify the following legal document into appropriate categories:
      
      Document: {document}
      
      Categories to consider:
      - Contract Law
      - Tort Law
      - Criminal Law
      - Constitutional Law
      - Administrative Law
      - Intellectual Property
      - Employment Law
      - Real Estate Law
      - Corporate Law
      - Family Law
      
      Primary Category:
      Secondary Categories:
      Practice Areas:
    `);

    this.riskAssessmentTemplate = PromptTemplate.fromTemplate(`
      Assess the legal risks and compliance issues in this document:
      
      Document: {document}
      
      Evaluate:
      1. Compliance with regulations
      2. Potential legal vulnerabilities
      3. Risk factors and their severity
      4. Recommended actions
      
      Risk Assessment:
    `);
  }

  async analyzeDocument(document: {
    id: string;
    content: string;
    title?: string;
    metadata?: any;
  }): Promise<SemanticAnalysisResult> {
    if (!this.initialized) {
      await this.initialize();
    }

    const startTime = Date.now();
    const warnings: string[] = [];

    try {
      console.log(`🔍 Analyzing document: ${document.id}`);

      // Split document into chunks for processing
      const chunks = await this.textSplitter.splitText(document.content);
      const docObjects = chunks.map(chunk => new Document({ pageContent: chunk }));

      // Parallel analysis tasks
      const [
        entities,
        sentiment,
        classification,
        summary,
        keyPhrases,
        structuralAnalysis,
        userIntent,
        riskAssessment
      ] = await Promise.allSettled([
        this.extractEntities(document.content),
        this.analyzeSentiment(document.content),
        this.classifyDocument(document.content),
        this.generateSummary(document.content),
        this.extractKeyPhrases(document.content),
        this.analyzeStructure(document.content),
        this.detectUserIntent(document.content),
        this.assessRisks(document.content)
      ]);

      // Process results and handle failures
      const processingTime = Date.now() - startTime;

      const result: SemanticAnalysisResult = {
        documentId: document.id,
        analysis: this.processAnalysisResult(summary, classification, document.content),
        entities: entities.status === 'fulfilled' ? entities.value : [],
        sentiment: sentiment.status === 'fulfilled' ? sentiment.value : this.getDefaultSentiment(),
        classification: classification.status === 'fulfilled' ? classification.value : this.getDefaultClassification(),
        riskAssessment: riskAssessment.status === 'fulfilled' ? riskAssessment.value : this.getDefaultRiskAssessment(),
        keyPhrases: keyPhrases.status === 'fulfilled' ? keyPhrases.value : [],
        structuralAnalysis: structuralAnalysis.status === 'fulfilled' ? structuralAnalysis.value : this.getDefaultStructuralAnalysis(),
        userIntent: userIntent.status === 'fulfilled' ? userIntent.value : this.getDefaultUserIntent(),
        processingMetadata: {
          processingTime,
          modelsUsed: Array.from(this.models.keys()),
          chunks: chunks.length,
          confidence: this.calculateOverallConfidence([
            entities, sentiment, classification, summary, keyPhrases, structuralAnalysis, userIntent, riskAssessment
          ]),
          warnings: this.collectWarnings([
            entities, sentiment, classification, summary, keyPhrases, structuralAnalysis, userIntent, riskAssessment
          ])
        }
      };

      console.log(`✅ Document analysis completed in ${processingTime}ms`);
      return result;

    } catch (error) {
      console.error('❌ Document analysis failed:', error);
      throw error;
    }
  }

  private async extractEntities(text: string): Promise<LegalEntity[]> {
    try {
      const nerModel = this.models.get('ner');
      if (!nerModel) throw new Error('NER model not loaded');

      const entities = await nerModel(text);
      const legalEntities: LegalEntity[] = [];

      for (const entity of entities) {
        if (entity.score >= this.config.thresholds.entityMin) {
          const legalCategory = this.mapToLegalCategory(entity.entity);
          const context = this.extractEntityContext(text, entity.start, entity.end);

          legalEntities.push({
            text: entity.word,
            label: entity.entity,
            confidence: entity.score,
            start: entity.start,
            end: entity.end,
            context,
            legalCategory
          });
        }
      }

      // Add legal-specific entity extraction
      const legalSpecificEntities = await this.extractLegalSpecificEntities(text);
      legalEntities.push(...legalSpecificEntities);

      return legalEntities;
    } catch (error) {
      console.error('Entity extraction failed:', error);
      return [];
    }
  }

  private async analyzeSentiment(text: string): Promise<any> {
    try {
      const sentimentModel = this.models.get('sentiment');
      if (!sentimentModel) throw new Error('Sentiment model not loaded');

      const result = await sentimentModel(text);
      const legalTone = this.analyzeLegalTone(text);

      return {
        overall: result[0].label === 'POSITIVE' ? result[0].score : -result[0].score,
        confidence: result[0].score,
        distribution: this.calculateSentimentDistribution(result),
        legalTone
      };
    } catch (error) {
      console.error('Sentiment analysis failed:', error);
      return this.getDefaultSentiment();
    }
  }

  private async classifyDocument(text: string): Promise<any> {
    try {
      const classificationModel = this.models.get('classification');
      if (!classificationModel) throw new Error('Classification model not loaded');

      const legalCategories = [
        'Contract Law', 'Tort Law', 'Criminal Law', 'Constitutional Law',
        'Administrative Law', 'Intellectual Property', 'Employment Law',
        'Real Estate Law', 'Corporate Law', 'Family Law'
      ];

      const result = await classificationModel(text, legalCategories);
      
      return {
        primaryCategory: result.labels[0],
        confidence: result.scores[0],
        secondaryCategories: result.labels.slice(1, 3).map((label, i) => ({
          category: label,
          confidence: result.scores[i + 1]
        })),
        practiceAreas: this.mapToPracticeAreas(result.labels[0])
      };
    } catch (error) {
      console.error('Document classification failed:', error);
      return this.getDefaultClassification();
    }
  }

  private async generateSummary(text: string): Promise<string> {
    try {
      const gemmaModel = this.models.get('gemma2b');
      if (!gemmaModel) throw new Error('Gemma model not loaded');

      const prompt = await this.summaryTemplate.format({ document: text.substring(0, 2000) });
      const result = await gemmaModel(prompt, {
        max_length: 500,
        temperature: 0.3,
        do_sample: false
      });

      return result[0].generated_text || 'Summary generation failed';
    } catch (error) {
      console.error('Summary generation failed:', error);
      return 'Unable to generate summary';
    }
  }

  private async extractKeyPhrases(text: string): Promise<Array<{ phrase: string; importance: number; category: string; context: string }>> {
    try {
      // Use Legal-BERT for key phrase extraction
      const legalBertModel = this.models.get('legalBert');
      if (!legalBertModel) throw new Error('Legal-BERT model not loaded');

      // Extract noun phrases and legal terms
      const phrases = this.extractNounPhrases(text);
      const legalTerms = this.extractLegalTerms(text);
      
      const keyPhrases = [];

      // Score and categorize phrases
      for (const phrase of [...phrases, ...legalTerms]) {
        const importance = this.calculatePhraseImportance(phrase, text);
        const category = this.categorizeLegalPhrase(phrase);
        const context = this.extractPhraseContext(text, phrase);

        if (importance > 0.5) {
          keyPhrases.push({
            phrase,
            importance,
            category,
            context
          });
        }
      }

      return keyPhrases.sort((a, b) => b.importance - a.importance).slice(0, 20);
    } catch (error) {
      console.error('Key phrase extraction failed:', error);
      return [];
    }
  }

  private async analyzeStructure(text: string): Promise<any> {
    try {
      const sections = this.extractSections(text);
      const citations = this.extractCitations(text);
      const definitions = this.extractDefinitions(text);

      return {
        sections: sections.map(section => ({
          title: section.title,
          content: section.content.substring(0, 200),
          importance: this.calculateSectionImportance(section)
        })),
        citations: citations.map(citation => ({
          text: citation,
          type: this.classifyCitation(citation),
          verified: false // Would need external verification
        })),
        definitions: definitions.map(def => ({
          term: def.term,
          definition: def.definition,
          source: 'document'
        }))
      };
    } catch (error) {
      console.error('Structural analysis failed:', error);
      return this.getDefaultStructuralAnalysis();
    }
  }

  private async detectUserIntent(text: string): Promise<any> {
    try {
      const intentCategories = [
        'contract_review', 'legal_research', 'compliance_check',
        'risk_assessment', 'document_drafting', 'case_analysis',
        'regulatory_guidance', 'litigation_support'
      ];

      const classificationModel = this.models.get('classification');
      if (!classificationModel) throw new Error('Classification model not loaded');

      const result = await classificationModel(text, intentCategories);
      
      const urgency = this.calculateUrgency(text);
      const suggestedActions = this.generateActionSuggestions(result.labels[0]);

      return {
        primaryIntent: result.labels[0],
        confidence: result.scores[0],
        intentCategories: result.labels.slice(0, 3),
        suggestedActions,
        urgency
      };
    } catch (error) {
      console.error('Intent detection failed:', error);
      return this.getDefaultUserIntent();
    }
  }

  private async assessRisks(text: string): Promise<any> {
    try {
      const riskFactors = this.identifyRiskFactors(text);
      const complianceIssues = this.identifyComplianceIssues(text);
      const overallRisk = this.calculateOverallRisk(riskFactors);
      const recommendations = this.generateRiskRecommendations(riskFactors, complianceIssues);

      return {
        overallRisk,
        riskFactors,
        complianceIssues,
        recommendations
      };
    } catch (error) {
      console.error('Risk assessment failed:', error);
      return this.getDefaultRiskAssessment();
    }
  }

  // Utility methods
  private mapToLegalCategory(entityType: string): LegalEntity['legalCategory'] {
    const mapping = {
      'PERSON': 'person',
      'ORG': 'organization',
      'GPE': 'location',
      'LAW': 'statute',
      'DATE': 'date',
      'MONEY': 'money'
    };
    return mapping[entityType] || 'case_law';
  }

  private extractEntityContext(text: string, start: number, end: number): string {
    const contextRadius = 50;
    const contextStart = Math.max(0, start - contextRadius);
    const contextEnd = Math.min(text.length, end + contextRadius);
    return text.substring(contextStart, contextEnd);
  }

  private async extractLegalSpecificEntities(text: string): Promise<LegalEntity[]> {
    const entities: LegalEntity[] = [];
    
    // Legal case citations
    const casePattern = /\b[A-Z][a-z]+ v\. [A-Z][a-z]+\b/g;
    let match;
    while ((match = casePattern.exec(text)) !== null) {
      entities.push({
        text: match[0],
        label: 'CASE',
        confidence: 0.9,
        start: match.index,
        end: match.index + match[0].length,
        context: this.extractEntityContext(text, match.index, match.index + match[0].length),
        legalCategory: 'case_law'
      });
    }

    // Statutes
    const statutePattern = /\b\d+\s+(U\.S\.C\.|USC)\s+§\s*\d+\b/g;
    while ((match = statutePattern.exec(text)) !== null) {
      entities.push({
        text: match[0],
        label: 'STATUTE',
        confidence: 0.95,
        start: match.index,
        end: match.index + match[0].length,
        context: this.extractEntityContext(text, match.index, match.index + match[0].length),
        legalCategory: 'statute'
      });
    }

    return entities;
  }

  private analyzeLegalTone(text: string): 'formal' | 'informal' | 'technical' | 'persuasive' {
    const formalIndicators = ['hereby', 'whereas', 'notwithstanding', 'pursuant to'];
    const technicalIndicators = ['§', 'subsection', 'regulation', 'statute'];
    const persuasiveIndicators = ['therefore', 'accordingly', 'it is clear that', 'undoubtedly'];
    
    const formalCount = formalIndicators.filter(word => text.toLowerCase().includes(word)).length;
    const technicalCount = technicalIndicators.filter(word => text.toLowerCase().includes(word)).length;
    const persuasiveCount = persuasiveIndicators.filter(word => text.toLowerCase().includes(word)).length;
    
    if (technicalCount > formalCount && technicalCount > persuasiveCount) return 'technical';
    if (persuasiveCount > formalCount) return 'persuasive';
    if (formalCount > 0) return 'formal';
    return 'informal';
  }

  private calculateSentimentDistribution(result: any): { positive: number; neutral: number; negative: number } {
    // Simplified distribution calculation
    const positive = result[0].label === 'POSITIVE' ? result[0].score : 1 - result[0].score;
    const negative = 1 - positive;
    const neutral = Math.abs(positive - negative) < 0.2 ? 0.3 : 0.1;
    
    return { positive, neutral, negative };
  }

  private mapToPracticeAreas(category: string): string[] {
    const mapping = {
      'Contract Law': ['Commercial Law', 'Business Law'],
      'Tort Law': ['Personal Injury', 'Product Liability'],
      'Criminal Law': ['White Collar Crime', 'Criminal Defense'],
      'Corporate Law': ['Securities Law', 'Mergers & Acquisitions'],
      'Employment Law': ['Labor Relations', 'Workplace Rights']
    };
    return mapping[category] || [category];
  }

  private extractNounPhrases(text: string): string[] {
    // Simplified noun phrase extraction
    const nounPhrasePattern = /\b(?:[A-Z][a-z]*\s+)*[A-Z][a-z]*\b/g;
    return Array.from(text.matchAll(nounPhrasePattern)).map(match => match[0]);
  }

  private extractLegalTerms(text: string): string[] {
    const legalTerms = [
      'consideration', 'breach', 'damages', 'liability', 'indemnification',
      'warranty', 'representation', 'covenant', 'force majeure', 'arbitration',
      'jurisdiction', 'venue', 'governing law', 'severability', 'assignment'
    ];
    
    return legalTerms.filter(term => 
      text.toLowerCase().includes(term.toLowerCase())
    );
  }

  private calculatePhraseImportance(phrase: string, text: string): number {
    const frequency = (text.toLowerCase().match(new RegExp(phrase.toLowerCase(), 'g')) || []).length;
    const position = text.toLowerCase().indexOf(phrase.toLowerCase()) / text.length;
    const length = phrase.length;
    
    // Combine frequency, position, and length for importance score
    return (frequency * 0.4) + ((1 - position) * 0.3) + (Math.min(length / 20, 1) * 0.3);
  }

  private categorizeLegalPhrase(phrase: string): string {
    const categories = {
      'contractual': ['agreement', 'contract', 'terms', 'conditions'],
      'liability': ['damages', 'liability', 'responsibility', 'fault'],
      'procedural': ['jurisdiction', 'venue', 'procedure', 'process'],
      'regulatory': ['compliance', 'regulation', 'statute', 'law']
    };
    
    for (const [category, keywords] of Object.entries(categories)) {
      if (keywords.some(keyword => phrase.toLowerCase().includes(keyword))) {
        return category;
      }
    }
    
    return 'general';
  }

  private extractPhraseContext(text: string, phrase: string): string {
    const index = text.toLowerCase().indexOf(phrase.toLowerCase());
    if (index === -1) return '';
    
    const start = Math.max(0, index - 30);
    const end = Math.min(text.length, index + phrase.length + 30);
    return text.substring(start, end);
  }

  private extractSections(text: string): Array<{ title: string; content: string }> {
    const sections = [];
    const sectionPattern = /^([A-Z][A-Z\s]+)$/gm;
    const matches = Array.from(text.matchAll(sectionPattern));
    
    for (let i = 0; i < matches.length; i++) {
      const title = matches[i][1].trim();
      const start = matches[i].index + matches[i][0].length;
      const end = i < matches.length - 1 ? matches[i + 1].index : text.length;
      const content = text.substring(start, end).trim();
      
      sections.push({ title, content });
    }
    
    return sections;
  }

  private extractCitations(text: string): string[] {
    const citations = [];
    
    // Case citations
    const casePattern = /\b[A-Z][a-z]+ v\. [A-Z][a-z]+,?\s*\d+\s+[A-Z][a-z\.]+\s+\d+/g;
    citations.push(...Array.from(text.matchAll(casePattern)).map(match => match[0]));
    
    // Statute citations
    const statutePattern = /\b\d+\s+(U\.S\.C\.|USC|CFR)\s+§\s*\d+/g;
    citations.push(...Array.from(text.matchAll(statutePattern)).map(match => match[0]));
    
    return citations;
  }

  private extractDefinitions(text: string): Array<{ term: string; definition: string }> {
    const definitions = [];
    const definitionPattern = /"([^"]+)"\s+means\s+([^.]+)/g;
    
    let match;
    while ((match = definitionPattern.exec(text)) !== null) {
      definitions.push({
        term: match[1],
        definition: match[2]
      });
    }
    
    return definitions;
  }

  private calculateSectionImportance(section: { title: string; content: string }): number {
    const importantTitles = ['definitions', 'terms', 'conditions', 'obligations', 'liability'];
    const titleImportance = importantTitles.some(title => 
      section.title.toLowerCase().includes(title)
    ) ? 0.8 : 0.5;
    
    const lengthImportance = Math.min(section.content.length / 1000, 1);
    
    return (titleImportance + lengthImportance) / 2;
  }

  private classifyCitation(citation: string): 'case' | 'statute' | 'regulation' {
    if (citation.includes('v.')) return 'case';
    if (citation.includes('CFR')) return 'regulation';
    return 'statute';
  }

  private calculateUrgency(text: string): 'low' | 'medium' | 'high' {
    const urgentKeywords = ['urgent', 'immediate', 'deadline', 'expires', 'emergency'];
    const urgentCount = urgentKeywords.filter(keyword => 
      text.toLowerCase().includes(keyword)
    ).length;
    
    if (urgentCount >= 2) return 'high';
    if (urgentCount >= 1) return 'medium';
    return 'low';
  }

  private generateActionSuggestions(intent: string): string[] {
    const suggestions = {
      'contract_review': ['Review key terms', 'Check for compliance', 'Identify risks'],
      'legal_research': ['Search precedents', 'Analyze case law', 'Review statutes'],
      'compliance_check': ['Verify regulations', 'Check requirements', 'Update policies'],
      'risk_assessment': ['Identify risks', 'Evaluate impact', 'Develop mitigation'],
      'document_drafting': ['Use templates', 'Include standard clauses', 'Review format'],
      'case_analysis': ['Review facts', 'Analyze legal issues', 'Research precedents']
    };
    
    return suggestions[intent] || ['Consult legal professional'];
  }

  private identifyRiskFactors(text: string): Array<{ factor: string; severity: number; description: string }> {
    const riskPatterns = {
      'Unlimited liability': { pattern: /unlimited.{0,20}liability/i, severity: 0.9 },
      'No limitation of damages': { pattern: /no.{0,20}limitation.{0,20}damages/i, severity: 0.8 },
      'Personal guarantee': { pattern: /personal.{0,20}guarantee/i, severity: 0.7 },
      'Broad indemnification': { pattern: /indemnif.{0,50}harmless/i, severity: 0.6 }
    };
    
    const riskFactors = [];
    
    for (const [factor, config] of Object.entries(riskPatterns)) {
      if (config.pattern.test(text)) {
        riskFactors.push({
          factor,
          severity: config.severity,
          description: `Potential ${factor.toLowerCase()} clause identified`
        });
      }
    }
    
    return riskFactors;
  }

  private identifyComplianceIssues(text: string): string[] {
    const compliancePatterns = [
      /GDPR/i,
      /CCPA/i,
      /SOX/i,
      /HIPAA/i,
      /PCI.{0,10}DSS/i
    ];
    
    const issues = [];
    
    for (const pattern of compliancePatterns) {
      if (pattern.test(text)) {
        issues.push(`Potential ${pattern.source} compliance requirement`);
      }
    }
    
    return issues;
  }

  private calculateOverallRisk(riskFactors: any[]): 'low' | 'medium' | 'high' {
    if (riskFactors.length === 0) return 'low';
    
    const maxSeverity = Math.max(...riskFactors.map(rf => rf.severity));
    
    if (maxSeverity >= 0.8) return 'high';
    if (maxSeverity >= 0.6) return 'medium';
    return 'low';
  }

  private generateRiskRecommendations(riskFactors: any[], complianceIssues: string[]): string[] {
    const recommendations = [];
    
    if (riskFactors.length > 0) {
      recommendations.push('Review and negotiate risk allocation clauses');
      recommendations.push('Consider liability limitations and caps');
    }
    
    if (complianceIssues.length > 0) {
      recommendations.push('Ensure compliance with applicable regulations');
      recommendations.push('Implement necessary compliance procedures');
    }
    
    recommendations.push('Consult with qualified legal counsel');
    
    return recommendations;
  }

  // Default values for failed analyses
  private getDefaultSentiment(): any {
    return {
      overall: 0,
      confidence: 0,
      distribution: { positive: 0.33, neutral: 0.34, negative: 0.33 },
      legalTone: 'formal'
    };
  }

  private getDefaultClassification(): any {
    return {
      primaryCategory: 'General Legal',
      confidence: 0,
      secondaryCategories: [],
      practiceAreas: ['General Practice']
    };
  }

  private getDefaultRiskAssessment(): any {
    return {
      overallRisk: 'medium',
      riskFactors: [],
      complianceIssues: [],
      recommendations: ['Professional legal review recommended']
    };
  }

  private getDefaultStructuralAnalysis(): any {
    return {
      sections: [],
      citations: [],
      definitions: []
    };
  }

  private getDefaultUserIntent(): any {
    return {
      primaryIntent: 'document_review',
      confidence: 0,
      intentCategories: ['document_review'],
      suggestedActions: ['Review document thoroughly'],
      urgency: 'medium'
    };
  }

  private processAnalysisResult(summary: any, classification: any, content: string): any {
    const summaryText = summary.status === 'fulfilled' ? summary.value : 'Unable to generate summary';
    const classificationData = classification.status === 'fulfilled' ? classification.value : this.getDefaultClassification();
    
    return {
      summary: summaryText,
      keyTopics: this.extractKeyTopics(content),
      legalCategories: [classificationData.primaryCategory],
      complexity: this.assessComplexity(content),
      readabilityScore: this.calculateReadabilityScore(content)
    };
  }

  private extractKeyTopics(text: string): string[] {
    const topics = ['contract', 'liability', 'damages', 'warranty', 'indemnification', 'termination'];
    return topics.filter(topic => text.toLowerCase().includes(topic));
  }

  private assessComplexity(text: string): 'low' | 'medium' | 'high' {
    const avgSentenceLength = text.split(/[.!?]+/).reduce((sum, sentence) => sum + sentence.split(' ').length, 0) / text.split(/[.!?]+/).length;
    
    if (avgSentenceLength > 25) return 'high';
    if (avgSentenceLength > 15) return 'medium';
    return 'low';
  }

  private calculateReadabilityScore(text: string): number {
    // Simplified Flesch Reading Ease score
    const words = text.split(/\s+/).length;
    const sentences = text.split(/[.!?]+/).length;
    const syllables = this.countSyllables(text);
    
    const score = 206.835 - (1.015 * (words / sentences)) - (84.6 * (syllables / words));
    return Math.max(0, Math.min(100, score));
  }

  private countSyllables(text: string): number {
    // Simplified syllable counting
    return text.toLowerCase().split(/[aeiou]+/).length - 1;
  }

  private calculateOverallConfidence(results: PromiseSettledResult<any>[]): number {
    const successful = results.filter(r => r.status === 'fulfilled').length;
    return successful / results.length;
  }

  private collectWarnings(results: PromiseSettledResult<any>[]): string[] {
    const warnings = [];
    
    results.forEach((result, index) => {
      if (result.status === 'rejected') {
        warnings.push(`Analysis component ${index} failed: ${result.reason}`);
      }
    });
    
    return warnings;
  }

  async healthCheck(): Promise<{ status: string; details: any }> {
    return {
      status: this.initialized ? 'healthy' : 'not initialized',
      details: {
        initialized: this.initialized,
        modelsLoaded: this.models.size,
        availableModels: Array.from(this.models.keys()),
        config: this.config
      }
    };
  }
}

export const semanticAnalysisPipeline = new SemanticAnalysisPipeline();