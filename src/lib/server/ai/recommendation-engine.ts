// @ts-nocheck
/**
 * Advanced Recommendation Engine with Self-Organizing Maps (SOM)
 * Uses Kohonen networks for pattern recognition and intelligent clustering
 */

import { EventEmitter } from 'events';
import { Matrix, Vector } from 'ml-matrix';
import { kmeans } from 'ml-kmeans';
import Fuse from 'fuse.js';

interface SOMNode {
  id: string;
  position: { x: number; y: number };
  weights: number[];
  patterns: RecommendationPattern[];
  activationCount: number;
  lastActivation: number;
}

interface RecommendationPattern {
  id: string;
  userId: string;
  sessionId: string;
  features: number[];
  metadata: {
    documentType: string;
    analysisType: string;
    confidence: number;
    processingTime: number;
    satisfaction: number;
    tags: string[];
    timestamp: number;
  };
}

interface RecommendationResult {
  type: 'model' | 'feature' | 'workflow' | 'content' | 'optimization';
  title: string;
  description: string;
  confidence: number;
  priority: 'low' | 'medium' | 'high' | 'critical';
  reasoning: string;
  actionableSteps: string[];
  expectedImprovement: number; // 0-1 scale
  category: string;
  tags: string[];
  metadata: any;
}

interface UserContext {
  userId: string;
  role: string;
  experience: string;
  recentPatterns: RecommendationPattern[];
  preferences: any;
  behavioralSignals: {
    averageSessionTime: number;
    preferredComplexity: number;
    modelPreference: string;
    satisfactionTrend: number;
    featureUsage: Map<string, number>;
  };
}

export class RecommendationEngine extends EventEmitter {
  private som: SOMNode[][];
  private patterns: Map<string, RecommendationPattern> = new Map();
  private userContexts: Map<string, UserContext> = new Map();
  private fuse: Fuse<RecommendationResult>;
  private config: {
    somWidth: number;
    somHeight: number;
    learningRate: number;
    neighborhoodRadius: number;
    iterations: number;
    featureDimensions: number;
  };

  constructor(options: {
    somWidth?: number;
    somHeight?: number;
    learningRate?: number;
    neighborhoodRadius?: number;
    iterations?: number;
  } = {}) {
    super();

    this.config = {
      somWidth: options.somWidth || 20,
      somHeight: options.somHeight || 20,
      learningRate: options.learningRate || 0.1,
      neighborhoodRadius: options.neighborhoodRadius || 3.0,
      iterations: options.iterations || 1000,
      featureDimensions: 50 // Number of features we extract from user interactions
    };

    this.initializeSOM();
    this.setupFuseSearch();

    console.log(`🧠 SOM Recommendation Engine initialized (${this.config.somWidth}x${this.config.somHeight})`);
  }

  private initializeSOM(): void {
    this.som = [];
    
    for (let x = 0; x < this.config.somWidth; x++) {
      this.som[x] = [];
      for (let y = 0; y < this.config.somHeight; y++) {
        this.som[x][y] = {
          id: `som_${x}_${y}`,
          position: { x, y },
          weights: this.randomizeWeights(),
          patterns: [],
          activationCount: 0,
          lastActivation: 0
        };
      }
    }
  }

  private randomizeWeights(): number[] {
    const weights = [];
    for (let i = 0; i < this.config.featureDimensions; i++) {
      weights.push(Math.random() * 2 - 1); // Random between -1 and 1
    }
    return weights;
  }

  private setupFuseSearch(): void {
    // Initialize empty Fuse instance - will be populated as recommendations are generated
    this.fuse = new Fuse([], {
      keys: [
        { name: 'title', weight: 0.4 },
        { name: 'description', weight: 0.3 },
        { name: 'category', weight: 0.2 },
        { name: 'tags', weight: 0.1 }
      ],
      threshold: 0.3,
      includeScore: true
    });
  }

  // Main recommendation generation
  public async generateRecommendations(
    userId: string,
    sessionChunks: any[],
    userHistory: any[]
  ): Promise<RecommendationResult[]> {
    console.log(`🎯 Generating recommendations for user ${userId}`);

    // Extract user context and patterns
    const userContext = await this.buildUserContext(userId, userHistory);
    const currentPattern = this.extractPatternFromSession(userId, sessionChunks);

    // Add pattern to SOM and get best matching unit (BMU)
    const bmu = await this.findBestMatchingUnit(currentPattern);
    await this.updateSOM(currentPattern, bmu);

    // Generate different types of recommendations
    const recommendations = [
      ...await this.generateModelRecommendations(userContext, currentPattern, bmu),
      ...await this.generateFeatureRecommendations(userContext, currentPattern, bmu),
      ...await this.generateWorkflowRecommendations(userContext, currentPattern, bmu),
      ...await this.generateContentRecommendations(userContext, currentPattern, bmu),
      ...await this.generateOptimizationRecommendations(userContext, currentPattern, bmu)
    ];

    // Rank recommendations by relevance and confidence
    const rankedRecommendations = this.rankRecommendations(recommendations, userContext);

    // Update Fuse search index
    this.fuse.setCollection(rankedRecommendations);

    console.log(`✨ Generated ${rankedRecommendations.length} recommendations for user ${userId}`);
    return rankedRecommendations;
  }

  // Pattern extraction and feature engineering
  private extractPatternFromSession(userId: string, sessionChunks: any[]): RecommendationPattern {
    const features = new Array(this.config.featureDimensions).fill(0);

    // Feature engineering from session chunks
    sessionChunks.forEach((chunk, index) => {
      // Chunk type features (one-hot encoding)
      if (chunk.type === 'legal-bert') features[0] = 1;
      if (chunk.type === 'local-llm') features[1] = 1;
      if (chunk.type === 'enhanced-rag') features[2] = 1;
      if (chunk.type === 'user-history') features[3] = 1;
      if (chunk.type === 'semantic-tokens') features[4] = 1;

      // Confidence and performance features
      features[5] += chunk.confidence || 0;
      features[6] += chunk.processingTime || 0;

      // Status indicators
      if (chunk.status === 'complete') features[7]++;
      if (chunk.status === 'error') features[8]++;
    });

    // Normalize features
    features[5] /= Math.max(sessionChunks.length, 1); // Average confidence
    features[6] /= Math.max(sessionChunks.length, 1); // Average processing time

    // Add temporal features
    const hour = new Date().getHours();
    features[9] = hour / 24.0; // Time of day (0-1)
    features[10] = new Date().getDay() / 7.0; // Day of week (0-1)

    // Add complexity indicators
    const totalChunks = sessionChunks.length;
    features[11] = Math.min(totalChunks / 10, 1.0); // Session complexity (0-1)

    // Fill remaining features with contextual data
    for (let i = 12; i < this.config.featureDimensions; i++) {
      features[i] = Math.random() * 0.1; // Small random noise for diversity
    }

    return {
      id: this.generateId(),
      userId,
      sessionId: `session_${Date.now()}`,
      features,
      metadata: {
        documentType: 'legal',
        analysisType: 'comprehensive',
        confidence: features[5],
        processingTime: features[6],
        satisfaction: 0.8, // Default, will be updated with feedback
        tags: ['legal-analysis', 'multi-model'],
        timestamp: Date.now()
      }
    };
  }

  private async buildUserContext(userId: string, userHistory: any[]): Promise<UserContext> {
    const existingContext = this.userContexts.get(userId);
    
    const context: UserContext = {
      userId,
      role: existingContext?.role || 'user',
      experience: existingContext?.experience || 'intermediate',
      recentPatterns: this.getRecentPatterns(userId, 10),
      preferences: existingContext?.preferences || {},
      behavioralSignals: {
        averageSessionTime: this.calculateAverageSessionTime(userHistory),
        preferredComplexity: this.calculatePreferredComplexity(userHistory),
        modelPreference: this.calculateModelPreference(userHistory),
        satisfactionTrend: this.calculateSatisfactionTrend(userHistory),
        featureUsage: this.calculateFeatureUsage(userHistory)
      }
    };

    this.userContexts.set(userId, context);
    return context;
  }

  // SOM operations
  private async findBestMatchingUnit(pattern: RecommendationPattern): Promise<SOMNode> {
    let bestNode = this.som[0][0];
    let minDistance = this.euclideanDistance(pattern.features, bestNode.weights);

    for (let x = 0; x < this.config.somWidth; x++) {
      for (let y = 0; y < this.config.somHeight; y++) {
        const node = this.som[x][y];
        const distance = this.euclideanDistance(pattern.features, node.weights);
        
        if (distance < minDistance) {
          minDistance = distance;
          bestNode = node;
        }
      }
    }

    bestNode.activationCount++;
    bestNode.lastActivation = Date.now();
    return bestNode;
  }

  private async updateSOM(pattern: RecommendationPattern, bmu: SOMNode): Promise<void> {
    const currentIteration = bmu.activationCount;
    const learningRate = this.config.learningRate * Math.exp(-currentIteration / 1000);
    const radius = this.config.neighborhoodRadius * Math.exp(-currentIteration / 1000);

    for (let x = 0; x < this.config.somWidth; x++) {
      for (let y = 0; y < this.config.somHeight; y++) {
        const node = this.som[x][y];
        const distance = this.manhattanDistance(
          { x: bmu.position.x, y: bmu.position.y },
          { x, y }
        );

        if (distance <= radius) {
          const influence = Math.exp(-(distance * distance) / (2 * radius * radius));
          
          for (let i = 0; i < this.config.featureDimensions; i++) {
            const delta = learningRate * influence * (pattern.features[i] - node.weights[i]);
            node.weights[i] += delta;
          }
        }
      }
    }

    // Add pattern to BMU
    bmu.patterns.push(pattern);
    this.patterns.set(pattern.id, pattern);
  }

  // Recommendation generators for different categories
  private async generateModelRecommendations(
    context: UserContext,
    pattern: RecommendationPattern,
    bmu: SOMNode
  ): Promise<RecommendationResult[]> {
    const recommendations: RecommendationResult[] = [];

    // Model performance analysis
    const avgConfidence = pattern.metadata.confidence;
    if (avgConfidence < 0.7) {
      recommendations.push({
        type: 'model',
        title: 'Switch to Higher Accuracy Model',
        description: 'Your recent analyses show lower confidence scores. Consider using Legal-BERT for more accurate results.',
        confidence: 0.85,
        priority: 'high',
        reasoning: `Current average confidence: ${(avgConfidence * 100).toFixed(1)}%. Legal-BERT typically achieves 15-20% higher accuracy for legal documents.`,
        actionableSteps: [
          'Go to Model Settings',
          'Select "Legal-BERT" as primary model',
          'Re-run recent analysis for comparison'
        ],
        expectedImprovement: 0.2,
        category: 'Model Optimization',
        tags: ['accuracy', 'legal-bert', 'confidence'],
        metadata: { currentConfidence: avgConfidence, recommendedModel: 'legal-bert' }
      });
    }

    // Multi-model recommendation
    if (context.experience === 'expert' && !this.hasUsedMultiModel(context)) {
      recommendations.push({
        type: 'model',
        title: 'Try Multi-Model Ensemble',
        description: 'Combine multiple AI models for enhanced accuracy and comprehensive analysis.',
        confidence: 0.9,
        priority: 'medium',
        reasoning: 'Expert users benefit from ensemble methods that combine Legal-BERT, local LLMs, and RAG systems.',
        actionableSteps: [
          'Enable "Multi-Model" mode in settings',
          'Configure model weights based on your use case',
          'Monitor ensemble confidence scores'
        ],
        expectedImprovement: 0.25,
        category: 'Advanced Features',
        tags: ['ensemble', 'multi-model', 'expert'],
        metadata: { userExperience: context.experience }
      });
    }

    return recommendations;
  }

  private async generateFeatureRecommendations(
    context: UserContext,
    pattern: RecommendationPattern,
    bmu: SOMNode
  ): Promise<RecommendationResult[]> {
    const recommendations: RecommendationResult[] = [];

    // Feature usage analysis
    const featureUsage = context.behavioralSignals.featureUsage;
    
    if (!featureUsage.has('document-comparison') && context.role === 'prosecutor') {
      recommendations.push({
        type: 'feature',
        title: 'Document Comparison for Legal Cases',
        description: 'Compare legal documents side-by-side to identify similarities, differences, and potential conflicts.',
        confidence: 0.8,
        priority: 'medium',
        reasoning: 'Prosecutors frequently need to compare evidence documents, contracts, and legal precedents.',
        actionableSteps: [
          'Click the "Comparison" tab in the analysis interface',
          'Upload two documents for comparison',
          'Review the generated similarity analysis'
        ],
        expectedImprovement: 0.3,
        category: 'Productivity Features',
        tags: ['comparison', 'prosecutor', 'legal-analysis'],
        metadata: { userRole: context.role }
      });
    }

    if (!featureUsage.has('custom-analysis') && context.experience !== 'beginner') {
      recommendations.push({
        type: 'feature',
        title: 'Custom Analysis Templates',
        description: 'Create and save custom analysis prompts for your specific legal specialization.',
        confidence: 0.75,
        priority: 'low',
        reasoning: 'Experienced users can create specialized templates for contract analysis, compliance checks, or risk assessments.',
        actionableSteps: [
          'Navigate to Custom Analysis tab',
          'Try the pre-built templates',
          'Create your own specialized prompts'
        ],
        expectedImprovement: 0.2,
        category: 'Customization',
        tags: ['templates', 'custom-prompts', 'specialization'],
        metadata: { userExperience: context.experience }
      });
    }

    return recommendations;
  }

  private async generateWorkflowRecommendations(
    context: UserContext,
    pattern: RecommendationPattern,
    bmu: SOMNode
  ): Promise<RecommendationResult[]> {
    const recommendations: RecommendationResult[] = [];

    // Session length optimization
    if (context.behavioralSignals.averageSessionTime > 45 * 60 * 1000) { // 45 minutes
      recommendations.push({
        type: 'workflow',
        title: 'Optimize Long Analysis Sessions',
        description: 'Break down lengthy analysis sessions into focused, efficient workflows.',
        confidence: 0.7,
        priority: 'medium',
        reasoning: `Your average session time is ${(context.behavioralSignals.averageSessionTime / 60000).toFixed(1)} minutes. Shorter, focused sessions often yield better results.`,
        actionableSteps: [
          'Use batch processing for multiple documents',
          'Set time limits for individual analyses',
          'Take breaks between complex analyses'
        ],
        expectedImprovement: 0.25,
        category: 'Workflow Optimization',
        tags: ['productivity', 'session-management', 'efficiency'],
        metadata: { currentSessionTime: context.behavioralSignals.averageSessionTime }
      });
    }

    // Automation suggestions
    if (this.hasRepetitivePatterns(bmu)) {
      recommendations.push({
        type: 'workflow',
        title: 'Automate Repetitive Analysis',
        description: 'Set up automated workflows for commonly performed analysis types.',
        confidence: 0.85,
        priority: 'high',
        reasoning: 'Detected repetitive analysis patterns that could benefit from automation.',
        actionableSteps: [
          'Review your common analysis types',
          'Set up automated processing rules',
          'Configure notification preferences'
        ],
        expectedImprovement: 0.4,
        category: 'Automation',
        tags: ['automation', 'repetitive-tasks', 'efficiency'],
        metadata: { detectedPatterns: bmu.patterns.length }
      });
    }

    return recommendations;
  }

  private async generateContentRecommendations(
    context: UserContext,
    pattern: RecommendationPattern,
    bmu: SOMNode
  ): Promise<RecommendationResult[]> {
    const recommendations: RecommendationResult[] = [];

    // Content quality suggestions
    if (pattern.metadata.confidence < 0.6) {
      recommendations.push({
        type: 'content',
        title: 'Improve Document Quality for Better Analysis',
        description: 'Enhance document preparation to achieve higher analysis confidence.',
        confidence: 0.8,
        priority: 'medium',
        reasoning: 'Low confidence scores often indicate issues with document quality, formatting, or content clarity.',
        actionableSteps: [
          'Ensure documents are high-quality scans or native PDFs',
          'Remove unnecessary headers, footers, or watermarks',
          'Consider OCR processing for scanned documents'
        ],
        expectedImprovement: 0.3,
        category: 'Content Quality',
        tags: ['document-quality', 'confidence', 'preprocessing'],
        metadata: { currentConfidence: pattern.metadata.confidence }
      });
    }

    // Specialized content recommendations based on patterns
    const similarPatterns = bmu.patterns.filter(p => p.userId !== context.userId);
    if (similarPatterns.length > 5) {
      recommendations.push({
        type: 'content',
        title: 'Explore Similar Legal Cases',
        description: 'Other users with similar analysis patterns have worked on related legal cases.',
        confidence: 0.65,
        priority: 'low',
        reasoning: `Found ${similarPatterns.length} similar analysis patterns from other users.`,
        actionableSteps: [
          'Browse recommended case studies',
          'Review similar document types',
          'Connect with users working on related matters'
        ],
        expectedImprovement: 0.15,
        category: 'Content Discovery',
        tags: ['similar-cases', 'collaboration', 'knowledge-sharing'],
        metadata: { similarPatternCount: similarPatterns.length }
      });
    }

    return recommendations;
  }

  private async generateOptimizationRecommendations(
    context: UserContext,
    pattern: RecommendationPattern,
    bmu: SOMNode
  ): Promise<RecommendationResult[]> {
    const recommendations: RecommendationResult[] = [];

    // Performance optimization
    if (pattern.metadata.processingTime > 30000) { // 30 seconds
      recommendations.push({
        type: 'optimization',
        title: 'Optimize Processing Performance',
        description: 'Improve analysis speed through configuration optimizations.',
        confidence: 0.75,
        priority: 'medium',
        reasoning: `Current processing time: ${(pattern.metadata.processingTime / 1000).toFixed(1)}s. Several optimizations can improve speed.`,
        actionableSteps: [
          'Enable parallel processing for multiple documents',
          'Adjust model complexity settings',
          'Consider using faster models for initial analysis'
        ],
        expectedImprovement: 0.35,
        category: 'Performance',
        tags: ['speed', 'optimization', 'configuration'],
        metadata: { currentProcessingTime: pattern.metadata.processingTime }
      });
    }

    // Resource usage optimization
    if (context.behavioralSignals.featureUsage.size > 8) {
      recommendations.push({
        type: 'optimization',
        title: 'Streamline Feature Usage',
        description: 'Focus on core features that provide the most value for your workflow.',
        confidence: 0.6,
        priority: 'low',
        reasoning: 'Using many features simultaneously can impact performance and create complexity.',
        actionableSteps: [
          'Review feature usage analytics',
          'Identify your most valuable features',
          'Customize interface to highlight key tools'
        ],
        expectedImprovement: 0.2,
        category: 'User Experience',
        tags: ['streamlining', 'focus', 'customization'],
        metadata: { featureCount: context.behavioralSignals.featureUsage.size }
      });
    }

    return recommendations;
  }

  // Ranking and relevance
  private rankRecommendations(
    recommendations: RecommendationResult[],
    context: UserContext
  ): RecommendationResult[] {
    return recommendations
      .map(rec => ({
        ...rec,
        relevanceScore: this.calculateRelevanceScore(rec, context)
      }))
      .sort((a, b) => {
        // Sort by priority first, then by relevance and confidence
        const priorityOrder = { critical: 4, high: 3, medium: 2, low: 1 };
        const priorityDiff = priorityOrder[b.priority] - priorityOrder[a.priority];
        
        if (priorityDiff !== 0) return priorityDiff;
        
        return (b.relevanceScore * b.confidence) - (a.relevanceScore * a.confidence);
      })
      .slice(0, 10); // Return top 10 recommendations
  }

  private calculateRelevanceScore(rec: RecommendationResult, context: UserContext): number {
    let score = 0.5; // Base score

    // Role-based relevance
    if (context.role === 'prosecutor' && rec.tags.includes('prosecutor')) score += 0.2;
    if (context.role === 'detective' && rec.tags.includes('detective')) score += 0.2;

    // Experience-based relevance
    if (context.experience === 'expert' && rec.tags.includes('advanced')) score += 0.15;
    if (context.experience === 'beginner' && rec.tags.includes('beginner')) score += 0.15;

    // Behavioral signal alignment
    if (rec.type === 'optimization' && context.behavioralSignals.averageSessionTime > 30 * 60 * 1000) {
      score += 0.1;
    }

    // Expected improvement weight
    score += rec.expectedImprovement * 0.2;

    return Math.min(score, 1.0);
  }

  // Utility methods
  private euclideanDistance(a: number[], b: number[]): number {
    if (a.length !== b.length) throw new Error('Vector dimensions must match');
    
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
      sum += Math.pow(a[i] - b[i], 2);
    }
    return Math.sqrt(sum);
  }

  private manhattanDistance(a: { x: number; y: number }, b: { x: number; y: number }): number {
    return Math.abs(a.x - b.x) + Math.abs(a.y - b.y);
  }

  private getRecentPatterns(userId: string, limit: number): RecommendationPattern[] {
    return Array.from(this.patterns.values())
      .filter(p => p.userId === userId)
      .sort((a, b) => b.metadata.timestamp - a.metadata.timestamp)
      .slice(0, limit);
  }

  // Analytics calculations
  private calculateAverageSessionTime(userHistory: any[]): number {
    const sessionTimes = userHistory
      .filter(h => h.eventType === 'session-complete')
      .map(h => h.data.duration || 0);
    
    return sessionTimes.length > 0 
      ? sessionTimes.reduce((sum, time) => sum + time, 0) / sessionTimes.length 
      : 15 * 60 * 1000; // Default 15 minutes
  }

  private calculatePreferredComplexity(userHistory: any[]): number {
    // Analyze user's tendency toward simple vs complex analyses
    const complexityScores = userHistory
      .filter(h => h.data.analysisType)
      .map(h => {
        switch (h.data.analysisType) {
          case 'quick': return 0.2;
          case 'standard': return 0.5;
          case 'detailed': return 0.8;
          case 'comprehensive': return 1.0;
          default: return 0.5;
        }
      });

    return complexityScores.length > 0
      ? complexityScores.reduce((sum, score) => sum + score, 0) / complexityScores.length
      : 0.5;
  }

  private calculateModelPreference(userHistory: any[]): string {
    const modelUsage = new Map<string, number>();
    
    userHistory
      .filter(h => h.data.modelUsed)
      .forEach(h => {
        const model = h.data.modelUsed;
        modelUsage.set(model, (modelUsage.get(model) || 0) + 1);
      });

    if (modelUsage.size === 0) return 'auto';

    return Array.from(modelUsage.entries())
      .sort(([,a], [,b]) => b - a)[0][0];
  }

  private calculateSatisfactionTrend(userHistory: any[]): number {
    const satisfactionScores = userHistory
      .filter(h => h.data.resultSatisfaction)
      .map(h => h.data.resultSatisfaction)
      .slice(-10); // Last 10 ratings

    return satisfactionScores.length > 0
      ? satisfactionScores.reduce((sum, score) => sum + score, 0) / satisfactionScores.length
      : 3.0; // Default neutral
  }

  private calculateFeatureUsage(userHistory: any[]): Map<string, number> {
    const usage = new Map<string, number>();
    
    userHistory.forEach(h => {
      if (h.interactionType) {
        usage.set(h.interactionType, (usage.get(h.interactionType) || 0) + 1);
      }
    });

    return usage;
  }

  private hasUsedMultiModel(context: UserContext): boolean {
    return context.behavioralSignals.featureUsage.has('multi-model') ||
           context.behavioralSignals.modelPreference === 'ensemble';
  }

  private hasRepetitivePatterns(bmu: SOMNode): boolean {
    const patternTypes = bmu.patterns.map(p => p.metadata.analysisType);
    const uniqueTypes = new Set(patternTypes);
    
    // If there are many patterns but few unique types, it's repetitive
    return bmu.patterns.length > 5 && uniqueTypes.size < 3;
  }

  private generateId(): string {
    return `pattern_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  // Public methods for external updates
  public async updateFromSession(session: any, synthesizedResult: any): Promise<void> {
    const pattern = this.extractPatternFromSession(session.userId, session.chunks);
    
    // Update pattern with results
    pattern.metadata.confidence = synthesizedResult.overallConfidence || 0.5;
    pattern.metadata.satisfaction = 4.0; // Default, will be updated with user feedback

    const bmu = await this.findBestMatchingUnit(pattern);
    await this.updateSOM(pattern, bmu);

    console.log(`📈 Updated SOM with session data for user ${session.userId}`);
  }

  public async searchRecommendations(query: string): Promise<RecommendationResult[]> {
    const results = this.fuse.search(query);
    return results.map(result => ({
      ...result.item,
      searchScore: 1 - (result.score || 0)
    }));
  }

  public getSOMVisualization(): any {
    // Return SOM state for visualization
    return {
      dimensions: { width: this.config.somWidth, height: this.config.somHeight },
      nodes: this.som.flat().map(node => ({
        id: node.id,
        position: node.position,
        activationCount: node.activationCount,
        patternCount: node.patterns.length,
        lastActivation: node.lastActivation
      }))
    };
  }

  public getRecommendationStats(): any {
    return {
      totalPatterns: this.patterns.size,
      activeNodes: this.som.flat().filter(node => node.patterns.length > 0).length,
      totalNodes: this.config.somWidth * this.config.somHeight,
      userContexts: this.userContexts.size
    };
  }

  public async shutdown(): Promise<void> {
    console.log('🔄 Shutting down Recommendation Engine...');
    
    // Save SOM state if needed
    // Clear maps and arrays
    this.patterns.clear();
    this.userContexts.clear();
    
    console.log('✅ Recommendation Engine shutdown complete');
  }
}