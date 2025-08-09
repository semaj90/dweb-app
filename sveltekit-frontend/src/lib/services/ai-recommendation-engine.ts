// @ts-nocheck
import { writable, derived, get } from 'svelte/store';
import { advancedCache } from './advanced-cache-manager';

export interface RecommendationContext {
  userQuery: string;
  legalDomain: 'contract' | 'litigation' | 'compliance' | 'intellectual_property' | 'employment' | 'general';
  userRole: 'prosecutor' | 'detective' | 'legal_analyst' | 'paralegal' | 'client';
  caseType?: string;
  jurisdiction?: string;
  priority: 'low' | 'medium' | 'high' | 'urgent';
}

export interface Recommendation {
  id: string;
  type: 'suggestion' | 'correction' | 'enhancement' | 'alternative';
  confidence: number;
  content: string;
  reasoning: string;
  legalBasis?: string;
  precedents?: string[];
  riskLevel: 'low' | 'medium' | 'high' | 'critical';
  actionable: boolean;
  estimatedTime?: string;
  requiredExpertise?: string[];
}

export interface DidYouMeanSuggestion {
  originalQuery: string;
  suggestedQuery: string;
  confidence: number;
  reasoning: string;
  improvements: string[];
  legalTerms: string[];
}

export interface QueryEnhancement {
  enhanced_query: string;
  added_context: string[];
  legal_terms_clarified: string[];
  missing_elements: string[];
  confidence_score: number;
}

class AIRecommendationEngine {
  private recommendations = writable<Recommendation[]>([]);
  private queryHistory = writable<string[]>([]);
  private userPatterns = writable<Map<string, number>>(new Map());
  private legalKnowledgeBase = new Map<string, any>();
  
  // Legal domain expertise mapping
  private domainExperts = {
    contract: ['contract_analysis', 'clause_review', 'liability_assessment'],
    litigation: ['case_strategy', 'evidence_analysis', 'precedent_research'],
    compliance: ['regulatory_review', 'risk_assessment', 'audit_preparation'],
    intellectual_property: ['patent_analysis', 'trademark_search', 'copyright_review'],
    employment: ['hr_policy', 'discrimination_analysis', 'termination_review'],
    general: ['legal_research', 'document_review', 'consultation']
  };

  // Common legal term corrections and suggestions
  private legalTermCorrections = new Map([
    ['contract', ['agreement', 'pact', 'deal', 'arrangement', 'covenant']],
    ['liability', ['responsibility', 'obligation', 'accountability', 'culpability']],
    ['plaintiff', ['complainant', 'petitioner', 'claimant', 'appellant']],
    ['defendant', ['respondent', 'accused', 'appellee', 'defending party']],
    ['precedent', ['case law', 'judicial precedent', 'legal authority', 'binding authority']],
    ['jurisdiction', ['legal authority', 'court system', 'territorial authority', 'legal domain']],
    ['statute of limitations', ['time limit', 'limitation period', 'prescriptive period']],
    ['due process', ['legal process', 'procedural fairness', 'constitutional protection']],
    ['habeas corpus', ['legal protection', 'unlawful detention protection', 'court order']],
    ['pro bono', ['free legal service', 'volunteer legal work', 'charity legal work']]
  ]);

  // Advanced pattern matching for legal queries
  private legalPatterns = [
    {
      pattern: /\b(contract|agreement|deal)\b.*\b(review|analyze|check)\b/i,
      domain: 'contract',
      suggestions: ['clause analysis', 'risk assessment', 'compliance check']
    },
    {
      pattern: /\b(liability|responsible|blame|fault)\b/i,
      domain: 'contract',
      suggestions: ['liability assessment', 'indemnification review', 'insurance coverage']
    },
    {
      pattern: /\b(sue|lawsuit|litigation|court)\b/i,
      domain: 'litigation',
      suggestions: ['case strategy', 'evidence gathering', 'settlement options']
    },
    {
      pattern: /\b(compliance|regulation|rule|law)\b/i,
      domain: 'compliance',
      suggestions: ['regulatory review', 'audit preparation', 'policy updates']
    },
    {
      pattern: /\b(patent|trademark|copyright|IP|intellectual property)\b/i,
      domain: 'intellectual_property',
      suggestions: ['IP search', 'infringement analysis', 'protection strategy']
    },
    {
      pattern: /\b(employee|employment|HR|workplace|termination)\b/i,
      domain: 'employment',
      suggestions: ['policy review', 'compliance check', 'risk assessment']
    }
  ];

  constructor() {
    this.initializeLegalKnowledgeBase();
    this.loadUserPatterns();
  }

  // Main recommendation generation
  async generateRecommendations(context: RecommendationContext): Promise<Recommendation[]> {
    const cacheKey = `recommendations_${this.hashContext(context)}`;
    
    // Check cache first
    const cached = await advancedCache.get<Recommendation[]>(cacheKey);
    if (cached) {
      this.recommendations.set(cached);
      return cached;
    }

    const recommendations: Recommendation[] = [];

    // 1. Generate "Did You Mean" suggestions
    const didYouMean = await this.generateDidYouMeanSuggestions(context.userQuery, context.legalDomain);
    if (didYouMean) {
      recommendations.push({
        id: `dym_${Date.now()}`,
        type: 'suggestion',
        confidence: didYouMean.confidence,
        content: `Did you mean: "${didYouMean.suggestedQuery}"?`,
        reasoning: didYouMean.reasoning,
        riskLevel: 'low',
        actionable: true,
        estimatedTime: '1 minute'
      });
    }

    // 2. Generate contextual enhancements
    const enhancements = await this.generateQueryEnhancements(context);
    recommendations.push(...enhancements);

    // 3. Generate domain-specific recommendations
    const domainRecs = await this.generateDomainRecommendations(context);
    recommendations.push(...domainRecs);

    // 4. Generate user pattern-based suggestions
    const patternRecs = await this.generatePatternBasedRecommendations(context);
    recommendations.push(...patternRecs);

    // 5. Risk-based recommendations
    const riskRecs = await this.generateRiskRecommendations(context);
    recommendations.push(...riskRecs);

    // Sort by confidence and priority
    const sortedRecommendations = recommendations
      .sort((a, b) => b.confidence - a.confidence)
      .slice(0, 8); // Limit to top 8 recommendations

    // Cache the results
    await advancedCache.set(cacheKey, sortedRecommendations, {
      priority: 'high',
      ttl: 5 * 60 * 1000, // 5 minutes
      tags: ['recommendations', context.legalDomain, context.userRole]
    });

    this.recommendations.set(sortedRecommendations);
    this.updateUserPatterns(context.userQuery);
    
    return sortedRecommendations;
  }

  // Generate "Did You Mean" suggestions with legal context
  async generateDidYouMeanSuggestions(
    query: string, 
    domain: string
  ): Promise<DidYouMeanSuggestion | null> {
    const cacheKey = `dym_${this.hashString(query)}_${domain}`;
    
    const cached = await advancedCache.get<DidYouMeanSuggestion>(cacheKey);
    if (cached) return cached;

    // Check for common legal term corrections
    const correctedTerms: string[] = [];
    let suggestedQuery = query;
    
    for (const [correctTerm, alternatives] of this.legalTermCorrections.entries()) {
      for (const alternative of alternatives) {
        if (query.toLowerCase().includes(alternative.toLowerCase()) && 
            !query.toLowerCase().includes(correctTerm.toLowerCase())) {
          suggestedQuery = suggestedQuery.replace(
            new RegExp(alternative, 'gi'), 
            correctTerm
          );
          correctedTerms.push(`"${alternative}" → "${correctTerm}"`);
        }
      }
    }

    // Check for incomplete legal phrases
    const improvements: string[] = [];
    const legalTerms: string[] = [];

    if (query.includes('contract') && !query.includes('review') && !query.includes('analyze')) {
      suggestedQuery += ' review and analysis';
      improvements.push('Added specific action: review and analysis');
    }

    if (query.includes('liability') && !query.includes('assessment') && !query.includes('risk')) {
      suggestedQuery += ' risk assessment';
      improvements.push('Added risk assessment context');
    }

    // Add domain-specific enhancements
    const domainTerms = this.getLegalTermsForDomain(domain);
    for (const term of domainTerms) {
      if (!query.toLowerCase().includes(term.toLowerCase())) {
        legalTerms.push(term);
      }
    }

    // Calculate confidence based on improvements made
    let confidence = 0;
    if (correctedTerms.length > 0) confidence += 0.4;
    if (improvements.length > 0) confidence += 0.3;
    if (legalTerms.length > 0) confidence += 0.2;
    if (suggestedQuery.length > query.length * 1.2) confidence += 0.1;

    if (confidence < 0.3 || suggestedQuery === query) {
      return null; // Not confident enough to suggest changes
    }

    const suggestion: DidYouMeanSuggestion = {
      originalQuery: query,
      suggestedQuery,
      confidence: Math.min(confidence, 0.95),
      reasoning: `Enhanced query with ${correctedTerms.length + improvements.length} improvements`,
      improvements: [...correctedTerms, ...improvements],
      legalTerms
    };

    // Cache the suggestion
    await advancedCache.set(cacheKey, suggestion, {
      priority: 'medium',
      ttl: 30 * 60 * 1000, // 30 minutes
      tags: ['did-you-mean', domain]
    });

    return suggestion;
  }

  // Generate query enhancements
  async generateQueryEnhancements(context: RecommendationContext): Promise<Recommendation[]> {
    const recommendations: Recommendation[] = [];
    const query = context.userQuery.toLowerCase();

    // Pattern-based enhancements
    for (const pattern of this.legalPatterns) {
      if (pattern.pattern.test(context.userQuery)) {
        for (const suggestion of pattern.suggestions) {
          recommendations.push({
            id: `enhance_${Date.now()}_${Math.random()}`,
            type: 'enhancement',
            confidence: 0.75,
            content: `Consider adding: ${suggestion}`,
            reasoning: `Pattern match for ${pattern.domain} domain`,
            riskLevel: 'low',
            actionable: true,
            estimatedTime: '2-5 minutes',
            requiredExpertise: [pattern.domain]
          });
        }
        break; // Only match first pattern to avoid overwhelming
      }
    }

    // Missing context recommendations
    if (!query.includes('jurisdiction') && !query.includes('state') && !query.includes('federal')) {
      recommendations.push({
        id: `context_jurisdiction_${Date.now()}`,
        type: 'enhancement',
        confidence: 0.6,
        content: 'Specify jurisdiction (state/federal) for more accurate legal guidance',
        reasoning: 'Legal requirements vary significantly by jurisdiction',
        riskLevel: 'medium',
        actionable: true,
        estimatedTime: '1 minute'
      });
    }

    if (!query.includes('timeline') && !query.includes('deadline') && !query.includes('urgent')) {
      recommendations.push({
        id: `context_timeline_${Date.now()}`,
        type: 'enhancement',
        confidence: 0.5,
        content: 'Consider specifying timeline or urgency level',
        reasoning: 'Urgency affects legal strategy and approach',
        riskLevel: 'low',
        actionable: true,
        estimatedTime: '30 seconds'
      });
    }

    return recommendations;
  }

  // Domain-specific recommendations
  async generateDomainRecommendations(context: RecommendationContext): Promise<Recommendation[]> {
    const recommendations: Recommendation[] = [];
    const expertises = this.domainExperts[context.legalDomain] || [];

    for (const expertise of expertises) {
      const confidence = this.calculateDomainConfidence(context.userQuery, expertise);
      
      if (confidence > 0.4) {
        recommendations.push({
          id: `domain_${expertise}_${Date.now()}`,
          type: 'suggestion',
          confidence,
          content: `Consider ${expertise.replace('_', ' ')} approach`,
          reasoning: `Relevant to ${context.legalDomain} domain`,
          riskLevel: this.assessRiskLevel(expertise),
          actionable: true,
          estimatedTime: this.getEstimatedTime(expertise),
          requiredExpertise: [context.legalDomain, expertise]
        });
      }
    }

    return recommendations.slice(0, 3); // Limit to top 3 domain recommendations
  }

  // Pattern-based recommendations from user history
  async generatePatternBasedRecommendations(context: RecommendationContext): Promise<Recommendation[]> {
    const recommendations: Recommendation[] = [];
    const patterns = get(this.userPatterns);
    const query = context.userQuery.toLowerCase();

    // Find similar queries in history
    const similarQueries = Array.from(patterns.entries())
      .filter(([historicalQuery, frequency]) => 
        frequency > 1 && this.calculateSimilarity(query, historicalQuery) > 0.6
      )
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3);

    for (const [similarQuery, frequency] of similarQueries) {
      recommendations.push({
        id: `pattern_${Date.now()}_${Math.random()}`,
        type: 'suggestion',
        confidence: 0.6 + (frequency * 0.05),
        content: `Based on your history: "${similarQuery}"`,
        reasoning: `Similar to ${frequency} previous queries`,
        riskLevel: 'low',
        actionable: true,
        estimatedTime: '30 seconds'
      });
    }

    return recommendations;
  }

  // Risk-based recommendations
  async generateRiskRecommendations(context: RecommendationContext): Promise<Recommendation[]> {
    const recommendations: Recommendation[] = [];
    const query = context.userQuery.toLowerCase();

    // High-risk indicators
    const riskIndicators = [
      { terms: ['sue', 'lawsuit', 'court', 'litigation'], risk: 'critical', action: 'immediate legal consultation' },
      { terms: ['deadline', 'statute of limitations', 'time limit'], risk: 'high', action: 'urgency assessment' },
      { terms: ['breach', 'violation', 'non-compliance'], risk: 'high', action: 'risk mitigation planning' },
      { terms: ['penalty', 'fine', 'damages', 'liability'], risk: 'medium', action: 'liability assessment' }
    ];

    for (const indicator of riskIndicators) {
      const hasRiskTerms = indicator.terms.some((term: any) => query.includes(term));
      
      if (hasRiskTerms) {
        recommendations.push({
          id: `risk_${indicator.risk}_${Date.now()}`,
          type: 'suggestion',
          confidence: 0.8,
          content: `⚠️ ${indicator.action} recommended`,
          reasoning: `Detected ${indicator.risk} risk indicators`,
          riskLevel: indicator.risk as any,
          actionable: true,
          estimatedTime: indicator.risk === 'critical' ? 'Immediate' : '15-30 minutes',
          requiredExpertise: ['risk_assessment', 'legal_consultation']
        });
        break; // Only show highest priority risk
      }
    }

    return recommendations;
  }

  // Helper methods
  private initializeLegalKnowledgeBase() {
    // Initialize with common legal concepts and their relationships
    this.legalKnowledgeBase.set('contracts', {
      related_terms: ['agreement', 'clause', 'liability', 'breach'],
      common_issues: ['ambiguous terms', 'missing clauses', 'liability allocation'],
      expert_areas: ['contract_law', 'commercial_law', 'liability_law']
    });

    this.legalKnowledgeBase.set('litigation', {
      related_terms: ['lawsuit', 'plaintiff', 'defendant', 'evidence'],
      common_issues: ['jurisdiction', 'standing', 'statute of limitations'],
      expert_areas: ['civil_procedure', 'evidence_law', 'trial_advocacy']
    });
  }

  private async loadUserPatterns() {
    const cached = await advancedCache.get<Map<string, number>>('user_patterns');
    if (cached) {
      this.userPatterns.set(new Map(cached));
    }
  }

  private updateUserPatterns(query: string) {
    this.userPatterns.update((patterns: any) => {
      const current = patterns.get(query) || 0;
      patterns.set(query, current + 1);
      
      // Cache updated patterns
      advancedCache.set('user_patterns', Array.from(patterns.entries()), {
        priority: 'medium',
        ttl: 24 * 60 * 60 * 1000, // 24 hours
        tags: ['user_patterns', 'personalization']
      });
      
      return patterns;
    });
  }

  private getLegalTermsForDomain(domain: string): string[] {
    const domainTerms: { [key: string]: string[] } = {
      contract: ['clause', 'liability', 'breach', 'consideration', 'performance'],
      litigation: ['plaintiff', 'defendant', 'evidence', 'discovery', 'motion'],
      compliance: ['regulation', 'audit', 'violation', 'penalty', 'reporting'],
      intellectual_property: ['patent', 'trademark', 'copyright', 'infringement', 'license'],
      employment: ['termination', 'discrimination', 'harassment', 'wage', 'benefit'],
      general: ['jurisdiction', 'precedent', 'statute', 'common law', 'due process']
    };

    return domainTerms[domain] || domainTerms.general;
  }

  private calculateDomainConfidence(query: string, expertise: string): number {
    const expertiseTerms = expertise.split('_');
    let confidence = 0;

    for (const term of expertiseTerms) {
      if (query.toLowerCase().includes(term.toLowerCase())) {
        confidence += 0.3;
      }
    }

    return Math.min(confidence, 0.9);
  }

  private assessRiskLevel(expertise: string): 'low' | 'medium' | 'high' | 'critical' {
    const highRiskAreas = ['litigation', 'compliance', 'liability'];
    const mediumRiskAreas = ['contract', 'employment', 'audit'];

    if (highRiskAreas.some((area: any) => expertise.includes(area))) {
      return 'high';
    } else if (mediumRiskAreas.some((area: any) => expertise.includes(area))) {
      return 'medium';
    }
    
    return 'low';
  }

  private getEstimatedTime(expertise: string): string {
    const timeMapping: { [key: string]: string } = {
      'contract_analysis': '15-30 minutes',
      'clause_review': '10-20 minutes',
      'liability_assessment': '20-45 minutes',
      'case_strategy': '30-60 minutes',
      'evidence_analysis': '20-40 minutes',
      'precedent_research': '30-90 minutes',
      'regulatory_review': '25-50 minutes',
      'risk_assessment': '15-35 minutes',
      'audit_preparation': '45-120 minutes'
    };

    return timeMapping[expertise] || '10-30 minutes';
  }

  private calculateSimilarity(str1: string, str2: string): number {
    const longer = str1.length > str2.length ? str1 : str2;
    const shorter = str1.length > str2.length ? str2 : str1;
    
    if (longer.length === 0) return 1.0;
    
    const distance = this.levenshteinDistance(longer, shorter);
    return (longer.length - distance) / longer.length;
  }

  private levenshteinDistance(str1: string, str2: string): number {
    const matrix = Array(str2.length + 1).fill(null).map(() => 
      Array(str1.length + 1).fill(null)
    );
    
    for (let i = 0; i <= str1.length; i++) matrix[0][i] = i;
    for (let j = 0; j <= str2.length; j++) matrix[j][0] = j;
    
    for (let j = 1; j <= str2.length; j++) {
      for (let i = 1; i <= str1.length; i++) {
        const indicator = str1[i - 1] === str2[j - 1] ? 0 : 1;
        matrix[j][i] = Math.min(
          matrix[j][i - 1] + 1,
          matrix[j - 1][i] + 1,
          matrix[j - 1][i - 1] + indicator
        );
      }
    }
    
    return matrix[str2.length][str1.length];
  }

  private hashContext(context: RecommendationContext): string {
    return btoa(JSON.stringify({
      query: context.userQuery,
      domain: context.legalDomain,
      role: context.userRole,
      priority: context.priority
    })).replace(/[^a-zA-Z0-9]/g, '').substring(0, 32);
  }

  private hashString(str: string): string {
    return btoa(str).replace(/[^a-zA-Z0-9]/g, '').substring(0, 16);
  }

  // Public API methods
  getRecommendations() {
    return this.recommendations;
  }

  getQueryHistory() {
    return this.queryHistory;
  }

  getUserPatterns() {
    return this.userPatterns;
  }

  async clearRecommendations() {
    this.recommendations.set([]);
    await advancedCache.invalidateByTags(['recommendations']);
  }

  async getRecommendationStats() {
    const patterns = get(this.userPatterns);
    const recommendations = get(this.recommendations);
    
    return {
      totalQueries: Array.from(patterns.values()).reduce((sum, count) => sum + count, 0),
      uniqueQueries: patterns.size,
      activeRecommendations: recommendations.length,
      highConfidenceRecs: recommendations.filter((r: any) => r.confidence > 0.7).length,
      actionableRecs: recommendations.filter((r: any) => r.actionable).length
    };
  }
}

export const aiRecommendationEngine = new AIRecommendationEngine();