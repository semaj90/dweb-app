// Legal Document Service - Clean Architecture Implementation
import type { LegalDocument, LegalAnalysis, AIInsights, ComplianceCheck } from '$lib/types/legal';

export interface DocumentRepository {
  findById(id: string): Promise<LegalDocument>;
  findByCase(caseId: string): Promise<LegalDocument[]>;
  save(document: LegalDocument): Promise<void>;
  search(query: string): Promise<LegalDocument[]>;
}

export interface AIAnalysisService {
  analyze(document: LegalDocument): Promise<AIInsights>;
  summarize(content: string): Promise<string>;
  extractEntities(content: string): Promise<any[]>;
}

export interface ComplianceService {
  validateCompliance(insights: AIInsights): Promise<ComplianceCheck[]>;
  calculateRisk(insights: AIInsights): Promise<number>;
}

export class LegalDocumentService {
  constructor(
    private aiAnalysisService: AIAnalysisService,
    private documentRepository: DocumentRepository,
    private complianceService: ComplianceService
  ) {}

  async analyzeDocument(documentId: string): Promise<LegalAnalysis> {
    try {
      // Fetch document
      const document = await this.documentRepository.findById(documentId);
      
      if (!document) {
        throw new Error(`Document ${documentId} not found`);
      }

      // AI analysis
      const aiInsights = await this.aiAnalysisService.analyze(document);
      
      // Compliance validation
      const complianceChecks = await this.complianceService.validateCompliance(aiInsights);
      
      // Risk assessment
      const riskScore = await this.complianceService.calculateRisk(aiInsights);
      
      return {
        document,
        insights: aiInsights,
        complianceChecks,
        riskAssessment: {
          score: riskScore,
          level: this.getRiskLevel(riskScore),
          recommendations: this.generateRecommendations(riskScore, complianceChecks)
        },
        analyzedAt: new Date()
      };
    } catch (error) {
      console.error('Document analysis failed:', error);
      throw new Error(`Failed to analyze document: ${error.message}`);
    }
  }

  async analyzeCase(caseId: string): Promise<LegalAnalysis[]> {
    const documents = await this.documentRepository.findByCase(caseId);
    
    const analyses = await Promise.all(
      documents.map(doc => this.analyzeDocument(doc.id))
    );
    
    return analyses;
  }

  private getRiskLevel(score: number): 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL' {
    if (score >= 0.8) return 'CRITICAL';
    if (score >= 0.6) return 'HIGH';
    if (score >= 0.4) return 'MEDIUM';
    return 'LOW';
  }

  private generateRecommendations(
    riskScore: number, 
    complianceChecks: ComplianceCheck[]
  ): string[] {
    const recommendations: string[] = [];
    
    // High-risk recommendations
    if (riskScore >= 0.7) {
      recommendations.push('Immediate legal review required');
      recommendations.push('Consider senior partner consultation');
    }
    
    // Compliance-based recommendations
    const failedChecks = complianceChecks.filter(check => !check.passed);
    if (failedChecks.length > 0) {
      recommendations.push(`Address ${failedChecks.length} compliance issues`);
    }
    
    return recommendations;
  }
}

// Audit Service for Legal Compliance
export class LegalAuditService {
  async logAction(action: {
    type: string;
    entityType: string;
    entityId: string;
    userId: string;
    details?: any;
  }) {
    const auditEntry = {
      ...action,
      timestamp: new Date(),
      ipAddress: await this.getClientIP(),
      userAgent: navigator.userAgent
    };
    
    // Store in database
    await fetch('/api/audit', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(auditEntry)
    });
  }

  private async getClientIP(): Promise<string> {
    try {
      const response = await fetch('https://api.ipify.org?format=json');
      const data = await response.json();
      return data.ip;
    } catch {
      return 'unknown';
    }
  }
}
