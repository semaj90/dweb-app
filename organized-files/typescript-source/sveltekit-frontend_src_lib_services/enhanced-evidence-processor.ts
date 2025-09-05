/**
 * Enhanced Evidence Processor Service
 * Integrates OCR, Enhanced RAG, and Legal AI processing with file uploads
 */

import type { EvidenceMetadata } from '$lib/server/db/schema-unified-postgres.js';

export interface ProcessingOptions {
  enableOcr: boolean;
  enableAiAnalysis: boolean;
  enableEmbeddings: boolean;
  enableSummarization: boolean;
  caseId?: string;
  userId?: string;
}

export interface EvidenceProcessingResult {
  success: boolean;
  evidenceId: string;
  metadata: EvidenceMetadata;
  ocrResult?: {
    extractedText: string;
    confidence: number;
    legalConcepts: string[];
    citations: string[];
    pageCount?: number;
  };
  aiAnalysis?: {
    summary: string;
    keyPoints: string[];
    entities: Array<{ name: string; type: string; confidence: number }>;
    sentiment?: { score: number; label: 'positive' | 'negative' | 'neutral' };
    categories: string[];
    confidence: number;
  };
  embeddings?: number[];
  processingTime: number;
  error?: string;
}

export class EnhancedEvidenceProcessor {;
  private readonly ocrEndpoint = '/api/ocr/extract';
  private readonly ragEndpoint = '/api/v1/rag';
  private readonly enhancedRagEndpoint = 'http://localhost:8095/api/rag'; // Go service
  
  /**
   * Process uploaded evidence with comprehensive AI pipeline
   */
  async processEvidence(
    file: File, 
    evidenceType: string,
    options: ProcessingOptions
  ): Promise<EvidenceProcessingResult> {
    const startTime = Date.now();
    
    try {
      console.log(`Starting evidence processing for: ${file.name} (${evidenceType})`);
      
      let ocrResult = null;
      let aiAnalysis = null;
      let embeddings = null;
      
      // Step 1: OCR Processing for supported file types
      if (options.enableOcr && (evidenceType === 'PDF' || evidenceType === 'IMAGE')) {
        ocrResult = await this.performOCR(file);
        console.log('OCR completed:', {
          confidence: ocrResult?.confidence,
          textLength: ocrResult?.extractedText?.length,
          conceptsFound: ocrResult?.legalConcepts?.length,
          citationsFound: ocrResult?.citations?.length
        });
      }
      
      // Step 2: Enhanced RAG Analysis
      if (options.enableAiAnalysis && (ocrResult?.extractedText || evidenceType === 'TEXT')) {
        const contentToAnalyze = ocrResult?.extractedText || await this.extractTextFromFile(file);
        if (contentToAnalyze) {
          aiAnalysis = await this.performEnhancedRAGAnalysis(
            contentToAnalyze, 
            {
              caseId: options.caseId,
              userId: options.userId,
              documentType: evidenceType,
              filename: file.name
            }
          );
          console.log('Enhanced RAG analysis completed:', {
            summaryLength: aiAnalysis?.summary?.length,
            keyPointsCount: aiAnalysis?.keyPoints?.length,
            entitiesCount: aiAnalysis?.entities?.length,
            confidence: aiAnalysis?.confidence
          });
        }
      }
      
      // Step 3: Vector Embeddings Generation
      if (options.enableEmbeddings && (ocrResult?.extractedText || aiAnalysis?.summary)) {
        const textForEmbedding = aiAnalysis?.summary || ocrResult?.extractedText || file.name;
        embeddings = await this.generateEmbeddings(textForEmbedding);
        console.log('Embeddings generated:', { dimensions: embeddings?.length });
      }
      
      // Step 4: Generate Rich Metadata
      const metadata = this.generateRichMetadata(file, evidenceType, {
        ocrResult,
        aiAnalysis,
        embeddings,
        processingOptions: options
      });
      
      const processingTime = Date.now() - startTime;
      
      return {
        success: true,
        evidenceId: `evidence_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        metadata,
        ocrResult,
        aiAnalysis,
        embeddings,
        processingTime
      };
      
    } catch (error: any) {
      console.error('Evidence processing error:', error);
      
      return {
        success: false,
        evidenceId: '',
        metadata: { kind: 'UNKNOWN' } as EvidenceMetadata,
        processingTime: Date.now() - startTime,
        error: error instanceof Error ? error.message : 'Processing failed'
      };
    }
  }
  
  /**
   * Perform OCR using the existing API endpoint
   */
  private async performOCR(file: File) {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch(this.ocrEndpoint, {
      method: 'POST',
      body: formData
    });
    
    if (!response.ok) {
      throw new Error(`OCR failed: ${response.statusText}`);
    }
    
    const result = await response.json();
    
    return {
      extractedText: result.text,
      confidence: result.averageConfidence,
      legalConcepts: result.legalConcepts || [],
      citations: result.citations || [],
      pageCount: result.pages
    };
  }
  
  /**
   * Perform Enhanced RAG Analysis using the Go microservice
   */
  private async performEnhancedRAGAnalysis(
    text: string, 
    context: { caseId?: string; userId?: string; documentType: string; filename: string }
  ) {
    try {
      // Try enhanced RAG service (Go microservice) first
      const enhancedResponse = await fetch(this.enhancedRagEndpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: `Analyze this ${context.documentType.toLowerCase()} document: ${text}`,
          options: {
            caseId: context.caseId,
            userId: context.userId,
            useCache: true,
            includeContext7: true,
            priority: 'high',
            enableFallback: true
          },
          context: {
            documentType: context.documentType,
            filename: context.filename,
            textLength: text.length
          }
        })
      });
      
      if (enhancedResponse.ok) {
        const enhancedResult = await enhancedResponse.json();
        console.log('Enhanced RAG service responded:', enhancedResult.metadata);
        
        return this.parseRAGResponse(enhancedResult);
      } else {
        console.warn('Enhanced RAG service unavailable, falling back to local analysis');
      }
    } catch (error: any) {
      console.warn('Enhanced RAG service error, using fallback:', error);
    }
    
    // Fallback to local SvelteKit endpoint
    const fallbackResponse = await fetch(this.ragEndpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: `Analyze and summarize this ${context.documentType.toLowerCase()} document`,
        text: text.substring(0, 4000), // Limit for fallback
        context
      })
    });
    
    if (!fallbackResponse.ok) {
      throw new Error('RAG analysis failed');
    }
    
    const fallbackResult = await fallbackResponse.json();
    return this.parseRAGResponse(fallbackResult);
  }
  
  /**
   * Parse RAG service response into standardized format
   */
  private parseRAGResponse(ragResult: any) {
    return {
      summary: ragResult.output || ragResult.summary || '',
      keyPoints: ragResult.keyPoints || this.extractKeyPoints(ragResult.output),
      entities: ragResult.entities || [],
      sentiment: ragResult.sentiment,
      categories: ragResult.categories || this.inferCategories(ragResult.output),
      confidence: ragResult.score || ragResult.confidence || 0.8
    };
  }
  
  /**
   * Generate vector embeddings
   */
  private async generateEmbeddings(text: string): Promise<number[]> {
    // Try to use existing embedding service or Enhanced RAG
    try {
      const response = await fetch('/api/embeddings/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text.substring(0, 2000) })
      });
      
      if (response.ok) {
        const result = await response.json();
        const embeddings = result.embedding || [];
        
        // Ensure we have exactly 384 dimensions
        if (embeddings.length === 384) {
          return embeddings;
        } else {
          console.warn(`Received ${embeddings.length} dimensions, expected 384. Using mock embeddings.`);
        }
      }
    } catch (error: any) {
      console.warn('Embeddings service unavailable:', error);
    }
    
    // Return mock embeddings for development (exactly 384 dimensions)
    return Array.from({ length: 384 }, () => Math.random() * 2 - 1);
  }
  
  /**
   * Extract text from different file types
   */
  private async extractTextFromFile(file: File): Promise<string> {
    if (file.type.startsWith('text/')) {
      return await file.text();
    }
    
    // For other file types, return filename as fallback
    return file.name;
  }
  
  /**
   * Generate rich metadata combining all processing results
   */
  private generateRichMetadata(
    file: File, 
    evidenceType: string, 
    processingResults: any
  ): EvidenceMetadata {
    const baseMetadata = {
      fileSize: file.size,
      uploadedAt: new Date().toISOString(),
      processingOptions: processingResults.processingOptions
    };
    
    switch (evidenceType) {
      case 'PDF':
        return {
          kind: 'PDF',
          pageCount: processingResults.ocrResult?.pageCount || 1,
          isEncrypted: false,
          title: file.name,
          extractedText: processingResults.ocrResult?.extractedText,
          legalConcepts: processingResults.ocrResult?.legalConcepts || [],
          citations: processingResults.ocrResult?.citations || [],
          aiSummary: processingResults.aiAnalysis?.summary,
          confidence: processingResults.ocrResult?.confidence,
          ...baseMetadata
        } as EvidenceMetadata;
        
      case 'IMAGE':
        return {
          kind: 'IMAGE',
          resolution: { width: 0, height: 0 },
          format: file.type.split('/')[1] as any,
          hasAlphaChannel: file.type === 'image/png',
          extractedText: processingResults.ocrResult?.extractedText,
          ocrConfidence: processingResults.ocrResult?.confidence,
          ...baseMetadata
        } as EvidenceMetadata;
        
      case 'VIDEO':
        return {
          kind: 'VIDEO',
          durationSeconds: 0,
          resolution: { width: 0, height: 0 },
          codec: 'unknown',
          frameRate: 0,
          ...baseMetadata
        } as EvidenceMetadata;
        
      case 'AUDIO':
        return {
          kind: 'AUDIO',
          durationSeconds: 0,
          codec: 'unknown',
          sampleRate: 44100,
          channels: 2,
          ...baseMetadata
        } as EvidenceMetadata;
        
      case 'TEXT':
        return {
          kind: 'TEXT',
          wordCount: processingResults.ocrResult?.extractedText?.split(/\s+/).length || 0,
          characterCount: processingResults.ocrResult?.extractedText?.length || 0,
          language: 'unknown',
          aiSummary: processingResults.aiAnalysis?.summary,
          ...baseMetadata
        } as EvidenceMetadata;
        
      default:
        return {
          kind: 'UNKNOWN',
          ...baseMetadata
        } as EvidenceMetadata;
    }
  }
  
  /**
   * Helper: Extract key points from text
   */
  private extractKeyPoints(text: string): string[] {
    if (!text) return [];
    
    // Simple extraction logic - look for bullet points, numbered items, or sentences with keywords
    const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 20);
    const keyWords = ['important', 'key', 'critical', 'must', 'shall', 'required', 'obligation'];
    
    return sentences
      .filter(sentence => 
        keyWords.some(keyword => sentence.toLowerCase().includes(keyword))
      )
      .slice(0, 5)
      .map(s => s.trim());
  }
  
  /**
   * Helper: Infer document categories
   */
  private inferCategories(text: string): string[] {
    if (!text) return [];
    
    const categories: string[] = [];
    const lowerText = text.toLowerCase();
    
    if (lowerText.includes('contract') || lowerText.includes('agreement')) {
      categories.push('contract');
    }
    if (lowerText.includes('evidence') || lowerText.includes('exhibit')) {
      categories.push('evidence');
    }
    if (lowerText.includes('liability') || lowerText.includes('damages')) {
      categories.push('liability');
    }
    if (lowerText.includes('intellectual property') || lowerText.includes('copyright')) {
      categories.push('intellectual_property');
    }
    
    return categories;
  }
}

// Export singleton instance
export const enhancedEvidenceProcessor = new EnhancedEvidenceProcessor();