/**
 * Enhanced Inline Suggestion Service
 * Orchestrates AI-powered suggestions using XState machines and Go microservice
 */

import { createActor } from 'xstate';
import { aiProcessingMachine, goMicroserviceMachine, createAITask, aiTaskCreators } from '$lib/stores/machines';
import { enhancedRAGStore } from '$lib/stores';
import { debounce } from 'lodash-es';

export interface SuggestionContext {
  text: string;
  cursorPosition: number;
  contextBefore: string;
  contextAfter: string;
  documentType?: 'legal' | 'evidence' | 'notes' | 'general';
  caseId?: string;
  userId?: string;
}

export interface InlineSuggestion {
  id: string;
  type: 'completion' | 'grammar' | 'legal_term' | 'case_reference' | 'citation';
  text: string;
  replacement?: string;
  confidence: number;
  reasoning: string;
  range?: { start: number; end: number };
  metadata?: {
    sourceType?: string;
    relevanceScore?: number;
    legalAccuracy?: number;
  };
}

export interface SuggestionOptions {
  enableAutoComplete: boolean;
  enableGrammarCheck: boolean;
  enableLegalTerms: boolean;
  enableCaseReferences: boolean;
  enableCitations: boolean;
  minCharacters: number;
  maxSuggestions: number;
  suggestionDelay: number;
  aiModel: string;
}

export class InlineSuggestionService {
  private aiActor: any;
  private goServiceActor: any;
  private isInitialized = false;
  private pendingTasks = new Map<string, Promise<any>>();
  
  constructor(private options: SuggestionOptions = {
    enableAutoComplete: true,
    enableGrammarCheck: true,
    enableLegalTerms: true,
    enableCaseReferences: true,
    enableCitations: true,
    minCharacters: 10,
    maxSuggestions: 5,
    suggestionDelay: 800,
    aiModel: 'gemma3-legal'
  }) {
    this.initialize();
  }

  private initialize() {
    // Initialize XState actors
    this.aiActor = createActor(aiProcessingMachine);
    this.goServiceActor = createActor(goMicroserviceMachine);
    
    this.aiActor.start();
    this.goServiceActor.start();
    
    // Connect to Go microservice
    this.goServiceActor.send({ 
      type: 'CONNECT', 
      endpoint: 'http://localhost:8080' 
    });
    
    this.isInitialized = true;
  }

  /**
   * Generate AI-powered suggestions for the given context
   */
  async generateSuggestions(context: SuggestionContext): Promise<InlineSuggestion[]> {
    if (!this.isInitialized) {
      this.initialize();
    }

    if (context.text.length < this.options.minCharacters) {
      return [];
    }

    const suggestions: InlineSuggestion[] = [];
    const parallel: Promise<InlineSuggestion[]>[] = [];

    // 1. Auto-completion suggestions
    if (this.options.enableAutoComplete) {
      parallel.push(this.generateCompletionSuggestions(context));
    }

    // 2. Grammar and style suggestions
    if (this.options.enableGrammarCheck) {
      parallel.push(this.generateGrammarSuggestions(context));
    }

    // 3. Legal terminology suggestions
    if (this.options.enableLegalTerms) {
      parallel.push(this.generateLegalTermSuggestions(context));
    }

    // 4. Case reference suggestions
    if (this.options.enableCaseReferences) {
      parallel.push(this.generateCaseReferenceSuggestions(context));
    }

    // 5. Citation suggestions
    if (this.options.enableCitations) {
      parallel.push(this.generateCitationSuggestions(context));
    }

    // Execute all suggestion generators in parallel
    const results = await Promise.allSettled(parallel);
    
    results.forEach((result) => {
      if (result.status === 'fulfilled') {
        suggestions.push(...result.value);
      } else {
        console.warn('Suggestion generation failed:', result.reason);
      }
    });

    // Sort by confidence and limit results
    return suggestions
      .sort((a, b) => b.confidence - a.confidence)
      .slice(0, this.options.maxSuggestions);
  }

  /**
   * Generate text completion suggestions
   */
  private async generateCompletionSuggestions(context: SuggestionContext): Promise<InlineSuggestion[]> {
    const task = aiTaskCreators.analyzeDocument(
      `Complete this ${context.documentType || 'legal'} text naturally:
      
      Context before cursor: "${context.contextBefore}"
      Context after cursor: "${context.contextAfter}"
      
      Provide 3 natural completions that:
      1. Maintain professional legal tone
      2. Use appropriate legal terminology
      3. Complete the current thought or sentence
      4. Consider the document type: ${context.documentType || 'legal'}
      
      Return JSON array: [{"text": "completion", "confidence": 0.8, "reasoning": "explanation"}]`,
      this.options.aiModel,
      'json'
    );

    const result = await this.executeAITask(task);
    
    if (result?.success && result.result) {
      const completions = Array.isArray(result.result) ? result.result : result.result.completions || [];
      
      return completions.map((completion: any, index: number) => ({
        id: `completion_${Date.now()}_${index}`,
        type: 'completion' as const,
        text: completion.text || completion,
        confidence: completion.confidence || 0.7,
        reasoning: completion.reasoning || 'AI-generated text completion',
        metadata: {
          sourceType: 'llm_completion',
          relevanceScore: completion.confidence || 0.7
        }
      }));
    }

    return [];
  }

  /**
   * Generate grammar and style suggestions
   */
  private async generateGrammarSuggestions(context: SuggestionContext): Promise<InlineSuggestion[]> {
    const task = aiTaskCreators.analyzeDocument(
      `Analyze this ${context.documentType || 'legal'} text for grammar, style, and legal writing improvements:
      
      Text: "${context.text}"
      
      Focus on:
      1. Grammar errors and corrections
      2. Legal writing style improvements
      3. Clarity and precision enhancements
      4. Professional tone adjustments
      5. Sentence structure optimization
      
      Return JSON array with specific suggestions:
      [{"text": "suggestion", "replacement": "corrected text", "confidence": 0.9, "reasoning": "explanation", "range": {"start": 10, "end": 20}}]`,
      this.options.aiModel,
      'json'
    );

    const result = await this.executeAITask(task);
    
    if (result?.success && result.result) {
      const suggestions = Array.isArray(result.result) ? result.result : result.result.suggestions || [];
      
      return suggestions.map((suggestion: any, index: number) => ({
        id: `grammar_${Date.now()}_${index}`,
        type: 'grammar' as const,
        text: suggestion.text,
        replacement: suggestion.replacement,
        confidence: suggestion.confidence || 0.8,
        reasoning: suggestion.reasoning || 'Grammar/style improvement',
        range: suggestion.range,
        metadata: {
          sourceType: 'grammar_check',
          legalAccuracy: suggestion.legalAccuracy || 0.8
        }
      }));
    }

    return [];
  }

  /**
   * Generate legal terminology suggestions
   */
  private async generateLegalTermSuggestions(context: SuggestionContext): Promise<InlineSuggestion[]> {
    try {
      // First get semantic embedding for the context
      const embeddingTask = createAITask('embed', {
        text: context.contextBefore,
        model: 'nomic-embed-text'
      }, { priority: 'medium' });

      const embeddingResult = await this.executeAITask(embeddingTask);
      
      if (embeddingResult?.success) {
        // Use enhanced RAG to find related legal terms
        const ragResults = await enhancedRAGStore.search(
          context.contextBefore,
          {
            topK: 10,
            useEnhancedMode: true,
            userId: context.userId || 'inline-editor',
            sessionId: Date.now().toString(),
            filters: { 
              confidenceThreshold: 0.6,
              documentTypes: ['legal', 'case_law', 'statute']
            }
          }
        );

        if (ragResults && typeof ragResults === 'object' && 'results' in ragResults && ragResults.results?.length > 0) {
          // Process RAG results into legal term suggestions
          const legalTermTask = aiTaskCreators.analyzeDocument(
            `Based on these legal documents and the user's context, suggest relevant legal terms:
            
            User context: "${context.contextBefore}"
            Document type: ${context.documentType || 'legal'}
            
            Legal documents found:
            ${(ragResults as any).results.slice(0, 5).map((r: any) => `- ${r.content.slice(0, 200)}...`).join('\n')}
            
            Suggest 3-5 relevant legal terms, phrases, or concepts that would improve the text.
            Return JSON: [{"term": "legal term", "definition": "brief definition", "confidence": 0.8, "context_relevance": 0.9}]`,
            this.options.aiModel,
            'json'
          );

          const termResult = await this.executeAITask(legalTermTask);
          
          if (termResult?.success && termResult.result) {
            const terms = Array.isArray(termResult.result) ? termResult.result : termResult.result.terms || [];
            
            return terms.map((term: any, index: number) => ({
              id: `legal_term_${Date.now()}_${index}`,
              type: 'legal_term' as const,
              text: term.term,
              confidence: term.confidence || 0.7,
              reasoning: `Legal term: ${term.definition || 'Relevant legal concept'}`,
              metadata: {
                sourceType: 'rag_legal_terms',
                relevanceScore: term.context_relevance || 0.7,
                legalAccuracy: term.confidence || 0.7
              }
            }));
          }
        }
      }
    } catch (error) {
      console.warn('Legal term suggestion failed:', error);
    }

    return [];
  }

  /**
   * Generate case reference suggestions
   */
  private async generateCaseReferenceSuggestions(context: SuggestionContext): Promise<InlineSuggestion[]> {
    if (!context.caseId) {
      return [];
    }

    try {
      // Search for related cases using RAG
      const ragResults = await enhancedRAGStore.search(
        `${context.contextBefore} case references precedent`,
        {
          topK: 5,
          useEnhancedMode: true,
          userId: context.userId || 'inline-editor',
          sessionId: Date.now().toString(),
          caseId: context.caseId,
          filters: { 
            confidenceThreshold: 0.7,
            documentTypes: ['case', 'precedent']
          }
        }
      );

      if (ragResults && typeof ragResults === 'object' && 'results' in ragResults && ragResults.results?.length > 0) {
        return (ragResults as any).results.slice(0, 3).map((result: any, index: number) => ({
          id: `case_ref_${Date.now()}_${index}`,
          type: 'case_reference' as const,
          text: result.summary || `Reference to ${result.metadata?.title || 'related case'}`,
          confidence: result.confidence,
          reasoning: `Related case reference: ${result.metadata?.caseNumber || 'Similar case'}`,
          metadata: {
            sourceType: 'case_reference',
            relevanceScore: result.score,
            legalAccuracy: result.confidence
          }
        }));
      }
    } catch (error) {
      console.warn('Case reference suggestion failed:', error);
    }

    return [];
  }

  /**
   * Generate citation suggestions
   */
  private async generateCitationSuggestions(context: SuggestionContext): Promise<InlineSuggestion[]> {
    const task = aiTaskCreators.analyzeDocument(
      `Suggest legal citations for this ${context.documentType || 'legal'} text:
      
      Text: "${context.contextBefore}"
      
      Provide relevant citations that support the statements made. Include:
      1. Case law citations
      2. Statutory references
      3. Regulatory citations
      4. Legal authorities
      
      Return JSON: [{"citation": "citation text", "authority": "type of authority", "relevance": "explanation", "confidence": 0.8}]`,
      this.options.aiModel,
      'json'
    );

    const result = await this.executeAITask(task);
    
    if (result?.success && result.result) {
      const citations = Array.isArray(result.result) ? result.result : result.result.citations || [];
      
      return citations.map((citation: any, index: number) => ({
        id: `citation_${Date.now()}_${index}`,
        type: 'citation' as const,
        text: citation.citation,
        confidence: citation.confidence || 0.6,
        reasoning: `${citation.authority}: ${citation.relevance}`,
        metadata: {
          sourceType: 'legal_citation',
          relevanceScore: citation.confidence || 0.6,
          legalAccuracy: citation.confidence || 0.6
        }
      }));
    }

    return [];
  }

  /**
   * Execute AI task and wait for result
   */
  private async executeAITask(task: any): Promise<any> {
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error('AI task timeout'));
      }, 15000);

      const subscription = this.aiActor.subscribe((state: any) => {
        if (state.context.result?.taskId === task.id) {
          clearTimeout(timeout);
          subscription.unsubscribe();
          resolve(state.context.result);
        }
        if (state.context.error && state.context.task?.id === task.id) {
          clearTimeout(timeout);
          subscription.unsubscribe();
          reject(new Error(state.context.error));
        }
      });

      this.aiActor.send({ type: 'START_PROCESSING', task });
    });
  }

  /**
   * Update service options
   */
  updateOptions(options: Partial<SuggestionOptions>) {
    this.options = { ...this.options, ...options };
  }

  /**
   * Check if service is ready
   */
  isReady(): boolean {
    return this.isInitialized && 
           this.aiActor?.getSnapshot().matches('idle') &&
           this.goServiceActor?.getSnapshot().context.connectionStatus === 'connected';
  }

  /**
   * Cleanup resources
   */
  destroy() {
    this.aiActor?.stop();
    this.goServiceActor?.stop();
    this.pendingTasks.clear();
    this.isInitialized = false;
  }
}

// Singleton instance for global use
export const inlineSuggestionService = new InlineSuggestionService();

// Debounced wrapper for real-time usage
export const debouncedSuggestionGenerator = debounce(
  (context: SuggestionContext) => inlineSuggestionService.generateSuggestions(context),
  800
);

// Utility functions
export const createSuggestionContext = (
  text: string,
  cursorPosition: number,
  options: Partial<SuggestionContext> = {}
): SuggestionContext => ({
  text,
  cursorPosition,
  contextBefore: text.slice(Math.max(0, cursorPosition - 150), cursorPosition),
  contextAfter: text.slice(cursorPosition, Math.min(text.length, cursorPosition + 50)),
  documentType: 'legal',
  ...options
});

export const applySuggestionToText = (
  text: string,
  suggestion: InlineSuggestion,
  cursorPosition: number
): { newText: string; newCursorPosition: number } => {
  if (suggestion.type === 'completion') {
    const newText = text.slice(0, cursorPosition) + suggestion.text + text.slice(cursorPosition);
    return {
      newText,
      newCursorPosition: cursorPosition + suggestion.text.length
    };
  } else if (suggestion.replacement && suggestion.range) {
    const newText = text.slice(0, suggestion.range.start) + 
                   suggestion.replacement + 
                   text.slice(suggestion.range.end);
    return {
      newText,
      newCursorPosition: suggestion.range.start + suggestion.replacement.length
    };
  }
  
  return { newText: text, newCursorPosition: cursorPosition };
};