
/**
 * Enhanced Inline Suggestion Service
 * Orchestrates AI-powered suggestions using XState machines and Go microservice
 */

import { createActor, type ActorRefFrom } from "xstate";
import { aiProcessingMachine, goMicroserviceMachine, createAITask, aiTaskCreators } from "$lib/machines";
import { enhancedRAGStore } from "$lib/stores";
import { debounce } from "$lib/utils";

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

// Lightweight result/type guards to avoid pervasive `any` usage
interface AITaskResult<T = unknown> {
  success: boolean;
  taskId?: string;
  result?: T;
  error?: string;
}

type AIActor = ActorRefFrom<typeof aiProcessingMachine>;
type GoActor = ActorRefFrom<typeof goMicroserviceMachine>;

// Simple LRU helper for tracking pending task promises (prevents unbounded growth)
class LRUMap<K, V> extends Map<K, V> {
  constructor(private maxSize = 50) { super(); }
  set(key: K, value: V) {
    if (this.has(key)) super.delete(key); // refresh order
    super.set(key, value);
    if (this.size > this.maxSize) {
      // delete oldest (Map iteration order = insertion order)
      const firstKey = this.keys().next().value as K | undefined;
      if (firstKey !== undefined) super.delete(firstKey);
    }
    return this;
  }
}

export class InlineSuggestionService {
  private aiActor: AIActor | null = null;
  private goServiceActor: GoActor | null = null;
  private isInitialized = false;
  private pendingTasks = new LRUMap<string, Promise<AITaskResult>>();
  private abortedTaskIds = new Set<string>();
  private defaultModel = (typeof process !== 'undefined' && (process as any)?.env?.VITE_LEGAL_AI_MODEL) || (typeof import.meta !== 'undefined' && (import.meta as any).env?.VITE_LEGAL_AI_MODEL) || 'gemma3-legal';

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
    // SSR / non-browser guard: defer initialization until first browser call
    if (typeof window === 'undefined') {
      return; // will lazily init on first client invocation
    }
    if (this.isInitialized) return;

    try {
      this.aiActor = createActor(aiProcessingMachine);
      this.goServiceActor = createActor(goMicroserviceMachine);

      this.aiActor.start();
      this.goServiceActor.start();

      // Connect to Go microservice (TODO: centralize base URL config)
      this.goServiceActor.send({
        type: 'CONNECT',
        endpoint: 'http://localhost:8080'
      });

      this.isInitialized = true;
    } catch (err) {
      console.warn('[InlineSuggestionService] init failed', err);
      this.isInitialized = false;
    }
  }

  /**
   * Generate AI-powered suggestions for the given context
   */
  async generateSuggestions(context: SuggestionContext): Promise<InlineSuggestion[]> {
    if (!this.isInitialized) this.initialize();
    // Still not initialized (SSR) => short‑circuit
    if (!this.isInitialized) return [];

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
      const raw = (result as any).result;
      const completions = Array.isArray(raw) ? raw : (raw?.completions && Array.isArray(raw.completions) ? raw.completions : []);

      return completions.filter(Boolean).map((completion: any, index: number) => ({
        id: `completion_${Date.now()}_${index}`,
        type: 'completion' as const,
        text: typeof completion === 'string' ? completion : (completion.text ?? ''),
        confidence: typeof completion?.confidence === 'number' ? completion.confidence : 0.7,
        reasoning: completion.reasoning || 'AI-generated text completion',
        metadata: {
          sourceType: 'llm_completion',
          relevanceScore: typeof completion?.confidence === 'number' ? completion.confidence : 0.7
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
      const raw = (result as any).result;
      const suggestions = Array.isArray(raw) ? raw : (raw?.suggestions && Array.isArray(raw.suggestions) ? raw.suggestions : []);

      return suggestions.filter(Boolean).map((suggestion: any, index: number) => ({
        id: `grammar_${Date.now()}_${index}`,
        type: 'grammar' as const,
        text: suggestion?.text ?? '',
        replacement: suggestion?.replacement,
        confidence: typeof suggestion?.confidence === 'number' ? suggestion.confidence : 0.8,
        reasoning: suggestion.reasoning || 'Grammar/style improvement',
        range: suggestion?.range,
        metadata: {
          sourceType: 'grammar_check',
          legalAccuracy: typeof suggestion?.legalAccuracy === 'number' ? suggestion.legalAccuracy : 0.8
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
            const raw = (termResult as any).result;
            const terms = Array.isArray(raw) ? raw : (raw?.terms && Array.isArray(raw.terms) ? raw.terms : []);

            return terms.filter(Boolean).map((term: any, index: number) => ({
              id: `legal_term_${Date.now()}_${index}`,
              type: 'legal_term' as const,
              text: term?.term ?? '',
              confidence: typeof term?.confidence === 'number' ? term.confidence : 0.7,
              reasoning: `Legal term: ${term?.definition || 'Relevant legal concept'}`,
              metadata: {
                sourceType: 'rag_legal_terms',
                relevanceScore: typeof term?.context_relevance === 'number' ? term.context_relevance : 0.7,
                legalAccuracy: typeof term?.confidence === 'number' ? term.confidence : 0.7
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
      const raw = (result as any).result;
      const citations = Array.isArray(raw) ? raw : (raw?.citations && Array.isArray(raw.citations) ? raw.citations : []);

      return citations.filter(Boolean).map((citation: any, index: number) => ({
        id: `citation_${Date.now()}_${index}`,
        type: 'citation' as const,
        text: citation?.citation ?? '',
        confidence: typeof citation?.confidence === 'number' ? citation.confidence : 0.6,
        reasoning: `${citation?.authority || 'authority'}: ${citation?.relevance || 'relevant support'}`,
        metadata: {
          sourceType: 'legal_citation',
          relevanceScore: typeof citation?.confidence === 'number' ? citation.confidence : 0.6,
          legalAccuracy: typeof citation?.confidence === 'number' ? citation.confidence : 0.6
        }
      }));
    }

    return [];
  }

  /**
   * Execute AI task and wait for result
   */
  private async executeAITask(task: any, opts: { timeoutMs?: number; signal?: AbortSignal } = {}): Promise<AITaskResult> {
    if (!this.aiActor) throw new Error('AI actor not initialized');
    const { timeoutMs = 15000, signal } = opts;
    if (signal?.aborted) {
      return { success: false, taskId: task?.id, error: 'aborted' };
    }

    return new Promise<AITaskResult>((resolve, reject) => {
      let settled = false;
      const onAbort = () => {
        this.abortedTaskIds.add(task.id);
        cleanup();
        const res: AITaskResult = { success: false, taskId: task.id, error: 'aborted' };
        resolve(res);
      };
      const timer = setTimeout(() => {
        if (settled) return;
        settled = true;
        cleanup();
        reject(new Error('AI task timeout'));
      }, timeoutMs);

      const subscription = this.aiActor!.subscribe((state: any) => {
        try {
          const ctx = state?.context;
          if (!ctx) return;
          if (ctx.result?.taskId === task.id && !settled) {
            settled = true;
            cleanup();
            resolve(ctx.result as AITaskResult);
          } else if (ctx.error && ctx.task?.id === task.id && !settled) {
            settled = true;
            cleanup();
            reject(new Error(ctx.error));
          }
        } catch (e) {
        // swallow; defensive guard
        }
      });

      const cleanup = () => {
        clearTimeout(timer);
        subscription.unsubscribe();
        signal?.removeEventListener('abort', onAbort);
      };

      signal?.addEventListener('abort', onAbort, { once: true });
      this.aiActor!.send({ type: 'START_PROCESSING', task });
    });
  }

  /**
   * Update service options
   */
  updateOptions(options: Partial<SuggestionOptions>) {
    this.options = { ...this.options, ...options };
    // Allow dynamic model override if none explicitly provided yet
    if (!options.aiModel && this.options.aiModel === 'gemma3-legal' && this.defaultModel !== 'gemma3-legal') {
      this.options.aiModel = this.defaultModel; // reflect env override (GEMMA3_COMPLETE_SOLUTION)
    }
  }

  /**
   * Check if service is ready
   */
  isReady(): boolean {
    if (!this.isInitialized || !this.aiActor || !this.goServiceActor) return false;
    try {
      const aiSnap = this.aiActor.getSnapshot();
      const goSnap = this.goServiceActor.getSnapshot();
      return !!aiSnap && aiSnap.matches?.('idle') && !!goSnap?.context && goSnap.context.connectionStatus === 'connected';
    } catch {
      return false;
    }
  }

  /**
   * Cleanup resources
   */
  destroy() {
    try { this.aiActor?.stop(); } catch { }
    try { this.goServiceActor?.stop(); } catch { }
    this.pendingTasks.clear();
    this.abortedTaskIds.clear();
    this.isInitialized = false;
  }

  /** Cancel (mark aborted) all currently pending AI tasks */
  cancelAll() {
    for (const [taskId] of this.pendingTasks) {
      this.abortedTaskIds.add(taskId);
    }
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