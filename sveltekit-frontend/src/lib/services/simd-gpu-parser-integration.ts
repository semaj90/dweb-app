/**
 * SIMD GPU Parser Integration with "Did You Mean" System
 * High-performance legal document parsing with intelligent suggestions
 */

import type { GPUDevice, GPUBuffer, GPUComputePipeline } from '@webgpu/types';

// Types for SIMD GPU Parser
export interface ParsedDocument {
  id: string;
  content: string;
  entities: ExtractedEntity[];
  metadata: DocumentMetadata;
  confidence: number;
  suggestions: DidYouMeanSuggestion[];
  processingTime: number;
}

export interface ExtractedEntity {
  type: 'person' | 'organization' | 'location' | 'date' | 'legal_term' | 'case_reference';
  text: string;
  confidence: number;
  position: { start: number; end: number };
  alternatives: string[];
}

export interface DocumentMetadata {
  title?: string;
  documentType: 'contract' | 'brief' | 'evidence' | 'citation' | 'statute' | 'regulation';
  jurisdiction?: string;
  practiceArea?: string[];
  language: string;
  pageCount?: number;
  wordCount: number;
}

export interface DidYouMeanSuggestion {
  original: string;
  suggestion: string;
  confidence: number;
  type: 'spelling' | 'legal_term' | 'entity' | 'context' | 'citation_format';
  context: string;
  alternatives: string[];
}

export interface ParsingConfig {
  enableSpellCheck: boolean;
  enableEntityExtraction: boolean;
  enableLegalTermSuggestions: boolean;
  enableCitationValidation: boolean;
  confidenceThreshold: number;
  maxSuggestions: number;
  simdOptimization: boolean;
  gpuAcceleration: boolean;
}

export interface SIMDStats {
  vectorOperationsPerSecond: number;
  parallelizationFactor: number;
  memoryBandwidth: number;
  cacheHitRate: number;
  averageLatency: number;
}

export class SIMDGPUParserIntegration {
  private gpuDevice: GPUDevice | null = null;
  private computePipeline: GPUComputePipeline | null = null;
  private gpuBuffers: Map<string, GPUBuffer> = new Map();
  
  // Legal terminology dictionary (loaded from resources)
  private legalTerms: Set<string> = new Set();
  private commonMisspellings: Map<string, string[]> = new Map();
  private citationPatterns: RegExp[] = [];
  
  // SIMD processing buffers
  private textBuffer: ArrayBuffer = new ArrayBuffer(0);
  private entityBuffer: Float32Array = new Float32Array(0);
  private suggestionBuffer: Uint32Array = new Uint32Array(0);
  
  // Performance tracking
  private simdStats: SIMDStats = {
    vectorOperationsPerSecond: 0,
    parallelizationFactor: 1,
    memoryBandwidth: 0,
    cacheHitRate: 0,
    averageLatency: 0
  };
  
  // Levenshtein distance cache for performance
  private levenshteinCache: Map<string, Map<string, number>> = new Map();
  
  constructor(private config: ParsingConfig) {
    this.initializeLegalDictionaries();
    this.initializeCitationPatterns();
  }

  /**
   * Initialize WebGPU for SIMD-optimized parsing
   */
  async initializeGPU(): Promise<void> {
    if (!this.config.gpuAcceleration) {
      console.log('📝 GPU acceleration disabled, using CPU SIMD');
      return;
    }

    try {
      console.log('🎮 Initializing WebGPU for SIMD parsing...');
      
      if (!navigator.gpu) {
        throw new Error('WebGPU not available');
      }

      const adapter = await navigator.gpu.requestAdapter({
        powerPreference: 'high-performance',
      });
      
      if (!adapter) {
        throw new Error('WebGPU adapter not available');
      }

      this.gpuDevice = await adapter.requestDevice({
        requiredFeatures: [],
        requiredLimits: {
          maxComputeWorkgroupSizeX: 256,
          maxComputeWorkgroupSizeY: 256,
        }
      });

      // Create compute shader for parallel text processing
      await this.createComputePipeline();
      await this.createGPUBuffers();
      
      console.log('✅ WebGPU SIMD parser initialized');
      
    } catch (error) {
      console.warn('⚠️ WebGPU initialization failed, falling back to CPU:', error);
    }
  }

  /**
   * Create WebGPU compute pipeline for text processing
   */
  private async createComputePipeline(): Promise<void> {
    if (!this.gpuDevice) return;

    const computeShaderCode = /* wgsl */ `
      struct TextData {
        characters: array<u32>,
        length: u32,
      }

      struct EntityResult {
        position: u32,
        entity_type: u32,
        confidence: f32,
        length: u32,
      }

      @group(0) @binding(0) var<storage, read> input_text: TextData;
      @group(0) @binding(1) var<storage, read_write> entity_results: array<EntityResult>;
      @group(0) @binding(2) var<storage, read> legal_terms: array<u32>;
      
      // SIMD-optimized entity detection
      @compute @workgroup_size(256)
      fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let index = global_id.x;
        
        if (index >= input_text.length) {
          return;
        }
        
        // Process 4 characters in parallel using SIMD-style operations
        let char_group = vec4<u32>(
          input_text.characters[index],
          input_text.characters[min(index + 1u, input_text.length - 1u)],
          input_text.characters[min(index + 2u, input_text.length - 1u)],
          input_text.characters[min(index + 3u, input_text.length - 1u)]
        );
        
        // Parallel pattern matching for legal terms
        var match_scores = vec4<f32>(0.0);
        
        // Process multiple patterns simultaneously
        for (var pattern_idx = 0u; pattern_idx < 1024u; pattern_idx += 1u) {
          let pattern_char = legal_terms[pattern_idx];
          let matches = select(vec4<f32>(0.0), vec4<f32>(1.0), char_group == vec4<u32>(pattern_char));
          match_scores += matches * 0.25; // Weight each character match
        }
        
        // Find best match and confidence
        let max_score = max(max(match_scores.x, match_scores.y), max(match_scores.z, match_scores.w));
        
        if (max_score > 0.7) {
          // Store entity result
          entity_results[index] = EntityResult(
            index,
            1u, // entity_type (will be refined on CPU)
            max_score,
            4u  // processed character group size
          );
        }
      }
    `;

    const shaderModule = this.gpuDevice.createShaderModule({
      code: computeShaderCode,
    });

    this.computePipeline = this.gpuDevice.createComputePipeline({
      layout: 'auto',
      compute: {
        module: shaderModule,
        entryPoint: 'main',
      },
    });

    console.log('✅ WebGPU compute pipeline created for SIMD parsing');
  }

  /**
   * Create GPU buffers for parallel processing
   */
  private async createGPUBuffers(): Promise<void> {
    if (!this.gpuDevice) return;

    const maxTextLength = 1024 * 1024; // 1MB text buffer
    const maxEntities = 10000; // Maximum entities per document

    const buffers = {
      'input_text': maxTextLength * 4, // u32 characters
      'entity_results': maxEntities * 16, // EntityResult struct size
      'legal_terms': 100000 * 4, // Legal terms dictionary
      'output_buffer': maxEntities * 16, // Output results
    };

    for (const [name, size] of Object.entries(buffers)) {
      const buffer = this.gpuDevice.createBuffer({
        size,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        mappedAtCreation: false,
      });
      
      this.gpuBuffers.set(name, buffer);
    }

    // Initialize legal terms buffer
    await this.uploadLegalTermsToGPU();
  }

  /**
   * Upload legal terms dictionary to GPU memory
   */
  private async uploadLegalTermsToGPU(): Promise<void> {
    if (!this.gpuDevice) return;

    const termsBuffer = this.gpuBuffers.get('legal_terms');
    if (!termsBuffer) return;

    // Convert legal terms to character codes for GPU processing
    const termsArray = Array.from(this.legalTerms);
    const termsData = new Uint32Array(100000); // Pre-allocated size
    
    let offset = 0;
    for (const term of termsArray) {
      for (let i = 0; i < term.length && offset < termsData.length; i++) {
        termsData[offset++] = term.charCodeAt(i);
      }
    }

    // Upload to GPU
    this.gpuDevice.queue.writeBuffer(
      termsBuffer,
      0,
      termsData.buffer,
      0,
      Math.min(termsData.byteLength, 100000 * 4)
    );
  }

  /**
   * Initialize legal terminology dictionaries
   */
  private initializeLegalDictionaries(): void {
    console.log('📚 Initializing legal terminology dictionaries...');
    
    // Legal terms (subset for demo - would be loaded from comprehensive database)
    const legalTermsSample = [
      'plaintiff', 'defendant', 'tort', 'contract', 'liability', 'jurisdiction',
      'precedent', 'statute', 'regulation', 'constitutional', 'amendment',
      'habeas corpus', 'due process', 'equal protection', 'substantive due process',
      'procedural due process', 'strict scrutiny', 'intermediate scrutiny',
      'rational basis', 'compelling interest', 'proximate cause', 'negligence',
      'breach of contract', 'specific performance', 'injunctive relief',
      'damages', 'punitive damages', 'compensatory damages', 'restitution',
      'estoppel', 'waiver', 'laches', 'res judicata', 'collateral estoppel',
      'summary judgment', 'motion to dismiss', 'discovery', 'deposition',
      'interrogatories', 'requests for production', 'subpoena', 'contempt'
    ];

    for (const term of legalTermsSample) {
      this.legalTerms.add(term.toLowerCase());
    }

    // Common legal misspellings
    this.commonMisspellings.set('pliantiff', ['plaintiff']);
    this.commonMisspellings.set('defendannt', ['defendant']);
    this.commonMisspellings.set('liablity', ['liability']);
    this.commonMisspellings.set('jurisidiction', ['jurisdiction']);
    this.commonMisspellings.set('precendent', ['precedent']);
    this.commonMisspellings.set('neglegence', ['negligence']);
    this.commonMisspellings.set('compensitory', ['compensatory']);
    this.commonMisspellings.set('injuctive', ['injunctive']);

    console.log(`✅ Loaded ${this.legalTerms.size} legal terms and ${this.commonMisspellings.size} common misspellings`);
  }

  /**
   * Initialize citation pattern recognition
   */
  private initializeCitationPatterns(): void {
    console.log('📖 Initializing citation patterns...');
    
    // Common legal citation patterns
    this.citationPatterns = [
      // Federal cases: Vol F.Supp Page (Court Year)
      /\d+\s+F\.\s?Supp\.\s?\d+\s+\([A-Z\.]+\s+\d{4}\)/gi,
      // State cases: Vol State_Reporter Page (Court Year)  
      /\d+\s+[A-Z][a-z]*\.\s?\d+\s+\([A-Z\.]+\s+\d{4}\)/gi,
      // U.S. Code: Title U.S.C. § Section
      /\d+\s+U\.S\.C\.\s+§\s+\d+/gi,
      // Code of Federal Regulations: Title C.F.R. § Section
      /\d+\s+C\.F\.R\.\s+§\s+\d+/gi,
      // Federal Register: Vol Fed. Reg. Page (date)
      /\d+\s+Fed\.\s+Reg\.\s+\d+\s+\([A-Z][a-z]+\s+\d+,\s+\d{4}\)/gi,
      // Law Review: Vol Law_Review Page (Year)
      /\d+\s+[A-Z][a-z\s]*L\.\s*Rev\.\s+\d+\s+\(\d{4}\)/gi,
    ];

    console.log(`✅ Initialized ${this.citationPatterns.length} citation patterns`);
  }

  /**
   * Parse document with SIMD GPU acceleration and "did you mean" suggestions
   */
  async parseDocument(
    content: string,
    metadata: Partial<DocumentMetadata> = {}
  ): Promise<ParsedDocument> {
    const startTime = performance.now();
    console.log(`📝 Parsing document with SIMD GPU acceleration...`);

    try {
      // Initialize GPU if not done yet
      if (this.config.gpuAcceleration && !this.gpuDevice) {
        await this.initializeGPU();
      }

      // Prepare document metadata
      const documentMetadata: DocumentMetadata = {
        documentType: metadata.documentType || 'contract',
        language: metadata.language || 'en',
        wordCount: content.split(/\s+/).length,
        ...metadata
      };

      // Parallel processing pipeline
      const [entities, suggestions] = await Promise.all([
        this.extractEntitiesSIMD(content),
        this.generateDidYouMeanSuggestions(content)
      ]);

      // Calculate overall confidence
      const averageEntityConfidence = entities.length > 0 
        ? entities.reduce((sum, e) => sum + e.confidence, 0) / entities.length 
        : 0;
      
      const suggestionPenalty = Math.min(suggestions.length * 0.02, 0.2); // Max 20% penalty
      const confidence = Math.max(0, averageEntityConfidence - suggestionPenalty);

      const processingTime = performance.now() - startTime;
      
      // Update performance stats
      this.updateSIMDStats(processingTime, content.length);

      const result: ParsedDocument = {
        id: crypto.randomUUID(),
        content,
        entities,
        metadata: documentMetadata,
        confidence,
        suggestions,
        processingTime
      };

      console.log(`✅ Document parsed in ${processingTime.toFixed(2)}ms with ${entities.length} entities and ${suggestions.length} suggestions`);
      
      return result;

    } catch (error) {
      console.error('❌ Document parsing failed:', error);
      throw new Error(`SIMD GPU parsing failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Extract entities using SIMD-optimized processing
   */
  private async extractEntitiesSIMD(content: string): Promise<ExtractedEntity[]> {
    if (!this.config.enableEntityExtraction) return [];

    const entities: ExtractedEntity[] = [];
    
    if (this.gpuDevice && this.computePipeline) {
      // GPU-accelerated entity extraction
      entities.push(...await this.extractEntitiesGPU(content));
    } else {
      // CPU SIMD-optimized fallback
      entities.push(...await this.extractEntitiesCPU(content));
    }

    return entities;
  }

  /**
   * GPU-accelerated entity extraction
   */
  private async extractEntitiesGPU(content: string): Promise<ExtractedEntity[]> {
    if (!this.gpuDevice || !this.computePipeline) return [];

    console.log('🎮 Running GPU-accelerated entity extraction...');

    try {
      // Prepare input data
      const inputBuffer = this.gpuBuffers.get('input_text')!;
      const outputBuffer = this.gpuBuffers.get('entity_results')!;
      
      // Convert text to character codes
      const textData = new Uint32Array(content.length + 1);
      textData[0] = content.length;
      for (let i = 0; i < content.length; i++) {
        textData[i + 1] = content.charCodeAt(i);
      }

      // Upload text to GPU
      this.gpuDevice.queue.writeBuffer(inputBuffer, 0, textData.buffer);

      // Create bind group
      const bindGroup = this.gpuDevice.createBindGroup({
        layout: this.computePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: inputBuffer } },
          { binding: 1, resource: { buffer: outputBuffer } },
          { binding: 2, resource: { buffer: this.gpuBuffers.get('legal_terms')! } },
        ],
      });

      // Execute compute shader
      const commandEncoder = this.gpuDevice.createCommandEncoder();
      const computePass = commandEncoder.beginComputePass();
      
      computePass.setPipeline(this.computePipeline);
      computePass.setBindGroup(0, bindGroup);
      computePass.dispatchWorkgroups(Math.ceil(content.length / 256));
      computePass.end();

      this.gpuDevice.queue.submit([commandEncoder.finish()]);

      // Read results back from GPU
      const resultBuffer = this.gpuDevice.createBuffer({
        size: outputBuffer.size,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });

      const copyEncoder = this.gpuDevice.createCommandEncoder();
      copyEncoder.copyBufferToBuffer(outputBuffer, 0, resultBuffer, 0, outputBuffer.size);
      this.gpuDevice.queue.submit([copyEncoder.finish()]);

      await resultBuffer.mapAsync(GPUMapMode.READ);
      const resultData = new Float32Array(resultBuffer.getMappedRange());
      
      // Process GPU results into entities
      const entities = this.processGPUResults(resultData, content);
      
      resultBuffer.unmap();
      resultBuffer.destroy();

      return entities;

    } catch (error) {
      console.warn('⚠️ GPU entity extraction failed, falling back to CPU:', error);
      return this.extractEntitiesCPU(content);
    }
  }

  /**
   * Process GPU computation results into entity objects
   */
  private processGPUResults(resultData: Float32Array, content: string): ExtractedEntity[] {
    const entities: ExtractedEntity[] = [];
    
    // Process results in groups of 4 (EntityResult struct)
    for (let i = 0; i < resultData.length; i += 4) {
      const position = Math.floor(resultData[i]);
      const entityType = Math.floor(resultData[i + 1]);
      const confidence = resultData[i + 2];
      const length = Math.floor(resultData[i + 3]);
      
      if (confidence > this.config.confidenceThreshold && position < content.length) {
        const text = content.substring(position, Math.min(position + length, content.length));
        
        if (text.trim().length > 0) {
          entities.push({
            type: this.mapEntityType(entityType),
            text: text.trim(),
            confidence,
            position: { start: position, end: position + text.length },
            alternatives: []
          });
        }
      }
    }

    return entities;
  }

  /**
   * CPU SIMD-optimized entity extraction fallback
   */
  private async extractEntitiesCPU(content: string): Promise<ExtractedEntity[]> {
    console.log('⚡ Running CPU SIMD-optimized entity extraction...');
    
    const entities: ExtractedEntity[] = [];
    const words = content.split(/\s+/);
    
    // Process words in batches for better cache performance
    const batchSize = 64; // SIMD-friendly batch size
    
    for (let i = 0; i < words.length; i += batchSize) {
      const batch = words.slice(i, i + batchSize);
      
      // Parallel processing of word batch
      const batchEntities = await Promise.all(
        batch.map(async (word, idx) => {
          const globalIndex = i + idx;
          return this.analyzeWordForEntities(word, globalIndex, content);
        })
      );
      
      // Flatten and filter results
      entities.push(...batchEntities.filter(e => e !== null) as ExtractedEntity[]);
    }

    return entities;
  }

  /**
   * Analyze individual word for entity classification
   */
  private async analyzeWordForEntities(
    word: string, 
    wordIndex: number, 
    fullContent: string
  ): Promise<ExtractedEntity | null> {
    const cleanWord = word.toLowerCase().replace(/[^\w]/g, '');
    
    if (cleanWord.length < 2) return null;

    // Check against legal terms dictionary
    if (this.legalTerms.has(cleanWord)) {
      return {
        type: 'legal_term',
        text: word,
        confidence: 0.9,
        position: this.findWordPosition(word, wordIndex, fullContent),
        alternatives: []
      };
    }

    // Pattern-based entity recognition
    
    // Date patterns
    if (/^\d{1,2}\/\d{1,2}\/\d{4}$/.test(word) || /^\d{4}-\d{2}-\d{2}$/.test(word)) {
      return {
        type: 'date',
        text: word,
        confidence: 0.95,
        position: this.findWordPosition(word, wordIndex, fullContent),
        alternatives: []
      };
    }

    // Organization patterns (Inc., LLC, Corp.)
    if (/\b\w+\s+(inc|llc|corp|ltd|co)\b/i.test(word)) {
      return {
        type: 'organization',
        text: word,
        confidence: 0.8,
        position: this.findWordPosition(word, wordIndex, fullContent),
        alternatives: []
      };
    }

    // Person name patterns (capitalized words)
    if (/^[A-Z][a-z]+$/.test(word) && word.length > 2) {
      return {
        type: 'person',
        text: word,
        confidence: 0.6, // Lower confidence for name detection
        position: this.findWordPosition(word, wordIndex, fullContent),
        alternatives: []
      };
    }

    return null;
  }

  /**
   * Find word position in full content
   */
  private findWordPosition(word: string, wordIndex: number, content: string): { start: number; end: number } {
    // Simple approximation - in production, would maintain word position mapping
    const words = content.split(/\s+/);
    let position = 0;
    
    for (let i = 0; i < Math.min(wordIndex, words.length); i++) {
      position += words[i].length + 1; // +1 for space
    }
    
    return {
      start: position,
      end: position + word.length
    };
  }

  /**
   * Generate "Did You Mean" suggestions for misspellings and improvements
   */
  private async generateDidYouMeanSuggestions(content: string): Promise<DidYouMeanSuggestion[]> {
    if (!this.config.enableSpellCheck && !this.config.enableLegalTermSuggestions) {
      return [];
    }

    console.log('🔍 Generating "Did You Mean" suggestions...');
    
    const suggestions: DidYouMeanSuggestion[] = [];
    const words = content.toLowerCase().split(/\s+/);
    
    // Process words in parallel batches
    const batchSize = 32;
    const suggestionPromises: Promise<DidYouMeanSuggestion[]>[] = [];
    
    for (let i = 0; i < words.length; i += batchSize) {
      const batch = words.slice(i, i + batchSize);
      suggestionPromises.push(this.processSuggestionBatch(batch, i, content));
    }
    
    const batchResults = await Promise.all(suggestionPromises);
    suggestions.push(...batchResults.flat());

    // Limit suggestions to avoid overwhelming the user
    return suggestions.slice(0, this.config.maxSuggestions);
  }

  /**
   * Process batch of words for suggestions
   */
  private async processSuggestionBatch(
    words: string[], 
    startIndex: number, 
    content: string
  ): Promise<DidYouMeanSuggestion[]> {
    const suggestions: DidYouMeanSuggestion[] = [];
    
    for (let i = 0; i < words.length; i++) {
      const word = words[i].replace(/[^\w]/g, '');
      const globalIndex = startIndex + i;
      
      if (word.length < 3) continue;

      // Check for common misspellings
      if (this.commonMisspellings.has(word)) {
        const alternatives = this.commonMisspellings.get(word)!;
        suggestions.push({
          original: word,
          suggestion: alternatives[0],
          confidence: 0.9,
          type: 'spelling',
          context: this.getWordContext(word, content),
          alternatives
        });
        continue;
      }

      // Check for similar legal terms using Levenshtein distance
      const similarTerms = this.findSimilarLegalTerms(word);
      if (similarTerms.length > 0) {
        suggestions.push({
          original: word,
          suggestion: similarTerms[0].term,
          confidence: similarTerms[0].similarity,
          type: 'legal_term',
          context: this.getWordContext(word, content),
          alternatives: similarTerms.slice(1, 4).map(t => t.term)
        });
      }
    }

    return suggestions;
  }

  /**
   * Find similar legal terms using cached Levenshtein distance
   */
  private findSimilarLegalTerms(word: string): Array<{ term: string; similarity: number }> {
    const results: Array<{ term: string; similarity: number }> = [];
    const maxDistance = Math.floor(word.length * 0.3); // Allow 30% character differences
    
    for (const term of this.legalTerms) {
      if (Math.abs(term.length - word.length) > maxDistance) continue;
      
      const distance = this.getCachedLevenshteinDistance(word, term);
      if (distance <= maxDistance) {
        const similarity = 1 - (distance / Math.max(word.length, term.length));
        if (similarity >= 0.6) { // 60% similarity threshold
          results.push({ term, similarity });
        }
      }
    }
    
    return results.sort((a, b) => b.similarity - a.similarity).slice(0, 5);
  }

  /**
   * Get cached Levenshtein distance with memoization
   */
  private getCachedLevenshteinDistance(a: string, b: string): number {
    if (!this.levenshteinCache.has(a)) {
      this.levenshteinCache.set(a, new Map());
    }
    
    const aCache = this.levenshteinCache.get(a)!;
    if (aCache.has(b)) {
      return aCache.get(b)!;
    }
    
    const distance = this.calculateLevenshteinDistance(a, b);
    aCache.set(b, distance);
    
    // Prune cache if it gets too large
    if (this.levenshteinCache.size > 10000) {
      this.pruneLevenshteinCache();
    }
    
    return distance;
  }

  /**
   * Calculate Levenshtein distance between two strings
   */
  private calculateLevenshteinDistance(a: string, b: string): number {
    const matrix: number[][] = [];
    
    for (let i = 0; i <= b.length; i++) {
      matrix[i] = [i];
    }
    
    for (let j = 0; j <= a.length; j++) {
      matrix[0][j] = j;
    }
    
    for (let i = 1; i <= b.length; i++) {
      for (let j = 1; j <= a.length; j++) {
        if (b.charAt(i - 1) === a.charAt(j - 1)) {
          matrix[i][j] = matrix[i - 1][j - 1];
        } else {
          matrix[i][j] = Math.min(
            matrix[i - 1][j - 1] + 1, // substitution
            matrix[i][j - 1] + 1,     // insertion
            matrix[i - 1][j] + 1      // deletion
          );
        }
      }
    }
    
    return matrix[b.length][a.length];
  }

  /**
   * Prune Levenshtein distance cache to prevent memory bloat
   */
  private pruneLevenshteinCache(): void {
    const keys = Array.from(this.levenshteinCache.keys());
    // Remove oldest entries (simple FIFO)
    const toRemove = keys.slice(0, keys.length / 2);
    
    for (const key of toRemove) {
      this.levenshteinCache.delete(key);
    }
  }

  /**
   * Get surrounding context for a word
   */
  private getWordContext(word: string, content: string, contextSize: number = 50): string {
    const index = content.toLowerCase().indexOf(word.toLowerCase());
    if (index === -1) return '';
    
    const start = Math.max(0, index - contextSize);
    const end = Math.min(content.length, index + word.length + contextSize);
    
    return content.substring(start, end).trim();
  }

  /**
   * Map numeric entity type to string
   */
  private mapEntityType(typeId: number): ExtractedEntity['type'] {
    const typeMap: Record<number, ExtractedEntity['type']> = {
      0: 'person',
      1: 'organization',
      2: 'location',
      3: 'date',
      4: 'legal_term',
      5: 'case_reference'
    };
    
    return typeMap[typeId] || 'legal_term';
  }

  /**
   * Update SIMD performance statistics
   */
  private updateSIMDStats(processingTime: number, contentLength: number): void {
    const operationsPerSecond = contentLength / (processingTime / 1000);
    const parallelizationFactor = this.config.simdOptimization ? 4 : 1; // Estimate
    
    this.simdStats = {
      vectorOperationsPerSecond: operationsPerSecond,
      parallelizationFactor,
      memoryBandwidth: (contentLength * 4) / (processingTime / 1000) / 1024 / 1024, // MB/s
      cacheHitRate: this.calculateCacheHitRate(),
      averageLatency: processingTime
    };
  }

  /**
   * Calculate cache hit rate for performance metrics
   */
  private calculateCacheHitRate(): number {
    const totalCacheQueries = Array.from(this.levenshteinCache.values())
      .reduce((sum, cache) => sum + cache.size, 0);
    
    return totalCacheQueries > 0 ? 0.8 : 0; // Estimate - would track actual hits/misses
  }

  /**
   * Get SIMD performance statistics
   */
  getSIMDStats(): SIMDStats {
    return { ...this.simdStats };
  }

  /**
   * Cleanup resources
   */
  async cleanup(): Promise<void> {
    console.log('🧹 Cleaning up SIMD GPU parser...');
    
    // Release GPU buffers
    for (const [name, buffer] of this.gpuBuffers) {
      buffer.destroy();
    }
    this.gpuBuffers.clear();
    
    // Clear caches
    this.levenshteinCache.clear();
    
    console.log('✅ SIMD GPU parser cleanup completed');
  }
}

export default SIMDGPUParserIntegration;