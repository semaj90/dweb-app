// Advanced text chunking with overlap and structure preservation
// Optimized for legal documents with section awareness

export class ChunkSplitter {
  constructor(options = {}) {
    this.chunkSize = options.chunkSize || 500; // tokens
    this.overlap = options.overlap || 50; // overlap tokens
    this.preserveStructure = options.preserveStructure ?? true;
    this.minChunkSize = options.minChunkSize || 50;
    
    // Legal document section patterns
    this.sectionPatterns = [
      /^(ARTICLE|SECTION|CHAPTER)\s+[IVXLCDM]+/i,
      /^\d+\.\s+/,  // Numbered sections
      /^[A-Z][^a-z]*:$/,  // ALL CAPS headers
      /^WHEREAS/i,
      /^NOW THEREFORE/i,
      /^IN WITNESS WHEREOF/i,
    ];
  }

  // Estimate token count (rough approximation)
  estimateTokens(text) {
    return Math.ceil(text.split(/\s+/).length * 1.3); // Account for punctuation
  }

  // Split text into sentences while preserving structure
  splitSentences(text) {
    // Improved sentence splitting for legal text
    const sentences = text.split(/(?<=[.!?])\s+(?=[A-Z])/);
    return sentences.filter(s => s.trim().length > 0);
  }

  // Detect if line is a section header
  isSection(line) {
    return this.sectionPatterns.some(pattern => pattern.test(line.trim()));
  }

  // Create hierarchical chunks with overlap
  async splitText(text, metadata = {}) {
    const chunks = [];
    const paragraphs = text.split(/\n\s*\n/).filter(p => p.trim().length > 0);
    
    let currentChunk = '';
    let currentTokens = 0;
    let chunkIndex = 0;
    let currentSection = null;
    
    for (let i = 0; i < paragraphs.length; i++) {
      const paragraph = paragraphs[i].trim();
      const paragraphTokens = this.estimateTokens(paragraph);
      
      // Check if this is a section header
      if (this.preserveStructure && this.isSection(paragraph)) {
        // Finish current chunk if it exists
        if (currentChunk.trim()) {
          chunks.push(this.createChunk(
            currentChunk.trim(),
            chunkIndex++,
            currentTokens,
            { section: currentSection, ...metadata }
          ));
          currentChunk = '';
          currentTokens = 0;
        }
        
        currentSection = paragraph;
      }
      
      // Check if adding this paragraph would exceed chunk size
      if (currentTokens + paragraphTokens > this.chunkSize && currentChunk.trim()) {
        // Create chunk with current content
        chunks.push(this.createChunk(
          currentChunk.trim(),
          chunkIndex++,
          currentTokens,
          { section: currentSection, ...metadata }
        ));
        
        // Start new chunk with overlap
        const overlapText = this.createOverlap(currentChunk, this.overlap);
        currentChunk = overlapText + '\n\n' + paragraph;
        currentTokens = this.estimateTokens(currentChunk);
      } else {
        // Add paragraph to current chunk
        if (currentChunk) {
          currentChunk += '\n\n' + paragraph;
        } else {
          currentChunk = paragraph;
        }
        currentTokens += paragraphTokens;
      }
    }
    
    // Add final chunk if it exists
    if (currentChunk.trim() && this.estimateTokens(currentChunk) >= this.minChunkSize) {
      chunks.push(this.createChunk(
        currentChunk.trim(),
        chunkIndex,
        currentTokens,
        { section: currentSection, ...metadata }
      ));
    }
    
    // Post-process chunks to add positional information
    return this.addPositionalMetadata(chunks, text);
  }

  // Create overlap text from end of previous chunk
  createOverlap(text, overlapTokens) {
    const sentences = this.splitSentences(text);
    let overlapText = '';
    let overlapCount = 0;
    
    // Take last sentences up to overlap token limit
    for (let i = sentences.length - 1; i >= 0 && overlapCount < overlapTokens; i--) {
      const sentence = sentences[i];
      const sentenceTokens = this.estimateTokens(sentence);
      
      if (overlapCount + sentenceTokens <= overlapTokens) {
        overlapText = sentence + ' ' + overlapText;
        overlapCount += sentenceTokens;
      } else {
        break;
      }
    }
    
    return overlapText.trim();
  }

  // Create chunk object with metadata
  createChunk(text, index, tokens, metadata = {}) {
    return {
      text,
      index,
      tokens,
      metadata: {
        ...metadata,
        length: text.length,
        sentences: this.splitSentences(text).length,
        createdAt: new Date().toISOString(),
      }
    };
  }

  // Add positional metadata to chunks
  addPositionalMetadata(chunks, originalText) {
    return chunks.map(chunk => {
      const startOffset = originalText.indexOf(chunk.text.substring(0, 50));
      const endOffset = startOffset + chunk.text.length;
      
      return {
        ...chunk,
        startOffset: startOffset >= 0 ? startOffset : 0,
        endOffset: startOffset >= 0 ? endOffset : chunk.text.length,
        metadata: {
          ...chunk.metadata,
          totalChunks: chunks.length,
          relativePosition: chunk.index / chunks.length,
        }
      };
    });
  }

  // Hierarchical chunking - create parent chunks from child chunks
  async createHierarchicalChunks(chunks, parentChunkSize = 1000) {
    const parentChunks = [];
    let currentParent = '';
    let currentTokens = 0;
    let childChunkIds = [];
    
    for (const chunk of chunks) {
      if (currentTokens + chunk.tokens > parentChunkSize && currentParent) {
        parentChunks.push({
          text: currentParent.trim(),
          tokens: currentTokens,
          level: 1,
          childChunkIds: [...childChunkIds],
          metadata: {
            type: 'parent',
            childCount: childChunkIds.length,
          }
        });
        
        currentParent = '';
        currentTokens = 0;
        childChunkIds = [];
      }
      
      currentParent += chunk.text + '\n\n';
      currentTokens += chunk.tokens;
      childChunkIds.push(chunk.index);
    }
    
    // Add final parent chunk
    if (currentParent.trim()) {
      parentChunks.push({
        text: currentParent.trim(),
        tokens: currentTokens,
        level: 1,
        childChunkIds,
        metadata: {
          type: 'parent',
          childCount: childChunkIds.length,
        }
      });
    }
    
    return parentChunks;
  }

  // Split by specific patterns (useful for contracts, legal docs)
  splitByPattern(text, patterns = []) {
    if (!patterns.length) {
      return [text];
    }
    
    const combinedPattern = new RegExp(patterns.map(p => `(${p.source})`).join('|'), 'gi');
    const splits = text.split(combinedPattern).filter(s => s && s.trim());
    
    return splits.map(split => split.trim()).filter(s => s.length > 0);
  }

  // Adaptive chunking based on content density
  async adaptiveChunk(text, metadata = {}) {
    const sentences = this.splitSentences(text);
    const chunks = [];
    let currentChunk = '';
    let currentTokens = 0;
    let chunkIndex = 0;
    
    for (const sentence of sentences) {
      const sentenceTokens = this.estimateTokens(sentence);
      
      // Dynamic chunk size based on sentence complexity
      const adaptiveSize = this.calculateAdaptiveSize(sentence, metadata);
      
      if (currentTokens + sentenceTokens > adaptiveSize && currentChunk.trim()) {
        chunks.push(this.createChunk(
          currentChunk.trim(),
          chunkIndex++,
          currentTokens,
          { ...metadata, adaptive: true, chunkSize: adaptiveSize }
        ));
        
        currentChunk = sentence;
        currentTokens = sentenceTokens;
      } else {
        currentChunk += (currentChunk ? ' ' : '') + sentence;
        currentTokens += sentenceTokens;
      }
    }
    
    if (currentChunk.trim()) {
      chunks.push(this.createChunk(
        currentChunk.trim(),
        chunkIndex,
        currentTokens,
        { ...metadata, adaptive: true }
      ));
    }
    
    return chunks;
  }

  // Calculate adaptive chunk size based on content
  calculateAdaptiveSize(sentence, metadata) {
    let size = this.chunkSize;
    
    // Increase chunk size for dense legal text
    if (/\b(whereas|therefore|herein|thereof|hereby)\b/i.test(sentence)) {
      size *= 1.2;
    }
    
    // Decrease for lists or simple statements
    if (/^\d+\.|^[a-z]\)|^-/.test(sentence.trim())) {
      size *= 0.8;
    }
    
    // Adjust based on document type
    if (metadata.documentType === 'contract') {
      size *= 1.1; // Larger chunks for contracts
    } else if (metadata.documentType === 'evidence') {
      size *= 0.9; // Smaller chunks for evidence
    }
    
    return Math.max(Math.round(size), this.minChunkSize);
  }
}