const { Worker, isMainThread, parentPort, workerData } = require('worker_threads');
const path = require('path');
const fs = require('fs').promises;

/**
 * Document Processing Service Worker
 * Handles legal document parsing, analysis, and vector embedding preparation
 */
class DocumentProcessorWorker {
  constructor(data) {
    this.workerId = data.workerId;
    this.services = data.services;
    this.processedCount = 0;
    this.startTime = Date.now();
    
    // Document processing pipeline configuration
    this.pipeline = {
      maxFileSize: 50 * 1024 * 1024, // 50MB
      supportedTypes: ['.pdf', '.docx', '.txt', '.md', '.rtf'],
      chunkSize: 1000, // Characters per chunk
      overlapSize: 200, // Overlap between chunks
      maxChunks: 1000 // Maximum chunks per document
    };
    
    this.init();
  }
  
  async init() {
    console.log(`[DOC-PROCESSOR-${this.workerId}] Document processor worker starting`);
    
    // Setup message handling
    this.setupMessageHandling();
    
    // Initialize document parsing libraries
    await this.initializeProcessors();
    
    // Setup periodic cleanup
    setInterval(() => {
      this.performCleanup();
    }, 300000); // 5 minutes
    
    this.sendMessage({
      type: 'worker-ready',
      worker: 'document-processor',
      pid: process.pid
    });
  }
  
  setupMessageHandling() {
    parentPort.on('message', async (message) => {
      try {
        await this.handleMessage(message);
      } catch (error) {
        console.error(`[DOC-PROCESSOR-${this.workerId}] Message handling error:`, error);
        this.sendMessage({
          type: 'error',
          worker: 'document-processor',
          error: error.message,
          timestamp: Date.now()
        });
      }
    });
  }
  
  async handleMessage(message) {
    switch (message.type) {
      case 'process-document':
        await this.processDocument(message.data);
        break;
        
      case 'process-batch':
        await this.processBatch(message.data);
        break;
        
      case 'extract-text':
        await this.extractText(message.data);
        break;
        
      case 'chunk-document':
        await this.chunkDocument(message.data);
        break;
        
      case 'analyze-metadata':
        await this.analyzeMetadata(message.data);
        break;
        
      case 'memory-cleanup':
        await this.performCleanup();
        break;
        
      case 'health-check':
        this.sendHealthReport();
        break;
        
      default:
        console.log(`[DOC-PROCESSOR-${this.workerId}] Unknown message type: ${message.type}`);
    }
  }
  
  async initializeProcessors() {
    try {
      // Initialize PDF processing
      this.pdfParse = require('pdf-parse');
      
      // Initialize DOCX processing
      this.mammoth = require('mammoth');
      
      // Initialize natural language processing
      this.natural = require('natural');
      this.tokenizer = new this.natural.WordTokenizer();
      this.stemmer = this.natural.PorterStemmer;
      
      console.log(`[DOC-PROCESSOR-${this.workerId}] Document processors initialized`);
      
    } catch (error) {
      console.error(`[DOC-PROCESSOR-${this.workerId}] Failed to initialize processors:`, error);
      // Continue with limited functionality
    }
  }
  
  async processDocument(documentData) {
    const startTime = Date.now();
    
    try {
      console.log(`[DOC-PROCESSOR-${this.workerId}] Processing document: ${documentData.filename}`);
      
      // Validate document
      const validation = await this.validateDocument(documentData);
      if (!validation.valid) {
        throw new Error(`Document validation failed: ${validation.reason}`);
      }
      
      // Extract text content
      const textData = await this.extractText(documentData);
      
      // Analyze document metadata
      const metadata = await this.analyzeMetadata(documentData, textData);
      
      // Chunk document for vector processing
      const chunks = await this.chunkDocument(textData, metadata);
      
      // Prepare for legal analysis
      const legalAnalysis = await this.prepareLegalAnalysis(textData, metadata);
      
      const result = {
        documentId: documentData.documentId,
        filename: documentData.filename,
        textContent: textData.content,
        metadata: metadata,
        chunks: chunks,
        legalAnalysis: legalAnalysis,
        processingTime: Date.now() - startTime,
        processedBy: this.workerId,
        timestamp: Date.now()
      };
      
      this.processedCount++;
      
      this.sendMessage({
        type: 'document-processed',
        data: result
      });
      
      console.log(`[DOC-PROCESSOR-${this.workerId}] Document processed in ${result.processingTime}ms`);
      
    } catch (error) {
      console.error(`[DOC-PROCESSOR-${this.workerId}] Document processing error:`, error);
      
      this.sendMessage({
        type: 'processing-error',
        data: {
          documentId: documentData.documentId,
          error: error.message,
          processingTime: Date.now() - startTime
        }
      });
    }
  }
  
  async validateDocument(documentData) {
    // Check file size
    if (documentData.size > this.pipeline.maxFileSize) {
      return {
        valid: false,
        reason: `File size ${documentData.size} exceeds maximum ${this.pipeline.maxFileSize}`
      };
    }
    
    // Check file type
    const ext = path.extname(documentData.filename).toLowerCase();
    if (!this.pipeline.supportedTypes.includes(ext)) {
      return {
        valid: false,
        reason: `File type ${ext} not supported. Supported types: ${this.pipeline.supportedTypes.join(', ')}`
      };
    }
    
    // Check if file exists and is readable
    try {
      if (documentData.filepath) {
        await fs.access(documentData.filepath, fs.constants.R_OK);
      }
    } catch (error) {
      return {
        valid: false,
        reason: `File not accessible: ${error.message}`
      };
    }
    
    return { valid: true };
  }
  
  async extractText(documentData) {
    const ext = path.extname(documentData.filename).toLowerCase();
    let content = '';
    let rawText = '';
    
    try {
      switch (ext) {
        case '.pdf':
          content = await this.extractPDFText(documentData);
          break;
          
        case '.docx':
          content = await this.extractDOCXText(documentData);
          break;
          
        case '.txt':
        case '.md':
          content = await this.extractPlainText(documentData);
          break;
          
        default:
          throw new Error(`Unsupported file type: ${ext}`);
      }
      
      // Clean and normalize text
      rawText = content;
      content = this.cleanText(content);
      
      return {
        content: content,
        rawText: rawText,
        length: content.length,
        wordCount: this.tokenizer.tokenize(content).length,
        extraction: {
          method: ext,
          success: true,
          timestamp: Date.now()
        }
      };
      
    } catch (error) {
      console.error(`[DOC-PROCESSOR-${this.workerId}] Text extraction error:`, error);
      throw new Error(`Text extraction failed: ${error.message}`);
    }
  }
  
  async extractPDFText(documentData) {
    if (!this.pdfParse) {
      throw new Error('PDF parser not available');
    }
    
    const buffer = documentData.buffer || await fs.readFile(documentData.filepath);
    const data = await this.pdfParse(buffer);
    
    return data.text;
  }
  
  async extractDOCXText(documentData) {
    if (!this.mammoth) {
      throw new Error('DOCX parser not available');
    }
    
    const buffer = documentData.buffer || await fs.readFile(documentData.filepath);
    const result = await this.mammoth.extractRawText({ buffer: buffer });
    
    return result.value;
  }
  
  async extractPlainText(documentData) {
    const buffer = documentData.buffer || await fs.readFile(documentData.filepath);
    return buffer.toString('utf-8');
  }
  
  cleanText(text) {
    // Remove excessive whitespace
    text = text.replace(/\s+/g, ' ');
    
    // Remove control characters
    text = text.replace(/[\x00-\x1F\x7F]/g, '');
    
    // Normalize quotes and dashes
    text = text.replace(/[""]/g, '"');
    text = text.replace(/['']/g, "'");
    text = text.replace(/[—–]/g, '-');
    
    // Trim whitespace
    text = text.trim();
    
    return text;
  }
  
  async analyzeMetadata(documentData, textData) {
    const tokens = this.tokenizer.tokenize(textData.content);
    const uniqueWords = [...new Set(tokens.map(token => token.toLowerCase()))];
    
    // Calculate readability metrics
    const sentences = textData.content.split(/[.!?]+/).length;
    const avgWordsPerSentence = tokens.length / sentences;
    const avgCharsPerWord = tokens.reduce((sum, word) => sum + word.length, 0) / tokens.length;
    
    // Detect legal terminology
    const legalTerms = this.detectLegalTerms(tokens);
    
    // Calculate document complexity
    const complexity = this.calculateComplexity(tokens, sentences, legalTerms);
    
    return {
      filename: documentData.filename,
      fileType: path.extname(documentData.filename),
      fileSize: documentData.size,
      textLength: textData.length,
      wordCount: tokens.length,
      uniqueWords: uniqueWords.length,
      sentenceCount: sentences,
      avgWordsPerSentence: Math.round(avgWordsPerSentence * 100) / 100,
      avgCharsPerWord: Math.round(avgCharsPerWord * 100) / 100,
      readabilityScore: this.calculateReadabilityScore(tokens.length, sentences, complexity),
      legalTerms: legalTerms,
      complexity: complexity,
      language: 'en', // Could be enhanced with language detection
      encoding: 'utf-8',
      processedAt: Date.now(),
      processedBy: this.workerId
    };
  }
  
  detectLegalTerms(tokens) {
    const legalKeywords = [
      'contract', 'agreement', 'clause', 'provision', 'statute', 'regulation',
      'defendant', 'plaintiff', 'court', 'judgment', 'liability', 'damages',
      'jurisdiction', 'precedent', 'appeal', 'motion', 'discovery', 'deposition',
      'subpoena', 'evidence', 'testimony', 'witness', 'attorney', 'counsel',
      'breach', 'warranty', 'indemnification', 'arbitration', 'litigation'
    ];
    
    const foundTerms = [];
    const normalizedTokens = tokens.map(token => token.toLowerCase());
    
    legalKeywords.forEach(term => {
      const count = normalizedTokens.filter(token => token.includes(term)).length;
      if (count > 0) {
        foundTerms.push({ term, count });
      }
    });
    
    return foundTerms.sort((a, b) => b.count - a.count);
  }
  
  calculateComplexity(tokens, sentences, legalTerms) {
    // Base complexity on various factors
    const avgWordLength = tokens.reduce((sum, word) => sum + word.length, 0) / tokens.length;
    const avgSentenceLength = tokens.length / sentences;
    const legalDensity = legalTerms.reduce((sum, term) => sum + term.count, 0) / tokens.length;
    
    // Weighted complexity score (0-10)
    const complexity = Math.min(10, 
      (avgWordLength * 0.5) + 
      (avgSentenceLength * 0.1) + 
      (legalDensity * 100)
    );
    
    return Math.round(complexity * 100) / 100;
  }
  
  calculateReadabilityScore(wordCount, sentenceCount, complexity) {
    // Simplified readability score (higher = easier to read)
    const avgWordsPerSentence = wordCount / sentenceCount;
    const baseScore = 100 - (avgWordsPerSentence * 2) - (complexity * 5);
    
    return Math.max(0, Math.min(100, Math.round(baseScore)));
  }
  
  async chunkDocument(textData, metadata) {
    const content = textData.content;
    const chunks = [];
    const chunkSize = this.pipeline.chunkSize;
    const overlap = this.pipeline.overlapSize;
    
    let start = 0;
    let chunkIndex = 0;
    
    while (start < content.length && chunkIndex < this.pipeline.maxChunks) {
      const end = Math.min(start + chunkSize, content.length);
      const chunkText = content.substring(start, end);
      
      // Find a good break point (sentence boundary)
      let breakPoint = end;
      if (end < content.length) {
        const lastSentence = chunkText.lastIndexOf('.');
        const lastQuestion = chunkText.lastIndexOf('?');
        const lastExclamation = chunkText.lastIndexOf('!');
        
        const lastPunctuation = Math.max(lastSentence, lastQuestion, lastExclamation);
        if (lastPunctuation > start + (chunkSize * 0.5)) {
          breakPoint = start + lastPunctuation + 1;
        }
      }
      
      const finalChunk = content.substring(start, breakPoint);
      
      chunks.push({
        index: chunkIndex,
        text: finalChunk.trim(),
        startChar: start,
        endChar: breakPoint,
        length: finalChunk.length,
        wordCount: this.tokenizer.tokenize(finalChunk).length,
        metadata: {
          documentId: metadata.documentId || 'unknown',
          chunkIndex: chunkIndex,
          totalChunks: 0, // Will be updated after processing
          overlap: chunkIndex > 0 ? overlap : 0
        }
      });
      
      start = breakPoint - overlap;
      chunkIndex++;
    }
    
    // Update total chunks count
    chunks.forEach(chunk => {
      chunk.metadata.totalChunks = chunks.length;
    });
    
    return chunks;
  }
  
  async prepareLegalAnalysis(textData, metadata) {
    const analysis = {
      documentType: this.classifyDocumentType(textData.content, metadata),
      keyEntities: this.extractLegalEntities(textData.content),
      contractElements: this.identifyContractElements(textData.content),
      riskFactors: this.identifyRiskFactors(textData.content),
      actionItems: this.extractActionItems(textData.content),
      priority: this.calculatePriority(metadata, textData.content)
    };
    
    return analysis;
  }
  
  classifyDocumentType(content, metadata) {
    const types = {
      'contract': ['agreement', 'contract', 'terms', 'conditions', 'party'],
      'legal_brief': ['court', 'motion', 'brief', 'argument', 'ruling'],
      'regulation': ['regulation', 'rule', 'statute', 'code', 'section'],
      'correspondence': ['dear', 'sincerely', 'letter', 'memo', 'regarding'],
      'filing': ['petition', 'complaint', 'answer', 'filing', 'docket']
    };
    
    const scores = {};
    const lowerContent = content.toLowerCase();
    
    Object.entries(types).forEach(([type, keywords]) => {
      scores[type] = keywords.reduce((score, keyword) => {
        const matches = (lowerContent.match(new RegExp(keyword, 'g')) || []).length;
        return score + matches;
      }, 0);
    });
    
    const topType = Object.entries(scores).reduce((a, b) => scores[a[0]] > scores[b[0]] ? a : b);
    
    return {
      primary: topType[0],
      confidence: topType[1] / content.split(' ').length,
      scores: scores
    };
  }
  
  extractLegalEntities(content) {
    // Simplified entity extraction
    const entities = {
      parties: this.extractPattern(content, /(?:party|parties?|plaintiff|defendant|company|corporation)\s+([A-Z][a-z\s]+)/gi),
      dates: this.extractPattern(content, /(\b\d{1,2}\/\d{1,2}\/\d{4}\b|\b\d{1,2}-\d{1,2}-\d{4}\b)/g),
      amounts: this.extractPattern(content, /\$[\d,]+(?:\.\d{2})?/g),
      sections: this.extractPattern(content, /Section\s+(\d+(?:\.\d+)*)/gi)
    };
    
    return entities;
  }
  
  extractPattern(content, pattern) {
    const matches = content.match(pattern) || [];
    return [...new Set(matches)].slice(0, 10); // Limit to 10 unique matches
  }
  
  identifyContractElements(content) {
    const elements = {
      hasSignature: /signature|signed|executed/i.test(content),
      hasEffectiveDate: /effective\s+date|effective\s+as\s+of/i.test(content),
      hasTermination: /termination|expire|end\s+date/i.test(content),
      hasGoverningLaw: /governing\s+law|governed\s+by/i.test(content),
      hasDispute: /dispute|arbitration|litigation/i.test(content),
      hasConfidentiality: /confidential|non-disclosure|nda/i.test(content)
    };
    
    return elements;
  }
  
  identifyRiskFactors(content) {
    const riskKeywords = [
      'breach', 'default', 'penalty', 'damages', 'liability', 'indemnify',
      'terminate', 'void', 'invalid', 'dispute', 'lawsuit', 'claim'
    ];
    
    const risks = [];
    const lowerContent = content.toLowerCase();
    
    riskKeywords.forEach(keyword => {
      if (lowerContent.includes(keyword)) {
        const matches = (lowerContent.match(new RegExp(keyword, 'g')) || []).length;
        risks.push({ keyword, frequency: matches });
      }
    });
    
    return risks.sort((a, b) => b.frequency - a.frequency);
  }
  
  extractActionItems(content) {
    const actionPatterns = [
      /shall\s+([^.]+)/gi,
      /must\s+([^.]+)/gi,
      /will\s+([^.]+)/gi,
      /required\s+to\s+([^.]+)/gi
    ];
    
    const actions = [];
    
    actionPatterns.forEach(pattern => {
      const matches = content.match(pattern) || [];
      matches.slice(0, 5).forEach(match => { // Limit to 5 per pattern
        actions.push(match.trim());
      });
    });
    
    return [...new Set(actions)]; // Remove duplicates
  }
  
  calculatePriority(metadata, content) {
    let priority = 5; // Base priority (1-10 scale)
    
    // Increase priority for legal documents
    if (metadata.legalTerms.length > 10) priority += 2;
    
    // Increase priority for complex documents
    if (metadata.complexity > 7) priority += 1;
    
    // Increase priority for contracts
    if (content.toLowerCase().includes('contract') || content.toLowerCase().includes('agreement')) {
      priority += 2;
    }
    
    // Increase priority for time-sensitive content
    if (/urgent|immediate|asap|deadline/i.test(content)) {
      priority += 3;
    }
    
    return Math.min(10, priority);
  }
  
  async processBatch(batchData) {
    console.log(`[DOC-PROCESSOR-${this.workerId}] Processing batch of ${batchData.documents.length} documents`);
    
    const results = [];
    
    for (const document of batchData.documents) {
      try {
        await this.processDocument(document);
      } catch (error) {
        console.error(`[DOC-PROCESSOR-${this.workerId}] Batch processing error for ${document.filename}:`, error);
        results.push({
          documentId: document.documentId,
          error: error.message,
          status: 'failed'
        });
      }
    }
    
    this.sendMessage({
      type: 'batch-processed',
      data: {
        batchId: batchData.batchId,
        totalDocuments: batchData.documents.length,
        results: results,
        timestamp: Date.now()
      }
    });
  }
  
  performCleanup() {
    // Force garbage collection if available
    if (global.gc) {
      global.gc();
    }
    
    console.log(`[DOC-PROCESSOR-${this.workerId}] Cleanup completed. Memory usage:`, process.memoryUsage());
  }
  
  sendHealthReport() {
    const health = {
      worker: 'document-processor',
      workerId: this.workerId,
      pid: process.pid,
      uptime: Date.now() - this.startTime,
      processedCount: this.processedCount,
      memoryUsage: process.memoryUsage(),
      timestamp: Date.now()
    };
    
    this.sendMessage({
      type: 'health-report',
      data: health
    });
  }
  
  sendMessage(message) {
    try {
      parentPort.postMessage(message);
    } catch (error) {
      console.error(`[DOC-PROCESSOR-${this.workerId}] Failed to send message:`, error);
    }
  }
}

// Initialize worker if running in worker thread
if (!isMainThread) {
  new DocumentProcessorWorker(workerData);
}

module.exports = DocumentProcessorWorker;