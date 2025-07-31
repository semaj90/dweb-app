/**
 * Document Processor Service
 * Handles document ingestion, parsing, chunking, and indexing
 */

import pdfParse from 'pdf-parse';
import mammoth from 'mammoth';
import { createWorker } from 'tesseract.js';
import sharp from 'sharp';
import * as cheerio from 'cheerio';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import path from 'path';
import fs from 'fs/promises';

export class DocumentProcessor {
  constructor(services) {
    this.ollama = services.ollama;
    this.vector = services.vector;
    this.cache = services.cache;
    
    // Initialize text splitter
    this.textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
      separators: ['\n\n', '\n', '. ', ' ', '']
    });

    // OCR worker (lazy initialization)
    this.ocrWorker = null;
  }

  /**
   * Process uploaded document
   */
  async processDocument(filePath, metadata = {}) {
    try {
      console.log(`📄 Processing document: ${filePath}`);
      
      // Extract file information
      const fileInfo = await this.getFileInfo(filePath);
      const fileExtension = path.extname(filePath).toLowerCase();

      // Extract text based on file type
      let extractedText = '';
      let extractedMetadata = {};

      switch (fileExtension) {
        case '.pdf':
          const pdfResult = await this.processPDF(filePath);
          extractedText = pdfResult.text;
          extractedMetadata = pdfResult.metadata;
          break;
        
        case '.docx':
        case '.doc':
          const docResult = await this.processWord(filePath);
          extractedText = docResult.text;
          extractedMetadata = docResult.metadata;
          break;
        
        case '.txt':
          extractedText = await this.processText(filePath);
          break;
        
        case '.html':
        case '.htm':
          const htmlResult = await this.processHTML(filePath);
          extractedText = htmlResult.text;
          extractedMetadata = htmlResult.metadata;
          break;
        
        case '.jpg':
        case '.jpeg':
        case '.png':
        case '.tiff':
        case '.bmp':
          extractedText = await this.processImage(filePath);
          break;
        
        default:
          throw new Error(`Unsupported file type: ${fileExtension}`);
      }

      if (!extractedText || extractedText.trim().length === 0) {
        throw new Error('No text content extracted from document');
      }

      // Create document record
      const documentData = {
        title: metadata.title || path.basename(filePath),
        content: extractedText,
        filePath: filePath,
        fileType: fileExtension.slice(1),
        fileSize: fileInfo.size,
        documentType: metadata.documentType || 'general',
        caseId: metadata.caseId,
        metadata: {
          ...extractedMetadata,
          ...metadata,
          processingDate: new Date().toISOString(),
          wordCount: extractedText.split(/\s+/).length,
          characterCount: extractedText.length
        }
      };

      // Add document with embedding
      const document = await this.vector.addDocument(documentData);

      // Create and process chunks
      const chunks = await this.createDocumentChunks(extractedText, {
        documentId: document.id,
        metadata: document.metadata
      });

      // Add chunks with embeddings
      const chunkResults = await this.vector.addDocumentChunks(document.id, chunks);

      // Generate AI summary and analysis
      const aiAnalysis = await this.generateAIAnalysis(extractedText, metadata);

      // Cache processing results
      await this.cache.cacheDocumentProcessing(document.id, {
        document,
        chunks: chunkResults,
        aiAnalysis
      });

      console.log(`✅ Document processed successfully: ${document.id}`);
      console.log(`📊 Created ${chunkResults.length} chunks`);

      return {
        document,
        chunks: chunkResults,
        aiAnalysis,
        stats: {
          originalSize: fileInfo.size,
          textLength: extractedText.length,
          chunkCount: chunkResults.length,
          processingTime: Date.now() - performance.now()
        }
      };

    } catch (error) {
      console.error('Document processing failed:', error);
      throw error;
    }
  }

  /**
   * Process PDF document
   */
  async processPDF(filePath) {
    try {
      const buffer = await fs.readFile(filePath);
      const data = await pdfParse(buffer);

      return {
        text: data.text,
        metadata: {
          pages: data.numpages,
          info: data.info,
          version: data.version
        }
      };
    } catch (error) {
      console.error('PDF processing failed:', error);
      throw error;
    }
  }

  /**
   * Process Word document
   */
  async processWord(filePath) {
    try {
      const buffer = await fs.readFile(filePath);
      const result = await mammoth.extractRawText({ buffer });

      return {
        text: result.value,
        metadata: {
          messages: result.messages
        }
      };
    } catch (error) {
      console.error('Word document processing failed:', error);
      throw error;
    }
  }

  /**
   * Process text file
   */
  async processText(filePath) {
    try {
      return await fs.readFile(filePath, 'utf-8');
    } catch (error) {
      console.error('Text file processing failed:', error);
      throw error;
    }
  }

  /**
   * Process HTML file
   */
  async processHTML(filePath) {
    try {
      const html = await fs.readFile(filePath, 'utf-8');
      const $ = cheerio.load(html);
      
      // Remove script and style elements
      $('script, style').remove();
      
      // Extract text content
      const text = $('body').text().replace(/\s+/g, ' ').trim();
      
      // Extract metadata
      const title = $('title').text();
      const metaDescription = $('meta[name="description"]').attr('content');
      const metaKeywords = $('meta[name="keywords"]').attr('content');

      return {
        text,
        metadata: {
          title,
          description: metaDescription,
          keywords: metaKeywords
        }
      };
    } catch (error) {
      console.error('HTML processing failed:', error);
      throw error;
    }
  }

  /**
   * Process image with OCR
   */
  async processImage(filePath) {
    try {
      // Initialize OCR worker if not already done
      if (!this.ocrWorker) {
        this.ocrWorker = await createWorker('eng');
      }

      // Optimize image for OCR
      const optimizedBuffer = await sharp(filePath)
        .resize({ width: 2000, withoutEnlargement: true })
        .sharpen()
        .normalize()
        .toBuffer();

      // Perform OCR
      const { data: { text } } = await this.ocrWorker.recognize(optimizedBuffer);

      return text;
    } catch (error) {
      console.error('Image OCR processing failed:', error);
      throw error;
    }
  }

  /**
   * Create document chunks
   */
  async createDocumentChunks(text, options = {}) {
    try {
      const {
        documentId,
        metadata = {},
        chunkSize = 1000,
        chunkOverlap = 200
      } = options;

      // Configure text splitter
      const splitter = new RecursiveCharacterTextSplitter({
        chunkSize,
        chunkOverlap,
        separators: ['\n\n', '\n', '. ', ' ', '']
      });

      // Split text into chunks
      const textChunks = await splitter.splitText(text);

      // Create chunk objects with metadata
      const chunks = textChunks.map((chunk, index) => ({
        content: chunk,
        metadata: {
          ...metadata,
          chunkIndex: index,
          chunkSize: chunk.length,
          startChar: textChunks.slice(0, index).join('').length,
          endChar: textChunks.slice(0, index + 1).join('').length
        }
      }));

      console.log(`📝 Created ${chunks.length} chunks from document`);
      return chunks;
    } catch (error) {
      console.error('Chunk creation failed:', error);
      throw error;
    }
  }

  /**
   * Generate AI analysis of document
   */
  async generateAIAnalysis(text, metadata = {}) {
    try {
      // Check cache first
      const cacheKey = `ai_analysis:${Buffer.from(text.substring(0, 1000)).toString('base64')}`;
      const cached = await this.cache.get(cacheKey);
      if (cached) {
        console.log('🔄 Using cached AI analysis');
        return cached;
      }

      // Determine analysis type based on metadata
      const analysisType = metadata.documentType === 'contract' ? 'contract' :
                          metadata.documentType === 'evidence' ? 'evidence' :
                          metadata.documentType === 'legal' ? 'litigation' :
                          'general';

      // Generate AI analysis
      const analysis = await this.ollama.analyzeLegalDocument(text, analysisType, {
        maxTokens: 1024
      });

      // Generate summary
      const summary = await this.ollama.summarizeText(text, {
        summaryLength: 'medium',
        maxTokens: 512
      });

      // Extract key information
      const entities = await this.ollama.extractKeyInfo(text, 'entities', {
        maxTokens: 512
      });

      const keyDates = await this.ollama.extractKeyInfo(text, 'dates', {
        maxTokens: 256
      });

      const result = {
        analysis: analysis.analysis,
        summary: summary.summary,
        entities: entities.extractedInfo,
        keyDates: keyDates.extractedInfo,
        analysisType,
        confidence: this.calculateAnalysisConfidence(analysis, summary),
        processingTime: analysis.processingTime,
        generatedAt: new Date().toISOString()
      };

      // Cache the result
      await this.cache.set(cacheKey, result, 86400); // 24 hours

      return result;
    } catch (error) {
      console.error('AI analysis failed:', error);
      return {
        analysis: 'AI analysis failed',
        summary: 'Summary generation failed',
        entities: 'Entity extraction failed',
        keyDates: 'Date extraction failed',
        error: error.message,
        generatedAt: new Date().toISOString()
      };
    }
  }

  /**
   * Calculate confidence score for AI analysis
   */
  calculateAnalysisConfidence(analysis, summary) {
    let confidence = 0.5; // Base confidence

    // Check if analysis contains meaningful content
    if (analysis.analysis && analysis.analysis.length > 100) {
      confidence += 0.2;
    }

    // Check if summary is coherent
    if (summary.summary && summary.summary.length > 50) {
      confidence += 0.2;
    }

    // Check processing time (faster usually means more confident)
    if (analysis.processingTime && analysis.processingTime < 10000) {
      confidence += 0.1;
    }

    return Math.min(confidence, 1.0);
  }

  /**
   * Batch process multiple documents
   */
  async batchProcessDocuments(filePaths, metadata = {}) {
    const results = [];
    const errors = [];

    console.log(`📦 Starting batch processing of ${filePaths.length} documents`);

    for (let i = 0; i < filePaths.length; i++) {
      const filePath = filePaths[i];
      
      try {
        console.log(`📄 Processing ${i + 1}/${filePaths.length}: ${filePath}`);
        
        const result = await this.processDocument(filePath, {
          ...metadata,
          batchIndex: i,
          batchTotal: filePaths.length
        });
        
        results.push({
          filePath,
          success: true,
          result
        });

      } catch (error) {
        console.error(`❌ Failed to process ${filePath}:`, error);
        
        errors.push({
          filePath,
          error: error.message
        });
        
        results.push({
          filePath,
          success: false,
          error: error.message
        });
      }
    }

    console.log(`✅ Batch processing complete: ${results.filter(r => r.success).length} successful, ${errors.length} failed`);

    return {
      results,
      errors,
      stats: {
        total: filePaths.length,
        successful: results.filter(r => r.success).length,
        failed: errors.length,
        successRate: (results.filter(r => r.success).length / filePaths.length * 100).toFixed(1) + '%'
      }
    };
  }

  /**
   * Get file information
   */
  async getFileInfo(filePath) {
    try {
      const stats = await fs.stat(filePath);
      return {
        size: stats.size,
        created: stats.birthtime,
        modified: stats.mtime,
        isFile: stats.isFile(),
        isDirectory: stats.isDirectory()
      };
    } catch (error) {
      console.error('Failed to get file info:', error);
      throw error;
    }
  }

  /**
   * Reprocess document with new settings
   */
  async reprocessDocument(documentId, options = {}) {
    try {
      // Get existing document
      const document = await this.vector.db.getDocument(documentId);
      if (!document) {
        throw new Error('Document not found');
      }

      // Clear existing cache
      await this.cache.delete(`ai_analysis:${documentId}`);

      // Reprocess with new options
      const result = await this.processDocument(document.file_path, {
        ...document.metadata,
        ...options,
        reprocessing: true
      });

      console.log(`♻️ Document reprocessed: ${documentId}`);
      return result;
    } catch (error) {
      console.error('Document reprocessing failed:', error);
      throw error;
    }
  }

  /**
   * Get processing statistics
   */
  async getProcessingStats() {
    try {
      const dbStats = await this.vector.db.getHealthStats();
      
      return {
        documentsProcessed: dbStats.totalDocuments,
        chunksCreated: dbStats.totalChunks,
        indexedDocuments: dbStats.indexedDocuments,
        averageProcessingTime: 'N/A', // Would need to track this
        supportedFormats: ['.pdf', '.docx', '.doc', '.txt', '.html', '.htm', '.jpg', '.jpeg', '.png', '.tiff', '.bmp'],
        ocrEnabled: this.ocrWorker !== null,
        aiAnalysisEnabled: this.ollama.isHealthy
      };
    } catch (error) {
      console.error('Failed to get processing stats:', error);
      return {
        error: error.message
      };
    }
  }

  /**
   * Cleanup OCR worker
   */
  async cleanup() {
    if (this.ocrWorker) {
      try {
        await this.ocrWorker.terminate();
        this.ocrWorker = null;
        console.log('✅ OCR worker terminated');
      } catch (error) {
        console.error('Error terminating OCR worker:', error);
      }
    }
  }
}