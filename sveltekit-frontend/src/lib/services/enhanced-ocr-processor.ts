/**
 * Enhanced OCR Processor Service - Legal AI Platform
 * Advanced document processing with OCR, PDF conversion, and legal document optimization
 * Integrates with Tesseract.js, Sharp, and custom legal document analysis
 */

import sharp from "sharp";
import fs from "fs/promises";
import path from "path";
import { EventEmitter } from "events";
import { createWorker } from "tesseract.js";

export interface OCRResult {
  text: string;
  confidence: number;
  pages: number;
  processingTime: number;
  metadata: {
    filename: string;
    fileSize: number;
    mimeType: string;
    pageCount?: number;
    language?: string;
    documentType?: 'legal' | 'contract' | 'evidence' | 'general';
    legalEntities?: string[];
    confidentialityLevel?: 'public' | 'confidential' | 'privileged';
  };
  analysisResults?: {
    legalKeywords: string[];
    documentStructure: string[];
    confidenceBySection: number[];
    extractedDates: string[];
    extractedNumbers: string[];
  };
}

export interface ProcessingOptions {
  language?: string;
  psm?: number; // Page Segmentation Mode
  oem?: number; // OCR Engine Mode
  dpi?: number;
  outputQuality?: number;
  enhanceImage?: boolean;
  parallel?: boolean;
  enableLegalAnalysis?: boolean;
  enableEntityExtraction?: boolean;
  confidentialityDetection?: boolean;
}

export interface LegalDocumentAnalysis {
  documentType: 'contract' | 'legal_brief' | 'evidence' | 'correspondence' | 'other';
  legalKeywords: string[];
  parties: string[];
  dates: string[];
  amounts: string[];
  confidentialityMarkers: string[];
  structuralElements: {
    headers: string[];
    sections: string[];
    signatures: boolean;
    seals: boolean;
  };
}

export interface OCRWorkerConfig {
  id: string;
  worker: any;
  status: 'idle' | 'busy' | 'error';
  language: string;
  processedPages: number;
  errors: number;
}

export class EnhancedOCRProcessor extends EventEmitter {
  private workers: OCRWorkerConfig[] = [];
  private maxConcurrentWorkers: number;
  private tempDir: string;
  private initialized: boolean = false;
  private processingQueue: Array<{ resolve: Function; reject: Function; task: any }> = [];
  private legalKeywords: Set<string>;

  constructor(maxWorkers = 4, tempDir = "./temp") {
    super();
    this.maxConcurrentWorkers = maxWorkers;
    this.tempDir = tempDir;
    this.legalKeywords = new Set([
      'contract', 'agreement', 'defendant', 'plaintiff', 'evidence', 'testimony',
      'witness', 'attorney', 'counsel', 'court', 'jurisdiction', 'liability',
      'indemnification', 'confidential', 'privileged', 'attorney-client',
      'whereas', 'heretofore', 'hereinafter', 'ipso facto', 'prima facie',
      'subpoena', 'deposition', 'affidavit', 'statute', 'regulation',
      'settlement', 'damages', 'breach', 'breach of contract', 'negligence'
    ]);
    this.initializeWorkers();
  }

  private async initializeWorkers(): Promise<void> {
    try {
      // Ensure temp directory exists
      await fs.mkdir(this.tempDir, { recursive: true });

      // Initialize Tesseract workers
      for (let i = 0; i < this.maxConcurrentWorkers; i++) {
        const worker = await createWorker();
        await (worker as any).loadLanguage("eng");
        await (worker as any).initialize("eng");
        
        const workerConfig: OCRWorkerConfig = {
          id: `worker-${i}`,
          worker,
          status: 'idle',
          language: 'eng',
          processedPages: 0,
          errors: 0
        };
        
        this.workers.push(workerConfig);
      }

      this.initialized = true;
      this.emit("initialized", `${this.workers.length} OCR workers ready`);
      this.processQueue();
    } catch (error) {
      this.emit("error", `Failed to initialize OCR workers: ${error}`);
      throw error;
    }
  }

  /**
   * Process a file and extract text using OCR with legal document analysis
   */
  async processFile(
    filePath: string,
    options: ProcessingOptions = {}
  ): Promise<OCRResult> {
    const startTime = Date.now();
    const filename = path.basename(filePath);
    
    try {
      const stats = await fs.stat(filePath);
      this.emit("processing:start", { filename, size: stats.size });

      const mimeType = this.getMimeType(filename);
      let result: OCRResult;

      switch (mimeType) {
        case "application/pdf":
          result = await this.processPDFEnhanced(filePath, options);
          break;
        case "image/jpeg":
        case "image/png":
        case "image/tiff":
        case "image/bmp":
        case "image/webp":
          result = await this.processImageEnhanced(filePath, options);
          break;
        default:
          throw new Error(`Unsupported file type: ${mimeType}`);
      }

      result.processingTime = Date.now() - startTime;
      result.metadata = {
        ...result.metadata,
        filename,
        fileSize: stats.size,
        mimeType,
      };

      // Perform legal document analysis if enabled
      if (options.enableLegalAnalysis) {
        result.analysisResults = await this.performLegalAnalysis(result.text);
        result.metadata.documentType = await this.detectDocumentType(result.text);
        
        if (options.confidentialityDetection) {
          result.metadata.confidentialityLevel = await this.detectConfidentiality(result.text);
        }
      }

      this.emit("processing:complete", result);
      return result;
    } catch (error) {
      this.emit("processing:error", { filename, error: error.message });
      throw error;
    }
  }

  /**
   * Enhanced PDF processing with better error handling and legal optimization
   */
  private async processPDFEnhanced(
    filePath: string,
    options: ProcessingOptions
  ): Promise<OCRResult> {
    try {
      // For production, we'll simulate PDF to image conversion
      // In a real implementation, you'd use pdf2pic or pdf-poppler
      const pageImages = await this.simulatePDFConversion(filePath, options);
      
      const results: string[] = [];
      let totalConfidence = 0;
      const confidenceBySection: number[] = [];

      // Process pages with enhanced error handling
      if (options.parallel && pageImages.length > 1) {
        const promises = pageImages.map((imagePath, index) =>
          this.processImagePageEnhanced(imagePath, options, index)
        );
        const pageResults = await Promise.allSettled(promises);

        pageResults.forEach((result, index) => {
          if (result.status === 'fulfilled') {
            results.push(result.value.text);
            totalConfidence += result.value.confidence;
            confidenceBySection.push(result.value.confidence);
          } else {
            this.emit("page:error", { page: index + 1, error: result.reason });
            results.push("");
            confidenceBySection.push(0);
          }
        });
      } else {
        // Sequential processing for better memory management
        for (let i = 0; i < pageImages.length; i++) {
          try {
            const result = await this.processImagePageEnhanced(pageImages[i], options, i);
            results.push(result.text);
            totalConfidence += result.confidence;
            confidenceBySection.push(result.confidence);

            this.emit("page:processed", {
              page: i + 1,
              total: pageImages.length,
              confidence: result.confidence,
            });
          } catch (error) {
            this.emit("page:error", { page: i + 1, error: error.message });
            results.push("");
            confidenceBySection.push(0);
          }
        }
      }

      // Cleanup temporary images
      await this.cleanupImages(pageImages);

      return {
        text: results.join("\n\n"),
        confidence: pageImages.length > 0 ? totalConfidence / pageImages.length : 0,
        pages: pageImages.length,
        processingTime: 0, // Set by caller
        metadata: {
          filename: path.basename(filePath),
          fileSize: 0, // Set by caller
          mimeType: "application/pdf",
          pageCount: pageImages.length,
          language: options.language || "eng",
        },
        analysisResults: {
          legalKeywords: [],
          documentStructure: [],
          confidenceBySection,
          extractedDates: [],
          extractedNumbers: [],
        },
      };
    } catch (error) {
      throw new Error(`Enhanced PDF processing failed: ${error}`);
    }
  }

  /**
   * Enhanced image processing with legal document optimization
   */
  private async processImageEnhanced(
    filePath: string,
    options: ProcessingOptions
  ): Promise<OCRResult> {
    let processedImagePath = filePath;

    try {
      // Enhance image for better OCR results
      if (options.enhanceImage) {
        processedImagePath = await this.enhanceImageForLegal(filePath, options);
      }

      const result = await this.processImagePageEnhanced(
        processedImagePath,
        options,
        0
      );

      // Cleanup enhanced image if created
      if (processedImagePath !== filePath) {
        await fs.unlink(processedImagePath).catch(() => {}); // Ignore cleanup errors
      }

      return {
        text: result.text,
        confidence: result.confidence,
        pages: 1,
        processingTime: 0, // Set by caller
        metadata: {
          filename: path.basename(filePath),
          fileSize: 0, // Set by caller
          mimeType: this.getMimeType(filePath),
          language: options.language || "eng",
        },
        analysisResults: {
          legalKeywords: [],
          documentStructure: [],
          confidenceBySection: [result.confidence],
          extractedDates: [],
          extractedNumbers: [],
        },
      };
    } catch (error) {
      throw new Error(`Enhanced image processing failed: ${error}`);
    }
  }

  /**
   * Process a single image page with enhanced OCR settings
   */
  private async processImagePageEnhanced(
    imagePath: string,
    options: ProcessingOptions,
    pageIndex: number
  ): Promise<{ text: string; confidence: number }> {
    return new Promise((resolve, reject) => {
      if (!this.initialized) {
        this.processingQueue.push({ resolve, reject, task: { imagePath, options, pageIndex } });
        return;
      }

      this.processImagePageNow(imagePath, options, pageIndex)
        .then(resolve)
        .catch(reject);
    });
  }

  private async processImagePageNow(
    imagePath: string,
    options: ProcessingOptions,
    pageIndex: number
  ): Promise<{ text: string; confidence: number }> {
    const worker = this.getAvailableWorker();
    
    if (!worker) {
      throw new Error("No OCR workers available");
    }

    worker.status = 'busy';

    try {
      // Configure OCR settings optimized for legal documents
      await worker.worker.setParameters({
        tessedit_pageseg_mode: options.psm || 6, // Uniform block of text
        tessedit_ocr_engine_mode: options.oem || 3, // Default + LSTM
        preserve_interword_spaces: '1',
        tessedit_char_whitelist: 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,;:!?\'"-()[]{}$%&/*+=<>|\\@#^_`~', // Legal document characters
      });

      if (options.language && options.language !== worker.language) {
        await worker.worker.loadLanguage(options.language);
        await worker.worker.initialize(options.language);
        worker.language = options.language;
      }

      const {
        data: { text, confidence },
      } = await worker.worker.recognize(imagePath);

      worker.processedPages++;
      worker.status = 'idle';

      return {
        text: this.cleanLegalText(text),
        confidence: confidence || 0,
      };
    } catch (error) {
      worker.errors++;
      worker.status = 'error';
      
      // Reset worker if too many errors
      if (worker.errors > 5) {
        try {
          await worker.worker.terminate();
          const newWorker = await createWorker();
          await newWorker.loadLanguage(worker.language);
          await newWorker.initialize(worker.language);
          worker.worker = newWorker;
          worker.status = 'idle';
          worker.errors = 0;
        } catch (resetError) {
          this.emit("worker:reset:failed", { workerId: worker.id, error: resetError });
        }
      } else {
        worker.status = 'idle';
      }
      
      throw new Error(`OCR failed for page ${pageIndex}: ${error}`);
    }
  }

  /**
   * Enhanced image enhancement specifically for legal documents
   */
  private async enhanceImageForLegal(
    imagePath: string,
    options: ProcessingOptions
  ): Promise<string> {
    const enhancedPath = path.join(
      this.tempDir,
      `enhanced_legal_${Date.now()}_${path.basename(imagePath)}`
    );

    try {
      await sharp(imagePath)
        .greyscale()
        .normalize()
        .sharpen({ sigma: 1.2, flat: 1, jagged: 2 }) // Optimized for text
        .threshold(128) // High contrast for text
        .jpeg({ quality: options.outputQuality || 98, mozjpeg: true })
        .toFile(enhancedPath);

      return enhancedPath;
    } catch (error) {
      throw new Error(`Legal image enhancement failed: ${error}`);
    }
  }

  /**
   * Clean text specifically for legal documents
   */
  private cleanLegalText(text: string): string {
    return text
      .replace(/\s+/g, " ") // Normalize whitespace
      .replace(/\n\s*\n/g, "\n") // Remove empty lines
      .replace(/[^\w\s.,;:!?'"()-[\]{}$%&/*+=<>|\\@#^_`~]/g, '') // Remove invalid characters
      .replace(/(\w)\s+(\w)/g, '$1 $2') // Normalize word spacing
      .trim();
  }

  /**
   * Perform legal document analysis on extracted text
   */
  private async performLegalAnalysis(text: string): Promise<{
    legalKeywords: string[];
    documentStructure: string[];
    confidenceBySection: number[];
    extractedDates: string[];
    extractedNumbers: string[];
  }> {
    const legalKeywords = Array.from(this.legalKeywords).filter(keyword =>
      text.toLowerCase().includes(keyword.toLowerCase())
    );

    const documentStructure = this.extractDocumentStructure(text);
    const extractedDates = this.extractDates(text);
    const extractedNumbers = this.extractNumbers(text);

    return {
      legalKeywords,
      documentStructure,
      confidenceBySection: [], // Filled by caller
      extractedDates,
      extractedNumbers,
    };
  }

  /**
   * Detect document type based on content
   */
  private async detectDocumentType(text: string): Promise<'legal' | 'contract' | 'evidence' | 'general'> {
    const lowerText = text.toLowerCase();
    
    if (lowerText.includes('contract') || lowerText.includes('agreement') || lowerText.includes('whereas')) {
      return 'contract';
    } else if (lowerText.includes('evidence') || lowerText.includes('exhibit') || lowerText.includes('testimony')) {
      return 'evidence';
    } else if (this.legalKeywords.size > 0 && Array.from(this.legalKeywords).some(keyword => 
      lowerText.includes(keyword.toLowerCase())
    )) {
      return 'legal';
    }
    
    return 'general';
  }

  /**
   * Detect confidentiality level
   */
  private async detectConfidentiality(text: string): Promise<'public' | 'confidential' | 'privileged'> {
    const lowerText = text.toLowerCase();
    
    if (lowerText.includes('attorney-client') || lowerText.includes('privileged')) {
      return 'privileged';
    } else if (lowerText.includes('confidential') || lowerText.includes('proprietary')) {
      return 'confidential';
    }
    
    return 'public';
  }

  /**
   * Extract document structure elements
   */
  private extractDocumentStructure(text: string): string[] {
    const structure: string[] = [];
    const lines = text.split('\n');
    
    for (const line of lines) {
      const trimmedLine = line.trim();
      
      // Detect headers (all caps, short lines)
      if (trimmedLine.length > 0 && trimmedLine.length < 100 && 
          trimmedLine === trimmedLine.toUpperCase() && 
          /^[A-Z\s\d.,;:-]+$/.test(trimmedLine)) {
        structure.push(`HEADER: ${trimmedLine}`);
      }
      
      // Detect numbered sections
      if (/^\d+\.\s/.test(trimmedLine)) {
        structure.push(`SECTION: ${trimmedLine.substring(0, 50)}...`);
      }
    }
    
    return structure;
  }

  /**
   * Extract dates from text
   */
  private extractDates(text: string): string[] {
    const datePatterns = [
      /\d{1,2}\/\d{1,2}\/\d{2,4}/g, // MM/DD/YYYY
      /\d{1,2}-\d{1,2}-\d{2,4}/g,   // MM-DD-YYYY
      /\w+\s+\d{1,2},\s+\d{4}/g,    // Month DD, YYYY
    ];
    
    const dates: string[] = [];
    for (const pattern of datePatterns) {
      const matches = text.match(pattern);
      if (matches) {
        dates.push(...matches);
      }
    }
    
    return [...new Set(dates)]; // Remove duplicates
  }

  /**
   * Extract numbers and amounts from text
   */
  private extractNumbers(text: string): string[] {
    const numberPatterns = [
      /\$[\d,]+\.?\d*/g,      // Currency amounts
      /\d+%/g,                // Percentages
      /\b\d{4,}\b/g,          // Large numbers (case numbers, etc.)
    ];
    
    const numbers: string[] = [];
    for (const pattern of numberPatterns) {
      const matches = text.match(pattern);
      if (matches) {
        numbers.push(...matches);
      }
    }
    
    return [...new Set(numbers)]; // Remove duplicates
  }

  /**
   * Simulate PDF conversion (placeholder for real implementation)
   */
  private async simulatePDFConversion(filePath: string, options: ProcessingOptions): Promise<string[]> {
    // In a real implementation, use pdf2pic or pdf-poppler
    // For now, return the original file path as if it were converted
    return [filePath];
  }

  /**
   * Get available OCR worker with load balancing
   */
  private getAvailableWorker(): OCRWorkerConfig | null {
    const idleWorkers = this.workers.filter(w => w.status === 'idle');
    
    if (idleWorkers.length === 0) {
      return null;
    }
    
    // Return worker with least processed pages for load balancing
    return idleWorkers.reduce((prev, current) => 
      prev.processedPages < current.processedPages ? prev : current
    );
  }

  /**
   * Process queued tasks
   */
  private async processQueue(): Promise<void> {
    while (this.processingQueue.length > 0) {
      const queueItem = this.processingQueue.shift();
      if (queueItem) {
        try {
          const result = await this.processImagePageNow(
            queueItem.task.imagePath,
            queueItem.task.options,
            queueItem.task.pageIndex
          );
          queueItem.resolve(result);
        } catch (error) {
          queueItem.reject(error);
        }
      }
    }
  }

  /**
   * Cleanup temporary image files
   */
  private async cleanupImages(imagePaths: string[]): Promise<void> {
    const cleanupPromises = imagePaths.map(async (imagePath) => {
      try {
        await fs.unlink(imagePath);
      } catch (error) {
        // Ignore cleanup errors
      }
    });

    await Promise.allSettled(cleanupPromises);
  }

  /**
   * Get MIME type from filename
   */
  private getMimeType(filename: string): string {
    const ext = path.extname(filename).toLowerCase();
    const mimeTypes: Record<string, string> = {
      ".pdf": "application/pdf",
      ".jpg": "image/jpeg",
      ".jpeg": "image/jpeg",
      ".png": "image/png",
      ".tiff": "image/tiff",
      ".tif": "image/tiff",
      ".bmp": "image/bmp",
      ".webp": "image/webp",
      ".gif": "image/gif",
    };

    return mimeTypes[ext] || "application/octet-stream";
  }

  /**
   * Batch process multiple files with progress tracking
   */
  async processMultipleFiles(
    filePaths: string[],
    options: ProcessingOptions = {}
  ): Promise<OCRResult[]> {
    const results: OCRResult[] = [];

    this.emit("batch:start", { fileCount: filePaths.length });

    for (let i = 0; i < filePaths.length; i++) {
      try {
        const result = await this.processFile(filePaths[i], options);
        results.push(result);

        this.emit("batch:progress", {
          completed: i + 1,
          total: filePaths.length,
          filename: path.basename(filePaths[i]),
          confidence: result.confidence,
        });
      } catch (error) {
        this.emit("batch:error", {
          file: filePaths[i],
          error: error.message,
        });

        // Add failed result to maintain order
        results.push({
          text: "",
          confidence: 0,
          pages: 0,
          processingTime: 0,
          metadata: {
            filename: path.basename(filePaths[i]),
            fileSize: 0,
            mimeType: "application/octet-stream",
          },
        });
      }
    }

    this.emit("batch:complete", { results: results.length, totalSuccessful: results.filter(r => r.confidence > 0).length });
    return results;
  }

  /**
   * Gracefully shutdown all workers
   */
  async destroy(): Promise<void> {
    try {
      const terminationPromises = this.workers.map(async (workerConfig) => {
        try {
          await workerConfig.worker.terminate();
        } catch (error) {
          this.emit("worker:termination:error", { workerId: workerConfig.id, error });
        }
      });
      
      await Promise.allSettled(terminationPromises);
      this.workers = [];
      this.initialized = false;

      this.emit("destroyed", "All OCR workers terminated");
    } catch (error) {
      this.emit("error", `Worker termination failed: ${error}`);
    }
  }

  /**
   * Get comprehensive system status
   */
  getStatus(): {
    workersActive: number;
    workersIdle: number;
    workersBusy: number;
    workersError: number;
    tempDir: string;
    isReady: boolean;
    queueLength: number;
    totalPagesProcessed: number;
    totalErrors: number;
  } {
    const idleWorkers = this.workers.filter(w => w.status === 'idle').length;
    const busyWorkers = this.workers.filter(w => w.status === 'busy').length;
    const errorWorkers = this.workers.filter(w => w.status === 'error').length;
    const totalPagesProcessed = this.workers.reduce((sum, w) => sum + w.processedPages, 0);
    const totalErrors = this.workers.reduce((sum, w) => sum + w.errors, 0);

    return {
      workersActive: this.workers.length,
      workersIdle: idleWorkers,
      workersBusy: busyWorkers,
      workersError: errorWorkers,
      tempDir: this.tempDir,
      isReady: this.initialized && this.workers.length > 0,
      queueLength: this.processingQueue.length,
      totalPagesProcessed,
      totalErrors,
    };
  }

  /**
   * Add legal keywords for enhanced detection
   */
  addLegalKeywords(keywords: string[]): void {
    keywords.forEach(keyword => this.legalKeywords.add(keyword.toLowerCase()));
    this.emit("keywords:updated", { totalKeywords: this.legalKeywords.size });
  }

  /**
   * Get legal keyword statistics
   */
  getLegalKeywordStats(): { total: number; keywords: string[] } {
    return {
      total: this.legalKeywords.size,
      keywords: Array.from(this.legalKeywords).sort(),
    };
  }
}

// Export singleton instance for global use
export const ocrProcessor = new EnhancedOCRProcessor();

// Export factory function for custom instances
export function createOCRProcessor(
  maxWorkers?: number,
  tempDir?: string
): EnhancedOCRProcessor {
  return new EnhancedOCRProcessor(maxWorkers, tempDir);
}

// Export utility functions
export const ocrUtils = {
  /**
   * Validate file for OCR processing
   */
  validateFile: async (filePath: string): Promise<{ valid: boolean; reason?: string }> => {
    try {
      const stats = await fs.stat(filePath);
      
      if (stats.size === 0) {
        return { valid: false, reason: "File is empty" };
      }
      
      if (stats.size > 100 * 1024 * 1024) { // 100MB limit
        return { valid: false, reason: "File too large (>100MB)" };
      }
      
      const ext = path.extname(filePath).toLowerCase();
      const supportedExtensions = ['.pdf', '.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.webp'];
      
      if (!supportedExtensions.includes(ext)) {
        return { valid: false, reason: `Unsupported file type: ${ext}` };
      }
      
      return { valid: true };
    } catch (error) {
      return { valid: false, reason: `File access error: ${error}` };
    }
  },

  /**
   * Estimate processing time based on file size and type
   */
  estimateProcessingTime: (filePath: string, fileSize: number): number => {
    const ext = path.extname(filePath).toLowerCase();
    
    // Base time in seconds per MB
    const timePerMB = ext === '.pdf' ? 10 : 5;
    const sizeInMB = fileSize / (1024 * 1024);
    
    return Math.max(sizeInMB * timePerMB, 2); // Minimum 2 seconds
  },

  /**
   * Get optimal OCR settings for document type
   */
  getOptimalSettings: (documentType: 'contract' | 'handwritten' | 'typed' | 'mixed'): ProcessingOptions => {
    const baseSettings = {
      language: 'eng',
      dpi: 300,
      outputQuality: 95,
      enhanceImage: true,
      enableLegalAnalysis: true,
      confidentialityDetection: true,
    };

    switch (documentType) {
      case 'contract':
        return {
          ...baseSettings,
          psm: 6, // Uniform block of text
          oem: 3, // Default + LSTM
        };
      case 'handwritten':
        return {
          ...baseSettings,
          psm: 8, // Single word
          oem: 2, // LSTM only
          dpi: 400,
        };
      case 'typed':
        return {
          ...baseSettings,
          psm: 4, // Single column of text
          oem: 3,
        };
      case 'mixed':
        return {
          ...baseSettings,
          psm: 3, // Fully automatic page segmentation
          oem: 3,
          parallel: true,
        };
      default:
        return baseSettings;
    }
  }
};

export default EnhancedOCRProcessor;