/**
 * Production OCR Processor Service
 * Handles PDF, image, and document processing with enhanced error handling
 */

import { createWorker } from "tesseract.js";
import PDFParser from "pdf2pic";
import sharp from "sharp";
import fs from "fs/promises";
import path from "path";
import { pipeline } from "stream/promises";
import { EventEmitter } from "events";

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
}

export class EnhancedOCRProcessor extends EventEmitter {
  private workers: any[] = [];
  private maxConcurrentWorkers: number;
  private tempDir: string;

  constructor(maxWorkers = 4, tempDir = "./temp") {
    super();
    this.maxConcurrentWorkers = maxWorkers;
    this.tempDir = tempDir;
    this.initializeWorkers();
  }

  private async initializeWorkers(): Promise<void> {
    try {
      // Ensure temp directory exists
      await fs.mkdir(this.tempDir, { recursive: true });

      // Initialize Tesseract workers
      for (let i = 0; i < this.maxConcurrentWorkers; i++) {
        const worker = await createWorker();
        await worker.loadLanguage("eng");
        await worker.initialize("eng");
        this.workers.push(worker);
      }

      this.emit("initialized", `${this.workers.length} OCR workers ready`);
    } catch (error) {
      this.emit("error", `Failed to initialize OCR workers: ${error.message}`);
      throw error;
    }
  }

  /**
   * Process a file and extract text using OCR
   */
  async processFile(
    filePath: string,
    options: ProcessingOptions = {}
  ): Promise<OCRResult> {
    const startTime = Date.now();
    const filename = path.basename(filePath);
    const stats = await fs.stat(filePath);

    this.emit("processing:start", { filename, size: stats.size });

    try {
      const mimeType = this.getMimeType(filename);
      let result: OCRResult;

      switch (mimeType) {
        case "application/pdf":
          result = await this.processPDF(filePath, options);
          break;
        case "image/jpeg":
        case "image/png":
        case "image/tiff":
        case "image/bmp":
          result = await this.processImage(filePath, options);
          break;
        default:
          throw new Error(`Unsupported file type: ${mimeType}`);
      }

      result.processingTime = Date.now() - startTime;
      result.metadata = {
        filename,
        fileSize: stats.size,
        mimeType,
        ...result.metadata,
      };

      this.emit("processing:complete", result);
      return result;
    } catch (error) {
      this.emit("processing:error", { filename, error: error.message });
      throw error;
    }
  }

  /**
   * Process PDF files by converting to images first
   */
  private async processPDF(
    filePath: string,
    options: ProcessingOptions
  ): Promise<OCRResult> {
    const pdfParser = new PDFParser({
      density: options.dpi || 300,
      saveFilename: "page",
      savePath: this.tempDir,
      format: "png",
      width: 2480,
      height: 3508,
    });

    try {
      // Convert PDF pages to images
      const pages = await pdfParser.convertBulk(filePath, -1);
      this.emit("pdf:converted", `${pages.length} pages converted`);

      const results: string[] = [];
      let totalConfidence = 0;

      // Process pages in parallel or sequential based on options
      if (options.parallel && pages.length > 1) {
        const promises = pages.map((page, index) =>
          this.processImagePage(page.path, options, index)
        );
        const pageResults = await Promise.all(promises);

        pageResults.forEach((result) => {
          results.push(result.text);
          totalConfidence += result.confidence;
        });
      } else {
        // Sequential processing for better memory management
        for (let i = 0; i < pages.length; i++) {
          const result = await this.processImagePage(pages[i].path, options, i);
          results.push(result.text);
          totalConfidence += result.confidence;

          this.emit("page:processed", {
            page: i + 1,
            total: pages.length,
            confidence: result.confidence,
          });
        }
      }

      // Cleanup temporary images
      await this.cleanupImages(pages.map((p) => p.path));

      return {
        text: results.join("\n\n"),
        confidence: totalConfidence / pages.length,
        pages: pages.length,
        processingTime: 0, // Set by caller
        metadata: {
          pageCount: pages.length,
          language: options.language || "eng",
        },
      };
    } catch (error) {
      throw new Error(`PDF processing failed: ${error.message}`);
    }
  }

  /**
   * Process image files directly
   */
  private async processImage(
    filePath: string,
    options: ProcessingOptions
  ): Promise<OCRResult> {
    let processedImagePath = filePath;

    try {
      // Enhance image if requested
      if (options.enhanceImage) {
        processedImagePath = await this.enhanceImage(filePath, options);
      }

      const result = await this.processImagePage(
        processedImagePath,
        options,
        0
      );

      // Cleanup enhanced image if created
      if (processedImagePath !== filePath) {
        await fs.unlink(processedImagePath);
      }

      return {
        text: result.text,
        confidence: result.confidence,
        pages: 1,
        processingTime: 0, // Set by caller
        metadata: {
          language: options.language || "eng",
        },
      };
    } catch (error) {
      throw new Error(`Image processing failed: ${error.message}`);
    }
  }

  /**
   * Process a single image page with OCR
   */
  private async processImagePage(
    imagePath: string,
    options: ProcessingOptions,
    pageIndex: number
  ): Promise<{ text: string; confidence: number }> {
    const worker = this.getAvailableWorker();

    try {
      // Configure OCR settings
      await worker.setParameters({
        tessedit_pageseg_mode: options.psm || 3,
        tessedit_ocr_engine_mode: options.oem || 3,
      });

      if (options.language && options.language !== "eng") {
        await worker.loadLanguage(options.language);
        await worker.initialize(options.language);
      }

      const {
        data: { text, confidence },
      } = await worker.recognize(imagePath);

      return {
        text: this.cleanText(text),
        confidence: confidence || 0,
      };
    } catch (error) {
      throw new Error(`OCR failed for page ${pageIndex}: ${error.message}`);
    }
  }

  /**
   * Enhance image quality for better OCR results
   */
  private async enhanceImage(
    imagePath: string,
    options: ProcessingOptions
  ): Promise<string> {
    const enhancedPath = path.join(
      this.tempDir,
      `enhanced_${Date.now()}_${path.basename(imagePath)}`
    );

    try {
      await sharp(imagePath)
        .greyscale()
        .normalize()
        .sharpen({ sigma: 1.0 })
        .jpeg({ quality: options.outputQuality || 95 })
        .toFile(enhancedPath);

      return enhancedPath;
    } catch (error) {
      throw new Error(`Image enhancement failed: ${error.message}`);
    }
  }

  /**
   * Clean up extracted text
   */
  private cleanText(text: string): string {
    return text
      .replace(/\s+/g, " ") // Normalize whitespace
      .replace(/\n\s*\n/g, "\n") // Remove empty lines
      .trim();
  }

  /**
   * Get available OCR worker
   */
  private getAvailableWorker(): any {
    if (this.workers.length === 0) {
      throw new Error("No OCR workers available");
    }
    return this.workers[Math.floor(Math.random() * this.workers.length)];
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
    };

    return mimeTypes[ext] || "application/octet-stream";
  }

  /**
   * Batch process multiple files
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
            mimeType: "error",
            error: error.message,
          },
        });
      }
    }

    this.emit("batch:complete", { results: results.length });
    return results;
  }

  /**
   * Gracefully shutdown all workers
   */
  async destroy(): Promise<void> {
    try {
      const terminationPromises = this.workers.map((worker) =>
        worker.terminate()
      );
      await Promise.all(terminationPromises);
      this.workers = [];

      this.emit("destroyed", "All OCR workers terminated");
    } catch (error) {
      this.emit("error", `Worker termination failed: ${error.message}`);
    }
  }

  /**
   * Get system status
   */
  getStatus(): {
    workersActive: number;
    tempDir: string;
    isReady: boolean;
  } {
    return {
      workersActive: this.workers.length,
      tempDir: this.tempDir,
      isReady: this.workers.length > 0,
    };
  }
}

// Export singleton instance
export const ocrProcessor = new EnhancedOCRProcessor();

// Export factory function for custom instances
export function createOCRProcessor(
  maxWorkers?: number,
  tempDir?: string
): EnhancedOCRProcessor {
  return new EnhancedOCRProcessor(maxWorkers, tempDir);
}
