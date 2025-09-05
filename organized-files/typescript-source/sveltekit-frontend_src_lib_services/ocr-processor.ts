// @ts-nocheck
import PDFDocument from "pdfkit";
import fs from "fs";
import path from "path";
import { Readable } from "stream";
import Tesseract from "tesseract.js";
import pdf2pic from "pdf2pic";
import sharp from "sharp";

export interface OCRResult {
  text: string;
  confidence: number;
  pages: PageResult[];
  metadata: DocumentMetadata;
  processing_time: number;
}

export interface PageResult {
  page_number: number;
  text: string;
  confidence: number;
  blocks: TextBlock[];
  images: ImageRegion[];
}

export interface TextBlock {
  text: string;
  bbox: BoundingBox;
  confidence: number;
  font_size?: number;
  font_family?: string;
}

export interface ImageRegion {
  bbox: BoundingBox;
  type: "figure" | "table" | "diagram" | "signature";
}

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface DocumentMetadata {
  title?: string;
  author?: string;
  subject?: string;
  creator?: string;
  creation_date?: Date;
  modification_date?: Date;
  page_count: number;
  file_size: number;
  content_type: string;
}

export class EnhancedOCRProcessor {
  private readonly supportedFormats = [
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".tiff",
    ".bmp",
  ];
  private readonly ocrOptions = {
    logger: (m: any) => console.log(m),
    errorHandler: (err: any) => console.error(err),
  };

  constructor() {
    this.initializeTesseract();
  }

  private async initializeTesseract(): Promise<void> {
    try {
      await Tesseract.recognize(
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
        "eng"
      );
      console.log("✅ Tesseract initialized successfully");
    } catch (error) {
      console.error("❌ Failed to initialize Tesseract:", error);
    }
  }

  async processDocument(filePath: string): Promise<OCRResult> {
    const startTime = Date.now();
    const fileExtension = path.extname(filePath).toLowerCase();

    if (!this.supportedFormats.includes(fileExtension)) {
      throw new Error(`Unsupported file format: ${fileExtension}`);
    }

    try {
      let result: OCRResult;

      if (fileExtension === ".pdf") {
        result = await this.processPDF(filePath);
      } else {
        result = await this.processImage(filePath);
      }

      result.processing_time = Date.now() - startTime;
      return result;
    } catch (error) {
      console.error("❌ Document processing failed:", error);
      throw error;
    }
  }

  private async processPDF(filePath: string): Promise<OCRResult> {
    const fileBuffer = fs.readFileSync(filePath);
    const fileStats = fs.statSync(filePath);

    // Extract PDF metadata using PDFKit
    const metadata = await this.extractPDFMetadata(fileBuffer);

    // Convert PDF pages to images for OCR
    const convert = pdf2pic.fromBuffer(fileBuffer, {
      density: 300,
      saveFilename: "page",
      savePath: "/tmp",
      format: "png",
      width: 2480,
      height: 3508,
    });

    const pages: PageResult[] = [];
    let totalConfidence = 0;
    let totalText = "";

    try {
      // Process each page
      for (let pageNum = 1; pageNum <= metadata.page_count; pageNum++) {
        console.log(`📄 Processing PDF page ${pageNum}/${metadata.page_count}`);

        const imageResult = await convert(pageNum);
        const imagePath = imageResult.path;

        // Enhance image for better OCR
        const enhancedImagePath = await this.enhanceImageForOCR(imagePath);

        // Perform OCR on the page
        const ocrResult = await Tesseract.recognize(
          enhancedImagePath,
          "eng",
          this.ocrOptions
        );

        const pageResult: PageResult = {
          page_number: pageNum,
          text: ocrResult.data.text,
          confidence: ocrResult.data.confidence,
          blocks: this.extractTextBlocks(ocrResult.data),
          images: this.detectImageRegions(ocrResult.data),
        };

        pages.push(pageResult);
        totalText += pageResult.text + "\\n";
        totalConfidence += pageResult.confidence;

        // Clean up temporary files
        this.cleanupTempFiles([imagePath, enhancedImagePath]);
      }

      const avgConfidence =
        pages.length > 0 ? totalConfidence / pages.length : 0;

      return {
        text: totalText.trim(),
        confidence: avgConfidence,
        pages,
        metadata: {
          ...metadata,
          file_size: fileStats.size,
          content_type: "application/pdf",
        },
        processing_time: 0, // Will be set by caller
      };
    } catch (error) {
      console.error("❌ PDF processing error:", error);
      throw error;
    }
  }

  private async processImage(filePath: string): Promise<OCRResult> {
    const fileStats = fs.statSync(filePath);

    // Enhance image for better OCR
    const enhancedImagePath = await this.enhanceImageForOCR(filePath);

    // Perform OCR
    const ocrResult = await Tesseract.recognize(
      enhancedImagePath,
      "eng",
      this.ocrOptions
    );

    const pageResult: PageResult = {
      page_number: 1,
      text: ocrResult.data.text,
      confidence: ocrResult.data.confidence,
      blocks: this.extractTextBlocks(ocrResult.data),
      images: this.detectImageRegions(ocrResult.data),
    };

    // Clean up enhanced image if it's different from original
    if (enhancedImagePath !== filePath) {
      this.cleanupTempFiles([enhancedImagePath]);
    }

    return {
      text: ocrResult.data.text,
      confidence: ocrResult.data.confidence,
      pages: [pageResult],
      metadata: {
        page_count: 1,
        file_size: fileStats.size,
        content_type: `image/${path.extname(filePath).slice(1)}`,
      },
      processing_time: 0,
    };
  }

  private async extractPDFMetadata(buffer: Buffer): Promise<DocumentMetadata> {
    try {
      // Use a simple PDF parser for metadata
      const pdfText = buffer.toString("binary");

      // Extract basic metadata patterns
      const titleMatch = pdfText.match(/\/Title\s*\((.*?)\)/);
      const authorMatch = pdfText.match(/\/Author\s*\((.*?)\)/);
      const subjectMatch = pdfText.match(/\/Subject\s*\((.*?)\)/);
      const creatorMatch = pdfText.match(/\/Creator\s*\((.*?)\)/);

      // Count pages by looking for page objects
      const pageMatches = pdfText.match(/\/Type\s*\/Page[^s]/g);
      const pageCount = pageMatches ? pageMatches.length : 1;

      return {
        title: titleMatch ? titleMatch[1] : undefined,
        author: authorMatch ? authorMatch[1] : undefined,
        subject: subjectMatch ? subjectMatch[1] : undefined,
        creator: creatorMatch ? creatorMatch[1] : undefined,
        page_count: pageCount,
        file_size: buffer.length,
        content_type: "application/pdf",
      };
    } catch (error) {
      console.warn("⚠️ Could not extract PDF metadata:", error);
      return {
        page_count: 1,
        file_size: buffer.length,
        content_type: "application/pdf",
      };
    }
  }

  private async enhanceImageForOCR(imagePath: string): Promise<string> {
    try {
      const enhancedPath = imagePath.replace(/\\.(\\w+)$/, "_enhanced.$1");

      await sharp(imagePath)
        .greyscale()
        .normalize()
        .sharpen()
        .threshold(128)
        .toFile(enhancedPath);

      return enhancedPath;
    } catch (error) {
      console.warn("⚠️ Image enhancement failed, using original:", error);
      return imagePath;
    }
  }

  private extractTextBlocks(data: Tesseract.Page): TextBlock[] {
    const blocks: TextBlock[] = [];

    if (data.blocks) {
      for (const block of data.blocks) {
        blocks.push({
          text: block.text,
          bbox: {
            x: block.bbox.x0,
            y: block.bbox.y0,
            width: block.bbox.x1 - block.bbox.x0,
            height: block.bbox.y1 - block.bbox.y0,
          },
          confidence: block.confidence,
        });
      }
    }

    return blocks;
  }

  private detectImageRegions(data: Tesseract.Page): ImageRegion[] {
    // Simple heuristic for detecting image regions
    // In a real implementation, you might use additional ML models
    const regions: ImageRegion[] = [];

    if (data.blocks) {
      for (const block of data.blocks) {
        // Detect potential image regions based on low text confidence and size
        if (
          block.confidence < 30 &&
          block.bbox.x1 - block.bbox.x0 > 100 &&
          block.bbox.y1 - block.bbox.y0 > 100
        ) {
          regions.push({
            bbox: {
              x: block.bbox.x0,
              y: block.bbox.y0,
              width: block.bbox.x1 - block.bbox.x0,
              height: block.bbox.y1 - block.bbox.y0,
            },
            type: "figure",
          });
        }
      }
    }

    return regions;
  }

  private cleanupTempFiles(filePaths: string[]): void {
    filePaths.forEach((filePath) => {
      try {
        if (fs.existsSync(filePath)) {
          fs.unlinkSync(filePath);
        }
      } catch (error) {
        console.warn(`⚠️ Failed to cleanup temp file ${filePath}:`, error);
      }
    });
  }

  async batchProcessDocuments(filePaths: string[]): Promise<OCRResult[]> {
    const results: OCRResult[] = [];

    console.log(
      `📚 Starting batch processing of ${filePaths.length} documents`
    );

    for (let i = 0; i < filePaths.length; i++) {
      const filePath = filePaths[i];
      console.log(
        `📄 Processing document ${i + 1}/${filePaths.length}: ${path.basename(filePath)}`
      );

      try {
        const result = await this.processDocument(filePath);
        results.push(result);
        console.log(
          `✅ Completed processing ${path.basename(filePath)} in ${result.processing_time}ms`
        );
      } catch (error) {
        console.error(
          `❌ Failed to process ${path.basename(filePath)}:`,
          error
        );
        // Continue with other documents
      }
    }

    console.log(
      `🎉 Batch processing completed. ${results.length}/${filePaths.length} documents processed successfully`
    );
    return results;
  }

  getProcessingStats(): {
    supportedFormats: string[];
    tesseractReady: boolean;
  } {
    return {
      supportedFormats: this.supportedFormats,
      tesseractReady: true,
    };
  }
}

// Export singleton instance
export const ocrProcessor = new EnhancedOCRProcessor();
