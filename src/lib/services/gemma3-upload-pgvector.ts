/**
 * LLVM-Quality WebAssembly Gemma3 with File Upload & pgvector Integration
 * Complete file processing pipeline with PostgreSQL vector storage
 */

import { gemma3LokiIntegration } from './gemma3-loki-integration';
import { vertexBufferImageAnalyzer } from './vertex-buffer-image-analyzer';

interface UploadProcessingOptions {
  extractText?: boolean;
  generateEmbeddings?: boolean;
  performLegalAnalysis?: boolean;
  storeInDatabase?: boolean;
  analysisType?: 'comprehensive' | 'quick' | 'risk-focused' | 'legal-precedent';
  userId: string;
  caseId?: string;
  enableImageAnalysis?: boolean;
  useWebAssembly?: boolean;
}

interface ProcessedFile {
  id: string;
  originalName: string;
  mimeType: string;
  size: number;
  extractedText?: string;
  analysis?: any;
  embeddings?: {
    vector: number[];
    dimensions: number;
    model: string;
  };
  imageAnalysis?: any;
  databaseId?: string;
  processingTime: number;
  method: 'webassembly' | 'ollama' | 'hybrid';
}

interface DatabaseStorageResult {
  documentId: string;
  vectorId: string;
  evidenceId?: string;
  success: boolean;
  error?: string;
}

export class Gemma3UploadPgVectorService {
  private initialized = false;
  private metrics = {
    filesProcessed: 0,
    totalProcessingTime: 0,
    webassemblyProcessing: 0,
    databaseStores: 0,
    vectorEmbeddings: 0,
    errors: 0
  };

  constructor() {
    this.initialize();
  }

  private async initialize(): Promise<void> {
    if (this.initialized) return;

    try {
      // Initialize dependent services
      await Promise.all([
        gemma3LokiIntegration.initialize?.() || Promise.resolve(),
        vertexBufferImageAnalyzer.initialize()
      ]);

      this.initialized = true;
      console.log('🚀 Gemma3 Upload & pgvector Service initialized');

    } catch (error) {
      console.error('❌ Failed to initialize Gemma3 Upload Service:', error);
      throw error;
    }
  }

  /**
   * Process uploaded file with complete AI pipeline
   */
  async processUploadedFile(
    file: File,
    options: UploadProcessingOptions
  ): Promise<ProcessedFile> {
    await this.initialize();

    const startTime = performance.now();
    const fileId = crypto.randomUUID();

    console.log(`📁 Processing file: ${file.name} (${file.size} bytes)`);

    try {
      const result: ProcessedFile = {
        id: fileId,
        originalName: file.name,
        mimeType: file.type,
        size: file.size,
        processingTime: 0,
        method: 'webassembly'
      };

      // Step 1: Extract text content
      if (options.extractText || options.performLegalAnalysis) {
        console.log('📄 Extracting text content...');
        result.extractedText = await this.extractTextFromFile(file);
      }

      // Step 2: Analyze images if it's an image file
      if (options.enableImageAnalysis && this.isImageFile(file)) {
        console.log('🖼️ Analyzing image with vertex buffer extraction...');
        result.imageAnalysis = await this.analyzeImageFile(file);
      }

      // Step 3: Perform legal AI analysis
      if (options.performLegalAnalysis && result.extractedText) {
        console.log('⚖️ Performing legal AI analysis...');
        
        const analysisResult = await gemma3LokiIntegration.analyzeLegalDocument({
          content: result.extractedText,
          title: file.name,
          caseId: options.caseId,
          analysisType: options.analysisType || 'comprehensive',
          useCache: true,
          storeResults: false, // We'll handle storage ourselves
          userId: options.userId
        });

        result.analysis = analysisResult.analysis;
        result.embeddings = {
          vector: Array.from(analysisResult.embeddings.vector),
          dimensions: analysisResult.embeddings.dimensions,
          model: analysisResult.embeddings.model
        };
        result.method = analysisResult.analysis.method;
      } else if (options.generateEmbeddings && result.extractedText) {
        // Generate embeddings only
        console.log('🧠 Generating embeddings...');
        const embeddings = await this.generateEmbeddings(result.extractedText);
        result.embeddings = embeddings;
      }

      // Step 4: Store in PostgreSQL database with pgvector
      if (options.storeInDatabase) {
        console.log('💾 Storing in PostgreSQL with pgvector...');
        const storageResult = await this.storeInDatabase(result, options);
        result.databaseId = storageResult.documentId;
      }

      const totalTime = performance.now() - startTime;
      result.processingTime = totalTime;

      // Update metrics
      this.updateMetrics(result, totalTime);

      console.log(`✅ File processing completed in ${totalTime.toFixed(2)}ms using ${result.method}`);

      return result;

    } catch (error) {
      this.metrics.errors++;
      console.error('❌ File processing failed:', error);
      throw error;
    }
  }

  /**
   * Batch process multiple files
   */
  async processBatchUpload(
    files: FileList | File[],
    options: UploadProcessingOptions,
    maxConcurrency: number = 3
  ): Promise<ProcessedFile[]> {
    await this.initialize();

    const fileArray = Array.from(files);
    console.log(`📦 Processing batch of ${fileArray.length} files`);

    const results: ProcessedFile[] = [];
    const errors: any[] = [];

    // Process files in chunks to avoid overwhelming the system
    for (let i = 0; i < fileArray.length; i += maxConcurrency) {
      const chunk = fileArray.slice(i, i + maxConcurrency);
      
      const chunkPromises = chunk.map(async (file) => {
        try {
          return await this.processUploadedFile(file, options);
        } catch (error) {
          console.error(`Failed to process ${file.name}:`, error);
          errors.push({ fileName: file.name, error });
          return null;
        }
      });

      const chunkResults = await Promise.all(chunkPromises);
      results.push(...chunkResults.filter((r): r is ProcessedFile => r !== null));
    }

    console.log(`✅ Batch processing completed: ${results.length} successful, ${errors.length} failed`);

    return results;
  }

  /**
   * Search processed documents using pgvector similarity
   */
  async searchDocuments(
    query: string,
    options: {
      threshold?: number;
      maxResults?: number;
      userId?: string;
      caseId?: string;
      includeAnalysis?: boolean;
    } = {}
  ): Promise<any[]> {
    await this.initialize();

    const {
      threshold = 0.7,
      maxResults = 20,
      userId,
      caseId,
      includeAnalysis = true
    } = options;

    try {
      // Generate query embedding
      const queryEmbeddings = await this.generateEmbeddings(query);

      // Search in PostgreSQL with pgvector
      const response = await fetch('/api/database/vector-search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          embedding: queryEmbeddings.vector,
          threshold,
          maxResults,
          userId,
          caseId,
          includeAnalysis
        })
      });

      if (!response.ok) {
        throw new Error(`Vector search failed: ${response.status}`);
      }

      const results = await response.json();
      console.log(`🔍 Vector search found ${results.length} similar documents`);

      return results;

    } catch (error) {
      console.error('❌ Document search failed:', error);
      throw error;
    }
  }

  /**
   * Get file processing statistics
   */
  getProcessingStats() {
    const avgProcessingTime = this.metrics.filesProcessed > 0 
      ? this.metrics.totalProcessingTime / this.metrics.filesProcessed 
      : 0;

    return {
      ...this.metrics,
      averageProcessingTime: avgProcessingTime,
      webassemblySuccessRate: this.metrics.filesProcessed > 0 
        ? (this.metrics.webassemblyProcessing / this.metrics.filesProcessed) * 100 
        : 0,
      errorRate: this.metrics.filesProcessed > 0 
        ? (this.metrics.errors / this.metrics.filesProcessed) * 100 
        : 0
    };
  }

  // Private helper methods

  private async extractTextFromFile(file: File): Promise<string> {
    const mimeType = file.type;

    // Handle different file types
    if (mimeType === 'text/plain') {
      return await this.readTextFile(file);
    } else if (mimeType === 'application/pdf') {
      return await this.extractPDFText(file);
    } else if (mimeType.includes('image/')) {
      return await this.extractImageText(file);
    } else if (mimeType.includes('word') || mimeType.includes('doc')) {
      return await this.extractDocumentText(file);
    } else {
      // Try to read as text for other formats
      try {
        return await this.readTextFile(file);
      } catch {
        throw new Error(`Unsupported file type: ${mimeType}`);
      }
    }
  }

  private async readTextFile(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (e) => resolve(e.target?.result as string);
      reader.onerror = (e) => reject(e);
      reader.readAsText(file);
    });
  }

  private async extractPDFText(file: File): Promise<string> {
    // Use enhanced RAG service for PDF extraction
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch('/api/enhanced-rag/extract-pdf', {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      throw new Error(`PDF extraction failed: ${response.status}`);
    }

    const result = await response.json();
    return result.text || '';
  }

  private async extractImageText(file: File): Promise<string> {
    // Use OCR service for image text extraction
    const formData = new FormData();
    formData.append('image', file);

    const response = await fetch('/api/ocr/extract', {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      throw new Error(`OCR extraction failed: ${response.status}`);
    }

    const result = await response.json();
    return result.text || '';
  }

  private async extractDocumentText(file: File): Promise<string> {
    // Use upload service for document parsing
    const formData = new FormData();
    formData.append('document', file);

    const response = await fetch('/api/upload/extract-text', {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      throw new Error(`Document extraction failed: ${response.status}`);
    }

    const result = await response.json();
    return result.text || '';
  }

  private async analyzeImageFile(file: File): Promise<any> {
    try {
      // Convert file to ImageData for vertex analysis
      const imageData = await this.fileToImageData(file);
      
      const analysisResult = await vertexBufferImageAnalyzer.analyzeImage(imageData, {
        extractVertexBuffers: true,
        generateGeometryFeatures: true,
        createWebGPUTextures: false, // Skip for upload processing
        enableCaching: true,
        outputFormat: 'float32'
      });

      return {
        vertexCount: analysisResult.metadata.vertexCount,
        faceCount: analysisResult.metadata.faceCount,
        detectedObjects: analysisResult.metadata.detectedObjects,
        qualityScore: analysisResult.metadata.qualityScore,
        geometryFeatures: analysisResult.geometryFeatures,
        processingTime: analysisResult.metadata.processingTimeMs
      };

    } catch (error) {
      console.warn('⚠️ Image analysis failed:', error);
      return null;
    }
  }

  private async fileToImageData(file: File): Promise<ImageData> {
    return new Promise((resolve, reject) => {
      const img = new Image();
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');

      img.onload = () => {
        canvas.width = img.width;
        canvas.height = img.height;
        ctx?.drawImage(img, 0, 0);
        
        const imageData = ctx?.getImageData(0, 0, canvas.width, canvas.height);
        if (imageData) {
          resolve(imageData);
        } else {
          reject(new Error('Failed to get image data'));
        }
      };

      img.onerror = reject;
      img.src = URL.createObjectURL(file);
    });
  }

  private async generateEmbeddings(text: string): Promise<{
    vector: number[];
    dimensions: number;
    model: string;
  }> {
    try {
      // Use nomic-embed-text via Ollama
      const response = await fetch('http://localhost:11434/api/embeddings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: 'nomic-embed-text',
          prompt: text.substring(0, 2000) // Limit length for embedding
        })
      });

      if (!response.ok) {
        throw new Error(`Embedding generation failed: ${response.status}`);
      }

      const result = await response.json();
      this.metrics.vectorEmbeddings++;

      return {
        vector: result.embedding,
        dimensions: result.embedding.length,
        model: 'nomic-embed-text'
      };

    } catch (error) {
      console.error('❌ Embedding generation failed:', error);
      throw error;
    }
  }

  private async storeInDatabase(
    file: ProcessedFile,
    options: UploadProcessingOptions
  ): Promise<DatabaseStorageResult> {
    try {
      const payload = {
        originalName: file.originalName,
        mimeType: file.mimeType,
        size: file.size,
        extractedText: file.extractedText,
        analysis: file.analysis,
        embeddings: file.embeddings,
        imageAnalysis: file.imageAnalysis,
        userId: options.userId,
        caseId: options.caseId,
        processingMethod: file.method,
        processingTime: file.processingTime
      };

      const response = await fetch('/api/database/store-document', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        throw new Error(`Database storage failed: ${response.status}`);
      }

      const result = await response.json();
      this.metrics.databaseStores++;

      return {
        documentId: result.documentId,
        vectorId: result.vectorId,
        evidenceId: result.evidenceId,
        success: true
      };

    } catch (error) {
      console.error('❌ Database storage failed:', error);
      return {
        documentId: '',
        vectorId: '',
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error'
      };
    }
  }

  private isImageFile(file: File): boolean {
    return file.type.startsWith('image/');
  }

  private updateMetrics(file: ProcessedFile, processingTime: number): void {
    this.metrics.filesProcessed++;
    this.metrics.totalProcessingTime += processingTime;
    
    if (file.method === 'webassembly') {
      this.metrics.webassemblyProcessing++;
    }
  }
}

// Export singleton instance
export const gemma3UploadPgVectorService = new Gemma3UploadPgVectorService();