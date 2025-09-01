import type { RequestHandler } from './$types.js';

/**
 * Unified Document Processing API
 * Endpoint for complete document processing pipeline
 */

import { unifiedDocumentProcessor, type DocumentProcessingConfig } from "$lib/services/unified-document-processor.js";
import { URL } from "url";

export const POST: RequestHandler = async ({ request }) => {
  try {
    console.log('📄 Starting unified document processing request...');

    const formData = await request.formData();
    
    // Extract files
    const files: File[] = [];
    const fileEntries = formData.getAll('files');
    for (const entry of fileEntries) {
      if (entry instanceof File) {
        files.push(entry);
      }
    }
    
    // Single file upload
    const singleFile = formData.get('file') as File;
    if (singleFile && !files.length) {
      files.push(singleFile);
    }

    if (files.length === 0) {
      throw error(400, 'No files provided');
    }

    // Extract configuration
    const config: DocumentProcessingConfig = {
      enableOCR: formData.get('enableOCR') === 'true' || true, // Default to true
      enableLegalBERT: formData.get('enableLegalBERT') === 'true' || true,
      enableEmbeddings: formData.get('enableEmbeddings') === 'true' || true,
      enableSummarization: formData.get('enableSummarization') === 'true' || true,
      enableMinIOStorage: formData.get('enableMinIOStorage') === 'true' || false,
      model: 'gemma3-legal:latest',
      chunkSize: parseInt(formData.get('chunkSize') as string) || 1000,
      confidence: parseFloat(formData.get('confidence') as string) || 0.7
    };

    // Extract metadata
    const metadata = {
      caseId: formData.get('caseId') as string,
      documentType: formData.get('documentType') as string || 'general',
      description: formData.get('description') as string || '',
      tags: (formData.get('tags') as string || '').split(',').map(t => t.trim()).filter(Boolean),
      userId: formData.get('userId') as string || 'anonymous',
      priority: formData.get('priority') as string || 'medium'
    };

    if (!metadata.caseId) {
      throw error(400, 'Case ID is required');
    }

    console.log(`📁 Processing ${files.length} file(s) for case ${metadata.caseId}`);
    console.log(`⚙️ Config: OCR=${config.enableOCR}, LegalBERT=${config.enableLegalBERT}, Embeddings=${config.enableEmbeddings}, Summarization=${config.enableSummarization}`);

    let results;
    
    if (files.length === 1) {
      // Single file processing
      results = await unifiedDocumentProcessor.processDocument(files[0], config, metadata);
    } else {
      // Batch processing
      results = await unifiedDocumentProcessor.batchProcess(files, config, metadata);
    }

    console.log('✅ Document processing completed successfully');

    return json({
      success: true,
      results,
      metadata: {
        filesProcessed: files.length,
        processingTime: Array.isArray(results) 
          ? results.reduce((sum, r) => sum + r.metadata.processingTime, 0)
          : results.metadata.processingTime,
        timestamp: new Date().toISOString()
      }
    });

  } catch (err: any) {
    console.error('❌ Document processing failed:', err);
    
    return json({
      success: false,
      error: err.message || 'Document processing failed',
      details: err.stack
    }, { 
      status: err.status || 500 
    });
  }
};

// Search endpoint
export const GET: RequestHandler = async ({ url }) => {
  try {
    const query = url.searchParams.get('q');
    const caseId = url.searchParams.get('caseId');
    const documentType = url.searchParams.get('documentType');
    const limit = parseInt(url.searchParams.get('limit') || '10');
    const threshold = parseFloat(url.searchParams.get('threshold') || '0.7');

    if (!query) {
      throw error(400, 'Query parameter "q" is required');
    }

    console.log(`🔍 Semantic search query: "${query}"`);

    const results = await unifiedDocumentProcessor.semanticSearch(query, {
      caseId: caseId || undefined,
      documentType: documentType || undefined,
      limit,
      threshold
    });

    console.log(`✅ Search completed: ${results.results.length} results found`);

    return json({
      success: true,
      query,
      results: results.results,
      metadata: {
        resultCount: results.results.length,
        processingTime: results.processingTime,
        threshold,
        timestamp: new Date().toISOString()
      }
    });

  } catch (err: any) {
    console.error('❌ Semantic search failed:', err);
    
    return json({
      success: false,
      error: err.message || 'Semantic search failed',
      details: err.stack
    }, { 
      status: err.status || 500 
    });
  }
};

// Health check endpoint
export const OPTIONS: RequestHandler = async () => {
  try {
    const health = await unifiedDocumentProcessor.healthCheck();
    
    return json({
      success: true,
      health,
      timestamp: new Date().toISOString()
    });
  } catch (err: any) {
    return json({
      success: false,
      error: err.message || 'Health check failed'
    }, { 
      status: 500 
    });
  }
};