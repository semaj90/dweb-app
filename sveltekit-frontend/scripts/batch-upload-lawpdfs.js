#!/usr/bin/env node
/**
 * Batch Legal PDF Processor
 * Uploads and processes all PDF documents from the lawpdfs folder
 * Uses the real database-integrated RAG pipeline
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Configuration
const LAWPDFS_FOLDER = path.join(__dirname, '../../lawpdfs');
const EVIDENCE_PROCESS_URL = 'http://localhost:5173/api/evidence/process';
const WS_STREAM_URL = 'ws://localhost:5173/api/evidence/stream';
const OLD_RAG_API_URL = 'http://localhost:5177/api/rag/process'; // Fallback
const SEARCH_API_URL = 'http://localhost:5177/api/rag/search';

// Colors for console output
const colors = {
  reset: '\x1b[0m',
  green: '\x1b[32m',
  red: '\x1b[31m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  cyan: '\x1b[36m'
};

async function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function uploadDocument(filePath, filename) {
  try {
    console.log(`${colors.blue}📄 Processing: ${filename}${colors.reset}`);

    // First, try the new evidence processing system
    try {
      const evidenceId = `evidence_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      
      console.log(`${colors.cyan}🔄 Starting evidence processing pipeline...${colors.reset}`);

      // Start evidence processing
      const processResponse = await fetch(EVIDENCE_PROCESS_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          evidenceId: evidenceId,
          steps: ['ocr', 'embedding', 'analysis']
        })
      });

      if (!processResponse.ok) {
        throw new Error(`Evidence processing failed: ${processResponse.status}`);
      }

      const { sessionId } = await processResponse.json();
      console.log(`${colors.green}✅ Evidence processing started`);
      console.log(`   📝 Evidence ID: ${evidenceId}`);
      console.log(`   🔗 Session ID: ${sessionId}`);

      // Monitor progress via WebSocket (simplified for batch script)
      const progressResult = await monitorProcessingProgress(sessionId, evidenceId);
      
      return { 
        success: true, 
        documentId: evidenceId, 
        sessionId: sessionId,
        filename,
        processingResult: progressResult
      };

    } catch (newSystemError) {
      console.log(`${colors.yellow}⚠️ New system failed, trying fallback...${colors.reset}`);
      console.log(`   New system error: ${newSystemError.message}`);

      // Fallback to old RAG system
      return await uploadDocumentFallback(filePath, filename);
    }

  } catch (error) {
    console.log(`${colors.red}❌ Failed to process: ${filename}`);
    console.log(`   Error: ${error.message}`);
    return { success: false, filename, error: error.message };
  }
}

async function uploadDocumentFallback(filePath, filename) {
  try {
    console.log(`${colors.yellow}📄 Fallback processing: ${filename}${colors.reset}`);

    // Read the PDF file
    const fileBuffer = fs.readFileSync(filePath);

    // Create FormData for upload
    const formData = new FormData();
    const file = new File([fileBuffer], filename, { type: 'application/pdf' });

    formData.append('files', file);
    formData.append('enableOCR', 'true');
    formData.append('enableEmbedding', 'true');
    formData.append('enableRAG', 'true');

    // Upload to old RAG processing pipeline
    const response = await fetch(OLD_RAG_API_URL, {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const result = await response.json();

    if (result.success && result.results && result.results.length > 0) {
      const docResult = result.results[0];
      console.log(`${colors.green}✅ Successfully processed (fallback): ${filename}`);
      console.log(`   📝 Document ID: ${docResult.documentId}`);
      console.log(`   📊 Content length: ${docResult.contentLength || 'N/A'} characters`);
      console.log(`   🔗 Embeddings: ${docResult.embeddingGenerated ? 'Generated' : 'Skipped'}`);
      console.log(`   ⚡ Processing time: ${docResult.processingTime || 'N/A'}`);
      return { success: true, documentId: docResult.documentId, filename, fallback: true };
    } else {
      throw new Error(result.error || 'Unknown processing error');
    }

  } catch (error) {
    throw error; // Re-throw to be handled by main function
  }
}

async function monitorProcessingProgress(sessionId, evidenceId) {
  return new Promise((resolve) => {
    const WebSocket = globalThis.WebSocket || require('ws');
    
    try {
      const wsUrl = `${WS_STREAM_URL}/${sessionId}`;
      console.log(`${colors.blue}📡 Connecting to WebSocket: ${wsUrl}${colors.reset}`);
      
      const ws = new WebSocket(wsUrl);
      let progressData = {};
      
      const timeout = setTimeout(() => {
        ws.close();
        resolve({ status: 'timeout', message: 'Processing monitoring timed out after 30 seconds' });
      }, 30000); // 30 second timeout
      
      ws.on('open', () => {
        console.log(`${colors.green}🔗 WebSocket connected for ${evidenceId}${colors.reset}`);
      });
      
      ws.on('message', (data) => {
        try {
          const message = JSON.parse(data.toString());
          
          if (message.type === 'processing-step') {
            console.log(`${colors.cyan}  📊 Step: ${message.step} (${message.stepProgress || 0}%)${colors.reset}`);
            progressData[message.step] = message.stepProgress;
          } else if (message.type === 'processing-complete') {
            console.log(`${colors.green}  ✅ Processing completed!${colors.reset}`);
            clearTimeout(timeout);
            ws.close();
            resolve({ status: 'completed', result: message.finalResult, progress: progressData });
          } else if (message.type === 'error') {
            console.log(`${colors.red}  ❌ Processing error: ${message.error.message}${colors.reset}`);
            clearTimeout(timeout);
            ws.close();
            resolve({ status: 'error', error: message.error, progress: progressData });
          }
        } catch (parseError) {
          console.log(`${colors.yellow}  ⚠️ WebSocket message parse error: ${parseError.message}${colors.reset}`);
        }
      });
      
      ws.on('error', (error) => {
        console.log(`${colors.yellow}  ⚠️ WebSocket error: ${error.message}${colors.reset}`);
        clearTimeout(timeout);
        resolve({ status: 'websocket_error', error: error.message, progress: progressData });
      });
      
      ws.on('close', () => {
        console.log(`${colors.blue}  📡 WebSocket connection closed${colors.reset}`);
        clearTimeout(timeout);
      });
      
    } catch (wsError) {
      console.log(`${colors.yellow}  ⚠️ WebSocket setup failed: ${wsError.message}${colors.reset}`);
      resolve({ status: 'websocket_setup_error', error: wsError.message });
    }
  });
}

async function testSearch(query) {
  try {
    console.log(`${colors.cyan}🔍 Testing search: "${query}"${colors.reset}`);

    const response = await fetch(SEARCH_API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        searchType: 'hybrid',
        limit: 5,
        threshold: 0.7
      })
    });

    if (!response.ok) {
      throw new Error(`Search failed: ${response.status}`);
    }

    const result = await response.json();

    if (result.success && result.results) {
      console.log(`${colors.green}📊 Found ${result.results.length} results in ${result.processingTime}${colors.reset}`);

      result.results.forEach((doc, index) => {
        console.log(`   ${index + 1}. ${doc.filename} (similarity: ${(doc.similarity * 100).toFixed(1)}%)`);
        console.log(`      Content preview: ${doc.content.substring(0, 100)}...`);
      });
    } else {
      console.log(`${colors.yellow}⚠️  No results found for: "${query}"${colors.reset}`);
    }

  } catch (error) {
    console.log(`${colors.red}❌ Search error: ${error.message}${colors.reset}`);
  }
}

async function main() {
  console.log(`${colors.cyan}🚀 Starting Legal PDF Batch Processing${colors.reset}`);
  console.log(`📁 Source folder: ${LAWPDFS_FOLDER}`);

  // Check if folder exists
  if (!fs.existsSync(LAWPDFS_FOLDER)) {
    console.log(`${colors.red}❌ Folder not found: ${LAWPDFS_FOLDER}${colors.reset}`);
    process.exit(1);
  }

  // Get all PDF files
  const files = fs.readdirSync(LAWPDFS_FOLDER)
    .filter(file => file.toLowerCase().endsWith('.pdf'))
    .sort();

  console.log(`📄 Found ${files.length} PDF documents to process`);
  console.log('');

  const results = {
    successful: [],
    failed: [],
    startTime: new Date()
  };

  // Process each PDF
  for (let i = 0; i < files.length; i++) {
    const filename = files[i];
    const filePath = path.join(LAWPDFS_FOLDER, filename);

    console.log(`${colors.yellow}[${i + 1}/${files.length}]${colors.reset} Processing...`);

    const result = await uploadDocument(filePath, filename);

    if (result.success) {
      results.successful.push(result);
    } else {
      results.failed.push(result);
    }

    // Add delay to avoid overwhelming the system
    if (i < files.length - 1) {
      console.log(`${colors.blue}⏳ Waiting 2 seconds before next upload...${colors.reset}`);
      await sleep(2000);
    }

    console.log('');
  }

  // Display final results
  const endTime = new Date();
  const totalTime = ((endTime - results.startTime) / 1000).toFixed(1);

  console.log(`${colors.cyan}📊 BATCH PROCESSING COMPLETE${colors.reset}`);
  console.log('');
  console.log(`⏰ Total time: ${totalTime} seconds`);
  console.log(`${colors.green}✅ Successful: ${results.successful.length}${colors.reset}`);
  console.log(`${colors.red}❌ Failed: ${results.failed.length}${colors.reset}`);

  if (results.failed.length > 0) {
    console.log('');
    console.log(`${colors.red}Failed Documents:${colors.reset}`);
    results.failed.forEach(item => {
      console.log(`  • ${item.filename}: ${item.error}`);
    });
  }

  // Run some test searches if we have successful uploads
  if (results.successful.length > 0) {
    console.log('');
    console.log(`${colors.cyan}🔍 TESTING SEARCH FUNCTIONALITY${colors.reset}`);
    console.log('');

    const testQueries = [
      'human trafficking legislation',
      'criminal sentencing guidelines',
      'federal prison corruption',
      'sex offender registry',
      'People v. Villegas case',
      'force majeure contract law'
    ];

    for (const query of testQueries) {
      await testSearch(query);
      await sleep(1000); // Brief pause between searches
      console.log('');
    }
  }

  console.log(`${colors.green}🎉 Legal PDF processing pipeline complete!${colors.reset}`);
}

// Handle errors gracefully
process.on('unhandledRejection', (error) => {
  console.log(`${colors.red}❌ Unhandled error: ${error.message}${colors.reset}`);
  process.exit(1);
});

// Run the batch processor
main().catch(error => {
  console.log(`${colors.red}❌ Fatal error: ${error.message}${colors.reset}`);
  process.exit(1);
});
