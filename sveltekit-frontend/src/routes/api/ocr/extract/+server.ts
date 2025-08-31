import * as pdfjsLib from "pdfjs-dist";
import { createWorker } from "tesseract.js";
import type { RequestHandler } from './$types';


export const POST: RequestHandler = async ({ request }) => {
  try {
    const formData = await request.formData();
    const file = formData.get('file') as File;

    if (!file) {
      throw error(400, 'No file provided');
    }

    if (!file.type.includes('pdf')) {
      throw error(400, 'Only PDF files are supported');
    }

    console.log(`Processing PDF: ${file.name} (${file.size} bytes)`);

    // Convert PDF to images and then OCR
    const arrayBuffer = await file.arrayBuffer();
    const typedArray = new Uint8Array(arrayBuffer);
    
    const loadingTask = pdfjsLib.getDocument(typedArray);
    const pdf = await loadingTask.promise;
    
    const ocrResults = [];
    let totalConfidence = 0;
    let totalCharacters = 0;

    // Process each page
    for (let pageNum = 1; pageNum <= pdf.numPages; pageNum++) {
      const page = await pdf.getPage(pageNum);
      
      // Get page text content first (if available)
      const textContent = await page.getTextContent();
      let pageText = '';
      
      if (textContent.items.length > 0) {
        // PDF has extractable text
        pageText = textContent.items
          .filter((item): item is any => 'str' in item)
          .map((item) => item.str)
          .join(' ');
        
        ocrResults.push({
          page: pageNum,
          text: pageText,
          confidence: 95, // High confidence for extractable text
          method: 'text_extraction'
        });
        
        totalCharacters += pageText.length;
        totalConfidence += 95;
      } else {
        // Need OCR for scanned pages
        const viewport = page.getViewport({ scale: 2.0 });
        const canvas = new OffscreenCanvas(viewport.width, viewport.height);
        const context = canvas.getContext('2d') as any as CanvasRenderingContext2D;
        
        await page.render({
          canvas: canvas as any,
          canvasContext: context,
          viewport: viewport
        }).promise;
        
        // Initialize Tesseract worker
        const worker = await createWorker('eng', 1, {
          workerPath: '/tesseract-worker.js',
          corePath: '/tesseract-core.js',
        });
        
        // Perform OCR using the canvas directly
        const { data } = await worker.recognize(canvas);
        await worker.terminate();
        
        ocrResults.push({
          page: pageNum,
          text: data.text,
          confidence: data.confidence,
          method: 'ocr'
        });
        
        totalCharacters += data.text.length;
        totalConfidence += data.confidence;
      }
    }

    // Calculate average confidence
    const averageConfidence = totalConfidence / pdf.numPages;

    // Extract legal concepts and citations
    const allText = ocrResults.map(r => r.text).join('\n\n');
    const legalConcepts = extractLegalConcepts(allText);
    const citations = extractCitations(allText);

    const result = {
      filename: file.name,
      pages: pdf.numPages,
      totalCharacters,
      averageConfidence: Math.round(averageConfidence),
      text: allText,
      pageResults: ocrResults,
      legalConcepts,
      citations,
      extractedAt: new Date().toISOString(),
      processingStats: {
        ocrPages: ocrResults.filter(r => r.method === 'ocr').length,
        extractedPages: ocrResults.filter(r => r.method === 'text_extraction').length
      }
    };

    return json(result);

  } catch (err: any) {
    console.error('OCR processing error:', err);
    throw error(500, `OCR processing failed: ${err.message}`);
  }
};

function extractLegalConcepts(text: string): string[] {
  const concepts = new Set<string>();
  
  // Legal concept patterns
  const patterns = [
    // Contract terms
    /(?:breach\s+of\s+contract|contract\s+formation|consideration|offer\s+and\s+acceptance)/gi,
    // Liability terms
    /(?:negligence|liability|damages|indemnification|limitation\s+of\s+liability)/gi,
    // Intellectual property
    /(?:copyright|trademark|patent|intellectual\s+property|trade\s+secret)/gi,
    // Corporate terms
    /(?:shareholder|board\s+of\s+directors|merger|acquisition|corporate\s+governance)/gi,
    // Legal procedures
    /(?:discovery|deposition|motion\s+to\s+dismiss|summary\s+judgment|trial)/gi
  ];

  patterns.forEach(pattern => {
    const matches = text.match(pattern);
    if (matches) {
      matches.forEach(match => concepts.add(match.toLowerCase()));
    }
  });

  return Array.from(concepts);
}

function extractCitations(text: string): string[] {
  const citations = new Set<string>();
  
  // Citation patterns
  const patterns = [
    // Federal courts
    /\b\d+\s+F\.\s*(?:2d|3d)\s+\d+/g,
    // Supreme Court
    /\b\d+\s+U\.S\.\s+\d+/g,
    // State courts
    /\b\d+\s+[A-Z][a-z]*\s+(?:2d|3d)?\s*\d+/g,
    // Statutes
    /\b\d+\s+U\.S\.C\.\s*§?\s*\d+/g,
    // Code of Federal Regulations
    /\b\d+\s+C\.F\.R\.\s*§?\s*\d+/g
  ];

  patterns.forEach(pattern => {
    const matches = text.match(pattern);
    if (matches) {
      matches.forEach(match => citations.add(match.trim()));
    }
  });

  return Array.from(citations);
}