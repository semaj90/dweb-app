
import type { RequestHandler } from './$types';

export const POST: RequestHandler = async ({ request }) => {
  try {
    const { ocrData } = await request.json();

    if (!ocrData) {
      throw error(400, 'No OCR data provided');
    }

    console.log('Converting OCR data to structured JSON...');

    // Structure the OCR data into a comprehensive legal document JSON
    const structuredData = {
      document: {
        metadata: {
          filename: ocrData.filename,
          processedAt: ocrData.extractedAt,
          totalPages: ocrData.pages,
          totalCharacters: ocrData.totalCharacters,
          averageConfidence: ocrData.averageConfidence,
          processingMethod: ocrData.processingStats
        },
        content: {
          fullText: ocrData.text,
          pages: ocrData.pageResults?.map((page: any, index: number) => ({
            pageNumber: page.page || index + 1,
            text: page.text,
            confidence: page.confidence,
            extractionMethod: page.method,
            wordCount: page.text?.split(/\s+/).length || 0,
            sections: extractSections(page.text || '')
          })) || []
        },
        legalAnalysis: {
          concepts: ocrData.legalConcepts || [],
          citations: ocrData.citations || [],
          documentType: classifyDocumentType(ocrData.text || ''),
          jurisdiction: extractJurisdiction(ocrData.text || ''),
          parties: extractParties(ocrData.text || ''),
          dates: extractDates(ocrData.text || ''),
          amounts: extractMonetaryAmounts(ocrData.text || '')
        },
        structure: {
          sections: identifyDocumentSections(ocrData.text || ''),
          headings: extractHeadings(ocrData.text || ''),
          paragraphs: (ocrData.text || '').split(/\n\s*\n/).filter(p => p.trim().length > 0),
          tableOfContents: generateTableOfContents(ocrData.text || '')
        },
        vectorization: {
          embeddings: generateEmbeddingIds(ocrData.text || ''),
          chunks: chunkTextForEmbedding(ocrData.text || ''),
          semanticSections: identifySemanticSections(ocrData.text || '')
        },
        qualityMetrics: {
          confidence: ocrData.averageConfidence,
          completeness: calculateCompleteness(ocrData),
          readability: calculateReadabilityScore(ocrData.text || ''),
          legalSpecificity: calculateLegalSpecificity(ocrData.legalConcepts || [])
        }
      }
    };

    return json({
      success: true,
      convertedAt: new Date().toISOString(),
      data: structuredData,
      stats: {
        jsonSize: JSON.stringify(structuredData).length,
        sections: structuredData.document.structure.sections.length,
        chunks: structuredData.document.vectorization.chunks.length,
        concepts: structuredData.document.legalAnalysis.concepts.length
      }
    });

  } catch (err: any) {
    console.error('JSON conversion error:', err);
    throw error(500, `JSON conversion failed: ${err.message}`);
  }
};

function extractSections(text: string): unknown[] {
  const sections = [];
  const lines = text.split('\n');
  let currentSection = { title: '', content: '', startLine: 0 };
  
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    
    // Check if line looks like a section header
    if (isHeaderLine(line)) {
      if (currentSection.content) {
        sections.push({ ...currentSection, endLine: i });
      }
      currentSection = {
        title: line,
        content: '',
        startLine: i
      };
    } else {
      currentSection.content += line + '\n';
    }
  }
  
  if (currentSection.content) {
    sections.push({ ...currentSection, endLine: lines.length });
  }
  
  return sections;
}

function isHeaderLine(line: string): boolean {
  // Patterns that suggest a header
  const headerPatterns = [
    /^[A-Z][A-Z\s]{2,}$/, // ALL CAPS
    /^\d+\.\s+[A-Z]/, // Numbered sections
    /^ARTICLE\s+[IVX]+/i, // Articles
    /^SECTION\s+\d+/i, // Sections
    /^EXHIBIT\s+[A-Z]/i // Exhibits
  ];
  
  return headerPatterns.some(pattern => pattern.test(line)) && line.length < 100;
}

function classifyDocumentType(text: string): string {
  const types = {
    'contract': /(?:agreement|contract|hereby agree|whereas)/gi,
    'motion': /(?:motion to|respectfully moves|comes now)/gi,
    'brief': /(?:brief in|memorandum of law|statement of facts)/gi,
    'complaint': /(?:complaint for|plaintiff alleges|cause of action)/gi,
    'settlement': /(?:settlement agreement|hereby settle|release and discharge)/gi,
    'lease': /(?:lease agreement|landlord|tenant|premises)/gi,
    'will': /(?:last will|testament|hereby bequeath)/gi,
    'corporate': /(?:board resolution|articles of incorporation|bylaws)/gi
  };
  
  for (const [type, pattern] of Object.entries(types)) {
    if (pattern.test(text)) {
      return type;
    }
  }
  
  return 'unknown';
}

function extractJurisdiction(text: string): string {
  const jurisdictionPatterns = [
    /(?:State of|Commonwealth of)\s+([A-Z][a-z]+)/g,
    /(?:United States|Federal|U\.S\.)/g,
    /(?:County of)\s+([A-Z][a-z]+)/g
  ];
  
  for (const pattern of jurisdictionPatterns) {
    const matches = text.match(pattern);
    if (matches) {
      return matches[0];
    }
  }
  
  return 'unknown';
}

function extractParties(text: string): string[] {
  const parties = new Set<string>();
  
  // Look for party designations
  const partyPatterns = [
    /(?:plaintiff|defendant|appellant|appellee|petitioner|respondent)[\s:]+([A-Z][a-zA-Z\s&,.]+?)(?:\s+(?:and|v\.|vs\.|versus))/gi,
    /([A-Z][a-zA-Z\s&,.]+?)\s+(?:v\.|vs\.|versus)\s+([A-Z][a-zA-Z\s&,.]+)/gi
  ];
  
  partyPatterns.forEach(pattern => {
    const matches = text.matchAll(pattern);
    for (const match of matches) {
      if (match[1]) parties.add(match[1].trim());
      if (match[2]) parties.add(match[2].trim());
    }
  });
  
  return Array.from(parties).slice(0, 10); // Limit to 10 parties
}

function extractDates(text: string): string[] {
  const dates = new Set<string>();
  
  const datePatterns = [
    /\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}/g,
    /\b\d{1,2}\/\d{1,2}\/\d{2,4}/g,
    /\b\d{1,2}-\d{1,2}-\d{2,4}/g
  ];
  
  datePatterns.forEach(pattern => {
    const matches = text.match(pattern);
    if (matches) {
      matches.forEach(match => dates.add(match));
    }
  });
  
  return Array.from(dates);
}

function extractMonetaryAmounts(text: string): string[] {
  const amounts = new Set<string>();
  
  const moneyPatterns = [
    /\$[\d,]+(?:\.\d{2})?/g,
    /(?:USD|dollars?)\s+[\d,]+(?:\.\d{2})?/gi,
    /[\d,]+(?:\.\d{2})?\s+(?:dollars?|USD)/gi
  ];
  
  moneyPatterns.forEach(pattern => {
    const matches = text.match(pattern);
    if (matches) {
      matches.forEach(match => amounts.add(match));
    }
  });
  
  return Array.from(amounts);
}

function identifyDocumentSections(text: string): unknown[] {
  const commonSections = [
    'recitals', 'definitions', 'scope of work', 'payment terms',
    'termination', 'confidentiality', 'governing law', 'signatures',
    'exhibits', 'schedules', 'appendices'
  ];
  
  return commonSections
    .filter(section => {
      const pattern = new RegExp(`\\b${section}\\b`, 'gi');
      return pattern.test(text);
    })
    .map(section => ({
      name: section,
      found: true,
      position: text.toLowerCase().indexOf(section.toLowerCase())
    }))
    .sort((a, b) => a.position - b.position);
}

function extractHeadings(text: string): string[] {
  const lines = text.split('\n');
  return lines
    .filter(line => isHeaderLine(line.trim()))
    .map(line => line.trim())
    .slice(0, 20); // Limit to 20 headings
}

function generateTableOfContents(text: string): unknown[] {
  const headings = extractHeadings(text);
  return headings.map((heading, index) => ({
    level: determineHeadingLevel(heading),
    title: heading,
    order: index + 1
  }));
}

function determineHeadingLevel(heading: string): number {
  if (/^\d+\.\s+/.test(heading)) return 1; // "1. Main Section"
  if (/^\d+\.\d+\s+/.test(heading)) return 2; // "1.1 Subsection"
  if (/^[A-Z]{3,}/.test(heading)) return 1; // "ARTICLE I"
  return 2; // Default subsection
}

function generateEmbeddingIds(text: string): string[] {
  // Generate placeholder embedding IDs for vector database
  const chunks = chunkTextForEmbedding(text);
  return chunks.map((_, index) => `embedding_${Date.now()}_${index}`);
}

function chunkTextForEmbedding(text: string, maxChunkSize: number = 512): string[] {
  const sentences = text.match(/[^\.!?]+[\.!?]+/g) || [text];
  const chunks = [];
  let currentChunk = '';
  
  for (const sentence of sentences) {
    if ((currentChunk + sentence).length > maxChunkSize && currentChunk) {
      chunks.push(currentChunk.trim());
      currentChunk = sentence;
    } else {
      currentChunk += sentence;
    }
  }
  
  if (currentChunk) {
    chunks.push(currentChunk.trim());
  }
  
  return chunks.filter(chunk => chunk.length > 0);
}

function identifySemanticSections(text: string): unknown[] {
  const semanticPatterns = [
    { type: 'legal_obligation', pattern: /(?:shall|must|required to|obligated to)/gi },
    { type: 'conditional_clause', pattern: /(?:if|unless|provided that|subject to)/gi },
    { type: 'definition', pattern: /(?:means|defined as|refers to)/gi },
    { type: 'temporal_reference', pattern: /(?:within|after|before|during|upon)/gi }
  ];
  
  return semanticPatterns
    .map(({ type, pattern }) => {
      const matches = text.match(pattern) || [];
      return {
        type,
        count: matches.length,
        density: matches.length / (text.length / 1000) // matches per 1000 chars
      };
    })
    .filter(section => section.count > 0);
}

function calculateCompleteness(ocrData: any): number {
  let score = 0;
  
  if (ocrData.averageConfidence > 85) score += 25;
  if (ocrData.legalConcepts?.length > 0) score += 25;
  if (ocrData.citations?.length > 0) score += 25;
  if (ocrData.totalCharacters > 1000) score += 25;
  
  return Math.min(score, 100);
}

function calculateReadabilityScore(text: string): number {
  const sentences = (text.match(/[.!?]+/g) || []).length;
  const words = text.split(/\s+/).length;
  const averageWordsPerSentence = words / Math.max(sentences, 1);
  
  // Simplified readability score (lower is more readable)
  return Math.max(0, 100 - Math.min(averageWordsPerSentence * 2, 100));
}

function calculateLegalSpecificity(concepts: string[]): number {
  const specificTerms = [
    'breach', 'liability', 'negligence', 'contract', 'tort',
    'damages', 'jurisdiction', 'statute', 'precedent', 'remedy'
  ];
  
  const specificCount = concepts.filter(concept =>
    specificTerms.some(term => concept.includes(term))
  ).length;
  
  return Math.min((specificCount / Math.max(concepts.length, 1)) * 100, 100);
}