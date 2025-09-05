// Legal entity extraction for Neo4j knowledge graph
// Extracts persons, organizations, locations, legal concepts

import { createHash } from 'crypto';

export class EntityExtractor {
  constructor(options = {}) {
    this.confidenceThreshold = options.confidenceThreshold || 0.7;
    this.enableLegalConcepts = options.enableLegalConcepts ?? true;
    
    // Legal-specific entity patterns
    this.patterns = {
      // Person names (common legal patterns)
      PERSON: [
        /\b([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b/g, // John Smith, Mary Jane Doe
        /\b(Mr\.|Mrs\.|Ms\.|Dr\.)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)/gi,
        /\b([A-Z][a-z]+),\s+(Esq\.|Attorney|Counsel)/gi,
      ],
      
      // Organizations
      ORGANIZATION: [
        /\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(Inc\.|Corp\.|LLC|LLP|Ltd\.|Company|Corporation)/gi,
        /\b(The\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(Company|Corporation|Group)/gi,
        /\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(Bank|Insurance|Fund|Trust)/gi,
      ],
      
      // Locations
      LOCATION: [
        /\b(\d+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Street|Avenue|Road|Drive|Lane|Boulevard))/gi,
        /\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),\s+([A-Z]{2})\s+(\d{5}(?:-\d{4})?)/g, // City, State ZIP
        /\b(County\s+of\s+[A-Z][a-z]+)/gi,
        /\b(State\s+of\s+[A-Z][a-z]+)/gi,
      ],
      
      // Legal concepts and terms
      LEGAL_CONCEPT: [
        /\b(breach\s+of\s+contract|negligence|liability|damages|injunction|restraining\s+order)\b/gi,
        /\b(intellectual\s+property|trademark|copyright|patent|trade\s+secret)\b/gi,
        /\b(employment\s+agreement|non-disclosure|confidentiality|non-compete)\b/gi,
        /\b(force\s+majeure|indemnification|arbitration|jurisdiction|venue)\b/gi,
        /\b(plaintiff|defendant|petitioner|respondent|appellant|appellee)\b/gi,
      ],
      
      // Dates and case references
      DATE: [
        /\b(\d{1,2}\/\d{1,2}\/\d{2,4})\b/g,
        /\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}/gi,
      ],
      
      // Case citations
      CASE_CITATION: [
        /\b(\d+\s+[A-Z][a-z]+\.?\s*\d+)/g, // 123 F.3d 456
        /\b([A-Z][a-z]+\s+v\.\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)/g, // Smith v. Jones
      ],
      
      // Money amounts
      MONEY: [
        /\$[\d,]+(?:\.\d{2})?/g,
        /\b(\d+(?:,\d{3})*(?:\.\d{2})?\s+dollars?)/gi,
      ],
      
      // Contract terms
      CONTRACT_TERM: [
        /\b(term|duration|period)\s+of\s+(\d+\s+(?:year|month|day)s?)/gi,
        /\b(effective\s+date|expiration\s+date|termination\s+date)/gi,
        /\b(payment\s+terms?|billing\s+cycle|due\s+date)/gi,
      ],
    };

    // Legal stop words to filter out
    this.stopWords = new Set([
      'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
      'said', 'such', 'any', 'all', 'each', 'every', 'other', 'same', 'including',
      'shall', 'will', 'may', 'must', 'should', 'would', 'could', 'might',
      'party', 'parties', 'agreement', 'contract', 'document', 'section', 'clause',
    ]);
  }

  // Main extraction method
  static async extractEntities(text, options = {}) {
    const extractor = new EntityExtractor(options);
    return extractor.extract(text);
  }

  async extract(text) {
    const entities = [];
    
    // Extract by pattern type
    for (const [type, patterns] of Object.entries(this.patterns)) {
      for (const pattern of patterns) {
        let match;
        while ((match = pattern.exec(text)) !== null) {
          const entityText = match[1] || match[0];
          const confidence = this.calculateConfidence(entityText, type, text);
          
          if (confidence >= this.confidenceThreshold) {
            entities.push({
              text: entityText.trim(),
              label: type,
              confidence,
              startOffset: match.index,
              endOffset: match.index + match[0].length,
              context: this.extractContext(text, match.index, 100),
            });
          }
        }
      }
    }

    // Deduplicate and normalize
    const deduped = this.deduplicateEntities(entities);
    
    // Add relationships
    const withRelations = this.extractRelationships(deduped, text);
    
    return withRelations.sort((a, b) => b.confidence - a.confidence);
  }

  // Calculate confidence score for entity
  calculateConfidence(entityText, type, fullText) {
    let confidence = 0.5; // Base confidence
    
    // Boost confidence based on context
    const context = fullText.toLowerCase();
    const entity = entityText.toLowerCase();
    
    switch (type) {
      case 'PERSON':
        // Higher confidence for names with titles
        if (/\b(mr|mrs|ms|dr|attorney|counsel|esq)\b/.test(context)) confidence += 0.2;
        // Lower confidence for common words
        if (this.isCommonWord(entity)) confidence -= 0.3;
        // Proper capitalization
        if (/^[A-Z][a-z]+\s+[A-Z][a-z]+/.test(entityText)) confidence += 0.2;
        break;
        
      case 'ORGANIZATION':
        // Legal entity suffixes
        if (/(inc|corp|llc|ltd)\.?\s*$/i.test(entityText)) confidence += 0.3;
        // Context clues
        if (/\b(company|corporation|business|firm)\b/.test(context)) confidence += 0.1;
        break;
        
      case 'LOCATION':
        // Address patterns
        if (/\d+\s+/.test(entityText)) confidence += 0.2;
        // State abbreviations
        if (/\b[A-Z]{2}\b/.test(entityText)) confidence += 0.2;
        break;
        
      case 'LEGAL_CONCEPT':
        // Legal document context
        if (/\b(whereas|therefore|party|agreement|contract)\b/i.test(context)) confidence += 0.2;
        break;
        
      case 'CASE_CITATION':
        // Citation format validation
        if (/\d+.*\d+/.test(entityText)) confidence += 0.3;
        break;
        
      case 'MONEY':
        // Currency symbols and context
        if (/\$/.test(entityText)) confidence += 0.2;
        if (/\b(damages|payment|fee|cost|price)\b/i.test(context)) confidence += 0.1;
        break;
    }
    
    // Frequency boost (mentioned multiple times)
    const occurrences = (fullText.match(new RegExp(this.escapeRegex(entityText), 'gi')) || []).length;
    if (occurrences > 1) {
      confidence += Math.min(0.2, occurrences * 0.05);
    }
    
    return Math.min(1.0, Math.max(0.0, confidence));
  }

  // Check if text is a common word
  isCommonWord(text) {
    return this.stopWords.has(text.toLowerCase()) || 
           /^(a|an|the|this|that|these|those)$/i.test(text);
  }

  // Extract surrounding context
  extractContext(text, position, windowSize = 100) {
    const start = Math.max(0, position - windowSize);
    const end = Math.min(text.length, position + windowSize);
    return text.slice(start, end).trim();
  }

  // Remove duplicate entities
  deduplicateEntities(entities) {
    const seen = new Map();
    const deduplicated = [];
    
    for (const entity of entities) {
      const key = `${entity.label}:${entity.text.toLowerCase()}`;
      
      if (seen.has(key)) {
        const existing = seen.get(key);
        if (entity.confidence > existing.confidence) {
          // Replace with higher confidence entity
          const index = deduplicated.findIndex(e => e === existing);
          if (index >= 0) {
            deduplicated[index] = entity;
            seen.set(key, entity);
          }
        }
      } else {
        deduplicated.push(entity);
        seen.set(key, entity);
      }
    }
    
    return deduplicated;
  }

  // Extract relationships between entities
  extractRelationships(entities, text) {
    const relationships = [];
    
    // Simple co-occurrence relationships
    for (let i = 0; i < entities.length; i++) {
      for (let j = i + 1; j < entities.length; j++) {
        const entity1 = entities[i];
        const entity2 = entities[j];
        
        // Check if entities appear in same sentence/paragraph
        const distance = Math.abs(entity1.startOffset - entity2.startOffset);
        if (distance < 200) { // Within 200 characters
          const relationship = this.inferRelationshipType(entity1, entity2, text);
          
          if (relationship) {
            relationships.push({
              source: entity1.text,
              target: entity2.text,
              type: relationship,
              confidence: Math.min(entity1.confidence, entity2.confidence) * 0.8,
              distance,
            });
          }
        }
      }
    }

    // Add relationships to entities
    return entities.map(entity => ({
      ...entity,
      relationships: relationships.filter(r => 
        r.source === entity.text || r.target === entity.text
      ),
    }));
  }

  // Infer relationship type between entities
  inferRelationshipType(entity1, entity2, text) {
    const context = text.slice(
      Math.min(entity1.startOffset, entity2.startOffset) - 50,
      Math.max(entity1.endOffset, entity2.endOffset) + 50
    ).toLowerCase();

    // Person-Organization relationships
    if (entity1.label === 'PERSON' && entity2.label === 'ORGANIZATION') {
      if (/\b(employee|work|employ|hire)\b/.test(context)) return 'EMPLOYED_BY';
      if (/\b(represent|attorney|counsel)\b/.test(context)) return 'REPRESENTS';
      if (/\b(ceo|president|director|manager)\b/.test(context)) return 'LEADS';
    }

    // Person-Person relationships
    if (entity1.label === 'PERSON' && entity2.label === 'PERSON') {
      if (/\b(spouse|husband|wife|married)\b/.test(context)) return 'MARRIED_TO';
      if (/\b(partner|business partner)\b/.test(context)) return 'PARTNER_OF';
      if (/\bv\.\b/.test(context)) return 'LEGAL_DISPUTE'; // "Smith v. Jones"
    }

    // Organization-Location relationships
    if (entity1.label === 'ORGANIZATION' && entity2.label === 'LOCATION') {
      if (/\b(located|headquarter|office|address)\b/.test(context)) return 'LOCATED_AT';
    }

    // Legal concept relationships
    if (entity1.label === 'LEGAL_CONCEPT' || entity2.label === 'LEGAL_CONCEPT') {
      return 'INVOLVES';
    }

    // Contract term relationships
    if (entity1.label === 'CONTRACT_TERM' || entity2.label === 'CONTRACT_TERM') {
      return 'CONTRACT_ELEMENT';
    }

    // Default co-occurrence
    return 'MENTIONED_WITH';
  }

  // Escape regex special characters
  escapeRegex(text) {
    return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  }

  // Generate entity ID for Neo4j
  generateEntityId(entityText, entityType) {
    const hash = createHash('sha256')
      .update(`${entityType}:${entityText.toLowerCase().trim()}`)
      .digest('hex')
      .substring(0, 8);
    return `${entityType.toLowerCase()}_${hash}`;
  }

  // Normalize entity name for deduplication
  normalizeEntityName(text, type) {
    let normalized = text.toLowerCase().trim();
    
    // Remove common prefixes/suffixes
    if (type === 'PERSON') {
      normalized = normalized.replace(/\b(mr|mrs|ms|dr|esq|attorney|counsel)\.?\s*/gi, '');
    } else if (type === 'ORGANIZATION') {
      normalized = normalized.replace(/\b(inc|corp|llc|ltd|company|corporation)\.?\s*$/gi, '');
    }
    
    // Remove extra whitespace
    normalized = normalized.replace(/\s+/g, ' ');
    
    return normalized;
  }

  // Advanced pattern matching for specific legal document types
  extractContractEntities(text) {
    const contractPatterns = {
      PARTIES: /\b(party|parties)\s+(?:to\s+)?(?:this\s+)?(?:agreement|contract)/gi,
      EFFECTIVE_DATE: /effective\s+(?:date|as\s+of)\s*:?\s*([^,\n]+)/gi,
      TERMINATION: /terminat\w+\s+(?:date|on|upon)\s*:?\s*([^,\n]+)/gi,
      GOVERNING_LAW: /govern\w+\s+by\s+(?:the\s+)?laws?\s+of\s+([^,\n]+)/gi,
      JURISDICTION: /jurisdiction\s+(?:of|in)\s+([^,\n]+)/gi,
    };

    const contractEntities = [];
    
    for (const [type, pattern] of Object.entries(contractPatterns)) {
      let match;
      while ((match = pattern.exec(text)) !== null) {
        contractEntities.push({
          text: match[1] || match[0],
          label: `CONTRACT_${type}`,
          confidence: 0.8,
          startOffset: match.index,
          endOffset: match.index + match[0].length,
          context: this.extractContext(text, match.index, 80),
        });
      }
    }
    
    return contractEntities;
  }
}