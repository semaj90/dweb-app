// Enhanced Sentence Splitter Service
// Provides advanced sentence splitting with configurable options

export interface SentenceSplitterOptions {
  minLength?: number;
  maxLength?: number;
  preserveAbbreviations?: boolean;
  customAbbreviations?: string[];
}

export class EnhancedSentenceSplitter {
  private options: Required<SentenceSplitterOptions>;
  private abbreviations: Set<string>;

  constructor(options: SentenceSplitterOptions = {}) {
    this.options = {
      minLength: options.minLength || 10,
      maxLength: options.maxLength || 500,
      preserveAbbreviations: options.preserveAbbreviations ?? true,
      customAbbreviations: options.customAbbreviations || []
    };

    // Common legal abbreviations
    this.abbreviations = new Set([
      'v.', 'vs.', 'No.', 'Inc.', 'Corp.', 'LLC', 'Ltd.', 'Co.',
      'U.S.', 'F.2d', 'F.3d', 'F.Supp.', 'P.2d', 'P.3d',
      'Cal.', 'N.Y.', 'Tex.', 'Fla.', 'Ill.', 'Pa.',
      'J.', 'C.J.', 'App.', 'Ct.', 'D.C.', 'Cir.',
      'Sec.', 'Subd.', 'Art.', 'Ch.', 'Div.', 'Par.',
      'e.g.', 'i.e.', 'etc.', 'cf.', 'id.', 'ibid.',
      ...this.options.customAbbreviations
    ]);
  }

  split(text: string): string[] {
    if (!text || typeof text !== 'string') {
      return [];
    }

    // Pre-process text to handle abbreviations
    const processedText = this.preprocessText(text);
    
    // Split on sentence boundaries
    const preliminarySentences = this.splitOnBoundaries(processedText);
    
    // Post-process to handle edge cases
    const cleanedSentences = this.postProcess(preliminarySentences);
    
    // Filter by length requirements
    return this.filterByLength(cleanedSentences);
  }

  addAbbreviations(abbreviations: string[]): void {
    for (const abbr of abbreviations) {
      this.abbreviations.add(abbr);
    }
  }

  private preprocessText(text: string): string {
    // Replace abbreviations with placeholders to prevent splitting
    let processed = text;
    
    if (this.options.preserveAbbreviations) {
      for (const abbr of this.abbreviations) {
        const escaped = abbr.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        const regex = new RegExp(`\\b${escaped}`, 'gi');
        processed = processed.replace(regex, abbr.replace('.', '||DOT||'));
      }
    }
    
    return processed;
  }

  private splitOnBoundaries(text: string): string[] {
    // Enhanced sentence boundary detection
    const sentences: string[] = [];
    let current = '';
    let inQuotes = false;
    let quoteChar = '';
    
    for (let i = 0; i < text.length; i++) {
      const char = text[i];
      const nextChar = text[i + 1] || '';
      const prevChar = text[i - 1] || '';
      
      current += char;
      
      // Track quote state
      if ((char === '"' || char === "'") && prevChar !== '\\') {
        if (!inQuotes) {
          inQuotes = true;
          quoteChar = char;
        } else if (char === quoteChar) {
          inQuotes = false;
          quoteChar = '';
        }
      }
      
      // Check for sentence boundary
      if (!inQuotes && this.isSentenceBoundary(char, nextChar, text, i)) {
        sentences.push(current.trim());
        current = '';
      }
    }
    
    // Add remaining text
    if (current.trim()) {
      sentences.push(current.trim());
    }
    
    return sentences;
  }

  private isSentenceBoundary(char: string, nextChar: string, text: string, position: number): boolean {
    if (!['.', '!', '?'].includes(char)) {
      return false;
    }
    
    // Don't split on abbreviations
    const before = text.substring(Math.max(0, position - 10), position + 1);
    for (const abbr of this.abbreviations) {
      if (before.toLowerCase().endsWith(abbr.replace('.', '||DOT||').toLowerCase())) {
        return false;
      }
    }
    
    // Don't split on decimal numbers
    const prevChar = text[position - 1] || '';
    if (char === '.' && /\d/.test(prevChar) && /\d/.test(nextChar)) {
      return false;
    }
    
    // Must be followed by whitespace and capital letter or end of text
    return !nextChar || /\s/.test(nextChar) && /[A-Z]/.test(text[position + 2] || '');
  }

  private postProcess(sentences: string[]): string[] {
    return sentences.map(sentence => {
      // Restore abbreviations
      return sentence.replace(/\|\|DOT\|\|/g, '.');
    }).filter(sentence => sentence.length > 0);
  }

  private filterByLength(sentences: string[]): string[] {
    return sentences.filter(sentence => {
      const length = sentence.length;
      return length >= this.options.minLength && length <= this.options.maxLength;
    });
  }

  // Static factory method for backward compatibility
  static create(options: SentenceSplitterOptions = {}): EnhancedSentenceSplitter {
    return new EnhancedSentenceSplitter(options);
  }
}

// Default export for backward compatibility
export default function createStreamingSplitter(options: SentenceSplitterOptions = {}): EnhancedSentenceSplitter {
  return new EnhancedSentenceSplitter(options);
}

// Named export
export { createStreamingSplitter };
