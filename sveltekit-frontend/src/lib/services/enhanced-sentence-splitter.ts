/**
 * Enhanced Sentence Splitter for text processing
 * Supports legal abbreviations, fragment merging, and streaming
 */

interface SplitterOptions {
  minLength?: number;
  maxLength?: number;
  minFragmentLength?: number;
  mergeThreshold?: number;
  streamBufferSize?: number;
  headingPatterns?: RegExp[]; // custom heading patterns to merge with following sentence
}

interface StreamingContext {
  buffer: string;
  processedSentences: string[];
}

export class EnhancedSentenceSplitter {
  private minLength: number;
  private maxLength: number;
  private minFragmentLength: number;
  private mergeThreshold: number;
  private streamBufferSize: number;
  private customAbbreviations: Set<string>;
  private abbreviationRegexes: { abbr: string; regex: RegExp }[];
  private headingPatterns: RegExp[];

  // Common legal and business abbreviations
  private readonly defaultAbbreviations = new Set([
    'Inc.', 'Corp.', 'Ltd.', 'L.L.C.', 'LLC', 'P.C.', 'P.A.',
    'U.S.', 'U.S.C.', 'U.S.A.', 'Art.', 'Sec.', 'Para.',
    'v.', 'vs.', 'No.', 'Dr.', 'Mr.', 'Mrs.', 'Ms.',
    'Ph.D.', 'M.D.', 'J.D.', 'Esq.', 'Jr.', 'Sr.',
    'Co.', 'Bros.', 'Assoc.', 'Dept.', 'Est.',
    'Jan.', 'Feb.', 'Mar.', 'Apr.', 'Jun.', 'Jul.',
    'Aug.', 'Sep.', 'Sept.', 'Oct.', 'Nov.', 'Dec.',
    'Mon.', 'Tue.', 'Wed.', 'Thu.', 'Fri.', 'Sat.', 'Sun.',
    'Cust.', 'Spec.', 'e.g.', 'i.e.', 'cf.', 'et al.'
  ]);

  constructor(options: SplitterOptions = {}) {
    this.minLength = options.minLength || 10;
    this.maxLength = options.maxLength || 500;
    this.minFragmentLength = options.minFragmentLength || 20;
    this.mergeThreshold = options.mergeThreshold || 15;
    this.streamBufferSize = options.streamBufferSize || 100;
    this.customAbbreviations = new Set();
    // Precompile abbreviation regexes (except conditional 'No.') for performance
    this.abbreviationRegexes = Array.from(this.defaultAbbreviations)
      .filter(a => a !== 'No.')
      .map(abbr => ({ abbr, regex: new RegExp(abbr.replace(/\./g, '\\.'), 'g') }));
    // Default heading patterns (merge with following sentence)
    this.headingPatterns = options.headingPatterns || [
      /^(Section|Chapter|Article|Exhibit|Annex|Appendix|Schedule)\s+(?:[0-9]+|[IVXLC]+|[A-Z])\.$/i
    ];
  }

  /**
   * Add custom abbreviations to avoid splitting on
   */
  addAbbreviations(abbreviations: string[]): void {
    abbreviations.forEach(abbr => this.customAbbreviations.add(abbr));
  }

  /**
   * Main method to split sentences (for compatibility with tests)
   */
  splitSentences(text: string): string[] {
    return this.split(text);
  }

  /**
   * Split text into sentences with legal abbreviation handling
   */
  split(text: string): string[] {
    if (!text || !text.trim()) {
      return [];
    }

    // Combine default and custom abbreviations
    const allAbbreviations = new Set([...this.defaultAbbreviations, ...this.customAbbreviations]);

    // Protect abbreviations by temporarily replacing periods
    let protectedText = text;
    const replacements: Map<string, string> = new Map();
    let replacementIndex = 0;
    // First handle conditional 'No.' separately
    if (allAbbreviations.has('No.')) {
      const placeholder = `__ABBR_${replacementIndex++}__`;
      replacements.set(placeholder, 'No.');
      protectedText = protectedText.replace(/No\.(?=\s+\d)/g, placeholder);
    }
    // Then handle precompiled regexes
    this.abbreviationRegexes.forEach(({ abbr, regex }) => {
      if (!allAbbreviations.has(abbr)) return; // skip if removed later dynamically
      const placeholder = `__ABBR_${replacementIndex++}__`;
      replacements.set(placeholder, abbr);
      protectedText = protectedText.replace(regex, placeholder);
    });

  // Split on sentence boundaries; include final fragment without terminal punctuation
  const sentences = protectedText.match(/[^.!?]+[.!?]+|[^.!?]+$/g) || [];

    // Restore abbreviations and process sentences
    const processedSentences = sentences.map(sentence => {
      let restored = sentence.trim();
      replacements.forEach((original, placeholder) => {
        restored = restored.replace(new RegExp(placeholder, 'g'), original);
      });
      return restored;
    });

    // Merge heading patterns with the following sentence
    for (let i = 0; i < processedSentences.length - 1; i++) {
      const heading = processedSentences[i].trim();
      if (this.headingPatterns.some(r => r.test(heading))) {
        processedSentences[i + 1] = processedSentences[i] + ' ' + processedSentences[i + 1];
        processedSentences.splice(i, 1);
        i--;
      }
    }

    // Filter and merge fragments
    return this.mergeFragments(processedSentences);
  }

  /**
   * Merge short fragments with neighboring sentences
   */
  private mergeFragments(sentences: string[]): string[] {
    const result: string[] = [];
    let i = 0;
    while (i < sentences.length) {
      const current = sentences[i];
      const curLen = current.length;
      const terminal = current.at(-1) || '';

      // Always keep punctuation-ended short questions/exclamations as standalone
      const isForceKeep = /[!?]/.test(terminal);

      if (curLen >= this.minFragmentLength || isForceKeep) {
        result.push(current);
        i++;
        continue;
      }

      // Collect run of consecutive short fragments (ending with '.')
      const runStart = i;
      let runEnd = i;
      while (runEnd + 1 < sentences.length && sentences[runEnd + 1].length < this.minFragmentLength && /\.$/.test(sentences[runEnd + 1].trim())) {
        runEnd++;
      }

      const runLength = runEnd - runStart + 1;
      const nextSentence = sentences[runEnd + 1];
      const hasFollowingLong = !!nextSentence && nextSentence.length >= this.minFragmentLength;

      if (runLength === 1 && hasFollowingLong) {
        // Merge the single short fragment with the following long sentence
        sentences[runEnd + 1] = sentences[runStart] + ' ' + nextSentence;
        i = runEnd + 1; // Skip to merged long sentence next iteration to be processed
        continue;
      }

      if (runLength > 1 && hasFollowingLong) {
        // Drop all short fragments before long sentence
        i = runEnd + 1; // advance to the long sentence; do not add fragments
        continue;
      }

      // No following long sentence to merge with; keep fragments that satisfy minLength
      for (let j = runStart; j <= runEnd; j++) {
        if (sentences[j].length >= this.minLength) {
          result.push(sentences[j]);
        }
      }
      i = runEnd + 1;
    }

    return result.filter(s => s.length <= this.maxLength);
  }

  /**
   * Split text into chunks of approximately equal size
   */
  splitIntoChunks(text: string, chunkSize: number = 1000): string[] {
    const chunks: string[] = [];
    const sentences = this.split(text);
    let currentChunk = '';

    for (const sentence of sentences) {
      if ((currentChunk + sentence).length > chunkSize && currentChunk) {
        chunks.push(currentChunk.trim());
        currentChunk = sentence;
      } else {
        currentChunk += (currentChunk ? ' ' : '') + sentence;
      }
    }

    if (currentChunk) {
      chunks.push(currentChunk.trim());
    }

    return chunks;
  }

  /**
   * Process streaming chunk of text
   */
  processStreamingChunk(chunk: string, context: StreamingContext): string[] {
    context.buffer += chunk;
    const sentences: string[] = [];

    // Look for complete sentences
    const matches = context.buffer.match(/[^.!?]+[.!?]+/g);
    if (matches) {
      // Process complete sentences
      matches.forEach(match => {
        const processed = this.split(match.trim());
        sentences.push(...processed);
      });

      // Keep the remainder in buffer
      const lastMatch = matches[matches.length - 1];
      const lastIndex = context.buffer.lastIndexOf(lastMatch) + lastMatch.length;
      context.buffer = context.buffer.substring(lastIndex);

      // If buffer grows too large without terminal punctuation (unlikely due to regex),
      // we safeguard by returning early once buffer exceeds streamBufferSize.
      if (context.buffer.length > this.streamBufferSize) {
        // Emit buffer as a sentence fragment (will be validated / merged later)
        const frag = context.buffer.trim();
        if (frag.length >= this.minLength) {
          sentences.push(frag);
          context.processedSentences.push(frag);
          context.buffer = '';
        }
      }
    }

    // Store processed sentences
    context.processedSentences.push(...sentences);
    return sentences;
  }

  /**
   * Finalize streaming by processing remaining buffer
   */
  finalizeStreaming(context: StreamingContext): string[] {
    const remaining = context.buffer.trim();
    if (remaining) {
      let sentences = this.split(remaining);
      // If no sentences were detected (e.g., trailing fragment without terminal punctuation),
      // treat the remaining buffer as a single sentence so streaming consumers don't lose data.
      if (sentences.length === 0 && remaining.length >= this.minLength) {
        sentences = [remaining];
      }
      context.processedSentences.push(...sentences);
      return sentences;
    }
    return [];
  }
}

export default EnhancedSentenceSplitter;

/**
 * Convenience function for splitting sentences
 */
export function splitSentencesEnhanced(text: string): string[] {
  const splitter = new EnhancedSentenceSplitter();
  return splitter.split(text);
}

/**
 * Create a streaming splitter with context
 */
export function createStreamingSplitter(options: SplitterOptions = {}): {
  splitter: EnhancedSentenceSplitter;
  context: StreamingContext;
} {
  const splitter = new EnhancedSentenceSplitter(options);
  const context: StreamingContext = {
    buffer: '',
    processedSentences: []
  };
  return { splitter, context };
}
