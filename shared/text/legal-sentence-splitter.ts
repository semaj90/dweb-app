// Centralized Legal Sentence Splitter Factory
// Provides a shared instance & configuration for legal document processing across services.

import { EnhancedSentenceSplitter } from './enhanced-sentence-splitter';

export interface LegalSentenceSplitterConfig {
  minLength?: number;
  maxLength?: number;
  minFragmentLength?: number;
  mergeThreshold?: number;
  streamBufferSize?: number;
  headingPatterns?: RegExp[];
  extraAbbreviations?: string[];
}

const defaultConfig: Required<Omit<LegalSentenceSplitterConfig, 'headingPatterns' | 'extraAbbreviations'>> = {
  minLength: 10,
  maxLength: 600,
  minFragmentLength: 20,
  mergeThreshold: 15,
  streamBufferSize: 120,
};

const defaultHeadingPatterns: RegExp[] = [
  /^(Section|Chapter|Article|Exhibit|Annex|Appendix|Schedule)\s+(?:[0-9]+|[IVXLC]+|[A-Z])\.$/i,
  /^Recital\s+[A-Z]\.$/i,
  /^WHEREAS,$/,
];

const defaultExtraAbbreviations = ['Sec.', 'Art.', 'Exh.', 'Sched.', 'Recital'];

// Lazy singleton instance (can be reset for tests)
let sharedInstance: unknown | null = null;

export function createLegalSentenceSplitter(config: LegalSentenceSplitterConfig = {}) {
  const { headingPatterns, extraAbbreviations, ...rest } = config;
  const splitter = new EnhancedSentenceSplitter({
    ...defaultConfig,
    ...rest,
    headingPatterns: headingPatterns || defaultHeadingPatterns,
  });
  if (extraAbbreviations && extraAbbreviations.length) {
    splitter.addAbbreviations([...defaultExtraAbbreviations, ...extraAbbreviations]);
  } else {
    splitter.addAbbreviations(defaultExtraAbbreviations);
  }
  return splitter;
}

export function getSharedLegalSentenceSplitter() {
  if (!sharedInstance) {
    sharedInstance = createLegalSentenceSplitter();
  }
  return sharedInstance;
}

export function resetSharedLegalSentenceSplitter() {
  sharedInstance = null;
}
