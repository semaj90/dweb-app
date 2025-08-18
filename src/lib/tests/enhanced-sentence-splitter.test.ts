// Enhanced Sentence Splitter Tests
// Test suite for the enhanced sentence splitter service

import { describe, it, expect } from 'vitest';
import { 
  EnhancedSentenceSplitter, 
  createStreamingSplitter 
} from '../services/enhanced-sentence-splitter';

describe('EnhancedSentenceSplitter', () => {
  it('should split basic sentences correctly', () => {
    const splitter = new EnhancedSentenceSplitter({
      minLength: 5,
      maxLength: 100
    });

    const text = 'This is the first sentence. This is the second sentence.';
    const result = splitter.split(text);

    expect(result).toHaveLength(2);
    expect(result[0]).toBe('This is the first sentence.');
    expect(result[1]).toBe('This is the second sentence.');
  });

  it('should respect minimum length requirements', () => {
    const splitter = new EnhancedSentenceSplitter({
      minLength: 25,
      maxLength: 200
    });

    const text = 'Short. This is a much longer sentence that meets the minimum length requirement.';
    const result = splitter.split(text);

    expect(result).toHaveLength(1);
    expect(result[0]).toBe('This is a much longer sentence that meets the minimum length requirement.');
  });

  it('should respect maximum length requirements', () => {
    const splitter = new EnhancedSentenceSplitter({
      minLength: 10,
      maxLength: 30
    });

    const text = 'This sentence is exactly the right length. This is a very long sentence that exceeds the maximum length requirement and should be filtered out.';
    const result = splitter.split(text);

    expect(result).toHaveLength(1);
    expect(result[0]).toBe('This sentence is exactly the right length.');
  });

  it('should handle legal abbreviations correctly', () => {
    const splitter = new EnhancedSentenceSplitter();

    const text = 'The case Smith v. Jones was decided in the U.S. Supreme Court. The ruling was unanimous.';
    const result = splitter.split(text);

    expect(result).toHaveLength(2);
    expect(result[0]).toBe('The case Smith v. Jones was decided in the U.S. Supreme Court.');
    expect(result[1]).toBe('The ruling was unanimous.');
  });

  it('should handle question marks and exclamation points', () => {
    const splitter = new EnhancedSentenceSplitter();

    const text = 'What is the legal precedent? The court ruled decisively! This establishes new precedent.';
    const result = splitter.split(text);

    expect(result).toHaveLength(3);
    expect(result[0]).toBe('What is the legal precedent?');
    expect(result[1]).toBe('The court ruled decisively!');
    expect(result[2]).toBe('This establishes new precedent.');
  });

  it('should handle quoted text correctly', () => {
    const splitter = new EnhancedSentenceSplitter();

    const text = 'The witness stated "I saw the defendant. He was running." This was crucial testimony.';
    const result = splitter.split(text);

    expect(result).toHaveLength(2);
    expect(result[0]).toBe('The witness stated "I saw the defendant. He was running."');
    expect(result[1]).toBe('This was crucial testimony.');
  });

  it('should allow adding custom abbreviations', () => {
    const splitter = new EnhancedSentenceSplitter();
    
    splitter.addAbbreviations(['Cust.', 'Spec.']);

    const text = 'The Cust. Agreement specifies terms. The Spec. Document outlines requirements.';
    const result = splitter.split(text);

    expect(result).toHaveLength(2);
    expect(result[0]).toBe('The Cust. Agreement specifies terms.');
    expect(result[1]).toBe('The Spec. Document outlines requirements.');
  });

  it('should handle decimal numbers without splitting', () => {
    const splitter = new EnhancedSentenceSplitter();

    const text = 'The damages awarded were $1,234.56 in total. This was considered fair compensation.';
    const result = splitter.split(text);

    expect(result).toHaveLength(2);
    expect(result[0]).toBe('The damages awarded were $1,234.56 in total.');
    expect(result[1]).toBe('This was considered fair compensation.');
  });

  it('should handle empty and invalid input', () => {
    const splitter = new EnhancedSentenceSplitter();

    expect(splitter.split('')).toEqual([]);
    expect(splitter.split('   ')).toEqual([]);
    expect(splitter.split(null as any)).toEqual([]);
    expect(splitter.split(undefined as any)).toEqual([]);
  });

  describe('createStreamingSplitter factory function', () => {
    it('should create a working splitter instance', () => {
      const splitter = createStreamingSplitter({
        minLength: 15,
        maxLength: 100
      });

      const text = 'This is a test sentence. This is another test sentence.';
      const result = splitter.split(text);

      expect(result).toHaveLength(2);
    });

    it('should work with default options', () => {
      const splitter = createStreamingSplitter();

      const text = 'This is a test sentence. This is another test sentence.';
      const result = splitter.split(text);

      expect(result).toHaveLength(2);
    });
  });

  describe('EnhancedSentenceSplitter.create static method', () => {
    it('should create a working splitter instance', () => {
      const splitter = EnhancedSentenceSplitter.create({
        minLength: 15,
        maxLength: 100
      });

      const text = 'This is a test sentence. This is another test sentence.';
      const result = splitter.split(text);

      expect(result).toHaveLength(2);
    });
  });

  describe('Complex legal text handling', () => {
    it('should handle complex legal citations', () => {
      const splitter = new EnhancedSentenceSplitter();

      const text = 'In Brown v. Board of Education, 347 U.S. 483 (1954), the Court held that separate educational facilities are inherently unequal. This landmark decision overturned Plessy v. Ferguson, 163 U.S. 537 (1896).';
      const result = splitter.split(text);

      expect(result).toHaveLength(2);
      expect(result[0]).toContain('347 U.S. 483 (1954)');
      expect(result[1]).toContain('163 U.S. 537 (1896)');
    });

    it('should handle numbered paragraphs and sections', () => {
      const splitter = new EnhancedSentenceSplitter();

      const text = 'Pursuant to Section 1.2.3 of the Agreement, the parties agree to arbitration. Section 2.1 outlines the dispute resolution process.';
      const result = splitter.split(text);

      expect(result).toHaveLength(2);
      expect(result[0]).toContain('Section 1.2.3');
      expect(result[1]).toContain('Section 2.1');
    });
  });
});
