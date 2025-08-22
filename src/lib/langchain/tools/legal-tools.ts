// Legal AI Tools for LangChain Integration
// Specialized tools for legal document analysis, search, and processing

import type { BaseTool } from '../langchain-manager';
import { webgpuRAGService } from '../../webgpu/webgpu-rag-service';
import { multiProtocolRouter, routerHelpers } from '../../services/multi-protocol-router';

/**
 * Legal Document Search Tool
 * Performs semantic search across legal documents using WebGPU acceleration
 */
export class LegalSearchTool implements BaseTool {
  name = 'legal_search';
  description = 'Search through legal documents, cases, and statutes for relevant information';
  schema = {
    type: 'object',
    properties: {
      query: { type: 'string', description: 'Search query for legal documents' },
      filters: {
        type: 'object',
        properties: {
          documentTypes: { type: 'array', items: { type: 'string' } },
          jurisdiction: { type: 'string' },
          dateRange: { type: 'object' },
          caseId: { type: 'string' }
        }
      },
      options: {
        type: 'object',
        properties: {
          topK: { type: 'number', default: 5 },
          threshold: { type: 'number', default: 0.7 },
          useGPU: { type: 'boolean', default: true }
        }
      }
    },
    required: ['query']
  };

  async call(input: string, options: unknown = {}): Promise<string> {
    try {
      let query: string;
      let filters: unknown = {};
      let searchOptions: unknown = { topK: 5, threshold: 0.7, useGPU: true };

      // Parse input
      try {
        const parsed = JSON.parse(input);
        query = parsed.query;
        filters = parsed.filters || {};
        searchOptions = { ...searchOptions, ...parsed.options };
      } catch {
        // Treat input as plain text query
        query = input;
      }

      console.log(`🔍 Legal search: "${query}"`);

      // Use WebGPU-accelerated search if available
      if (webgpuRAGService.isReady() && searchOptions.useGPU) {
        const results = await webgpuRAGService.semanticSearch(query, {
          ...searchOptions,
          ...filters
        });

        const formattedResults = results.results.map(result => ({
          title: result.metadata?.title || 'Legal Document',
          snippet: result.text.substring(0, 200) + '...',
          similarity: (result.similarity * 100).toFixed(1) + '%',
          documentType: result.metadata?.type || 'document',
          caseId: result.metadata?.caseId
        }));

        return JSON.stringify({
          query,
          totalResults: results.results.length,
          processingTime: results.processingTime,
          usedGPU: results.usedGPU,
          results: formattedResults
        });
      }

      // Fallback to traditional search
      const searchResult = await routerHelpers.search({
        query,
        filters,
        ...searchOptions
      });

      // Type assertion to ensure searchResult has the expected properties
      const resultObj = searchResult as { total?: number; results?: unknown[] };

      return JSON.stringify({
        query,
        totalResults: resultObj.total || 0,
        results: resultObj.results || [],
        source: 'traditional_search'
      });

    } catch (error) {
      console.error('Legal search tool error:', error);
      return JSON.stringify({
        error: `Search failed: ${error.message}`,
        query: input
      });
    }
  }
}

/**
 * Legal Case Analysis Tool
 * Analyzes legal cases for precedents, holdings, and key facts
 */
export class LegalCaseAnalysisTool implements BaseTool {
  name = 'case_analysis';
  description = 'Analyze legal cases to extract key facts, holdings, precedents, and legal reasoning';
  schema = {
    type: 'object',
    properties: {
      caseText: { type: 'string', description: 'Full text of the legal case' },
      analysisType: {
        type: 'string',
        enum: ['summary', 'precedents', 'facts', 'holding', 'reasoning'],
        default: 'summary'
      },
      jurisdiction: { type: 'string', description: 'Legal jurisdiction' }
    },
    required: ['caseText']
  };

  async call(input: string, options: unknown = {}): Promise<string> {
    try {
      let caseText: string;
      let analysisType: string = 'summary';
      let jurisdiction: string = '';

      // Parse input
      try {
        const parsed = JSON.parse(input);
        caseText = parsed.caseText;
        analysisType = parsed.analysisType || 'summary';
        jurisdiction = parsed.jurisdiction || '';
      } catch {
        caseText = input;
      }

      console.log(`⚖️ Analyzing case (${analysisType})`);

      // Extract key information based on analysis type
      const analysis = await this.performCaseAnalysis(caseText, analysisType, jurisdiction);

      return JSON.stringify({
        analysisType,
        jurisdiction,
        caseLength: caseText.length,
        analysis,
        extractedAt: new Date().toISOString()
      });

    } catch (error) {
      console.error('Case analysis tool error:', error);
      return JSON.stringify({
        error: `Case analysis failed: ${error.message}`,
        input: input.substring(0, 100) + '...'
      });
    }
  }

  private async performCaseAnalysis(caseText: string, analysisType: string, jurisdiction: string): Promise<any> {
    // Simplified analysis - in production, this would use advanced NLP/LLM processing
    const text = caseText.toLowerCase();

    switch (analysisType) {
      case 'summary':
        return {
          summary: this.extractSummary(caseText),
          keyTerms: this.extractKeyTerms(text),
          parties: this.extractParties(text),
          jurisdiction
        };

      case 'precedents':
        return {
          citedCases: this.extractCitations(text),
          precedents: this.extractPrecedents(text),
          applicableLaw: this.extractApplicableLaw(text)
        };

      case 'facts':
        return {
          keyFacts: this.extractKeyFacts(text),
          timeline: this.extractTimeline(text),
          evidence: this.extractEvidence(text)
        };

      case 'holding':
        return {
          holding: this.extractHolding(text),
          reasoning: this.extractReasoning(text),
          outcome: this.extractOutcome(text)
        };

      case 'reasoning':
        return {
          legalReasoning: this.extractLegalReasoning(text),
          analysis: this.extractAnalysis(text),
          rationale: this.extractRationale(text)
        };

      default:
        return { error: `Unknown analysis type: ${analysisType}` };
    }
  }

  private extractSummary(text: string): string {
    // Extract first significant paragraph as summary
    const sentences = text.split('.').slice(0, 3);
    return sentences.join('.') + '.';
  }

  private extractKeyTerms(text: string): string[] {
    const legalTerms = [
      'negligence', 'liability', 'damages', 'breach', 'contract', 'tort',
      'defendant', 'plaintiff', 'evidence', 'burden of proof', 'precedent',
      'statute', 'regulation', 'due process', 'jurisdiction', 'appeal'
    ];

    return legalTerms.filter(term => text.includes(term));
  }

  private extractParties(text: string): unknown {
    // Simple regex patterns to identify parties
    const plaintiffMatch = text.match(/plaintiff[s]?:?\s+([^,.\n]+)/i);
    const defendantMatch = text.match(/defendant[s]?:?\s+([^,.\n]+)/i);

    return {
      plaintiff: plaintiffMatch?.[1]?.trim() || 'Not identified',
      defendant: defendantMatch?.[1]?.trim() || 'Not identified'
    };
  }

  private extractCitations(text: string): string[] {
    // Simple pattern for case citations
    const citationPattern = /\d+\s+[\w\s]+\s+\d+/g;
    return text.match(citationPattern) || [];
  }

  private extractPrecedents(text: string): string[] {
    const precedentKeywords = ['established in', 'pursuant to', 'following', 'in accordance with'];
    const precedents: string[] = [];

    precedentKeywords.forEach(keyword => {
      const regex = new RegExp(`${keyword}\\s+([^.]+)`, 'gi');
      const matches = text.match(regex);
      if (matches) {
        precedents.push(...matches);
      }
    });

    return precedents;
  }

  private extractApplicableLaw(text: string): string[] {
    const lawPatterns = [
      /section\s+\d+/gi,
      /title\s+\d+/gi,
      /code\s+section\s+\d+/gi,
      /statute\s+\d+/gi
    ];

    const laws: string[] = [];
    lawPatterns.forEach(pattern => {
      const matches = text.match(pattern);
      if (matches) {
        laws.push(...matches);
      }
    });

    return laws;
  }

  private extractKeyFacts(text: string): string[] {
    // Look for fact patterns
    const factIndicators = ['the facts show', 'it is undisputed', 'the evidence reveals', 'plaintiff alleges'];
    const facts: string[] = [];

    factIndicators.forEach(indicator => {
      const index = text.indexOf(indicator);
      if (index !== -1) {
        const sentence = text.substring(index, text.indexOf('.', index) + 1);
        facts.push(sentence);
      }
    });

    return facts;
  }

  private extractTimeline(text: string): unknown[] {
    // Simple date extraction
    const datePattern = /\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}\b/gi;
    const dates = text.match(datePattern) || [];

    return dates.map(date => ({
      date,
      context: 'Date mentioned in case'
    }));
  }

  private extractEvidence(text: string): string[] {
    const evidenceKeywords = ['exhibit', 'testimony', 'witness', 'document', 'record'];
    const evidence: string[] = [];

    evidenceKeywords.forEach(keyword => {
      if (text.includes(keyword)) {
        evidence.push(`${keyword} mentioned`);
      }
    });

    return evidence;
  }

  private extractHolding(text: string): string {
    // Look for holding indicators
    const holdingIndicators = ['we hold', 'the court holds', 'it is held', 'holding'];

    for (const indicator of holdingIndicators) {
      const index = text.indexOf(indicator);
      if (index !== -1) {
        const sentence = text.substring(index, text.indexOf('.', index) + 1);
        return sentence;
      }
    }

    return 'Holding not clearly identified';
  }

  private extractReasoning(text: string): string {
    // Look for reasoning indicators
    const reasoningIndicators = ['because', 'therefore', 'consequently', 'as a result'];

    for (const indicator of reasoningIndicators) {
      const index = text.indexOf(indicator);
      if (index !== -1) {
        const sentence = text.substring(index, text.indexOf('.', index) + 1);
        return sentence;
      }
    }

    return 'Reasoning not clearly identified';
  }

  private extractOutcome(text: string): string {
    const outcomeKeywords = ['granted', 'denied', 'reversed', 'affirmed', 'dismissed'];

    for (const keyword of outcomeKeywords) {
      if (text.includes(keyword)) {
        return `Case ${keyword}`;
      }
    }

    return 'Outcome not clearly identified';
  }

  private extractLegalReasoning(text: string): string {
    return 'Legal reasoning analysis would be performed here';
  }

  private extractAnalysis(text: string): string {
    return 'Detailed legal analysis would be performed here';
  }

  private extractRationale(text: string): string {
    return 'Judicial rationale extraction would be performed here';
  }
}

/**
 * Legal Document Drafting Tool
 * Assists in creating legal documents based on templates and requirements
 */
export class LegalDraftingTool implements BaseTool {
  name = 'legal_drafting';
  description = 'Generate legal document templates and assist with legal document drafting';
  schema = {
    type: 'object',
    properties: {
      documentType: {
        type: 'string',
        enum: ['contract', 'motion', 'brief', 'agreement', 'notice', 'pleading'],
        description: 'Type of legal document to draft'
      },
      parties: {
        type: 'object',
        properties: {
          party1: { type: 'string' },
          party2: { type: 'string' },
          additionalParties: { type: 'array', items: { type: 'string' } }
        }
      },
      terms: { type: 'object', description: 'Specific terms and conditions' },
      jurisdiction: { type: 'string', description: 'Legal jurisdiction' },
      purpose: { type: 'string', description: 'Purpose of the document' }
    },
    required: ['documentType', 'purpose']
  };

  async call(input: string, options: unknown = {}): Promise<string> {
    try {
      let documentType: string;
      let parties: unknown = {};
      let terms: unknown = {};
      let jurisdiction: string = '';
      let purpose: string = '';

      // Parse input
      try {
        const parsed = JSON.parse(input);
        documentType = parsed.documentType;
        parties = parsed.parties || {};
        terms = parsed.terms || {};
        jurisdiction = parsed.jurisdiction || '';
        purpose = parsed.purpose;
      } catch {
        // Try to extract from plain text
        documentType = this.extractDocumentType(input);
        purpose = input;
      }

      console.log(`📝 Drafting ${documentType} document`);

      const template = await this.generateDocumentTemplate(documentType, parties, terms, jurisdiction, purpose);

      return JSON.stringify({
        documentType,
        jurisdiction,
        parties,
        template,
        clauses: this.getSuggestedClauses(documentType),
        warnings: this.getWarnings(documentType),
        generatedAt: new Date().toISOString()
      });

    } catch (error) {
      console.error('Legal drafting tool error:', error);
      return JSON.stringify({
        error: `Document drafting failed: ${error.message}`,
        input: input.substring(0, 100) + '...'
      });
    }
  }

  private extractDocumentType(input: string): string {
    const text = input.toLowerCase();

    if (text.includes('contract') || text.includes('agreement')) return 'contract';
    if (text.includes('motion')) return 'motion';
    if (text.includes('brief')) return 'brief';
    if (text.includes('notice')) return 'notice';
    if (text.includes('pleading')) return 'pleading';

    return 'contract'; // default
  }

  private async generateDocumentTemplate(
    documentType: string,
    parties: unknown,
    terms: unknown,
    jurisdiction: string,
    purpose: string
  ): Promise<string> {
    const templates = {
      contract: this.generateContractTemplate(parties, terms, purpose),
      motion: this.generateMotionTemplate(parties, terms, purpose),
      brief: this.generateBriefTemplate(parties, terms, purpose),
      agreement: this.generateAgreementTemplate(parties, terms, purpose),
      notice: this.generateNoticeTemplate(parties, terms, purpose),
      pleading: this.generatePleadingTemplate(parties, terms, purpose)
    };

    return templates[documentType as keyof typeof templates] || templates.contract;
  }

  private generateContractTemplate(parties: unknown, terms: unknown, purpose: string): string {
    return `CONTRACT AGREEMENT

This Agreement is entered into on [DATE] between:

Party 1: ${parties.party1 || '[PARTY 1 NAME]'}
Party 2: ${parties.party2 || '[PARTY 2 NAME]'}

PURPOSE: ${purpose}

TERMS AND CONDITIONS:

1. SCOPE OF WORK
${terms.scope || '[DEFINE SCOPE OF WORK]'}

2. CONSIDERATION
${terms.payment || '[PAYMENT TERMS]'}

3. TERM
${terms.duration || '[CONTRACT DURATION]'}

4. TERMINATION
Either party may terminate this agreement with [NOTICE PERIOD] written notice.

5. GOVERNING LAW
This agreement shall be governed by the laws of [JURISDICTION].

6. DISPUTE RESOLUTION
Any disputes shall be resolved through [DISPUTE RESOLUTION METHOD].

IN WITNESS WHEREOF, the parties have executed this Agreement.

_____________________          _____________________
${parties.party1 || '[PARTY 1]'}                    ${parties.party2 || '[PARTY 2]'}

Date: _______________          Date: _______________`;
  }

  private generateMotionTemplate(parties: unknown, terms: unknown, purpose: string): string {
    return `MOTION TO [SPECIFY RELIEF SOUGHT]

TO THE HONORABLE COURT:

NOW COMES ${parties.party1 || '[MOVANT]'}, by and through undersigned counsel, and respectfully moves this Court for [RELIEF SOUGHT] and in support thereof states:

STATEMENT OF FACTS

[FACTUAL BACKGROUND]

ARGUMENT

I. [LEGAL ARGUMENT HEADING]

${purpose}

[LEGAL ANALYSIS AND AUTHORITIES]

CONCLUSION

WHEREFORE, ${parties.party1 || '[MOVANT]'} respectfully requests that this Court [GRANT THE RELIEF REQUESTED].

Respectfully submitted,

_____________________
[ATTORNEY NAME]
[ATTORNEY INFORMATION]`;
  }

  private generateBriefTemplate(parties: unknown, terms: unknown, purpose: string): string {
    return `LEGAL BRIEF

IN THE MATTER OF: ${purpose}

TABLE OF CONTENTS
I. STATEMENT OF THE CASE
II. STATEMENT OF FACTS
III. ARGUMENT
IV. CONCLUSION

I. STATEMENT OF THE CASE

[PROCEDURAL HISTORY AND NATURE OF THE CASE]

II. STATEMENT OF FACTS

[RELEVANT FACTS]

III. ARGUMENT

A. [FIRST LEGAL ARGUMENT]

[ANALYSIS]

B. [SECOND LEGAL ARGUMENT]

[ANALYSIS]

IV. CONCLUSION

For the foregoing reasons, [REQUESTED RELIEF].

Respectfully submitted,

_____________________
[ATTORNEY INFORMATION]`;
  }

  private generateAgreementTemplate(parties: unknown, terms: unknown, purpose: string): string {
    return this.generateContractTemplate(parties, terms, purpose); // Similar to contract
  }

  private generateNoticeTemplate(parties: unknown, terms: unknown, purpose: string): string {
    return `NOTICE TO ${parties.party2 || '[RECIPIENT]'}

TO: ${parties.party2 || '[RECIPIENT NAME AND ADDRESS]'}

YOU ARE HEREBY NOTIFIED that:

${purpose}

[SPECIFIC NOTICE CONTENT]

This notice is served upon you as required by law.

Date: _______________

_____________________
${parties.party1 || '[SENDER]'}`;
  }

  private generatePleadingTemplate(parties: unknown, terms: unknown, purpose: string): string {
    return `[COURT HEADER]

${parties.party1 || '[PLAINTIFF]'},
                    Plaintiff,
v.                                          Case No. [CASE NUMBER]

${parties.party2 || '[DEFENDANT]'},
                    Defendant.

[PLEADING TYPE]

NOW COMES ${parties.party1 || '[PARTY]'} and for [HIS/HER] [PLEADING TYPE] against ${parties.party2 || '[OPPOSING PARTY]'} states:

COUNT I
[CAUSE OF ACTION]

1. [FACTUAL ALLEGATIONS]

2. [ADDITIONAL ALLEGATIONS]

WHEREFORE, ${parties.party1 || '[PARTY]'} demands judgment against ${parties.party2 || '[OPPOSING PARTY]'} for [RELIEF REQUESTED].

_____________________
[ATTORNEY SIGNATURE]`;
  }

  private getSuggestedClauses(documentType: string): string[] {
    const clauses: Record<string, string[]> = {
      contract: [
        'Force Majeure Clause',
        'Confidentiality Clause',
        'Indemnification Clause',
        'Limitation of Liability',
        'Intellectual Property Rights'
      ],
      motion: [
        'Statement of Relief Sought',
        'Legal Standard',
        'Factual Support',
        'Legal Authority'
      ],
      brief: [
        'Issue Statement',
        'Standard of Review',
        'Argument Headings',
        'Conclusion'
      ]
    };

    return clauses[documentType] || [];
  }

  private getWarnings(documentType: string): string[] {
    return [
      'This is a template and should be reviewed by qualified legal counsel',
      'Customize all bracketed placeholders with specific information',
      'Ensure compliance with local jurisdiction requirements',
      'Consider additional clauses specific to your situation'
    ];
  }
}

/**
 * Legal Citation Checker Tool
 * Validates and formats legal citations according to standard formats
 */
export class LegalCitationTool implements BaseTool {
  name = 'citation_checker';
  description = 'Check, validate, and format legal citations according to standard citation formats';
  schema = {
    type: 'object',
    properties: {
      citations: {
        type: 'array',
        items: { type: 'string' },
        description: 'List of citations to check'
      },
      format: {
        type: 'string',
        enum: ['bluebook', 'alwd', 'chicago', 'mla'],
        default: 'bluebook'
      }
    },
    required: ['citations']
  };

  async call(input: string, options: unknown = {}): Promise<string> {
    try {
      let citations: string[];
      let format: string = 'bluebook';

      // Parse input
      try {
        const parsed = JSON.parse(input);
        citations = parsed.citations;
        format = parsed.format || 'bluebook';
      } catch {
        // Treat input as single citation
        citations = [input];
      }

      console.log(`📚 Checking ${citations.length} citations in ${format} format`);

      const checkedCitations = citations.map(citation => this.checkCitation(citation, format));

      return JSON.stringify({
        format,
        totalCitations: citations.length,
        results: checkedCitations,
        checkedAt: new Date().toISOString()
      });

    } catch (error) {
      console.error('Citation checker tool error:', error);
      return JSON.stringify({
        error: `Citation checking failed: ${error.message}`,
        input: input.substring(0, 100) + '...'
      });
    }
  }

  private checkCitation(citation: string, format: string): unknown {
    // Simplified citation checking - in production, this would use sophisticated parsing
    const result = {
      original: citation,
      formatted: citation,
      isValid: false,
      errors: [] as string[],
      suggestions: [] as string[],
      type: 'unknown'
    };

    // Detect citation type
    result.type = this.detectCitationType(citation);

    // Basic validation
    const validation = this.validateCitation(citation, result.type, format);
    result.isValid = validation.isValid;
    result.errors = validation.errors;
    result.suggestions = validation.suggestions;

    // Format citation
    result.formatted = this.formatCitation(citation, result.type, format);

    return result;
  }

  private detectCitationType(citation: string): string {
    const text = citation.toLowerCase();

    if (text.includes('u.s.') || text.includes('supreme court')) return 'case';
    if (text.includes('f.2d') || text.includes('f.3d')) return 'federal_case';
    if (text.includes('§') || text.includes('section')) return 'statute';
    if (text.includes('vol.') || text.includes('law review')) return 'law_review';
    if (text.includes('restatement')) return 'restatement';

    return 'case'; // default assumption
  }

  private validateCitation(citation: string, type: string, format: string): unknown {
    const result = { isValid: true, errors: [], suggestions: [] };

    // Basic validation rules
    if (citation.length < 10) {
      result.isValid = false;
      result.errors.push('Citation appears too short');
    }

    if (!citation.includes(' ')) {
      result.isValid = false;
      result.errors.push('Citation should contain spaces');
    }

    // Type-specific validation
    switch (type) {
      case 'case':
        if (!/\d+/.test(citation)) {
          result.errors.push('Case citation should include page or volume numbers');
          result.isValid = false;
        }
        break;

      case 'statute':
        if (!citation.includes('§') && !citation.includes('section')) {
          result.suggestions.push('Consider using § symbol for statute citations');
        }
        break;
    }

    return result;
  }

  private formatCitation(citation: string, type: string, format: string): string {
    // Simplified formatting - in production, this would be much more sophisticated
    switch (format) {
      case 'bluebook':
        return this.formatBluebook(citation, type);
      case 'alwd':
        return this.formatALWD(citation, type);
      default:
        return citation;
    }
  }

  private formatBluebook(citation: string, type: string): string {
    // Basic Bluebook formatting
    return citation.replace(/\s+/g, ' ').trim();
  }

  private formatALWD(citation: string, type: string): string {
    // Basic ALWD formatting
    return citation.replace(/\s+/g, ' ').trim();
  }
}

// Export all tools as a collection
export const legalTools: BaseTool[] = [
  new LegalSearchTool(),
  new LegalCaseAnalysisTool(),
  new LegalDraftingTool(),
  new LegalCitationTool()
];

export default legalTools;