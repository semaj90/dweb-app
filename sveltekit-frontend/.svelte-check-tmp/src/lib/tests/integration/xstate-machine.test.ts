/**
 * XState Legal Case Machine Integration Tests
 * Tests XState machine flows, selectors, and SSR/hydration compatibility
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { createActor } from 'xstate';
import { legalCaseMachine } from '../../machines/legalCaseMachine';
import type { LegalCaseContext, LegalCaseEvent } from '../../machines/legalCaseMachine';

// TODO: Import from proper selectors file once created
// Mock selectors for now to fix compilation
const legalCaseSelectors = {
  isLoading: (snapshot: any) => snapshot.matches('loading'),
  hasError: (snapshot: any) => snapshot.context.error !== null,
  getCurrentCase: (snapshot: any) => snapshot.context.case,
  getEvidence: (snapshot: any) => snapshot.context.evidence || [],
  canStartAIAnalysis: (snapshot: any) => !snapshot.context.isAnalyzing,
  getAISummary: (snapshot: any) => snapshot.context.aiSummary,
  getActiveTab: (snapshot: any) => snapshot.context.activeTab || 'overview',
  getWorkflowStage: (snapshot: any) => snapshot.context.workflowStage || 'initial',
  getStats: (snapshot: any) => snapshot.context.metrics,
  getSimilarCases: (snapshot: any) => snapshot.context.similarCases || []
};

// Mock external services
vi.mock('$lib/services/ai-summarization-service.js', () => ({
  aiSummarizationService: {
    summarizeCase: vi.fn().mockResolvedValue('AI generated summary'),
    analyzeEvidence: vi.fn().mockResolvedValue({ confidence: 0.85, insights: [] })
  }
}));

vi.mock('$lib/services/vector-search-service.js', () => ({
  vectorSearchService: {
    findSimilarCases: vi.fn().mockResolvedValue([
      { id: '123', title: 'Similar Case 1', similarity: 0.89 },
      { id: '456', title: 'Similar Case 2', similarity: 0.76 }
    ])
  }
}));

// Mock fetch for API calls
global.fetch = vi.fn();

describe('XState Legal Case Machine', () => {
  let actor: ReturnType<typeof createActor>;
  
  beforeEach(() => {
    vi.clearAllMocks();
    actor = createActor(legalCaseMachine);
  });

  afterEach(() => {
    actor?.stop();
  });

  describe('Initial State', () => {
    it('should start in idle state', () => {
      actor.start();
      expect(actor.getSnapshot().value).toBe('idle');
    });

    it('should have correct initial context', () => {
      actor.start();
      const context = actor.getSnapshot().context;
      
      expect(context.case).toBeNull();
      expect(context.caseId).toBeNull();
      expect(context.evidence).toEqual([]);
      expect(context.isLoading).toBe(false);
      expect(context.error).toBeNull();
      expect(context.activeTab).toBe('overview');
      expect(context.workflowStage).toBe('investigation');
    });
  });

  describe('Case Loading Flow', () => {
    it('should transition to loading state when LOAD_CASE event is sent', () => {
      actor.start();
      
      actor.send({ type: 'LOAD_CASE', caseId: 'test-case-123' });
      
      const snapshot = actor.getSnapshot();
      expect(snapshot.value).toMatch(/loading/);
      expect(legalCaseSelectors.isLoading(snapshot)).toBe(true);
    });

    it('should successfully load case data', async () => {
      const mockCaseData = {
        id: 'test-case-123',
        title: 'Test Case',
        description: 'Test Description',
        caseNumber: 'TC-001',
        status: 'active'
      };

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockCaseData)
      });

      actor.start();
      actor.send({ type: 'LOAD_CASE', caseId: 'test-case-123' });

      // Wait for async operations
      await new Promise(resolve => setTimeout(resolve, 100));

      const snapshot = actor.getSnapshot();
      expect(legalCaseSelectors.getCurrentCase(snapshot)).toEqual(mockCaseData);
      expect(legalCaseSelectors.hasError(snapshot)).toBe(false);
    });

    it('should handle case loading errors', async () => {
      (global.fetch as any).mockRejectedValueOnce(new Error('Network error'));

      actor.start();
      actor.send({ type: 'LOAD_CASE', caseId: 'invalid-case' });

      // Wait for async operations
      await new Promise(resolve => setTimeout(resolve, 100));

      const snapshot = actor.getSnapshot();
      expect(legalCaseSelectors.hasError(snapshot)).toBe(true);
      expect(snapshot.context.error).toContain('Network error');
    });
  });

  describe('Case Creation Flow', () => {
    it('should create new case with form data', async () => {
      const newCaseData = {
        title: 'New Test Case',
        description: 'New case description',
        caseNumber: 'NTC-001',
        status: 'active' as const
      };

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ id: 'new-case-123', ...newCaseData })
      });

      actor.start();
      
      // Update form data first
      actor.send({ type: 'UPDATE_CASE_FORM', data: newCaseData });
      
      // Create case
      actor.send({ type: 'CREATE_CASE', caseData: newCaseData });

      // Wait for async operations
      await new Promise(resolve => setTimeout(resolve, 100));

      const snapshot = actor.getSnapshot();
      const currentCase = legalCaseSelectors.getCurrentCase(snapshot);
      expect(currentCase?.title).toBe(newCaseData.title);
    });
  });

  describe('Evidence Management', () => {
    it('should add evidence files to upload queue', () => {
      const mockFiles = [
        new File(['content1'], 'evidence1.pdf', { type: 'application/pdf' }),
        new File(['content2'], 'evidence2.jpg', { type: 'image/jpeg' })
      ];

      actor.start();
      actor.send({ type: 'ADD_EVIDENCE', files: mockFiles });

      const snapshot = actor.getSnapshot();
      expect(snapshot.context.uploadQueue).toHaveLength(2);
    });

    it('should select evidence item', () => {
      const mockEvidence = {
        id: 'evidence-123',
        title: 'Test Evidence',
        type: 'document',
        caseId: 'case-123'
      };

      actor.start();
      actor.send({ type: 'SELECT_EVIDENCE', evidence: mockEvidence });

      const snapshot = actor.getSnapshot();
      expect(snapshot.context.selectedEvidence).toEqual(mockEvidence);
    });
  });

  describe('AI Analysis Flow', () => {
    it('should start AI analysis with proper conditions', () => {
      // Set up machine with case data
      actor.start();
      actor.send({ 
        type: 'LOAD_CASE', 
        caseId: 'test-case-123' 
      });

      const snapshot = actor.getSnapshot();
      if (legalCaseSelectors.canStartAIAnalysis(snapshot)) {
        actor.send({ type: 'START_AI_ANALYSIS' });
        
        const updatedSnapshot = actor.getSnapshot();
        expect(updatedSnapshot.value).toMatch(/aiAnalysis/);
      }
    });

    it('should update AI analysis progress', () => {
      actor.start();
      actor.send({ type: 'AI_ANALYSIS_PROGRESS', progress: 50 });

      const snapshot = actor.getSnapshot();
      expect(snapshot.context.aiAnalysisProgress).toBe(50);
    });

    it('should complete AI analysis with summary', () => {
      const mockSummary = 'AI generated case summary';

      actor.start();
      actor.send({ type: 'AI_ANALYSIS_COMPLETE', summary: mockSummary });

      const snapshot = actor.getSnapshot();
      expect(legalCaseSelectors.getAISummary(snapshot)).toBe(mockSummary);
    });
  });

  describe('Navigation and UI State', () => {
    it('should switch tabs correctly', () => {
      actor.start();
      
      actor.send({ type: 'SWITCH_TAB', tab: 'evidence' });
      expect(legalCaseSelectors.getActiveTab(actor.getSnapshot())).toBe('evidence');

      actor.send({ type: 'SWITCH_TAB', tab: 'analysis' });
      expect(legalCaseSelectors.getActiveTab(actor.getSnapshot())).toBe('analysis');
    });

    it('should update workflow stage', () => {
      actor.start();
      
      actor.send({ type: 'SET_WORKFLOW_STAGE', stage: 'analysis' });
      expect(legalCaseSelectors.getWorkflowStage(actor.getSnapshot())).toBe('analysis');
    });
  });

  describe('Search Functionality', () => {
    it('should handle search queries', () => {
      const searchQuery = 'contract violation';

      actor.start();
      actor.send({ type: 'SEARCH', query: searchQuery });

      const snapshot = actor.getSnapshot();
      expect(snapshot.context.searchQuery).toBe(searchQuery);
    });

    it('should clear search results', () => {
      actor.start();
      
      // Set search state
      actor.send({ type: 'SEARCH', query: 'test query' });
      
      // Clear search
      actor.send({ type: 'CLEAR_SEARCH' });

      const snapshot = actor.getSnapshot();
      expect(snapshot.context.searchQuery).toBe('');
      expect(snapshot.context.searchResults).toEqual([]);
    });
  });

  describe('Error Handling', () => {
    it('should handle RETRY event correctly', () => {
      actor.start();
      
      // Simulate error state
      actor.send({ type: 'ERROR', message: 'Test error' });
      
      // Retry
      actor.send({ type: 'RETRY' });

      const snapshot = actor.getSnapshot();
      expect(legalCaseSelectors.hasError(snapshot)).toBe(false);
    });

    it('should dismiss errors', () => {
      actor.start();
      
      // Set error
      actor.send({ type: 'ERROR', message: 'Test error' });
      
      // Dismiss error
      actor.send({ type: 'DISMISS_ERROR' });

      const snapshot = actor.getSnapshot();
      expect(snapshot.context.error).toBeNull();
    });
  });
});

describe('Legal Case Selectors', () => {
  let mockSnapshot: any;

  beforeEach(() => {
    mockSnapshot = {
      context: {
        case: null,
        caseId: null,
        evidence: [],
        selectedEvidence: null,
        aiSummary: null,
        similarCases: [],
        isLoading: false,
        error: null,
        activeTab: 'overview' as const,
        workflowStage: 'investigation' as const,
        nextActions: [],
        stats: {
          totalEvidence: 0,
          processedEvidence: 0,
          averageConfidence: 0,
          processingTime: 0
        }
      },
      value: 'idle'
    };
  });

  it('should return correct loading state', () => {
    mockSnapshot.context.isLoading = true;
    expect(legalCaseSelectors.isLoading(mockSnapshot)).toBe(true);

    mockSnapshot.context.isLoading = false;
    expect(legalCaseSelectors.isLoading(mockSnapshot)).toBe(false);
  });

  it('should detect error state correctly', () => {
    mockSnapshot.context.error = 'Test error';
    expect(legalCaseSelectors.hasError(mockSnapshot)).toBe(true);

    mockSnapshot.context.error = null;
    expect(legalCaseSelectors.hasError(mockSnapshot)).toBe(false);
  });

  it('should return current case data', () => {
    const mockCase = { id: '123', title: 'Test Case' };
    mockSnapshot.context.case = mockCase;
    
    expect(legalCaseSelectors.getCurrentCase(mockSnapshot)).toEqual(mockCase);
  });

  it('should return evidence list', () => {
    const mockEvidence = [
      { id: '1', title: 'Evidence 1' },
      { id: '2', title: 'Evidence 2' }
    ];
    mockSnapshot.context.evidence = mockEvidence;
    
    expect(legalCaseSelectors.getEvidence(mockSnapshot)).toEqual(mockEvidence);
  });

  it('should determine if AI analysis can start', () => {
    // Without case, cannot start
    expect(legalCaseSelectors.canStartAIAnalysis(mockSnapshot)).toBe(false);

    // With case, can start
    mockSnapshot.context.case = { id: '123', title: 'Test Case' };
    expect(legalCaseSelectors.canStartAIAnalysis(mockSnapshot)).toBe(true);
  });

  it('should return stats correctly', () => {
    const mockStats = {
      totalEvidence: 5,
      processedEvidence: 3,
      averageConfidence: 85,
      processingTime: 1250
    };
    mockSnapshot.context.stats = mockStats;
    
    expect(legalCaseSelectors.getStats(mockSnapshot)).toEqual(mockStats);
  });
});

describe('SSR/Hydration Compatibility', () => {
  it('should create machine without browser-only dependencies', () => {
    // Test that machine can be instantiated in server environment
    expect(() => {
      const machine = legalCaseMachine;
      expect(machine).toBeDefined();
    }).not.toThrow();
  });

  it('should have serializable initial context', () => {
    const machine = legalCaseMachine;
    const initialContext = machine.context;
    
    // Test that context can be JSON serialized/deserialized
    expect(() => {
      const serialized = JSON.stringify(initialContext);
      const deserialized = JSON.parse(serialized);
      expect(deserialized).toEqual(initialContext);
    }).not.toThrow();
  });

  it('should handle undefined/null states gracefully', () => {
    const actor = createActor(legalCaseMachine);
    actor.start();
    
    // Test selectors with incomplete state
    const snapshot = actor.getSnapshot();
    
    expect(legalCaseSelectors.getCurrentCase(snapshot)).toBeNull();
    expect(legalCaseSelectors.getEvidence(snapshot)).toEqual([]);
    expect(legalCaseSelectors.getAISummary(snapshot)).toBeNull();
    expect(legalCaseSelectors.getSimilarCases(snapshot)).toEqual([]);
  });
});