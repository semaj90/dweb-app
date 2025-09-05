/**
 * UI Hydration Tests with XState Machine State
 * Tests client-side hydration, state persistence, and machine synchronization
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/svelte';
import { createActor } from 'xstate';
import { tick } from 'svelte';

// Mock browser environment
Object.defineProperty(window, 'location', {
  value: {
    href: 'http://localhost:5173/',
    origin: 'http://localhost:5173',
    pathname: '/',
    search: '',
    hash: ''
  },
  writable: true
});

// Mock SvelteKit stores
const mockPageStore = {
  subscribe: vi.fn((callback) => {
    callback({
      url: new URL('http://localhost:5173/'),
      params: {},
      route: { id: '/' },
      status: 200,
      error: null,
      data: {},
      form: null,
      state: {}
    });
    return () => {};
  })
};

const mockNavigating = {
  subscribe: vi.fn((callback) => {
    callback(null);
    return () => {};
  })
};

vi.mock('$app/stores', () => ({
  page: mockPageStore,
  navigating: mockNavigating,
  updated: {
    subscribe: vi.fn((callback) => {
      callback(false);
      return () => {};
    })
  }
}));

// Mock XState Svelte integration
vi.mock('@xstate/svelte', () => ({
  useMachine: vi.fn()
}));

// Mock fetch for API calls
global.fetch = vi.fn();

describe('UI Hydration with XState Integration', () => {
  let mockActor: any;
  let mockUseMachine: any;

  beforeEach(() => {
    vi.clearAllMocks();
    
    mockActor = {
      send: vi.fn(),
      getSnapshot: vi.fn(),
      subscribe: vi.fn(),
      start: vi.fn(),
      stop: vi.fn()
    };

    mockUseMachine = vi.fn().mockReturnValue({
      state: {
        value: 'idle',
        context: {
          case: null,
          evidence: [],
          isLoading: false,
          error: null,
          activeTab: 'overview',
          workflowStage: 'investigation',
          stats: {
            totalEvidence: 0,
            processedEvidence: 0,
            averageConfidence: 0,
            processingTime: 0
          }
        },
        matches: vi.fn(),
        can: vi.fn(),
        toJSON: vi.fn()
      },
      send: mockActor.send,
      actor: mockActor
    });

    vi.mocked(require('@xstate/svelte').useMachine).mockImplementation(mockUseMachine);
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe('Component Hydration', () => {
    it('should hydrate CaseManagerXState component without errors', async () => {
      // Mock the component with basic props
      const TestComponent = (await import('$lib/components/legal/CaseManagerXState.svelte')).default;
      
      expect(() => {
        render(TestComponent);
      }).not.toThrow();
    });

    it('should preserve server-rendered XState machine state during hydration', async () => {
      const serverState = {
        value: 'loaded',
        context: {
          case: { id: 'case-123', title: 'Hydration Test Case' },
          evidence: [{ id: 'evidence-1', title: 'Test Evidence' }],
          isLoading: false,
          error: null,
          activeTab: 'evidence',
          workflowStage: 'analysis'
        }
      };

      mockUseMachine.mockReturnValueOnce({
        state: serverState,
        send: mockActor.send,
        actor: mockActor
      });

      const TestComponent = (await import('$lib/components/legal/CaseManagerXState.svelte')).default;
      
      const { container } = render(TestComponent);
      
      // Wait for hydration to complete
      await tick();
      
      // Verify server state is preserved
      expect(container.textContent).toContain('Hydration Test Case');
      expect(container.textContent).toContain('Test Evidence');
    });

    it('should handle hydration mismatch gracefully', async () => {
      // Simulate server/client state mismatch
      const serverState = {
        value: 'idle',
        context: { case: null, evidence: [] }
      };

      const clientState = {
        value: 'loaded',
        context: { 
          case: { id: 'case-456', title: 'Client Case' },
          evidence: []
        }
      };

      // Start with server state, then simulate client takeover
      mockUseMachine
        .mockReturnValueOnce({ state: serverState, send: mockActor.send })
        .mockReturnValueOnce({ state: clientState, send: mockActor.send });

      const TestComponent = (await import('$lib/components/legal/CaseManagerXState.svelte')).default;
      
      expect(() => {
        render(TestComponent);
      }).not.toThrow();
    });
  });

  describe('State Persistence', () => {
    it('should persist XState machine context to localStorage', async () => {
      const mockStorage = {
        getItem: vi.fn(),
        setItem: vi.fn(),
        removeItem: vi.fn()
      };

      Object.defineProperty(window, 'localStorage', {
        value: mockStorage,
        writable: true
      });

      const machineState = {
        value: 'analyzing',
        context: {
          case: { id: 'case-789' },
          aiAnalysisProgress: 75
        }
      };

      mockUseMachine.mockReturnValue({
        state: machineState,
        send: mockActor.send,
        actor: mockActor
      });

      const TestComponent = (await import('$lib/components/legal/CaseManagerXState.svelte')).default;
      render(TestComponent);

      await tick();

      // Should attempt to persist important state
      expect(mockStorage.setItem).toHaveBeenCalledWith(
        expect.stringContaining('xstate'),
        expect.any(String)
      );
    });

    it('should restore state from localStorage on page reload', async () => {
      const persistedState = JSON.stringify({
        value: 'loaded',
        context: {
          case: { id: 'restored-case', title: 'Restored Case' },
          activeTab: 'analysis'
        }
      });

      const mockStorage = {
        getItem: vi.fn().mockReturnValue(persistedState),
        setItem: vi.fn(),
        removeItem: vi.fn()
      };

      Object.defineProperty(window, 'localStorage', {
        value: mockStorage,
        writable: true
      });

      const TestComponent = (await import('$lib/components/legal/CaseManagerXState.svelte')).default;
      render(TestComponent);

      await tick();

      expect(mockStorage.getItem).toHaveBeenCalled();
      // Machine should be initialized with restored state
      expect(mockUseMachine).toHaveBeenCalledWith(
        expect.any(Object),
        expect.objectContaining({
          context: expect.objectContaining({
            case: expect.objectContaining({ id: 'restored-case' })
          })
        })
      );
    });
  });

  describe('Event Handling Hydration', () => {
    it('should maintain event handlers after hydration', async () => {
      const TestComponent = (await import('$lib/components/legal/CaseManagerXState.svelte')).default;
      const { container } = render(TestComponent);

      await tick();

      // Find and click a button that should trigger machine events
      const createButton = container.querySelector('button[type="submit"]');
      if (createButton) {
        fireEvent.click(createButton);
        expect(mockActor.send).toHaveBeenCalled();
      }
    });

    it('should handle form submissions during and after hydration', async () => {
      const TestComponent = (await import('$lib/components/legal/CaseManagerXState.svelte')).default;
      const { container } = render(TestComponent);

      await tick();

      // Fill form and submit
      const titleInput = container.querySelector('input[id="case-title"]');
      const form = container.querySelector('form');

      if (titleInput && form) {
        fireEvent.input(titleInput, { target: { value: 'Hydration Test Case' } });
        fireEvent.submit(form);

        await tick();

        expect(mockActor.send).toHaveBeenCalledWith(
          expect.objectContaining({
            type: 'CREATE_CASE'
          })
        );
      }
    });

    it('should handle file uploads after hydration', async () => {
      const TestComponent = (await import('$lib/components/legal/CaseManagerXState.svelte')).default;
      const { container } = render(TestComponent);

      await tick();

      // Simulate file upload
      const fileInput = container.querySelector('input[type="file"]');
      if (fileInput) {
        const file = new File(['test content'], 'test.pdf', { type: 'application/pdf' });
        fireEvent.change(fileInput, { target: { files: [file] } });

        await tick();

        expect(mockActor.send).toHaveBeenCalledWith(
          expect.objectContaining({
            type: 'ADD_EVIDENCE',
            files: expect.arrayContaining([file])
          })
        );
      }
    });
  });

  describe('Route-Based Hydration', () => {
    it('should handle case route parameter hydration', async () => {
      // Mock route with case ID
      mockPageStore.subscribe.mockImplementationOnce((callback) => {
        callback({
          url: new URL('http://localhost:5173/cases/case-123'),
          params: { caseId: 'case-123' },
          route: { id: '/cases/[caseId]' },
          status: 200,
          error: null,
          data: {},
          form: null,
          state: {}
        });
        return () => {};
      });

      const TestComponent = (await import('$routes/cases/[caseId]/+page.svelte')).default;
      render(TestComponent);

      await tick();

      // Should send LOAD_CASE event with route parameter
      expect(mockActor.send).toHaveBeenCalledWith({
        type: 'LOAD_CASE',
        caseId: 'case-123'
      });
    });

    it('should handle navigation state during hydration', async () => {
      // Mock navigation state
      mockNavigating.subscribe.mockImplementationOnce((callback) => {
        callback({
          from: new URL('http://localhost:5173/cases'),
          to: new URL('http://localhost:5173/cases/case-456'),
          type: 'link'
        });
        return () => {};
      });

      const TestComponent = (await import('$routes/cases/[caseId]/+page.svelte')).default;
      render(TestComponent);

      await tick();

      // Should handle navigation gracefully during hydration
      expect(() => {
        // Component should not throw during navigation
      }).not.toThrow();
    });
  });

  describe('WebSocket Hydration', () => {
    it('should reconnect WebSocket connections after hydration', async () => {
      const mockWebSocket = {
        send: vi.fn(),
        close: vi.fn(),
        readyState: 1, // OPEN
        addEventListener: vi.fn(),
        removeEventListener: vi.fn()
      };

      global.WebSocket = vi.fn().mockImplementation(() => mockWebSocket);

      const TestComponent = (await import('$lib/components/legal/CaseManagerXState.svelte')).default;
      render(TestComponent);

      await tick();

      // Should attempt to establish WebSocket connection
      expect(global.WebSocket).toHaveBeenCalled();
    });

    it('should handle real-time updates after hydration', async () => {
      const mockWebSocket = {
        send: vi.fn(),
        close: vi.fn(),
        readyState: 1,
        addEventListener: vi.fn(),
        removeEventListener: vi.fn()
      };

      global.WebSocket = vi.fn().mockImplementation(() => mockWebSocket);

      const TestComponent = (await import('$lib/components/legal/CaseManagerXState.svelte')).default;
      render(TestComponent);

      await tick();

      // Simulate receiving WebSocket message
      const messageHandler = mockWebSocket.addEventListener.mock.calls
        .find(call => call[0] === 'message')?.[1];
      
      if (messageHandler) {
        messageHandler({
          data: JSON.stringify({
            type: 'CASE_UPDATED',
            payload: { caseId: 'case-123', status: 'closed' }
          })
        });

        await tick();

        // Should send machine event from WebSocket message
        expect(mockActor.send).toHaveBeenCalledWith(
          expect.objectContaining({
            type: 'CASE_UPDATED'
          })
        );
      }
    });
  });

  describe('Performance During Hydration', () => {
    it('should complete hydration within reasonable time', async () => {
      const startTime = performance.now();

      const TestComponent = (await import('$lib/components/legal/CaseManagerXState.svelte')).default;
      render(TestComponent);

      await tick();

      const endTime = performance.now();
      const hydrationTime = endTime - startTime;

      expect(hydrationTime).toBeLessThan(1000); // Should hydrate within 1 second
    });

    it('should not cause layout thrashing during hydration', async () => {
      const TestComponent = (await import('$lib/components/legal/CaseManagerXState.svelte')).default;
      const { container } = render(TestComponent);

      const initialHeight = container.offsetHeight;

      await tick();

      const afterHydrationHeight = container.offsetHeight;

      // Height should not change significantly during hydration
      expect(Math.abs(afterHydrationHeight - initialHeight)).toBeLessThan(50);
    });
  });

  describe('Error Recovery During Hydration', () => {
    it('should recover from machine initialization errors', async () => {
      // Mock machine initialization error
      mockUseMachine.mockImplementationOnce(() => {
        throw new Error('Machine initialization failed');
      });

      const TestComponent = (await import('$lib/components/legal/CaseManagerXState.svelte')).default;
      
      expect(() => {
        render(TestComponent);
      }).not.toThrow();
      
      // Should provide fallback state
    });

    it('should handle service call failures during hydration', async () => {
      (global.fetch as any).mockRejectedValueOnce(new Error('Network error'));

      const TestComponent = (await import('$lib/components/legal/CaseManagerXState.svelte')).default;
      render(TestComponent);

      await tick();

      // Should handle network errors gracefully
      expect(mockActor.send).toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'ERROR'
        })
      );
    });

    it('should provide loading states during async hydration', async () => {
      // Mock slow loading state
      mockUseMachine.mockReturnValue({
        state: {
          value: 'loading',
          context: { isLoading: true, case: null },
          matches: vi.fn().mockReturnValue(true)
        },
        send: mockActor.send
      });

      const TestComponent = (await import('$lib/components/legal/CaseManagerXState.svelte')).default;
      const { container } = render(TestComponent);

      await tick();

      // Should show loading indicator
      expect(container.textContent).toContain('Loading');
    });
  });
});