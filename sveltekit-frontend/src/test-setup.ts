import { beforeAll, vi } from 'vitest';

// Mock browser environment
beforeAll(() => {
  // Mock global fetch
  global.fetch = vi.fn();
  
  // Mock performance if not available
  if (typeof global.performance === 'undefined') {
    global.performance = {
      now: vi.fn(() => Date.now()),
      mark: vi.fn(),
      measure: vi.fn(),
      getEntriesByName: vi.fn(() => []),
      getEntriesByType: vi.fn(() => []),
      clearMarks: vi.fn(),
      clearMeasures: vi.fn()
    } as any;
  }

  // Mock AbortSignal timeout for Node.js environments
  if (typeof AbortSignal.timeout === 'undefined') {
    AbortSignal.timeout = vi.fn((ms: number) => {
      const controller = new AbortController();
      setTimeout(() => controller.abort(), ms);
      return controller.signal;
    });
  }
});

// Extend vitest matchers
import { expect } from 'vitest';

expect.extend({
  toBeOneOf(received, array) {
    const pass = array.includes(received);
    if (pass) {
      return {
        message: () => `expected ${received} not to be one of ${array}`,
        pass: true,
      };
    } else {
      return {
        message: () => `expected ${received} to be one of ${array}`,
        pass: false,
      };
    }
  },
});