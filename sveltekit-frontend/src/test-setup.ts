/// <reference types="vitest" />
import { vi } from 'vitest';


// Mock browser environment for tests. Avoid writing to read-only globals directly.
beforeAll(() => {
  // Mock global fetch if not present
  if (typeof global.fetch === 'undefined') {
    (global as any).fetch = vi.fn();
  }

  // Mock performance if not available
  if (typeof (global as any).performance === 'undefined') {
    (global as any).performance = {
      now: vi.fn(() => Date.now()),
      mark: vi.fn(),
      measure: vi.fn(),
      getEntriesByName: vi.fn(() => []),
      getEntriesByType: vi.fn(() => []),
      clearMarks: vi.fn(),
      clearMeasures: vi.fn()
    } as any;
  }

  // Provide AbortSignal.timeout shim only if missing
  if (typeof (AbortSignal as any).timeout === 'undefined') {
    (AbortSignal as any).timeout = vi.fn((ms: number) => {
      const controller = new AbortController();
      setTimeout(() => controller.abort(), ms);
      return controller.signal;
    });
  }
});

// Extend vitest matchers
expect.extend({
  toBeOneOf(received: any, array: any[]) {
    const pass = array.includes(received);
    if (pass) {
      return {
        message: () => `expected ${received} not to be one of ${array}`,
        pass: true,
      } as any;
    } else {
      return {
        message: () => `expected ${received} to be one of ${array}`,
        pass: false,
      } as any;
    }
  },
});