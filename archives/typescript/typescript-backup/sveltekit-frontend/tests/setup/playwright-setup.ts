import { expect } from '@playwright/test';

// Register safe shim for toHaveProperty where objects may be plain JSON
// If toHaveProperty is not present or fails due to typing, we provide a helper that falls back
// to a simple property existence check using Object.prototype.hasOwnProperty.
// This reduces test failures from env differences.

// @ts-ignore allow augment
declare global {
  // eslint-disable-next-line @typescript-eslint/no-namespace
  namespace PlaywrightTest {
    interface Matchers<R> {}
  }
}

// Provide a tiny wrapper helper used in tests: expectHas(obj, prop)
export function expectHas(obj: any, prop: string) {
  try {
    // prefer Playwright's matcher when available
    // @ts-ignore
    if (expect(obj).toHaveProperty) {
      // @ts-ignore
      return expect(obj).toHaveProperty(prop);
    }
  } catch {
    // ignore and fallback
  }
  // fallback implementation
  if (obj && Object.prototype.hasOwnProperty.call(obj, prop)) {
    return expect(true).toBe(true);
  }
  return expect(false).toBe(true);
}

// No default export; tests can import { expectHas } from './setup/playwright-setup'
