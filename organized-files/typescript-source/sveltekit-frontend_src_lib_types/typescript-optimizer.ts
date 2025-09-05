/**
 * Lightweight TypeScript helper utilities to centralize common types
 * and ensure this file is treated as a module.
 *
 * These helpers are intentionally minimal and safe to import from any part
 * of the codebase without introducing runtime side-effects.
 */

export type Dict<T = any> = Record<string, T>;

export type Mutable<T> = { -readonly [P in keyof T]: T[P] };

export type DeepPartial<T> = {
  [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
};

export type Awaitable<T> = T | Promise<T>;

export type UnwrapPromise<T> = T extends Promise<infer U> ? U : T;

/**
 * Identity helper (useful in generics-heavy code to help TS infer types).
 */
export function identity<T>(value: T): T {
  return value;
}

/**
 * Ensure this file is treated as a module even if only types are imported.
 */
export default {};
