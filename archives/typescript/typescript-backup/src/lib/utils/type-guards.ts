/**
 * Type guard utilities for handling unknown types safely
 */

export function isObject(value: any): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

export function isString(value: any): value is string {
  return typeof value === 'string';
}

export function isNumber(value: any): value is number {
  return typeof value === 'number' && !isNaN(value);
}

export function hasProperty<T extends string>(
  obj: any,
  prop: T
): obj is Record<T, unknown> {
  return isObject(obj) && prop in obj;
}

export function safeSpread(obj: any): Record<string, unknown> {
  return isObject(obj) ? obj : {};
}

export function safeString(value: any, fallback = ''): string {
  return isString(value) ? value : fallback;
}

export function safeNumber(value: any, fallback = 0): number {
  return isNumber(value) ? value : fallback;
}

export function getProperty(obj: any, prop: string): any {
  return isObject(obj) ? obj[prop] : undefined;
}