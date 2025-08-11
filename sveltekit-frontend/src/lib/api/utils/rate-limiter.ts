// Simple no-op rate limiter stub
export function rateLimit<T>(fn: (...args: any[]) => Promise<T>) {
  return fn;
}
