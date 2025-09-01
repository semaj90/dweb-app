
// Debounce utility functions for performance optimization

/**
 * Debounce function that delays execution until after wait milliseconds
 * have elapsed since the last time it was invoked
 */
export function debounce<T extends (...args: any[]) => any>(
  func: T,
  wait: number,
  immediate = false
): (...args: Parameters<T>) => void {
  let timeout: ReturnType<typeof setTimeout> | null = null;

  return function executedFunction(...args: Parameters<T>) {
    const later = () => {
      timeout = null;
      if (!immediate) func(...args);
    };

    const callNow = immediate && !timeout;

    if (timeout) clearTimeout(timeout);
    timeout = setTimeout(later, wait);

    if (callNow) func(...args);
  };
}

/**
 * Throttle function that limits execution to once per wait period
 */
export function throttle<T extends (...args: any[]) => any>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let inThrottle: boolean;

  return function executedFunction(...args: Parameters<T>) {
    if (!inThrottle) {
      func.apply(this, args);
      inThrottle = true;
      setTimeout(() => (inThrottle = false), wait);
    }
  };
}

/**
 * Async debounce for promises
 */
export function debounceAsync<T extends (...args: any[]) => Promise<any>>(
  func: T,
  wait: number
): (...args: Parameters<T>) => Promise<ReturnType<T>> {
  let timeout: ReturnType<typeof setTimeout> | null = null;
  let resolveList: Array<(value: any) => void> = [];
  let rejectList: Array<(reason: any) => void> = [];

  return function executedFunction(...args: Parameters<T>): Promise<ReturnType<T>> {
    return new Promise((resolve, reject) => {
      resolveList.push(resolve);
      rejectList.push(reject);

      if (timeout) clearTimeout(timeout);

      timeout = setTimeout(async () => {
        try {
          const result = await func(...args);
          resolveList.forEach((r: any) => r(result));
        } catch (error: any) {
          rejectList.forEach((r: any) => r(error));
        } finally {
          resolveList = [];
          rejectList = [];
          timeout = null;
        }
      }, wait);
    });
  };
}

/**
 * Search debounce specifically for input fields
 */
export function createSearchDebounce(
  callback: (query: string) => void,
  delay = 300
) {
  return debounce(callback, delay);
}

/**
 * Auto-save debounce for form fields
 */
export function createAutoSaveDebounce(
  callback: () => void | Promise<void>,
  delay = 1000
) {
  return debounce(callback, delay);
}