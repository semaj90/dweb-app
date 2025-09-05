/**
 * Creates a debounced function that delays invoking the provided function 
 * until after the specified delay has elapsed since the last time it was invoked.
 */
export function debounce<T extends (...args: any[]) => any>(
	func: T,
	delay: number
): (...args: Parameters<T>) => void {
	let timeoutId: NodeJS.Timeout | null = null;

	return function debounced(...args: Parameters<T>) {
		if (timeoutId !== null) {
			clearTimeout(timeoutId);
		}

		timeoutId = setTimeout(() => {
			func(...args);
			timeoutId = null;
		}, delay);
	};
}

/**
 * Creates a throttled function that only invokes the provided function 
 * at most once per the specified delay period.
 */
export function throttle<T extends (...args: any[]) => any>(
	func: T,
	delay: number
): (...args: Parameters<T>) => void {
	let lastCall = 0;
	let timeoutId: NodeJS.Timeout | null = null;

	return function throttled(...args: Parameters<T>) {
		const now = Date.now();

		if (now - lastCall >= delay) {
			func(...args);
			lastCall = now;
		} else if (timeoutId === null) {
			timeoutId = setTimeout(() => {
				func(...args);
				lastCall = Date.now();
				timeoutId = null;
			}, delay - (now - lastCall));
		}
	};
}
