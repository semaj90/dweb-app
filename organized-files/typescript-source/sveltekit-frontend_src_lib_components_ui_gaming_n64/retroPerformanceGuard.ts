// Retro performance guard for N64 UI: determines if retro effects should be enabled
// based on device memory, CPU cores, network conditions and prefers-reduced-motion.

export function shouldEnableRetroEffects(): boolean {
  try {
	if (typeof window === 'undefined') return false;

	// Respect user preference for reduced motion.
	const prefersReducedMotion =
	  typeof window !== 'undefined' &&
	  window.matchMedia &&
	  window.matchMedia('(prefers-reduced-motion: reduce)').matches;
	if (prefersReducedMotion) return false;

	const nav: any = navigator as any;

	// deviceMemory is in GB (may be undefined)
	const deviceMemory: number = nav.deviceMemory || 0;
	// number of logical processors (may be undefined)
	const cores: number = navigator.hardwareConcurrency || 0;

	// Network effective type: 'slow-2g', '2g', '3g', '4g' etc. (may be undefined)
	const connection = nav.connection || nav.mozConnection || nav.webkitConnection;
	const effectiveType: string | undefined = connection ? connection.effectiveType : undefined;
	const slowNetwork = effectiveType && ['slow-2g', '2g', '3g'].includes(effectiveType);

	// Heuristics: require moderate device memory or multiple cores and a non-slow network.
	const hasEnoughMemory = deviceMemory >= 3; // 3GB+ considered ok
	const hasEnoughCores = cores >= 4;
	if (slowNetwork) return false;

	return hasEnoughMemory || hasEnoughCores;
  } catch {
	return false;
  }
}

/**
 * Apply or remove a CSS class on the documentElement (or provided root element)
 * to allow CSS to toggle retro animations/styles based on runtime capability.
 *
 * Usage: import { applyRetroPerformanceGuard } and call it on app startup.
 */
export function applyRetroPerformanceGuard(root: Document | HTMLElement = document): void {
  if (typeof window === 'undefined') return;
  const enable = shouldEnableRetroEffects();
  const el = root instanceof Document ? document.documentElement : root;
  const className = 'n64-retro--enabled';

  if (enable) {
	el.classList.add(className);
  } else {
	el.classList.remove(className);
  }
}

export default applyRetroPerformanceGuard;
