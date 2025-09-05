// Svelte action / utility to apply retro-style transform effects (pointer + gyro)
// Minimal, safe implementation that no-ops during SSR and cleans up listeners.

export default function useRetroTransform(node, options = {}) {
  if (typeof window === 'undefined' || !node) {
	// SSR or invalid node: return an action-compatible no-op
	return {
	  update() {},
	  destroy() {},
	};
  }

  const opts = {
	strength: 0.03,
	maxTranslate: 40,
	pointer: true,
	gyro: false,
	resetOnLeave: true,
	...options,
  };

  let bounds = null;

  function clamp(v, min, max) {
	return Math.max(min, Math.min(max, v));
  }

  function updateBounds() {
	bounds = node.getBoundingClientRect();
  }

  function applyTransform(x, y) {
	// x,y are -1..1 where -1 is left/top and 1 is right/bottom
	const tx = clamp(x * opts.maxTranslate * opts.strength * 100, -opts.maxTranslate, opts.maxTranslate);
	const ty = clamp(y * opts.maxTranslate * opts.strength * 100, -opts.maxTranslate, opts.maxTranslate);
	node.style.transform = `translate3d(${tx}px, ${ty}px, 0)`;
	node.style.willChange = 'transform';
  }

  function resetTransform() {
	node.style.transform = '';
	node.style.willChange = '';
  }

  function onPointerMove(e) {
	if (!bounds) updateBounds();
	const clientX = e.clientX ?? (e.touches && e.touches[0] && e.touches[0].clientX) ?? 0;
	const clientY = e.clientY ?? (e.touches && e.touches[0] && e.touches[0].clientY) ?? 0;
	const x = ((clientX - bounds.left) / bounds.width) * 2 - 1;
	const y = ((clientY - bounds.top) / bounds.height) * 2 - 1;
	applyTransform(x, y);
  }

  function onLeave() {
	if (opts.resetOnLeave) resetTransform();
  }

  function onDeviceOrientation(e) {
	// Use gamma (left/right) and beta (front/back) to create small parallax
	const gamma = e.gamma ?? 0; // -90 .. 90
	const beta = e.beta ?? 0; // -180 .. 180
	// normalize to -1..1 with reasonable clamping
	const x = clamp(gamma / 30, -1, 1);
	const y = clamp(beta / 30, -1, 1);
	applyTransform(x, y);
  }

  if (opts.pointer) {
	node.addEventListener('pointermove', onPointerMove, { passive: true });
	node.addEventListener('pointerleave', onLeave);
	window.addEventListener('resize', updateBounds);
	updateBounds();
  }

  if (opts.gyro && typeof window !== 'undefined' && 'DeviceOrientationEvent' in window) {
	window.addEventListener('deviceorientation', onDeviceOrientation);
  }

  return {
	update(newOptions = {}) {
	  Object.assign(opts, newOptions);
	  updateBounds();
	},
	destroy() {
	  node.removeEventListener('pointermove', onPointerMove);
	  node.removeEventListener('pointerleave', onLeave);
	  window.removeEventListener('resize', updateBounds);
	  window.removeEventListener('deviceorientation', onDeviceOrientation);
	  resetTransform();
	},
  };
}
