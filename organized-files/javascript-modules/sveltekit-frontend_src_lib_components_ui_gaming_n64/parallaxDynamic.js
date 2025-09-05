import { shouldEnableRetroEffects } from './retroPerformanceGuard';

// parallaxDynamic.js
// Lightweight parallax controller for N64 UI.
// Respects runtime performance guard (if available) and prefers-reduced-motion.


const DEFAULT_OPTIONS = {
    selector: '[data-parallax-layer]',
    rootElement: document, // document or a container element
    strength: 0.03, // multiplier for translation based on layer depth
    maxTranslate: 40, // px clamp for translation
    pointer: true,
    gyro: true,
    resetOnLeave: true,
};

export default function createParallax(userOptions = {}) {
    if (typeof window === 'undefined') {
        return {
            enable: () => {},
            disable: () => {},
            destroy: () => {},
            isEnabled: false,
        };
    }

    const opts = { ...DEFAULT_OPTIONS, ...userOptions };
    let enabled = false;
    let rafId = null;
    let lastPointer = { x: 0, y: 0 };
    let target = { x: 0, y: 0 };
    let layers = [];
    let rootEl = opts.rootElement instanceof Document ? document : opts.rootElement;

    function queryLayers() {
        layers = Array.from((rootEl).querySelectorAll(opts.selector)).map((el) => {
            const depth = parseFloat(el.getAttribute('data-depth')) || 0;
            return { el, depth };
        });
        layers.forEach((l) => {
            l.el.style.willChange = 'transform';
            l.el.style.transition = 'transform 0.08s linear';
        });
    }

    function clamp(v, min, max) {
        return Math.max(min, Math.min(max, v));
    }

    function applyTransforms() {
        const strength = opts.strength;
        const maxT = opts.maxTranslate;

        layers.forEach(({ el, depth }) => {
            // deeper (larger depth) means move more
            const factor = depth * strength;
            const tx = clamp(target.x * factor, -maxT, maxT);
            const ty = clamp(target.y * factor, -maxT, maxT);
            el.style.transform = `translate3d(${tx}px, ${ty}px, 0)`;
        });
    }

    function rafLoop() {
        // simple lerp to smooth motion
        lastPointer.x += (target.x - lastPointer.x) * 0.12;
        lastPointer.y += (target.y - lastPointer.y) * 0.12;
        applyTransforms();
        rafId = requestAnimationFrame(rafLoop);
    }

    function onPointerMove(e) {
        const rect = (rootEl instanceof Document ? document.documentElement : rootEl).getBoundingClientRect();
        const cx = rect.left + rect.width / 2;
        const cy = rect.top + rect.height / 2;
        const clientX = e.touches ? e.touches[0].clientX : e.clientX;
        const clientY = e.touches ? e.touches[0].clientY : e.clientY;
        // normalized -1..1
        const nx = (clientX - cx) / (rect.width / 2);
        const ny = (clientY - cy) / (rect.height / 2);
        target.x = nx * opts.maxTranslate;
        target.y = ny * opts.maxTranslate;
    }

    function onDeviceOrientation(e) {
        // Use gamma (left/right) and beta (front/back)
        const gamma = e.gamma || 0;
        const beta = e.beta || 0;
        // small scale to keep movement subtle
        target.x = clamp(gamma / 45 * opts.maxTranslate, -opts.maxTranslate, opts.maxTranslate);
        target.y = clamp(beta / 45 * opts.maxTranslate, -opts.maxTranslate, opts.maxTranslate);
    }

    function onLeave() {
        if (!opts.resetOnLeave) return;
        target.x = 0;
        target.y = 0;
    }

    function addListeners() {
        if (opts.pointer) {
            window.addEventListener('mousemove', onPointerMove, { passive: true });
            window.addEventListener('touchmove', onPointerMove, { passive: true });
            window.addEventListener('mouseleave', onLeave);
            window.addEventListener('touchend', onLeave);
        }
        if (opts.gyro && 'DeviceOrientationEvent' in window) {
            // prefer permission flow on iOS 13+ if needed — caller may handle permission
            window.addEventListener('deviceorientation', onDeviceOrientation, { passive: true });
        }
    }

    function removeListeners() {
        window.removeEventListener('mousemove', onPointerMove);
        window.removeEventListener('touchmove', onPointerMove);
        window.removeEventListener('mouseleave', onLeave);
        window.removeEventListener('touchend', onLeave);
        window.removeEventListener('deviceorientation', onDeviceOrientation);
    }

    function enable() {
        // Respect reduced motion and performance guard if available
        const prefersReducedMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
        if (prefersReducedMotion) return (enabled = false);

        // if the imported guard exists and returns false, disable
        try {
            if (typeof shouldEnableRetroEffects === 'function' && !shouldEnableRetroEffects()) {
                return (enabled = false);
            }
        } catch {
            // ignore and proceed
        }

        queryLayers();
        if (!layers.length) return (enabled = false);

        // set CSS hook
        document.documentElement.classList.add('n64-parallax--enabled');

        addListeners();
        if (!rafId) rafLoop();
        enabled = true;
        return enabled;
    }

    function disable() {
        cancelAnimationFrame(rafId || 0);
        rafId = null;
        removeListeners();
        layers.forEach((l) => {
            l.el.style.transform = '';
            l.el.style.willChange = '';
            l.el.style.transition = '';
        });
        document.documentElement.classList.remove('n64-parallax--enabled');
        enabled = false;
    }

    function destroy() {
        disable();
        layers = [];
    }

    // auto-enable unless user opts out
    if (userOptions.auto !== false) {
        enable();
    }

    return {
        enable,
        disable,
        destroy,
        isEnabled: () => enabled,
    };
}