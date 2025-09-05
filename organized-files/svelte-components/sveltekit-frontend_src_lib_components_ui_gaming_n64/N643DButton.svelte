<script lang="ts">
    import { onMount, onDestroy, createEventDispatcher } from 'svelte';
    import createParallax from './parallaxDynamic.js';

    export let disabled: boolean = false;
    export let ariaLabel: string | undefined;
    export let size: 'sm' | 'md' | 'lg' = 'md';
    export let className = '';

    const dispatch = createEventDispatcher();
    let rootEl: HTMLElement
    let parallax: any

    function handleClick(e: MouseEvent) {
        if (disabled) {
            e.preventDefault();
            e.stopPropagation();
            return;
        }
        dispatch('click', e);
    }

    function onKeydown(e: KeyboardEvent) {
        if (disabled) return;
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            // emulate click
            dispatch('click', { keyboard: true });
        }
    }

    onMount(() => {
        // initialize scoped parallax for the button's layers
        parallax = createParallax({
            rootElement: rootEl,
            selector: '[data-parallax-layer]',
            strength: 0.035,
            maxTranslate: 18,
            pointer: true,
            gyro: false, // keep gyro off for small UI controls
            resetOnLeave: true,
        });

        return () => {};
    });

    onDestroy(() => {
        if (parallax && typeof parallax.destroy === 'function') parallax.destroy();
    });
</script>

<button
    bind:this={rootEl}
    class="n64-3d-button {size} {disabled ? 'is-disabled' : ''} {className}"
    on:click={handleClick}
    on:keydown={onKeydown}
    aria-label={ariaLabel}
    aria-disabled={disabled}
    disabled={disabled}
    type="button"
>
    <!-- Shadow layer (furthest, small movement) -->
    <span class="layer shadow" data-parallax-layer data-depth="0.06"></span>

    <!-- Base / bezel (stationary) -->
    <span class="layer bezel" data-parallax-layer data-depth="0.02"></span>

    <!-- Main face (visible content) -->
    <span class="layer face" data-parallax-layer data-depth="0.22">
        <slot>Button</slot>
    </span>

    <!-- Highlight (top gloss) -->
    <span class="layer highlight" data-parallax-layer data-depth="0.34" aria-hidden="true"></span>
</button>

<style>
    .n64-3d-button {
        --bg: #3a2f1b;
        --face: #ffd26f;
        --bezel: #0f0b07;
        --shadow: rgba(0,0,0,0.45);
        --highlight: rgba(255,255,255,0.18);

        display: inline-block;
        position: relative
        border: none
        background: transparent
        padding: 0;
        cursor: pointer
        outline: none
        user-select: none
        -webkit-tap-highlight-color: transparent
        transform-style: preserve-3d;
        vertical-align: middle
    }

    .n64-3d-button.is-disabled {
        opacity: 0.5;
        cursor: not-allowed;
    }

    /* sizes */
    .n64-3d-button.sm { width: 96px; height: 40px; }
    .n64-3d-button.md { width: 140px; height: 56px; }
    .n64-3d-button.lg { width: 220px; height: 72px; }

    .n64-3d-button .layer {
        position: absolute
        left: 0;
        top: 0;
        right: 0;
        bottom: 0;
        border-radius: 10px;
        pointer-events: none
        transform-origin: 50% 50%;
        transition: transform 120ms linear;
    }

    .n64-3d-button .shadow {
        background: linear-gradient(180deg, transparent 0%, var(--shadow) 100%);
        filter: blur(6px);
        transform: translateZ(-12px);
        z-index: 0;
    }

    .n64-3d-button .bezel {
        z-index: 5;
        background: linear-gradient(180deg, #20160b, #0f0b07);
        box-shadow: inset 0 2px 0 rgba(255,255,255,0.03), 0 6px 0 rgba(0,0,0,0.25);
        transform: translateZ(-6px);
    }

    .n64-3d-button .face {
        z-index: 10;
        display: flex
        align-items: center
        justify-content: center
        font-weight: 700;
        color: #1b1309;
        background: var(--face);
        box-shadow: inset 0 -4px 0 rgba(0,0,0,0.06);
        transform: translateZ(0);
        padding: 0 12px;
        font-family: system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
    }

    .n64-3d-button .highlight {
        z-index: 15;
        background: linear-gradient(180deg, rgba(255,255,255,0.5) 0%, transparent 40%);
        mix-blend-mode: overlay
        pointer-events: none
        transform: translateZ(6px);
    }

    /* focus ring */
    .n64-3d-button:focus .face {
        box-shadow: 0 0 0 3px rgba(255, 210, 111, 0.22);
    }

    /* Respect reduced motion by minimizing transitions if user prefers reduced motion */
    @media (prefers-reduced-motion: reduce) {
        .n64-3d-button .layer {
            transition: none !important;
        }
    }

    /* When global retro class disabled, reduce fancy visuals */
    :global(.n64-retro--enabled) .n64-3d-button { /* nothing extra */ }
    :global(.n64-retro--enabled) .n64-3d-button .face { /* keep retro styles */ }
</style>