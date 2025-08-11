<script lang="ts">
	import type { ComponentProps } from 'svelte';
	
	interface GamingButtonProps {
		variant?: 'primary' | 'secondary' | 'danger' | 'success' | 'warning';
		size?: 'sm' | 'md' | 'lg';
		disabled?: boolean;
		loading?: boolean;
		glowEffect?: boolean;
		soundEnabled?: boolean;
		onclick?: () => void;
		children: any;
	}
	
	let { 
		variant = 'primary',
		size = 'md',
		disabled = false,
		loading = false,
		glowEffect = false,
		soundEnabled = true,
		onclick,
		children
	}: GamingButtonProps = $props();
	
	let buttonElement = $state<HTMLButtonElement>();
	let isPressed = $state(false);
	
	function handleClick(event: MouseEvent) {
		if (disabled || loading) return;
		
		isPressed = true;
		setTimeout(() => isPressed = false, 150);
		
		// Gaming sound effect (optional)
		if (soundEnabled) {
			playClickSound();
		}
		
		onclick?.(event);
	}
	
	function playClickSound() {
		// Create audio context for gaming click sound
		try {
			const audioContext = new (window.AudioContext || window.webkitAudioContext)();
			const oscillator = audioContext.createOscillator();
			const gainNode = audioContext.createGain();
			
			oscillator.connect(gainNode);
			gainNode.connect(audioContext.destination);
			
			oscillator.frequency.setValueAtTime(800, audioContext.currentTime);
			oscillator.frequency.exponentialRampToValueAtTime(400, audioContext.currentTime + 0.1);
			
			gainNode.gain.setValueAtTime(0.1, audioContext.currentTime);
			gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.1);
			
			oscillator.start(audioContext.currentTime);
			oscillator.stop(audioContext.currentTime + 0.1);
		} catch (error) {
			// Audio context not supported, silently fail
		}
	}
</script>

<button
	bind:this={buttonElement}
	class="gaming-button {variant} {size}"
	class:disabled
	class:loading
	class:pressed={isPressed}
	class:glow={glowEffect}
	{disabled}
	onclick={handleClick}
	{...$$restProps}
>
	{#if loading}
		<div class="loading-spinner"></div>
	{/if}
	
	<span class="button-content" class:loading>
		{@render children()}
	</span>
	
	<!-- Gaming button effects -->
	<div class="button-overlay"></div>
	<div class="scan-line"></div>
</button>

<style>
	.gaming-button {
		position: relative;
		display: inline-flex;
		align-items: center;
		justify-content: center;
		gap: 8px;
		border: none;
		border-radius: 4px;
		font-family: 'Orbitron', 'Courier New', monospace;
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 1px;
		cursor: pointer;
		transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
		overflow: hidden;
		user-select: none;
		outline: none;
		background: linear-gradient(135deg, transparent 0%, rgba(255,255,255,0.1) 50%, transparent 100%);
	}
	
	/* Size Variants */
	.gaming-button.sm {
		padding: 8px 16px;
		font-size: 12px;
		min-height: 32px;
	}
	
	.gaming-button.md {
		padding: 12px 24px;
		font-size: 14px;
		min-height: 40px;
	}
	
	.gaming-button.lg {
		padding: 16px 32px;
		font-size: 16px;
		min-height: 48px;
	}
	
	/* Color Variants */
	.gaming-button.primary {
		background: linear-gradient(135deg, #0066cc 0%, #004499 100%);
		border: 2px solid #0088ff;
		color: #ffffff;
		box-shadow: 
			0 0 20px rgba(0, 136, 255, 0.3),
			inset 0 1px 0 rgba(255, 255, 255, 0.2);
	}
	
	.gaming-button.primary:hover:not(:disabled) {
		background: linear-gradient(135deg, #0077dd 0%, #0055aa 100%);
		border-color: #00aaff;
		box-shadow: 
			0 0 30px rgba(0, 136, 255, 0.5),
			inset 0 1px 0 rgba(255, 255, 255, 0.3);
		transform: translateY(-2px);
	}
	
	.gaming-button.secondary {
		background: linear-gradient(135deg, #333333 0%, #1a1a1a 100%);
		border: 2px solid #555555;
		color: #ffffff;
		box-shadow: 
			0 0 20px rgba(255, 255, 255, 0.1),
			inset 0 1px 0 rgba(255, 255, 255, 0.1);
	}
	
	.gaming-button.secondary:hover:not(:disabled) {
		background: linear-gradient(135deg, #444444 0%, #2a2a2a 100%);
		border-color: #777777;
		box-shadow: 
			0 0 30px rgba(255, 255, 255, 0.2),
			inset 0 1px 0 rgba(255, 255, 255, 0.2);
		transform: translateY(-2px);
	}
	
	.gaming-button.success {
		background: linear-gradient(135deg, #00cc66 0%, #009944 100%);
		border: 2px solid #00ff88;
		color: #ffffff;
		box-shadow: 
			0 0 20px rgba(0, 255, 136, 0.3),
			inset 0 1px 0 rgba(255, 255, 255, 0.2);
	}
	
	.gaming-button.success:hover:not(:disabled) {
		background: linear-gradient(135deg, #00dd77 0%, #00aa55 100%);
		border-color: #00ffaa;
		box-shadow: 
			0 0 30px rgba(0, 255, 136, 0.5),
			inset 0 1px 0 rgba(255, 255, 255, 0.3);
		transform: translateY(-2px);
	}
	
	.gaming-button.danger {
		background: linear-gradient(135deg, #cc3333 0%, #aa1111 100%);
		border: 2px solid #ff4444;
		color: #ffffff;
		box-shadow: 
			0 0 20px rgba(255, 68, 68, 0.3),
			inset 0 1px 0 rgba(255, 255, 255, 0.2);
	}
	
	.gaming-button.danger:hover:not(:disabled) {
		background: linear-gradient(135deg, #dd4444 0%, #bb2222 100%);
		border-color: #ff6666;
		box-shadow: 
			0 0 30px rgba(255, 68, 68, 0.5),
			inset 0 1px 0 rgba(255, 255, 255, 0.3);
		transform: translateY(-2px);
	}
	
	.gaming-button.warning {
		background: linear-gradient(135deg, #ff8800 0%, #cc6600 100%);
		border: 2px solid #ffaa00;
		color: #ffffff;
		box-shadow: 
			0 0 20px rgba(255, 170, 0, 0.3),
			inset 0 1px 0 rgba(255, 255, 255, 0.2);
	}
	
	.gaming-button.warning:hover:not(:disabled) {
		background: linear-gradient(135deg, #ff9900 0%, #dd7700 100%);
		border-color: #ffbb00;
		box-shadow: 
			0 0 30px rgba(255, 170, 0, 0.5),
			inset 0 1px 0 rgba(255, 255, 255, 0.3);
		transform: translateY(-2px);
	}
	
	/* States */
	.gaming-button:disabled {
		opacity: 0.5;
		cursor: not-allowed;
		transform: none !important;
		box-shadow: none !important;
		background: #333333 !important;
		border-color: #555555 !important;
	}
	
	.gaming-button.pressed {
		transform: translateY(1px) scale(0.98);
		box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.3) !important;
	}
	
	.gaming-button.glow {
		animation: gaming-glow 2s ease-in-out infinite alternate;
	}
	
	/* Button Content */
	.button-content {
		position: relative;
		z-index: 2;
		display: flex;
		align-items: center;
		gap: 8px;
		transition: opacity 0.2s ease;
	}
	
	.button-content.loading {
		opacity: 0.7;
	}
	
	/* Loading Spinner */
	.loading-spinner {
		position: absolute;
		width: 20px;
		height: 20px;
		border: 2px solid rgba(255, 255, 255, 0.3);
		border-top: 2px solid currentColor;
		border-radius: 50%;
		animation: spin 1s linear infinite;
		z-index: 3;
	}
	
	/* Gaming Effects */
	.button-overlay {
		position: absolute;
		top: 0;
		left: -100%;
		width: 100%;
		height: 100%;
		background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.2) 50%, transparent 100%);
		transition: left 0.5s ease;
		z-index: 1;
	}
	
	.gaming-button:hover:not(:disabled) .button-overlay {
		left: 100%;
	}
	
	.scan-line {
		position: absolute;
		top: 0;
		left: 0;
		right: 0;
		height: 1px;
		background: linear-gradient(90deg, transparent 0%, currentColor 50%, transparent 100%);
		opacity: 0.6;
		animation: scan 3s ease-in-out infinite;
		z-index: 1;
	}
	
	/* Animations */
	@keyframes spin {
		0% { transform: rotate(0deg); }
		100% { transform: rotate(360deg); }
	}
	
	@keyframes gaming-glow {
		0% {
			box-shadow: 
				0 0 20px currentColor,
				inset 0 1px 0 rgba(255, 255, 255, 0.2);
		}
		100% {
			box-shadow: 
				0 0 40px currentColor,
				0 0 60px currentColor,
				inset 0 1px 0 rgba(255, 255, 255, 0.3);
		}
	}
	
	@keyframes scan {
		0%, 100% {
			transform: translateX(-100%);
			opacity: 0;
		}
		50% {
			transform: translateX(0%);
			opacity: 0.6;
		}
	}
	
	/* Focus styles for accessibility */
	.gaming-button:focus-visible {
		outline: 2px solid currentColor;
		outline-offset: 2px;
	}
</style>