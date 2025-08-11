<script lang="ts">
	import { onMount } from 'svelte';
	import type { ComponentProps } from 'svelte';
	
	// Gaming-themed props using Svelte 5 patterns
	interface GamingHUDProps {
		userLevel?: number;
		experience?: number;
		maxExperience?: number;
		currentCase?: string;
		documentsAnalyzed?: number;
		accuracyScore?: number;
		isOnline?: boolean;
	}
	
	let { 
		userLevel = 1,
		experience = 750,
		maxExperience = 1000,
		currentCase = "CASE-2024-001",
		documentsAnalyzed = 47,
		accuracyScore = 94.2,
		isOnline = true
	}: GamingHUDProps = $props();
	
	let currentTime = $state('00:00:00');
	let glowEffect = $state(false);
	
	// Experience bar percentage
	let experiencePercent = $derived(() => Math.round((experience / maxExperience) * 100));
	
	onMount(() => {
		// Update time every second
		const timeInterval = setInterval(() => {
			const now = new Date();
			currentTime = now.toLocaleTimeString();
		}, 1000);
		
		// Glow effect animation
		const glowInterval = setInterval(() => {
			glowEffect = !glowEffect;
		}, 2000);
		
		return () => {
			clearInterval(timeInterval);
			clearInterval(glowInterval);
		};
	});
</script>

<!-- Gaming HUD Container -->
<div class="gaming-hud">
	<!-- Top Bar -->
	<div class="hud-top-bar">
		<!-- User Level & Experience -->
		<div class="level-section">
			<div class="level-badge" class:glow={glowEffect}>
				<span class="level-text">LVL</span>
				<span class="level-number">{userLevel}</span>
			</div>
			<div class="experience-bar">
				<div class="exp-background">
					<div 
						class="exp-fill" 
						style="width: {experiencePercent}%"
					></div>
				</div>
				<span class="exp-text">{experience}/{maxExperience} EXP</span>
			</div>
		</div>
		
		<!-- Current Case Info -->
		<div class="case-section">
			<div class="case-label">ACTIVE CASE</div>
			<div class="case-id">{currentCase}</div>
		</div>
		
		<!-- System Status -->
		<div class="status-section">
			<div class="status-indicator" class:online={isOnline} class:offline={!isOnline}>
				<div class="status-dot"></div>
				<span>{isOnline ? 'ONLINE' : 'OFFLINE'}</span>
			</div>
			<div class="system-time">{currentTime}</div>
		</div>
	</div>
	
	<!-- Stats Panel -->
	<div class="stats-panel">
		<div class="stat-item">
			<div class="stat-icon">📊</div>
			<div class="stat-content">
				<div class="stat-label">DOCUMENTS</div>
				<div class="stat-value">{documentsAnalyzed}</div>
			</div>
		</div>
		
		<div class="stat-item">
			<div class="stat-icon">🎯</div>
			<div class="stat-content">
				<div class="stat-label">ACCURACY</div>
				<div class="stat-value">{accuracyScore}%</div>
			</div>
		</div>
		
		<div class="stat-item">
			<div class="stat-icon">⚡</div>
			<div class="stat-content">
				<div class="stat-label">AI STATUS</div>
				<div class="stat-value">ACTIVE</div>
			</div>
		</div>
	</div>
</div>

<style>
	.gaming-hud {
		position: fixed;
		top: 0;
		left: 0;
		right: 0;
		z-index: 1000;
		background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
		border-bottom: 2px solid #00ff88;
		box-shadow: 0 4px 20px rgba(0, 255, 136, 0.3);
		font-family: 'Orbitron', 'Courier New', monospace;
		backdrop-filter: blur(10px);
	}
	
	.hud-top-bar {
		display: flex;
		justify-content: space-between;
		align-items: center;
		padding: 12px 24px;
		background: rgba(0, 0, 0, 0.4);
	}
	
	/* Level Section */
	.level-section {
		display: flex;
		align-items: center;
		gap: 16px;
	}
	
	.level-badge {
		display: flex;
		align-items: center;
		background: linear-gradient(45deg, #ff6b00, #ff8c42);
		padding: 8px 16px;
		border-radius: 20px;
		border: 2px solid #ffaa00;
		transition: all 0.3s ease;
	}
	
	.level-badge.glow {
		box-shadow: 0 0 20px rgba(255, 170, 0, 0.8);
		transform: scale(1.05);
	}
	
	.level-text {
		font-size: 12px;
		font-weight: bold;
		color: #fff;
		margin-right: 4px;
	}
	
	.level-number {
		font-size: 18px;
		font-weight: bold;
		color: #fff;
	}
	
	.experience-bar {
		position: relative;
		width: 200px;
	}
	
	.exp-background {
		width: 100%;
		height: 8px;
		background: rgba(255, 255, 255, 0.1);
		border-radius: 4px;
		overflow: hidden;
		border: 1px solid #333;
	}
	
	.exp-fill {
		height: 100%;
		background: linear-gradient(90deg, #00ff88, #00cc70);
		border-radius: 4px;
		transition: width 0.5s ease;
		box-shadow: 0 0 10px rgba(0, 255, 136, 0.5);
	}
	
	.exp-text {
		position: absolute;
		top: -20px;
		left: 0;
		font-size: 11px;
		color: #00ff88;
		font-weight: bold;
	}
	
	/* Case Section */
	.case-section {
		text-align: center;
	}
	
	.case-label {
		font-size: 10px;
		color: #888;
		margin-bottom: 2px;
		letter-spacing: 1px;
	}
	
	.case-id {
		font-size: 16px;
		color: #00ff88;
		font-weight: bold;
		text-shadow: 0 0 8px rgba(0, 255, 136, 0.5);
	}
	
	/* Status Section */
	.status-section {
		text-align: right;
	}
	
	.status-indicator {
		display: flex;
		align-items: center;
		gap: 8px;
		margin-bottom: 4px;
		font-size: 12px;
		font-weight: bold;
	}
	
	.status-dot {
		width: 8px;
		height: 8px;
		border-radius: 50%;
		animation: pulse 2s infinite;
	}
	
	.status-indicator.online {
		color: #00ff88;
	}
	
	.status-indicator.online .status-dot {
		background: #00ff88;
		box-shadow: 0 0 10px rgba(0, 255, 136, 0.7);
	}
	
	.status-indicator.offline {
		color: #ff4444;
	}
	
	.status-indicator.offline .status-dot {
		background: #ff4444;
		box-shadow: 0 0 10px rgba(255, 68, 68, 0.7);
	}
	
	.system-time {
		font-size: 14px;
		color: #fff;
		font-family: 'Courier New', monospace;
	}
	
	/* Stats Panel */
	.stats-panel {
		display: flex;
		justify-content: center;
		gap: 32px;
		padding: 8px 24px 12px;
		background: rgba(0, 0, 0, 0.2);
	}
	
	.stat-item {
		display: flex;
		align-items: center;
		gap: 8px;
		padding: 8px 16px;
		background: rgba(0, 255, 136, 0.1);
		border: 1px solid rgba(0, 255, 136, 0.3);
		border-radius: 8px;
		transition: all 0.3s ease;
	}
	
	.stat-item:hover {
		background: rgba(0, 255, 136, 0.2);
		border-color: #00ff88;
		transform: translateY(-2px);
		box-shadow: 0 4px 12px rgba(0, 255, 136, 0.3);
	}
	
	.stat-icon {
		font-size: 18px;
	}
	
	.stat-content {
		text-align: center;
	}
	
	.stat-label {
		font-size: 9px;
		color: #888;
		margin-bottom: 2px;
		letter-spacing: 0.5px;
	}
	
	.stat-value {
		font-size: 14px;
		color: #00ff88;
		font-weight: bold;
	}
	
	@keyframes pulse {
		0%, 100% {
			opacity: 1;
		}
		50% {
			opacity: 0.5;
		}
	}
	
	/* Responsive Design */
	@media (max-width: 768px) {
		.hud-top-bar {
			flex-direction: column;
			gap: 12px;
			padding: 16px;
		}
		
		.stats-panel {
			flex-wrap: wrap;
			gap: 16px;
		}
		
		.experience-bar {
			width: 150px;
		}
	}
</style>