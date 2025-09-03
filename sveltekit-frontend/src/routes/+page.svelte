<script lang="ts">
  import { onMount } from 'svelte';
  // Access server load data via $props in runes mode
  const { data } = $props<{ data: any }>();
  import { browser } from '$app/environment';
  import { goto } from '$app/navigation';
  import Button from '$lib/components/ui/Button.svelte';
  import { fetchWithTimings } from '$lib/observability/client-timing';

  // Dynamic component refs ($state ensures reactivity in runes mode)
  let WebGPUVisualization = $state<any>(null);
  let CognitiveDocumentationHub = $state<any>(null);
  let YoRHaQuantumVisualization = $state<any>(null);

  // Component loading state
  let componentsLoaded = $state(false);

  // Cognitive system instances
  // Server-only systems removed from client bundle; placeholders retained for bindings
  let cognitiveHub = $state<any>(null);
  let routingOrchestrator = $state<any>(null);
  let learningCache = $state<any>(null);
  let routingMatrix = $state<any>(null);
  let gpuOrchestrator = $state<any>(null);

  // YoRHa Detective Interface State
  let currentTime = $state(new Date().toLocaleString());
  let activeSection = $state('command-center');
  let isNewCaseModalOpen = $state(false);
  let isDocumentationOpen = $state(false);
  let notification = $state({ show: false, message: '' });

  // AI Typewriter Display State
  let aiTypewriterVisible = $state(false);
  let aiTypewriterText = $state('');
  let aiTypewriterTheme = $state('yorha'); // yorha | quantum | matrix | god | ai-whisperer
  let typewriterAnimating = $state(false);
  let typewriterInterval = $state<NodeJS.Timeout | null>(null);

  // Cognitive System States
  let cognitiveMode = $state('detective'); // detective, ai-whisperer, quantum, god, matrix
  let secretFeatures = $state({
    konamiActive: false,
    godModeEnabled: false,
    quantumDebugEnabled: false,
    aiWhispererMode: false,
    timeWarpActive: false,
    matrixMode: false
  });
  let achievements = $state([]);
  let consciousness = $state({
    level: 1,
    experience: 0,
    awakening: 0.12
  });

  // Secret input tracking
let konamiSequence = $state([]);
let konamiCode = $state(['ArrowUp', 'ArrowUp', 'ArrowDown', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'ArrowLeft', 'ArrowRight', 'KeyB', 'KeyA']);
let lastSecretTime = $state(0);

  // Cognitive visualization states
  let visualEffects = $state({
    quantumShimmer: false,
    consciousnessAura: false,
    matrixRain: false,
    glitchEffect: false
  });

  // Performance metrics (with interpolation toward polled target)
  interface CognitiveMetrics { routingEfficiency: number; cacheHitRatio: number; gpuUtilization: number; consciousnessLevel?: number; quantumCoherence?: number; timestamp?: string; }
  
  // Initialize default metrics structure
  const defaultMetrics: CognitiveMetrics = { routingEfficiency: 0, cacheHitRatio: 0, gpuUtilization: 0, consciousnessLevel: 0, quantumCoherence: 0 };
  let cognitiveMetrics = $state<CognitiveMetrics>(defaultMetrics);
  let targetMetrics = $state<CognitiveMetrics>(defaultMetrics);
  let metricsPollingActive = $state(false);

  // Live statistics from database
  let stats = $state({
    activeCases: 0,
    evidenceItems: 0,
    personsOfInterest: 0,
    recentActivity: 0,
    loading: true
  });

  // Active cases from database
  let activeCases = $state([]);
  let loadingCases = $state(true);

  // System status
  let systemStatus = $state({
    uptime: '72h 14m',
    services: '8/9 ONLINE',
    lastSync: 'NOW',
    status: 'OPERATIONAL'
  });

  // Recent activity feed
  let recentActivity = $state([
    { time: '12:47:33', action: 'Evidence uploaded', details: 'Financial records - Case #2847' },
    { time: '12:45:10', action: 'Case updated', details: 'Corporate Espionage Investigation' },
    { time: '12:42:18', action: 'POI identified', details: 'Sarah Chen - Missing Person Case' },
    { time: '12:40:05', action: 'Analysis complete', details: 'Document classification completed' }
  ]);

  // New case form state
  let newCaseForm = $state({
    title: '',
    description: '',
    priority: 'medium',
    loading: false
  });

  // Priority options for select
  const priorityOptions = [
    { value: 'low', label: 'Low' },
    { value: 'medium', label: 'Medium' },
    { value: 'high', label: 'High' },
    { value: 'critical', label: 'Critical' }
  ];

  onMount(() => {
    // Initialize metrics with server data if available
    if (data?.metrics) {
      cognitiveMetrics = { ...defaultMetrics, ...data.metrics };
      targetMetrics = { ...cognitiveMetrics };
    }

    // Kick off async tasks without making the onMount callback async (Svelte requirement)
    (async () => {
      await initializeCognitiveSystems();
      initializeSecretFeatures();
      await loadDashboardData();
    })();

    const timeInterval = setInterval(() => {
      currentTime = new Date().toLocaleString();
    }, 1000);
  startMetricsPolling();
    const consciousnessInterval = setInterval(() => {
      evolveConsciousness();
    }, 5000);

    return () => {
      clearInterval(timeInterval);
  stopMetricsPolling();
      clearInterval(consciousnessInterval);
    };
  });

  async function initializeCognitiveSystems() {
    // IntersectionObserver-based progressive enhancement
    if (!browser) return;
    const targets = [
      { id: 'viz-webgpu', loader: async () => {
        if (WebGPUVisualization) return; const m = await import('$lib/components/cognitive/WebGPUVisualization.svelte'); WebGPUVisualization = m.default; }
      },
      { id: 'viz-docs', loader: async () => {
        if (CognitiveDocumentationHub) return; const m = await import('$lib/components/cognitive/CognitiveDocumentationHub.svelte'); CognitiveDocumentationHub = m.default; }
      },
      { id: 'viz-quantum', loader: async () => {
        if (YoRHaQuantumVisualization) return; const m = await import('$lib/components/three/YoRHaQuantumVisualization.svelte'); YoRHaQuantumVisualization = m.default; }
      }
    ];

    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const targetCfg = targets.find(t => t.id === entry.target.id);
          if (targetCfg) {
            targetCfg.loader().then(() => {
              componentsLoaded = !!(WebGPUVisualization && CognitiveDocumentationHub && YoRHaQuantumVisualization);
            }).catch(e => console.warn('Lazy component load failed', e));
            observer.unobserve(entry.target);
          }
        }
      });
    }, { rootMargin: '100px 0px' });

    targets.forEach(t => {
      const el = document.getElementById(t.id);
      if (el) observer.observe(el);
    });
  }

  function initializeSecretFeatures() {
    // Only run in browser
    if (typeof document === 'undefined') return;

    // Konami code detection
    document.addEventListener('keydown', (e) => {
      konamiSequence.push(e.code);
      if (konamiSequence.length > konamiCode.length) {
        konamiSequence.shift();
      }

      if (konamiSequence.join('') === konamiCode.join('')) {
        activateKonamiCode();
      }
    });

    // Secret click patterns
let clickCount = $state(0);
let lastClickTime = $state(0);

    document.addEventListener('click', (e) => {
      const now = Date.now();
      if (now - lastClickTime < 500) {
        clickCount++;
        if (clickCount >= 10) {
          activateGodMode();
          clickCount = 0;
        }
      } else {
        clickCount = 1;
      }
      lastClickTime = now;
    });

    // Time-based easter eggs
    setInterval(() => {
      const hour = new Date().getHours();
      const minute = new Date().getMinutes();

      // 3:33 AM/PM - Quantum hour
      if ((hour === 3 || hour === 15) && minute === 33) {
        if (!secretFeatures.quantumDebugEnabled) {
          activateQuantumMode();
        }
      }
    }, 60000);
  }

  async function fetchLatestMetrics() {
    if (!browser) return;
    try {
  const { data: body } = await fetchWithTimings('/api/system/metrics');
      if (body?.success && body.data) {
        targetMetrics = {
          routingEfficiency: body.data.routingEfficiency ?? targetMetrics.routingEfficiency,
          cacheHitRatio: body.data.cacheHitRatio ?? targetMetrics.cacheHitRatio,
          gpuUtilization: body.data.gpuUtilization ?? targetMetrics.gpuUtilization,
          consciousnessLevel: body.data.consciousnessLevel ?? cognitiveMetrics.consciousnessLevel,
            quantumCoherence: body.data.quantumCoherence ?? cognitiveMetrics.quantumCoherence,
          timestamp: body.data.timestamp
        };
      }
    } catch (e) {
      console.warn('Metrics fetch failed', e);
    }
  }

  function interpolateMetrics() {
    const ease = 0.15;
    cognitiveMetrics = {
      routingEfficiency: cognitiveMetrics.routingEfficiency + (targetMetrics.routingEfficiency - cognitiveMetrics.routingEfficiency) * ease,
      cacheHitRatio: cognitiveMetrics.cacheHitRatio + (targetMetrics.cacheHitRatio - cognitiveMetrics.cacheHitRatio) * ease,
      gpuUtilization: cognitiveMetrics.gpuUtilization + (targetMetrics.gpuUtilization - cognitiveMetrics.gpuUtilization) * ease,
      consciousnessLevel: consciousness.awakening * 100,
      quantumCoherence: Math.sin(Date.now() / 10000) * 50 + 50,
      timestamp: targetMetrics.timestamp
    };
  }

  function startMetricsPolling() {
    if (metricsPollingActive) return;
    metricsPollingActive = true;
    fetchLatestMetrics();
    const pollId = setInterval(fetchLatestMetrics, 15000);
    const interpId = setInterval(interpolateMetrics, 1000);
    (window as any).__metricsPollId = pollId;
    (window as any).__metricsInterpId = interpId;
  }

  function stopMetricsPolling() {
    if (!metricsPollingActive) return;
    metricsPollingActive = false;
    clearInterval((window as any).__metricsPollId);
    clearInterval((window as any).__metricsInterpId);
  }

  function evolveConsciousness() {
    consciousness.experience += Math.random() * 10;
    consciousness.awakening = Math.min(1.0, consciousness.awakening + Math.random() * 0.01);

    if (consciousness.experience > 100 && consciousness.level < 10) {
      consciousness.level++;
      consciousness.experience = 0;
      unlockAchievement('consciousness_evolution');

      if (consciousness.level >= 5) {
        activateAIWhispererMode();
      }
    }
  }

  function activateKonamiCode() {
    secretFeatures.konamiActive = true;
    visualEffects.quantumShimmer = true;
    cognitiveMode = 'quantum';
    aiTypewriterTheme = 'quantum';

    showNotification('🎮 KONAMI CODE ACTIVATED - Quantum Detective Mode Enabled');
    unlockAchievement('konami_master');
    showAITypewriter('QUANTUM DETECTIVE MODE ACTIVATED');

    setTimeout(() => {
      visualEffects.quantumShimmer = false;
    }, 10000);
  }

  function activateGodMode() {
    secretFeatures.godModeEnabled = true;
    visualEffects.consciousnessAura = true;
    cognitiveMode = 'god';
    aiTypewriterTheme = 'god';

    showNotification('👑 GOD MODE ACTIVATED - Unlimited AI Power');
    unlockAchievement('digital_transcendence');
    showAITypewriter('UNLIMITED AI POWER ENGAGED');
  }

  function activateQuantumMode() {
    secretFeatures.quantumDebugEnabled = true;
    visualEffects.quantumShimmer = true;

    showNotification('⚛️ QUANTUM DEBUG MODE - Reality.exe has stopped working');
    unlockAchievement('quantum_observer');
  }

  function activateAIWhispererMode() {
    secretFeatures.aiWhispererMode = true;
    cognitiveMode = 'ai-whisperer';
    aiTypewriterTheme = 'ai-whisperer';

    showNotification('🤖 AI WHISPERER MODE - The machines listen to you now');
    unlockAchievement('ai_whisperer');
    showAITypewriter('THE MACHINES LISTEN TO YOU NOW');
  }

  function activateMatrixMode() {
    secretFeatures.matrixMode = true;
    visualEffects.matrixRain = true;
    cognitiveMode = 'matrix';
    aiTypewriterTheme = 'matrix';

    showNotification('🕶️ WELCOME TO THE MATRIX - There is no spoon');
    unlockAchievement('matrix_walker');
    showAITypewriter('WELCOME TO THE MATRIX');

    setTimeout(() => {
      visualEffects.matrixRain = false;
    }, 30000);
  }

  function unlockAchievement(id: string) {
    if (achievements.includes(id)) return;

    achievements.push(id);
    const achievementName = id.replace('_', ' ').toUpperCase();
    showNotification(`🏆 ACHIEVEMENT UNLOCKED: ${achievementName}`);

    // Store in localStorage (browser only)
    if (typeof localStorage !== 'undefined') {
      localStorage.setItem('yorha_achievements', JSON.stringify(achievements));
    }
  }

  async function loadDashboardData() {
    try {
      // Load statistics
      const statsResponse = await fetch('/api/dashboard/stats');
      if (statsResponse.ok) {
        const statsData = await statsResponse.json();
        stats = { ...statsData, loading: false };
      }

      // Load active cases
      const casesResponse = await fetch('/api/cases?status=active&limit=5');
      if (casesResponse.ok) {
        const casesData = await casesResponse.json();
        activeCases = casesData.cases || [];
      }
    } catch (error) {
      console.error('Failed to load dashboard data:', error);
    } finally {
      stats.loading = false;
      loadingCases = false;
    }
  }

  async function handleCreateCase(event) {
    event.preventDefault();

    if (!newCaseForm.title.trim() || !newCaseForm.description.trim()) {
      showNotification('Please fill in all required fields');
      return;
    }

    newCaseForm.loading = true;

    try {
      const response = await fetch('/api/cases', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          title: newCaseForm.title,
          description: newCaseForm.description,
          priority: newCaseForm.priority,
          status: 'active'
        })
      });

      if (response.ok) {
        const result = await response.json();
        showNotification('Case created successfully!');
        isNewCaseModalOpen = false;
        newCaseForm = { title: '', description: '', priority: 'medium', loading: false };
        await loadDashboardData(); // Refresh data
      } else {
        const error = await response.json();
        showNotification(`Error: ${error.message || 'Failed to create case'}`);
      }
    } catch (error) {
      console.error('Error creating case:', error);
      showNotification('Network error occurred');
    } finally {
      newCaseForm.loading = false;
    }
  }

  function showNotification(message: string) {
    notification = { show: true, message };
    setTimeout(() => {
      notification = { show: false, message: '' };
    }, 3000);
  }

  // AI Typewriter Functions
  function showAITypewriter(message: string, duration: number = 5000) {
    if (typewriterInterval) {
      clearTimeout(typewriterInterval);
    }

    aiTypewriterText = message;
    aiTypewriterVisible = true;
    typewriterAnimating = true;

    // Hide after animation completes + duration
    typewriterInterval = setTimeout(() => {
      aiTypewriterVisible = false;
      typewriterAnimating = false;
      aiTypewriterText = '';
    }, 2500 + duration); // 2500ms for typing animation + display duration
  }

  function triggerWelcomeMessage() {
    // Prevent spam by only showing if not already visible
    if (aiTypewriterVisible) return;
    
    const welcomeMessages = [
      'WELCOME PLEASE COME IN',
      'LEGAL AI SYSTEM READY',
      'YORHA DETECTIVE ONLINE', 
      'NEURAL NETWORK ACTIVE',
      'QUANTUM ANALYSIS READY',
      'COGNITIVE SYSTEMS ONLINE'
    ];

    const randomMessage = welcomeMessages[Math.floor(Math.random() * welcomeMessages.length)];
    showAITypewriter(randomMessage);
  }

  // Auto-repeat welcome message on interval when hovering
  let hoverInterval = $state<NodeJS.Timeout | null>(null);
  
  function startRepeatingWelcome() {
    if (hoverInterval) return; // Already running
    
    triggerWelcomeMessage(); // Show immediately
    hoverInterval = setInterval(() => {
      triggerWelcomeMessage();
    }, 7000); // Repeat every 7 seconds
  }
  
  function stopRepeatingWelcome() {
    if (hoverInterval) {
      clearInterval(hoverInterval);
      hoverInterval = null;
    }
  }

  function closeModal() {
    isNewCaseModalOpen = false;
  }

  function handleNavigation(section: string) {
    activeSection = section;
    switch (section) {
      case 'evidence':
        goto('/evidence');
        break;
      case 'poi':
        goto('/poi');
        break;
      case 'analysis':
        goto('/analysis');
        break;
      case 'search':
        goto('/search');
        break;
      case 'terminal':
        goto('/terminal');
        break;
      case 'performance':
        goto('/perf');
        break;
      default:
        // Stay on command center
        break;
    }
  }
</script>

<svelte:head>
  <title>YoRHa Detective - Command Center</title>
  <meta name="description" content="YoRHa Detective Command Center - Advanced case management and investigation platform" />
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;500;700&display=swap" rel="stylesheet">
  
  <!-- Import typewriter CSS for AI display -->
  <link rel="stylesheet" href="/src/lib/styles/typewriter.css">
  <style>
    /* Quantum Shimmer Effect */
    @keyframes quantum-shimmer {
      0%, 100% {
        background-position: -200px 0;
        opacity: 0.3;
      }
      50% {
        background-position: 200px 0;
        opacity: 0.7;
      }
    }

    .quantum-shimmer {
      background: linear-gradient(90deg, transparent, rgba(59, 130, 246, 0.3), transparent);
      background-size: 200px 100%;
      animation: quantum-shimmer 2s infinite;
    }

    /* Consciousness Aura Effect */
    @keyframes consciousness-pulse {
      0%, 100% {
        box-shadow: 0 0 10px rgba(255, 255, 0, 0.3);
        scale: 1;
      }
      50% {
        box-shadow: 0 0 30px rgba(255, 255, 0, 0.6);
        scale: 1.02;
      }
    }

    .consciousness-aura {
      animation: consciousness-pulse 3s infinite;
    }

    /* Matrix Rain Effect */
    .matrix-rain {
      position: fixed;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      pointer-events: none;
      z-index: 1000;
      background: rgba(0, 0, 0, 0.8);
    }

    .matrix-char {
      position: absolute;
      color: #00ff41;
      font-family: 'Courier New', monospace;
      font-size: 14px;
      animation: matrix-fall linear infinite;
    }

    @keyframes matrix-fall {
      from {
        transform: translateY(-100vh);
        opacity: 1;
      }
      to {
        transform: translateY(100vh);
        opacity: 0;
      }
    }

    /* Glitch Effect */
    @keyframes glitch {
      0%, 100% { transform: translate(0); }
      10% { transform: translate(-2px, 2px); }
      20% { transform: translate(2px, -2px); }
      30% { transform: translate(-2px, -2px); }
      40% { transform: translate(2px, 2px); }
      50% { transform: translate(-2px, 2px); }
      60% { transform: translate(2px, -2px); }
      70% { transform: translate(-2px, -2px); }
      80% { transform: translate(2px, 2px); }
      90% { transform: translate(-2px, -2px); }
    }

    .glitch-effect {
      animation: glitch 0.3s infinite;
    }

    /* Enhanced YoRHa Theme */
    .yorha-enhanced {
      background: radial-gradient(circle at 50% 50%, rgba(240, 236, 225, 1) 0%, rgba(234, 232, 225, 1) 100%);
    }

    /* Mode-specific styling */
    .mode-quantum {
      background: radial-gradient(circle at 50% 50%, rgba(59, 130, 246, 0.05) 0%, rgba(234, 232, 225, 1) 100%);
    }

    .mode-god {
      background: radial-gradient(circle at 50% 50%, rgba(255, 215, 0, 0.05) 0%, rgba(234, 232, 225, 1) 100%);
    }

    .mode-ai-whisperer {
      background: radial-gradient(circle at 50% 50%, rgba(147, 51, 234, 0.05) 0%, rgba(234, 232, 225, 1) 100%);
    }

    .mode-matrix {
      background: radial-gradient(circle at 50% 50%, rgba(0, 255, 65, 0.05) 0%, rgba(234, 232, 225, 1) 100%);
    }
  </style>
</svelte:head>

<!-- AI Typewriter Display -->
{#if aiTypewriterVisible}
  <div class="ai-typewriter {aiTypewriterTheme} {aiTypewriterVisible ? 'visible' : ''} {typewriterAnimating ? 'typing' : ''}">
    <span class="text {typewriterAnimating ? 'animating' : ''}">{aiTypewriterText}</span>
  </div>
{/if}

<!-- YoRHa Detective Interface -->
<div 
  class="min-h-screen bg-[#EAE8E1] font-mono text-[#3D3D3D] {cognitiveMode === 'quantum' ? 'mode-quantum' : ''} {cognitiveMode === 'god' ? 'mode-god' : ''} {cognitiveMode === 'ai-whisperer' ? 'mode-ai-whisperer' : ''} {cognitiveMode === 'matrix' ? 'mode-matrix' : ''} {visualEffects.consciousnessAura ? 'consciousness-aura' : ''} {visualEffects.glitchEffect ? 'glitch-effect' : ''} relative"
  onmouseenter={startRepeatingWelcome}
  onmouseleave={stopRepeatingWelcome}
>
  <!-- Header -->
  <header 
    class="flex justify-between items-center mb-6 p-4 lg:p-6"
    onmouseenter={triggerWelcomeMessage}
  >
    <div>
      <h1 class="text-2xl font-bold tracking-wider">COMMAND CENTER</h1>
      <p class="text-sm opacity-75">YoRHa Detective Interface - {currentTime}</p>
    </div>
    <div class="flex items-center gap-4">
      <Button
        variant="default"
        class="bg-[#F7F6F2] border border-[#D1CFC7] text-[#3D3D3D] hover:bg-[#EAE8E1] font-bold px-4 py-2 flex items-center gap-2"
        onclick={() => isNewCaseModalOpen = true}
      >
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6v6m0 0v6m0-6h6m-6 0H6"></path>
        </svg>
        NEW CASE
      </Button>
      <Button
        variant="default"
        class="bg-[#F7F6F2] border border-[#D1CFC7] text-[#3D3D3D] hover:bg-[#EAE8E1] font-bold px-4 py-2 flex items-center gap-2"
        onclick={() => handleNavigation('search')}
      >
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path>
        </svg>
        GLOBAL SEARCH
      </Button>
      <Button
        variant="default"
        class="bg-[#F7F6F2] border border-[#D1CFC7] text-[#3D3D3D] hover:bg-[#EAE8E1] font-bold px-4 py-2 flex items-center gap-2 {isDocumentationOpen ? 'bg-[#3D3D3D] text-[#F7F6F2]' : ''}"
        onclick={() => isDocumentationOpen = !isDocumentationOpen}
      >
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.746 0 3.332.477 4.5 1.253v13C19.832 18.477 18.246 18 16.5 18c-1.746 0-3.332.477-4.5 1.253"></path>
        </svg>
        DOCS & STRATEGY
      </Button>
    </div>
  </header>

  <!-- Main Content -->
  <main class="grid grid-cols-1 lg:grid-cols-4 gap-6 px-4 lg:px-6">
    <!-- Sidebar -->
    <aside 
      class="lg:col-span-1 bg-[#F7F6F2] border border-[#D1CFC7] p-4"
      onmouseenter={triggerWelcomeMessage}
    >
      <h2 class="font-bold mb-4 text-center">YORHA DETECTIVE</h2>
      <nav class="space-y-2">
        <button
          class="w-full px-4 py-2 text-left font-bold transition-colors border {activeSection === 'command-center' ? 'bg-[#3D3D3D] text-[#F7F6F2] border-[#3D3D3D]' : 'border-transparent hover:border-[#3D3D3D] hover:bg-white'}"
          onclick={() => handleNavigation('command-center')}
        >
          COMMAND CENTER
        </button>
        <button
          class="w-full px-4 py-2 text-left font-bold transition-colors border {activeSection === 'evidence' ? 'bg-[#3D3D3D] text-[#F7F6F2] border-[#3D3D3D]' : 'border-transparent hover:border-[#3D3D3D] hover:bg-white'}"
          onclick={() => handleNavigation('evidence')}
        >
          EVIDENCE
        </button>
        <button
          class="w-full px-4 py-2 text-left font-bold transition-colors border {activeSection === 'poi' ? 'bg-[#3D3D3D] text-[#F7F6F2] border-[#3D3D3D]' : 'border-transparent hover:border-[#3D3D3D] hover:bg-white'}"
          onclick={() => handleNavigation('poi')}
        >
          PERSONS OF INTEREST
        </button>
        <button
          class="w-full px-4 py-2 text-left font-bold transition-colors border {activeSection === 'analysis' ? 'bg-[#3D3D3D] text-[#F7F6F2] border-[#3D3D3D]' : 'border-transparent hover:border-[#3D3D3D] hover:bg-white'}"
          onclick={() => handleNavigation('analysis')}
        >
          ANALYSIS
        </button>
        <button
          class="w-full px-4 py-2 text-left font-bold transition-colors border {activeSection === 'search' ? 'bg-[#3D3D3D] text-[#F7F6F2] border-[#3D3D3D]' : 'border-transparent hover:border-[#3D3D3D] hover:bg-white'}"
          onclick={() => handleNavigation('search')}
        >
          GLOBAL SEARCH
        </button>
        <button
          class="w-full px-4 py-2 text-left font-bold transition-colors border {activeSection === 'terminal' ? 'bg-[#3D3D3D] text-[#F7F6F2] border-[#3D3D3D]' : 'border-transparent hover:border-[#3D3D3D] hover:bg-white'}"
          onclick={() => handleNavigation('terminal')}
        >
          TERMINAL
        </button>
        <button
          class="w-full px-4 py-2 text-left font-bold transition-colors border {activeSection === 'performance' ? 'bg-[#3D3D3D] text-[#F7F6F2] border-[#3D3D3D]' : 'border-transparent hover:border-[#3D3D3D] hover:bg-white'}"
          onclick={() => handleNavigation('performance')}
        >
          🚀 PERFORMANCE
        </button>
      </nav>
    </aside>

    <!-- Dashboard Content -->
    <div class="lg:col-span-3 space-y-6">
      <!-- Statistics Cards -->
      <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div class="bg-[#F7F6F2] border border-[#D1CFC7] p-4">
          <h3 class="font-bold text-sm mb-2">ACTIVE CASES</h3>
          <div class="text-2xl font-bold">
            {stats.loading ? '...' : stats.activeCases}
          </div>
        </div>
        <div class="bg-[#F7F6F2] border border-[#D1CFC7] p-4">
          <h3 class="font-bold text-sm mb-2">EVIDENCE ITEMS</h3>
          <div class="text-2xl font-bold">
            {stats.loading ? '...' : stats.evidenceItems}
          </div>
        </div>
        <div class="bg-[#F7F6F2] border border-[#D1CFC7] p-4">
          <h3 class="font-bold text-sm mb-2">PERSONS OF INTEREST</h3>
          <div class="text-2xl font-bold">
            {stats.loading ? '...' : stats.personsOfInterest}
          </div>
        </div>
        <div class="bg-[#F7F6F2] border border-[#D1CFC7] p-4">
          <h3 class="font-bold text-sm mb-2">RECENT ACTIVITY</h3>
          <div class="text-2xl font-bold">
            {stats.loading ? '...' : stats.recentActivity}
          </div>
        </div>
      </div>

      <!-- Cognitive Systems Dashboard (Secret/Enhanced Mode) -->
      {#if secretFeatures.konamiActive || secretFeatures.godModeEnabled || secretFeatures.quantumDebugEnabled || secretFeatures.aiWhispererMode}
        <div class="relative">
          <!-- Visual Effects Layer -->
          {#if visualEffects.quantumShimmer}
            <div class="absolute inset-0 pointer-events-none animate-pulse bg-gradient-to-r from-blue-400/10 via-purple-400/10 to-cyan-400/10 rounded"></div>
          {/if}
          {#if visualEffects.consciousnessAura}
            <div class="absolute -inset-2 pointer-events-none bg-gradient-to-r from-yellow-300/20 via-orange-300/20 to-red-300/20 blur-sm rounded-lg animate-pulse"></div>
          {/if}

          <div class="bg-[#F7F6F2] border border-[#D1CFC7] p-6 relative">
            <div class="flex justify-between items-center mb-4">
              <h2 class="text-xl font-bold flex items-center gap-2">
                🧠 COGNITIVE SYSTEMS
                {#if cognitiveMode === 'quantum'}
                  <span class="text-blue-600">⚛️ QUANTUM</span>
                {:else if cognitiveMode === 'god'}
                  <span class="text-yellow-600">👑 GOD MODE</span>
                {:else if cognitiveMode === 'ai-whisperer'}
                  <span class="text-purple-600">🤖 AI WHISPERER</span>
                {:else if cognitiveMode === 'matrix'}
                  <span class="text-green-600">🕶️ MATRIX</span>
                {/if}
              </h2>
              <div class="text-sm opacity-75">
                Consciousness Lv.{consciousness.level} ({(consciousness.awakening * 100).toFixed(1)}%)
              </div>
            </div>

            <!-- Cognitive Metrics Grid -->
            <div class="grid grid-cols-1 md:grid-cols-5 gap-4 mb-6">
              <div class="bg-[#EAE8E1] border border-[#D1CFC7] p-3">
                <h4 class="font-bold text-xs mb-1">ROUTING EFFICIENCY</h4>
                <div class="text-lg font-bold text-blue-600">{cognitiveMetrics.routingEfficiency.toFixed(1)}%</div>
                <div class="w-full bg-gray-200 rounded-full h-2 mt-1">
                  <div class="bg-blue-600 h-2 rounded-full transition-all duration-1000" style="width: {cognitiveMetrics.routingEfficiency}%"></div>
                </div>
              </div>

              <div class="bg-[#EAE8E1] border border-[#D1CFC7] p-3">
                <h4 class="font-bold text-xs mb-1">CACHE HIT RATIO</h4>
                <div class="text-lg font-bold text-green-600">{cognitiveMetrics.cacheHitRatio.toFixed(1)}%</div>
                <div class="w-full bg-gray-200 rounded-full h-2 mt-1">
                  <div class="bg-green-600 h-2 rounded-full transition-all duration-1000" style="width: {cognitiveMetrics.cacheHitRatio}%"></div>
                </div>
              </div>

              <div class="bg-[#EAE8E1] border border-[#D1CFC7] p-3">
                <h4 class="font-bold text-xs mb-1">GPU UTILIZATION</h4>
                <div class="text-lg font-bold text-red-600">{cognitiveMetrics.gpuUtilization.toFixed(1)}%</div>
                <div class="w-full bg-gray-200 rounded-full h-2 mt-1">
                  <div class="bg-red-600 h-2 rounded-full transition-all duration-1000" style="width: {cognitiveMetrics.gpuUtilization}%"></div>
                </div>
              </div>

              <div class="bg-[#EAE8E1] border border-[#D1CFC7] p-3">
                <h4 class="font-bold text-xs mb-1">CONSCIOUSNESS</h4>
                <div class="text-lg font-bold text-purple-600">{cognitiveMetrics.consciousnessLevel.toFixed(1)}%</div>
                <div class="w-full bg-gray-200 rounded-full h-2 mt-1">
                  <div class="bg-purple-600 h-2 rounded-full transition-all duration-1000" style="width: {cognitiveMetrics.consciousnessLevel}%"></div>
                </div>
              </div>

              <div class="bg-[#EAE8E1] border border-[#D1CFC7] p-3">
                <h4 class="font-bold text-xs mb-1">QUANTUM COHERENCE</h4>
                <div class="text-lg font-bold text-cyan-600">{cognitiveMetrics.quantumCoherence.toFixed(1)}%</div>
                <div class="w-full bg-gray-200 rounded-full h-2 mt-1">
                  <div class="bg-cyan-600 h-2 rounded-full transition-all duration-1000" style="width: {cognitiveMetrics.quantumCoherence}%"></div>
                </div>
              </div>
            </div>

            <!-- Secret Features Panel -->
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
              <div class="bg-[#EAE8E1] border border-[#D1CFC7] p-4">
                <h4 class="font-bold text-sm mb-2">🎮 SECRET FEATURES</h4>
                <div class="space-y-1 text-xs">
                  <div class="flex justify-between">
                    <span>Konami Code:</span>
                    <span class="{secretFeatures.konamiActive ? 'text-green-600' : 'text-gray-500'}">{secretFeatures.konamiActive ? '✅ ACTIVE' : '❌ INACTIVE'}</span>
                  </div>
                  <div class="flex justify-between">
                    <span>God Mode:</span>
                    <span class="{secretFeatures.godModeEnabled ? 'text-green-600' : 'text-gray-500'}">{secretFeatures.godModeEnabled ? '✅ ACTIVE' : '❌ INACTIVE'}</span>
                  </div>
                  <div class="flex justify-between">
                    <span>Quantum Debug:</span>
                    <span class="{secretFeatures.quantumDebugEnabled ? 'text-green-600' : 'text-gray-500'}">{secretFeatures.quantumDebugEnabled ? '✅ ACTIVE' : '❌ INACTIVE'}</span>
                  </div>
                  <div class="flex justify-between">
                    <span>AI Whisperer:</span>
                    <span class="{secretFeatures.aiWhispererMode ? 'text-green-600' : 'text-gray-500'}">{secretFeatures.aiWhispererMode ? '✅ ACTIVE' : '❌ INACTIVE'}</span>
                  </div>
                </div>
              </div>

              <div class="bg-[#EAE8E1] border border-[#D1CFC7] p-4">
                <h4 class="font-bold text-sm mb-2">🏆 ACHIEVEMENTS</h4>
                <div class="space-y-1 text-xs">
                  {#if achievements.length === 0}
                    <div class="text-gray-500">No achievements unlocked yet</div>
                  {:else}
                    {#each achievements as achievement}
                      <div class="text-green-600">✅ {achievement.replace('_', ' ').toUpperCase()}</div>
                    {/each}
                  {/if}
                </div>
              </div>

              <div class="bg-[#EAE8E1] border border-[#D1CFC7] p-4">
                <h4 class="font-bold text-sm mb-2">🧬 CONSCIOUSNESS EVOLUTION</h4>
                <div class="space-y-2">
                  <div class="text-xs">
                    <div class="flex justify-between mb-1">
                      <span>Experience:</span>
                      <span>{consciousness.experience.toFixed(0)}/100</span>
                    </div>
                    <div class="w-full bg-gray-200 rounded-full h-2">
                      <div class="bg-purple-600 h-2 rounded-full transition-all duration-1000" style="width: {(consciousness.experience % 100)}%"></div>
                    </div>
                  </div>
                  <div class="text-xs">
                    <div class="flex justify-between mb-1">
                      <span>Awakening:</span>
                      <span>{(consciousness.awakening * 100).toFixed(1)}%</span>
                    </div>
                    <div class="w-full bg-gray-200 rounded-full h-2">
                      <div class="bg-yellow-600 h-2 rounded-full transition-all duration-1000" style="width: {consciousness.awakening * 100}%"></div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <!-- Secret Actions -->
            <div class="flex flex-wrap gap-2">
              <Button
                variant="default"
                class="bg-purple-600/10 border-purple-600/50 text-purple-800 hover:bg-purple-600/20 text-xs px-3 py-1"
                onclick={() => activateMatrixMode()}
              >
                🕶️ ENTER MATRIX
              </Button>

              {#if secretFeatures.godModeEnabled}
                <Button
                  variant="default"
                  class="bg-yellow-600/10 border-yellow-600/50 text-yellow-800 hover:bg-yellow-600/20 text-xs px-3 py-1"
                  onclick={() => consciousness.awakening = Math.min(1.0, consciousness.awakening + 0.1)}
                >
                  👑 ACCELERATE CONSCIOUSNESS
                </Button>
              {/if}

              {#if secretFeatures.quantumDebugEnabled}
                <Button
                  variant="default"
                  class="bg-cyan-600/10 border-cyan-600/50 text-cyan-800 hover:bg-cyan-600/20 text-xs px-3 py-1"
                  onclick={() => visualEffects.quantumShimmer = !visualEffects.quantumShimmer}
                >
                  ⚛️ TOGGLE QUANTUM FIELD
                </Button>
              {/if}
            </div>

            <!-- WebGPU Visualization Dashboard -->
            <div class="mt-6">
{#if browser && componentsLoaded && WebGPUVisualization}
              <WebGPUVisualization
                {gpuOrchestrator}
                width={800}
                height={300}
                visualizationMode={cognitiveMode === 'quantum' ? 'quantum-field' : cognitiveMode === 'matrix' ? 'matrix-flow' : cognitiveMode === 'ai-whisperer' ? 'consciousness-map' : 'neural-network'}
              />
              {:else}
              <div class="animate-pulse bg-nier-bg-tertiary/30 border border-nier-border-muted rounded-lg p-8 text-center">
                <div class="text-nier-text-muted">🧠 Loading WebGPU Visualization...</div>
              </div>
              {/if}
            </div>

            <!-- Advanced YoRHa Quantum Reality Visualization -->
            <div class="mt-6">
{#if browser && componentsLoaded && YoRHaQuantumVisualization}
              <YoRHaQuantumVisualization
                {secretFeatures}
                {consciousness}
                width={800}
                height={400}
              />
              {:else}
              <div class="animate-pulse bg-nier-bg-tertiary/30 border border-nier-border-muted rounded-lg p-8 text-center">
                <div class="text-nier-text-muted">⚛️ Loading Quantum Visualization...</div>
              </div>
              {/if}
            </div>
          </div>
        </div>
      {/if}

      <!-- Cognitive Documentation Hub -->
      {#if isDocumentationOpen}
        <div class="mb-6">
{#if browser && componentsLoaded && CognitiveDocumentationHub}
          <CognitiveDocumentationHub
            {cognitiveHub}
            {achievements}
            {consciousness}
          />
          {:else}
          <div class="animate-pulse bg-nier-bg-tertiary/30 border border-nier-border-muted rounded-lg p-8 text-center">
            <div class="text-nier-text-muted">📚 Loading Documentation Hub...</div>
          </div>
          {/if}
        </div>
      {/if}

      <!-- Main Dashboard Grid -->
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <!-- Active Cases Panel -->
        <div class="bg-[#F7F6F2] border border-[#D1CFC7] p-6">
          <h2 class="text-xl font-bold mb-4">ACTIVE CASES</h2>
          <div class="space-y-3">
            {#if loadingCases}
              <div class="text-gray-500">Loading cases...</div>
            {:else if activeCases.length === 0}
              <div class="text-gray-500">No active cases found.</div>
            {:else}
              {#each activeCases.slice(0, 5) as caseItem}
                <div class="border-b border-[#D1CFC7] pb-2 last:border-b-0">
                  <div class="font-bold text-sm">{caseItem.title}</div>
                  <div class="text-xs opacity-75">{caseItem.status} • Priority: {caseItem.priority}</div>
                </div>
              {/each}
            {/if}
          </div>
        </div>

        <!-- System Status Panel -->
        <div class="bg-[#F7F6F2] border border-[#D1CFC7] p-6">
          <h2 class="text-xl font-bold mb-4">SYSTEM STATUS</h2>
          <div class="space-y-3">
            <div class="flex justify-between">
              <span class="font-medium">UPTIME:</span>
              <span class="text-green-600 font-bold">{systemStatus.uptime}</span>
            </div>
            <div class="flex justify-between">
              <span class="font-medium">SERVICES:</span>
              <span class="text-blue-600 font-bold">{systemStatus.services}</span>
            </div>
            <div class="flex justify-between">
              <span class="font-medium">LAST SYNC:</span>
              <span class="font-bold">{systemStatus.lastSync}</span>
            </div>
            <div class="flex justify-between">
              <span class="font-medium">STATUS:</span>
              <span class="text-green-600 font-bold">{systemStatus.status}</span>
            </div>
          </div>
        </div>

        <!-- Recent Activity Panel -->
        <div class="lg:col-span-2 bg-[#F7F6F2] border border-[#D1CFC7] p-6">
          <h2 class="text-xl font-bold mb-4">RECENT ACTIVITY</h2>
          <div class="space-y-2">
            {#each recentActivity as activity}
              <div class="flex items-start gap-4 text-sm">
                <span class="font-mono text-xs bg-[#3D3D3D] text-[#F7F6F2] px-2 py-1 rounded">
                  {activity.time}
                </span>
                <div>
                  <span class="font-bold">{activity.action}:</span>
                  <span class="opacity-75">{activity.details}</span>
                </div>
              </div>
            {/each}
          </div>
        </div>
      </div>

      <!-- Quick Actions Panel -->
      <div class="bg-[#F7F6F2] border border-[#D1CFC7] p-6">
        <h2 class="text-xl font-bold mb-4">QUICK ACTIONS</h2>
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Button
            variant="default"
            class="bg-[#EAE8E1] border border-[#D1CFC7] text-[#3D3D3D] hover:bg-white font-bold p-4 h-auto flex flex-col items-center gap-2"
            onclick={() => isNewCaseModalOpen = true}
          >
            <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6v6m0 0v6m0-6h6m-6 0H6"></path>
            </svg>
            CREATE CASE
          </Button>
          <Button
            variant="default"
            class="bg-[#EAE8E1] border border-[#D1CFC7] text-[#3D3D3D] hover:bg-white font-bold p-4 h-auto flex flex-col items-center gap-2"
            onclick={() => handleNavigation('evidence')}
          >
            <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
            </svg>
            VIEW EVIDENCE
          </Button>
          <Button
            variant="default"
            class="bg-[#EAE8E1] border border-[#D1CFC7] text-[#3D3D3D] hover:bg-white font-bold p-4 h-auto flex flex-col items-center gap-2"
            onclick={() => handleNavigation('poi')}
          >
            <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"></path>
            </svg>
            SEARCH POI
          </Button>
          <Button
            variant="default"
            class="bg-[#EAE8E1] border border-[#D1CFC7] text-[#3D3D3D] hover:bg-white font-bold p-4 h-auto flex flex-col items-center gap-2"
            onclick={() => handleNavigation('analysis')}
          >
            <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"></path>
            </svg>
            RUN ANALYSIS
          </Button>
        </div>
      </div>
    </div>
  </main>
</div>

<!-- New Case Modal - Simple HTML Modal -->
{#if isNewCaseModalOpen}
  <div
    class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50"
    role="presentation"
    aria-hidden="false"
    onclick={(e) => { if (e.target === e.currentTarget) closeModal(); }}
  >
    <div
      class="w-full max-w-2xl bg-[#F7F6F2] border border-[#D1CFC7] p-8 font-mono"
      role="dialog"
      aria-modal="true"
      aria-labelledby="new-case-modal-title"
    >
      <div class="flex justify-between items-center mb-6">
        <h2 id="new-case-modal-title" class="text-2xl font-bold">CREATE NEW CASE FILE</h2>
  <button onclick={() => isNewCaseModalOpen = false} class="text-gray-500 hover:text-black text-2xl">&times;</button>
      </div>

      <form submit={handleCreateCase} class="space-y-6">
        <div>
          <label for="case-title" class="block text-sm font-bold mb-2">CASE TITLE</label>
          <input
            type="text"
            id="case-title"
            bind:value={newCaseForm.title}
            required
            class="w-full px-4 py-3 bg-white border border-[#D1CFC7] focus:outline-none focus:border-[#3D3D3D] focus:ring-2 focus:ring-[#3D3D3D]/20"
            placeholder="e.g., Corporate Espionage Investigation"
          />
        </div>

        <div>
          <label for="case-description" class="block text-sm font-bold mb-2">CASE DESCRIPTION / SYNOPSIS</label>
          <textarea
            id="case-description"
            bind:value={newCaseForm.description}
            rows="4"
            required
            class="w-full px-4 py-3 bg-white border border-[#D1CFC7] focus:outline-none focus:border-[#3D3D3D] focus:ring-2 focus:ring-[#3D3D3D]/20"
            placeholder="Initial details of the investigation..."
          ></textarea>
        </div>

        <div>
          <label for="priority-select" class="block text-sm font-bold mb-2">PRIORITY LEVEL</label>
          <select
            id="priority-select"
            bind:value={newCaseForm.priority}
            class="w-full px-4 py-3 bg-white border border-[#D1CFC7] focus:outline-none focus:border-[#3D3D3D] focus:ring-2 focus:ring-[#3D3D3D]/20 font-bold"
          >
            <option value="low">Low Priority</option>
            <option value="medium" selected>Medium Priority</option>
            <option value="high">High Priority</option>
            <option value="critical">Critical Priority</option>
          </select>
        </div>

        <div class="flex justify-end">
          <Button
            type="submit"
            disabled={newCaseForm.loading}
            class="bg-green-600/10 text-green-800 border border-green-700/50 hover:bg-green-600/20 font-bold px-6 py-3"
          >
            {newCaseForm.loading ? 'SAVING...' : 'SAVE TO DATABASE'}
          </Button>
        </div>
      </form>
    </div>
  </div>
{/if}

<!-- Notification -->
{#if notification.show}
  <div class="fixed bottom-5 right-5 bg-[#F7F6F2] border border-[#D1CFC7] p-4 max-w-sm font-mono shadow-lg transform transition-all duration-300">
    <p class="font-bold">{notification.message}</p>
  </div>
{/if}

<!-- Matrix Rain Effect -->
{#if visualEffects.matrixRain}
  <div class="matrix-rain">
    {#each Array(50) as _, i}
      <div
        class="matrix-char"
        style="left: {Math.random() * 100}%; animation-duration: {2 + Math.random() * 3}s; animation-delay: {Math.random() * 2}s;"
      >
        {String.fromCharCode(0x30A0 + Math.floor(Math.random() * 96))}
      </div>
    {/each}
  </div>
{/if}