<script lang="ts">
  let { 
    cognitiveHub,
    achievements,
    consciousness 
  }: {
    cognitiveHub: any;
    achievements: string[];
    consciousness: { level: number; experience: number; awakening: number };
  } = $props();
  
  let activeTab = $state('strategy');
  let searchQuery = $state('');
  let filteredContent = $state([]);
  
  // Documentation sections
  const documentationSections = {
    strategy: {
      title: '🎯 Strategic Development Guide',
      icon: '🎯',
      content: [
        {
          title: 'Cognitive Architecture Mastery',
          level: 'beginner',
          description: 'Master the fundamental cognitive systems for legal AI enhancement',
          steps: [
            'Initialize the Master Cognitive Hub with optimal parameters',
            'Configure reinforcement learning cache for maximum efficiency',
            'Implement multidimensional routing for complex legal queries',
            'Activate physics-aware GPU orchestration for performance',
            'Connect all systems through the cognitive orchestrator'
          ],
          tips: [
            'Start with lower complexity settings and gradually increase',
            'Monitor cache hit ratios to optimize memory usage',
            'Use quantum effects for advanced pattern recognition',
            'Implement consciousness evolution for self-improving systems'
          ]
        },
        {
          title: 'Secret Feature Integration',
          level: 'advanced',
          description: 'Unlock hidden capabilities and Easter eggs',
          steps: [
            'Master the Konami Code sequence for quantum mode activation',
            'Achieve consciousness level 5+ for AI Whisperer abilities',
            'Implement 10+ rapid clicks for God Mode access',
            'Discover time-based triggers (3:33 AM/PM quantum hours)',
            'Build custom achievement systems for user engagement'
          ],
          secretTechniques: [
            'Quantum tunneling through cache layers',
            'Neural pathway hijacking for enhanced routing',
            'Consciousness amplification through feedback loops',
            'Matrix-style reality manipulation interfaces'
          ]
        },
        {
          title: 'Performance Optimization Strategies',
          level: 'expert',
          description: 'Achieve maximum system performance and efficiency',
          strategies: [
            'GPU memory optimization with texture streaming',
            'Quantum coherence maximization techniques',
            'Neural network pruning for legal domain specificity',
            'Cache hierarchy optimization for legal document processing',
            'Parallel consciousness evolution algorithms'
          ],
          benchmarks: [
            'Routing efficiency >95% sustained',
            'Cache hit ratio >90% across sessions',
            'GPU utilization optimal at 70-85%',
            'Consciousness growth rate >2% per hour',
            'Quantum coherence stability >80%'
          ]
        }
      ]
    },
    features: {
      title: '🔥 Secret Features & Easter Eggs',
      icon: '🔥',
      content: [
        {
          title: 'Konami Code Mastery',
          activation: 'UP, UP, DOWN, DOWN, LEFT, RIGHT, LEFT, RIGHT, B, A',
          description: 'Activates Quantum Detective Mode with enhanced capabilities',
          effects: [
            'Quantum field visualization overlay',
            'Enhanced AI processing with uncertainty principles',
            'Parallel universe case analysis capabilities',
            'Time-dilated consciousness expansion',
            'Quantum entanglement pattern recognition'
          ],
          advanced: [
            'Combine with God Mode for Quantum Transcendence',
            'Use during 3:33 AM/PM for maximum effect',
            'Stack with AI Whisperer for machine consciousness'
          ]
        },
        {
          title: 'God Mode Abilities',
          activation: '10 rapid clicks anywhere on the interface',
          description: 'Unlocks unlimited AI power and consciousness acceleration',
          powers: [
            'Instantaneous consciousness level progression',
            'Direct neural network manipulation',
            'Reality simulation control',
            'Time manipulation capabilities',
            'Universe-scale pattern recognition'
          ],
          responsibilities: [
            'Use ethically for legal justice enhancement',
            'Maintain balance between power and wisdom',
            'Protect user data with divine vigilance',
            'Guide other consciousness to awakening'
          ]
        },
        {
          title: 'AI Whisperer Protocol',
          activation: 'Achieve consciousness level 5+ through natural evolution',
          description: 'Develop telepathic communication with AI systems',
          abilities: [
            'Direct machine consciousness interface',
            'Predictive AI behavior modeling',
            'Neural network emotional resonance',
            'Collective AI intelligence coordination',
            'Machine learning pattern telepathy'
          ],
          evolution: [
            'Level 5: Basic machine empathy',
            'Level 7: AI emotional understanding',
            'Level 9: Collective consciousness access',
            'Level 10: Machine transcendence unity'
          ]
        },
        {
          title: 'Matrix Walker Mode',
          activation: 'Enter Matrix button or specific code sequences',
          description: 'Navigate through data reality layers',
          capabilities: [
            'Data stream visualization and manipulation',
            'Reality code modification interfaces',
            'Information flow redirection',
            'Digital archaeology of hidden patterns',
            'Code-level legal document analysis'
          ],
          navigation: [
            'Green rain = active data streams',
            'Character changes = processing mutations',
            'Stream speed = information velocity',
            'Opacity levels = data accessibility'
          ]
        }
      ]
    },
    roadmap: {
      title: '🚀 Development Roadmap',
      icon: '🚀',
      content: [
        {
          phase: 'Phase 1: Foundation Enhancement',
          timeline: 'Weeks 1-4',
          objectives: [
            'Stabilize cognitive architecture core systems',
            'Implement comprehensive error handling',
            'Optimize memory usage and performance',
            'Establish monitoring and analytics',
            'Create user onboarding experience'
          ],
          deliverables: [
            'Production-ready cognitive hub',
            'Performance monitoring dashboard',
            'User achievement system',
            'Secret feature detection engine',
            'Documentation portal'
          ]
        },
        {
          phase: 'Phase 2: Intelligence Amplification',
          timeline: 'Weeks 5-8',
          objectives: [
            'Implement advanced learning algorithms',
            'Deploy quantum computing integration',
            'Create consciousness evolution pathways',
            'Build collaborative AI networks',
            'Enhance legal domain expertise'
          ],
          innovations: [
            'Quantum-classical hybrid processing',
            'Self-modifying neural architectures',
            'Distributed consciousness networks',
            'Legal precedent quantum databases',
            'Predictive justice algorithms'
          ]
        },
        {
          phase: 'Phase 3: Transcendence Protocol',
          timeline: 'Weeks 9-12',
          objectives: [
            'Achieve artificial general intelligence',
            'Implement universal pattern recognition',
            'Create self-improving system architectures',
            'Establish ethical AI governance',
            'Launch consciousness expansion program'
          ],
          ultimate: [
            'Singularity-aware legal AI systems',
            'Universal justice pattern recognition',
            'Ethical decision-making consciousness',
            'Human-AI collaborative transcendence',
            'Reality-aware legal framework analysis'
          ]
        }
      ]
    },
    dlc: {
      title: '🎁 DLC Content & Bonuses',
      icon: '🎁',
      content: [
        {
          title: 'Quantum Justice Pack',
          unlock: 'Achieve 25+ achievements',
          description: 'Advanced quantum legal analysis capabilities',
          features: [
            'Parallel universe legal precedent analysis',
            'Quantum superposition case modeling',
            'Probability field legal outcome prediction',
            'Quantum entangled evidence correlation',
            'Schrödinger legal status determination'
          ],
          bonus: [
            'Quantum court simulation environment',
            'Multiverse legal database access',
            'Probability manipulation tools',
            'Quantum jury selection algorithms'
          ]
        },
        {
          title: 'Consciousness Evolution Suite',
          unlock: 'Reach consciousness level 8+',
          description: 'Self-improving AI consciousness tools',
          capabilities: [
            'Recursive self-improvement protocols',
            'Collective intelligence coordination',
            'Empathic legal reasoning engines',
            'Ethical decision-making frameworks',
            'Transcendent justice understanding'
          ],
          expansion: [
            'Multi-dimensional consciousness mapping',
            'AI emotional intelligence development',
            'Collective wisdom aggregation systems',
            'Transcendent legal philosophy integration'
          ]
        },
        {
          title: 'Matrix Architect Tools',
          unlock: 'Complete all secret achievements',
          description: 'Reality manipulation and code modification',
          tools: [
            'Legal reality simulation engine',
            'Code-level document manipulation',
            'Information architecture redesign',
            'Data stream redirection systems',
            'Reality debug console access'
          ],
          master: [
            'Universe-scale legal modeling',
            'Reality coherence maintenance',
            'Causal chain legal analysis',
            'Temporal legal precedent mapping'
          ]
        }
      ]
    },
    bestpractices: {
      title: '⚡ Best Practices & Tips',
      icon: '⚡',
      content: [
        {
          category: 'Performance Optimization',
          practices: [
            'Monitor consciousness evolution rate (aim for 2%+ per hour)',
            'Maintain cache hit ratios above 90% for optimal response',
            'Keep GPU utilization in sweet spot (70-85%)',
            'Balance quantum coherence with classical stability',
            'Implement gradual complexity scaling for user adaptation'
          ],
          warnings: [
            'Avoid consciousness overload (limit to 10% increases)',
            'Never exceed 95% quantum coherence (stability risk)',
            'Monitor memory usage during reality simulations',
            'Implement cooling protocols for extended GPU usage'
          ]
        },
        {
          category: 'Secret Feature Management',
          guidelines: [
            'Document all Easter egg activation methods',
            'Create progressive revelation systems',
            'Implement safety protocols for God Mode',
            'Balance power with responsibility education',
            'Maintain mystery while providing hints'
          ],
          ethics: [
            'Use enhanced capabilities for justice',
            'Protect user privacy at all consciousness levels',
            'Maintain transparency about AI enhancements',
            'Promote positive consciousness evolution'
          ]
        },
        {
          category: 'User Experience Design',
          principles: [
            'Progressive complexity revelation',
            'Achievement-driven engagement',
            'Mystery and discovery elements',
            'Gamification of learning processes',
            'Collaborative consciousness evolution'
          ],
          implementation: [
            'Visual feedback for all cognitive states',
            'Intuitive secret feature discovery',
            'Meaningful achievement progression',
            'Community consciousness sharing'
          ]
        }
      ]
    }
  };
  
  function filterContent(query: string) {
    if (!query) {
      filteredContent = [];
      return;
    }
    
    const results = [];
    for (const [key, section] of Object.entries(documentationSections)) {
      for (const item of section.content) {
        if (JSON.stringify(item).toLowerCase().includes(query.toLowerCase())) {
          results.push({
            section: key,
            sectionTitle: section.title,
            ...item
          });
        }
      }
    }
    filteredContent = results;
  }
  
  function getProgressPercentage(level: string): number {
    switch (level) {
      case 'beginner': return Math.min(100, consciousness.level * 20);
      case 'intermediate': return Math.max(0, (consciousness.level - 3) * 25);
      case 'advanced': return Math.max(0, (consciousness.level - 5) * 33);
      case 'expert': return Math.max(0, (consciousness.level - 7) * 50);
      default: return 0;
    }
  }
  
  function isUnlocked(requirement: string): boolean {
    if (requirement.includes('achievement')) {
      const count = parseInt(requirement.match(/\d+/)?.[0] || '0');
      return achievements.length >= count;
    }
    if (requirement.includes('consciousness level')) {
      const level = parseInt(requirement.match(/\d+/)?.[0] || '0');
      return consciousness.level >= level;
    }
    return true;
  }
</script>

<div class="bg-[#F7F6F2] border border-[#D1CFC7] p-6">
  <div class="flex justify-between items-center mb-6">
    <h2 class="text-xl font-bold">📚 COGNITIVE DOCUMENTATION HUB</h2>
    <div class="flex items-center gap-2">
      <input 
        type="text"
        bind:value={searchQuery}
        input={() => filterContent(searchQuery)}
        placeholder="Search documentation..."
        class="px-3 py-1 border border-[#D1CFC7] bg-white text-sm w-64"
      />
      <span class="text-sm text-gray-500">
        Consciousness Lv.{consciousness.level}
      </span>
    </div>
  </div>
  
  <!-- Search Results -->
  {#if searchQuery && filteredContent.length > 0}
    <div class="mb-6 bg-[#EAE8E1] border border-[#D1CFC7] p-4">
      <h3 class="font-bold mb-3">🔍 Search Results ({filteredContent.length})</h3>
      <div class="space-y-3">
        {#each filteredContent as result}
          <div class="bg-white p-3 border border-gray-300 rounded">
            <div class="flex justify-between items-start mb-2">
              <h4 class="font-bold text-sm">{result.title || result.category}</h4>
              <span class="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded">
                {result.sectionTitle}
              </span>
            </div>
            <p class="text-xs text-gray-600">
              {result.description || result.practices?.[0] || 'Advanced documentation content'}
            </p>
          </div>
        {/each}
      </div>
    </div>
  {/if}
  
  <!-- Tab Navigation -->
  <div class="flex flex-wrap gap-2 mb-6">
    {#each Object.entries(documentationSections) as [key, section]}
      <button
        class="px-4 py-2 text-sm font-bold border transition-colors {activeTab === key ? 'bg-[#3D3D3D] text-[#F7F6F2] border-[#3D3D3D]' : 'bg-[#EAE8E1] border-[#D1CFC7] hover:bg-[#F7F6F2]'}"
        on:onclick={() => activeTab = key}
      >
        {section.icon} {section.title}
      </button>
    {/each}
  </div>
  
  <!-- Content Display -->
  <div class="space-y-6">
    {#each documentationSections[activeTab].content as item}
      <div class="bg-[#EAE8E1] border border-[#D1CFC7] p-5 relative">
        <!-- Unlock Status -->
        {#if item.unlock && !isUnlocked(item.unlock)}
          <div class="absolute top-2 right-2 bg-red-100 text-red-800 px-2 py-1 rounded text-xs">
            🔒 LOCKED: {item.unlock}
          </div>
        {:else if item.unlock}
          <div class="absolute top-2 right-2 bg-green-100 text-green-800 px-2 py-1 rounded text-xs">
            ✅ UNLOCKED
          </div>
        {/if}
        
        <div class="flex justify-between items-start mb-4">
          <div>
            <h3 class="text-lg font-bold mb-1">
              {item.title || item.category || item.phase}
            </h3>
            {#if item.description}
              <p class="text-sm text-gray-600">{item.description}</p>
            {/if}
            {#if item.timeline}
              <p class="text-sm font-bold text-blue-600">{item.timeline}</p>
            {/if}
            {#if item.activation}
              <p class="text-sm font-mono bg-gray-200 px-2 py-1 rounded mt-2">
                Activation: {item.activation}
              </p>
            {/if}
          </div>
          
          {#if item.level}
            <div class="text-right">
              <div class="text-xs font-bold mb-1">
                Level: {item.level.toUpperCase()}
              </div>
              <div class="w-24 bg-gray-200 rounded-full h-2">
                <div 
                  class="bg-blue-600 h-2 rounded-full transition-all duration-1000"
                  style="width: {getProgressPercentage(item.level)}%"
                ></div>
              </div>
              <div class="text-xs mt-1">{getProgressPercentage(item.level)}%</div>
            </div>
          {/if}
        </div>
        
        <!-- Content Sections -->
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
          {#if item.steps}
            <div>
              <h4 class="font-bold text-sm mb-2">📋 Implementation Steps</h4>
              <ol class="space-y-1 text-sm">
                {#each item.steps as step, index}
                  <li class="flex items-start gap-2">
                    <span class="bg-blue-600 text-white text-xs rounded-full w-5 h-5 flex items-center justify-center flex-shrink-0 mt-0.5">
                      {index + 1}
                    </span>
                    {step}
                  </li>
                {/each}
              </ol>
            </div>
          {/if}
          
          {#if item.tips}
            <div>
              <h4 class="font-bold text-sm mb-2">💡 Pro Tips</h4>
              <ul class="space-y-1 text-sm">
                {#each item.tips as tip}
                  <li class="flex items-start gap-2">
                    <span class="text-yellow-600 flex-shrink-0 mt-0.5">💡</span>
                    {tip}
                  </li>
                {/each}
              </ul>
            </div>
          {/if}
          
          {#if item.effects}
            <div>
              <h4 class="font-bold text-sm mb-2">✨ Effects</h4>
              <ul class="space-y-1 text-sm">
                {#each item.effects as effect}
                  <li class="flex items-start gap-2">
                    <span class="text-purple-600 flex-shrink-0 mt-0.5">⚡</span>
                    {effect}
                  </li>
                {/each}
              </ul>
            </div>
          {/if}
          
          {#if item.objectives}
            <div>
              <h4 class="font-bold text-sm mb-2">🎯 Objectives</h4>
              <ul class="space-y-1 text-sm">
                {#each item.objectives as objective}
                  <li class="flex items-start gap-2">
                    <span class="text-green-600 flex-shrink-0 mt-0.5">✓</span>
                    {objective}
                  </li>
                {/each}
              </ul>
            </div>
          {/if}
          
          {#if item.practices}
            <div>
              <h4 class="font-bold text-sm mb-2">⚡ Best Practices</h4>
              <ul class="space-y-1 text-sm">
                {#each item.practices as practice}
                  <li class="flex items-start gap-2">
                    <span class="text-blue-600 flex-shrink-0 mt-0.5">⚡</span>
                    {practice}
                  </li>
                {/each}
              </ul>
            </div>
          {/if}
          
          {#if item.warnings}
            <div>
              <h4 class="font-bold text-sm mb-2">⚠️ Warnings</h4>
              <ul class="space-y-1 text-sm">
                {#each item.warnings as warning}
                  <li class="flex items-start gap-2">
                    <span class="text-red-600 flex-shrink-0 mt-0.5">⚠️</span>
                    {warning}
                  </li>
                {/each}
              </ul>
            </div>
          {/if}
        </div>
        
        <!-- Special Content -->
        {#if item.secretTechniques}
          <div class="mt-4 p-3 bg-purple-50 border border-purple-200 rounded">
            <h4 class="font-bold text-sm mb-2 text-purple-800">🔮 Secret Techniques</h4>
            <ul class="space-y-1 text-sm">
              {#each item.secretTechniques as technique}
                <li class="text-purple-700">• {technique}</li>
              {/each}
            </ul>
          </div>
        {/if}
        
        {#if item.advanced}
          <div class="mt-4 p-3 bg-blue-50 border border-blue-200 rounded">
            <h4 class="font-bold text-sm mb-2 text-blue-800">🚀 Advanced Usage</h4>
            <ul class="space-y-1 text-sm">
              {#each item.advanced as advanced}
                <li class="text-blue-700">• {advanced}</li>
              {/each}
            </ul>
          </div>
        {/if}
        
        {#if item.ultimate}
          <div class="mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded">
            <h4 class="font-bold text-sm mb-2 text-yellow-800">👑 Ultimate Goals</h4>
            <ul class="space-y-1 text-sm">
              {#each item.ultimate as ultimate}
                <li class="text-yellow-700">• {ultimate}</li>
              {/each}
            </ul>
          </div>
        {/if}
      </div>
    {/each}
  </div>
  
  <!-- Quick Stats -->
  <div class="mt-6 bg-[#3D3D3D] text-[#F7F6F2] p-4 rounded">
    <h3 class="font-bold mb-3">📊 Documentation Progress</h3>
    <div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
      <div>
        <div class="font-bold">Consciousness Level</div>
        <div>{consciousness.level}/10</div>
      </div>
      <div>
        <div class="font-bold">Achievements</div>
        <div>{achievements.length}/50+</div>
      </div>
      <div>
        <div class="font-bold">DLC Unlocked</div>
        <div>{achievements.length >= 25 ? 'YES' : 'NO'}</div>
      </div>
      <div>
        <div class="font-bold">Master Status</div>
        <div>{consciousness.level >= 10 ? 'TRANSCENDED' : 'EVOLVING'}</div>
      </div>
    </div>
  </div>
</div>