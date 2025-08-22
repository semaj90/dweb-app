<!-- YoRHa Demos Single Page App with ScrollArea -->
<script lang="ts">
  import { $state, $derived } from 'svelte';
  import { onMount } from 'svelte';
  import { goto } from '$app/navigation';
  import { Button } from 'bits-ui';
  import { ScrollArea } from '$lib/components/ui/scrollarea';
  import { Badge } from '$lib/components/ui/badge';
  import { 
    Bot, 
    FileText, 
    Shield, 
    Search, 
    Upload,
    Settings,
    Brain,
    Globe,
    ChevronRight,
    Activity,
    CheckCircle2,
    MessageSquare,
    BarChart3,
    Gavel,
    Eye,
    Terminal,
    Monitor,
    Server,
    Cpu,
    Play,
    Code,
    TestTube,
    Gamepad2,
    Home,
    ArrowLeft,
    Navigation,
    Layers
  } from 'lucide-svelte';

  // Svelte 5 state management
  let selectedDemo = $state('overview');
  let isLoading = $state(false);
  let scrollElement: HTMLElement;

  // All demo routes organized by category
  const demoCategories = [
    {
      id: 'overview',
      title: 'DEMOS OVERVIEW',
      icon: Monitor,
      color: 'text-amber-400',
      description: 'Complete list of all interactive demonstrations',
      demos: []
    },
    {
      id: 'ai-demos',
      title: 'AI DEMONSTRATIONS',
      icon: Bot,
      color: 'text-green-400',
      description: 'Artificial Intelligence and Machine Learning demos',
      demos: [
        { 
          path: '/enhanced-ai-demo', 
          title: 'Enhanced AI Demo', 
          description: 'Advanced AI capabilities with Context7 integration',
          status: 'production',
          tags: ['AI', 'Context7', 'RAG']
        },
        { 
          path: '/gpu-chat', 
          title: 'GPU Chat Interface', 
          description: 'GPU-accelerated chat with RTX 3060 Ti optimization',
          status: 'production',
          tags: ['GPU', 'Chat', 'Performance']
        },
        { 
          path: '/ai-demo', 
          title: 'AI Demo Showcase', 
          description: 'Interactive AI demonstrations and testing',
          status: 'production',
          tags: ['AI', 'Demo', 'Interactive']
        },
        { 
          path: '/ai-assistant', 
          title: 'AI Assistant Interface', 
          description: 'Advanced AI chat interface with XState integration',
          status: 'production',
          tags: ['AI', 'XState', 'Assistant']
        },
        { 
          path: '/local-ai-demo', 
          title: 'Local AI Demo', 
          description: 'Local LLM integration with Ollama',
          status: 'production',
          tags: ['Ollama', 'Local', 'LLM']
        }
      ]
    },
    {
      id: 'legal-demos',
      title: 'LEGAL AI DEMONSTRATIONS',
      icon: Gavel,
      color: 'text-blue-400',
      description: 'Legal document analysis and case management demos',
      demos: [
        { 
          path: '/rag-demo', 
          title: 'RAG Demo', 
          description: 'Retrieval-Augmented Generation for legal documents',
          status: 'production',
          tags: ['RAG', 'Legal', 'Documents']
        },
        { 
          path: '/semantic-search-demo', 
          title: 'Semantic Search', 
          description: 'AI-powered semantic search for legal content',
          status: 'production',
          tags: ['Search', 'Semantic', 'Legal']
        },
        { 
          path: '/evidenceboard', 
          title: 'Evidence Board', 
          description: 'Visual evidence management and analysis',
          status: 'production',
          tags: ['Evidence', 'Visualization', 'Legal']
        },
        { 
          path: '/evidence-editor', 
          title: 'Evidence Editor', 
          description: 'Interactive evidence editing and annotation',
          status: 'production',
          tags: ['Evidence', 'Editor', 'Annotation']
        },
        { 
          path: '/canvas-editor', 
          title: 'Canvas Editor', 
          description: 'Visual case canvas with drag-and-drop interface',
          status: 'production',
          tags: ['Canvas', 'Visual', 'Cases']
        }
      ]
    },
    {
      id: 'ui-demos',
      title: 'UI/UX DEMONSTRATIONS',
      icon: Layers,
      color: 'text-purple-400',
      description: 'User interface components and theming demos',
      demos: [
        { 
          path: '/yorha-demo', 
          title: 'YoRHa Theme Demo', 
          description: 'YoRHa interface theming and component showcase',
          status: 'production',
          tags: ['YoRHa', 'Theme', 'UI']
        },
        { 
          path: '/yorha-dashboard', 
          title: 'YoRHa Dashboard', 
          description: 'Complete YoRHa-themed dashboard interface',
          status: 'production',
          tags: ['YoRHa', 'Dashboard', 'Interface']
        },
        { 
          path: '/yorha-terminal', 
          title: 'YoRHa Terminal', 
          description: 'Command line interface with YoRHa styling',
          status: 'production',
          tags: ['Terminal', 'CLI', 'YoRHa']
        },
        { 
          path: '/frameworks-demo', 
          title: 'Frameworks Demo', 
          description: 'UI framework demonstrations and comparisons',
          status: 'production',
          tags: ['Frameworks', 'UI', 'Components']
        },
        { 
          path: '/demo/component-gallery', 
          title: 'Component Gallery', 
          description: 'Comprehensive UI component showcase',
          status: 'production',
          tags: ['Components', 'Gallery', 'Showcase']
        }
      ]
    },
    {
      id: 'development-demos',
      title: 'DEVELOPMENT TOOLS',
      icon: Code,
      color: 'text-yellow-400',
      description: 'Development and testing tool demonstrations',
      demos: [
        { 
          path: '/dev/mcp-tools', 
          title: 'MCP Tools Demo', 
          description: 'Model Context Protocol tools and integration testing',
          status: 'production',
          tags: ['MCP', 'Tools', 'Development']
        },
        { 
          path: '/dev/self-prompting-demo', 
          title: 'Self-Prompting Demo', 
          description: 'Autonomous AI prompting and decision making',
          status: 'production',
          tags: ['AI', 'Autonomous', 'Prompting']
        },
        { 
          path: '/dev/context7-test', 
          title: 'Context7 Test', 
          description: 'Context7 integration testing and validation',
          status: 'production',
          tags: ['Context7', 'Testing', 'Integration']
        },
        { 
          path: '/test', 
          title: 'System Tests', 
          description: 'Comprehensive system testing and validation',
          status: 'production',
          tags: ['Testing', 'Validation', 'System']
        },
        { 
          path: '/test-simple', 
          title: 'Simple Tests', 
          description: 'Basic functionality and unit testing',
          status: 'production',
          tags: ['Testing', 'Unit', 'Basic']
        }
      ]
    },
    {
      id: 'messaging-demos',
      title: 'MESSAGING & COMMUNICATION',
      icon: MessageSquare,
      color: 'text-indigo-400',
      description: 'Real-time messaging and communication demos',
      demos: [
        { 
          path: '/demos/nats-messaging', 
          title: 'NATS Messaging Demo', 
          description: 'Real-time messaging with NATS and WebSocket support',
          status: 'production',
          tags: ['NATS', 'Messaging', 'Real-time']
        }
      ]
    },
    {
      id: 'analytics-demos',
      title: 'ANALYTICS & MONITORING',
      icon: BarChart3,
      color: 'text-pink-400',
      description: 'System analytics and monitoring demonstrations',
      demos: [
        { 
          path: '/dashboard', 
          title: 'Analytics Dashboard', 
          description: 'System analytics and performance insights',
          status: 'production',
          tags: ['Analytics', 'Dashboard', 'Metrics']
        },
        { 
          path: '/memory-dashboard', 
          title: 'Memory Dashboard', 
          description: 'System memory monitoring and optimization',
          status: 'production',
          tags: ['Memory', 'Monitoring', 'Performance']
        },
        { 
          path: '/optimization-dashboard', 
          title: 'Optimization Dashboard', 
          description: 'Performance optimization and tuning tools',
          status: 'production',
          tags: ['Optimization', 'Performance', 'Tuning']
        },
        { 
          path: '/demo/ai-dashboard', 
          title: 'AI Dashboard Demo', 
          description: 'AI system monitoring and analytics dashboard',
          status: 'production',
          tags: ['AI', 'Dashboard', 'Monitoring']
        }
      ]
    },
    {
      id: 'admin-demos',
      title: 'ADMINISTRATION TOOLS',
      icon: Settings,
      color: 'text-red-400',
      description: 'System administration and configuration demos',
      demos: [
        { 
          path: '/admin/cluster', 
          title: 'Cluster Administration', 
          description: 'Cluster management and orchestration tools',
          status: 'production',
          tags: ['Cluster', 'Admin', 'Management']
        },
        { 
          path: '/admin/gpu-demo', 
          title: 'GPU Administration', 
          description: 'GPU management and monitoring tools',
          status: 'production',
          tags: ['GPU', 'Admin', 'Management']
        }
      ]
    }
  ];

  // Get all demos for overview
  const allDemos = $derived(
    demoCategories
      .filter(cat => cat.id !== 'overview')
      .flatMap(cat => cat.demos)
      .sort((a, b) => a.title.localeCompare(b.title))
  );

  // Navigation functions
  function handleDemoNavigation(path: string) {
    isLoading = true;
    setTimeout(() => {
      goto(path);
      isLoading = false;
    }, 300);
  }

  function scrollToCategory(categoryId: string) {
    selectedDemo = categoryId;
    const element = document.getElementById(`category-${categoryId}`);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }

  function goHome() {
    goto('/');
  }

  // Status badge colors
  function getStatusColor(status: string) {
    switch (status) {
      case 'production':
        return 'bg-green-500 text-green-100';
      case 'beta':
        return 'bg-yellow-500 text-yellow-100';
      case 'alpha':
        return 'bg-blue-500 text-blue-100';
      default:
        return 'bg-gray-500 text-gray-100';
    }
  }

  onMount(() => {
    // Auto-scroll to fragment if present in URL
    const hash = window.location.hash.substring(1);
    if (hash) {
      setTimeout(() => scrollToCategory(hash), 100);
    }
  });
</script>

<svelte:head>
  <title>YoRHa Legal AI - Demo Center | Interactive Demonstrations</title>
  <meta name="description" content="Comprehensive demo center for YoRHa Legal AI platform featuring interactive demonstrations of AI, legal tools, UI components, and system features." />
</svelte:head>

<div class="yorha-demos-container">
  <!-- Header Section -->
  <section class="yorha-demos-header">
    <div class="yorha-header-content">
      <!-- Navigation Bar -->
      <div class="yorha-nav-bar">
        <Button.Root 
          class="yorha-nav-btn"
          onclick={goHome}
        >
          <Home size={16} />
          HOME
        </Button.Root>
        
        <div class="yorha-breadcrumb">
          <span class="yorha-breadcrumb-item">YoRHa Legal AI</span>
          <ChevronRight size={14} />
          <span class="yorha-breadcrumb-current">Demo Center</span>
        </div>
      </div>

      <!-- Hero Title -->
      <div class="yorha-demos-hero">
        <h1 class="yorha-demos-title">
          <span class="yorha-title-icon">
            <Gamepad2 size={48} />
          </span>
          <span class="yorha-title-text">
            DEMO CENTER
          </span>
        </h1>
        <p class="yorha-demos-subtitle">
          Interactive demonstrations of YoRHa Legal AI platform capabilities
        </p>
        <div class="yorha-demos-stats">
          <div class="yorha-stat">
            <span class="yorha-stat-number">{allDemos.length}</span>
            <span class="yorha-stat-label">TOTAL DEMOS</span>
          </div>
          <div class="yorha-stat">
            <span class="yorha-stat-number">{demoCategories.length - 1}</span>
            <span class="yorha-stat-label">CATEGORIES</span>
          </div>
        </div>
      </div>

      <!-- Category Navigation -->
      <div class="yorha-category-nav">
        {#each demoCategories as category}
          <button 
            class="yorha-category-btn {selectedDemo === category.id ? 'active' : ''}"
            onclick={() => scrollToCategory(category.id)}
          >
            <svelte:component this={category.icon} size={16} />
            {category.title}
          </button>
        {/each}
      </div>
    </div>
  </section>

  <!-- Scrollable Content Area -->
  <ScrollArea class="yorha-demos-scroll-area">
    <div class="yorha-demos-content">
      <!-- Overview Section -->
      <section id="category-overview" class="yorha-demo-category">
        <div class="yorha-category-header">
          <h2 class="yorha-category-title">
            <Monitor size={24} />
            DEMOS OVERVIEW
          </h2>
          <p class="yorha-category-description">
            Complete collection of interactive demonstrations showcasing the full capabilities 
            of the YoRHa Legal AI platform including AI services, legal tools, UI components, 
            and development utilities.
          </p>
        </div>

        <div class="yorha-overview-grid">
          {#each allDemos as demo}
            <div class="yorha-demo-card yorha-demo-card-overview" onclick={() => handleDemoNavigation(demo.path)}>
              <div class="yorha-demo-header">
                <h3 class="yorha-demo-title">{demo.title}</h3>
                <Badge class="{getStatusColor(demo.status)}">
                  {demo.status.toUpperCase()}
                </Badge>
              </div>
              <p class="yorha-demo-description">{demo.description}</p>
              <div class="yorha-demo-tags">
                {#each demo.tags as tag}
                  <span class="yorha-tag">{tag}</span>
                {/each}
              </div>
              <div class="yorha-demo-footer">
                <span class="yorha-demo-path">{demo.path}</span>
                <ChevronRight size={16} class="yorha-demo-arrow" />
              </div>
            </div>
          {/each}
        </div>
      </section>

      <!-- Category Sections -->
      {#each demoCategories.filter(cat => cat.id !== 'overview') as category}
        <section id="category-{category.id}" class="yorha-demo-category">
          <div class="yorha-category-header">
            <h2 class="yorha-category-title" style="color: {category.color}">
              <svelte:component this={category.icon} size={24} />
              {category.title}
            </h2>
            <p class="yorha-category-description">{category.description}</p>
          </div>

          <div class="yorha-demos-grid">
            {#each category.demos as demo}
              <div class="yorha-demo-card" onclick={() => handleDemoNavigation(demo.path)}>
                <div class="yorha-demo-header">
                  <h3 class="yorha-demo-title">{demo.title}</h3>
                  <Badge class="{getStatusColor(demo.status)}">
                    {demo.status.toUpperCase()}
                  </Badge>
                </div>
                <p class="yorha-demo-description">{demo.description}</p>
                <div class="yorha-demo-tags">
                  {#each demo.tags as tag}
                    <span class="yorha-tag">{tag}</span>
                  {/each}
                </div>
                <div class="yorha-demo-footer">
                  <span class="yorha-demo-path">{demo.path}</span>
                  <div class="yorha-demo-actions">
                    <Button.Root 
                      class="yorha-demo-btn"
                      onclick={(e) => {
                        e.stopPropagation();
                        handleDemoNavigation(demo.path);
                      }}
                    >
                      <Play size={14} />
                      LAUNCH
                    </Button.Root>
                  </div>
                </div>
              </div>
            {/each}
          </div>
        </section>
      {/each}

      <!-- Footer Section -->
      <section class="yorha-demos-footer">
        <div class="yorha-footer-content">
          <h3 class="yorha-footer-title">
            <Terminal size={20} />
            TECHNICAL SPECIFICATIONS
          </h3>
          <div class="yorha-tech-specs">
            <div class="yorha-spec-item">
              <strong>Frontend:</strong> SvelteKit 2 + Svelte 5 runes
            </div>
            <div class="yorha-spec-item">
              <strong>UI Components:</strong> bits-ui + melt-ui + shadcn-svelte
            </div>
            <div class="yorha-spec-item">
              <strong>AI Engine:</strong> Multi-core Ollama + NVIDIA GPU acceleration
            </div>
            <div class="yorha-spec-item">
              <strong>Database:</strong> PostgreSQL + pgvector + Neo4j
            </div>
            <div class="yorha-spec-item">
              <strong>State Management:</strong> XState + enhanced reactive stores
            </div>
            <div class="yorha-spec-item">
              <strong>Messaging:</strong> NATS + WebSocket + real-time communication
            </div>
          </div>
          
          <div class="yorha-footer-actions">
            <Button.Root 
              class="yorha-btn yorha-btn-primary"
              onclick={goHome}
            >
              <ArrowLeft size={16} />
              RETURN TO HOMEPAGE
            </Button.Root>
          </div>
        </div>
      </section>
    </div>
  </ScrollArea>

  <!-- Loading Overlay -->
  {#if isLoading}
    <div class="yorha-loading-overlay">
      <div class="yorha-loading-content">
        <div class="yorha-spinner-large"></div>
        <div class="yorha-loading-text">INITIALIZING DEMO...</div>
      </div>
    </div>
  {/if}
</div>

<style>
  .yorha-demos-container {
    @apply min-h-screen bg-black text-amber-400 font-mono flex flex-col;
    font-family: 'Courier New', monospace;
    background-image: 
      radial-gradient(circle at 20% 50%, rgba(255, 191, 0, 0.03) 0%, transparent 50%),
      radial-gradient(circle at 80% 20%, rgba(255, 191, 0, 0.03) 0%, transparent 50%);
  }

  /* Header Section */
  .yorha-demos-header {
    @apply border-b border-amber-400 border-opacity-30 bg-gray-900 bg-opacity-50;
  }

  .yorha-header-content {
    @apply max-w-7xl mx-auto px-6 py-8 space-y-8;
  }

  .yorha-nav-bar {
    @apply flex items-center justify-between;
  }

  .yorha-nav-btn {
    @apply px-4 py-2 bg-amber-400 text-black font-mono text-sm tracking-wider;
    @apply hover:bg-amber-300 transition-colors flex items-center gap-2;
  }

  .yorha-breadcrumb {
    @apply flex items-center gap-2 text-sm text-amber-300;
  }

  .yorha-breadcrumb-item {
    @apply opacity-60;
  }

  .yorha-breadcrumb-current {
    @apply text-amber-400 font-bold;
  }

  .yorha-demos-hero {
    @apply text-center space-y-6;
  }

  .yorha-demos-title {
    @apply flex items-center justify-center gap-4 text-6xl md:text-8xl font-bold tracking-wider;
    text-shadow: 0 0 20px rgba(255, 191, 0, 0.5);
  }

  .yorha-title-icon {
    @apply text-amber-400;
  }

  .yorha-title-text {
    @apply text-amber-400;
  }

  .yorha-demos-subtitle {
    @apply text-xl text-amber-300 tracking-wide opacity-80;
  }

  .yorha-demos-stats {
    @apply flex justify-center gap-8;
  }

  .yorha-stat {
    @apply text-center;
  }

  .yorha-stat-number {
    @apply block text-3xl font-bold text-amber-400;
  }

  .yorha-stat-label {
    @apply block text-sm text-amber-300 tracking-wider;
  }

  .yorha-category-nav {
    @apply flex flex-wrap justify-center gap-2;
  }

  .yorha-category-btn {
    @apply px-4 py-2 bg-gray-900 border border-amber-400 border-opacity-30 text-amber-400;
    @apply font-mono text-sm tracking-wider transition-all duration-300;
    @apply hover:border-opacity-60 hover:bg-amber-400 hover:text-black;
    @apply flex items-center gap-2;
  }

  .yorha-category-btn.active {
    @apply bg-amber-400 text-black border-opacity-100;
  }

  /* Scrollable Content */
  .yorha-demos-scroll-area {
    @apply flex-1 h-0;
  }

  .yorha-demos-content {
    @apply space-y-16 pb-16;
  }

  /* Demo Categories */
  .yorha-demo-category {
    @apply px-6;
  }

  .yorha-category-header {
    @apply max-w-7xl mx-auto mb-12 text-center space-y-4;
  }

  .yorha-category-title {
    @apply text-3xl font-bold text-amber-400 flex items-center justify-center gap-3;
    @apply tracking-wider;
  }

  .yorha-category-description {
    @apply text-amber-300 max-w-3xl mx-auto leading-relaxed;
  }

  /* Demo Grids */
  .yorha-overview-grid {
    @apply grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6 max-w-7xl mx-auto;
  }

  .yorha-demos-grid {
    @apply grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 max-w-7xl mx-auto;
  }

  /* Demo Cards */
  .yorha-demo-card {
    @apply bg-gray-900 border border-amber-400 border-opacity-30 p-6 cursor-pointer;
    @apply hover:border-opacity-60 transition-all duration-300 hover:bg-amber-900 hover:bg-opacity-10;
    @apply space-y-4;
  }

  .yorha-demo-card-overview {
    @apply hover:scale-105 hover:shadow-lg;
    box-shadow: 0 0 20px rgba(255, 191, 0, 0.1);
  }

  .yorha-demo-header {
    @apply flex items-start justify-between gap-3;
  }

  .yorha-demo-title {
    @apply font-bold text-amber-400 tracking-wider flex-1;
  }

  .yorha-demo-description {
    @apply text-sm text-amber-300 leading-relaxed;
  }

  .yorha-demo-tags {
    @apply flex flex-wrap gap-2;
  }

  .yorha-tag {
    @apply px-2 py-1 bg-amber-400 bg-opacity-10 border border-amber-400 border-opacity-30;
    @apply text-xs text-amber-400 font-mono tracking-wider;
  }

  .yorha-demo-footer {
    @apply flex items-center justify-between pt-2 border-t border-amber-400 border-opacity-20;
  }

  .yorha-demo-path {
    @apply text-xs text-amber-400 opacity-60 font-mono;
  }

  .yorha-demo-arrow {
    @apply text-amber-300 opacity-60;
  }

  .yorha-demo-actions {
    @apply flex gap-2;
  }

  .yorha-demo-btn {
    @apply px-3 py-1 bg-amber-400 text-black text-xs font-mono;
    @apply hover:bg-amber-300 transition-colors flex items-center gap-1;
  }

  /* Footer */
  .yorha-demos-footer {
    @apply border-t border-amber-400 border-opacity-30 bg-gray-900 bg-opacity-50 px-6 py-16;
  }

  .yorha-footer-content {
    @apply max-w-7xl mx-auto space-y-8;
  }

  .yorha-footer-title {
    @apply text-2xl font-bold text-amber-400 flex items-center justify-center gap-3;
    @apply tracking-wider mb-6;
  }

  .yorha-tech-specs {
    @apply grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 text-sm;
  }

  .yorha-spec-item {
    @apply text-amber-300 border-l-2 border-amber-400 border-opacity-30 pl-4;
  }

  .yorha-footer-actions {
    @apply text-center;
  }

  /* Button Styles */
  .yorha-btn {
    @apply px-8 py-3 font-mono text-sm tracking-wider border transition-all duration-300;
    @apply flex items-center gap-3 justify-center;
  }

  .yorha-btn-primary {
    @apply bg-amber-400 text-black border-amber-400 hover:bg-amber-300 hover:border-amber-300;
    box-shadow: 0 0 15px rgba(255, 191, 0, 0.3);
  }

  /* Loading Overlay */
  .yorha-loading-overlay {
    @apply fixed inset-0 bg-black bg-opacity-80 flex items-center justify-center z-50;
  }

  .yorha-loading-content {
    @apply text-center space-y-6;
  }

  .yorha-spinner-large {
    @apply w-16 h-16 border-4 border-amber-400 border-t-transparent rounded-full animate-spin mx-auto;
  }

  .yorha-loading-text {
    @apply text-amber-400 font-mono tracking-wider text-lg;
  }

  /* Responsive Design */
  @media (max-width: 768px) {
    .yorha-demos-title {
      @apply text-4xl;
    }
    
    .yorha-category-nav {
      @apply grid grid-cols-2 gap-2;
    }
    
    .yorha-overview-grid {
      @apply grid-cols-1 gap-4;
    }
    
    .yorha-demos-grid {
      @apply grid-cols-1 gap-4;
    }
    
    .yorha-tech-specs {
      @apply grid-cols-1 gap-3;
    }
  }

  /* Custom scrollbar for ScrollArea */
  :global(.yorha-demos-scroll-area [data-bits-scroll-area-viewport]) {
    height: 100%;
  }

  :global(.yorha-demos-scroll-area::-webkit-scrollbar) {
    width: 8px;
  }

  :global(.yorha-demos-scroll-area::-webkit-scrollbar-track) {
    background: #1f2937;
  }

  :global(.yorha-demos-scroll-area::-webkit-scrollbar-thumb) {
    background: #ffbf00;
    border-radius: 4px;
  }

  :global(.yorha-demos-scroll-area::-webkit-scrollbar-thumb:hover) {
    background: #ffd700;
  }
</style>