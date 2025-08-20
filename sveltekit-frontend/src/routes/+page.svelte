<!-- YoRHa Legal AI Platform - Comprehensive Homepage -->
<!-- Built with Svelte 5 + Bits UI v2 + Context7 Best Practices -->

<script lang="ts">
  import { onMount } from 'svelte';
  import { goto } from '$app/navigation';
  import { page } from '$app/stores';
  import { Button } from 'bits-ui';
  import { Dialog } from 'bits-ui';
  import { Card } from '$lib/components/ui/card';
  import { Badge } from '$lib/components/ui/badge';
  import { 
    Bot, 
    FileText, 
    Shield, 
    Zap, 
    Database, 
    Search, 
    Upload,
    Settings,
    Users,
    Brain,
    Lock,
    Globe,
    ChevronRight,
    Activity,
    AlertTriangle,
    CheckCircle2,
    MessageSquare,
    BarChart3,
    Gavel,
    Eye,
    Terminal,
    Monitor,
    Server,
    Cpu,
    Layers,
    Map,
    Filter,
    Archive,
    Play,
    Code,
    TestTube,
    Puzzle,
    Gamepad2
  } from 'lucide-svelte';
  import YoRHaTable from '$lib/components/yorha/YoRHaTable.svelte';
  import AIAssistantButton from '$lib/components/AIAssistantButton.svelte';
  import { authStore } from '$lib/stores/auth-store';
  import { systemHealthStore } from '$lib/stores/system-health-store';
  import { aiAgentStore, isAIConnected, currentConversation } from '$lib/stores/ai-agent';

  // Svelte 5 state management
  let isLoginDialogOpen = $state(false);
  let isAIChatOpen = $state(false);
  let selectedCategory = $state('overview');
  let systemStatus = $state<'loading' | 'operational' | 'degraded' | 'offline'>('loading');
  let quickStats = $state({
    activeCases: 247,
    documentsProcessed: 15842,
    aiAnalysisRunning: 12,
    systemUptime: '99.94%'
  });

  // Chat integration
  let chatState = $state({
    message: '',
    isLoading: false,
    error: null
  });

  // Navigation categories with all routes organized
  const navigationCategories = [
    {
      id: 'overview',
      title: 'SYSTEM OVERVIEW',
      icon: Activity,
      color: 'text-amber-400',
      routes: []
    },
    {
      id: 'ai',
      title: 'AI SERVICES',
      icon: Bot,
      color: 'text-green-400',
      routes: [
        { path: '/ai-assistant', title: 'AI Assistant', description: 'Advanced AI chat interface' },
        { path: '/ai-demo', title: 'AI Demo', description: 'Interactive AI demonstrations' },
        { path: '/ai-summary', title: 'AI Summary', description: 'Document summarization' },
        { path: '/ai-test', title: 'AI Testing', description: 'AI system testing tools' },
        { path: '/ai-upload-demo', title: 'AI Upload Demo', description: 'File upload with AI processing' },
        { path: '/enhanced-ai-demo', title: 'Enhanced AI Demo', description: 'Advanced AI capabilities' },
        { path: '/local-ai-demo', title: 'Local AI Demo', description: 'Local LLM integration' },
        { path: '/gpu-chat', title: 'GPU Chat', description: 'GPU-accelerated chat' }
      ]
    },
    {
      id: 'cases',
      title: 'CASE MANAGEMENT',
      icon: Gavel,
      color: 'text-blue-400',
      routes: [
        { path: '/cases', title: 'Cases Dashboard', description: 'Manage legal cases' },
        { path: '/cases/new', title: 'New Case', description: 'Create new legal case' },
        { path: '/prosecutor', title: 'Prosecutor Tools', description: 'Prosecution management' },
        { path: '/detective', title: 'Detective Board', description: 'Investigation workspace' }
      ]
    },
    {
      id: 'evidence',
      title: 'EVIDENCE ANALYSIS',
      icon: Eye,
      color: 'text-purple-400',
      routes: [
        { path: '/evidence', title: 'Evidence Manager', description: 'Evidence processing and analysis' },
        { path: '/evidence/upload', title: 'Upload Evidence', description: 'Upload and process evidence' },
        { path: '/evidence/analyze', title: 'Analyze Evidence', description: 'AI-powered evidence analysis' },
        { path: '/evidence/hash', title: 'Hash Verification', description: 'Evidence integrity verification' },
        { path: '/evidence/realtime', title: 'Real-time Processing', description: 'Live evidence processing' },
        { path: '/evidence-editor', title: 'Evidence Editor', description: 'Visual evidence editing' },
        { path: '/evidenceboard', title: 'Evidence Board', description: 'Evidence visualization' }
      ]
    },
    {
      id: 'documents',
      title: 'DOCUMENT PROCESSING',
      icon: FileText,
      color: 'text-cyan-400',
      routes: [
        { path: '/legal/documents', title: 'Legal Documents', description: 'Legal document management' },
        { path: '/upload', title: 'Document Upload', description: 'Upload and process documents' },
        { path: '/editor', title: 'Document Editor', description: 'Advanced document editing' },
        { path: '/document-editor-demo', title: 'Editor Demo', description: 'Document editor demonstration' },
        { path: '/report-builder', title: 'Report Builder', description: 'Generate legal reports' },
        { path: '/reports', title: 'Reports', description: 'View and manage reports' }
      ]
    },
    {
      id: 'search',
      title: 'SEARCH & DISCOVERY',
      icon: Search,
      color: 'text-orange-400',
      routes: [
        { path: '/search', title: 'Advanced Search', description: 'Intelligent legal search' },
        { path: '/semantic-search-demo', title: 'Semantic Search', description: 'AI-powered semantic search' },
        { path: '/rag-demo', title: 'RAG Demo', description: 'Retrieval-augmented generation' },
        { path: '/enhanced', title: 'Enhanced Search', description: 'Enhanced search capabilities' },
        { path: '/laws', title: 'Legal Database', description: 'Browse legal statutes' }
      ]
    },
    {
      id: 'analytics',
      title: 'ANALYTICS & INSIGHTS',
      icon: BarChart3,
      color: 'text-pink-400',
      routes: [
        { path: '/dashboard', title: 'Analytics Dashboard', description: 'System analytics and insights' },
        { path: '/memory-dashboard', title: 'Memory Dashboard', description: 'System memory monitoring' },
        { path: '/optimization-dashboard', title: 'Optimization', description: 'Performance optimization' },
        { path: '/crud-dashboard', title: 'CRUD Dashboard', description: 'Data management interface' }
      ]
    },
    {
      id: 'admin',
      title: 'ADMINISTRATION',
      icon: Settings,
      color: 'text-red-400',
      routes: [
        { path: '/admin/cluster', title: 'Cluster Admin', description: 'Cluster management' },
        { path: '/admin/gpu-demo', title: 'GPU Demo', description: 'GPU administration' },
        { path: '/settings', title: 'System Settings', description: 'Configure system settings' },
        { path: '/security', title: 'Security Center', description: 'Security monitoring' },
        { path: '/help', title: 'Help & Support', description: 'Documentation and support' }
      ]
    },
    {
      id: 'demos',
      title: 'DEMONSTRATIONS',
      icon: Play,
      color: 'text-indigo-400',
      routes: [
        // Main Demo Center
        { path: '/demos', title: 'Demo Center', description: 'Interactive demo center with all demonstrations' },
        
        // Core Demos
        { path: '/demo', title: 'Feature Demos', description: 'System feature demonstrations' },
        { path: '/demo/xstate-auth', title: 'XState Auth Demo', description: 'Complete authentication flow with GPU acceleration' },
        { path: '/demo/ai-dashboard', title: 'AI Dashboard Demo', description: 'AI dashboard showcase' },
        { path: '/demo/component-gallery', title: 'Component Gallery', description: 'UI component showcase' },
        { path: '/demo/ai-assistant', title: 'AI Assistant Demo', description: 'Interactive AI assistant showcase' },
        { path: '/demo/ai-integration', title: 'AI Integration Demo', description: 'Complete AI system integration' },
        { path: '/demo/ai-pipeline', title: 'AI Pipeline Demo', description: 'AI processing pipeline demonstration' },
        { path: '/demo/enhanced-rag-semantic', title: 'Enhanced RAG', description: 'Semantic search with RAG' },
        { path: '/demo/vector-intelligence', title: 'Vector Intelligence', description: 'Advanced vector operations' },
        { path: '/demo/webgpu-acceleration', title: 'WebGPU Demo', description: 'GPU-accelerated processing' },
        { path: '/demo/integrated-system', title: 'System Integration', description: 'Full system demonstration' },
        
        // Legal AI Demos
        { path: '/demo/legal-ai-complete', title: 'Legal AI Complete', description: 'Complete legal AI workflow' },
        { path: '/demo/gpu-legal-ai', title: 'GPU Legal AI', description: 'GPU-accelerated legal analysis' },
        { path: '/demo/document-ai', title: 'Document AI', description: 'AI-powered document processing' },
        { path: '/demo/langextract-ollama', title: 'Language Extraction', description: 'Advanced text extraction' },
        
        // YoRHa Interface Demos
        { path: '/yorha-home', title: 'YoRHa Interface Home', description: 'YoRHa command center and interface hub' },
        { path: '/yorha-demo', title: 'YoRHa Theme Demo', description: 'YoRHa interface demonstration' },
        { path: '/yorha-dashboard', title: 'YoRHa Dashboard', description: 'YoRHa-themed dashboard' },
        { path: '/yorha-terminal', title: 'YoRHa Terminal', description: 'Command line interface' },
        { path: '/demo/yorha-tables', title: 'YoRHa Tables', description: 'Advanced data tables' },
        
        // Technical Demos  
        { path: '/demo/unified-architecture', title: 'Unified Architecture', description: 'System architecture showcase' },
        { path: '/demo/enhanced-semantic-architecture', title: 'Semantic Architecture', description: 'Advanced semantic processing' },
        { path: '/demo/neural-sprite-engine', title: 'Neural Sprite Engine', description: 'Advanced rendering engine' },
        { path: '/demo/unocss-svelte5', title: 'UnoCSS + Svelte 5', description: 'Modern CSS framework demo' },
        { path: '/demo/professional-editor', title: 'Professional Editor', description: 'Advanced text editor' },
        { path: '/demo/inline-suggestions', title: 'Inline Suggestions', description: 'AI-powered suggestions' },
        { path: '/demo/live-agents', title: 'Live Agents', description: 'Multi-agent system demo' },
        { path: '/demo/notes', title: 'Notes System', description: 'Advanced note-taking' }
      ]
    },
    {
      id: 'development',
      title: 'DEVELOPMENT TOOLS',
      icon: Code,
      color: 'text-yellow-400',
      routes: [
        // Development Tools
        { path: '/dev/mcp-tools', title: 'MCP Tools', description: 'Model Context Protocol tools' },
        { path: '/dev/context7-test', title: 'Context7 Test', description: 'Context7 integration testing' },
        { path: '/dev/self-prompting-demo', title: 'Self-Prompting', description: 'Autonomous prompting demo' },
        { path: '/dev/cache-demo', title: 'Cache Demo', description: 'Caching system demonstration' },
        { path: '/dev/metrics', title: 'Metrics Dashboard', description: 'System metrics and monitoring' },
        { path: '/dev/vector-search-demo', title: 'Vector Search Demo', description: 'Vector search testing' },
        { path: '/dev/enhanced-processor', title: 'Enhanced Processor', description: 'Advanced processing tools' },
        { path: '/dev/copilot-optimizer', title: 'Copilot Optimizer', description: 'AI optimization tools' },
        { path: '/dev/suggestions', title: 'Suggestions Engine', description: 'AI suggestions testing' },
        { path: '/dev/vite-error-demo', title: 'Vite Error Demo', description: 'Error handling demonstration' },
        
        // Testing Routes
        { path: '/test', title: 'System Tests', description: 'Comprehensive system testing' },
        { path: '/test-simple', title: 'Simple Tests', description: 'Basic functionality tests' },
        { path: '/test-ai-ask', title: 'AI Ask Test', description: 'AI query testing' },
        { path: '/test-components', title: 'Component Tests', description: 'UI component testing' },
        { path: '/test-gemma3', title: 'Gemma3 Test', description: 'Gemma3 model testing' },
        { path: '/test-integration', title: 'Integration Tests', description: 'System integration testing' },
        { path: '/test-upload', title: 'Upload Tests', description: 'File upload testing' },
        
        // Specialized Demos
        { path: '/auth/test', title: 'Auth Test', description: 'Authentication testing' },
        { path: '/bits-uno-demo', title: 'Bits + UnoCSS Demo', description: 'UI framework integration' },
        { path: '/context7-demo', title: 'Context7 Demo', description: 'Context7 system demonstration' },
        { path: '/frameworks-demo', title: 'Frameworks Demo', description: 'Framework comparison' },
        { path: '/gaming-demo', title: 'Gaming Interface', description: 'Gaming-style UI demonstration' },
        { path: '/golden-ratio-demo', title: 'Golden Ratio Demo', description: 'Design system showcase' },
        { path: '/modern-demo', title: 'Modern UI Demo', description: 'Modern interface patterns' },
        { path: '/nier-showcase', title: 'NieR Showcase', description: 'NieR-inspired interface' },
        { path: '/ui-demo', title: 'UI Components Demo', description: 'Complete UI component library' },
        { path: '/wasm-gpu-demo', title: 'WASM GPU Demo', description: 'WebAssembly GPU processing' }
      ]
    },
    {
      id: 'utilities',
      title: 'UTILITY TOOLS',
      icon: TestTube,
      color: 'text-teal-400',
      routes: [
        // Utility Pages
        { path: '/chat', title: 'Chat Interface', description: 'Direct chat interface' },
        { path: '/upload-test', title: 'Upload Test', description: 'File upload testing' },
        { path: '/simple-upload-test', title: 'Simple Upload', description: 'Basic file upload' },
        { path: '/semantic-search-demo', title: 'Semantic Search', description: 'Advanced semantic search' },
        { path: '/rag-demo', title: 'RAG Demonstration', description: 'Retrieval augmented generation' },
        { path: '/enhanced-ai-demo', title: 'Enhanced AI', description: 'Advanced AI capabilities' },
        { path: '/local-ai-demo', title: 'Local AI', description: 'Local AI model integration' },
        { path: '/gpu-chat', title: 'GPU Chat', description: 'GPU-accelerated chat' },
        { path: '/compiler-ai-demo', title: 'Compiler AI', description: 'AI-powered compilation' },
        { path: '/phase13-demo', title: 'Phase 13 Demo', description: 'Advanced system phase' },
        { path: '/windows-gguf-demo', title: 'Windows GGUF', description: 'GGUF model demonstration' },
        
        // Specialized Tools
        { path: '/copilot/autonomous', title: 'Autonomous Copilot', description: 'AI autonomous assistance' },
        { path: '/studio', title: 'Development Studio', description: 'Integrated development environment' },
        { path: '/showcase', title: 'System Showcase', description: 'Complete system overview' },
        { path: '/interactive-canvas', title: 'Interactive Canvas', description: 'Advanced canvas editing' },
        { path: '/detective/canvas', title: 'Detective Canvas', description: 'Investigation workspace' },
        { path: '/legal-ai-suite', title: 'Legal AI Suite', description: 'Complete legal AI toolkit' }
      ]
    }
  ];

  // Featured cases data
  const featuredCasesColumns = [
    { key: 'id', title: 'CASE ID', sortable: true, width: '120px' },
    { key: 'title', title: 'CASE TITLE', sortable: true },
    { key: 'status', title: 'STATUS', type: 'status', sortable: true, width: '140px' },
    { key: 'priority', title: 'PRIORITY', type: 'status', sortable: true, width: '120px' },
    { key: 'lastUpdate', title: 'LAST UPDATE', type: 'date', sortable: true, width: '140px' },
    { key: 'actions', title: 'ACTIONS', type: 'action', width: '120px' }
  ] as const;

  const featuredCasesData = $state([
    {
      id: 'LGL-2024-001',
      title: 'Corporate Compliance Audit',
      status: 'active',
      priority: 'high',
      lastUpdate: new Date('2024-08-20'),
      assignee: 'A. Smith'
    },
    {
      id: 'LGL-2024-002', 
      title: 'Contract Review - TechCorp',
      status: 'processing',
      priority: 'medium',
      lastUpdate: new Date('2024-08-19'),
      assignee: 'B. Jones'
    },
    {
      id: 'LGL-2024-003',
      title: 'IP Litigation Defense',
      status: 'pending',
      priority: 'critical',
      lastUpdate: new Date('2024-08-18'),
      assignee: 'C. Wilson'
    }
  ]);

  // Service status indicators
  const serviceStatuses = $derived([
    { name: 'Enhanced RAG', status: 'online', port: 8094, description: 'AI reasoning engine' },
    { name: 'Upload Service', status: 'online', port: 8093, description: 'File processing service' },
    { name: 'Ollama AI', status: 'online', port: 11434, description: 'Local LLM server' },
    { name: 'PostgreSQL', status: 'online', port: 5432, description: 'Vector database' },
    { name: 'Redis Cache', status: 'online', port: 6379, description: 'Caching layer' },
    { name: 'Neo4j Graph', status: 'pending', port: 7474, description: 'Knowledge graph' }
  ]);

  // System health check
  onMount(async () => {
    try {
      const healthResponse = await fetch('/api/v1/cluster/health');
      if (healthResponse.ok) {
        systemStatus = 'operational';
      } else {
        systemStatus = 'degraded';
      }
    } catch (error) {
      console.error('Health check failed:', error);
      systemStatus = 'offline';
    }

    // Initialize AI system
    try {
      await aiAgentStore.connect();
    } catch (error) {
      console.error('AI system initialization failed:', error);
    }

    // Simulate real-time updates
    const interval = setInterval(() => {
      quickStats.activeCases += Math.floor(Math.random() * 3) - 1;
      quickStats.documentsProcessed += Math.floor(Math.random() * 5);
      quickStats.aiAnalysisRunning = Math.max(0, quickStats.aiAnalysisRunning + Math.floor(Math.random() * 5) - 2);
    }, 5000);

    return () => clearInterval(interval);
  });

  // Navigation handlers
  function handleRouteNavigation(path: string) {
    if ($authStore.isAuthenticated || isPublicRoute(path)) {
      goto(path);
    } else {
      isLoginDialogOpen = true;
    }
  }

  function isPublicRoute(path: string): boolean {
    const publicRoutes = ['/demo', '/help', '/yorha-demo', '/test'];
    return publicRoutes.some(route => path.startsWith(route));
  }

  function handleAIChat() {
    isAIChatOpen = true;
  }

  // AI Chat functions
  async function sendAIMessage() {
    if (!chatState.message.trim() || chatState.isLoading) return;

    const userMessage = chatState.message.trim();
    chatState.message = '';
    chatState.isLoading = true;
    chatState.error = null;

    try {
      await aiAgentStore.sendMessage(userMessage, {
        timestamp: new Date(),
        source: 'yorha_homepage',
        userAgent: navigator.userAgent
      });
      chatState.isLoading = false;
    } catch (error) {
      console.error('AI chat error:', error);
      chatState.isLoading = false;
      chatState.error = `Failed to send message: ${(error as Error).message}`;
    }
  }

  function handleChatKeyDown(event: KeyboardEvent) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      sendAIMessage();
    }
  }

  // Status badge colors
  function getStatusColor(status: string) {
    switch (status) {
      case 'operational':
      case 'online':
        return 'bg-green-500 text-green-100';
      case 'degraded':
      case 'pending':
        return 'bg-yellow-500 text-yellow-100';
      case 'offline':
        return 'bg-red-500 text-red-100';
      default:
        return 'bg-gray-500 text-gray-100';
    }
  }
</script>

<svelte:head>
  <title>YoRHa Legal AI Platform - Advanced Legal Document Analysis</title>
  <meta name="description" content="Enterprise-grade legal AI platform with advanced document analysis, case management, and intelligent automation." />
</svelte:head>

<div class="yorha-home-container">
  <!-- Hero Section -->
  <section class="yorha-hero">
    <div class="yorha-hero-content">
      <div class="yorha-hero-header">
        <h1 class="yorha-hero-title">
          <span class="yorha-title-main">YoRHa</span>
          <span class="yorha-title-sub">LEGAL AI PLATFORM</span>
        </h1>
        <div class="yorha-hero-tagline">
          ADVANCED ARTIFICIAL INTELLIGENCE FOR LEGAL DOCUMENT ANALYSIS
        </div>
      </div>

      <!-- System Status Panel -->
      <div class="yorha-status-panel">
        <div class="yorha-status-header">
          <Activity size={16} />
          SYSTEM STATUS
          <Badge class="{getStatusColor(systemStatus)}">
            {systemStatus.toUpperCase()}
          </Badge>
        </div>
        
        <div class="yorha-status-grid">
          <div class="yorha-stat-item">
            <div class="yorha-stat-value">{quickStats.activeCases}</div>
            <div class="yorha-stat-label">ACTIVE CASES</div>
          </div>
          <div class="yorha-stat-item">
            <div class="yorha-stat-value">{quickStats.documentsProcessed.toLocaleString()}</div>
            <div class="yorha-stat-label">DOCUMENTS PROCESSED</div>
          </div>
          <div class="yorha-stat-item">
            <div class="yorha-stat-value">{quickStats.aiAnalysisRunning}</div>
            <div class="yorha-stat-label">AI ANALYSIS RUNNING</div>
          </div>
          <div class="yorha-stat-item">
            <div class="yorha-stat-value">{quickStats.systemUptime}</div>
            <div class="yorha-stat-label">SYSTEM UPTIME</div>
          </div>
        </div>
      </div>

      <!-- Action Buttons -->
      <div class="yorha-hero-actions">
        <Button.Root 
          class="yorha-btn yorha-btn-primary yorha-btn-demos"
          onclick={() => handleRouteNavigation('/demos')}
        >
          <Play size={20} />
          INTERACTIVE DEMOS
          <ChevronRight size={16} />
        </Button.Root>

        <Button.Root 
          class="yorha-btn yorha-btn-secondary"
          onclick={handleAIChat}
        >
          <MessageSquare size={20} />
          AI ASSISTANT
          <ChevronRight size={16} />
        </Button.Root>

        <Button.Root 
          class="yorha-btn yorha-btn-secondary"
          onclick={() => handleRouteNavigation('/cases')}
        >
          <Gavel size={20} />
          CASE MANAGEMENT
        </Button.Root>

        <Button.Root 
          class="yorha-btn yorha-btn-outline"
          onclick={() => handleRouteNavigation('/evidence')}
        >
          <Eye size={20} />
          EVIDENCE ANALYSIS
        </Button.Root>
      </div>
    </div>
  </section>

  <!-- Navigation Categories -->
  <section class="yorha-navigation-section">
    <div class="yorha-nav-header">
      <h2 class="yorha-section-title">
        <Monitor size={20} />
        SYSTEM MODULES
      </h2>
      
      <div class="yorha-category-tabs">
        {#each navigationCategories as category}
          <button 
            class="yorha-tab {selectedCategory === category.id ? 'active' : ''}"
            onclick={() => selectedCategory = category.id}
          >
            <svelte:component this={category.icon} size={16} />
            {category.title}
          </button>
        {/each}
      </div>
    </div>

    <div class="yorha-routes-grid">
      {#each navigationCategories.find(cat => cat.id === selectedCategory)?.routes || [] as route}
        <div class="yorha-route-card" onclick={() => handleRouteNavigation(route.path)}>
          <div class="yorha-route-header">
            <h3 class="yorha-route-title">{route.title}</h3>
            <ChevronRight size={16} class="yorha-route-arrow" />
          </div>
          <p class="yorha-route-description">{route.description}</p>
          <div class="yorha-route-path">{route.path}</div>
        </div>
      {/each}
    </div>
  </section>

  <!-- Services Status Section -->
  <section class="yorha-services-section">
    <h2 class="yorha-section-title">
      <Server size={20} />
      SERVICE STATUS
    </h2>
    
    <div class="yorha-services-grid">
      {#each serviceStatuses as service}
        <div class="yorha-service-card">
          <div class="yorha-service-header">
            <span class="yorha-service-name">{service.name}</span>
            <Badge class="{getStatusColor(service.status)}">
              {service.status.toUpperCase()}
            </Badge>
          </div>
          <div class="yorha-service-description">{service.description}</div>
          <div class="yorha-service-port">PORT: {service.port}</div>
          <div class="yorha-service-indicator {service.status}"></div>
        </div>
      {/each}
    </div>
  </section>

  <!-- Featured Cases Section -->
  <section class="yorha-cases-section">
    <h2 class="yorha-section-title">
      <FileText size={20} />
      RECENT CASE ACTIVITY
    </h2>
    
    <div class="yorha-table-container">
      <YoRHaTable 
        columns={featuredCasesColumns}
        data={featuredCasesData}
        pagination={true}
        pageSize={5}
        glitchEffect={true}
        theme="dark"
      >
        {#snippet actions({ row })}
          <Button.Root class="yorha-action-btn-sm" onclick={() => handleRouteNavigation(`/cases/${row.id}`)}>
            VIEW
          </Button.Root>
          <Button.Root class="yorha-action-btn-sm" onclick={() => handleRouteNavigation(`/cases/${row.id}/canvas`)}>
            EDIT
          </Button.Root>
        {/snippet}
      </YoRHaTable>
    </div>
  </section>

  <!-- AI Assistant Button -->
  <div class="yorha-ai-fab">
    <AIAssistantButton />
  </div>

  <!-- AI Chat Dialog -->
  <Dialog.Root bind:open={isAIChatOpen}>
    <Dialog.Portal>
      <Dialog.Overlay class="yorha-dialog-overlay" />
      <Dialog.Content class="yorha-chat-dialog">
        <Dialog.Title class="yorha-chat-title">
          <Bot size={20} />
          YoRHa AI ASSISTANT
          <Badge class="{getStatusColor($isAIConnected ? 'online' : 'offline')}">
            {$isAIConnected ? 'ONLINE' : 'OFFLINE'}
          </Badge>
        </Dialog.Title>
        
        <div class="yorha-chat-messages">
          {#if $currentConversation.length === 0}
            <div class="yorha-chat-welcome">
              <Bot size={48} class="yorha-welcome-icon" />
              <h3>YoRHa AI Assistant Active</h3>
              <p>How may I assist with your legal analysis today?</p>
            </div>
          {:else}
            {#each $currentConversation as message}
              <div class="yorha-message {message.role}">
                <div class="yorha-message-header">
                  <span class="yorha-message-role">
                    {message.role === 'user' ? 'USER' : 'AI ASSISTANT'}
                  </span>
                  <span class="yorha-message-time">
                    {new Date(message.timestamp).toLocaleTimeString()}
                  </span>
                </div>
                <div class="yorha-message-content">{message.content}</div>
              </div>
            {/each}
          {/if}
          
          {#if chatState.isLoading}
            <div class="yorha-message ai loading">
              <div class="yorha-loading-indicator">
                <div class="yorha-spinner"></div>
                AI PROCESSING...
              </div>
            </div>
          {/if}
        </div>

        {#if chatState.error}
          <div class="yorha-chat-error">
            <AlertTriangle size={16} />
            {chatState.error}
          </div>
        {/if}

        <div class="yorha-chat-input">
          <textarea
            bind:value={chatState.message}
            onkeydown={handleChatKeyDown}
            placeholder="Enter your query..."
            class="yorha-textarea"
            rows="3"
            disabled={chatState.isLoading || !$isAIConnected}
          ></textarea>
          <Button.Root 
            class="yorha-send-btn"
            onclick={sendAIMessage}
            disabled={!chatState.message.trim() || chatState.isLoading || !$isAIConnected}
          >
            {#if chatState.isLoading}
              <div class="yorha-spinner"></div>
            {:else}
              SEND
            {/if}
          </Button.Root>
        </div>
        
        <Dialog.Close class="yorha-dialog-close">
          ×
        </Dialog.Close>
      </Dialog.Content>
    </Dialog.Portal>
  </Dialog.Root>

  <!-- Authentication Required Dialog -->
  <Dialog.Root open={isLoginDialogOpen} onOpenChange={(open) => isLoginDialogOpen = open}>
    <Dialog.Portal>
      <Dialog.Overlay class="yorha-dialog-overlay" />
      <Dialog.Content class="yorha-dialog-content">
        <Dialog.Title class="yorha-dialog-title">
          <Lock size={20} />
          AUTHENTICATION REQUIRED
        </Dialog.Title>
        <Dialog.Description class="yorha-dialog-description">
          Access to this feature requires authentication. Please login or register to continue.
        </Dialog.Description>
        
        <div class="yorha-dialog-actions">
          <Button.Root 
            class="yorha-btn yorha-btn-primary"
            onclick={() => {
              isLoginDialogOpen = false;
              goto('/login');
            }}
          >
            <Users size={16} />
            LOGIN
          </Button.Root>
          
          <Button.Root 
            class="yorha-btn yorha-btn-outline"
            onclick={() => {
              isLoginDialogOpen = false;
              goto('/register');
            }}
          >
            <Globe size={16} />
            REGISTER
          </Button.Root>
        </div>
        
        <Dialog.Close class="yorha-dialog-close">
          ×
        </Dialog.Close>
      </Dialog.Content>
    </Dialog.Portal>
  </Dialog.Root>
</div>

<style>
  .yorha-home-container {
    @apply min-h-screen bg-black text-amber-400 font-mono overflow-x-hidden;
    font-family: 'Courier New', monospace;
    background-image: 
      radial-gradient(circle at 20% 50%, rgba(255, 191, 0, 0.03) 0%, transparent 50%),
      radial-gradient(circle at 80% 20%, rgba(255, 191, 0, 0.03) 0%, transparent 50%),
      radial-gradient(circle at 40% 80%, rgba(255, 191, 0, 0.03) 0%, transparent 50%);
  }

  /* Hero Section */
  .yorha-hero {
    @apply relative py-20 px-6 text-center border-b border-amber-400 border-opacity-30;
    background: linear-gradient(135deg, transparent 0%, rgba(255, 191, 0, 0.05) 100%);
  }

  .yorha-hero::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #ffbf00, transparent);
    animation: scanline 4s linear infinite;
  }

  .yorha-hero-content {
    @apply max-w-6xl mx-auto space-y-12;
  }

  .yorha-hero-header {
    @apply space-y-6;
  }

  .yorha-hero-title {
    @apply text-6xl md:text-8xl font-bold tracking-wider;
  }

  .yorha-title-main {
    @apply block text-amber-400;
    text-shadow: 0 0 20px rgba(255, 191, 0, 0.5);
  }

  .yorha-title-sub {
    @apply block text-2xl md:text-3xl text-amber-300 mt-2;
    font-weight: 300;
    letter-spacing: 0.3em;
  }

  .yorha-hero-tagline {
    @apply text-lg md:text-xl text-amber-300 tracking-wide;
    opacity: 0.8;
  }

  /* Status Panel */
  .yorha-status-panel {
    @apply bg-gray-900 border border-amber-400 p-6 rounded-none;
    box-shadow: 0 0 20px rgba(255, 191, 0, 0.2);
  }

  .yorha-status-header {
    @apply flex items-center justify-center gap-3 mb-6 text-lg font-bold tracking-wider;
  }

  .yorha-status-grid {
    @apply grid grid-cols-2 md:grid-cols-4 gap-6;
  }

  .yorha-stat-item {
    @apply text-center;
  }

  .yorha-stat-value {
    @apply text-2xl md:text-3xl font-bold text-amber-400 mb-2;
    text-shadow: 0 0 10px rgba(255, 191, 0, 0.3);
  }

  .yorha-stat-label {
    @apply text-xs text-amber-300 tracking-wider;
  }

  /* Action Buttons */
  .yorha-hero-actions {
    @apply flex flex-col md:flex-row gap-4 justify-center items-center;
  }

  /* Navigation Section */
  .yorha-navigation-section {
    @apply py-16 px-6;
  }

  .yorha-nav-header {
    @apply max-w-6xl mx-auto mb-12;
  }

  .yorha-section-title {
    @apply text-2xl md:text-3xl font-bold text-center mb-8 flex items-center justify-center gap-3;
    @apply tracking-wider text-amber-400;
  }

  .yorha-category-tabs {
    @apply flex flex-wrap justify-center gap-2;
  }

  .yorha-tab {
    @apply px-4 py-2 bg-gray-900 border border-amber-400 border-opacity-30 text-amber-400;
    @apply font-mono text-sm tracking-wider transition-all duration-300;
    @apply hover:border-opacity-60 hover:bg-amber-400 hover:text-black;
    @apply flex items-center gap-2;
  }

  .yorha-tab.active {
    @apply bg-amber-400 text-black border-opacity-100;
  }

  .yorha-routes-grid {
    @apply grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 max-w-6xl mx-auto mt-8;
  }

  .yorha-route-card {
    @apply bg-gray-900 border border-amber-400 border-opacity-30 p-6 cursor-pointer;
    @apply hover:border-opacity-60 transition-all duration-300 hover:bg-amber-900 hover:bg-opacity-10;
  }

  .yorha-route-header {
    @apply flex items-center justify-between mb-3;
  }

  .yorha-route-title {
    @apply font-bold text-amber-400 tracking-wider;
  }

  .yorha-route-arrow {
    @apply text-amber-300 opacity-60;
  }

  .yorha-route-description {
    @apply text-sm text-amber-300 mb-3 leading-relaxed;
  }

  .yorha-route-path {
    @apply text-xs text-amber-400 opacity-60 font-mono;
  }

  /* Services Section */
  .yorha-services-section {
    @apply py-16 px-6 border-t border-amber-400 border-opacity-30;
  }

  .yorha-services-grid {
    @apply grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 max-w-6xl mx-auto;
  }

  .yorha-service-card {
    @apply bg-gray-900 border border-amber-400 border-opacity-30 p-6 relative;
    @apply hover:border-opacity-60 transition-all duration-300;
  }

  .yorha-service-header {
    @apply flex items-center justify-between mb-3;
  }

  .yorha-service-name {
    @apply font-bold text-amber-400 tracking-wider;
  }

  .yorha-service-description {
    @apply text-sm text-amber-300 mb-3;
  }

  .yorha-service-port {
    @apply text-xs text-amber-300 mb-3;
  }

  .yorha-service-indicator {
    @apply absolute bottom-2 right-2 w-3 h-3 rounded-full;
  }

  .yorha-service-indicator.online {
    @apply bg-green-500;
    box-shadow: 0 0 10px rgba(34, 197, 94, 0.5);
    animation: pulse-green 2s infinite;
  }

  .yorha-service-indicator.pending {
    @apply bg-yellow-500;
    animation: pulse-yellow 2s infinite;
  }

  /* Cases Section */
  .yorha-cases-section {
    @apply py-16 px-6 border-t border-amber-400 border-opacity-30;
  }

  .yorha-table-container {
    @apply max-w-6xl mx-auto;
  }

  /* AI Assistant FAB */
  .yorha-ai-fab {
    @apply fixed bottom-6 right-6 z-50;
  }

  /* Button Styles */
  .yorha-btn {
    @apply px-8 py-3 font-mono text-sm tracking-wider border transition-all duration-300;
    @apply flex items-center gap-3 min-w-48 justify-center;
  }

  .yorha-btn-primary {
    @apply bg-amber-400 text-black border-amber-400 hover:bg-amber-300 hover:border-amber-300;
    box-shadow: 0 0 15px rgba(255, 191, 0, 0.3);
  }

  .yorha-btn-demos {
    @apply text-lg font-bold min-w-64;
    box-shadow: 0 0 25px rgba(255, 191, 0, 0.5);
    animation: pulse-glow 2s infinite;
  }

  @keyframes pulse-glow {
    0%, 100% { 
      box-shadow: 0 0 25px rgba(255, 191, 0, 0.5);
      transform: scale(1);
    }
    50% { 
      box-shadow: 0 0 35px rgba(255, 191, 0, 0.8);
      transform: scale(1.02);
    }
  }

  .yorha-btn-secondary {
    @apply bg-gray-800 text-amber-400 border-amber-400 hover:bg-gray-700;
  }

  .yorha-btn-outline {
    @apply bg-transparent text-amber-400 border-amber-400 hover:bg-amber-400 hover:text-black;
  }

  .yorha-action-btn-sm {
    @apply bg-amber-400 text-black px-2 py-1 text-xs font-mono hover:bg-amber-300 transition-colors;
    border: 1px solid #ffbf00;
  }

  /* Dialog Styles */
  .yorha-dialog-overlay {
    @apply fixed inset-0 bg-black bg-opacity-80 z-40;
  }

  .yorha-dialog-content {
    @apply fixed top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2;
    @apply bg-gray-900 border-2 border-amber-400 p-8 max-w-md w-full mx-4 z-50;
    box-shadow: 0 0 30px rgba(255, 191, 0, 0.3);
  }

  .yorha-dialog-title {
    @apply text-xl font-bold text-amber-400 mb-4 flex items-center gap-3;
  }

  .yorha-dialog-description {
    @apply text-amber-300 mb-6 leading-relaxed;
  }

  .yorha-dialog-actions {
    @apply space-y-3;
  }

  .yorha-dialog-close {
    @apply absolute top-3 right-3 text-amber-400 hover:text-amber-300 text-2xl cursor-pointer;
  }

  /* Chat Dialog */
  .yorha-chat-dialog {
    @apply fixed top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2;
    @apply bg-gray-900 border-2 border-amber-400 w-full max-w-2xl h-96 max-h-[80vh] z-50;
    @apply flex flex-col;
    box-shadow: 0 0 30px rgba(255, 191, 0, 0.3);
  }

  .yorha-chat-title {
    @apply p-4 border-b border-amber-400 border-opacity-30 text-lg font-bold text-amber-400;
    @apply flex items-center gap-3;
  }

  .yorha-chat-messages {
    @apply flex-1 p-4 overflow-y-auto space-y-4;
  }

  .yorha-chat-welcome {
    @apply text-center py-8 space-y-4;
  }

  .yorha-welcome-icon {
    @apply text-amber-400 mx-auto;
  }

  .yorha-message {
    @apply p-3 rounded border;
  }

  .yorha-message.user {
    @apply bg-amber-400 bg-opacity-10 border-amber-400 border-opacity-30 ml-8;
  }

  .yorha-message.ai {
    @apply bg-gray-800 border-gray-600 mr-8;
  }

  .yorha-message.loading {
    @apply border-amber-400 border-opacity-50;
  }

  .yorha-message-header {
    @apply flex justify-between items-center mb-2 text-xs opacity-70;
  }

  .yorha-message-role {
    @apply font-bold tracking-wider;
  }

  .yorha-message-content {
    @apply text-sm leading-relaxed;
  }

  .yorha-loading-indicator {
    @apply flex items-center gap-2 text-amber-400;
  }

  .yorha-chat-error {
    @apply p-3 bg-red-900 bg-opacity-50 border border-red-400 text-red-300 text-sm;
    @apply flex items-center gap-2;
  }

  .yorha-chat-input {
    @apply p-4 border-t border-amber-400 border-opacity-30 flex gap-3;
  }

  .yorha-textarea {
    @apply flex-1 bg-black border border-amber-400 text-amber-400 p-3 font-mono text-sm;
    @apply focus:outline-none focus:border-amber-300 resize-none;
  }

  .yorha-send-btn {
    @apply bg-amber-400 text-black px-6 py-2 font-mono text-sm hover:bg-amber-300;
    @apply disabled:opacity-50 disabled:cursor-not-allowed;
  }

  .yorha-spinner {
    @apply w-4 h-4 border-2 border-amber-400 border-t-transparent rounded-full animate-spin;
  }

  /* Animations */
  @keyframes scanline {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
  }

  @keyframes pulse-green {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
  }

  @keyframes pulse-yellow {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
  }

  /* Responsive Design */
  @media (max-width: 768px) {
    .yorha-hero-title {
      @apply text-4xl;
    }
    
    .yorha-title-sub {
      @apply text-lg;
    }
    
    .yorha-status-grid {
      @apply grid-cols-2 gap-4;
    }
    
    .yorha-category-tabs {
      @apply grid grid-cols-2 gap-2;
    }
    
    .yorha-routes-grid {
      @apply grid-cols-1 gap-4;
    }
    
    .yorha-services-grid {
      @apply grid-cols-1 gap-4;
    }

    .yorha-chat-dialog {
      @apply w-[95vw] h-[80vh];
    }
  }

  /* Custom scrollbar */
  :global(.overflow-y-auto::-webkit-scrollbar) {
    width: 6px;
  }

  :global(.overflow-y-auto::-webkit-scrollbar-track) {
    background: #1f2937;
  }

  :global(.overflow-y-auto::-webkit-scrollbar-thumb) {
    background: #ffbf00;
    border-radius: 3px;
  }

  :global(.overflow-y-auto::-webkit-scrollbar-thumb:hover) {
    background: #ffd700;
  }
</style>
