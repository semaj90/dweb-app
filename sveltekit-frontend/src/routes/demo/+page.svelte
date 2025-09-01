<!--
  Demo Hub - Legal AI Platform
  Comprehensive navigation for all demonstrations and testing interfaces
-->
<script lang="ts">
  import { goto } from '$app/navigation';
  import * as Card from '$lib/components/ui/card';
  import { Button } from '$lib/components/ui/button';
  import { Input } from '$lib/components/ui/input';
  import { Badge } from '$lib/components/ui/badge';
  import { 
    Search, Filter, Grid, List, 
    Zap, Bot, Shield, Database, 
    Cpu, FileText, Users, Settings,
    Palette, Code, TestTube, Monitor,
    Brain, Eye, Layers, Workflow,
    Upload, MessageSquare, BarChart3,
    Network, Cog, Play, Star,
    ChevronRight, ExternalLink
  } from 'lucide-svelte';

  // Demo categories and routes
  const demoCategories = [
    {
      id: 'ai-assistant',
      title: 'AI Assistant & Chat',
      icon: Bot,
      description: 'AI-powered chat interfaces, assistants, and conversation management',
      demos: [
        {
          name: 'AI Assistant',
          path: '/demo/ai-assistant',
          description: 'Primary AI assistant interface with Ollama integration',
          features: ['Chat Interface', 'Context Awareness', 'Legal AI'],
          status: 'production'
        },
        {
          name: 'AI Complete Test',
          path: '/demo/ai-complete-test',
          description: 'Complete AI system testing and validation',
          features: ['System Testing', 'Integration Tests', 'Performance'],
          status: 'stable'
        },
        {
          name: 'AI Dashboard',
          path: '/demo/ai-dashboard',
          description: 'Comprehensive AI system monitoring dashboard',
          features: ['Metrics', 'Real-time Status', 'Analytics'],
          status: 'production'
        },
        {
          name: 'AI Integration',
          path: '/demo/ai-integration',
          description: 'AI service integration and orchestration',
          features: ['Service Discovery', 'Load Balancing', 'Failover'],
          status: 'production'
        },
        {
          name: 'AI Pipeline',
          path: '/demo/ai-pipeline',
          description: 'AI processing pipeline and workflow management',
          features: ['Pipeline Management', 'Data Flow', 'Monitoring'],
          status: 'stable'
        },
        {
          name: 'AI Summary',
          path: '/demo/ai-summary',
          description: 'Document summarization and analysis',
          features: ['Text Summarization', 'Key Extraction', 'Analysis'],
          status: 'production'
        },
        {
          name: 'AI Test',
          path: '/demo/ai-test',
          description: 'AI system testing interface',
          features: ['Unit Tests', 'Integration Tests', 'Benchmarks'],
          status: 'development'
        }
      ]
    },
    {
      id: 'authentication',
      title: 'Authentication & Security',
      icon: Shield,
      description: 'Authentication flows, security testing, and user management',
      demos: [
        {
          name: 'XState Authentication Demo',
          path: '/demo/xstate-auth',
          description: 'Complete XState authentication flow with GPU orchestration',
          features: ['XState Integration', 'GPU Security', 'Session Management'],
          status: 'production'
        }
      ]
    },
    {
      id: 'gpu-processing',
      title: 'GPU & Processing',
      icon: Cpu,
      description: 'GPU acceleration, CUDA processing, and performance optimization',
      demos: [
        {
          name: 'GPU Legal AI',
          path: '/demo/gpu-legal-ai',
          description: 'GPU-accelerated legal document processing',
          features: ['CUDA Processing', 'Legal Analysis', 'Performance'],
          status: 'production'
        },
        {
          name: 'GPU Legal AI - Law PDFs',
          path: '/demo/gpu-legal-ai/lawpdfs',
          description: 'Specialized PDF processing for legal documents',
          features: ['PDF Processing', 'OCR', 'Text Extraction'],
          status: 'stable'
        },
        {
          name: 'WebGPU Acceleration',
          path: '/demo/webgpu-acceleration',
          description: 'Browser-based GPU acceleration demonstration',
          features: ['WebGPU', 'Browser Processing', 'Real-time'],
          status: 'experimental'
        }
      ]
    },
    {
      id: 'document-processing',
      title: 'Document Processing',
      icon: FileText,
      description: 'Document analysis, processing, and management systems',
      demos: [
        {
          name: 'Document AI',
          path: '/demo/document-ai',
          description: 'AI-powered document analysis and processing',
          features: ['Document Analysis', 'AI Processing', 'Metadata'],
          status: 'production'
        },
        {
          name: 'Language Extract Ollama',
          path: '/demo/langextract-ollama',
          description: 'Language extraction using Ollama models',
          features: ['Language Detection', 'Text Extraction', 'AI Models'],
          status: 'stable'
        },
        {
          name: 'Professional Editor',
          path: '/demo/professional-editor',
          description: 'Advanced text editor for legal documents',
          features: ['Rich Text', 'Legal Templates', 'Collaboration'],
          status: 'production'
        }
      ]
    },
    {
      id: 'vector-search',
      title: 'Vector Search & RAG',
      icon: Database,
      description: 'Vector databases, semantic search, and RAG implementations',
      demos: [
        {
          name: 'Vector Search',
          path: '/demo/vector-search',
          description: 'Vector similarity search and retrieval',
          features: ['Semantic Search', 'Vector DB', 'Similarity'],
          status: 'production'
        },
        {
          name: 'Vector Intelligence',
          path: '/demo/vector-intelligence',
          description: 'Advanced vector intelligence and analytics',
          features: ['Vector Analytics', 'Intelligence', 'Insights'],
          status: 'production'
        },
        {
          name: 'Enhanced RAG Semantic',
          path: '/demo/enhanced-rag-semantic',
          description: 'Enhanced retrieval-augmented generation with semantics',
          features: ['RAG', 'Semantic Enhancement', 'Context'],
          status: 'production'
        }
      ]
    },
    {
      id: 'agents-workflow',
      title: 'Agents & Workflow',
      icon: Workflow,
      description: 'Multi-agent systems, workflow automation, and orchestration',
      demos: [
        {
          name: 'Live Agents',
          path: '/demo/live-agents',
          description: 'Real-time multi-agent system demonstration',
          features: ['Multi-Agent', 'Real-time', 'Orchestration'],
          status: 'production'
        }
      ]
    },
    {
      id: 'ui-components',
      title: 'UI Components',
      icon: Palette,
      description: 'Component libraries, design systems, and interface elements',
      demos: [
        {
          name: 'Component Gallery',
          path: '/demo/component-gallery',
          description: 'Comprehensive UI component showcase',
          features: ['Bits UI', 'Melt UI', 'Components'],
          status: 'production'
        },
        {
          name: 'UnoCSS Svelte 5',
          path: '/demo/unocss-svelte5',
          description: 'UnoCSS utility classes with Svelte 5',
          features: ['UnoCSS', 'Svelte 5', 'Styling'],
          status: 'stable'
        },
        {
          name: 'YoRHa Tables',
          path: '/demo/yorha-tables',
          description: 'YoRHa-themed data tables and grids',
          features: ['Data Tables', 'YoRHa Theme', 'Responsive'],
          status: 'production'
        }
      ]
    },
    {
      id: 'system-architecture',
      title: 'System Architecture',
      icon: Network,
      description: 'System integration, architecture demos, and infrastructure',
      demos: [
        {
          name: 'Enhanced Semantic Architecture',
          path: '/demo/enhanced-semantic-architecture',
          description: 'Advanced semantic system architecture',
          features: ['Semantic Architecture', 'System Design', 'Integration'],
          status: 'production'
        },
        {
          name: 'Unified Architecture',
          path: '/demo/unified-architecture',
          description: 'Unified system architecture demonstration',
          features: ['Unified Design', 'Architecture', 'Integration'],
          status: 'production'
        },
        {
          name: 'System Summary',
          path: '/demo/system-summary',
          description: 'Complete system overview and status',
          features: ['System Overview', 'Status', 'Monitoring'],
          status: 'production'
        },
        {
          name: 'Integrated System',
          path: '/demo/integrated-system',
          description: 'Full system integration demonstration',
          features: ['Full Integration', 'End-to-End', 'Production'],
          status: 'production'
        }
      ]
    },
    {
      id: 'specialized',
      title: 'Specialized Tools',
      icon: Settings,
      description: 'Specialized tools, engines, and advanced features',
      demos: [
        {
          name: 'Neural Sprite Engine',
          path: '/demo/neural-sprite-engine',
          description: 'Advanced neural sprite rendering engine',
          features: ['Neural Networks', 'Sprites', 'Rendering'],
          status: 'experimental'
        },
        {
          name: 'Legal AI Complete',
          path: '/demo/legal-ai-complete',
          description: 'Complete legal AI system showcase',
          features: ['Legal AI', 'Complete System', 'Production'],
          status: 'production'
        },
        {
          name: 'Phase 5',
          path: '/demo/phase5',
          description: 'Phase 5 development milestones',
          features: ['Development', 'Milestones', 'Features'],
          status: 'development'
        },
        {
          name: 'Notes',
          path: '/demo/notes',
          description: 'Note-taking and documentation system',
          features: ['Notes', 'Documentation', 'Collaboration'],
          status: 'stable'
        },
        {
          name: 'Inline Suggestions',
          path: '/demo/inline-suggestions',
          description: 'Inline suggestion and completion system',
          features: ['Suggestions', 'Inline', 'AI Assistance'],
          status: 'stable'
        },
        {
          name: 'Simple Test',
          path: '/demo/simple-test',
          description: 'Basic system testing interface',
          features: ['Simple Testing', 'Basic', 'Validation'],
          status: 'stable'
        }
      ]
    }
  ];

  // State management
  let searchTerm = $state('');
  let selectedCategory = $state('all');
  let viewMode = $state<'grid' | 'list'>('grid');

  // Computed values
  const filteredCategories = $derived.by(() => {
    if (selectedCategory === 'all') {
      return demoCategories.map(category => ({
        ...category,
        demos: category.demos.filter(demo =>
          demo.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
          demo.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
          demo.features.some(feature => feature.toLowerCase().includes(searchTerm.toLowerCase()))
        )
      })).filter(category => category.demos.length > 0);
    } else {
      const category = demoCategories.find(cat => cat.id === selectedCategory);
      if (!category) return [];
      return [{
        ...category,
        demos: category.demos.filter(demo =>
          demo.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
          demo.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
          demo.features.some(feature => feature.toLowerCase().includes(searchTerm.toLowerCase()))
        )
      }].filter(category => category.demos.length > 0);
    }
  });

  const totalDemos = $derived.by(() => 
    demoCategories.reduce((total, category) => total + category.demos.length, 0)
  );

  const productionDemos = $derived.by(() =>
    demoCategories.reduce((total, category) => 
      total + category.demos.filter(demo => demo.status === 'production').length, 0)
  );

  // Functions
  function navigateToDemo(path: string) {
    goto(path);
  }

  function getStatusVariant(status: string): "default" | "secondary" | "destructive" | "outline" {
    switch (status) {
      case 'production': return 'default';
      case 'stable': return 'secondary';
      case 'development': return 'outline';
      case 'experimental': return 'destructive';
      default: return 'outline';
    }
  }

  function getStatusColor(status: string): string {
    switch (status) {
      case 'production': return 'text-green-600';
      case 'stable': return 'text-blue-600';
      case 'development': return 'text-orange-600';
      case 'experimental': return 'text-red-600';
      default: return 'text-gray-600';
    }
  }
</script>

<svelte:head>
  <title>Demo Hub - Legal AI Platform</title>
  <meta name="description" content="Comprehensive demonstration hub for the Legal AI Platform featuring AI assistants, authentication, GPU processing, and more" />
</svelte:head>

<div class="min-h-screen bg-background">
  <!-- Header -->
  <div class="border-b bg-card/50 backdrop-blur-sm sticky top-0 z-50">
    <div class="max-w-7xl mx-auto px-6 py-6">
      <div class="flex items-center justify-between mb-6">
        <div>
          <h1 class="text-3xl font-bold tracking-tight">Demo Hub</h1>
          <p class="text-muted-foreground mt-2">
            Explore {totalDemos} demonstrations across all platform features and systems
          </p>
        </div>
        <div class="flex items-center gap-3">
          <Badge variant="outline" class="text-sm">
            {productionDemos} Production Ready
          </Badge>
          <Button on:on:click={() => goto('/')} variant="outline">
            <ChevronRight class="h-4 w-4 mr-2 rotate-180" />
            Back to Home
          </Button>
        </div>
      </div>

      <!-- Search and Filters -->
      <div class="flex flex-col sm:flex-row gap-4">
        <div class="relative flex-1 max-w-md">
          <Search class="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            type="text"
            placeholder="Search demos, features, or descriptions..."
            bind:value={searchTerm}
            class="pl-10"
          />
        </div>
        
        <div class="flex items-center gap-2">
          <Filter class="h-4 w-4 text-muted-foreground" />
          <select 
            bind:value={selectedCategory}
            class="px-3 py-2 border border-input rounded-md text-sm bg-background"
          >
            <option value="all">All Categories</option>
            {#each demoCategories as category}
              <option value={category.id}>{category.title}</option>
            {/each}
          </select>
        </div>

        <div class="flex items-center border rounded-lg p-1">
          <Button
            variant={viewMode === 'grid' ? 'default' : 'ghost'}
            size="sm"
            on:on:click={() => viewMode = 'grid'}
          >
            <Grid class="h-4 w-4" />
          </Button>
          <Button
            variant={viewMode === 'list' ? 'default' : 'ghost'}
            size="sm"
            on:on:click={() => viewMode = 'list'}
          >
            <List class="h-4 w-4" />
          </Button>
        </div>
      </div>
    </div>
  </div>

  <!-- Content -->
  <div class="max-w-7xl mx-auto px-6 py-8">
    {#if filteredCategories.length === 0}
      <!-- No Results -->
      <div class="text-center py-12">
        <Search class="h-12 w-12 text-muted-foreground mx-auto mb-4" />
        <h3 class="text-lg font-semibold mb-2">No demos found</h3>
        <p class="text-muted-foreground">Try adjusting your search or filter criteria</p>
      </div>
    {:else}
      <!-- Demo Categories -->
      <div class="space-y-12">
        {#each filteredCategories as category}
          <section>
            <!-- Category Header -->
            <div class="flex items-center gap-4 mb-6">
              <div class="flex items-center gap-3">
                <div class="p-2 rounded-lg bg-primary/10 text-primary">
                  <svelte:component this={category.icon} class="h-6 w-6" />
                </div>
                <div>
                  <h2 class="text-2xl font-semibold">{category.title}</h2>
                  <p class="text-muted-foreground">{category.description}</p>
                </div>
              </div>
              <Badge variant="outline" class="ml-auto">
                {category.demos.length} demo{category.demos.length !== 1 ? 's' : ''}
              </Badge>
            </div>

            <!-- Demos Grid/List -->
            {#if viewMode === 'grid'}
              <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {#each category.demos as demo}
                  <Card.Root class="group hover:shadow-lg transition-all duration-200 cursor-pointer border-2 hover:border-primary/20">
                    <Card.Header class="pb-3">
                      <div class="flex items-start justify-between">
                        <div class="flex-1">
                          <Card.Title class="text-lg group-hover:text-primary transition-colors">
                            {demo.name}
                          </Card.Title>
                          <div class="flex items-center gap-2 mt-2">
                            <Badge variant={getStatusVariant(demo.status)} class="text-xs">
                              {demo.status}
                            </Badge>
                          </div>
                        </div>
                        <ExternalLink class="h-4 w-4 text-muted-foreground group-hover:text-primary transition-colors" />
                      </div>
                    </Card.Header>
                    
                    <Card.Content class="pt-0">
                      <p class="text-sm text-muted-foreground mb-4 line-clamp-2">
                        {demo.description}
                      </p>
                      
                      <div class="flex flex-wrap gap-1 mb-4">
                        {#each demo.features.slice(0, 3) as feature}
                          <Badge variant="outline" class="text-xs">
                            {feature}
                          </Badge>
                        {/each}
                        {#if demo.features.length > 3}
                          <Badge variant="outline" class="text-xs">
                            +{demo.features.length - 3} more
                          </Badge>
                        {/if}
                      </div>
                      
                      <Button 
                        on:on:click={() => navigateToDemo(demo.path)}
                        class="w-full group-hover:bg-primary group-hover:text-primary-foreground transition-colors"
                        variant="outline"
                      >
                        <Play class="h-4 w-4 mr-2" />
                        Launch Demo
                      </Button>
                    </Card.Content>
                  </Card.Root>
                {/each}
              </div>
            {:else}
              <!-- List View -->
              <div class="space-y-3">
                {#each category.demos as demo}
                  <Card.Root class="group hover:shadow-md transition-all duration-200">
                    <Card.Content class="p-4">
                      <div class="flex items-center justify-between">
                        <div class="flex-1">
                          <div class="flex items-center gap-4">
                            <div class="flex-1">
                              <h3 class="font-semibold group-hover:text-primary transition-colors">
                                {demo.name}
                              </h3>
                              <p class="text-sm text-muted-foreground mt-1">
                                {demo.description}
                              </p>
                              <div class="flex items-center gap-2 mt-2">
                                <Badge variant={getStatusVariant(demo.status)} class="text-xs">
                                  {demo.status}
                                </Badge>
                                <div class="flex gap-1">
                                  {#each demo.features.slice(0, 3) as feature}
                                    <Badge variant="outline" class="text-xs">
                                      {feature}
                                    </Badge>
                                  {/each}
                                </div>
                              </div>
                            </div>
                            <Button 
                              on:on:click={() => navigateToDemo(demo.path)}
                              variant="outline"
                              size="sm"
                            >
                              <Play class="h-4 w-4 mr-2" />
                              Launch
                            </Button>
                          </div>
                        </div>
                      </div>
                    </Card.Content>
                  </Card.Root>
                {/each}
              </div>
            {/if}
          </section>
        {/each}
      </div>
    {/if}
  </div>
</div>

<style>
  .line-clamp-2 {
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }
</style>