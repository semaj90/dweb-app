<!--
Vector Intelligence Demo Component
Comprehensive showcase of Phase 4 Vector Intelligence capabilities
-->
<script lang="ts">
  import { onMount } from 'svelte';
  import { Button } from '$lib/components/ui/button';
  import { Card, CardContent, CardHeader, CardTitle } from '$lib/components/ui/card';
  import { Badge } from '$lib/components/ui/badge';
  import { Textarea } from '$lib/components/ui/textarea';
  import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '$lib/components/ui/select';
  import { Separator } from '$lib/components/ui/separator';
  import {
    Brain,
    Search,
    Lightbulb,
    AlertTriangle,
    TrendingUp,
    Users,
    FileText,
    Target,
    Zap,
    Activity,
    BarChart3,
    Network,
    Eye,
    Settings,
    Play,
    Pause,
    RefreshCw,
    Download,
    Share,
    CheckCircle,
    Clock,
    Star,
    Tag,
    Link
  } from 'lucide-svelte';

  import { vectorIntelligenceService } from '$lib/services/vector-intelligence-service.js';
  import type {
    VectorSearchResult,
    IntelligenceRecommendation,
    SemanticAnalysisResult,
    VectorIntelligenceState
  } from '$lib/services/vector-intelligence-service.js';

  // Component state
  let searchQuery = $state('');
  let analysisContent = $state('');
  let recommendationContext = $state('');
  let selectedMode = $state<'search' | 'recommendations' | 'analysis'>('search');
  let selectedUserRole = $state<'prosecutor' | 'detective' | 'admin' | 'user'>('prosecutor');
  let selectedCaseType = $state<'criminal' | 'civil' | 'corporate' | 'general'>('criminal');

  // Results state
  let searchResults = $state<VectorSearchResult[]>([]);
  let recommendations = $state<IntelligenceRecommendation[]>([]);
  let semanticAnalysis = $state<SemanticAnalysisResult | null>(null);
  let systemHealth = $state<VectorIntelligenceState | null>(null);

  // UI state
  let isProcessing = $state(false);
  let processingStage = $state('');
  let activeTab = $state<'search' | 'recommendations' | 'analysis' | 'health'>('search');
  let showAdvancedOptions = $state(false);

  // Demo data
  const demoSearchQueries = [
    {
      query: 'contract liability clauses in employment agreements',
      description: 'Legal contract analysis for employment disputes',
      category: 'Legal Research'
    },
    {
      query: 'evidence tampering patterns in criminal investigations',
      description: 'Criminal investigation methodology',
      category: 'Investigation'
    },
    {
      query: 'corporate compliance violations and penalties',
      description: 'Corporate law and regulatory compliance',
      category: 'Compliance'
    },
    {
      query: 'witness testimony consistency analysis methods',
      description: 'Evidence evaluation techniques',
      category: 'Evidence Analysis'
    }
  ];

  const demoRecommendationContexts = [
    {
      context: 'I need to prepare a comprehensive case strategy for a high-profile criminal trial involving multiple defendants and complex evidence chains.',
      role: 'prosecutor',
      description: 'Complex criminal case preparation'
    },
    {
      context: 'Our investigation has uncovered potential digital evidence tampering. How should we proceed with forensic analysis and evidence preservation?',
      role: 'detective',
      description: 'Digital forensics investigation'
    },
    {
      context: 'We need to optimize our case management workflow to handle the increasing caseload more efficiently while maintaining quality.',
      role: 'admin',
      description: 'Workflow optimization analysis'
    }
  ];

  const demoAnalysisContent = [
    {
      content: `EMPLOYMENT AGREEMENT

This Employment Agreement ("Agreement") is entered into on January 15, 2024, between TechCorp Industries, a Delaware corporation ("Company"), and John Smith ("Employee").

1. POSITION AND DUTIES
Employee shall serve as Senior Software Engineer and shall perform such duties as assigned by the Company. Employee agrees to devote full business time and attention to the Company's business.

2. COMPENSATION
Company shall pay Employee a base salary of $150,000 per year, payable in accordance with Company's standard payroll practices.

3. CONFIDENTIALITY
Employee acknowledges that during employment, Employee may have access to confidential information. Employee agrees to maintain strict confidentiality.

4. TERMINATION
This Agreement may be terminated by either party with thirty (30) days written notice.`,
      description: 'Employment contract for legal analysis',
      type: 'Contract'
    },
    {
      content: `INCIDENT REPORT - Case #2024-CR-1892

Date: March 8, 2024
Location: 1425 Oak Street, Downtown District
Reporting Officer: Detective Sarah Johnson

SUMMARY:
Responded to reports of suspected break-in at residential property. Upon arrival, discovered evidence of forced entry through rear window. Victim John Doe reported missing electronics valued at approximately $3,500.

EVIDENCE COLLECTED:
- Fingerprints from window frame
- Footprint impressions in garden
- Security camera footage from neighboring property
- Witness statements from two neighbors

SUSPECTS:
Investigation ongoing. Similar pattern matches recent break-ins in the area.`,
      description: 'Criminal incident report for analysis',
      type: 'Incident Report'
    }
  ];

  onMount(async () => {
    await loadSystemHealth();
  });

  async function loadSystemHealth() {
    try {
      systemHealth = await vectorIntelligenceService.getSystemHealth();
    } catch (error) {
      console.error('Failed to load system health:', error);
    }
  }

  async function performSearch() {
    if (!searchQuery.trim() || isProcessing) return;

    isProcessing = true;
    processingStage = 'Performing semantic search...';
    searchResults = [];

    try {
      const results = await vectorIntelligenceService.semanticSearch({
        query: searchQuery,
        threshold: 0.7,
        limit: 10,
        includeMetadata: true,
        contextFilter: {
          evidenceType: selectedCaseType
        }
      });

      searchResults = results;
      processingStage = 'Search completed';

    } catch (error) {
      console.error('Search failed:', error);
      processingStage = `Search failed: ${error.message}`;
    } finally {
      isProcessing = false;
    }
  }

  async function generateRecommendations() {
    if (!recommendationContext.trim() || isProcessing) return;

    isProcessing = true;
    processingStage = 'Generating intelligent recommendations...';
    recommendations = [];

    try {
      const result = await vectorIntelligenceService.generateRecommendations({
        context: recommendationContext,
        userProfile: {
          role: selectedUserRole,
          experience: 'senior',
          specialization: ['legal-analysis', 'case-management']
        },
        currentCase: {
          id: 'DEMO-2024-001',
          type: selectedCaseType,
          priority: 'high',
          status: 'active'
        },
        preferences: {
          preferredActions: ['research', 'analysis', 'documentation'],
          workflowStyle: 'systematic'
        }
      });

      recommendations = result;
      processingStage = 'Recommendations generated';

    } catch (error) {
      console.error('Recommendation generation failed:', error);
      processingStage = `Failed: ${error.message}`;
    } finally {
      isProcessing = false;
    }
  }

  async function performSemanticAnalysis() {
    if (!analysisContent.trim() || isProcessing) return;

    isProcessing = true;
    processingStage = 'Performing semantic analysis...';
    semanticAnalysis = null;

    try {
      const result = await vectorIntelligenceService.analyzeSemantics(analysisContent);
      semanticAnalysis = result;
      processingStage = 'Analysis completed';

    } catch (error) {
      console.error('Semantic analysis failed:', error);
      processingStage = `Analysis failed: ${error.message}`;
    } finally {
      isProcessing = false;
    }
  }

  function loadDemoQuery(query: string) {
    searchQuery = query;
    activeTab = 'search';
  }

  function loadDemoContext(context: string, role: string) {
    recommendationContext = context;
    selectedUserRole = role as any;
    activeTab = 'recommendations';
  }

  function loadDemoContent(content: string) {
    analysisContent = content;
    activeTab = 'analysis';
  }

  function getRecommendationIcon(type: string) {
    switch (type) {
      case 'action': return Target;
      case 'insight': return Lightbulb;
      case 'warning': return AlertTriangle;
      case 'opportunity': return TrendingUp;
      default: return FileText;
    }
  }

  function getRecommendationColor(type: string) {
    switch (type) {
      case 'action': return 'recommendation-action';
      case 'insight': return 'recommendation-insight';
      case 'warning': return 'recommendation-warning';
      case 'opportunity': return 'recommendation-opportunity';
      default: return 'recommendation-item';
    }
  }

  function getEntityIcon(type: string) {
    switch (type) {
      case 'person': return Users;
      case 'organization': return Network;
      case 'location': return Target;
      case 'date': return Clock;
      case 'legal_concept': return FileText;
      default: return Tag;
    }
  }

  function getEntityColor(type: string) {
    switch (type) {
      case 'person': return 'semantic-entity-person';
      case 'organization': return 'semantic-entity-organization';
      case 'location': return 'semantic-entity-location';
      case 'date': return 'semantic-entity-date';
      case 'legal_concept': return 'semantic-entity-legal';
      default: return 'semantic-entity-tag';
    }
  }

  function getConfidenceColor(confidence: number) {
    if (confidence >= 0.8) return 'vector-confidence-high';
    if (confidence >= 0.6) return 'vector-confidence-medium';
    return 'vector-confidence-low';
  }

  function getHealthColor(health: string) {
    switch (health) {
      case 'excellent': return 'text-green-600';
      case 'good': return 'text-blue-600';
      case 'fair': return 'text-yellow-600';
      case 'poor': return 'text-red-600';
      default: return 'text-gray-600';
    }
  }
</script>

<div class="w-full max-w-7xl mx-auto p-6 space-y-6">
  <!-- Header -->
  <div class="text-center space-y-4">
    <h1 class="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
      Vector Intelligence Demo
    </h1>
    <p class="text-lg text-muted-foreground max-w-3xl mx-auto">
      Experience the power of Phase 4 Vector Intelligence with semantic search, AI recommendations,
      and advanced document analysis for legal professionals.
    </p>

    <div class="flex items-center justify-center gap-4 flex-wrap">
      <Badge class="bits-badge-default flex items-center gap-1">
        <Brain class="h-3 w-3" />
        Vector Intelligence
      </Badge>
      <Badge class="bits-badge-secondary flex items-center gap-1">
        <Search class="h-3 w-3" />
        Semantic Search
      </Badge>
      <Badge class="bits-badge-secondary flex items-center gap-1">
        <Lightbulb class="h-3 w-3" />
        AI Recommendations
      </Badge>
      <Badge class="bits-badge-secondary flex items-center gap-1">
        <Activity class="h-3 w-3" />
        Real-time Analysis
      </Badge>
    </div>
  </div>

  <!-- Demo Examples -->
  <Card class="shadcn-card">
    <CardHeader class="shadcn-card-header">
      <CardTitle class="flex items-center gap-2">
        <Play class="h-5 w-5" />
        Quick Demo Examples
      </CardTitle>
    </CardHeader>
    <CardContent class="shadcn-card-content">
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        <!-- Search Examples -->
        <div class="space-y-3">
          <h3 class="font-semibold text-sm flex items-center gap-2">
            <Search class="h-4 w-4" />
            Semantic Search
          </h3>
          {#each demoSearchQueries as example}
            <div class="demo-example-card" on:onclick={() => loadDemoQuery(example.query)}>
              <h4 class="font-medium text-sm mb-1">{example.category}</h4>
              <p class="text-xs text-muted-foreground mb-2">{example.description}</p>
              <p class="text-xs bg-muted p-2 rounded font-mono">{example.query}</p>
            </div>
          {/each}
        </div>

        <!-- Recommendation Examples -->
        <div class="space-y-3">
          <h3 class="font-semibold text-sm flex items-center gap-2">
            <Lightbulb class="h-4 w-4" />
            AI Recommendations
          </h3>
          {#each demoRecommendationContexts as example}
            <div class="demo-example-card" on:onclick={() => loadDemoContext(example.context, example.role)}>
              <h4 class="font-medium text-sm mb-1">{example.description}</h4>
              <Badge class="bits-badge-outline text-xs mb-2">{example.role}</Badge>
              <p class="text-xs text-muted-foreground">{example.context.substring(0, 80)}...</p>
            </div>
          {/each}
        </div>

        <!-- Analysis Examples -->
        <div class="space-y-3">
          <h3 class="font-semibold text-sm flex items-center gap-2">
            <FileText class="h-4 w-4" />
            Document Analysis
          </h3>
          {#each demoAnalysisContent as example}
            <div class="demo-example-card" on:onclick={() => loadDemoContent(example.content)}>
              <h4 class="font-medium text-sm mb-1">{example.type}</h4>
              <p class="text-xs text-muted-foreground mb-2">{example.description}</p>
              <p class="text-xs bg-muted p-2 rounded">{example.content.substring(0, 60)}...</p>
            </div>
          {/each}
        </div>
      </div>
    </CardContent>
  </Card>

  <!-- Tab Navigation -->
  <div class="flex items-center justify-center space-x-1 bg-muted p-1 rounded-lg w-fit mx-auto">
    <Button
      variant={activeTab === 'search' ? 'default' : 'ghost'}
      size="sm"
      on:on:click={() => activeTab = 'search'}
      class="flex items-center gap-2"
    >
      <Search class="h-4 w-4" />
      Search
    </Button>
    <Button
      variant={activeTab === 'recommendations' ? 'default' : 'ghost'}
      size="sm"
      on:on:click={() => activeTab = 'recommendations'}
      class="flex items-center gap-2"
    >
      <Lightbulb class="h-4 w-4" />
      Recommendations
    </Button>
    <Button
      variant={activeTab === 'analysis' ? 'default' : 'ghost'}
      size="sm"
      on:on:click={() => activeTab = 'analysis'}
      class="flex items-center gap-2"
    >
      <BarChart3 class="h-4 w-4" />
      Analysis
    </Button>
    <Button
      variant={activeTab === 'health' ? 'default' : 'ghost'}
      size="sm"
      on:on:click={() => activeTab = 'health'}
      class="flex items-center gap-2"
    >
      <Activity class="h-4 w-4" />
      Health
    </Button>
  </div>

  <!-- Processing Indicator -->
  {#if isProcessing}
    <Card class="shadcn-card">
      <CardContent class="shadcn-card-content">
        <div class="flex items-center justify-center space-x-3 py-8">
          <div class="animate-spin h-6 w-6 border border-primary border-t-transparent rounded-full"></div>
          <span class="text-muted-foreground">{processingStage}</span>
        </div>
      </CardContent>
    </Card>
  {/if}

  <!-- Tab Content -->
  {#if activeTab === 'search'}
    <!-- Semantic Search Tab -->
    <div class="space-y-6">
      <Card class="shadcn-card">
        <CardHeader class="shadcn-card-header">
          <CardTitle class="flex items-center gap-2">
            <Search class="h-5 w-5" />
            Semantic Vector Search
          </CardTitle>
        </CardHeader>
        <CardContent class="shadcn-card-content space-y-4">
          <div class="space-y-2">
            <label class="bits-label">Search Query</label>
            <Textarea
              bind:value={searchQuery}
              placeholder="Enter your legal search query..."
              class="vector-search-input min-h-[80px]"
            />
          </div>

          {#if showAdvancedOptions}
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div class="space-y-2">
                <label class="bits-label">Case Type Filter</label>
                <Select bind:value={selectedCaseType}>
                  <SelectTrigger class="bits-select-trigger">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="criminal">Criminal</SelectItem>
                    <SelectItem value="civil">Civil</SelectItem>
                    <SelectItem value="corporate">Corporate</SelectItem>
                    <SelectItem value="general">General</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
          {/if}

          <div class="flex items-center gap-2">
            <Button
              on:on:click={performSearch}
              disabled={isProcessing || !searchQuery.trim()}
              class="bits-btn-default"
            >
              {#if isProcessing}
                <Pause class="h-4 w-4 mr-2" />
                Searching...
              {:else}
                <Search class="h-4 w-4 mr-2" />
                Search
              {/if}
            </Button>
            <Button
              variant="outline"
              size="sm"
              on:on:click={() => showAdvancedOptions = !showAdvancedOptions}
            >
              <Settings class="h-4 w-4" />
            </Button>
          </div>
        </CardContent>
      </Card>

      <!-- Search Results -->
      {#if searchResults.length > 0}
        <Card class="shadcn-card">
          <CardHeader class="shadcn-card-header">
            <CardTitle class="flex items-center justify-between">
              <span class="flex items-center gap-2">
                <Eye class="h-5 w-5" />
                Search Results ({searchResults.length})
              </span>
              <Button variant="outline" size="sm">
                <Download class="h-4 w-4" />
              </Button>
            </CardTitle>
          </CardHeader>
          <CardContent class="shadcn-card-content">
            <div class="space-y-4">
              {#each searchResults as result}
                <div class="vector-result-item">
                  <div class="flex items-start justify-between mb-2">
                    <h3 class="font-semibold text-sm">{result.id}</h3>
                    <div class="flex items-center gap-2">
                      <Badge class={getConfidenceColor(result.similarity)}>
                        {Math.round(result.similarity * 100)}%
                      </Badge>
                      <Badge class="bits-badge-outline text-xs">{result.source}</Badge>
                    </div>
                  </div>

                  <p class="text-sm text-muted-foreground mb-3">
                    {result.content.substring(0, 200)}...
                  </p>

                  {#if result.highlights?.length > 0}
                    <div class="space-y-1">
                      <p class="text-xs font-medium">Highlights:</p>
                      {#each result.highlights as highlight}
                        <p class="text-xs bg-muted p-2 rounded">
                          <span class="vector-highlight">{highlight}</span>
                        </p>
                      {/each}
                    </div>
                  {/if}

                  <div class="vector-metadata-grid mt-3">
                    <span>Relevance: {result.relevanceScore.toFixed(2)}</span>
                    <span>Similarity: {result.similarity.toFixed(3)}</span>
                    <span>Source: {result.source}</span>
                    <span>ID: {result.id.substring(0, 8)}...</span>
                  </div>
                </div>
              {/each}
            </div>
          </CardContent>
        </Card>
      {/if}
    </div>

  {:else if activeTab === 'recommendations'}
    <!-- AI Recommendations Tab -->
    <div class="space-y-6">
      <Card class="shadcn-card">
        <CardHeader class="shadcn-card-header">
          <CardTitle class="flex items-center gap-2">
            <Lightbulb class="h-5 w-5" />
            AI Recommendations
          </CardTitle>
        </CardHeader>
        <CardContent class="shadcn-card-content space-y-4">
          <div class="space-y-2">
            <label class="bits-label">Context Description</label>
            <Textarea
              bind:value={recommendationContext}
              placeholder="Describe your current situation or challenge..."
              class="vector-search-input min-h-[100px]"
            />
          </div>

          <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div class="space-y-2">
              <label class="bits-label">Your Role</label>
              <Select bind:value={selectedUserRole}>
                <SelectTrigger class="bits-select-trigger">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="prosecutor">Prosecutor</SelectItem>
                  <SelectItem value="detective">Detective</SelectItem>
                  <SelectItem value="admin">Administrator</SelectItem>
                  <SelectItem value="user">General User</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div class="space-y-2">
              <label class="bits-label">Case Type</label>
              <Select bind:value={selectedCaseType}>
                <SelectTrigger class="bits-select-trigger">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="criminal">Criminal</SelectItem>
                  <SelectItem value="civil">Civil</SelectItem>
                  <SelectItem value="corporate">Corporate</SelectItem>
                  <SelectItem value="general">General</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          <Button
            on:on:click={generateRecommendations}
            disabled={isProcessing || !recommendationContext.trim()}
            class="bits-btn-default"
          >
            {#if isProcessing}
              <Pause class="h-4 w-4 mr-2" />
              Generating...
            {:else}
              <Zap class="h-4 w-4 mr-2" />
              Generate Recommendations
            {/if}
          </Button>
        </CardContent>
      </Card>

      <!-- Recommendations Results -->
      {#if recommendations.length > 0}
        <Card class="shadcn-card">
          <CardHeader class="shadcn-card-header">
            <CardTitle class="flex items-center justify-between">
              <span class="flex items-center gap-2">
                <Target class="h-5 w-5" />
                Intelligent Recommendations ({recommendations.length})
              </span>
              <div class="flex items-center gap-2">
                <Button variant="outline" size="sm">
                  <Share class="h-4 w-4" />
                </Button>
                <Button variant="outline" size="sm">
                  <Download class="h-4 w-4" />
                </Button>
              </div>
            </CardTitle>
          </CardHeader>
          <CardContent class="shadcn-card-content">
            <div class="recommendation-container">
              {#each recommendations as rec}
                {@const SvelteComponent = getRecommendationIcon(rec.type)}
                <div class={getRecommendationColor(rec.type)}>
                  <div class="recommendation-header">
                    <div class="flex items-center gap-2">
                      <SvelteComponent class="h-4 w-4" />
                      <span class="recommendation-title">{rec.title}</span>
                    </div>
                    <div class="flex items-center gap-2">
                      <Badge class="recommendation-category">{rec.category}</Badge>
                      <Badge class={getConfidenceColor(rec.confidence)}>
                        {Math.round(rec.confidence * 100)}%
                      </Badge>
                      <Badge class="bits-badge-outline text-xs">{rec.priority}</Badge>
                    </div>
                  </div>

                  <p class="recommendation-description">{rec.description}</p>

                  {#if rec.actionItems}
                    <div class="space-y-2 mb-3">
                      {#if rec.actionItems.immediate?.length > 0}
                        <div>
                          <p class="text-xs font-medium mb-1">Immediate Actions:</p>
                          {#each rec.actionItems.immediate as action}
                            <div class="flex items-center gap-2 text-xs">
                              <CheckCircle class="h-3 w-3 text-red-500" />
                              {action}
                            </div>
                          {/each}
                        </div>
                      {/if}
                    </div>
                  {/if}

                  <div class="recommendation-actions">
                    <span>Impact: {rec.estimatedImpact?.successProbability || 'N/A'}%</span>
                    <span>Time: {rec.estimatedImpact?.timeToComplete || 'N/A'}min</span>
                    <span>Priority: {rec.priority}</span>
                  </div>
                </div>
              {/each}
            </div>
          </CardContent>
        </Card>
      {/if}
    </div>

  {:else if activeTab === 'analysis'}
    <!-- Semantic Analysis Tab -->
    <div class="space-y-6">
      <Card class="shadcn-card">
        <CardHeader class="shadcn-card-header">
          <CardTitle class="flex items-center gap-2">
            <BarChart3 class="h-5 w-5" />
            Document Semantic Analysis
          </CardTitle>
        </CardHeader>
        <CardContent class="shadcn-card-content space-y-4">
          <div class="space-y-2">
            <label class="bits-label">Document Content</label>
            <Textarea
              bind:value={analysisContent}
              placeholder="Paste your legal document or text for analysis..."
              class="vector-search-input min-h-[150px]"
            />
          </div>

          <Button
            on:on:click={performSemanticAnalysis}
            disabled={isProcessing || !analysisContent.trim()}
            class="bits-btn-default"
          >
            {#if isProcessing}
              <Pause class="h-4 w-4 mr-2" />
              Analyzing...
            {:else}
              <BarChart3 class="h-4 w-4 mr-2" />
              Analyze Document
            {/if}
          </Button>
        </CardContent>
      </Card>

      <!-- Analysis Results -->
      {#if semanticAnalysis}
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <!-- Entities -->
          <Card class="shadcn-card">
            <CardHeader class="shadcn-card-header">
              <CardTitle class="flex items-center gap-2">
                <Users class="h-5 w-5" />
                Extracted Entities ({semanticAnalysis.entities?.length || 0})
              </CardTitle>
            </CardHeader>
            <CardContent class="shadcn-card-content">
              {#if semanticAnalysis.entities?.length > 0}
                <div class="semantic-entity-container">
                  {#each semanticAnalysis.entities as entity}
                    {@const SvelteComponent_1 = getEntityIcon(entity.type)}
                    <div class={getEntityColor(entity.type)}>
                      <SvelteComponent_1 class="h-3 w-3 mr-1" />
                      {entity.text}
                      <Badge class="ml-1 text-xs">{Math.round(entity.confidence * 100)}%</Badge>
                    </div>
                  {/each}
                </div>
              {:else}
                <p class="text-sm text-muted-foreground">No entities detected</p>
              {/if}
            </CardContent>
          </Card>

          <!-- Themes -->
          <Card class="shadcn-card">
            <CardHeader class="shadcn-card-header">
              <CardTitle class="flex items-center gap-2">
                <Tag class="h-5 w-5" />
                Document Themes ({semanticAnalysis.themes?.length || 0})
              </CardTitle>
            </CardHeader>
            <CardContent class="shadcn-card-content">
              {#if semanticAnalysis.themes?.length > 0}
                <div class="space-y-3">
                  {#each semanticAnalysis.themes as theme}
                    <div class="theme-item">
                      <div>
                        <p class="theme-title">{theme.topic}</p>
                        <p class="theme-weight">Weight: {theme.weight.toFixed(2)}</p>
                      </div>
                      <div class="theme-weight-bar w-20">
                        <div
                          class="theme-weight-fill"
                          style="width: {theme.weight * 100}%"
                        ></div>
                      </div>
                    </div>
                  {/each}
                </div>
              {:else}
                <p class="text-sm text-muted-foreground">No themes identified</p>
              {/if}
            </CardContent>
          </Card>

          <!-- Relationships -->
          <Card class="shadcn-card">
            <CardHeader class="shadcn-card-header">
              <CardTitle class="flex items-center gap-2">
                <Network class="h-5 w-5" />
                Entity Relationships ({semanticAnalysis.relationships?.length || 0})
              </CardTitle>
            </CardHeader>
            <CardContent class="shadcn-card-content">
              {#if semanticAnalysis.relationships?.length > 0}
                <div class="relationship-container">
                  {#each semanticAnalysis.relationships as rel}
                    <div class="relationship-item">
                      <span class="relationship-from">{rel.from}</span>
                      <span class="relationship-type">{rel.type}</span>
                      <span class="relationship-to">{rel.to}</span>
                      <span class="relationship-strength">{rel.strength.toFixed(2)}</span>
                    </div>
                  {/each}
                </div>
              {:else}
                <p class="text-sm text-muted-foreground">No relationships detected</p>
              {/if}
            </CardContent>
          </Card>

          <!-- Complexity Metrics -->
          <Card class="shadcn-card">
            <CardHeader class="shadcn-card-header">
              <CardTitle class="flex items-center gap-2">
                <BarChart3 class="h-5 w-5" />
                Complexity Analysis
              </CardTitle>
            </CardHeader>
            <CardContent class="shadcn-card-content">
              <div class="space-y-4">
                <div class="space-y-2">
                  <div class="flex justify-between text-sm">
                    <span>Readability</span>
                    <span>{Math.round((semanticAnalysis.complexity?.readability || 0) * 100)}%</span>
                  </div>
                  <div class="theme-weight-bar">
                    <div
                      class="theme-weight-fill bg-green-500"
                      style="width: {(semanticAnalysis.complexity?.readability || 0) * 100}%"
                    ></div>
                  </div>
                </div>

                <div class="space-y-2">
                  <div class="flex justify-between text-sm">
                    <span>Technical Level</span>
                    <span>{Math.round((semanticAnalysis.complexity?.technicalLevel || 0) * 100)}%</span>
                  </div>
                  <div class="theme-weight-bar">
                    <div
                      class="theme-weight-fill bg-blue-500"
                      style="width: {(semanticAnalysis.complexity?.technicalLevel || 0) * 100}%"
                    ></div>
                  </div>
                </div>

                <div class="space-y-2">
                  <div class="flex justify-between text-sm">
                    <span>Legal Complexity</span>
                    <span>{Math.round((semanticAnalysis.complexity?.legalComplexity || 0) * 100)}%</span>
                  </div>
                  <div class="theme-weight-bar">
                    <div
                      class="theme-weight-fill bg-purple-500"
                      style="width: {(semanticAnalysis.complexity?.legalComplexity || 0) * 100}%"
                    ></div>
                  </div>
                </div>

                <Separator class="bits-separator" />

                <div class="space-y-1 text-sm">
                  <div class="flex justify-between">
                    <span>Overall Sentiment</span>
                    <span class={semanticAnalysis.sentiment?.overall >= 0 ? 'text-green-600' : 'text-red-600'}>
                      {semanticAnalysis.sentiment?.overall?.toFixed(2) || 'N/A'}
                    </span>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      {/if}
    </div>

  {:else if activeTab === 'health'}
    <!-- System Health Tab -->
    <div class="space-y-6">
      <Card class="shadcn-card">
        <CardHeader class="shadcn-card-header">
          <CardTitle class="flex items-center justify-between">
            <span class="flex items-center gap-2">
              <Activity class="h-5 w-5" />
              Vector Intelligence System Health
            </span>
            <Button variant="outline" size="sm" on:on:click={loadSystemHealth}>
              <RefreshCw class="h-4 w-4" />
            </Button>
          </CardTitle>
        </CardHeader>
        <CardContent class="shadcn-card-content">
          {#if systemHealth}
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <Card class="border">
                <CardContent class="p-4">
                  <div class="flex items-center justify-between">
                    <div>
                      <p class="text-sm font-medium">System Status</p>
                      <p class="text-2xl font-bold {getHealthColor(systemHealth.systemHealth)}">
                        {systemHealth.systemHealth}
                      </p>
                    </div>
                    <Activity class="h-8 w-8 {getHealthColor(systemHealth.systemHealth)}" />
                  </div>
                </CardContent>
              </Card>

              <Card class="border">
                <CardContent class="p-4">
                  <div class="flex items-center justify-between">
                    <div>
                      <p class="text-sm font-medium">Model Confidence</p>
                      <p class="text-2xl font-bold">
                        {Math.round(systemHealth.modelConfidence * 100)}%
                      </p>
                    </div>
                    <Star class="h-8 w-8 text-yellow-600" />
                  </div>
                </CardContent>
              </Card>

              <Card class="border">
                <CardContent class="p-4">
                  <div class="flex items-center justify-between">
                    <div>
                      <p class="text-sm font-medium">Indexed Documents</p>
                      <p class="text-2xl font-bold">
                        {systemHealth.indexedDocuments.toLocaleString()}
                      </p>
                    </div>
                    <FileText class="h-8 w-8 text-blue-600" />
                  </div>
                </CardContent>
              </Card>

              <Card class="border">
                <CardContent class="p-4">
                  <div class="flex items-center justify-between">
                    <div>
                      <p class="text-sm font-medium">Vector Dimensions</p>
                      <p class="text-2xl font-bold">
                        {systemHealth.vectorDimensions}
                      </p>
                    </div>
                    <Network class="h-8 w-8 text-purple-600" />
                  </div>
                </CardContent>
              </Card>
            </div>

            <Separator class="bits-separator my-6" />

            <div class="space-y-4">
              <h3 class="text-lg font-semibold">System Information</h3>

              <div class="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                <div class="space-y-2">
                  <div class="flex justify-between">
                    <span class="text-muted-foreground">Embedding Model:</span>
                    <span class="font-mono">{systemHealth.embeddingModel}</span>
                  </div>
                  <div class="flex justify-between">
                    <span class="text-muted-foreground">Initialization Status:</span>
                    <Badge class={systemHealth.isInitialized ? 'bits-badge-default' : 'bits-badge-destructive'}>
                      {systemHealth.isInitialized ? 'Initialized' : 'Not Initialized'}
                    </Badge>
                  </div>
                </div>

                <div class="space-y-2">
                  <div class="flex justify-between">
                    <span class="text-muted-foreground">Last Update:</span>
                    <span>{new Date(systemHealth.lastUpdateTime).toLocaleString()}</span>
                  </div>
                  <div class="flex justify-between">
                    <span class="text-muted-foreground">Performance:</span>
                    <Badge class={getHealthColor(systemHealth.systemHealth).includes('green') ? 'bits-badge-default' : 'bits-badge-secondary'}>
                      Optimal
                    </Badge>
                  </div>
                </div>
              </div>
            </div>
          {:else}
            <div class="text-center py-8">
              <p class="text-muted-foreground">Loading system health information...</p>
            </div>
          {/if}
        </CardContent>
      </Card>
    </div>
  {/if}
</div>

<style>
  /* @unocss-include */
</style>

