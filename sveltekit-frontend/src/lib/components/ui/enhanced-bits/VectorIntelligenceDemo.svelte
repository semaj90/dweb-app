<script lang="ts">
  interface Props {
    class?: string;
    children?: import('svelte').Snippet;
  }
  import { Button, Select, Input, Card } from './index.js';
  import type { SelectOption } from './index.js';
  import type { VectorSearchResult, SemanticEntity } from '$lib/types/ai';
  import { cn } from '$lib/utils/cn';
  import { Search, Brain, FileText, Users, MapPin, Calendar, Scale, Zap } from 'lucide-svelte';

  // Vector Intelligence Demo State (Svelte 5 runes)
  let searchQuery = $state('');
  let searchResults = $state<VectorSearchResult[]>([]);
  let semanticEntities = $state<SemanticEntity[]>([]);
  let isSearching = $state(false);
  let selectedSearchType = $state('semantic');
  let selectedConfidence = $state('all');
  let analysisDepth = $state('standard');

  // Mock vector search configuration
  const searchTypes: SelectOption[] = [
    {
      value: 'semantic',
      label: 'Semantic Search',
      description: 'AI-powered contextual understanding',
      category: 'AI-Powered'
    },
    {
      value: 'vector',
      label: 'Vector Similarity',
      description: 'Embedding-based similarity matching',
      category: 'AI-Powered'
    },
    {
      value: 'hybrid',
      label: 'Hybrid Search',
      description: 'Combined semantic and keyword search',
      category: 'AI-Powered'
    },
    {
      value: 'legal',
      label: 'Legal Precedent',
      description: 'Case law and precedent matching',
      category: 'Legal-Specific'
    },
    {
      value: 'citation',
      label: 'Citation Analysis',
      description: 'Legal citation and reference tracking',
      category: 'Legal-Specific'
    }
  ];

  const confidenceFilters: SelectOption[] = [
    { value: 'all', label: 'All Results', description: 'Show all confidence levels' },
    { value: 'high', label: 'High Confidence', description: '90%+ confidence scores' },
    { value: 'medium', label: 'Medium Confidence', description: '70-89% confidence scores' },
    { value: 'low', label: 'Low Confidence', description: 'Below 70% confidence' }
  ];

  const analysisOptions: SelectOption[] = [
    { value: 'quick', label: 'Quick Analysis', description: 'Basic semantic parsing' },
    { value: 'standard', label: 'Standard Analysis', description: 'Full entity recognition' },
    { value: 'deep', label: 'Deep Analysis', description: 'Advanced relationship mapping' }
  ];

  // Mock search results data
  const mockSearchResults: VectorSearchResult[] = [
    {
      id: '1',
      content: 'Contract breach regarding non-disclosure agreement violation with evidence of corporate espionage through unauthorized access to proprietary systems.',
      metadata: {
        caseNumber: 'CV-2024-001',
        court: 'Superior Court of California',
        judge: 'Hon. Sarah Mitchell',
        date: '2024-01-15'
      },
      score: 0.94,
      highlights: ['non-disclosure agreement', 'corporate espionage', 'unauthorized access'],
      source: {
        type: 'case',
        name: 'TechCorp vs. StartupInc',
        url: '/cases/cv-2024-001'
      }
    },
    {
      id: '2',
      content: 'Employment termination dispute involving alleged discrimination based on protected class status under Title VII enforcement guidelines.',
      metadata: {
        caseNumber: 'EM-2024-042',
        jurisdiction: 'Federal District Court',
        statute: 'Title VII Civil Rights Act',
        precedent: 'McDonnell Douglas test'
      },
      score: 0.87,
      highlights: ['employment termination', 'discrimination', 'protected class'],
      source: {
        type: 'precedent',
        name: 'EEOC Guidelines on Discrimination',
        url: '/precedents/title-vii-enforcement'
      }
    },
    {
      id: '3',
      content: 'Intellectual property infringement case analyzing patent claims and prior art references in software development litigation.',
      metadata: {
        patentNumber: 'US 10,123,456',
        filingDate: '2020-03-15',
        inventor: 'Dr. Jane Smith',
        classification: 'G06F 16/00'
      },
      score: 0.82,
      highlights: ['intellectual property', 'patent claims', 'prior art'],
      source: {
        type: 'document',
        name: 'Patent Application Analysis',
        url: '/documents/patent-analysis-2024'
      }
    }
  ];

  const mockEntities: SemanticEntity[] = [
    { text: 'TechCorp', type: 'organization', confidence: 0.95, start: 0, end: 8 },
    { text: 'StartupInc', type: 'organization', confidence: 0.92, start: 12, end: 22 },
    { text: 'Hon. Sarah Mitchell', type: 'person', confidence: 0.98, start: 45, end: 64 },
    { text: 'Superior Court of California', type: 'organization', confidence: 0.89, start: 70, end: 98 },
    { text: '2024-01-15', type: 'date', confidence: 0.99, start: 103, end: 113 },
    { text: 'Title VII Civil Rights Act', type: 'legal_term', confidence: 0.96, start: 120, end: 146 },
    { text: 'McDonnell Douglas test', type: 'legal_term', confidence: 0.94, start: 150, end: 172 }
  ];

  // Reactive filtering of results based on confidence
  let filteredResults = $derived(
    selectedConfidence === 'all' ? searchResults : (() => {
      const thresholds = {
        high: 0.9,
        medium: 0.7,
        low: 0.0
      };

      const minScore = thresholds[selectedConfidence as keyof typeof thresholds];
      const maxScore = selectedConfidence === 'low' ? 0.7 : 1.0;

      return searchResults.filter(result =>
        result.score >= minScore && result.score < maxScore
      );
    })()
  );

  // Entity type icons mapping
  const entityIcons = {
    person: Users,
    organization: Scale,
    location: MapPin,
    date: Calendar,
    legal_term: FileText,
    case_citation: Scale
  };

  // Entity type colors
  const entityColors = {
    person: 'semantic-entity-person',
    organization: 'semantic-entity-organization',
    location: 'semantic-entity-location',
    date: 'semantic-entity-date',
    legal_term: 'semantic-entity-legal',
    case_citation: 'semantic-entity-legal'
  };

  // Search functionality
  async function performVectorSearch() {
    if (!searchQuery.trim()) return;

    isSearching = true;

    try {
      // Simulate API call delay
      await new Promise(resolve => setTimeout(resolve, 1500));

      // Mock results based on search type
      searchResults = mockSearchResults.map(result => ({
        ...result,
        score: Math.random() * 0.3 + 0.7, // Random score between 0.7-1.0
        highlights: result.highlights.filter(() => Math.random() > 0.3) // Random highlights
      }));

      // Mock entity extraction
      if (analysisDepth !== 'quick') {
        semanticEntities = mockEntities;
      }

    } catch (error) {
      console.error('Search failed:', error);
    } finally {
      isSearching = false;
    }
  }

  // Clear results
  function clearResults() {
    searchResults = [];
    semanticEntities = [];
    searchQuery = '';
  }

  // Get confidence badge class
  function getConfidenceBadgeClass(score: number): string {
    if (score >= 0.9) return 'vector-confidence-high';
    if (score >= 0.7) return 'vector-confidence-medium';
    return 'vector-confidence-low';
  }

  // Format confidence percentage
  function formatConfidence(score: number): string {
    return `${Math.round(score * 100)}%`;
  }
</script>

<div class="vector-intelligence-demo yorha-panel p-6 max-w-6xl mx-auto">
  <!-- Header -->
  <div class="yorha-panel-header mb-6">
    <div class="flex items-center gap-3 mb-2">
      <Brain class="w-8 h-8 text-blue-600" />
      <h1 class="text-2xl font-gothic tracking-wide text-nier-text-primary">
        Vector Intelligence Demo
      </h1>
    </div>
    <p class="text-nier-text-secondary">
      Advanced AI-powered search and semantic analysis for legal documents using vector embeddings and NLP.
    </p>
  </div>

  <!-- Search Configuration -->
  <div class="demo-config-section mb-6">
    <h2 class="text-lg font-gothic mb-4 text-nier-text-primary">Search Configuration</h2>

    <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
      <Select
        bind:value={selectedSearchType}
        options={searchTypes}
        placeholder="Select search type..."
        legal
        aiRecommendations
        label="Search Algorithm"
      />

      <Select
        bind:value={selectedConfidence}
        options={confidenceFilters}
        placeholder="Filter by confidence..."
        legal
        label="Confidence Filter"
      />

      <Select
        bind:value={analysisDepth}
        options={analysisOptions}
        placeholder="Analysis depth..."
        legal
        label="Analysis Depth"
      />
    </div>
  </div>

  <!-- Search Interface -->
  <div class="demo-config-section mb-6">
    <h2 class="text-lg font-gothic mb-4 text-nier-text-primary">Vector Search Interface</h2>

    <div class="flex gap-3 mb-4">
      <Input
        variant="search"
        placeholder="Enter legal query or document content..."
        bind:value={searchQuery}
        evidenceSearch
        legal
        fullWidth
        icon={Search}
        disabled={isSearching}
        class="flex-1"
      />

      <Button
        variant="primary"
        legal
        loading={isSearching}
        on:on:click={performVectorSearch}
        disabled={!searchQuery.trim()}
      >
        {#if isSearching}
          <Zap class="w-4 h-4 mr-2 animate-pulse" />
          Analyzing...
        {:else}
          <Search class="w-4 h-4 mr-2" />
          Search
        {/if}
      </Button>

      {#if searchResults.length > 0}
        <Button
          variant="outline"
          on:on:click={clearResults}
        >
          Clear
        </Button>
      {/if}
    </div>

    <!-- Search Status -->
    {#if isSearching}
      <div class="processing-overlay bg-nier-overlay rounded-lg p-4 mb-4">
        <div class="flex items-center justify-center gap-3">
          <div class="ai-status-indicator ai-status-processing w-6 h-6"></div>
          <span class="text-nier-text-primary font-medium">
            Processing vector embeddings and semantic analysis...
          </span>
        </div>
      </div>
    {/if}
  </div>

  <!-- Semantic Entities -->
  {#if semanticEntities.length > 0}
    <div class="demo-results-section mb-6">
      <h2 class="text-lg font-gothic mb-4 text-nier-text-primary">Extracted Entities</h2>

      <Card variant="yorha" legal class="p-4">
        <div class="semantic-entity-container">
          {#each semanticEntities as entity (entity.text)}
            {@const SvelteComponent = entityIcons[entity.type]}
            <div class={cn('semantic-entity-tag', entityColors[entity.type])}>
              <div class="flex items-center gap-1">
                <SvelteComponent
                  class="w-3 h-3"
                />
                <span class="text-xs font-medium">{entity.text}</span>
                <span class="text-xs opacity-75">
                  {formatConfidence(entity.confidence)}
                </span>
              </div>
            </div>
          {/each}
        </div>

        <div class="mt-3 text-xs text-nier-text-muted">
          <strong>{semanticEntities.length}</strong> entities extracted with
          <strong>{analysisDepth}</strong> analysis depth
        </div>
      </Card>
    </div>
  {/if}

  <!-- Search Results -->
  {#if filteredResults.length > 0}
    <div class="demo-results-section">
      <div class="flex items-center justify-between mb-4">
        <h2 class="text-lg font-gothic text-nier-text-primary">
          Search Results ({filteredResults.length})
        </h2>
        <div class="text-sm text-nier-text-muted">
          Showing {selectedConfidence} confidence results
        </div>
      </div>

      <div class="space-y-4">
        {#each filteredResults as result (result.id)}
          <Card
            variant="default"
            evidenceCard
            hoverable
            clickable
            class="vector-result-item"
          >
            {@const SvelteComponent_1 = result.source.type === 'case' ? Scale :
                          result.source.type === 'precedent' ? FileText :
                          FileText}
            <div class="space-y-3">
              <!-- Result Header -->
              <div class="flex items-start justify-between">
                <div class="flex items-center gap-2">
                  <SvelteComponent_1
                    class="w-4 h-4 text-nier-text-muted"
                  />
                  <span class="text-sm font-medium text-nier-text-primary">
                    {result.source.name}
                  </span>
                </div>

                <div class={cn('vector-confidence-badge', getConfidenceBadgeClass(result.score))}>
                  {formatConfidence(result.score)}
                </div>
              </div>

              <!-- Result Content -->
              <div class="text-sm text-nier-text-secondary leading-relaxed">
                {@html result.content.replace(
                  new RegExp(`(${result.highlights.join('|')})`, 'gi'),
                  '<span class="vector-highlight">$1</span>'
                )}
              </div>

              <!-- Result Metadata -->
              <div class="vector-metadata-grid">
                {#each Object.entries(result.metadata) as [key, value]}
                  <div class="flex flex-col">
                    <span class="text-xs font-medium text-nier-text-muted uppercase tracking-wide">
                      {key.replace(/([A-Z])/g, ' $1').toLowerCase()}
                    </span>
                    <span class="text-xs text-nier-text-secondary">
                      {value}
                    </span>
                  </div>
                {/each}
              </div>

              <!-- Action Buttons -->
              <div class="flex items-center justify-between pt-2 border-t border-nier-border-muted">
                <div class="flex gap-2">
                  {#each result.highlights as highlight}
                    <span class="text-xs px-2 py-1 bg-nier-bg-tertiary rounded text-nier-text-secondary">
                      {highlight}
                    </span>
                  {/each}
                </div>

                <div class="flex gap-2">
                  <Button size="sm" variant="outline">
                    View Full
                  </Button>
                  <Button size="sm" variant="primary">
                    Add to Case
                  </Button>
                </div>
              </div>
            </div>
          </Card>
        {/each}
      </div>
    </div>
  {:else if searchResults.length === 0 && !isSearching}
    <!-- Empty State -->
    <div class="text-center py-12">
      <Brain class="w-16 h-16 mx-auto text-nier-text-muted mb-4" />
      <h3 class="text-lg font-medium text-nier-text-primary mb-2">
        AI Vector Intelligence Ready
      </h3>
      <p class="text-nier-text-secondary mb-4">
        Enter a search query to demonstrate advanced semantic analysis and vector similarity matching.
      </p>
      <div class="flex justify-center gap-2">
        <Button
          variant="outline"
          on:on:click={() => searchQuery = 'contract breach non-disclosure agreement'}
        >
          Try Sample Query
        </Button>
      </div>
    </div>
  {/if}

  <!-- Performance Metrics -->
  {#if searchResults.length > 0}
    <div class="mt-6 pt-4 border-t border-nier-border-secondary">
      <div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
        <div class="agent-card p-3">
          <div class="text-lg font-bold text-nier-text-primary">
            {filteredResults.length}
          </div>
          <div class="text-xs text-nier-text-muted">Results Found</div>
        </div>

        <div class="agent-card p-3">
          <div class="text-lg font-bold text-nier-text-primary">
            {Math.round(Math.max(...searchResults.map(r => r.score)) * 100)}%
          </div>
          <div class="text-xs text-nier-text-muted">Max Confidence</div>
        </div>

        <div class="agent-card p-3">
          <div class="text-lg font-bold text-nier-text-primary">
            {semanticEntities.length}
          </div>
          <div class="text-xs text-nier-text-muted">Entities</div>
        </div>

        <div class="agent-card p-3">
          <div class="text-lg font-bold text-nier-text-primary">
            1.2s
          </div>
          <div class="text-xs text-nier-text-muted">Query Time</div>
        </div>
      </div>
    </div>
  {/if}
</div>

<style>
  /* @unocss-include */

  /* Vector Intelligence specific styling */
  .vector-intelligence-demo {
    background: linear-gradient(
      135deg,
      var(--color-nier-bg-primary) 0%,
      var(--color-nier-bg-secondary) 100%
    );
  }

  /* Enhanced result highlighting */
  :global(.vector-highlight) {
    background: linear-gradient(
      120deg,
      rgba(59, 130, 246, 0.2) 0%,
      rgba(16, 185, 129, 0.2) 100%
    );
    padding: 2px 4px;
    border-radius: 3px;
    font-weight: 600;
    color: var(--color-nier-text-primary);
  }

  /* Semantic entity styling enhancements */
  :global(.semantic-entity-container) {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-bottom: 1rem;
  }

  :global(.semantic-entity-tag) {
    transition: all 0.2s ease;
    cursor: pointer;
  }

  :global(.semantic-entity-tag:hover) {
    transform: translateY(-1px);
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  }

  /* Enhanced confidence badges */
  :global(.vector-confidence-badge) {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    letter-spacing: 0.5px;
    border: 1px solid currentColor;
  }

  /* Processing overlay enhancement */
  :global(.processing-overlay) {
    backdrop-filter: blur(4px);
    animation: processing-pulse 2s ease-in-out infinite;
  }

  @keyframes processing-pulse {
    0%, 100% {
      opacity: 0.8;
    }
    50% {
      opacity: 1;
    }
  }

  /* Vector result item enhancements */
  :global(.vector-result-item) {
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
  }

  :global(.vector-result-item::before) {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 2px;
    background: linear-gradient(
      90deg,
      transparent,
      var(--color-ai-status-online),
      transparent
    );
    transition: left 0.5s ease;
  }

  :global(.vector-result-item:hover::before) {
    left: 100%;
  }

  /* Metadata grid styling */
  :global(.vector-metadata-grid) {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: 0.75rem;
    padding: 0.75rem;
    background: rgba(58, 55, 47, 0.03);
    border-radius: 6px;
    border: 1px solid var(--color-nier-border-muted);
  }

  /* Empty state styling */
  .text-center {
    background: radial-gradient(
      ellipse at center,
      rgba(59, 130, 246, 0.05) 0%,
      transparent 70%
    );
    border-radius: 12px;
    padding: 3rem 2rem;
  }

  /* Performance metrics styling */
  :global(.agent-card) {
    background: linear-gradient(
      135deg,
      rgba(255, 255, 255, 0.8) 0%,
      rgba(255, 255, 255, 0.6) 100%
    );
    backdrop-filter: blur(8px);
    border: 1px solid rgba(58, 55, 47, 0.1);
  }
</style>