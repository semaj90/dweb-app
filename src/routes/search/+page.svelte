<script>
  import { onMount } from 'svelte';
  import { goto } from '$app/navigation';

  // State management
  let isLoading = $state(true);
  let searchQuery = $state('');
  let searchResults = $state([]);
  let searchType = $state('semantic');
  let isSearching = $state(false);
  let filters = $state({
    documentType: 'all',
    dateRange: 'all',
    relevanceScore: 0.7
  });

  // Mock search results
  let mockResults = [
    {
      id: 'result-001',
      title: 'Contract Liability Analysis - Smith vs. Jones',
      snippet: 'This case establishes precedent for liability limitations in commercial contracts...',
      type: 'case-law',
      relevance: 0.94,
      source: 'Federal Case Database',
      date: '2024-03-15',
      citations: 15,
      tags: ['contract', 'liability', 'commercial']
    },
    {
      id: 'result-002', 
      title: 'Corporate Compliance Guidelines 2024',
      snippet: 'Updated guidelines for corporate compliance including new regulatory requirements...',
      type: 'regulation',
      relevance: 0.89,
      source: 'Regulatory Database',
      date: '2024-01-10',
      citations: 8,
      tags: ['compliance', 'corporate', 'regulation']
    },
    {
      id: 'result-003',
      title: 'Employment Contract Template - Standard Terms',
      snippet: 'Comprehensive employment contract template with standard liability clauses...',
      type: 'document',
      relevance: 0.85,
      source: 'Legal Templates',
      date: '2023-12-20',
      citations: 12,
      tags: ['employment', 'contract', 'template']
    }
  ];

  // Functions
  function navigateHome() {
    goto('/');
  }

  async function performSearch() {
    if (!searchQuery.trim()) return;
    
    isSearching = true;
    searchResults = [];

    // Simulate search delay
    setTimeout(() => {
      searchResults = mockResults.filter(result => 
        result.relevance >= filters.relevanceScore
      );
      isSearching = false;
    }, 1000 + Math.random() * 1500);
  }

  function getTypeIcon(type) {
    switch (type) {
      case 'case-law': return '⚖️';
      case 'regulation': return '📋';
      case 'document': return '📄';
      case 'statute': return '📜';
      case 'precedent': return '🏛️';
      default: return '📁';
    }
  }

  function getTypeColor(type) {
    switch (type) {
      case 'case-law': return '#ff6b6b';
      case 'regulation': return '#4ecdc4';
      case 'document': return '#ffbf00';
      case 'statute': return '#a78bfa';
      case 'precedent': return '#00ff41';
      default: return '#888';
    }
  }

  function formatRelevance(score) {
    return Math.round(score * 100);
  }

  function clearSearch() {
    searchQuery = '';
    searchResults = [];
  }

  // Initialize component
  onMount(() => {
    setTimeout(() => {
      isLoading = false;
    }, 800);
  });
</script>

<svelte:head>
  <title>Legal Search - YoRHa Legal AI</title>
  <meta name="description" content="Semantic search across legal documents and case law">
</svelte:head>

<!-- Loading Screen -->
{#if isLoading}
  <div class="loading-screen">
    <div class="loading-content">
      <div class="loading-icon">🔎</div>
      <div class="loading-text">INITIALIZING LEGAL SEARCH ENGINE...</div>
      <div class="loading-bar">
        <div class="loading-progress"></div>
      </div>
    </div>
  </div>
{:else}
  <!-- Main Interface -->
  <div class="search-interface">
    
    <!-- Header -->
    <header class="search-header">
      <div class="header-left">
        <button class="back-button" onclick={navigateHome}>
          ← COMMAND CENTER
        </button>
        <div class="header-title">
          <h1>🔎 LEGAL SEARCH ENGINE</h1>
          <div class="header-subtitle">Semantic Search Across Legal Documents and Case Law</div>
        </div>
      </div>
      
      <div class="header-stats">
        <div class="stat-item">
          <div class="stat-value">2.3M</div>
          <div class="stat-label">DOCUMENTS</div>
        </div>
        <div class="stat-item">
          <div class="stat-value">750K</div>
          <div class="stat-label">CASES</div>
        </div>
        <div class="stat-item">
          <div class="stat-value">1.8M</div>
          <div class="stat-label">CITATIONS</div>
        </div>
      </div>
    </header>

    <!-- Search Section -->
    <section class="search-section">
      <div class="search-container">
        <div class="search-input-wrapper">
          <input
            bind:value={searchQuery}
            placeholder="SEARCH LEGAL DOCUMENTS, CASES, STATUTES..."
            class="search-input"
            disabled={isSearching}
          />
          <button 
            class="search-button"
            onclick={performSearch}
            disabled={!searchQuery.trim() || isSearching}
          >
            {isSearching ? '⚙️' : '🔍'}
          </button>
        </div>

        <div class="search-controls">
          <div class="search-type-selector">
            <label class="control-label">SEARCH TYPE:</label>
            <select bind:value={searchType} class="control-select">
              <option value="semantic">SEMANTIC SEARCH</option>
              <option value="keyword">KEYWORD SEARCH</option>
              <option value="citation">CITATION SEARCH</option>
              <option value="concept">CONCEPT SEARCH</option>
            </select>
          </div>

          <div class="relevance-filter">
            <label class="control-label">MIN RELEVANCE: {formatRelevance(filters.relevanceScore)}%</label>
            <input 
              type="range" 
              min="0.1" 
              max="1" 
              step="0.05" 
              bind:value={filters.relevanceScore}
              class="relevance-slider"
            />
          </div>

          <button class="clear-button" onclick={clearSearch}>CLEAR</button>
        </div>
      </div>
    </section>

    <!-- Results Section -->
    <main class="results-section">
      
      <!-- Search Status -->
      <div class="search-status">
        {#if isSearching}
          <div class="searching-indicator">
            <div class="searching-spinner">🔍</div>
            <div class="searching-text">SEARCHING LEGAL DATABASE...</div>
          </div>
        {:else if searchResults.length > 0}
          <div class="results-info">
            <div class="results-count">
              FOUND {searchResults.length} RESULTS FOR "{searchQuery}"
            </div>
            <div class="search-time">Search completed in 1.2 seconds</div>
          </div>
        {:else if searchQuery}
          <div class="no-results">
            <div class="no-results-icon">📭</div>
            <div class="no-results-text">NO RESULTS FOUND</div>
            <div class="no-results-suggestion">Try adjusting your search terms or filters</div>
          </div>
        {/if}
      </div>

      <!-- Results List -->
      {#if searchResults.length > 0}
        <div class="results-list">
          {#each searchResults as result (result.id)}
            <div class="result-card">
              <div class="result-header">
                <div class="result-type" style="color: {getTypeColor(result.type)}">
                  {getTypeIcon(result.type)} {result.type.toUpperCase().replace('-', ' ')}
                </div>
                <div class="result-relevance">
                  <div class="relevance-score">{formatRelevance(result.relevance)}% MATCH</div>
                  <div class="relevance-bar">
                    <div 
                      class="relevance-fill" 
                      style="width: {result.relevance * 100}%; background: {getTypeColor(result.type)}"
                    ></div>
                  </div>
                </div>
              </div>

              <div class="result-content">
                <h3 class="result-title">{result.title}</h3>
                <p class="result-snippet">{result.snippet}</p>
                
                <div class="result-metadata">
                  <div class="metadata-item">
                    <span class="metadata-label">SOURCE:</span>
                    <span class="metadata-value">{result.source}</span>
                  </div>
                  <div class="metadata-item">
                    <span class="metadata-label">DATE:</span>
                    <span class="metadata-value">{new Date(result.date).toLocaleDateString()}</span>
                  </div>
                  <div class="metadata-item">
                    <span class="metadata-label">CITATIONS:</span>
                    <span class="metadata-value">{result.citations}</span>
                  </div>
                </div>

                <div class="result-tags">
                  {#each result.tags as tag}
                    <span class="result-tag">{tag}</span>
                  {/each}
                </div>
              </div>

              <div class="result-actions">
                <button class="action-button primary">VIEW DOCUMENT</button>
                <button class="action-button secondary">CITE</button>
                <button class="action-button secondary">SAVE</button>
                <button class="action-button secondary">SHARE</button>
              </div>
            </div>
          {/each}
        </div>
      {/if}

      <!-- Search Tips -->
      {#if !searchQuery && !isSearching}
        <div class="search-tips">
          <h3 class="tips-title">🎯 SEARCH TIPS</h3>
          <div class="tips-grid">
            <div class="tip-card">
              <div class="tip-icon">🔍</div>
              <div class="tip-title">Semantic Search</div>
              <div class="tip-description">Search by concept rather than exact keywords</div>
              <div class="tip-example">Example: "contract breach remedies"</div>
            </div>
            <div class="tip-card">
              <div class="tip-icon">📎</div>
              <div class="tip-title">Citation Search</div>
              <div class="tip-description">Find documents that cite specific cases</div>
              <div class="tip-example">Example: "Smith v. Jones 2023"</div>
            </div>
            <div class="tip-card">
              <div class="tip-icon">🎨</div>
              <div class="tip-title">Concept Search</div>
              <div class="tip-description">Search by legal concepts and principles</div>
              <div class="tip-example">Example: "good faith dealings"</div>
            </div>
            <div class="tip-card">
              <div class="tip-icon">🔤</div>
              <div class="tip-title">Keyword Search</div>
              <div class="tip-description">Traditional keyword-based search</div>
              <div class="tip-example">Example: "liability AND damages"</div>
            </div>
          </div>
        </div>
      {/if}
    </main>

    <!-- Footer -->
    <footer class="search-footer">
      <div class="footer-info">
        <div class="database-stats">
          <div class="stat-group">
            <span class="stat-label">DATABASE STATUS:</span>
            <span class="stat-value active">ONLINE</span>
          </div>
          <div class="stat-group">
            <span class="stat-label">LAST UPDATE:</span>
            <span class="stat-value">{new Date().toLocaleDateString()}</span>
          </div>
          <div class="stat-group">
            <span class="stat-label">SEARCH ENGINE:</span>
            <span class="stat-value">VECTOR-POWERED</span>
          </div>
        </div>
      </div>
    </footer>
  </div>
{/if}

<style>
  /* === GLOBAL VARIABLES === */
  :global(:root) {
    --yorha-primary: #c4b49a;
    --yorha-secondary: #b5a48a;
    --yorha-accent-warm: #ffbf00;
    --yorha-accent-cool: #4ecdc4;
    --yorha-success: #00ff41;
    --yorha-warning: #ffbf00;
    --yorha-error: #ff6b6b;
    --yorha-light: #ffffff;
    --yorha-muted: #f0f0f0;
    --yorha-dark: #1a1a1a;
    --yorha-darker: #0a0a0a;
    --yorha-bg: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
  }

  /* === LOADING SCREEN === */
  .loading-screen {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100vh;
    background: var(--yorha-bg);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 9999;
    font-family: 'JetBrains Mono', monospace;
    color: var(--yorha-light);
  }

  .loading-content {
    text-align: center;
    animation: fadeInUp 0.8s ease-out;
  }

  .loading-icon {
    font-size: 4rem;
    margin-bottom: 2rem;
    color: var(--yorha-accent-warm);
    animation: pulse 2s ease-in-out infinite;
  }

  .loading-text {
    font-size: 1.2rem;
    color: var(--yorha-muted);
    letter-spacing: 2px;
    margin-bottom: 2rem;
  }

  .loading-bar {
    width: 300px;
    height: 4px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 2px;
    overflow: hidden;
    margin: 0 auto;
  }

  .loading-progress {
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, var(--yorha-accent-warm), var(--yorha-success));
    animation: loading 2s ease-in-out infinite;
  }

  /* === MAIN INTERFACE === */
  .search-interface {
    min-height: 100vh;
    background: var(--yorha-bg);
    color: var(--yorha-light);
    font-family: 'JetBrains Mono', monospace;
    animation: fadeIn 0.5s ease-out;
  }

  /* === HEADER === */
  .search-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 2rem;
    border-bottom: 2px solid var(--yorha-accent-warm);
    background: rgba(26, 26, 26, 0.9);
    backdrop-filter: blur(10px);
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 2rem;
  }

  .back-button {
    background: transparent;
    border: 2px solid var(--yorha-accent-cool);
    color: var(--yorha-accent-cool);
    padding: 0.8rem 1.5rem;
    font-family: inherit;
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    cursor: pointer;
    transition: all 0.3s ease;
  }

  .back-button:hover {
    background: var(--yorha-accent-cool);
    color: var(--yorha-dark);
    transform: translateX(-5px);
  }

  .header-title h1 {
    margin: 0;
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(45deg, var(--yorha-accent-warm), var(--yorha-success));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-transform: uppercase;
    letter-spacing: 2px;
  }

  .header-subtitle {
    font-size: 0.9rem;
    color: var(--yorha-muted);
    margin-top: 0.5rem;
    letter-spacing: 1px;
  }

  .header-stats {
    display: flex;
    gap: 2rem;
  }

  .stat-item {
    text-align: center;
  }

  .stat-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--yorha-accent-warm);
    line-height: 1;
  }

  .stat-label {
    font-size: 0.7rem;
    color: var(--yorha-muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 0.3rem;
  }

  /* === SEARCH SECTION === */
  .search-section {
    padding: 2rem;
    background: rgba(26, 26, 26, 0.5);
    border-bottom: 1px solid rgba(255, 191, 0, 0.3);
  }

  .search-container {
    max-width: 1200px;
    margin: 0 auto;
  }

  .search-input-wrapper {
    display: flex;
    margin-bottom: 1.5rem;
  }

  .search-input {
    flex: 1;
    background: rgba(26, 26, 26, 0.8);
    border: 2px solid rgba(255, 191, 0, 0.5);
    color: var(--yorha-light);
    padding: 1.2rem 2rem;
    font-family: inherit;
    font-size: 1.1rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    transition: all 0.3s ease;
    border-radius: 8px 0 0 8px;
  }

  .search-input:focus {
    outline: none;
    border-color: var(--yorha-accent-warm);
    box-shadow: 0 0 20px rgba(255, 191, 0, 0.3);
  }

  .search-input::placeholder {
    color: var(--yorha-muted);
    opacity: 0.7;
  }

  .search-input:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .search-button {
    background: var(--yorha-accent-warm);
    border: 2px solid var(--yorha-accent-warm);
    color: var(--yorha-dark);
    padding: 1.2rem 2rem;
    font-size: 1.5rem;
    cursor: pointer;
    transition: all 0.3s ease;
    border-radius: 0 8px 8px 0;
  }

  .search-button:hover:not(:disabled) {
    background: var(--yorha-success);
    border-color: var(--yorha-success);
    transform: scale(1.05);
  }

  .search-button:disabled {
    background: rgba(255, 255, 255, 0.1);
    border-color: rgba(255, 255, 255, 0.1);
    color: var(--yorha-muted);
    cursor: not-allowed;
    transform: none;
  }

  .search-controls {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 2rem;
    flex-wrap: wrap;
  }

  .search-type-selector,
  .relevance-filter {
    display: flex;
    align-items: center;
    gap: 1rem;
  }

  .control-label {
    font-size: 0.9rem;
    color: var(--yorha-muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    white-space: nowrap;
  }

  .control-select {
    background: rgba(26, 26, 26, 0.8);
    border: 2px solid rgba(78, 205, 196, 0.5);
    color: var(--yorha-light);
    padding: 0.8rem 1rem;
    font-family: inherit;
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    cursor: pointer;
    border-radius: 4px;
  }

  .control-select:focus {
    outline: none;
    border-color: var(--yorha-accent-cool);
  }

  .relevance-slider {
    width: 150px;
    height: 6px;
    background: rgba(255, 255, 255, 0.2);
    border-radius: 3px;
    outline: none;
    cursor: pointer;
  }

  .relevance-slider::-webkit-slider-thumb {
    appearance: none;
    width: 18px;
    height: 18px;
    background: var(--yorha-accent-warm);
    border-radius: 50%;
    cursor: pointer;
  }

  .clear-button {
    background: transparent;
    border: 2px solid var(--yorha-error);
    color: var(--yorha-error);
    padding: 0.8rem 1.5rem;
    font-family: inherit;
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    cursor: pointer;
    transition: all 0.3s ease;
    border-radius: 4px;
  }

  .clear-button:hover {
    background: var(--yorha-error);
    color: var(--yorha-light);
  }

  /* === RESULTS SECTION === */
  .results-section {
    padding: 2rem;
    max-width: 1200px;
    margin: 0 auto;
  }

  .search-status {
    margin-bottom: 2rem;
  }

  .searching-indicator {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 1rem;
    padding: 2rem;
  }

  .searching-spinner {
    font-size: 2rem;
    animation: spin 2s linear infinite;
  }

  .searching-text {
    font-size: 1.1rem;
    color: var(--yorha-accent-warm);
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  .results-info {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem 0;
    border-bottom: 1px solid rgba(255, 191, 0, 0.3);
  }

  .results-count {
    font-size: 1.1rem;
    color: var(--yorha-light);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  .search-time {
    font-size: 0.9rem;
    color: var(--yorha-muted);
  }

  .no-results {
    text-align: center;
    padding: 4rem 2rem;
  }

  .no-results-icon {
    font-size: 4rem;
    margin-bottom: 1rem;
    opacity: 0.5;
  }

  .no-results-text {
    font-size: 1.3rem;
    color: var(--yorha-muted);
    margin-bottom: 0.5rem;
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  .no-results-suggestion {
    color: var(--yorha-muted);
    opacity: 0.8;
  }

  /* === RESULTS LIST === */
  .results-list {
    display: flex;
    flex-direction: column;
    gap: 2rem;
  }

  .result-card {
    background: rgba(26, 26, 26, 0.8);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 8px;
    overflow: hidden;
    transition: all 0.3s ease;
  }

  .result-card:hover {
    border-color: var(--yorha-accent-warm);
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(255, 191, 0, 0.2);
  }

  .result-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem 1.5rem;
    background: rgba(42, 42, 42, 0.6);
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  }

  .result-type {
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  .result-relevance {
    display: flex;
    align-items: center;
    gap: 1rem;
  }

  .relevance-score {
    font-size: 0.8rem;
    color: var(--yorha-success);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  .relevance-bar {
    width: 100px;
    height: 6px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 3px;
    overflow: hidden;
  }

  .relevance-fill {
    height: 100%;
    transition: width 0.5s ease;
    border-radius: 3px;
  }

  .result-content {
    padding: 1.5rem;
  }

  .result-title {
    font-size: 1.3rem;
    color: var(--yorha-light);
    margin: 0 0 1rem 0;
    font-weight: 600;
    line-height: 1.3;
  }

  .result-snippet {
    color: var(--yorha-muted);
    line-height: 1.6;
    margin: 0 0 1.5rem 0;
    font-size: 0.95rem;
  }

  .result-metadata {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin-bottom: 1.5rem;
  }

  .metadata-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.8rem;
  }

  .metadata-label {
    color: var(--yorha-muted);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  .metadata-value {
    color: var(--yorha-light);
  }

  .result-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
  }

  .result-tag {
    background: rgba(78, 205, 196, 0.2);
    color: var(--yorha-accent-cool);
    padding: 0.3rem 0.6rem;
    border-radius: 3px;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    border: 1px solid var(--yorha-accent-cool);
  }

  .result-actions {
    display: flex;
    gap: 1rem;
    padding: 1rem 1.5rem;
    background: rgba(42, 42, 42, 0.4);
    border-top: 1px solid rgba(255, 255, 255, 0.1);
  }

  .action-button {
    padding: 0.6rem 1.2rem;
    font-family: inherit;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    cursor: pointer;
    transition: all 0.3s ease;
    border-radius: 4px;
    font-weight: 600;
  }

  .action-button.primary {
    background: var(--yorha-success);
    color: var(--yorha-dark);
    border: none;
  }

  .action-button.primary:hover {
    background: var(--yorha-accent-warm);
    transform: scale(1.05);
  }

  .action-button.secondary {
    background: transparent;
    color: var(--yorha-accent-cool);
    border: 2px solid var(--yorha-accent-cool);
  }

  .action-button.secondary:hover {
    background: var(--yorha-accent-cool);
    color: var(--yorha-dark);
  }

  /* === SEARCH TIPS === */
  .search-tips {
    padding: 2rem 0;
  }

  .tips-title {
    font-size: 1.5rem;
    color: var(--yorha-accent-warm);
    text-align: center;
    margin: 0 0 2rem 0;
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  .tips-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 1.5rem;
  }

  .tip-card {
    background: rgba(26, 26, 26, 0.8);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 8px;
    padding: 1.5rem;
    text-align: center;
    transition: all 0.3s ease;
  }

  .tip-card:hover {
    border-color: var(--yorha-accent-cool);
    transform: translateY(-3px);
  }

  .tip-icon {
    font-size: 2.5rem;
    margin-bottom: 1rem;
    color: var(--yorha-accent-cool);
  }

  .tip-title {
    font-size: 1.1rem;
    color: var(--yorha-light);
    margin-bottom: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  .tip-description {
    color: var(--yorha-muted);
    line-height: 1.5;
    margin-bottom: 1rem;
  }

  .tip-example {
    background: rgba(42, 42, 42, 0.6);
    padding: 0.8rem;
    border-radius: 4px;
    font-family: 'Courier New', monospace;
    font-size: 0.9rem;
    color: var(--yorha-accent-warm);
    border: 1px solid rgba(255, 191, 0, 0.3);
  }

  /* === FOOTER === */
  .search-footer {
    background: rgba(26, 26, 26, 0.9);
    border-top: 1px solid rgba(255, 191, 0, 0.3);
    padding: 1.5rem 2rem;
  }

  .footer-info {
    max-width: 1200px;
    margin: 0 auto;
  }

  .database-stats {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.8rem;
  }

  .stat-group {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  .stat-label {
    color: var(--yorha-muted);
  }

  .stat-value {
    color: var(--yorha-light);
    font-weight: 600;
  }

  .stat-value.active {
    color: var(--yorha-success);
  }

  /* === ANIMATIONS === */
  @keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
  }

  @keyframes fadeInUp {
    from {
      opacity: 0;
      transform: translateY(50px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }

  @keyframes loading {
    0% { transform: translateX(-100%); }
    50% { transform: translateX(0%); }
    100% { transform: translateX(100%); }
  }

  @keyframes pulse {
    0%, 100% { opacity: 0.7; transform: scale(1); }
    50% { opacity: 1; transform: scale(1.1); }
  }

  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }

  /* === RESPONSIVE DESIGN === */
  @media (max-width: 768px) {
    .search-header {
      flex-direction: column;
      gap: 1rem;
      text-align: center;
    }

    .header-left {
      flex-direction: column;
      gap: 1rem;
    }

    .header-stats {
      gap: 1rem;
    }

    .search-controls {
      flex-direction: column;
      gap: 1rem;
      align-items: stretch;
    }

    .tips-grid {
      grid-template-columns: 1fr;
    }

    .database-stats {
      flex-direction: column;
      gap: 1rem;
      text-align: center;
    }

    .result-actions {
      flex-wrap: wrap;
      gap: 0.5rem;
    }

    .action-button {
      flex: 1;
      min-width: 0;
    }
  }
</style>