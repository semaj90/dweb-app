<script lang="ts">
  import { onMount } from 'svelte';

  // Vector search state
  let searchQuery = $state('');
  let searchResults = $state([]);
  let isSearching = $state(false);
  let vectorSpace = $state('legal-documents');
  let similarityThreshold = $state(0.7);
  let maxResults = $state(10);
  let embeddingModel = $state('nomic-embed-text');
  let searchStats = $state({
    totalDocs: 0,
    searchTime: 0,
    vectorDim: 384,
    matchedDocs: 0
  });

  // Demo documents for visualization
  let demoDocuments = $state([
    {
      id: 'doc1',
      title: 'Commercial Lease Agreement',
      content: 'This lease agreement governs the rental of commercial property including liability clauses, termination conditions, and rent escalation terms.',
      vector: [0.8, 0.3, 0.6, 0.2], // Simplified 4D representation
      similarity: 0,
      category: 'Contract'
    },
    {
      id: 'doc2',
      title: 'Employment Contract Template',
      content: 'Standard employment agreement with non-compete clauses, compensation structure, and intellectual property assignments.',
      vector: [0.4, 0.9, 0.1, 0.7],
      similarity: 0,
      category: 'Employment'
    },
    {
      id: 'doc3',
      title: 'Liability Waiver Document',
      content: 'Legal waiver for liability protection in commercial activities, includes risk assumption and indemnification terms.',
      vector: [0.7, 0.2, 0.8, 0.4],
      similarity: 0,
      category: 'Legal'
    },
    {
      id: 'doc4',
      title: 'Intellectual Property License',
      content: 'Software licensing agreement covering usage rights, distribution terms, and intellectual property protection measures.',
      vector: [0.3, 0.6, 0.4, 0.9],
      similarity: 0,
      category: 'IP'
    },
    {
      id: 'doc5',
      title: 'Privacy Policy Template',
      content: 'Comprehensive privacy policy covering data collection, processing, storage, and user rights under GDPR compliance.',
      vector: [0.5, 0.4, 0.7, 0.3],
      similarity: 0,
      category: 'Privacy'
    }
  ]);

  const vectorSpaces = [
    { id: 'legal-documents', name: 'Legal Documents', description: 'Legal contracts and agreements' },
    { id: 'case-law', name: 'Case Law', description: 'Court decisions and precedents' },
    { id: 'regulations', name: 'Regulations', description: 'Government regulations and statutes' },
    { id: 'policies', name: 'Policies', description: 'Corporate policies and procedures' }
  ];

  const embeddingModels = [
    { id: 'nomic-embed-text', name: 'Nomic Embed Text', dimensions: 384 },
    { id: 'all-MiniLM-L6-v2', name: 'All-MiniLM-L6-v2', dimensions: 384 },
    { id: 'text-embedding-ada-002', name: 'OpenAI Ada-002', dimensions: 1536 },
    { id: 'instructor-xl', name: 'Instructor-XL', dimensions: 768 }
  ];

  // Simulate vector similarity calculation
  function calculateSimilarity(queryVector: number[], docVector: number[]): number {
    let dotProduct = 0;
    let queryMagnitude = 0;
    let docMagnitude = 0;
    
    for (let i = 0; i < Math.min(queryVector.length, docVector.length); i++) {
      dotProduct += queryVector[i] * docVector[i];
      queryMagnitude += queryVector[i] * queryVector[i];
      docMagnitude += docVector[i] * docVector[i];
    }
    
    return dotProduct / (Math.sqrt(queryMagnitude) * Math.sqrt(docMagnitude));
  }

  async function performVectorSearch() {
    if (!searchQuery.trim()) return;

    isSearching = true;
    searchResults = [];
    
    const startTime = performance.now();
    
    try {
      // Simulate embedding generation for query
      await new Promise(resolve => setTimeout(resolve, 800));
      
      // Generate synthetic query vector (normally would come from embedding model)
      const queryVector = [
        Math.random(),
        Math.random(),
        Math.random(),
        Math.random()
      ];
      
      // Calculate similarities with all documents
      const scoredDocs = demoDocuments.map(doc => {
        const similarity = calculateSimilarity(queryVector, doc.vector);
        return {
          ...doc,
          similarity,
          highlighted: highlightMatches(doc.content, searchQuery)
        };
      });
      
      // Filter and sort by similarity
      const filteredDocs = scoredDocs
        .filter(doc => doc.similarity >= similarityThreshold)
        .sort((a, b) => b.similarity - a.similarity)
        .slice(0, maxResults);
      
      const endTime = performance.now();
      
      searchResults = filteredDocs;
      searchStats = {
        totalDocs: demoDocuments.length,
        searchTime: Math.round(endTime - startTime),
        vectorDim: embeddingModels.find(m => m.id === embeddingModel)?.dimensions || 384,
        matchedDocs: filteredDocs.length
      };
      
    } catch (error) {
      console.error('Vector search error:', error);
    }
    
    isSearching = false;
  }

  function highlightMatches(text: string, query: string): string {
    if (!query.trim()) return text;
    
    const queryWords = query.toLowerCase().split(' ');
    let highlighted = text;
    
    queryWords.forEach(word => {
      const regex = new RegExp(`(${word})`, 'gi');
      highlighted = highlighted.replace(regex, '<mark>$1</mark>');
    });
    
    return highlighted;
  }

  function clearSearch() {
    searchQuery = '';
    searchResults = [];
    searchStats = {
      totalDocs: 0,
      searchTime: 0,
      vectorDim: 384,
      matchedDocs: 0
    };
  }

  function loadSampleQuery() {
    const sampleQueries = [
      'liability protection in commercial contracts',
      'intellectual property licensing terms',
      'employment contract non-compete clauses',
      'privacy policy data processing requirements',
      'commercial lease termination conditions'
    ];
    
    searchQuery = sampleQueries[Math.floor(Math.random() * sampleQueries.length)];
  }

  async function explainVector(doc: unknown) {
    // Simulate vector explanation
    const explanation = {
      document: doc.title,
      vector: doc.vector,
      dimensions: [
        { name: 'Legal Concepts', weight: doc.vector[0], description: 'Legal terminology and concepts' },
        { name: 'Commercial Terms', weight: doc.vector[1], description: 'Business and commercial language' },
        { name: 'Risk Factors', weight: doc.vector[2], description: 'Risk and liability related content' },
        { name: 'Technical Terms', weight: doc.vector[3], description: 'Technical and procedural language' }
      ],
      similarity: doc.similarity
    };
    
    // You could open a modal or update a sidebar with this explanation
    console.log('Vector explanation:', explanation);
  }

  onMount(() => {
    // Initialize demo
    searchStats.totalDocs = demoDocuments.length;
  });
</script>

<svelte:head>
  <title>Vector Search Development Tools</title>
</svelte:head>

<div class="vector-search-demo">
  <div class="demo-header">
    <h1 class="page-title">
      <span class="title-icon">🔍</span>
      VECTOR SEARCH DEVELOPMENT
    </h1>
    <div class="search-stats">
      <div class="stat-item">
        <div class="stat-value">{searchStats.matchedDocs}</div>
        <div class="stat-label">MATCHED DOCS</div>
      </div>
      <div class="stat-item">
        <div class="stat-value">{searchStats.vectorDim}D</div>
        <div class="stat-label">VECTOR SPACE</div>
      </div>
    </div>
  </div>

  <div class="demo-grid">
    <!-- Search Configuration -->
    <section class="config-panel">
      <h2 class="section-title">SEARCH CONFIGURATION</h2>
      
      <div class="config-group">
        <label class="config-label">Vector Space</label>
        <select bind:value={vectorSpace} class="config-select">
          {#each vectorSpaces as space}
            <option value={space.id}>{space.name} - {space.description}</option>
          {/each}
        </select>
      </div>

      <div class="config-group">
        <label class="config-label">Embedding Model</label>
        <select bind:value={embeddingModel} class="config-select">
          {#each embeddingModels as model}
            <option value={model.id}>{model.name} ({model.dimensions}D)</option>
          {/each}
        </select>
      </div>

      <div class="config-group">
        <label class="config-label">Similarity Threshold</label>
        <input
          type="range"
          min="0.1"
          max="1"
          step="0.05"
          bind:value={similarityThreshold}
          class="config-slider"
        />
        <div class="slider-value">{similarityThreshold.toFixed(2)}</div>
      </div>

      <div class="config-group">
        <label class="config-label">Max Results</label>
        <input
          type="range"
          min="5"
          max="50"
          step="5"
          bind:value={maxResults}
          class="config-slider"
        />
        <div class="slider-value">{maxResults} documents</div>
      </div>

      <div class="config-buttons">
        <button class="config-button sample-button" onclick={loadSampleQuery}>
          <div class="button-icon">🎯</div>
          <div class="button-text">SAMPLE QUERY</div>
        </button>
        <button class="config-button clear-button" onclick={clearSearch}>
          <div class="button-icon">🗑️</div>
          <div class="button-text">CLEAR SEARCH</div>
        </button>
      </div>
    </section>

    <!-- Search Interface -->
    <section class="search-panel">
      <h2 class="section-title">VECTOR SEARCH INTERFACE</h2>
      
      <div class="search-container">
        <div class="search-input-group">
          <input
            bind:value={searchQuery}
            placeholder="Enter semantic search query..."
            class="search-input"
            onkeydown={(e) => e.key === 'Enter' && performVectorSearch()}
          />
          <button
            class="search-button {isSearching ? 'searching' : ''}"
            onclick={performVectorSearch}
            disabled={isSearching || !searchQuery.trim()}
          >
            <div class="button-icon">
              {isSearching ? '⏳' : '🔍'}
            </div>
            <div class="button-text">
              {isSearching ? 'SEARCHING...' : 'VECTOR SEARCH'}
            </div>
          </button>
        </div>

        <div class="search-info">
          <div class="info-item">
            <span class="info-label">SEARCH TIME:</span>
            <span class="info-value">{searchStats.searchTime}ms</span>
          </div>
          <div class="info-item">
            <span class="info-label">TOTAL DOCS:</span>
            <span class="info-value">{searchStats.totalDocs}</span>
          </div>
          <div class="info-item">
            <span class="info-label">MODEL:</span>
            <span class="info-value">{embeddingModel}</span>
          </div>
        </div>
      </div>
    </section>

    <!-- Search Results -->
    <section class="results-panel">
      <h2 class="section-title">SEARCH RESULTS</h2>
      
      <div class="results-container">
        {#if searchResults.length === 0 && !isSearching}
          <div class="empty-results">
            <div class="empty-icon">🔍</div>
            <div class="empty-text">No search results</div>
            <div class="empty-subtext">
              {searchQuery ? 'Try adjusting similarity threshold or search terms' : 'Enter a search query to find similar documents'}
            </div>
          </div>
        {:else if isSearching}
          <div class="searching-state">
            <div class="search-loader">🔄</div>
            <div class="search-text">Processing vector search...</div>
            <div class="search-subtext">Generating embeddings and calculating similarities</div>
          </div>
        {:else}
          <div class="results-list">
            {#each searchResults as result, index}
              <div class="result-item">
                <div class="result-header">
                  <div class="result-rank">#{index + 1}</div>
                  <div class="result-title">{result.title}</div>
                  <div class="result-similarity">
                    {(result.similarity * 100).toFixed(1)}%
                  </div>
                </div>
                
                <div class="result-meta">
                  <div class="result-category">{result.category}</div>
                  <div class="result-id">ID: {result.id}</div>
                </div>
                
                <div class="result-content">
                  {@html result.highlighted || result.content}
                </div>
                
                <div class="result-actions">
                  <button class="action-button" onclick={() => explainVector(result)}>
                    <div class="button-icon">📊</div>
                    <div class="button-text">EXPLAIN VECTOR</div>
                  </button>
                  <div class="similarity-bar">
                    <div class="bar-fill" style="width: {result.similarity * 100}%"></div>
                    <div class="bar-label">Similarity</div>
                  </div>
                </div>
              </div>
            {/each}
          </div>
        {/if}
      </div>
    </section>

    <!-- Vector Visualization -->
    <section class="visualization-panel">
      <h2 class="section-title">VECTOR SPACE VISUALIZATION</h2>
      
      <div class="vector-space">
        <div class="space-info">
          <div class="space-title">4D Vector Space (Simplified View)</div>
          <div class="space-description">
            Each document is represented as a point in high-dimensional space.
            Distance between points indicates semantic similarity.
          </div>
        </div>
        
        <div class="vector-plot">
          <svg viewBox="0 0 300 300" class="plot-svg">
            <!-- Background grid -->
            <defs>
              <pattern id="grid" width="30" height="30" patternUnits="userSpaceOnUse">
                <path d="M 30 0 L 0 0 0 30" fill="none" stroke="rgba(255,255,255,0.1)" stroke-width="1"/>
              </pattern>
            </defs>
            <rect width="100%" height="100%" fill="url(#grid)" />
            
            <!-- Query point (if searching) -->
            {#if searchQuery && searchResults.length > 0}
              <circle cx="150" cy="150" r="8" fill="#fb7185" stroke="#ffffff" stroke-width="2">
                <title>Search Query</title>
              </circle>
              <text x="160" y="155" fill="#fb7185" font-size="10" font-weight="bold">Query</text>
            {/if}
            
            <!-- Document points -->
            {#each demoDocuments as doc, index}
              {@const x = 50 + doc.vector[0] * 200}
              {@const y = 50 + doc.vector[1] * 200}
              {@const similarity = searchResults.find(r => r.id === doc.id)?.similarity || 0}
              {@const isMatch = similarity >= similarityThreshold}
              
              <circle 
                cx={x} 
                cy={y} 
                r={isMatch ? 6 : 4} 
                fill={isMatch ? '#00ff41' : '#4ecdc4'} 
                stroke="#ffffff" 
                stroke-width="1"
                opacity={searchQuery ? (isMatch ? 1 : 0.4) : 0.8}
              >
                <title>{doc.title} (Similarity: {(similarity * 100).toFixed(1)}%)</title>
              </circle>
              
              {#if isMatch || !searchQuery}
                <text 
                  x={x + 8} 
                  y={y + 3} 
                  fill={isMatch ? '#00ff41' : '#4ecdc4'} 
                  font-size="8"
                  opacity={searchQuery ? (isMatch ? 1 : 0.4) : 0.8}
                >
                  {doc.category}
                </text>
              {/if}
            {/each}
          </svg>
        </div>
        
        <div class="legend">
          <div class="legend-item">
            <div class="legend-dot query"></div>
            <div class="legend-label">Search Query</div>
          </div>
          <div class="legend-item">
            <div class="legend-dot match"></div>
            <div class="legend-label">Matching Documents</div>
          </div>
          <div class="legend-item">
            <div class="legend-dot doc"></div>
            <div class="legend-label">All Documents</div>
          </div>
        </div>
      </div>
    </section>

    <!-- Performance Metrics -->
    <section class="metrics-panel">
      <h2 class="section-title">PERFORMANCE METRICS</h2>
      
      <div class="metrics-grid">
        <div class="metric-card">
          <div class="metric-icon">⚡</div>
          <div class="metric-content">
            <div class="metric-value">{searchStats.searchTime}ms</div>
            <div class="metric-label">Search Latency</div>
          </div>
        </div>
        
        <div class="metric-card">
          <div class="metric-icon">🎯</div>
          <div class="metric-content">
            <div class="metric-value">{searchResults.length > 0 ? (searchResults[0].similarity * 100).toFixed(1) : 0}%</div>
            <div class="metric-label">Top Similarity</div>
          </div>
        </div>
        
        <div class="metric-card">
          <div class="metric-icon">📊</div>
          <div class="metric-content">
            <div class="metric-value">{searchStats.vectorDim}</div>
            <div class="metric-label">Vector Dimensions</div>
          </div>
        </div>
        
        <div class="metric-card">
          <div class="metric-icon">🔍</div>
          <div class="metric-content">
            <div class="metric-value">{(similarityThreshold * 100).toFixed(0)}%</div>
            <div class="metric-label">Threshold</div>
          </div>
        </div>
      </div>
    </section>
  </div>

  <!-- Back Navigation -->
  <div class="navigation-footer">
    <a href="/dev/mcp-tools" class="back-button">
      <span class="button-icon">⬅️</span>
      BACK TO MCP TOOLS
    </a>
    <a href="/" class="home-button">
      <span class="button-icon">🏠</span>
      COMMAND CENTER
    </a>
  </div>
</div>

<style>
  .vector-search-demo {
    min-height: 100vh;
    background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
    color: #ffffff;
    font-family: 'JetBrains Mono', monospace;
    padding: 2rem;
  }

  .demo-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 3rem;
    padding: 2rem;
    background: linear-gradient(145deg, rgba(26, 26, 26, 0.9), rgba(42, 42, 42, 0.9));
    border: 2px solid #4ecdc4;
    border-radius: 8px;
  }

  .page-title {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin: 0;
    font-size: 2.5rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 3px;
    background: linear-gradient(45deg, #4ecdc4, #00ff41);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }

  .title-icon {
    font-size: 3rem;
    filter: drop-shadow(0 0 10px #4ecdc4);
  }

  .search-stats {
    display: flex;
    gap: 2rem;
  }

  .stat-item {
    text-align: center;
  }

  .stat-value {
    font-size: 2rem;
    font-weight: 700;
    color: #4ecdc4;
    margin-bottom: 0.5rem;
  }

  .stat-label {
    font-size: 0.8rem;
    color: #f0f0f0;
    opacity: 0.8;
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  .demo-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 2rem;
    margin-bottom: 3rem;
  }

  .section-title {
    font-size: 1.2rem;
    font-weight: 600;
    margin: 0 0 1.5rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #4ecdc4;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #4ecdc4;
  }

  /* Configuration Panel */
  .config-panel {
    background: linear-gradient(145deg, rgba(26, 26, 26, 0.9), rgba(42, 42, 42, 0.9));
    padding: 2rem;
    border-radius: 8px;
    border: 1px solid rgba(78, 205, 196, 0.3);
  }

  .config-group {
    margin-bottom: 2rem;
  }

  .config-label {
    display: block;
    font-size: 0.9rem;
    color: #f0f0f0;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.5rem;
  }

  .config-select {
    width: 100%;
    background: rgba(26, 26, 26, 0.8);
    border: 2px solid rgba(78, 205, 196, 0.5);
    color: #ffffff;
    padding: 1rem;
    font-family: inherit;
    font-size: 0.9rem;
    border-radius: 4px;
  }

  .config-slider {
    width: 100%;
    appearance: none;
    height: 6px;
    background: rgba(255, 255, 255, 0.2);
    outline: none;
    border-radius: 3px;
    margin-bottom: 0.5rem;
  }

  .config-slider::-webkit-slider-thumb {
    appearance: none;
    width: 20px;
    height: 20px;
    background: #4ecdc4;
    cursor: pointer;
    border-radius: 50%;
  }

  .slider-value {
    text-align: center;
    font-size: 0.9rem;
    color: #4ecdc4;
    font-weight: 600;
    padding: 0.3rem;
    background: rgba(78, 205, 196, 0.1);
    border-radius: 3px;
  }

  .config-buttons {
    display: flex;
    gap: 1rem;
  }

  .config-button {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.3rem;
    background: linear-gradient(145deg, #4ecdc4, #00ff41);
    color: #0a0a0a;
    border: none;
    padding: 1rem;
    font-family: inherit;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    cursor: pointer;
    transition: all 0.3s ease;
    border-radius: 6px;
  }

  .config-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 25px rgba(78, 205, 196, 0.3);
  }

  .clear-button {
    background: linear-gradient(145deg, #ff6b6b, #ffbf00);
  }

  .button-icon {
    font-size: 1.2rem;
  }

  .button-text {
    font-size: 0.8rem;
  }

  /* Search Panel */
  .search-panel {
    background: linear-gradient(145deg, rgba(26, 26, 26, 0.9), rgba(42, 42, 42, 0.9));
    padding: 2rem;
    border-radius: 8px;
    border: 1px solid rgba(78, 205, 196, 0.3);
  }

  .search-container {
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
  }

  .search-input-group {
    display: flex;
    gap: 1rem;
  }

  .search-input {
    flex: 1;
    background: rgba(26, 26, 26, 0.8);
    border: 2px solid rgba(78, 205, 196, 0.5);
    color: #ffffff;
    padding: 1rem;
    font-family: inherit;
    font-size: 1rem;
    border-radius: 4px;
    transition: all 0.3s ease;
  }

  .search-input:focus {
    outline: none;
    border-color: #4ecdc4;
    box-shadow: 0 0 20px rgba(78, 205, 196, 0.3);
  }

  .search-button {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    background: linear-gradient(145deg, #4ecdc4, #00ff41);
    color: #0a0a0a;
    border: none;
    padding: 1rem 2rem;
    font-family: inherit;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    cursor: pointer;
    transition: all 0.3s ease;
    border-radius: 4px;
  }

  .search-button:hover:not(:disabled) {
    transform: scale(1.05);
    box-shadow: 0 10px 25px rgba(78, 205, 196, 0.3);
  }

  .search-button:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }

  .search-button.searching {
    animation: pulse 1.5s ease-in-out infinite;
  }

  .search-info {
    display: flex;
    justify-content: space-between;
    gap: 1rem;
  }

  .info-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 4px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    font-size: 0.8rem;
  }

  .info-label {
    color: #f0f0f0;
    opacity: 0.8;
  }

  .info-value {
    color: #4ecdc4;
    font-weight: 600;
  }

  /* Results Panel */
  .results-panel {
    grid-column: 1 / -1;
    background: linear-gradient(145deg, rgba(26, 26, 26, 0.9), rgba(42, 42, 42, 0.9));
    padding: 2rem;
    border-radius: 8px;
    border: 1px solid rgba(78, 205, 196, 0.3);
  }

  .results-container {
    max-height: 500px;
    overflow-y: auto;
  }

  .empty-results,
  .searching-state {
    text-align: center;
    padding: 4rem;
    color: #f0f0f0;
    opacity: 0.7;
  }

  .empty-icon,
  .search-loader {
    font-size: 4rem;
    margin-bottom: 2rem;
  }

  .search-loader {
    animation: spin 1s linear infinite;
  }

  .empty-text,
  .search-text {
    font-size: 1.2rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
  }

  .empty-subtext,
  .search-subtext {
    font-size: 0.9rem;
    opacity: 0.8;
  }

  .results-list {
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
  }

  .result-item {
    background: rgba(255, 255, 255, 0.05);
    padding: 1.5rem;
    border-radius: 8px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    transition: all 0.3s ease;
  }

  .result-item:hover {
    background: rgba(78, 205, 196, 0.05);
    border-color: rgba(78, 205, 196, 0.3);
  }

  .result-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1rem;
  }

  .result-rank {
    background: #4ecdc4;
    color: #0a0a0a;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 700;
  }

  .result-title {
    flex: 1;
    font-size: 1.1rem;
    font-weight: 600;
    color: #ffffff;
  }

  .result-similarity {
    background: rgba(0, 255, 65, 0.2);
    color: #00ff41;
    padding: 0.3rem 0.8rem;
    border-radius: 4px;
    font-size: 0.8rem;
    font-weight: 700;
    border: 1px solid #00ff41;
  }

  .result-meta {
    display: flex;
    gap: 1rem;
    margin-bottom: 1rem;
    font-size: 0.8rem;
    opacity: 0.7;
  }

  .result-category {
    background: rgba(167, 139, 250, 0.2);
    color: #a78bfa;
    padding: 0.2rem 0.6rem;
    border-radius: 3px;
    border: 1px solid #a78bfa;
  }

  .result-id {
    color: #f0f0f0;
  }

  .result-content {
    margin-bottom: 1rem;
    line-height: 1.6;
    color: #f0f0f0;
  }

  .result-content :global(mark) {
    background: rgba(255, 191, 0, 0.3);
    color: #ffbf00;
    padding: 0.2rem 0.3rem;
    border-radius: 3px;
  }

  .result-actions {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 1rem;
  }

  .action-button {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    background: transparent;
    border: 2px solid #4ecdc4;
    color: #4ecdc4;
    padding: 0.5rem 1rem;
    font-family: inherit;
    font-size: 0.8rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
    border-radius: 4px;
  }

  .action-button:hover {
    background: #4ecdc4;
    color: #0a0a0a;
  }

  .similarity-bar {
    flex: 1;
    max-width: 200px;
    position: relative;
    background: rgba(255, 255, 255, 0.2);
    height: 8px;
    border-radius: 4px;
    overflow: hidden;
  }

  .bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #4ecdc4, #00ff41);
    border-radius: 4px;
    transition: width 0.5s ease;
  }

  .bar-label {
    position: absolute;
    top: -25px;
    right: 0;
    font-size: 0.7rem;
    color: #f0f0f0;
    opacity: 0.7;
  }

  /* Visualization Panel */
  .visualization-panel {
    grid-column: 1 / -1;
    background: linear-gradient(145deg, rgba(26, 26, 26, 0.9), rgba(42, 42, 42, 0.9));
    padding: 2rem;
    border-radius: 8px;
    border: 1px solid rgba(78, 205, 196, 0.3);
  }

  .vector-space {
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
  }

  .space-info {
    text-align: center;
  }

  .space-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #4ecdc4;
    margin-bottom: 0.5rem;
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  .space-description {
    font-size: 0.9rem;
    color: #f0f0f0;
    opacity: 0.8;
    line-height: 1.4;
  }

  .vector-plot {
    display: flex;
    justify-content: center;
  }

  .plot-svg {
    width: 100%;
    max-width: 400px;
    height: 300px;
    background: rgba(0, 0, 0, 0.3);
    border: 1px solid rgba(78, 205, 196, 0.3);
    border-radius: 8px;
  }

  .legend {
    display: flex;
    justify-content: center;
    gap: 2rem;
    flex-wrap: wrap;
  }

  .legend-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  .legend-dot {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    border: 2px solid #ffffff;
  }

  .legend-dot.query {
    background: #fb7185;
  }

  .legend-dot.match {
    background: #00ff41;
  }

  .legend-dot.doc {
    background: #4ecdc4;
  }

  .legend-label {
    font-size: 0.8rem;
    color: #f0f0f0;
    opacity: 0.8;
  }

  /* Metrics Panel */
  .metrics-panel {
    grid-column: 1 / -1;
    background: linear-gradient(145deg, rgba(26, 26, 26, 0.9), rgba(42, 42, 42, 0.9));
    padding: 2rem;
    border-radius: 8px;
    border: 1px solid rgba(78, 205, 196, 0.3);
  }

  .metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
  }

  .metric-card {
    display: flex;
    align-items: center;
    gap: 1rem;
    background: rgba(255, 255, 255, 0.05);
    padding: 1.5rem;
    border-radius: 6px;
    border: 1px solid rgba(255, 255, 255, 0.1);
  }

  .metric-icon {
    font-size: 2rem;
    color: #4ecdc4;
  }

  .metric-content {
    flex: 1;
  }

  .metric-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 0.3rem;
  }

  .metric-label {
    font-size: 0.8rem;
    color: #f0f0f0;
    opacity: 0.8;
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  /* Navigation Footer */
  .navigation-footer {
    display: flex;
    justify-content: center;
    gap: 2rem;
    margin-top: 3rem;
  }

  .back-button,
  .home-button {
    display: inline-flex;
    align-items: center;
    gap: 1rem;
    background: linear-gradient(145deg, #4ecdc4, #00ff41);
    color: #0a0a0a;
    padding: 1rem 2rem;
    border-radius: 6px;
    text-decoration: none;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    transition: all 0.3s ease;
  }

  .back-button:hover,
  .home-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 25px rgba(78, 205, 196, 0.3);
  }

  /* Animations */
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
  }

  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }

  /* Responsive Design */
  @media (max-width: 768px) {
    .demo-header {
      flex-direction: column;
      gap: 1rem;
      text-align: center;
    }

    .page-title {
      font-size: 2rem;
    }

    .demo-grid {
      grid-template-columns: 1fr;
    }

    .search-input-group {
      flex-direction: column;
    }

    .search-info {
      flex-direction: column;
      gap: 0.5rem;
    }

    .metrics-grid {
      grid-template-columns: 1fr;
    }

    .legend {
      flex-direction: column;
      gap: 1rem;
      align-items: center;
    }

    .navigation-footer {
      flex-direction: column;
      gap: 1rem;
    }
  }
</style>