<script lang="ts">
  import { onMount } from 'svelte';
  
  // Help system state
  let searchQuery = $state('');
  let selectedCategory = $state('all');
  let expandedSections = $state(new Set<string>());
  
  // Help categories and topics
  const helpCategories = {
    all: { label: 'ALL TOPICS', icon: '📚', color: '#ffbf00' },
    getting_started: { label: 'GETTING STARTED', icon: '🚀', color: '#00ff41' },
    ai_features: { label: 'AI FEATURES', icon: '🤖', color: '#4ecdc4' },
    legal_tools: { label: 'LEGAL TOOLS', icon: '⚖️', color: '#ff6b6b' },
    system: { label: 'SYSTEM', icon: '⚙️', color: '#a78bfa' },
    troubleshooting: { label: 'TROUBLESHOOTING', icon: '🔧', color: '#fb7185' }
  };

  const helpTopics = [
    {
      id: 'welcome',
      title: 'Welcome to YoRHa Legal AI',
      category: 'getting_started',
      content: 'YoRHa Legal AI is a comprehensive legal document analysis and AI-powered research platform. This system combines advanced artificial intelligence with legal expertise to provide powerful tools for legal professionals.',
      tags: ['welcome', 'introduction', 'overview']
    },
    {
      id: 'ai_assistant',
      title: 'Using the AI Assistant',
      category: 'ai_features',
      content: 'The AI Assistant provides intelligent support for legal research, document analysis, and case management. Ask questions in natural language and receive context-aware responses based on legal knowledge.',
      tags: ['ai', 'assistant', 'chat', 'questions']
    },
    {
      id: 'document_upload',
      title: 'Uploading Legal Documents',
      category: 'legal_tools',
      content: 'Upload legal documents in various formats (PDF, DOCX, TXT). The system will automatically extract text, analyze content, and make documents searchable through semantic search.',
      tags: ['upload', 'documents', 'pdf', 'analysis']
    },
    {
      id: 'vector_search',
      title: 'Semantic Search',
      category: 'ai_features',
      content: 'Use semantic search to find relevant information across your legal documents. The system understands context and meaning, not just exact keyword matches.',
      tags: ['search', 'semantic', 'vector', 'similarity']
    },
    {
      id: 'case_management',
      title: 'Managing Legal Cases',
      category: 'legal_tools',
      content: 'Create and manage legal cases with associated documents, evidence, and persons of interest. Track case progress and generate comprehensive reports.',
      tags: ['cases', 'management', 'evidence', 'tracking']
    },
    {
      id: 'gpu_acceleration',
      title: 'GPU Acceleration',
      category: 'system',
      content: 'The system uses GPU acceleration for AI processing, providing faster document analysis, embeddings generation, and model inference.',
      tags: ['gpu', 'performance', 'acceleration', 'speed']
    },
    {
      id: 'api_integration',
      title: 'API Integration',
      category: 'system',
      content: 'YoRHa Legal AI provides REST, gRPC, and QUIC APIs for integration with external systems. Comprehensive API documentation is available.',
      tags: ['api', 'integration', 'rest', 'grpc', 'quic']
    },
    {
      id: 'troubleshooting_upload',
      title: 'Document Upload Issues',
      category: 'troubleshooting',
      content: 'If document uploads fail, check file size limits (max 100MB), ensure supported formats (PDF, DOCX, TXT), and verify network connection.',
      tags: ['troubleshooting', 'upload', 'files', 'errors']
    },
    {
      id: 'performance_issues',
      title: 'Performance Optimization',
      category: 'troubleshooting',
      content: 'For optimal performance, ensure GPU drivers are updated, sufficient RAM is available, and close unnecessary applications. AI processing is resource-intensive.',
      tags: ['performance', 'optimization', 'gpu', 'memory']
    },
    {
      id: 'keyboard_shortcuts',
      title: 'Keyboard Shortcuts',
      category: 'getting_started',
      content: 'Speed up your workflow with keyboard shortcuts: Ctrl+K (Search), Ctrl+U (Upload), Ctrl+N (New Case), Ctrl+/ (Help), Ctrl+Shift+A (AI Assistant).',
      tags: ['keyboard', 'shortcuts', 'navigation', 'productivity']
    }
  ];

  // Filtered topics based on search and category
  let filteredTopics = $derived(() => {
    let topics = selectedCategory === 'all' 
      ? helpTopics 
      : helpTopics.filter(topic => topic.category === selectedCategory);

    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      topics = topics.filter(topic =>
        topic.title.toLowerCase().includes(query) ||
        topic.content.toLowerCase().includes(query) ||
        topic.tags.some(tag => tag.toLowerCase().includes(query))
      );
    }

    return topics;
  });

  function toggleSection(sectionId: string) {
    if (expandedSections.has(sectionId)) {
      expandedSections.delete(sectionId);
    } else {
      expandedSections.add(sectionId);
    }
    expandedSections = new Set(expandedSections);
  }

  function selectCategory(category: string) {
    selectedCategory = category;
    searchQuery = '';
  }

  onMount(() => {
    // Auto-expand first section
    if (helpTopics.length > 0) {
      expandedSections.add(helpTopics[0].id);
      expandedSections = new Set(expandedSections);
    }
  });
</script>

<svelte:head>
  <title>YoRHa Help & Documentation</title>
</svelte:head>

<div class="help-center">
  <div class="help-header">
    <h1 class="page-title">
      <span class="title-icon">❓</span>
      YORHA HELP & DOCUMENTATION
    </h1>
    <div class="help-stats">
      <div class="stat-item">
        <div class="stat-value">{filteredTopics.length}</div>
        <div class="stat-label">TOPICS FOUND</div>
      </div>
      <div class="stat-item">
        <div class="stat-value">{helpTopics.length}</div>
        <div class="stat-label">TOTAL TOPICS</div>
      </div>
    </div>
  </div>

  <!-- Search and Filter Controls -->
  <section class="search-controls">
    <div class="search-container">
      <input
        bind:value={searchQuery}
        placeholder="SEARCH HELP TOPICS..."
        class="search-input"
      />
      <div class="search-icon">🔍</div>
    </div>

    <div class="category-filters">
      {#each Object.entries(helpCategories) as [key, category]}
        <button
          class="category-button {selectedCategory === key ? 'active' : ''}"
          onclick={() => selectCategory(key)}
          style="border-color: {category.color}; color: {selectedCategory === key ? '#000' : category.color}"
        >
          {category.icon} {category.label}
        </button>
      {/each}
    </div>
  </section>

  <!-- Help Topics -->
  <section class="help-topics">
    <div class="topics-container">
      {#each filteredTopics as topic (topic.id)}
        <div class="topic-card">
          <button
            class="topic-header"
            onclick={() => toggleSection(topic.id)}
          >
            <div class="topic-info">
              <div class="topic-icon">{helpCategories[topic.category].icon}</div>
              <div class="topic-title">{topic.title}</div>
            </div>
            <div class="expand-icon {expandedSections.has(topic.id) ? 'expanded' : ''}">
              ▼
            </div>
          </button>

          {#if expandedSections.has(topic.id)}
            <div class="topic-content">
              <div class="content-text">{topic.content}</div>
              <div class="topic-tags">
                {#each topic.tags as tag}
                  <span class="tag">{tag}</span>
                {/each}
              </div>
            </div>
          {/if}
        </div>
      {/each}

      {#if filteredTopics.length === 0}
        <div class="empty-state">
          <div class="empty-icon">🔍</div>
          <div class="empty-title">NO HELP TOPICS FOUND</div>
          <div class="empty-description">
            No help topics match your current search criteria.
            Try adjusting your search terms or selecting a different category.
          </div>
          <button
            class="empty-action"
            onclick={() => { searchQuery = ''; selectedCategory = 'all'; }}
          >
            SHOW ALL TOPICS
          </button>
        </div>
      {/if}
    </div>
  </section>

  <!-- Quick Actions -->
  <section class="quick-actions">
    <h2 class="section-title">QUICK ACTIONS</h2>
    <div class="actions-grid">
      <button class="action-button">
        <div class="action-icon">🎯</div>
        <div class="action-content">
          <div class="action-title">INTERACTIVE TUTORIAL</div>
          <div class="action-description">Step-by-step system walkthrough</div>
        </div>
      </button>
      
      <button class="action-button">
        <div class="action-icon">📖</div>
        <div class="action-content">
          <div class="action-title">USER MANUAL</div>
          <div class="action-description">Complete system documentation</div>
        </div>
      </button>
      
      <button class="action-button">
        <div class="action-icon">🎬</div>
        <div class="action-content">
          <div class="action-title">VIDEO TUTORIALS</div>
          <div class="action-description">Visual learning resources</div>
        </div>
      </button>
      
      <button class="action-button">
        <div class="action-icon">💬</div>
        <div class="action-content">
          <div class="action-title">CONTACT SUPPORT</div>
          <div class="action-description">Get personalized assistance</div>
        </div>
      </button>
    </div>
  </section>

  <!-- System Information -->
  <section class="system-info">
    <h2 class="section-title">SYSTEM INFORMATION</h2>
    <div class="info-grid">
      <div class="info-item">
        <div class="info-label">VERSION</div>
        <div class="info-value">YoRHa Legal AI v2.1.5</div>
      </div>
      <div class="info-item">
        <div class="info-label">BUILD</div>
        <div class="info-value">Enhanced RAG Orchestration</div>
      </div>
      <div class="info-item">
        <div class="info-label">ARCHITECTURE</div>
        <div class="info-value">SvelteKit 2 + TypeScript</div>
      </div>
      <div class="info-item">
        <div class="info-label">AI MODELS</div>
        <div class="info-value">Ollama + Context7 Integration</div>
      </div>
    </div>
  </section>

  <!-- Back to Command Center -->
  <div class="navigation-footer">
    <a href="/" class="back-button">
      <span class="button-icon">⬅️</span>
      RETURN TO COMMAND CENTER
    </a>
  </div>
</div>

<style>
  .help-center {
    min-height: 100vh;
    background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
    color: #ffffff;
    font-family: 'JetBrains Mono', monospace;
    padding: 2rem;
  }

  .help-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 3rem;
    padding: 2rem;
    background: linear-gradient(145deg, rgba(26, 26, 26, 0.9), rgba(42, 42, 42, 0.9));
    border: 2px solid #ffbf00;
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
    background: linear-gradient(45deg, #ffbf00, #00ff41);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }

  .title-icon {
    font-size: 3rem;
    filter: drop-shadow(0 0 10px #ffbf00);
  }

  .help-stats {
    display: flex;
    gap: 2rem;
  }

  .stat-item {
    text-align: center;
  }

  .stat-value {
    font-size: 2rem;
    font-weight: 700;
    color: #00ff41;
    margin-bottom: 0.5rem;
  }

  .stat-label {
    font-size: 0.8rem;
    color: #f0f0f0;
    opacity: 0.8;
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  /* Search Controls */
  .search-controls {
    margin-bottom: 3rem;
    background: linear-gradient(145deg, rgba(26, 26, 26, 0.9), rgba(42, 42, 42, 0.9));
    padding: 2rem;
    border-radius: 8px;
    border: 1px solid rgba(255, 191, 0, 0.3);
  }

  .search-container {
    position: relative;
    margin-bottom: 2rem;
  }

  .search-input {
    width: 100%;
    background: rgba(26, 26, 26, 0.8);
    border: 2px solid rgba(255, 191, 0, 0.5);
    color: #ffffff;
    padding: 1rem 3rem 1rem 1.5rem;
    font-family: inherit;
    font-size: 1rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    border-radius: 4px;
    transition: all 0.3s ease;
  }

  .search-input:focus {
    outline: none;
    border-color: #ffbf00;
    box-shadow: 0 0 20px rgba(255, 191, 0, 0.3);
  }

  .search-input::placeholder {
    color: #f0f0f0;
    opacity: 0.7;
  }

  .search-icon {
    position: absolute;
    right: 1rem;
    top: 50%;
    transform: translateY(-50%);
    font-size: 1.2rem;
    color: #ffbf00;
    pointer-events: none;
  }

  .category-filters {
    display: flex;
    flex-wrap: wrap;
    gap: 1rem;
  }

  .category-button {
    background: transparent;
    border: 2px solid;
    color: #f0f0f0;
    padding: 0.8rem 1.5rem;
    font-family: inherit;
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    cursor: pointer;
    transition: all 0.3s ease;
    font-weight: 600;
    border-radius: 4px;
  }

  .category-button:hover {
    background: rgba(255, 255, 255, 0.1);
    transform: translateY(-2px);
  }

  .category-button.active {
    background: var(--category-color, #ffbf00);
    color: #0a0a0a;
  }

  /* Help Topics */
  .help-topics {
    margin-bottom: 3rem;
  }

  .topics-container {
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  .topic-card {
    background: linear-gradient(145deg, rgba(26, 26, 26, 0.9), rgba(42, 42, 42, 0.9));
    border: 1px solid rgba(255, 191, 0, 0.3);
    border-radius: 8px;
    overflow: hidden;
  }

  .topic-header {
    width: 100%;
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1.5rem 2rem;
    background: transparent;
    border: none;
    color: #ffffff;
    cursor: pointer;
    transition: all 0.3s ease;
  }

  .topic-header:hover {
    background: rgba(255, 191, 0, 0.1);
  }

  .topic-info {
    display: flex;
    align-items: center;
    gap: 1rem;
  }

  .topic-icon {
    font-size: 1.5rem;
  }

  .topic-title {
    font-size: 1.2rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    text-align: left;
  }

  .expand-icon {
    font-size: 1.2rem;
    color: #ffbf00;
    transition: transform 0.3s ease;
  }

  .expand-icon.expanded {
    transform: rotate(180deg);
  }

  .topic-content {
    padding: 0 2rem 2rem;
    border-top: 1px solid rgba(255, 191, 0, 0.2);
  }

  .content-text {
    font-size: 1rem;
    line-height: 1.6;
    margin-bottom: 1.5rem;
    color: #f0f0f0;
  }

  .topic-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
  }

  .tag {
    background: rgba(78, 205, 196, 0.2);
    color: #4ecdc4;
    padding: 0.3rem 0.6rem;
    border-radius: 3px;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    border: 1px solid #4ecdc4;
  }

  /* Empty State */
  .empty-state {
    text-align: center;
    padding: 4rem 2rem;
    color: #f0f0f0;
    opacity: 0.8;
  }

  .empty-icon {
    font-size: 4rem;
    margin-bottom: 2rem;
    opacity: 0.5;
  }

  .empty-title {
    font-size: 1.5rem;
    font-weight: 600;
    margin-bottom: 1rem;
    text-transform: uppercase;
    letter-spacing: 2px;
  }

  .empty-description {
    font-size: 1rem;
    line-height: 1.5;
    margin-bottom: 2rem;
    max-width: 500px;
    margin-left: auto;
    margin-right: auto;
  }

  .empty-action {
    background: #ffbf00;
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

  .empty-action:hover {
    background: #00ff41;
    transform: scale(1.05);
  }

  /* Quick Actions */
  .quick-actions {
    margin-bottom: 3rem;
    background: linear-gradient(145deg, rgba(26, 26, 26, 0.9), rgba(42, 42, 42, 0.9));
    padding: 2rem;
    border-radius: 8px;
    border: 1px solid rgba(255, 191, 0, 0.3);
  }

  .section-title {
    font-size: 1.2rem;
    font-weight: 600;
    margin: 0 0 1.5rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #ffbf00;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #ffbf00;
  }

  .actions-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 1rem;
  }

  .action-button {
    display: flex;
    align-items: center;
    gap: 1rem;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.2);
    color: #ffffff;
    padding: 1.5rem;
    border-radius: 6px;
    cursor: pointer;
    transition: all 0.3s ease;
    text-align: left;
  }

  .action-button:hover {
    background: rgba(255, 191, 0, 0.1);
    border-color: #ffbf00;
    transform: translateY(-2px);
  }

  .action-icon {
    font-size: 2rem;
    color: #ffbf00;
  }

  .action-content {
    flex: 1;
  }

  .action-title {
    font-size: 1rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  .action-description {
    font-size: 0.9rem;
    color: #f0f0f0;
    opacity: 0.8;
  }

  /* System Info */
  .system-info {
    margin-bottom: 3rem;
    background: linear-gradient(145deg, rgba(26, 26, 26, 0.9), rgba(42, 42, 42, 0.9));
    padding: 2rem;
    border-radius: 8px;
    border: 1px solid rgba(255, 191, 0, 0.3);
  }

  .info-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 1rem;
  }

  .info-item {
    background: rgba(255, 255, 255, 0.05);
    padding: 1.5rem;
    border-radius: 6px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    text-align: center;
  }

  .info-label {
    font-size: 0.9rem;
    color: #f0f0f0;
    opacity: 0.8;
    margin-bottom: 0.5rem;
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  .info-value {
    font-size: 1.1rem;
    font-weight: 600;
    color: #00ff41;
  }

  /* Navigation Footer */
  .navigation-footer {
    text-align: center;
    margin-top: 3rem;
  }

  .back-button {
    display: inline-flex;
    align-items: center;
    gap: 1rem;
    background: linear-gradient(145deg, #ffbf00, #00ff41);
    color: #0a0a0a;
    padding: 1rem 2rem;
    border-radius: 6px;
    text-decoration: none;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    transition: all 0.3s ease;
  }

  .back-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 25px rgba(255, 191, 0, 0.3);
  }

  /* Responsive Design */
  @media (max-width: 768px) {
    .help-header {
      flex-direction: column;
      gap: 1rem;
      text-align: center;
    }

    .page-title {
      font-size: 2rem;
    }

    .category-filters {
      justify-content: center;
    }

    .actions-grid, .info-grid {
      grid-template-columns: 1fr;
    }

    .topic-info {
      flex-direction: column;
      gap: 0.5rem;
    }
  }
</style>