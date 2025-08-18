<script>
  import { onMount } from 'svelte';
  import { goto } from '$app/navigation';
  import { page } from '$app/stores';

  // State management
  let isLoading = $state(true);
  let personsOfInterest = $state([]);
  let selectedPerson = $state(null);
  let searchQuery = $state('');
  let filterStatus = $state('all');

  // Mock persons data
  let mockPersons = [
    {
      id: 'poi-001',
      name: 'John Smith',
      alias: ['Johnny', 'J.S.'],
      status: 'active',
      riskLevel: 'medium',
      lastSeen: '2024-08-18T08:30:00Z',
      location: 'Downtown Legal District',
      occupation: 'Business Attorney',
      connections: ['poi-002', 'poi-004'],
      cases: ['case-001', 'case-003'],
      tags: ['legal', 'attorney', 'corporate'],
      description: 'Senior partner at Smith & Associates. Frequent involvement in corporate litigation cases.',
      evidence: ['doc-001', 'email-thread-001'],
      timeline: [
        { date: '2024-08-18', event: 'Meeting with client', location: 'Smith & Associates Office' },
        { date: '2024-08-17', event: 'Court appearance', location: 'City Courthouse' },
        { date: '2024-08-15', event: 'Contract signing', location: 'Corporate Plaza' }
      ]
    },
    {
      id: 'poi-002',
      name: 'Sarah Johnson',
      alias: ['S.J.', 'Sally'],
      status: 'monitoring',
      riskLevel: 'low',
      lastSeen: '2024-08-17T14:15:00Z',
      location: 'Financial District',
      occupation: 'Financial Analyst',
      connections: ['poi-001', 'poi-003'],
      cases: ['case-002'],
      tags: ['finance', 'analyst', 'compliance'],
      description: 'Senior financial analyst specializing in regulatory compliance and audit processes.',
      evidence: ['report-001', 'email-thread-002'],
      timeline: [
        { date: '2024-08-17', event: 'Compliance meeting', location: 'Financial Tower' },
        { date: '2024-08-16', event: 'Document review', location: 'Home office' },
        { date: '2024-08-14', event: 'Client consultation', location: 'Conference Center' }
      ]
    },
    {
      id: 'poi-003',
      name: 'Michael Rodriguez',
      alias: ['Mike', 'M.R.'],
      status: 'investigation',
      riskLevel: 'high',
      lastSeen: '2024-08-18T16:45:00Z',
      location: 'Industrial Zone',
      occupation: 'Import/Export Manager',
      connections: ['poi-002', 'poi-005'],
      cases: ['case-001', 'case-004'],
      tags: ['import', 'export', 'logistics'],
      description: 'Manager overseeing international trade operations. Currently under investigation for customs violations.',
      evidence: ['customs-doc-001', 'shipping-manifest-001'],
      timeline: [
        { date: '2024-08-18', event: 'Warehouse inspection', location: 'Port Authority' },
        { date: '2024-08-16', event: 'Customs meeting', location: 'Federal Building' },
        { date: '2024-08-13', event: 'Shipping coordination', location: 'Logistics Center' }
      ]
    }
  ];

  // Reactive computations
  let filteredPersons = $derived(() => {
    let filtered = personsOfInterest;
    
    if (filterStatus !== 'all') {
      filtered = filtered.filter(person => person.status === filterStatus);
    }
    
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(person => 
        person.name.toLowerCase().includes(query) ||
        person.occupation.toLowerCase().includes(query) ||
        person.tags.some(tag => tag.toLowerCase().includes(query)) ||
        person.location.toLowerCase().includes(query)
      );
    }
    
    return filtered;
  });

  // Navigation and interaction functions
  function navigateHome() {
    goto('/');
  }

  function selectPerson(person) {
    selectedPerson = person;
  }

  function getStatusColor(status) {
    switch (status) {
      case 'active': return '#00ff41';
      case 'monitoring': return '#4ecdc4';
      case 'investigation': return '#ff6b6b';
      case 'cleared': return '#888';
      default: return '#ffbf00';
    }
  }

  function getStatusIcon(status) {
    switch (status) {
      case 'active': return '✅';
      case 'monitoring': return '👁️';
      case 'investigation': return '🔍';
      case 'cleared': return '✓';
      default: return '❓';
    }
  }

  function getRiskColor(level) {
    switch (level) {
      case 'low': return '#00ff41';
      case 'medium': return '#ffbf00';
      case 'high': return '#ff6b6b';
      case 'critical': return '#ff3333';
      default: return '#888';
    }
  }

  function getRiskIcon(level) {
    switch (level) {
      case 'low': return '🟢';
      case 'medium': return '🟡';
      case 'high': return '🟠';
      case 'critical': return '🔴';
      default: return '⚪';
    }
  }

  // Initialize component
  onMount(() => {
    personsOfInterest = mockPersons;
    setTimeout(() => {
      isLoading = false;
    }, 800);
  });
</script>

<svelte:head>
  <title>Persons of Interest - YoRHa Legal AI</title>
  <meta name="description" content="Person tracking and relationship mapping system">
</svelte:head>

<!-- Loading Screen -->
{#if isLoading}
  <div class="loading-screen">
    <div class="loading-content">
      <div class="loading-icon">👤</div>
      <div class="loading-text">INITIALIZING PERSON TRACKING SYSTEM...</div>
      <div class="loading-bar">
        <div class="loading-progress"></div>
      </div>
    </div>
  </div>
{:else}
  <!-- Main Interface -->
  <div class="detective-interface">
    
    <!-- Header -->
    <header class="detective-header">
      <div class="header-left">
        <button class="back-button" onclick={navigateHome}>
          ← COMMAND CENTER
        </button>
        <div class="header-title">
          <h1>👤 PERSONS OF INTEREST</h1>
          <div class="header-subtitle">Person Tracking and Relationship Mapping</div>
        </div>
      </div>
      
      <div class="header-stats">
        <div class="stat-item">
          <div class="stat-value">{personsOfInterest.length}</div>
          <div class="stat-label">TOTAL PERSONS</div>
        </div>
        <div class="stat-item">
          <div class="stat-value">{personsOfInterest.filter(p => p.status === 'active').length}</div>
          <div class="stat-label">ACTIVE</div>
        </div>
        <div class="stat-item">
          <div class="stat-value">{personsOfInterest.filter(p => p.status === 'investigation').length}</div>
          <div class="stat-label">UNDER INVESTIGATION</div>
        </div>
      </div>
    </header>

    <!-- Search and Filter Controls -->
    <section class="control-panel">
      <div class="search-section">
        <div class="search-container">
          <input
            bind:value={searchQuery}
            placeholder="SEARCH PERSONS..."
            class="search-input"
          />
          <button class="search-button">🔍</button>
        </div>
      </div>

      <div class="filter-controls">
        <label class="filter-label">STATUS FILTER:</label>
        <select bind:value={filterStatus} class="filter-select">
          <option value="all">ALL STATUSES</option>
          <option value="active">ACTIVE</option>
          <option value="monitoring">MONITORING</option>
          <option value="investigation">INVESTIGATION</option>
          <option value="cleared">CLEARED</option>
        </select>
      </div>
    </section>

    <!-- Main Content Grid -->
    <div class="detective-content">
      
      <!-- Persons List -->
      <section class="persons-list">
        <h2 class="section-title">📋 PERSON DATABASE</h2>
        
        <div class="persons-grid">
          {#each filteredPersons as person (person.id)}
            <div 
              class="person-card {selectedPerson?.id === person.id ? 'selected' : ''}"
              onclick={() => selectPerson(person)}
            >
              <div class="person-header">
                <div class="person-avatar">
                  {person.name.charAt(0)}
                </div>
                <div class="person-status">
                  <div class="status-badge" style="color: {getStatusColor(person.status)}">
                    {getStatusIcon(person.status)} {person.status.toUpperCase()}
                  </div>
                  <div class="risk-badge" style="color: {getRiskColor(person.riskLevel)}">
                    {getRiskIcon(person.riskLevel)} {person.riskLevel.toUpperCase()} RISK
                  </div>
                </div>
              </div>

              <div class="person-info">
                <h3 class="person-name">{person.name}</h3>
                <div class="person-occupation">{person.occupation}</div>
                <div class="person-location">📍 {person.location}</div>
                <div class="person-last-seen">
                  Last seen: {new Date(person.lastSeen).toLocaleString()}
                </div>
              </div>

              <div class="person-tags">
                {#each person.tags.slice(0, 3) as tag}
                  <span class="tag">{tag}</span>
                {/each}
                {#if person.tags.length > 3}
                  <span class="tag-more">+{person.tags.length - 3}</span>
                {/if}
              </div>

              <div class="person-connections">
                <div class="connections-count">
                  🔗 {person.connections.length} connections
                </div>
                <div class="cases-count">
                  📁 {person.cases.length} active cases
                </div>
              </div>
            </div>
          {/each}
        </div>

        {#if filteredPersons.length === 0}
          <div class="empty-state">
            <div class="empty-icon">🔍</div>
            <div class="empty-text">NO PERSONS MATCH YOUR SEARCH</div>
            <button class="clear-filters-btn" onclick={() => { searchQuery = ''; filterStatus = 'all'; }}>
              CLEAR FILTERS
            </button>
          </div>
        {/if}
      </section>

      <!-- Person Details Panel -->
      <section class="person-details">
        <h2 class="section-title">🔍 DETAILED ANALYSIS</h2>
        
        {#if selectedPerson}
          <div class="details-content">
            <!-- Basic Information -->
            <div class="details-section">
              <h3>📄 BASIC INFORMATION</h3>
              <div class="info-grid">
                <div class="info-item">
                  <span class="info-label">Full Name:</span>
                  <span class="info-value">{selectedPerson.name}</span>
                </div>
                <div class="info-item">
                  <span class="info-label">Aliases:</span>
                  <span class="info-value">{selectedPerson.alias.join(', ')}</span>
                </div>
                <div class="info-item">
                  <span class="info-label">Occupation:</span>
                  <span class="info-value">{selectedPerson.occupation}</span>
                </div>
                <div class="info-item">
                  <span class="info-label">Status:</span>
                  <span class="info-value" style="color: {getStatusColor(selectedPerson.status)}">
                    {selectedPerson.status.toUpperCase()}
                  </span>
                </div>
                <div class="info-item">
                  <span class="info-label">Risk Level:</span>
                  <span class="info-value" style="color: {getRiskColor(selectedPerson.riskLevel)}">
                    {selectedPerson.riskLevel.toUpperCase()}
                  </span>
                </div>
                <div class="info-item">
                  <span class="info-label">Last Known Location:</span>
                  <span class="info-value">{selectedPerson.location}</span>
                </div>
              </div>
              <div class="description">
                <h4>Profile Description:</h4>
                <p>{selectedPerson.description}</p>
              </div>
            </div>

            <!-- Timeline -->
            <div class="details-section">
              <h3>📅 RECENT ACTIVITY</h3>
              <div class="timeline">
                {#each selectedPerson.timeline as event}
                  <div class="timeline-event">
                    <div class="event-date">{event.date}</div>
                    <div class="event-details">
                      <div class="event-title">{event.event}</div>
                      <div class="event-location">📍 {event.location}</div>
                    </div>
                  </div>
                {/each}
              </div>
            </div>

            <!-- Connections -->
            <div class="details-section">
              <h3>🔗 CONNECTIONS</h3>
              <div class="connections-list">
                {#each selectedPerson.connections as connectionId}
                  <div class="connection-item">
                    <div class="connection-avatar">
                      {personsOfInterest.find(p => p.id === connectionId)?.name.charAt(0) || '?'}
                    </div>
                    <div class="connection-name">
                      {personsOfInterest.find(p => p.id === connectionId)?.name || 'Unknown'}
                    </div>
                    <div class="connection-relationship">Associate</div>
                  </div>
                {/each}
              </div>
            </div>

            <!-- Evidence Links -->
            <div class="details-section">
              <h3>📎 LINKED EVIDENCE</h3>
              <div class="evidence-links">
                {#each selectedPerson.evidence as evidenceId}
                  <div class="evidence-link">
                    <div class="evidence-icon">📄</div>
                    <div class="evidence-name">{evidenceId}</div>
                    <button class="view-evidence-btn">VIEW</button>
                  </div>
                {/each}
              </div>
            </div>

            <!-- Tags -->
            <div class="details-section">
              <h3>🏷️ CLASSIFICATION TAGS</h3>
              <div class="tags-list">
                {#each selectedPerson.tags as tag}
                  <span class="detail-tag">{tag}</span>
                {/each}
              </div>
            </div>
          </div>
        {:else}
          <div class="no-selection">
            <div class="no-selection-icon">👤</div>
            <div class="no-selection-text">SELECT A PERSON TO VIEW DETAILS</div>
          </div>
        {/if}
      </section>
    </div>

    <!-- Footer Actions -->
    <footer class="detective-footer">
      <div class="footer-actions">
        <button class="footer-btn primary">ADD NEW PERSON</button>
        <button class="footer-btn secondary">EXPORT DATABASE</button>
        <button class="footer-btn secondary">GENERATE REPORT</button>
        <button class="footer-btn secondary">RELATIONSHIP MAP</button>
      </div>
      
      <div class="footer-info">
        <div class="system-status">
          <span class="status-label">TRACKING SYSTEM:</span>
          <span class="status-value active">OPERATIONAL</span>
        </div>
        <div class="timestamp">LAST UPDATE: {new Date().toLocaleString()}</div>
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
  .detective-interface {
    min-height: 100vh;
    background: var(--yorha-bg);
    color: var(--yorha-light);
    font-family: 'JetBrains Mono', monospace;
    animation: fadeIn 0.5s ease-out;
  }

  /* === HEADER === */
  .detective-header {
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
    font-size: 2rem;
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

  /* === CONTROL PANEL === */
  .control-panel {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1.5rem 2rem;
    border-bottom: 1px solid rgba(255, 191, 0, 0.3);
    background: rgba(26, 26, 26, 0.5);
  }

  .search-section {
    flex: 1;
    max-width: 500px;
  }

  .search-container {
    display: flex;
    position: relative;
  }

  .search-input {
    flex: 1;
    background: rgba(26, 26, 26, 0.8);
    border: 2px solid rgba(255, 191, 0, 0.5);
    color: var(--yorha-light);
    padding: 1rem 1.5rem;
    font-family: inherit;
    font-size: 1rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    transition: all 0.3s ease;
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

  .search-button {
    background: var(--yorha-accent-warm);
    border: none;
    color: var(--yorha-dark);
    padding: 1rem 1.5rem;
    font-size: 1.2rem;
    cursor: pointer;
    transition: all 0.3s ease;
  }

  .search-button:hover {
    background: var(--yorha-success);
    transform: scale(1.05);
  }

  .filter-controls {
    display: flex;
    align-items: center;
    gap: 1rem;
  }

  .filter-label {
    font-size: 0.9rem;
    color: var(--yorha-muted);
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  .filter-select {
    background: rgba(26, 26, 26, 0.8);
    border: 2px solid rgba(78, 205, 196, 0.5);
    color: var(--yorha-light);
    padding: 0.8rem 1rem;
    font-family: inherit;
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    cursor: pointer;
  }

  .filter-select:focus {
    outline: none;
    border-color: var(--yorha-accent-cool);
  }

  /* === MAIN CONTENT === */
  .detective-content {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 2rem;
    padding: 2rem;
    min-height: calc(100vh - 280px);
  }

  .section-title {
    font-size: 1.3rem;
    font-weight: 600;
    color: var(--yorha-accent-warm);
    margin: 0 0 1.5rem 0;
    text-transform: uppercase;
    letter-spacing: 1px;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(255, 191, 0, 0.3);
  }

  /* === PERSONS LIST === */
  .persons-list {
    background: rgba(26, 26, 26, 0.8);
    border: 1px solid rgba(255, 191, 0, 0.3);
    border-radius: 8px;
    padding: 1.5rem;
    overflow-y: auto;
    max-height: calc(100vh - 350px);
  }

  .persons-grid {
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  .person-card {
    background: rgba(42, 42, 42, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 8px;
    padding: 1.5rem;
    cursor: pointer;
    transition: all 0.3s ease;
  }

  .person-card:hover {
    border-color: var(--yorha-accent-warm);
    background: rgba(42, 42, 42, 0.8);
    transform: translateY(-2px);
  }

  .person-card.selected {
    border-color: var(--yorha-success);
    background: rgba(0, 255, 65, 0.1);
  }

  .person-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
  }

  .person-avatar {
    width: 50px;
    height: 50px;
    background: var(--yorha-accent-warm);
    color: var(--yorha-dark);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
    font-weight: 700;
  }

  .person-status {
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
    align-items: flex-end;
  }

  .status-badge,
  .risk-badge {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 1px;
    padding: 0.2rem 0.5rem;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 4px;
    border: 1px solid currentColor;
    text-transform: uppercase;
  }

  .person-info {
    margin-bottom: 1rem;
  }

  .person-name {
    font-size: 1.2rem;
    font-weight: 600;
    color: var(--yorha-light);
    margin: 0 0 0.5rem 0;
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  .person-occupation {
    color: var(--yorha-accent-cool);
    font-size: 0.9rem;
    margin-bottom: 0.3rem;
  }

  .person-location,
  .person-last-seen {
    color: var(--yorha-muted);
    font-size: 0.8rem;
    margin-bottom: 0.2rem;
  }

  .person-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-bottom: 1rem;
  }

  .tag {
    background: rgba(78, 205, 196, 0.2);
    color: var(--yorha-accent-cool);
    padding: 0.3rem 0.6rem;
    border-radius: 3px;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    border: 1px solid var(--yorha-accent-cool);
  }

  .tag-more {
    background: rgba(255, 255, 255, 0.1);
    color: var(--yorha-muted);
    padding: 0.3rem 0.6rem;
    border-radius: 3px;
    font-size: 0.75rem;
    border: 1px solid var(--yorha-muted);
  }

  .person-connections {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.8rem;
    color: var(--yorha-muted);
  }

  .connections-count,
  .cases-count {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  /* === PERSON DETAILS === */
  .person-details {
    background: rgba(26, 26, 26, 0.8);
    border: 1px solid rgba(255, 191, 0, 0.3);
    border-radius: 8px;
    padding: 1.5rem;
    overflow-y: auto;
    max-height: calc(100vh - 350px);
  }

  .details-content {
    display: flex;
    flex-direction: column;
    gap: 2rem;
  }

  .details-section h3 {
    font-size: 1.1rem;
    color: var(--yorha-accent-warm);
    margin: 0 0 1rem 0;
    text-transform: uppercase;
    letter-spacing: 1px;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(255, 191, 0, 0.2);
  }

  .info-grid {
    display: grid;
    gap: 0.8rem;
    margin-bottom: 1rem;
  }

  .info-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.5rem 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  }

  .info-label {
    color: var(--yorha-muted);
    font-size: 0.9rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  .info-value {
    color: var(--yorha-light);
    font-size: 0.9rem;
    text-align: right;
  }

  .description h4 {
    color: var(--yorha-accent-cool);
    font-size: 0.9rem;
    margin: 0 0 0.5rem 0;
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  .description p {
    color: var(--yorha-light);
    line-height: 1.5;
    margin: 0;
    padding: 1rem;
    background: rgba(42, 42, 42, 0.4);
    border-radius: 6px;
    border-left: 3px solid var(--yorha-accent-cool);
  }

  .timeline {
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  .timeline-event {
    display: flex;
    gap: 1rem;
    align-items: center;
    padding: 1rem;
    background: rgba(42, 42, 42, 0.4);
    border-radius: 6px;
    border-left: 3px solid var(--yorha-success);
  }

  .event-date {
    font-size: 0.8rem;
    color: var(--yorha-accent-warm);
    font-weight: 600;
    min-width: 80px;
  }

  .event-details {
    flex: 1;
  }

  .event-title {
    color: var(--yorha-light);
    font-weight: 600;
    margin-bottom: 0.3rem;
  }

  .event-location {
    color: var(--yorha-muted);
    font-size: 0.8rem;
  }

  .connections-list,
  .evidence-links {
    display: flex;
    flex-direction: column;
    gap: 0.8rem;
  }

  .connection-item {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 0.8rem;
    background: rgba(42, 42, 42, 0.4);
    border-radius: 6px;
    border: 1px solid rgba(78, 205, 196, 0.3);
  }

  .connection-avatar {
    width: 35px;
    height: 35px;
    background: var(--yorha-accent-cool);
    color: var(--yorha-dark);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
  }

  .connection-name {
    color: var(--yorha-light);
    font-weight: 600;
    flex: 1;
  }

  .connection-relationship {
    color: var(--yorha-muted);
    font-size: 0.8rem;
  }

  .evidence-link {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 0.8rem;
    background: rgba(42, 42, 42, 0.4);
    border-radius: 6px;
    border: 1px solid rgba(255, 191, 0, 0.3);
  }

  .evidence-icon {
    font-size: 1.5rem;
    color: var(--yorha-accent-warm);
  }

  .evidence-name {
    color: var(--yorha-light);
    flex: 1;
  }

  .view-evidence-btn {
    background: transparent;
    border: 1px solid var(--yorha-accent-warm);
    color: var(--yorha-accent-warm);
    padding: 0.4rem 0.8rem;
    font-family: inherit;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    cursor: pointer;
    transition: all 0.3s ease;
    border-radius: 3px;
  }

  .view-evidence-btn:hover {
    background: var(--yorha-accent-warm);
    color: var(--yorha-dark);
  }

  .tags-list {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
  }

  .detail-tag {
    background: rgba(78, 205, 196, 0.2);
    color: var(--yorha-accent-cool);
    padding: 0.4rem 0.8rem;
    border-radius: 4px;
    font-size: 0.8rem;
    border: 1px solid var(--yorha-accent-cool);
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .no-selection {
    text-align: center;
    padding: 3rem 1rem;
    color: var(--yorha-muted);
  }

  .no-selection-icon {
    font-size: 4rem;
    margin-bottom: 1rem;
    opacity: 0.5;
  }

  .no-selection-text {
    font-size: 1.1rem;
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  .empty-state {
    text-align: center;
    padding: 3rem 1rem;
    color: var(--yorha-muted);
  }

  .empty-icon {
    font-size: 4rem;
    margin-bottom: 1rem;
    opacity: 0.5;
  }

  .empty-text {
    font-size: 1.1rem;
    margin-bottom: 2rem;
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  .clear-filters-btn {
    background: var(--yorha-accent-warm);
    color: var(--yorha-dark);
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

  .clear-filters-btn:hover {
    background: var(--yorha-success);
    transform: scale(1.05);
  }

  /* === FOOTER === */
  .detective-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 2rem;
    border-top: 2px solid var(--yorha-accent-warm);
    background: rgba(26, 26, 26, 0.9);
  }

  .footer-actions {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
  }

  .footer-btn {
    padding: 1rem 1.5rem;
    font-family: inherit;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    cursor: pointer;
    transition: all 0.3s ease;
    border-radius: 4px;
    font-size: 0.9rem;
  }

  .footer-btn.primary {
    background: var(--yorha-success);
    color: var(--yorha-dark);
    border: none;
  }

  .footer-btn.primary:hover {
    background: var(--yorha-accent-warm);
    transform: scale(1.05);
  }

  .footer-btn.secondary {
    background: transparent;
    color: var(--yorha-accent-cool);
    border: 2px solid var(--yorha-accent-cool);
  }

  .footer-btn.secondary:hover {
    background: var(--yorha-accent-cool);
    color: var(--yorha-dark);
  }

  .footer-info {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 0.5rem;
    font-size: 0.8rem;
  }

  .system-status {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  .status-label {
    color: var(--yorha-muted);
  }

  .status-value.active {
    color: var(--yorha-success);
    font-weight: 600;
  }

  .timestamp {
    color: var(--yorha-muted);
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

  /* === RESPONSIVE DESIGN === */
  @media (max-width: 1200px) {
    .detective-content {
      grid-template-columns: 1fr;
    }
  }

  @media (max-width: 768px) {
    .detective-header {
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

    .control-panel {
      flex-direction: column;
      gap: 1rem;
    }

    .footer-actions {
      flex-direction: column;
      gap: 0.5rem;
    }
  }
</style>