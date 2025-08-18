<script lang="ts">
  import { Button } from '$lib/components/ui/button/index.js';
  import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '$lib/components/ui/card/index.js';
  import { Badge } from '$lib/components/ui/badge/index.js';
  import { goto } from '$app/navigation';
  import { page } from '$app/stores';
  import type { PageData } from './$types';

  interface Props {
    data: PageData;
  }

  let { data }: Props = $props();

  // Check for creation success message
  let showCreationSuccess = $state(false);
  $effect(() => {
    if ($page.url.searchParams.get('created') === 'true') {
      showCreationSuccess = true;
    }
  });

  function goBack() {
    goto('/yorha-demo');
  }

  function createCase() {
    goto('/cases/create');
  }

  function viewCase(caseId: string) {
    goto(`/cases/${caseId}`);
  }

  function editCase(caseId: string) {
    goto(`/cases/${caseId}/edit`);
  }

  function getPriorityColor(priority: string) {
    const colors = {
      low: '#68d391',
      medium: '#ed8936', 
      high: '#f56565',
      urgent: '#9f7aea'
    };
    return colors[priority] || '#a0aec0';
  }

  function getStatusColor(status: string) {
    const colors = {
      active: '#68d391',
      pending: '#ed8936',
      closed: '#a0aec0',
      archived: '#6b7280'
    };
    return colors[status] || '#a0aec0';
  }

  function formatDate(dateString: string) {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    });
  }
</script>

<svelte:head>
  <title>Case Dashboard - Legal AI System</title>
  <meta name="description" content="Manage legal cases in the AI-powered case management system" />
</svelte:head>

<div class="cases-container">
  <div class="cases-background"></div>
  
  <div class="cases-content">
    <!-- Header -->
    <div class="page-header">
      <Button variant="ghost" class="back-button" onclick={goBack}>
        ← Back to Demo Navigation
      </Button>
      
      <div class="header-content">
        <div class="header-left">
          <h1 class="page-title">Legal Case Dashboard</h1>
          <p class="page-subtitle">Manage and track legal cases with AI-powered insights</p>
        </div>
        <div class="header-right">
          <Button class="create-case-button" onclick={createCase}>
            <span class="button-icon">📁</span>
            Create New Case
          </Button>
        </div>
      </div>
    </div>

    <!-- Creation Success Alert -->
    {#if showCreationSuccess}
      <div class="alert alert-success">
        ✅ Case created successfully! It has been added to your case dashboard.
      </div>
    {/if}

    <!-- Statistics Cards -->
    <div class="stats-grid">
      <Card class="stat-card">
        <CardContent class="stat-content">
          <div class="stat-icon">📊</div>
          <div class="stat-info">
            <h3 class="stat-number">{data.stats.total}</h3>
            <p class="stat-label">Total Cases</p>
          </div>
        </CardContent>
      </Card>

      <Card class="stat-card">
        <CardContent class="stat-content">
          <div class="stat-icon active">🟢</div>
          <div class="stat-info">
            <h3 class="stat-number">{data.stats.active}</h3>
            <p class="stat-label">Active Cases</p>
          </div>
        </CardContent>
      </Card>

      <Card class="stat-card">
        <CardContent class="stat-content">
          <div class="stat-icon pending">🟡</div>
          <div class="stat-info">
            <h3 class="stat-number">{data.stats.pending}</h3>
            <p class="stat-label">Pending Cases</p>
          </div>
        </CardContent>
      </Card>

      <Card class="stat-card">
        <CardContent class="stat-content">
          <div class="stat-icon urgent">🔴</div>
          <div class="stat-info">
            <h3 class="stat-number">{data.stats.urgent}</h3>
            <p class="stat-label">Urgent Cases</p>
          </div>
        </CardContent>
      </Card>
    </div>

    <!-- Cases Grid -->
    <div class="cases-section">
      <div class="section-header">
        <h2 class="section-title">Recent Cases</h2>
        <div class="filter-controls">
          <select class="filter-select">
            <option value="all">All Cases</option>
            <option value="active">Active</option>
            <option value="pending">Pending</option>
            <option value="urgent">Urgent</option>
          </select>
        </div>
      </div>

      <div class="cases-grid">
        {#each data.cases as case}
          <Card class="case-card">
            <CardHeader class="case-header">
              <div class="case-title-row">
                <CardTitle class="case-title">{case.title}</CardTitle>
                <div class="case-badges">
                  <Badge 
                    class="priority-badge" 
                    style="background-color: {getPriorityColor(case.priority)}; color: #1a202c;"
                  >
                    {case.priority.toUpperCase()}
                  </Badge>
                  <Badge 
                    class="status-badge"
                    style="background-color: {getStatusColor(case.status)}; color: #1a202c;"
                  >
                    {case.status.toUpperCase()}
                  </Badge>
                </div>
              </div>
              <CardDescription class="case-description">
                Case #{case.caseNumber} • {case.caseType}
              </CardDescription>
            </CardHeader>

            <CardContent class="case-content">
              <div class="case-details">
                <div class="detail-item">
                  <span class="detail-icon">🏛️</span>
                  <span class="detail-text">{case.jurisdiction}</span>
                </div>
                {#if case.assignedDetective}
                  <div class="detail-item">
                    <span class="detail-icon">🕵️</span>
                    <span class="detail-text">{case.assignedDetective}</span>
                  </div>
                {/if}
                {#if case.assignedProsecutor}
                  <div class="detail-item">
                    <span class="detail-icon">⚖️</span>
                    <span class="detail-text">{case.assignedProsecutor}</span>
                  </div>
                {/if}
                <div class="detail-item">
                  <span class="detail-icon">📅</span>
                  <span class="detail-text">Created {formatDate(case.createdAt)}</span>
                </div>
              </div>

              <div class="case-actions">
                <Button 
                  variant="outline" 
                  size="sm" 
                  class="action-button"
                  onclick={() => viewCase(case.id)}
                >
                  View Details
                </Button>
                <Button 
                  variant="outline" 
                  size="sm" 
                  class="action-button"
                  onclick={() => editCase(case.id)}
                >
                  Edit Case
                </Button>
              </div>
            </CardContent>
          </Card>
        {/each}
      </div>

      {#if data.cases.length === 0}
        <div class="empty-state">
          <div class="empty-icon">📁</div>
          <h3 class="empty-title">No Cases Found</h3>
          <p class="empty-description">
            Get started by creating your first legal case using the AI-powered case management system.
          </p>
          <Button class="empty-action" onclick={createCase}>
            Create Your First Case
          </Button>
        </div>
      {/if}
    </div>

    <!-- AI Insights Section -->
    <Card class="insights-card">
      <CardHeader>
        <CardTitle class="insights-title">🤖 AI Insights</CardTitle>
        <CardDescription>AI-powered analytics and recommendations</CardDescription>
      </CardHeader>
      <CardContent>
        <div class="insights-grid">
          <div class="insight-item">
            <span class="insight-icon">📈</span>
            <div class="insight-content">
              <h4 class="insight-title">Case Resolution Time</h4>
              <p class="insight-value">Average: 8.3 months</p>
              <p class="insight-trend">↗️ 12% improvement this quarter</p>
            </div>
          </div>

          <div class="insight-item">
            <span class="insight-icon">🎯</span>
            <div class="insight-content">
              <h4 class="insight-title">Success Rate</h4>
              <p class="insight-value">87% conviction rate</p>
              <p class="insight-trend">↗️ Above department average</p>
            </div>
          </div>

          <div class="insight-item">
            <span class="insight-icon">⚠️</span>
            <div class="insight-content">
              <h4 class="insight-title">Cases Requiring Attention</h4>
              <p class="insight-value">3 cases overdue</p>
              <p class="insight-trend">→ Review recommended</p>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  </div>
</div>

<style>
  .cases-container {
    min-height: 100vh;
    position: relative;
    font-family: 'Rajdhani', 'Roboto Mono', monospace;
  }

  .cases-background {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 50%, #0f0f0f 100%);
    z-index: -2;
  }

  .cases-background::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: 
      radial-gradient(circle at 20% 80%, rgba(214, 158, 46, 0.03) 0%, transparent 50%),
      radial-gradient(circle at 80% 20%, rgba(68, 200, 245, 0.03) 0%, transparent 50%);
    z-index: -1;
  }

  .cases-content {
    position: relative;
    z-index: 10;
    padding: 2rem;
    max-width: 1400px;
    margin: 0 auto;
  }

  /* Header */
  .page-header {
    margin-bottom: 2rem;
  }

  .back-button {
    margin-bottom: 1rem;
    color: #a0aec0;
    border: 1px solid #4a5568;
    background: rgba(26, 32, 44, 0.8);
    backdrop-filter: blur(10px);
  }

  .back-button:hover {
    color: #d69e2e;
    border-color: #d69e2e;
  }

  .header-content {
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    gap: 2rem;
  }

  .header-left {
    flex: 1;
  }

  .page-title {
    color: #d69e2e;
    font-size: 2.5rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 0.5rem;
    text-shadow: 0 0 20px rgba(214, 158, 46, 0.3);
  }

  .page-subtitle {
    color: #a0aec0;
    font-size: 1.125rem;
    line-height: 1.6;
    margin: 0;
  }

  :global(.create-case-button) {
    background: linear-gradient(135deg, #d69e2e 0%, #ed8936 100%);
    border: none;
    color: #1a202c;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    padding: 1rem 2rem;
    border-radius: 0;
    transition: all 0.3s ease;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  :global(.create-case-button:hover) {
    transform: translateY(-2px);
    box-shadow: 0 10px 30px rgba(214, 158, 46, 0.3);
  }

  .button-icon {
    font-size: 1.125rem;
  }

  /* Alerts */
  .alert {
    padding: 1rem;
    border-radius: 0;
    margin-bottom: 2rem;
    font-weight: 500;
  }

  .alert-success {
    background: rgba(104, 211, 145, 0.1);
    border: 1px solid #68d391;
    color: #68d391;
  }

  /* Statistics */
  .stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 1.5rem;
    margin-bottom: 3rem;
  }

  :global(.stat-card) {
    background: rgba(45, 55, 72, 0.8);
    border: 1px solid rgba(214, 158, 46, 0.3);
    border-radius: 0;
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
  }

  :global(.stat-card:hover) {
    transform: translateY(-2px);
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
  }

  .stat-content {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 1.5rem;
  }

  .stat-icon {
    font-size: 2rem;
    flex-shrink: 0;
  }

  .stat-icon.active { color: #68d391; }
  .stat-icon.pending { color: #ed8936; }
  .stat-icon.urgent { color: #f56565; }

  .stat-info {
    flex: 1;
  }

  .stat-number {
    color: #d69e2e;
    font-size: 2rem;
    font-weight: 700;
    margin: 0 0 0.25rem 0;
  }

  .stat-label {
    color: #a0aec0;
    font-size: 0.875rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: 0;
  }

  /* Cases Section */
  .cases-section {
    margin-bottom: 3rem;
  }

  .section-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 2rem;
  }

  .section-title {
    color: #d69e2e;
    font-size: 1.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: 0;
  }

  .filter-controls {
    display: flex;
    gap: 1rem;
  }

  .filter-select {
    background: rgba(26, 32, 44, 0.8);
    border: 1px solid #4a5568;
    color: #e2e8f0;
    border-radius: 0;
    padding: 0.5rem 1rem;
    font-family: inherit;
  }

  .filter-select:focus {
    border-color: #d69e2e;
    outline: none;
  }

  .filter-select option {
    background: #1a202c;
    color: #e2e8f0;
  }

  /* Cases Grid */
  .cases-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
    gap: 1.5rem;
  }

  :global(.case-card) {
    background: rgba(45, 55, 72, 0.8);
    border: 1px solid rgba(74, 85, 104, 0.5);
    border-radius: 0;
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
  }

  :global(.case-card:hover) {
    transform: translateY(-4px);
    border-color: #d69e2e;
    box-shadow: 0 10px 30px rgba(214, 158, 46, 0.2);
  }

  .case-header {
    padding: 1.5rem 1.5rem 1rem;
  }

  .case-title-row {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 1rem;
    margin-bottom: 0.5rem;
  }

  :global(.case-title) {
    color: #e2e8f0;
    font-size: 1.25rem;
    font-weight: 600;
    flex: 1;
    margin: 0;
  }

  .case-badges {
    display: flex;
    gap: 0.5rem;
    flex-shrink: 0;
  }

  :global(.priority-badge),
  :global(.status-badge) {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    border-radius: 0;
    padding: 0.25rem 0.5rem;
  }

  :global(.case-description) {
    color: #a0aec0;
    font-size: 0.875rem;
    margin: 0;
  }

  .case-content {
    padding: 1rem 1.5rem 1.5rem;
  }

  .case-details {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    margin-bottom: 1.5rem;
  }

  .detail-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.875rem;
  }

  .detail-icon {
    font-size: 1rem;
    flex-shrink: 0;
  }

  .detail-text {
    color: #a0aec0;
  }

  .case-actions {
    display: flex;
    gap: 0.75rem;
  }

  :global(.action-button) {
    background: transparent;
    border: 1px solid #4a5568;
    color: #a0aec0;
    border-radius: 0;
    text-transform: uppercase;
    font-weight: 500;
    letter-spacing: 1px;
    font-size: 0.75rem;
  }

  :global(.action-button:hover) {
    border-color: #d69e2e;
    color: #d69e2e;
  }

  /* Empty State */
  .empty-state {
    text-align: center;
    padding: 4rem 2rem;
    background: rgba(26, 32, 44, 0.6);
    border: 1px solid rgba(74, 85, 104, 0.5);
    border-radius: 0;
    backdrop-filter: blur(10px);
  }

  .empty-icon {
    font-size: 4rem;
    margin-bottom: 1rem;
  }

  .empty-title {
    color: #e2e8f0;
    font-size: 1.5rem;
    font-weight: 600;
    margin-bottom: 1rem;
  }

  .empty-description {
    color: #a0aec0;
    font-size: 1rem;
    line-height: 1.6;
    margin-bottom: 2rem;
    max-width: 400px;
    margin-left: auto;
    margin-right: auto;
  }

  :global(.empty-action) {
    background: linear-gradient(135deg, #d69e2e 0%, #ed8936 100%);
    border: none;
    color: #1a202c;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    padding: 1rem 2rem;
    border-radius: 0;
  }

  /* AI Insights */
  :global(.insights-card) {
    background: rgba(26, 32, 44, 0.8);
    border: 1px solid rgba(104, 211, 145, 0.3);
    border-radius: 0;
    backdrop-filter: blur(10px);
    border-left: 4px solid #68d391;
  }

  :global(.insights-title) {
    color: #68d391;
    font-size: 1.5rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  .insights-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 2rem;
  }

  .insight-item {
    display: flex;
    align-items: center;
    gap: 1rem;
  }

  .insight-icon {
    font-size: 2rem;
    flex-shrink: 0;
  }

  .insight-content {
    flex: 1;
  }

  .insight-title {
    color: #e2e8f0;
    font-size: 1rem;
    font-weight: 600;
    margin: 0 0 0.25rem 0;
  }

  .insight-value {
    color: #68d391;
    font-size: 1.125rem;
    font-weight: 700;
    margin: 0 0 0.25rem 0;
  }

  .insight-trend {
    color: #a0aec0;
    font-size: 0.875rem;
    margin: 0;
  }

  /* Responsive Design */
  @media (max-width: 768px) {
    .cases-content {
      padding: 1rem;
    }

    .header-content {
      flex-direction: column;
      align-items: flex-start;
      gap: 1rem;
    }

    .page-title {
      font-size: 2rem;
    }

    .stats-grid {
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 1rem;
    }

    .cases-grid {
      grid-template-columns: 1fr;
    }

    .section-header {
      flex-direction: column;
      align-items: flex-start;
      gap: 1rem;
    }

    .insights-grid {
      grid-template-columns: 1fr;
      gap: 1.5rem;
    }
  }
</style>