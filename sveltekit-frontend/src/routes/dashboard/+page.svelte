<script lang="ts">
  import { onMount } from 'svelte';
  import { user } from '$lib/stores/user';
  import { goto } from '$app/navigation';

  // Mock data for demonstration
  let stats = {
    totalCases: 24,
    activeCases: 12,
    evidenceFiles: 156,
    recentActivity: 8
  };

  let recentCases = [
    { id: 1, title: 'Fraud Investigation #2024-001', status: 'active', lastUpdate: '2 hours ago' },
    { id: 2, title: 'Identity Theft Case #2024-002', status: 'pending', lastUpdate: '1 day ago' },
    { id: 3, title: 'Corporate Embezzlement #2024-003', status: 'closed', lastUpdate: '3 days ago' }
  ];

  onMount(() => {
    // Check if user is logged in
    if (!$user) {
      goto('/login');
    }
  });
</script>

<svelte:head>
  <title>Dashboard - Legal Case Management</title>
</svelte:head>

<div class="dashboard-container">
  {#if $user}
    <div class="dashboard-header">
      <h1>Welcome back, {$user.name || $user.email}</h1>
      <p>Here's what's happening with your cases today.</p>
    </div>

    <!-- Stats Grid -->
    <div class="stats-grid">
      <div class="stat-card">
        <div class="stat-icon">📁</div>
        <div class="stat-content">
          <h3>{stats.totalCases}</h3>
          <p>Total Cases</p>
        </div>
      </div>
      
      <div class="stat-card active">
        <div class="stat-icon">🔄</div>
        <div class="stat-content">
          <h3>{stats.activeCases}</h3>
          <p>Active Cases</p>
        </div>
      </div>
      
      <div class="stat-card">
        <div class="stat-icon">🔍</div>
        <div class="stat-content">
          <h3>{stats.evidenceFiles}</h3>
          <p>Evidence Files</p>
        </div>
      </div>
      
      <div class="stat-card">
        <div class="stat-icon">📊</div>
        <div class="stat-content">
          <h3>{stats.recentActivity}</h3>
          <p>Recent Updates</p>
        </div>
      </div>
    </div>

    <!-- Quick Actions -->
    <div class="quick-actions">
      <h2>Quick Actions</h2>
      <div class="action-grid">
        <a href="/cases/new" class="action-card">
          <div class="action-icon">➕</div>
          <h3>New Case</h3>
          <p>Create a new legal case</p>
        </a>
        
        <a href="/evidence" class="action-card">
          <div class="action-icon">📤</div>
          <h3>Upload Evidence</h3>
          <p>Add evidence to existing cases</p>
        </a>
        
        <a href="/search" class="action-card">
          <div class="action-icon">🔍</div>
          <h3>Search Cases</h3>
          <p>Find cases and evidence</p>
        </a>
        
        <a href="/reports" class="action-card">
          <div class="action-icon">📈</div>
          <h3>Generate Report</h3>
          <p>Create case reports</p>
        </a>
      </div>
    </div>

    <!-- Recent Cases -->
    <div class="recent-section">
      <h2>Recent Cases</h2>
      <div class="cases-list">
        {#each recentCases as case}
          <div class="case-item">
            <div class="case-info">
              <h3>{case.title}</h3>
              <p>Last updated: {case.lastUpdate}</p>
            </div>
            <div class="case-status status-{case.status}">
              {case.status}
            </div>
          </div>
        {/each}
      </div>
      
      <a href="/cases" class="view-all-btn">View All Cases →</a>
    </div>
  {:else}
    <div class="loading">
      <div class="spinner"></div>
      <p>Loading dashboard...</p>
    </div>
  {/if}
</div>

<style>
  .dashboard-container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem;
  }

  .dashboard-header {
    margin-bottom: 2rem;
  }

  .dashboard-header h1 {
    font-size: 2rem;
    font-weight: 700;
    color: #1f2937;
    margin-bottom: 0.5rem;
  }

  .dashboard-header p {
    color: #6b7280;
    font-size: 1rem;
  }

  .stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1.5rem;
    margin-bottom: 3rem;
  }

  .stat-card {
    background: white;
    padding: 1.5rem;
    border-radius: 0.75rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    display: flex;
    align-items: center;
    gap: 1rem;
    transition: transform 0.2s ease;
  }

  .stat-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  }

  .stat-card.active {
    background: linear-gradient(135deg, #3b82f6, #1d4ed8);
    color: white;
  }

  .stat-icon {
    font-size: 2rem;
    opacity: 0.8;
  }

  .stat-content h3 {
    font-size: 1.75rem;
    font-weight: 700;
    margin-bottom: 0.25rem;
  }

  .stat-content p {
    font-size: 0.875rem;
    opacity: 0.8;
  }

  .quick-actions {
    margin-bottom: 3rem;
  }

  .quick-actions h2 {
    font-size: 1.5rem;
    font-weight: 600;
    color: #1f2937;
    margin-bottom: 1rem;
  }

  .action-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
  }

  .action-card {
    background: white;
    padding: 1.5rem;
    border-radius: 0.75rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    text-decoration: none;
    color: inherit;
    transition: all 0.2s ease;
    text-align: center;
  }

  .action-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  }

  .action-icon {
    font-size: 2.5rem;
    margin-bottom: 1rem;
  }

  .action-card h3 {
    font-size: 1.125rem;
    font-weight: 600;
    color: #1f2937;
    margin-bottom: 0.5rem;
  }

  .action-card p {
    color: #6b7280;
    font-size: 0.875rem;
  }

  .recent-section h2 {
    font-size: 1.5rem;
    font-weight: 600;
    color: #1f2937;
    margin-bottom: 1rem;
  }

  .cases-list {
    background: white;
    border-radius: 0.75rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    overflow: hidden;
    margin-bottom: 1rem;
  }

  .case-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem 1.5rem;
    border-bottom: 1px solid #f3f4f6;
  }

  .case-item:last-child {
    border-bottom: none;
  }

  .case-info h3 {
    font-size: 1rem;
    font-weight: 600;
    color: #1f2937;
    margin-bottom: 0.25rem;
  }

  .case-info p {
    font-size: 0.875rem;
    color: #6b7280;
  }

  .case-status {
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
  }

  .status-active {
    background: #dcfce7;
    color: #166534;
  }

  .status-pending {
    background: #fef3c7;
    color: #92400e;
  }

  .status-closed {
    background: #f3f4f6;
    color: #374151;
  }

  .view-all-btn {
    display: inline-flex;
    align-items: center;
    color: #3b82f6;
    font-weight: 500;
    text-decoration: none;
    transition: color 0.2s ease;
  }

  .view-all-btn:hover {
    color: #1d4ed8;
  }

  .loading {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 50vh;
    text-align: center;
  }

  .spinner {
    width: 40px;
    height: 40px;
    border: 4px solid #f3f3f3;
    border-top: 4px solid #3b82f6;
    border-radius: 50%;
    animation: spin 1s linear infinite;
    margin-bottom: 1rem;
  }

  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }

  @media (max-width: 768px) {
    .dashboard-container {
      padding: 1rem;
    }
    
    .stats-grid {
      grid-template-columns: 1fr;
    }
    
    .action-grid {
      grid-template-columns: repeat(2, 1fr);
    }
  }

  @media (max-width: 480px) {
    .action-grid {
      grid-template-columns: 1fr;
    }
    
    .case-item {
      flex-direction: column;
      align-items: flex-start;
      gap: 0.5rem;
    }
  }
</style>