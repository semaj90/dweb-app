<script lang="ts">
  import { onMount } from 'svelte';
  import { user } from '$lib/stores/user';
  import { goto } from '$app/navigation';

  let cases = [
    { 
      id: 'case-001', 
      title: 'Sample Legal Case', 
      description: 'This is a sample case to demonstrate the system capabilities.',
      status: 'active',
      created: '2024-01-15',
      priority: 'high'
    },
    { 
      id: 'case-002', 
      title: 'Fraud Investigation', 
      description: 'Investigation into potential financial fraud.',
      status: 'pending',
      created: '2024-01-10',
      priority: 'medium'
    },
    { 
      id: 'case-003', 
      title: 'Identity Theft Case', 
      description: 'Case involving stolen personal information.',
      status: 'closed',
      created: '2024-01-05',
      priority: 'low'
    }
  ];

  onMount(() => {
    if (!$user?.id) {
      goto('/login');
    }
  });
</script>

<svelte:head>
  <title>Cases - Legal Case Management</title>
</svelte:head>

<div class="cases-container">
  <div class="cases-header">
    <div>
      <h1>Legal Cases</h1>
      <p>Manage and track all your legal cases</p>
    </div>
    
    <div class="header-actions">
      <button class="nier-button-primary">
        ➕ New Case
      </button>
      <button class="btn btn-secondary">
        📤 Import Cases
      </button>
    </div>
  </div>

  <div class="cases-filters">
    <div class="filter-group">
      <label for="status-filter">Filter by Status:</label>
      <select id="status-filter" class="form-select">
        <option value="">All Cases</option>
        <option value="active">Active</option>
        <option value="pending">Pending</option>
        <option value="closed">Closed</option>
      </select>
    </div>
    
    <div class="filter-group">
      <label for="search">Search Cases:</label>
      <input id="search" type="text" placeholder="Search by title or description..." class="form-input" />
    </div>
  </div>

  <div class="cases-grid">
    {#each cases as caseItem}
      <div class="case-card">
        <div class="case-header">
          <h3>{caseItem.title}</h3>
          <span class="case-status status-{caseItem.status}">{caseItem.status}</span>
        </div>
        
        <p class="case-description">{caseItem.description}</p>
        
        <div class="case-meta">
          <div class="meta-item">
            <span class="meta-label">Created:</span>
            <span class="meta-value">{caseItem.created}</span>
          </div>
          <div class="meta-item">
            <span class="meta-label">Priority:</span>
            <span class="meta-value priority-{caseItem.priority}">{caseItem.priority}</span>
          </div>
        </div>
        
        <div class="case-actions">
          <button class="btn btn-sm btn-primary">View Details</button>
          <button class="btn btn-sm btn-secondary">Edit Case</button>
          <button class="btn btn-sm btn-outline">Archive</button>
        </div>
      </div>
    {/each}
  </div>

  {#if cases.length === 0}
    <div class="empty-state">
      <div class="empty-icon">📁</div>
      <h3>No Cases Found</h3>
      <p>Create your first case to get started with case management.</p>
      <button class="nier-button-primary">Create First Case</button>
    </div>
  {/if}
</div>

<style>
  /* @unocss-include */
  .cases-container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem;
  }

  .cases-header {
    display: flex
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 2rem;
    flex-wrap: wrap
    gap: 1rem;
  }

  .cases-header h1 {
    font-size: 2rem;
    font-weight: 700;
    color: #1f2937;
    margin-bottom: 0.5rem;
  }

  .cases-header p {
    color: #6b7280;
    margin: 0;
  }

  .header-actions {
    display: flex
    gap: 1rem;
  }

  .cases-filters {
    background: white
    padding: 1.5rem;
    border-radius: 0.75rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    margin-bottom: 2rem;
    display: flex
    gap: 2rem;
    flex-wrap: wrap
    align-items: end
  }

  .filter-group {
    display: flex
    flex-direction: column
    gap: 0.5rem;
    flex: 1;
    min-width: 200px;
  }

  .filter-group label {
    font-weight: 500;
    color: #374151;
    font-size: 0.875rem;
  }

  .form-select, .form-input {
    padding: 0.5rem;
    border: 1px solid #d1d5db;
    border-radius: 0.375rem;
    font-size: 0.875rem;
  }

  .form-input {
    flex: 1;
  }

  .cases-grid {
    display: grid
    grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
    gap: 1.5rem;
  }

  .case-card {
    background: white
    border-radius: 0.75rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    padding: 1.5rem;
    transition: all 0.2s ease;
  }

  .case-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  }

  .case-header {
    display: flex
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 1rem;
    gap: 1rem;
  }

  .case-header h3 {
    font-size: 1.125rem;
    font-weight: 600;
    color: #1f2937;
    margin: 0;
    flex: 1;
  }

  .case-status {
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase
    white-space: nowrap
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

  .case-description {
    color: #6b7280;
    font-size: 0.875rem;
    line-height: 1.5;
    margin-bottom: 1rem;
  }

  .case-meta {
    display: flex
    flex-direction: column
    gap: 0.5rem;
    margin-bottom: 1rem;
    padding-top: 1rem;
    border-top: 1px solid #f3f4f6;
  }

  .meta-item {
    display: flex
    justify-content: space-between;
    font-size: 0.875rem;
  }

  .meta-label {
    color: #6b7280;
    font-weight: 500;
  }

  .meta-value {
    color: #1f2937;
    font-weight: 600;
  }

  .priority-high {
    color: #dc2626;
  }

  .priority-medium {
    color: #d97706;
  }

  .priority-low {
    color: #059669;
  }

  .case-actions {
    display: flex
    gap: 0.5rem;
    flex-wrap: wrap
  }

  .btn {
    padding: 0.5rem 1rem;
    border: none
    border-radius: 0.375rem;
    font-size: 0.875rem;
    font-weight: 500;
    cursor: pointer
    transition: all 0.2s ease;
    text-decoration: none
    display: inline-flex;
    align-items: center
    justify-content: center
  }

  .btn-sm {
    padding: 0.375rem 0.75rem;
    font-size: 0.75rem;
  }

  .btn-primary {
    background: #3b82f6;
    color: white
  }

  .btn-primary:hover {
    background: #2563eb;
  }

  .btn-secondary {
    background: #f3f4f6;
    color: #374151;
  }

  .btn-secondary:hover {
    background: #e5e7eb;
  }

  .btn-outline {
    background: transparent
    color: #6b7280;
    border: 1px solid #d1d5db;
  }

  .btn-outline:hover {
    background: #f9fafb;
    color: #374151;
  }

  .empty-state {
    text-align: center
    padding: 4rem 2rem;
    background: white
    border-radius: 0.75rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  }

  .empty-icon {
    font-size: 4rem;
    margin-bottom: 1rem;
    opacity: 0.5;
  }

  .empty-state h3 {
    font-size: 1.25rem;
    font-weight: 600;
    color: #1f2937;
    margin-bottom: 0.5rem;
  }

  .empty-state p {
    color: #6b7280;
    margin-bottom: 2rem;
  }

  @media (max-width: 768px) {
    .cases-container {
      padding: 1rem;
    }
    
    .cases-header {
      flex-direction: column
      align-items: stretch
    }
    
    .header-actions {
      flex-direction: column
    }
    
    .cases-filters {
      flex-direction: column
      gap: 1rem;
    }
    
    .cases-grid {
      grid-template-columns: 1fr;
    }
    
    .case-actions {
      flex-direction: column
    }
  }
</style>
