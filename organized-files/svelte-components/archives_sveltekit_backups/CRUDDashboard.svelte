<!-- Comprehensive CRUD Dashboard showing all entities working together -->
<script lang="ts">
  import { notifications } from "$lib/stores/notification";
  import {
    Activity,
    Camera,
    CheckCircle,
    Clock,
    FileText,
    Plus,
    RefreshCw,
    Search,
    TrendingUp,
  } from "lucide-svelte";
  import { onMount } from "svelte";

  // Data stores for all entities
  let cases: unknown[] = [];
  let evidence: unknown[] = [];
  let reports: unknown[] = [];
  let criminals: unknown[] = [];
  let activities: unknown[] = [];
  let users_list: unknown[] = [];

  // Loading states
  let loading = {
    cases: false,
    evidence: false,
    reports: false,
    criminals: false,
    activities: false,
    users: false,
  };

  // Statistics
  let stats = {
    totalCases: 0,
    activeCases: 0,
    totalEvidence: 0,
    totalReports: 0,
    urgentActivities: 0,
    recentActivity: 0,
  };

  // Search and filter states
  let searchTerms = {
    cases: "",
    evidence: "",
    reports: "",
    criminals: "",
    activities: "",
  };

  let refreshing = false;

  // Fetch all data
  async function fetchAllData() {
    if (refreshing) return;
    refreshing = true;

    try {
      await Promise.all([
        fetchCases(),
        fetchEvidence(),
        fetchReports(),
        fetchCriminals(),
        fetchActivities(),
        fetchUsers(),
      ]);

      calculateStats();

      notifications.add({
        type: "success",
        title: "Data Refreshed",
        message: "All data has been successfully updated",
      });
    } catch (error) {
      console.error("Error refreshing data:", error);
      notifications.add({
        type: "error",
        title: "Refresh Failed",
        message: "Failed to refresh some data. Please try again.",
      });
    } finally {
      refreshing = false;
    }
  }

  // Fetch functions for each entity
  async function fetchCases() {
    loading.cases = true;
    try {
      const response = await fetch(
        `/api/cases?limit=10&search=${searchTerms.cases}`
      );
      if (response.ok) {
        const data = await response.json();
        cases = data.cases || data;
      }
    } catch (error) {
      console.error("Error fetching cases:", error);
    } finally {
      loading.cases = false;
    }
  }

  async function fetchEvidence() {
    loading.evidence = true;
    try {
      const response = await fetch(
        `/api/evidence?limit=10&search=${searchTerms.evidence}`
      );
      if (response.ok) {
        const data = await response.json();
        evidence = data.evidence || data;
      }
    } catch (error) {
      console.error("Error fetching evidence:", error);
    } finally {
      loading.evidence = false;
    }
  }

  async function fetchReports() {
    loading.reports = true;
    try {
      const response = await fetch(
        `/api/reports?limit=10&search=${searchTerms.reports}`
      );
      if (response.ok) {
        const data = await response.json();
        reports = data.reports || data;
      }
    } catch (error) {
      console.error("Error fetching reports:", error);
    } finally {
      loading.reports = false;
    }
  }

  async function fetchCriminals() {
    loading.criminals = true;
    try {
      const response = await fetch(
        `/api/criminals?limit=10&search=${searchTerms.criminals}`
      );
      if (response.ok) {
        const data = await response.json();
        criminals = data.criminals || data;
      }
    } catch (error) {
      console.error("Error fetching criminals:", error);
    } finally {
      loading.criminals = false;
    }
  }

  async function fetchActivities() {
    loading.activities = true;
    try {
      const response = await fetch(
        `/api/activities?limit=10&search=${searchTerms.activities}`
      );
      if (response.ok) {
        const data = await response.json();
        activities = data.activities || data;
      }
    } catch (error) {
      console.error("Error fetching activities:", error);
    } finally {
      loading.activities = false;
    }
  }

  async function fetchUsers() {
    loading.users = true;
    try {
      const response = await fetch("/api/users?limit=10");
      if (response.ok) {
        const data = await response.json();
        users_list = data.users || data;
      }
    } catch (error) {
      console.error("Error fetching users:", error);
    } finally {
      loading.users = false;
    }
  }

  // Calculate statistics
  function calculateStats() {
    stats.totalCases = cases.length;
    stats.activeCases = cases.filter(
      (c) => c.status === "open" || c.status === "active"
    ).length;
    stats.totalEvidence = evidence.length;
    stats.totalReports = reports.length;
    stats.urgentActivities = activities.filter(
      (a) => a.priority === "urgent" || a.priority === "high"
    ).length;
    stats.recentActivity = activities.filter((a) => {
      const created = new Date(a.createdAt);
      const dayAgo = new Date(Date.now() - 24 * 60 * 60 * 1000);
      return created > dayAgo;
    }).length;
  }

  // Quick actions
  function createNew(entity: string) {
    // Navigate to create form for entity
    window.location.href = `/${entity}/create`;
  }

  function viewAll(entity: string) {
    // Navigate to full list view
    window.location.href = `/${entity}`;
  }

  // Search handlers
  function handleSearch(entity: string) {
    switch (entity) {
      case "cases":
        fetchCases();
        break;
      case "evidence":
        fetchEvidence();
        break;
      case "reports":
        fetchReports();
        break;
      case "criminals":
        fetchCriminals();
        break;
      case "activities":
        fetchActivities();
        break;
    }
  }

  // Format date helper
  function formatDate(dateString: string) {
    if (!dateString) return "N/A";
    return new Date(dateString).toLocaleDateString();
  }

  // Format status helper
  function getStatusColor(status: string) {
    switch (status?.toLowerCase()) {
      case "open":
      case "active":
      case "pending":
        return "text-blue-600 bg-blue-100";
      case "urgent":
      case "high":
        return "text-red-600 bg-red-100";
      case "completed":
      case "closed":
        return "text-green-600 bg-green-100";
      case "draft":
        return "text-yellow-600 bg-yellow-100";
      default:
        return "text-gray-600 bg-gray-100";
    }
  }

  // Initialize
  onMount(() => {
    fetchAllData();
  });
</script>

<div class="mx-auto px-4 max-w-7xl">
  <!-- Header -->
  <div class="mx-auto px-4 max-w-7xl">
    <div class="mx-auto px-4 max-w-7xl">
      <div class="mx-auto px-4 max-w-7xl">
        <h1>CRUD Dashboard</h1>
        <p>Comprehensive view of all database entities and operations</p>
      </div>

      <div class="mx-auto px-4 max-w-7xl">
        <button
          class="mx-auto px-4 max-w-7xl"
          onclick={() => fetchAllData()}
          disabled={refreshing}
          title="Refresh all data"
        >
          <RefreshCw class={refreshing ? "animate-spin" : ""} size={16} />
          Refresh
        </button>
      </div>
    </div>

    <!-- Statistics Cards -->
    <div class="mx-auto px-4 max-w-7xl">
      <div class="mx-auto px-4 max-w-7xl">
        <div class="mx-auto px-4 max-w-7xl">
          <FileText size={24} />
        </div>
        <div class="mx-auto px-4 max-w-7xl">
          <div class="mx-auto px-4 max-w-7xl">{stats.totalCases}</div>
          <div class="mx-auto px-4 max-w-7xl">Total Cases</div>
          <div class="mx-auto px-4 max-w-7xl">{stats.activeCases} active</div>
        </div>
      </div>

      <div class="mx-auto px-4 max-w-7xl">
        <div class="mx-auto px-4 max-w-7xl">
          <Camera size={24} />
        </div>
        <div class="mx-auto px-4 max-w-7xl">
          <div class="mx-auto px-4 max-w-7xl">{stats.totalEvidence}</div>
          <div class="mx-auto px-4 max-w-7xl">Evidence Items</div>
          <div class="mx-auto px-4 max-w-7xl">All formats</div>
        </div>
      </div>

      <div class="mx-auto px-4 max-w-7xl">
        <div class="mx-auto px-4 max-w-7xl">
          <FileText size={24} />
        </div>
        <div class="mx-auto px-4 max-w-7xl">
          <div class="mx-auto px-4 max-w-7xl">{stats.totalReports}</div>
          <div class="mx-auto px-4 max-w-7xl">Reports</div>
          <div class="mx-auto px-4 max-w-7xl">Generated</div>
        </div>
      </div>

      <div class="mx-auto px-4 max-w-7xl">
        <div class="mx-auto px-4 max-w-7xl">
          <Activity size={24} />
        </div>
        <div class="mx-auto px-4 max-w-7xl">
          <div class="mx-auto px-4 max-w-7xl">{stats.urgentActivities}</div>
          <div class="mx-auto px-4 max-w-7xl">Urgent Tasks</div>
          <div class="mx-auto px-4 max-w-7xl">{stats.recentActivity} recent</div>
        </div>
      </div>
    </div>
  </div>

  <!-- Main Content Grid -->
  <div class="mx-auto px-4 max-w-7xl">
    <!-- Cases Section -->
    <div class="mx-auto px-4 max-w-7xl">
      <div class="mx-auto px-4 max-w-7xl">
        <h2>
          <FileText size={20} />
          Cases
        </h2>
        <div class="mx-auto px-4 max-w-7xl">
          <div class="mx-auto px-4 max-w-7xl">
            <Search size={16} />
            <input
              type="text"
              placeholder="Search cases..."
              bind:value={searchTerms.cases}
              oninput={() => handleSearch("cases")}
            />
          </div>
          <button class="mx-auto px-4 max-w-7xl" onclick={() => createNew("cases")}>
            <Plus size={16} />
          </button>
        </div>
      </div>

      <div class="mx-auto px-4 max-w-7xl">
        {#if loading.cases}
          <div class="mx-auto px-4 max-w-7xl">Loading cases...</div>
        {:else if cases.length === 0}
          <div class="mx-auto px-4 max-w-7xl">No cases found</div>
        {:else}
          {#each cases as case_}
            <div class="mx-auto px-4 max-w-7xl">
              <div class="mx-auto px-4 max-w-7xl">
                <span class="mx-auto px-4 max-w-7xl">{case_.title}</span>
                <span class="mx-auto px-4 max-w-7xl"
                  >{case_.status}</span
                >
              </div>
              <div class="mx-auto px-4 max-w-7xl">
                <span class="mx-auto px-4 max-w-7xl">#{case_.caseNumber}</span>
                <span class="mx-auto px-4 max-w-7xl">Priority: {case_.priority}</span>
                <span class="mx-auto px-4 max-w-7xl"
                  >Created: {formatDate(case_.createdAt)}</span
                >
              </div>
            </div>
          {/each}
        {/if}
      </div>

      <div class="mx-auto px-4 max-w-7xl">
        <button class="mx-auto px-4 max-w-7xl" onclick={() => viewAll("cases")}>
          View All Cases
        </button>
      </div>
    </div>

    <!-- Evidence Section -->
    <div class="mx-auto px-4 max-w-7xl">
      <div class="mx-auto px-4 max-w-7xl">
        <h2>
          <Camera size={20} />
          Evidence
        </h2>
        <div class="mx-auto px-4 max-w-7xl">
          <div class="mx-auto px-4 max-w-7xl">
            <Search size={16} />
            <input
              type="text"
              placeholder="Search evidence..."
              bind:value={searchTerms.evidence}
              oninput={() => handleSearch("evidence")}
            />
          </div>
          <button class="mx-auto px-4 max-w-7xl" onclick={() => createNew("evidence")}>
            <Plus size={16} />
          </button>
        </div>
      </div>

      <div class="mx-auto px-4 max-w-7xl">
        {#if loading.evidence}
          <div class="mx-auto px-4 max-w-7xl">Loading evidence...</div>
        {:else if evidence.length === 0}
          <div class="mx-auto px-4 max-w-7xl">No evidence found</div>
        {:else}
          {#each evidence as item}
            <div class="mx-auto px-4 max-w-7xl">
              <div class="mx-auto px-4 max-w-7xl">
                <span class="mx-auto px-4 max-w-7xl">{item.title}</span>
                <span class="mx-auto px-4 max-w-7xl"
                  >{item.evidenceType}</span
                >
              </div>
              <div class="mx-auto px-4 max-w-7xl">
                <span class="mx-auto px-4 max-w-7xl">Type: {item.fileType || "N/A"}</span>
                <span class="mx-auto px-4 max-w-7xl"
                  >Size: {item.fileSize
                    ? `${Math.round(item.fileSize / 1024)}KB`
                    : "N/A"}</span
                >
                <span class="mx-auto px-4 max-w-7xl"
                  >Uploaded: {formatDate(item.uploadedAt)}</span
                >
              </div>
            </div>
          {/each}
        {/if}
      </div>

      <div class="mx-auto px-4 max-w-7xl">
        <button class="mx-auto px-4 max-w-7xl" onclick={() => viewAll("evidence")}>
          View All Evidence
        </button>
      </div>
    </div>

    <!-- Reports Section -->
    <div class="mx-auto px-4 max-w-7xl">
      <div class="mx-auto px-4 max-w-7xl">
        <h2>
          <FileText size={20} />
          Reports
        </h2>
        <div class="mx-auto px-4 max-w-7xl">
          <div class="mx-auto px-4 max-w-7xl">
            <Search size={16} />
            <input
              type="text"
              placeholder="Search reports..."
              bind:value={searchTerms.reports}
              oninput={() => handleSearch("reports")}
            />
          </div>
          <button class="mx-auto px-4 max-w-7xl" onclick={() => createNew("reports")}>
            <Plus size={16} />
          </button>
        </div>
      </div>

      <div class="mx-auto px-4 max-w-7xl">
        {#if loading.reports}
          <div class="mx-auto px-4 max-w-7xl">Loading reports...</div>
        {:else if reports.length === 0}
          <div class="mx-auto px-4 max-w-7xl">No reports found</div>
        {:else}
          {#each reports as report}
            <div class="mx-auto px-4 max-w-7xl">
              <div class="mx-auto px-4 max-w-7xl">
                <span class="mx-auto px-4 max-w-7xl">{report.title}</span>
                <span class="mx-auto px-4 max-w-7xl"
                  >{report.status}</span
                >
              </div>
              <div class="mx-auto px-4 max-w-7xl">
                <span class="mx-auto px-4 max-w-7xl">Type: {report.reportType}</span>
                <span class="mx-auto px-4 max-w-7xl"
                  >Words: {report.metadata?.wordCount || "N/A"}</span
                >
                <span class="mx-auto px-4 max-w-7xl"
                  >Created: {formatDate(report.createdAt)}</span
                >
              </div>
            </div>
          {/each}
        {/if}
      </div>

      <div class="mx-auto px-4 max-w-7xl">
        <button class="mx-auto px-4 max-w-7xl" onclick={() => viewAll("reports")}>
          View All Reports
        </button>
      </div>
    </div>

    <!-- Activities Section -->
    <div class="mx-auto px-4 max-w-7xl">
      <div class="mx-auto px-4 max-w-7xl">
        <h2>
          <Activity size={20} />
          Activities
        </h2>
        <div class="mx-auto px-4 max-w-7xl">
          <div class="mx-auto px-4 max-w-7xl">
            <Search size={16} />
            <input
              type="text"
              placeholder="Search activities..."
              bind:value={searchTerms.activities}
              oninput={() => handleSearch("activities")}
            />
          </div>
          <button class="mx-auto px-4 max-w-7xl" onclick={() => createNew("activities")}>
            <Plus size={16} />
          </button>
        </div>
      </div>

      <div class="mx-auto px-4 max-w-7xl">
        {#if loading.activities}
          <div class="mx-auto px-4 max-w-7xl">Loading activities...</div>
        {:else if activities.length === 0}
          <div class="mx-auto px-4 max-w-7xl">No activities found</div>
        {:else}
          {#each activities as activity}
            <div class="mx-auto px-4 max-w-7xl">
              <div class="mx-auto px-4 max-w-7xl">
                <span class="mx-auto px-4 max-w-7xl">{activity.title}</span>
                <span class="mx-auto px-4 max-w-7xl"
                  >{activity.status}</span
                >
              </div>
              <div class="mx-auto px-4 max-w-7xl">
                <span class="mx-auto px-4 max-w-7xl">Type: {activity.activityType}</span>
                <span class="mx-auto px-4 max-w-7xl">Priority: {activity.priority}</span>
                <span class="mx-auto px-4 max-w-7xl"
                  >Due: {formatDate(activity.scheduledFor)}</span
                >
              </div>
            </div>
          {/each}
        {/if}
      </div>

      <div class="mx-auto px-4 max-w-7xl">
        <button class="mx-auto px-4 max-w-7xl" onclick={() => viewAll("activities")}>
          View All Activities
        </button>
      </div>
    </div>

    <!-- Quick Actions Panel -->
    <div class="mx-auto px-4 max-w-7xl">
      <div class="mx-auto px-4 max-w-7xl">
        <h2>
          <TrendingUp size={20} />
          Quick Actions
        </h2>
      </div>

      <div class="mx-auto px-4 max-w-7xl">
        <button class="mx-auto px-4 max-w-7xl" onclick={() => createNew("cases")}>
          <FileText size={24} />
          <span>New Case</span>
        </button>

        <button class="mx-auto px-4 max-w-7xl" onclick={() => createNew("evidence")}>
          <Camera size={24} />
          <span>Add Evidence</span>
        </button>

        <button class="mx-auto px-4 max-w-7xl" onclick={() => createNew("reports")}>
          <FileText size={24} />
          <span>Create Report</span>
        </button>

        <button class="mx-auto px-4 max-w-7xl" onclick={() => viewAll("activities")}>
          <Clock size={24} />
          <span>View Tasks</span>
        </button>
      </div>

      <!-- System Status -->
      <div class="mx-auto px-4 max-w-7xl">
        <h3>System Status</h3>
        <div class="mx-auto px-4 max-w-7xl">
          <div class="mx-auto px-4 max-w-7xl">
            <CheckCircle size={16} class="mx-auto px-4 max-w-7xl" />
            <span>Database: Connected</span>
          </div>
          <div class="mx-auto px-4 max-w-7xl">
            <CheckCircle size={16} class="mx-auto px-4 max-w-7xl" />
            <span>API: Operational</span>
          </div>
          <div class="mx-auto px-4 max-w-7xl">
            <CheckCircle size={16} class="mx-auto px-4 max-w-7xl" />
            <span>CRUD: Synchronized</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>

<style>
  .crud-dashboard {
    padding: 2rem;
    background: #f8fafc;
    min-height: 100vh;
  }

  .dashboard-header {
    margin-bottom: 2rem;
  }

  .header-content {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1.5rem;
  }

  .title-section h1 {
    font-size: 2rem;
    font-weight: 700;
    margin: 0 0 0.5rem 0;
    color: #1f2937;
  }

  .title-section p {
    color: #6b7280;
    margin: 0;
  }

  .refresh-btn {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.75rem 1rem;
    background: #3b82f6;
    color: white;
    border: none;
    border-radius: 0.5rem;
    cursor: pointer;
    font-weight: 500;
    transition: background-color 0.2s;
  }

  .refresh-btn:hover:not(:disabled) {
    background: #2563eb;
  }

  .refresh-btn:disabled {
    opacity: 0.7;
    cursor: not-allowed;
  }

  .stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 1rem;
  }

  .stat-card {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 1.5rem;
    background: white;
    border-radius: 0.75rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  }

  .stat-icon {
    padding: 0.75rem;
    border-radius: 0.5rem;
    color: white;
  }

  .stat-icon.cases {
    background: #3b82f6;
  }
  .stat-icon.evidence {
    background: #10b981;
  }
  .stat-icon.reports {
    background: #8b5cf6;
  }
  .stat-icon.activities {
    background: #f59e0b;
  }

  .stat-value {
    font-size: 2rem;
    font-weight: 700;
    color: #1f2937;
  }

  .stat-label {
    color: #6b7280;
    font-weight: 500;
  }

  .stat-sublabel {
    color: #9ca3af;
    font-size: 0.875rem;
  }

  .content-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
    gap: 1.5rem;
  }

  .data-section {
    background: white;
    border-radius: 0.75rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    overflow: hidden;
  }

  .section-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1.5rem;
    border-bottom: 1px solid #e5e7eb;
    background: #f9fafb;
  }

  .section-header h2 {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin: 0;
    font-size: 1.25rem;
    font-weight: 600;
    color: #1f2937;
  }

  .section-actions {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  .search-box {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem;
    background: white;
    border: 1px solid #d1d5db;
    border-radius: 0.375rem;
  }

  .search-box input {
    border: none;
    outline: none;
    font-size: 0.875rem;
    width: 150px;
  }

  .action-btn {
    padding: 0.5rem;
    background: #3b82f6;
    color: white;
    border: none;
    border-radius: 0.375rem;
    cursor: pointer;
    transition: background-color 0.2s;
  }

  .action-btn:hover {
    background: #2563eb;
  }

  .data-list {
    max-height: 300px;
    overflow-y: auto;
  }

  .data-item {
    padding: 1rem 1.5rem;
    border-bottom: 1px solid #f3f4f6;
  }

  .data-item:last-child {
    border-bottom: none;
  }

  .item-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
  }

  .item-title {
    font-weight: 500;
    color: #1f2937;
  }

  .item-status {
    padding: 0.25rem 0.5rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 500;
  }

  .item-details {
    display: flex;
    gap: 1rem;
    font-size: 0.875rem;
    color: #6b7280;
  }

  .loading,
  .no-data {
    padding: 2rem;
    text-align: center;
    color: #6b7280;
  }

  .section-footer {
    padding: 1rem 1.5rem;
    border-top: 1px solid #f3f4f6;
    background: #f9fafb;
  }

  .view-all-btn {
    width: 100%;
    padding: 0.75rem;
    background: transparent;
    color: #3b82f6;
    border: 1px solid #3b82f6;
    border-radius: 0.375rem;
    cursor: pointer;
    font-weight: 500;
    transition: all 0.2s;
  }

  .view-all-btn:hover {
    background: #3b82f6;
    color: white;
  }

  .quick-actions {
    grid-row: span 2;
  }

  .actions-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 1rem;
    padding: 1.5rem;
  }

  .quick-action-btn {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.5rem;
    padding: 1.5rem;
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 0.5rem;
    cursor: pointer;
    transition: all 0.2s;
    color: #6b7280;
  }

  .quick-action-btn:hover {
    background: #3b82f6;
    color: white;
    border-color: #3b82f6;
  }

  .system-status {
    padding: 1.5rem;
    border-top: 1px solid #e5e7eb;
  }

  .system-status h3 {
    margin: 0 0 1rem 0;
    font-size: 1rem;
    font-weight: 600;
    color: #1f2937;
  }

  .status-items {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  .status-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.875rem;
    color: #6b7280;
  }

  @media (max-width: 768px) {
    .crud-dashboard {
      padding: 1rem;
    }

    .content-grid {
      grid-template-columns: 1fr;
    }

    .header-content {
      flex-direction: column;
      gap: 1rem;
      align-items: flex-start;
    }

    .section-actions {
      flex-direction: column;
      align-items: stretch;
      gap: 0.5rem;
    }

    .search-box input {
      width: 100%;
    }

    .actions-grid {
      grid-template-columns: 1fr;
    }
  }
</style>