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
  let cases: any[] = [];
  let evidence: any[] = [];
  let reports: any[] = [];
  let criminals: any[] = [];
  let activities: any[] = [];
  let users_list: any[] = [];

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
  // Initialize
  onMount(() => {
    fetchAllData();
  });
</script>

<div class="crud-dashboard container mx-auto px-4">
  <!-- Header -->
  <div class="space-y-4">
    <div class="space-y-4">
      <div class="space-y-4">
        <h1>CRUD Dashboard</h1>
        <p>Comprehensive view of all database entities and operations</p>
      </div>

      <div class="space-y-4">
        <button
          class="space-y-4"
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
    <div class="space-y-4">
      <div class="space-y-4">
        <div class="space-y-4">
          <FileText size={24} />
        </div>
        <div class="space-y-4">
          <div class="space-y-4">{stats.totalCases}</div>
          <div class="space-y-4">Total Cases</div>
          <div class="space-y-4">{stats.activeCases} active</div>
        </div>
      </div>

      <div class="space-y-4">
        <div class="space-y-4">
          <Camera size={24} />
        </div>
        <div class="space-y-4">
          <div class="space-y-4">{stats.totalEvidence}</div>
          <div class="space-y-4">Evidence Items</div>
          <div class="space-y-4">All formats</div>
        </div>
      </div>

      <div class="space-y-4">
        <div class="space-y-4">
          <FileText size={24} />
        </div>
        <div class="space-y-4">
          <div class="space-y-4">{stats.totalReports}</div>
          <div class="space-y-4">Reports</div>
          <div class="space-y-4">Generated</div>
        </div>
      </div>

      <div class="space-y-4">
        <div class="space-y-4">
          <Activity size={24} />
        </div>
        <div class="space-y-4">
          <div class="space-y-4">{stats.urgentActivities}</div>
          <div class="space-y-4">Urgent Tasks</div>
          <div class="space-y-4">{stats.recentActivity} recent</div>
        </div>
      </div>
    </div>
  </div>

  <!-- Main Content Grid -->
  <div class="space-y-4">
    <!-- Cases Section -->
    <div class="space-y-4">
      <div class="space-y-4">
        <h2>
          <FileText size={20} />
          Cases
        </h2>
        <div class="space-y-4">
          <div class="space-y-4">
            <Search size={16} />
            <input
              type="text"
              placeholder="Search cases..."
              bind:value={searchTerms.cases}
              oninput={() => handleSearch("cases")}
            />
          </div>
          <button class="space-y-4" onclick={() => createNew("cases")}>
            <Plus size={16} />
          </button>
        </div>
      </div>

      <div class="space-y-4">
        {#if loading.cases}
          <div class="space-y-4">Loading cases...</div>
        {:else if cases.length === 0}
          <div class="space-y-4">No cases found</div>
        {:else}
          {#each cases as case_}
            <div class="space-y-4">
              <div class="space-y-4">
                <span class="space-y-4">{case_.title}</span>
                <span class="space-y-4"
                  >{case_.status}</span
                >
              </div>
              <div class="space-y-4">
                <span class="space-y-4">#{case_.caseNumber}</span>
                <span class="space-y-4">Priority: {case_.priority}</span>
                <span class="space-y-4"
                  >Created: {formatDate(case_.createdAt)}</span
                >
              </div>
            </div>
          {/each}
        {/if}
      </div>

      <div class="space-y-4">
        <button class="space-y-4" onclick={() => viewAll("cases")}>
          View All Cases
        </button>
      </div>
    </div>

    <!-- Evidence Section -->
    <div class="space-y-4">
      <div class="space-y-4">
        <h2>
          <Camera size={20} />
          Evidence
        </h2>
        <div class="space-y-4">
          <div class="space-y-4">
            <Search size={16} />
            <input
              type="text"
              placeholder="Search evidence..."
              bind:value={searchTerms.evidence}
              oninput={() => handleSearch("evidence")}
            />
          </div>
          <button class="space-y-4" onclick={() => createNew("evidence")}>
            <Plus size={16} />
          </button>
        </div>
      </div>

      <div class="space-y-4">
        {#if loading.evidence}
          <div class="space-y-4">Loading evidence...</div>
        {:else if evidence.length === 0}
          <div class="space-y-4">No evidence found</div>
        {:else}
          {#each evidence as item}
            <div class="space-y-4">
              <div class="space-y-4">
                <span class="space-y-4">{item.title}</span>
                <span class="space-y-4"
                  >{item.evidenceType}</span
                >
              </div>
              <div class="space-y-4">
                <span class="space-y-4">Type: {item.fileType || "N/A"}</span>
                <span class="space-y-4"
                  >Size: {item.fileSize
                    ? `${Math.round(item.fileSize / 1024)}KB`
                    : "N/A"}</span
                >
                <span class="space-y-4"
                  >Uploaded: {formatDate(item.uploadedAt)}</span
                >
              </div>
            </div>
          {/each}
        {/if}
      </div>

      <div class="space-y-4">
        <button class="space-y-4" onclick={() => viewAll("evidence")}>
          View All Evidence
        </button>
      </div>
    </div>

    <!-- Reports Section -->
    <div class="space-y-4">
      <div class="space-y-4">
        <h2>
          <FileText size={20} />
          Reports
        </h2>
        <div class="space-y-4">
          <div class="space-y-4">
            <Search size={16} />
            <input
              type="text"
              placeholder="Search reports..."
              bind:value={searchTerms.reports}
              oninput={() => handleSearch("reports")}
            />
          </div>
          <button class="space-y-4" onclick={() => createNew("reports")}>
            <Plus size={16} />
          </button>
        </div>
      </div>

      <div class="space-y-4">
        {#if loading.reports}
          <div class="space-y-4">Loading reports...</div>
        {:else if reports.length === 0}
          <div class="space-y-4">No reports found</div>
        {:else}
          {#each reports as report}
            <div class="space-y-4">
              <div class="space-y-4">
                <span class="space-y-4">{report.title}</span>
                <span class="space-y-4"
                  >{report.status}</span
                >
              </div>
              <div class="space-y-4">
                <span class="space-y-4">Type: {report.reportType}</span>
                <span class="space-y-4"
                  >Words: {report.metadata?.wordCount || "N/A"}</span
                >
                <span class="space-y-4"
                  >Created: {formatDate(report.createdAt)}</span
                >
              </div>
            </div>
          {/each}
        {/if}
      </div>

      <div class="space-y-4">
        <button class="space-y-4" onclick={() => viewAll("reports")}>
          View All Reports
        </button>
      </div>
    </div>

    <!-- Activities Section -->
    <div class="space-y-4">
      <div class="space-y-4">
        <h2>
          <Activity size={20} />
          Activities
        </h2>
        <div class="space-y-4">
          <div class="space-y-4">
            <Search size={16} />
            <input
              type="text"
              placeholder="Search activities..."
              bind:value={searchTerms.activities}
              oninput={() => handleSearch("activities")}
            />
          </div>
          <button class="space-y-4" onclick={() => createNew("activities")}>
            <Plus size={16} />
          </button>
        </div>
      </div>

      <div class="space-y-4">
        {#if loading.activities}
          <div class="space-y-4">Loading activities...</div>
        {:else if activities.length === 0}
          <div class="space-y-4">No activities found</div>
        {:else}
          {#each activities as activity}
            <div class="space-y-4">
              <div class="space-y-4">
                <span class="space-y-4">{activity.title}</span>
                <span class="space-y-4"
                  >{activity.status}</span
                >
              </div>
              <div class="space-y-4">
                <span class="space-y-4">Type: {activity.activityType}</span>
                <span class="space-y-4">Priority: {activity.priority}</span>
                <span class="space-y-4"
                  >Due: {formatDate(activity.scheduledFor)}</span
                >
              </div>
            </div>
          {/each}
        {/if}
      </div>

      <div class="space-y-4">
        <button class="space-y-4" onclick={() => viewAll("activities")}>
          View All Activities
        </button>
      </div>
    </div>

    <!-- Quick Actions Panel -->
    <div class="space-y-4">
      <div class="space-y-4">
        <h2>
          <TrendingUp size={20} />
          Quick Actions
        </h2>
      </div>

      <div class="space-y-4">
        <button class="space-y-4" onclick={() => createNew("cases")}>
          <FileText size={24} />
          <span>New Case</span>
        </button>

        <button class="space-y-4" onclick={() => createNew("evidence")}>
          <Camera size={24} />
          <span>Add Evidence</span>
        </button>

        <button class="space-y-4" onclick={() => createNew("reports")}>
          <FileText size={24} />
          <span>Create Report</span>
        </button>

        <button class="space-y-4" onclick={() => viewAll("activities")}>
          <Clock size={24} />
          <span>View Tasks</span>
        </button>
      </div>

      <!-- System Status -->
      <div class="space-y-4">
        <h3>System Status</h3>
        <div class="space-y-4">
          <div class="space-y-4">
            <CheckCircle size={16} class="space-y-4" />
            <span>Database: Connected</span>
          </div>
          <div class="space-y-4">
            <CheckCircle size={16} class="space-y-4" />
            <span>API: Operational</span>
          </div>
          <div class="space-y-4">
            <CheckCircle size={16} class="space-y-4" />
            <span>CRUD: Synchronized</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>

<style>
  /* @unocss-include */
  .crud-dashboard {
    padding: 2rem;
    background: #f8fafc;
    min-height: 100vh;
}
  .dashboard-header {
    margin-bottom: 2rem;
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
    display: flex
    align-items: center
    gap: 0.5rem;
    padding: 0.75rem 1rem;
    background: #3b82f6;
  }
  @media (max-width: 768px) {
    .crud-dashboard {
      padding: 1rem;
}
    .content-grid {
      grid-template-columns: 1fr;
}
    .header-content {
      flex-direction: column
      gap: 1rem;
      align-items: flex-start;
}
    .section-actions {
      flex-direction: column
      align-items: stretch
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
