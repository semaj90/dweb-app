<!--
  Security Monitoring Dashboard
  Displays security events, system health, and security metrics
-->
<script lang="ts">
  import { $derived } from 'svelte';

  import { Button } from "$lib/components/ui/button";
  import { notifications } from "$lib/stores/notification";
  import { getSecurityEvents, type SecurityEvent } from "$lib/utils/security";
  import {
    Activity,
    AlertCircle,
    AlertTriangle,
    CheckCircle,
    Clock,
    Database,
    Download,
    Eye,
    Info,
    Key,
    Lock,
    Monitor,
    RefreshCw,
    Server,
    Shield,
    Trash2,
    Unlock,
    Users,
    XCircle,
  } from "lucide-svelte";
  import { onDestroy, onMount } from "svelte";
  import { writable } from "svelte/store";

  let securityEvents: SecurityEvent[] = [];
  let filteredEvents: SecurityEvent[] = [];
  let selectedSeverity = "";
  let selectedType = "";
  let showDetails = new Set<number>();
  let refreshInterval: number | null = null;
  let loading = false;

  // Security metrics
  let criticalEvents = $derived(securityEvents.filter(
    (e) => e.severity === "critical"
  ).length);
  let highEvents = $derived(securityEvents.filter((e) => e.severity === "high").length);
  let recentEvents = $derived(securityEvents.filter(
    (e) => Date.now() - e.timestamp < 24 * 60 * 60 * 1000
  ).length);
  let loginAttempts = $derived(securityEvents.filter((e) => e.type === "login").length);
  let accessDeniedEvents = $derived(securityEvents.filter(
    (e) => e.type === "access_denied"
  ).length);

  // System status
  const systemHealth = writable({
    database: "healthy",
    authentication: "healthy",
    fileSystem: "healthy",
    network: "healthy",
  });

  onMount(() => {
    loadSecurityEvents();
    startAutoRefresh();
  });

  onDestroy(() => {
    if (refreshInterval) {
      clearInterval(refreshInterval);
}
  });

  function loadSecurityEvents() {
    loading = true;
    try {
      securityEvents = getSecurityEvents();
      filterEvents();
    } catch (error) {
      console.error("Failed to load security events:", error);
      notifications.add({
        type: "error",
        title: "Failed to Load Security Events",
        message: "Unable to retrieve security monitoring data.",
      });
    } finally {
      loading = false;
}}
  function filterEvents() {
    let filtered = [...securityEvents];

    if (selectedSeverity) {
      filtered = filtered.filter((e) => e.severity === selectedSeverity);
}
    if (selectedType) {
      filtered = filtered.filter((e) => e.type === selectedType);
}
    // Sort by timestamp (most recent first)
    filtered.sort((a, b) => b.timestamp - a.timestamp);

    filteredEvents = filtered;
}
  function startAutoRefresh() {
    refreshInterval = window.setInterval(() => {
      loadSecurityEvents();
    }, 30000); // Refresh every 30 seconds
}
  function toggleEventDetails(index: number) {
    if (showDetails.has(index)) {
      showDetails.delete(index);
    } else {
      showDetails.add(index);
}
    showDetails = showDetails;
}
  function clearAllEvents() {
    if (
      confirm(
        "Are you sure you want to clear all security events? This cannot be undone."
      )
    ) {
      localStorage.removeItem("security_events");
      securityEvents = [];
      filteredEvents = [];
      notifications.add({
        type: "success",
        title: "Security Events Cleared",
        message: "All security events have been cleared.",
      });
}}
  function exportEvents() {
    const dataStr = JSON.stringify(securityEvents, null, 2);
    const dataBlob = new Blob([dataStr], { type: "application/json" });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `security_events_${new Date().toISOString().split("T")[0]}.json`;
    link.click();
    URL.revokeObjectURL(url);
}
  function getSeverityIcon(severity: string) {
    switch (severity) {
      case "critical":
        return AlertTriangle;
      case "high":
        return AlertCircle;
      case "medium":
        return Info;
      case "low":
        return CheckCircle;
      default:
        return Info;
}}
  function getSeverityColor(severity: string) {
    switch (severity) {
      case "critical":
        return "text-error bg-error/10 border-error/20";
      case "high":
        return "text-warning bg-warning/10 border-warning/20";
      case "medium":
        return "text-info bg-info/10 border-info/20";
      case "low":
        return "text-success bg-success/10 border-success/20";
      default:
        return "text-base-content bg-base-200 border-base-300";
}}
  function getTypeIcon(type: string) {
    switch (type) {
      case "login":
        return Users;
      case "logout":
        return Unlock;
      case "access_denied":
        return Lock;
      case "suspicious_activity":
        return AlertTriangle;
      case "file_upload":
        return Database;
      case "data_export":
        return Download;
      default:
        return Activity;
}}
  function formatTimestamp(timestamp: number) {
    return new Date(timestamp).toLocaleString();
}
  function getSystemHealthIcon(status: string) {
    switch (status) {
      case "healthy":
        return CheckCircle;
      case "warning":
        return AlertCircle;
      case "error":
        return XCircle;
      default:
        return Info;
}}
  function getSystemHealthColor(status: string) {
    switch (status) {
      case "healthy":
        return "text-success";
      case "warning":
        return "text-warning";
      case "error":
        return "text-error";
      default:
        return "text-base-content";
}}
  // Reactive statements
  $: if (selectedSeverity || selectedType) {
    filterEvents();
}
</script>

<div class="container mx-auto px-4">
  <!-- Header -->
  <div
    class="container mx-auto px-4"
  >
    <div>
      <h2 class="container mx-auto px-4">
        <Shield class="container mx-auto px-4" />
        Security Monitoring
      </h2>
      <p class="container mx-auto px-4">
        Monitor system security events and health status
      </p>
    </div>

    <div class="container mx-auto px-4">
      <Button
        variant="outline"
        size="sm"
        onclick={() => loadSecurityEvents()}
        disabled={loading}
      >
        <RefreshCw class={`w-4 h-4 ${loading ? "animate-spin" : ""}`} />
        Refresh
      </Button>
      <Button variant="outline" size="sm" onclick={() => exportEvents()}>
        <Download class="container mx-auto px-4" />
        Export
      </Button>
      <Button
        variant="outline"
        size="sm"
        onclick={() => clearAllEvents()}
        class="container mx-auto px-4"
      >
        <Trash2 class="container mx-auto px-4" />
        Clear All
      </Button>
    </div>
  </div>

  <!-- Security Metrics -->
  <div class="container mx-auto px-4">
    <div class="container mx-auto px-4">
      <div class="container mx-auto px-4">
        <AlertTriangle class="container mx-auto px-4" />
      </div>
      <div class="container mx-auto px-4">Critical Events</div>
      <div class="container mx-auto px-4">{criticalEvents}</div>
      <div class="container mx-auto px-4">Requiring immediate attention</div>
    </div>

    <div class="container mx-auto px-4">
      <div class="container mx-auto px-4">
        <AlertCircle class="container mx-auto px-4" />
      </div>
      <div class="container mx-auto px-4">High Priority</div>
      <div class="container mx-auto px-4">{highEvents}</div>
      <div class="container mx-auto px-4">Need investigation</div>
    </div>

    <div class="container mx-auto px-4">
      <div class="container mx-auto px-4">
        <Activity class="container mx-auto px-4" />
      </div>
      <div class="container mx-auto px-4">Recent Events</div>
      <div class="container mx-auto px-4">{recentEvents}</div>
      <div class="container mx-auto px-4">Last 24 hours</div>
    </div>

    <div class="container mx-auto px-4">
      <div class="container mx-auto px-4">
        <Users class="container mx-auto px-4" />
      </div>
      <div class="container mx-auto px-4">Login Attempts</div>
      <div class="container mx-auto px-4">{loginAttempts}</div>
      <div class="container mx-auto px-4">{accessDeniedEvents} denied</div>
    </div>
  </div>

  <!-- System Health -->
  <div class="container mx-auto px-4">
    <h3 class="container mx-auto px-4">
      <Monitor class="container mx-auto px-4" />
      System Health
    </h3>

    <div class="container mx-auto px-4">
      <div class="container mx-auto px-4">
        <Database class="container mx-auto px-4" />
        <div class="container mx-auto px-4">
          <div class="container mx-auto px-4">Database</div>
          <div class="container mx-auto px-4">Connection status</div>
        </div>
        <svelte:component
          this={getSystemHealthIcon($systemHealth.database)}
          class="container mx-auto px-4"
        />
      </div>

      <div class="container mx-auto px-4">
        <Key class="container mx-auto px-4" />
        <div class="container mx-auto px-4">
          <div class="container mx-auto px-4">Authentication</div>
          <div class="container mx-auto px-4">Service status</div>
        </div>
        <svelte:component
          this={getSystemHealthIcon($systemHealth.authentication)}
          class="container mx-auto px-4"
        />
      </div>

      <div class="container mx-auto px-4">
        <Server class="container mx-auto px-4" />
        <div class="container mx-auto px-4">
          <div class="container mx-auto px-4">File System</div>
          <div class="container mx-auto px-4">Storage access</div>
        </div>
        <svelte:component
          this={getSystemHealthIcon($systemHealth.fileSystem)}
          class="container mx-auto px-4"
        />
      </div>

      <div class="container mx-auto px-4">
        <Activity class="container mx-auto px-4" />
        <div class="container mx-auto px-4">
          <div class="container mx-auto px-4">Network</div>
          <div class="container mx-auto px-4">Connectivity</div>
        </div>
        <svelte:component
          this={getSystemHealthIcon($systemHealth.network)}
          class="container mx-auto px-4"
        />
      </div>
    </div>
  </div>

  <!-- Security Events -->
  <div class="container mx-auto px-4">
    <div class="container mx-auto px-4">
      <div
        class="container mx-auto px-4"
      >
        <h3 class="container mx-auto px-4">Security Events</h3>

        <!-- Filters -->
        <div class="container mx-auto px-4">
          <select
            class="container mx-auto px-4"
            bind:value={selectedSeverity}
          >
            <option value="">All Severities</option>
            <option value="critical">Critical</option>
            <option value="high">High</option>
            <option value="medium">Medium</option>
            <option value="low">Low</option>
          </select>

          <select
            class="container mx-auto px-4"
            bind:value={selectedType}
          >
            <option value="">All Types</option>
            <option value="login">Login</option>
            <option value="logout">Logout</option>
            <option value="access_denied">Access Denied</option>
            <option value="suspicious_activity">Suspicious Activity</option>
            <option value="file_upload">File Upload</option>
            <option value="data_export">Data Export</option>
          </select>
        </div>
      </div>
    </div>

    <div class="container mx-auto px-4">
      {#if loading}
        <div class="container mx-auto px-4">
          <div class="container mx-auto px-4"></div>
          <span class="container mx-auto px-4">Loading security events...</span>
        </div>
      {:else if filteredEvents.length === 0}
        <div class="container mx-auto px-4">
          <Shield class="container mx-auto px-4" />
          <h4 class="container mx-auto px-4">
            No Security Events
          </h4>
          <p class="container mx-auto px-4">
            {securityEvents.length === 0
              ? "No security events have been recorded yet."
              : "No events match the selected filters."}
          </p>
        </div>
      {:else}
        <div class="container mx-auto px-4">
          {#each filteredEvents as event, index}
            <div
              class="container mx-auto px-4"
            >
              <div class="container mx-auto px-4">
                <div class="container mx-auto px-4">
                  <svelte:component
                    this={getSeverityIcon(event.severity)}
                    class="container mx-auto px-4"
                  />
                  <div class="container mx-auto px-4">
                    <div class="container mx-auto px-4">
                      <svelte:component
                        this={getTypeIcon(event.type)}
                        class="container mx-auto px-4"
                      />
                      <span class="container mx-auto px-4"
                        >{event.type.replace("_", " ")}</span
                      >
                      <div class="container mx-auto px-4">{event.severity}</div>
                    </div>

                    <div class="container mx-auto px-4">
                      <Clock class="container mx-auto px-4" />
                      {formatTimestamp(event.timestamp)}
                      {#if event.userId}
                        • User: {event.userId}
                      {/if}
                    </div>

                    {#if showDetails.has(index)}
                      <div
                        class="container mx-auto px-4"
                      >
                        {#if event.details}
                          <div>
                            <strong>Details:</strong>
                            <pre
                              class="container mx-auto px-4">{JSON.stringify(
                                event.details,
                                null,
                                2
                              )}</pre>
                          </div>
                        {/if}
                        {#if event.ipAddress}
                          <div>
                            <strong>IP Address:</strong>
                            {event.ipAddress}
                          </div>
                        {/if}
                        {#if event.userAgent}
                          <div>
                            <strong>User Agent:</strong>
                            {event.userAgent}
                          </div>
                        {/if}
                      </div>
                    {/if}
                  </div>
                </div>

                <Button
                  variant="ghost"
                  size="sm"
                  onclick={() => toggleEventDetails(index)}
                  class="container mx-auto px-4"
                >
                  <Eye class="container mx-auto px-4" />
                </Button>
              </div>
            </div>
          {/each}
        </div>
      {/if}
    </div>
  </div>
</div>


