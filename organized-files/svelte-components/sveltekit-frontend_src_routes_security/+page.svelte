<script lang="ts">
  import SecurityMonitoringDashboard from "$lib/components/security/SecurityMonitoringDashboard.svelte";
  import { Button } from "$lib/components/ui/button";
  import { ArrowLeft, Shield } from "lucide-svelte";
</script>

<svelte:head>
  <title>Security Dashboard - WardenNet Detective Mode</title>
  <meta
    name="description"
    content="Monitor system security events, health status, and security metrics"
  />
</svelte:head>

<div class="space-y-4">
  <!-- Header -->
  <div class="space-y-4">
    <a href="/dashboard" class="space-y-4">
      <Button variant="outline" size="sm">
        <ArrowLeft class="space-y-4" />
        Back to Dashboard
      </Button>
    </a>
    <div>
      <h1 class="space-y-4">
        <Shield class="space-y-4" />
        Security Dashboard
      </h1>
      <p class="space-y-4">
        Comprehensive security monitoring and event management
      </p>
    </div>
  </div>

  <!-- Security Monitoring Dashboard -->
  <SecurityMonitoringDashboard />
</div>
