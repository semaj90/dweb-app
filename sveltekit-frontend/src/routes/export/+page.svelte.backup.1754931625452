<script lang="ts">
  import { Button } from "$lib/components/ui/button";
  import Tooltip from "$lib/components/ui/Tooltip.svelte";
  import TooltipContent from "$lib/components/ui/TooltipContent.svelte";
  import TooltipTrigger from "$lib/components/ui/TooltipTrigger.svelte";
  import type { Case } from "$lib/types/index";
  import {
    AlertTriangle,
    Calendar,
    CheckCircle,
    Database,
    Download,
    FileText,
    Filter,
  } from "lucide-svelte";
  import { onMount } from "svelte";

  // Export state
  let exportLoading = false;
  let exportError: string | null = null;
  let exportSuccess = false;
  let availableCases: Case[] = [];

  // Export configuration
  let format: "json" | "csv" | "xml" = "json";
  let includeEvidence = true;
  let includeCases = true;
  let includeAnalytics = false;
  let selectedCaseIds: string[] = [];
  let dateFrom = "";
  let dateTo = "";

  onMount(() => {
    loadAvailableCases();
  });

  async function loadAvailableCases() {
    try {
      const response = await fetch("/api/cases");
      if (response.ok) {
        const data = await response.json();
        availableCases = data.cases || [];
}
    } catch (error) {
      console.error("Failed to load cases:", error);
}}
  async function exportData() {
    exportLoading = true;
    exportError = null;
    exportSuccess = false;

    try {
      const exportRequest = {
        format,
        includeEvidence,
        includeCases,
        includeAnalytics,
        dateRange:
          dateFrom || dateTo
            ? {
                from: dateFrom || undefined,
                to: dateTo || undefined,
}
            : undefined,
        caseIds: selectedCaseIds.length > 0 ? selectedCaseIds : undefined,
      };

      const response = await fetch("/api/export", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(exportRequest),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Export failed");
}
      // Get the filename from the response headers
      const contentDisposition = response.headers.get("Content-Disposition");
      const filename =
        contentDisposition?.match(/filename="(.+)"/)?.[1] || `export.${format}`;

      // Download the file
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      a.click();
      URL.revokeObjectURL(url);

      exportSuccess = true;
      setTimeout(() => (exportSuccess = false), 3000);
    } catch (error) {
      console.error("Export failed:", error);
      exportError = error instanceof Error ? error.message : "Export failed";
    } finally {
      exportLoading = false;
}}
  function toggleCaseSelection(caseId: string) {
    if (selectedCaseIds.includes(caseId)) {
      selectedCaseIds = selectedCaseIds.filter((id) => id !== caseId);
    } else {
      selectedCaseIds = [...selectedCaseIds, caseId];
}}
  function selectAllCases() {
    selectedCaseIds = availableCases.map((c) => c.id);
}
  function clearCaseSelection() {
    selectedCaseIds = [];
}
</script>

<svelte:head>
  <title>Data Export - Legal Analysis Platform</title>
  <meta
    name="description"
    content="Export legal cases, evidence, and analytics data"
  />
</svelte:head>

<div class="container mx-auto px-4">
  <!-- Header -->
  <header class="container mx-auto px-4">
    <div class="container mx-auto px-4">
      <div class="container mx-auto px-4">
        <Download class="container mx-auto px-4" />
        <div>
          <h1 class="container mx-auto px-4">Data Export</h1>
          <p class="container mx-auto px-4">
            Export cases, evidence, and analytics in multiple formats
          </p>
        </div>
      </div>
    </div>
  </header>

  <div class="container mx-auto px-4">
    <div class="container mx-auto px-4">
      <!-- Export Configuration -->
      <div class="container mx-auto px-4">
        <div class="container mx-auto px-4">
          <h2 class="container mx-auto px-4">
            <FileText class="container mx-auto px-4" />
            Export Configuration
          </h2>

          <!-- Format Selection -->
          <div class="container mx-auto px-4">
            <div class="container mx-auto px-4">
              Export Format
            </div>
            <div class="container mx-auto px-4">
              {#each [{ value: "json", label: "JSON", description: "Structured data format" }, { value: "csv", label: "CSV", description: "Spreadsheet compatible" }, { value: "xml", label: "XML", description: "Standard markup format" }] as formatOption}
                <Tooltip>
                  <TooltipTrigger asChild>
                    <button
                      class="container mx-auto px-4"
                      on:click={() =>
                        (format = formatOption.value as "json" | "csv" | "xml")}
                    >
                      <div class="container mx-auto px-4">{formatOption.label}</div>
                      <div class="container mx-auto px-4">
                        {formatOption.description}
                      </div>
                    </button>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>Export data in {formatOption.label} format</p>
                  </TooltipContent>
                </Tooltip>
              {/each}
            </div>
          </div>

          <!-- Data Selection -->
          <div class="container mx-auto px-4">
            <div class="container mx-auto px-4">
              Data to Include
            </div>
            <div class="container mx-auto px-4">
              <label class="container mx-auto px-4">
                <input
                  type="checkbox"
                  bind:checked={includeCases}
                  class="container mx-auto px-4"
                />
                <span class="container mx-auto px-4">Cases</span>
              </label>
              <label class="container mx-auto px-4">
                <input
                  type="checkbox"
                  bind:checked={includeEvidence}
                  class="container mx-auto px-4"
                />
                <span class="container mx-auto px-4">Evidence</span>
              </label>
              <label class="container mx-auto px-4">
                <input
                  type="checkbox"
                  bind:checked={includeAnalytics}
                  class="container mx-auto px-4"
                />
                <span class="container mx-auto px-4">Analytics & Statistics</span>
              </label>
            </div>
          </div>

          <!-- Date Range -->
          <div class="container mx-auto px-4">
            <label
              class="container mx-auto px-4"
            >
              <Calendar class="container mx-auto px-4" />
              Date Range (Optional)
            </label>
            <div class="container mx-auto px-4">
              <div>
                <label for="date-from" class="container mx-auto px-4"
                  >From</label
                >
                <input
                  id="date-from"
                  type="date"
                  bind:value={dateFrom}
                  class="container mx-auto px-4"
                />
              </div>
              <div>
                <label for="date-to" class="container mx-auto px-4"
                  >To</label
                >
                <input
                  id="date-to"
                  type="date"
                  bind:value={dateTo}
                  class="container mx-auto px-4"
                />
              </div>
            </div>
          </div>

          <!-- Case Selection -->
          <div class="container mx-auto px-4">
            <div class="container mx-auto px-4">
              <label
                class="container mx-auto px-4"
              >
                <Filter class="container mx-auto px-4" />
                Case Filter (Optional)
              </label>
              <div class="container mx-auto px-4">
                <Button
                  variant="outline"
                  size="sm"
                  on:click={() => selectAllCases()}
                >
                  Select All
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  on:click={() => clearCaseSelection()}
                >
                  Clear
                </Button>
              </div>
            </div>

            {#if availableCases.length > 0}
              <div
                class="container mx-auto px-4"
              >
                {#each availableCases as caseItem}
                  <label
                    class="container mx-auto px-4"
                  >
                    <input
                      type="checkbox"
                      checked={selectedCaseIds.includes(caseItem.id)}
                      on:change={() => toggleCaseSelection(caseItem.id)}
                      class="container mx-auto px-4"
                    />
                    <span class="container mx-auto px-4">
                      <span class="container mx-auto px-4">{caseItem.title}</span>
                      <span class="container mx-auto px-4">({caseItem.id})</span>
                    </span>
                  </label>
                {/each}
              </div>
              <p class="container mx-auto px-4">
                {selectedCaseIds.length} of {availableCases.length} cases selected
              </p>
            {:else}
              <p class="container mx-auto px-4">No cases available</p>
            {/if}
          </div>

          <!-- Error/Success Messages -->
          {#if exportError}
            <div class="container mx-auto px-4">
              <div class="container mx-auto px-4">
                <AlertTriangle
                  class="container mx-auto px-4"
                />
                <div>
                  <h4 class="container mx-auto px-4">Export Failed</h4>
                  <p class="container mx-auto px-4">{exportError}</p>
                </div>
              </div>
            </div>
          {/if}

          {#if exportSuccess}
            <div
              class="container mx-auto px-4"
            >
              <div class="container mx-auto px-4">
                <CheckCircle
                  class="container mx-auto px-4"
                />
                <div>
                  <h4 class="container mx-auto px-4">Export Successful</h4>
                  <p class="container mx-auto px-4">
                    Your data has been downloaded successfully.
                  </p>
                </div>
              </div>
            </div>
          {/if}

          <!-- Export Button -->
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                on:click={() => exportData()}
                disabled={exportLoading || (!includeCases && !includeEvidence)}
                class="container mx-auto px-4"
              >
                {#if exportLoading}
                  <div
                    class="container mx-auto px-4"
                  ></div>
                  Exporting...
                {:else}
                  <Download class="container mx-auto px-4" />
                  Export Data
                {/if}
              </Button>
            </TooltipTrigger>
            <TooltipContent>
              <p>Download the configured data export</p>
            </TooltipContent>
          </Tooltip>
        </div>
      </div>

      <!-- Export Summary -->
      <div class="container mx-auto px-4">
        <div class="container mx-auto px-4">
          <h3 class="container mx-auto px-4">
            <Database class="container mx-auto px-4" />
            Export Summary
          </h3>

          <div class="container mx-auto px-4">
            <div class="container mx-auto px-4">
              <div class="container mx-auto px-4">Format</div>
              <div class="container mx-auto px-4">{format.toUpperCase()}</div>
            </div>

            <div class="container mx-auto px-4">
              <div class="container mx-auto px-4">Data Types</div>
              <div class="container mx-auto px-4">
                {#if includeCases}
                  <div class="container mx-auto px-4">
                    <CheckCircle class="container mx-auto px-4" />
                    Cases
                  </div>
                {/if}
                {#if includeEvidence}
                  <div class="container mx-auto px-4">
                    <CheckCircle class="container mx-auto px-4" />
                    Evidence
                  </div>
                {/if}
                {#if includeAnalytics}
                  <div class="container mx-auto px-4">
                    <CheckCircle class="container mx-auto px-4" />
                    Analytics
                  </div>
                {/if}
              </div>
            </div>

            {#if dateFrom || dateTo}
              <div class="container mx-auto px-4">
                <div class="container mx-auto px-4">Date Range</div>
                <div class="container mx-auto px-4">
                  {dateFrom || "Beginning"} to {dateTo || "End"}
                </div>
              </div>
            {/if}

            {#if selectedCaseIds.length > 0}
              <div class="container mx-auto px-4">
                <div class="container mx-auto px-4">
                  Selected Cases
                </div>
                <div class="container mx-auto px-4">
                  {selectedCaseIds.length} case{selectedCaseIds.length !== 1
                    ? "s"
                    : ""} selected
                </div>
              </div>
            {/if}
          </div>

          <!-- Export Instructions -->
          <div class="container mx-auto px-4">
            <h4 class="container mx-auto px-4">Export Instructions</h4>
            <ul class="container mx-auto px-4">
              <li>• Select your preferred format</li>
              <li>• Choose data types to include</li>
              <li>• Optionally filter by date or cases</li>
              <li>• Click "Export Data" to download</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>
