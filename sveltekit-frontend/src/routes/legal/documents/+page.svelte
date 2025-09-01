<script lang="ts">
  import { onMount } from "svelte";

  interface Document {
    id: string;
    title: string;
    type?: string;
    created: string;
    status: "draft" | "review" | "final";
  }
let documents = $state<Document[] >([]);
let loading = $state(true);

  // Mock data
  const mockDocuments: Document[] = [
    {
      id: "1",
      title: "Case File #2024-001",
      type: "Case Report",
      created: "2024-01-15",
      status: "final",
    },
    {
      id: "2",
      title: "Evidence Analysis Report",
      type: "Analysis",
      created: "2024-01-18",
      status: "review",
    },
    {
      id: "3",
      title: "Legal Brief Draft",
      type: "Brief",
      created: "2024-01-20",
      status: "draft",
    },
  ];

  onMount(async () => {
    try {
      await new Promise((resolve) => setTimeout(resolve, 500));
      documents = mockDocuments;
    } catch (error) {
      console.error("Error loading documents:", error);
    } finally {
      loading = false;
}
  });

  function getStatusClass(status: string): string {
    switch (status) {
      case "draft":
        return "bg-yellow-100 text-yellow-800";
      case "review":
        return "bg-blue-100 text-blue-800";
      case "final":
        return "bg-green-100 text-green-800";
      default:
        return "bg-gray-100 text-gray-800";
}}
</script>

<svelte:head>
  <title>Legal Documents - Legal Case Management</title>
</svelte:head>

<div class="space-y-4">
  <div class="space-y-4">
    <div>
      <h1 class="space-y-4">
        Legal Documents
      </h1>
      <p class="space-y-4">
        Manage case files, reports, and legal documentation
      </p>
    </div>

    <button
      class="space-y-4"
      type="button"
    >
      + New Document
    </button>
  </div>

  {#if loading}
    <div class="space-y-4">
      <div
        class="space-y-4"
      ></div>
      <span class="space-y-4"
        >Loading documents...</span
      >
    </div>
  {:else if documents.length === 0}
    <div class="space-y-4">
      <div class="space-y-4">📄</div>
      <h3 class="space-y-4">
        No documents found
      </h3>
      <p class="space-y-4">
        Create your first document to get started
      </p>
    </div>
  {:else}
    <div class="space-y-4">
      {#each documents as doc (doc.id)}
        <div
          class="space-y-4"
        >
          <div class="space-y-4">
            <h3
              class="space-y-4"
            >
              {doc.title}
            </h3>
            <button
              type="button"
              class="space-y-4"
              aria-label="Document options"
            >
              <svg class="space-y-4" fill="currentColor" viewBox="0 0 20 20">
                <path
                  d="M10 6a2 2 0 110-4 2 2 0 010 4zM10 12a2 2 0 110-4 2 2 0 010 4zM10 18a2 2 0 110-4 2 2 0 010 4z"
                />
              </svg>
            </button>
          </div>

          <div class="space-y-4">
            <div class="space-y-4">
              <span class="space-y-4">
                <svg class="space-y-4" fill="currentColor" viewBox="0 0 20 20">
                  <path
                    fill-rule="evenodd"
                    d="M4 4a2 2 0 012-2h8a2 2 0 012 2v12a2 2 0 01-2 2H6a2 2 0 01-2-2V4zm2 0v12h8V4H6z"
                    clip-rule="evenodd"
                  />
                </svg>
                {doc.type}
              </span>
              <span class="space-y-4">
                <svg class="space-y-4" fill="currentColor" viewBox="0 0 20 20">
                  <path
                    fill-rule="evenodd"
                    d="M6 2a1 1 0 00-1 1v1H4a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V6a2 2 0 00-2-2h-1V3a1 1 0 10-2 0v1H7V3a1 1 0 00-1-1zm0 5a1 1 0 000 2h8a1 1 0 100-2H6z"
                    clip-rule="evenodd"
                  />
                </svg>
                {doc.created}
              </span>
            </div>

            <span
              class={`px-2 py-1 rounded-full text-xs font-medium ${getStatusClass(doc.status)}`}
            >
              {doc.status.charAt(0).toUpperCase() + doc.status.slice(1)}
            </span>
          </div>
        </div>
      {/each}
    </div>
  {/if}
</div>

<style>
  /* @unocss-include */
  .line-clamp-2 {
    display: -webkit-box;
    -webkit-line-clamp: 2;
    line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
}
</style>
