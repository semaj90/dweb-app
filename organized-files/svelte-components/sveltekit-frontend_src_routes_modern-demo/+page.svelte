<script lang="ts">
  import EvidenceCard from "$lib/components/evidence/EvidenceCard.svelte";
  import ExpandGrid from "$lib/components/ui/ExpandGrid.svelte";
  import GoldenLayout from "$lib/components/ui/GoldenLayout.svelte";
  import SmartTextarea from "$lib/components/ui/SmartTextarea.svelte";
  import { citationStore } from "$lib/stores/citations";
  import type { Evidence } from "$lib/stores/report";
  import { onMount } from "svelte";

  let sidebarCollapsed = false;
  let textareaValue =
    "Try typing # or pressing Ctrl+K to open the command menu!\n\nThis smart textarea integrates with your citations and commands.";

  // Sample evidence data
  const sampleEvidence = [
    {
      id: "1",
      title: "Crime Scene Photo 1",
      description: "Primary evidence showing the scene of the incident",
      type: "image" as const,
      url: "https://via.placeholder.com/200x150/4f46e5/white?text=Evidence+1",
      tags: ["crime scene", "primary evidence"],
      metadata: { size: 2048576, format: "JPEG" },
      createdAt: new Date("2024-01-15"),
      updatedAt: new Date("2024-01-15"),
    },
    {
      id: "2",
      title: "Witness Statement Video",
      description: "Interview with key witness recorded on-site",
      type: "video" as const,
      url: "https://via.placeholder.com/200x150/10b981/white?text=Video+Evidence",
      tags: ["witness", "interview"],
      metadata: { size: 50331648, format: "MP4" },
      createdAt: new Date("2024-01-16"),
      updatedAt: new Date("2024-01-16"),
    },
    {
      id: "3",
      title: "Audio Recording",
      description: "Phone call recording related to the case",
      type: "audio" as const,
      tags: ["audio", "phone call"],
      metadata: { size: 1048576, format: "WAV" },
      createdAt: new Date("2024-01-17"),
      updatedAt: new Date("2024-01-17"),
    },
    {
      id: "4",
      title: "Legal Document",
      description: "Contract related to the dispute",
      type: "document" as const,
      tags: ["contract", "legal"],
      metadata: { size: 524288, format: "PDF" },
      createdAt: new Date("2024-01-18"),
      updatedAt: new Date("2024-01-18"),
    },
    {
      id: "5",
      title: "External Link",
      description: "Reference to online legal database",
      type: "link" as const,
      url: "https://example.com/legal-database",
      tags: ["reference", "database"],
      metadata: { format: "URL" },
      createdAt: new Date("2024-01-19"),
      updatedAt: new Date("2024-01-19"),
    },
    {
      id: "6",
      title: "Additional Evidence",
      description: "Supporting documentation for the case",
      type: "document" as const,
      tags: ["supporting", "documentation"],
      metadata: { size: 256000, format: "DOCX" },
      createdAt: new Date("2024-01-20"),
      updatedAt: new Date("2024-01-20"),
    },
  ];

  onMount(() => {
    // Load citations when component mounts
    citationStore.loadCitations();
  });

  function handleEvidenceView(evidence: Evidence) {
    console.log("View evidence:", evidence);
}
  function handleEvidenceEdit(evidence: Evidence) {
    console.log("Edit evidence:", evidence);
}
  function handleEvidenceDelete(evidence: Evidence) {
    console.log("Delete evidence:", evidence);
}
  function handleEvidenceDownload(evidence: Evidence) {
    console.log("Download evidence:", evidence);
}
  function handleGridExpand(event: CustomEvent) {
    console.log("Grid expand:", event.detail);
}
  function handleTextareaInput(event: CustomEvent) {
    console.log("Textarea input:", event.detail);
}
  function handleCommandInsert(event: CustomEvent) {
    console.log("Command insert:", event.detail);
}
</script>

<svelte:head>
  <title>Modern SvelteKit Components Demo</title>
  <meta
    name="description"
    content="Demonstration of modern SvelteKit components with Melt UI"
  />
</svelte:head>

<div class="space-y-4">
  <header class="space-y-4">
    <h1>Modern SvelteKit Components Demo</h1>
    <p>
      Showcasing modern layout techniques, command menus, and interactive
      components
    </p>
  </header>

  <GoldenLayout
    ratio="golden"
    collapsible={true}
    bind:collapsed={sidebarCollapsed}
    ontoggle={(e) => console.log("Sidebar toggled:", e.detail.collapsed)}
  >
    <div class="space-y-4" slot="default">
      <section class="space-y-4">
        <h2>📝 Smart Textarea with Command Menu</h2>
        <p>
          This textarea integrates a command menu system. Try typing <code
            >#</code
          >
          or pressing
          <kbd>Ctrl+K</kbd> to open the command menu. You can insert citations, navigate
          to different parts of the app, or insert common text snippets.
        </p>

        <SmartTextarea
          bind:value={textareaValue}
          placeholder="Type # for commands or Ctrl+K for command menu..."
          rows={6}
          oninput={handleTextareaInput}
          on:commandInsert={handleCommandInsert}
        />
      </section>

      <section class="space-y-4">
        <h2>🎨 Hover-Expanding Grid</h2>
        <p>
          This grid starts with 1 column and expands to 3 columns on hover.
          Perfect for showcasing cards or gallery items with smooth transitions.
        </p>

        <ExpandGrid
          columns={1}
          expandedColumns={3}
          expandDuration="0.4s"
          onexpand={handleGridExpand}
        >
          {#each sampleEvidence as evidence}
            <div class="space-y-4">
              <EvidenceCard
                {evidence}
                onView={handleEvidenceView}
                onEdit={handleEvidenceEdit}
                onDelete={handleEvidenceDelete}
                onDownload={handleEvidenceDownload}
                expandOnHover={true}
                compact={false}
              />
            </div>
          {/each}
        </ExpandGrid>
      </section>

      <section class="space-y-4">
        <h2>⚡ Fast SvelteKit Navigation</h2>
        <p>
          These links use standard <code>&lt;a&gt;</code> tags but SvelteKit automatically
          intercepts them for fast, SPA-style navigation without page refreshes.
        </p>

        <div class="space-y-4">
          <a href="/cases" class="space-y-4">📁 Cases</a>
          <a href="/evidence" class="space-y-4">🔍 Evidence</a>
          <a href="/search" class="space-y-4">🔎 Search</a>
          <a href="/ai-assistant" class="space-y-4">🤖 AI Assistant</a>
          <a href="/reports" class="space-y-4">📊 Reports</a>
        </div>
      </section>

      <section class="space-y-4">
        <h2>🎯 Features Demonstrated</h2>
        <div class="space-y-4">
          <div class="space-y-4">
            <h3>Golden Ratio Layout</h3>
            <p>
              Sidebar uses the golden ratio (1.618:1) for optimal visual balance
            </p>
          </div>
          <div class="space-y-4">
            <h3>Command Menu</h3>
            <p>Slash commands and keyboard shortcuts for power users</p>
          </div>
          <div class="space-y-4">
            <h3>Smart Components</h3>
            <p>
              Context-aware interactions with real-time search and filtering
            </p>
          </div>
          <div class="space-y-4">
            <h3>Responsive Design</h3>
            <p>Adapts to different screen sizes with mobile-first approach</p>
          </div>
        </div>
      </section>
    </div>

    <div class="space-y-4" slot="sidebar">
      <h3>📋 Citations</h3>
      <p>Recent citations from your library:</p>

      <div class="space-y-4">
        {#each citationStore.getRecentCitations($citationStore, 5) as citation}
          <div class="space-y-4">
            <div class="space-y-4">{citation.title}</div>
            <div class="space-y-4">
              {citation.source || citation.author}
            </div>
            <div class="space-y-4">{citation.date}</div>
          </div>
        {/each}
      </div>

      <h3>🔧 Keyboard Shortcuts</h3>
      <div class="space-y-4">
        <div class="space-y-4">
          <kbd>Ctrl</kbd> + <kbd>K</kbd>
          <span>Open command menu</span>
        </div>
        <div class="space-y-4">
          <kbd>#</kbd>
          <span>Trigger command menu</span>
        </div>
        <div class="space-y-4">
          <kbd>Ctrl</kbd> + <kbd>\</kbd>
          <span>Toggle sidebar</span>
        </div>
        <div class="space-y-4">
          <kbd>Ctrl</kbd> + <kbd>1-4</kbd>
          <span>Switch tabs</span>
        </div>
      </div>

      <h3>🎨 CSS Features</h3>
      <ul class="space-y-4">
        <li>CSS Grid with dynamic columns</li>
        <li>Flexbox golden ratio layouts</li>
        <li>Smooth transitions and animations</li>
        <li>Hover effects and micro-interactions</li>
        <li>Responsive breakpoints</li>
        <li>CSS custom properties</li>
      </ul>
    </div>
  </GoldenLayout>
</div>

<style>
  /* @unocss-include */
  .demo-page {
    height: 100vh;
    overflow: hidden
    background: #f8fafc;
}
  .demo-header {
    padding: 2rem;
    text-align: center
    background: #ffffff;
    border-bottom: 1px solid #e2e8f0;
}
  .demo-header h1 {
    margin: 0 0 0.5rem 0;
    color: #3b82f6;
    font-size: 2.5rem;
}
  .demo-header p {
    margin: 0;
    color: #6b7280;
    font-size: 1.125rem;
}
  .main-content {
    padding: 2rem;
    overflow-y: auto
    height: 100%;
}
  .demo-section {
    margin-bottom: 3rem;
}
  .demo-section h2 {
    margin: 0 0 1rem 0;
    color: #111827;
    font-size: 1.5rem;
}
  .demo-section p {
    margin: 0 0 1.5rem 0;
    color: #6b7280;
    line-height: 1.6;
}
  .navigation-demo {
    display: flex
    flex-wrap: wrap
    gap: 1rem;
}
  .demo-link {
    display: inline-flex;
    align-items: center
    padding: 0.75rem 1.5rem;
    background: #3b82f6;
    color: white
    text-decoration: none
    border-radius: 0.5rem;
    font-weight: 500;
    transition: all 0.2s ease;
}
  .demo-link:hover {
    background: #2563eb;
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
}
  .features-grid {
    display: grid
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 1.5rem;
    margin-top: 1.5rem;
}
  .feature-card {
    padding: 1.5rem;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 0.75rem;
    transition: all 0.2s ease;
}
  .feature-card:hover {
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    transform: translateY(-2px);
}
  .feature-card h3 {
    margin: 0 0 0.5rem 0;
    color: #3b82f6;
    font-size: 1.125rem;
}
  .feature-card p {
    margin: 0;
    color: #6b7280;
    font-size: 0.875rem;
}
  .sidebar-content {
    height: 100%;
    overflow-y: auto
}
  .sidebar-content h3 {
    margin: 0 0 1rem 0;
    color: #111827;
    font-size: 1.125rem;
}
  .citations-list {
    margin-bottom: 2rem;
}
  .citation-item {
    padding: 0.75rem;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 0.5rem;
    margin-bottom: 0.5rem;
}
  .citation-title {
    font-weight: 600;
    color: #111827;
    font-size: 0.875rem;
    margin-bottom: 0.25rem;
}
  .citation-source {
    color: #6b7280;
    font-size: 0.75rem;
    margin-bottom: 0.25rem;
}
  .citation-date {
    color: #9ca3af;
    font-size: 0.75rem;
}
  .shortcuts-list {
    margin-bottom: 2rem;
}
  .shortcut-item {
    display: flex
    align-items: center
    justify-content: space-between;
    padding: 0.5rem 0;
    border-bottom: 1px solid #e2e8f0;
}
  .shortcut-item:last-child {
    border-bottom: none
}
  .shortcut-item kbd {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 0.25rem;
    padding: 0.125rem 0.375rem;
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--pico-color, #111827);
}
  .shortcut-item span {
    color: var(--pico-muted-color, #6b7280);
    font-size: 0.875rem;
}
  .features-list {
    list-style: none
    padding: 0;
    margin: 0;
}
  .features-list li {
    padding: 0.5rem 0;
    color: var(--pico-muted-color, #6b7280);
    font-size: 0.875rem;
    border-bottom: 1px solid var(--pico-border-color, #e2e8f0);
}
  .features-list li:last-child {
    border-bottom: none
}
  .features-list li::before {
    content: "✓";
    color: var(--pico-primary, #3b82f6);
    font-weight: bold
    margin-right: 0.5rem;
}
  code {
    background: var(--pico-card-sectioning-background-color, #f1f5f9);
    padding: 0.125rem 0.375rem;
    border-radius: 0.25rem;
    font-size: 0.875rem;
    color: var(--pico-primary, #3b82f6);
}
  kbd {
    background: var(--pico-card-background-color, #ffffff);
    border: 1px solid var(--pico-border-color, #e2e8f0);
    border-radius: 0.25rem;
    padding: 0.125rem 0.375rem;
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--pico-color, #111827);
}
  /* Responsive design */
  @media (max-width: 768px) {
    .demo-header {
      padding: 1rem;
}
    .demo-header h1 {
      font-size: 2rem;
}
    .main-content {
      padding: 1rem;
}
    .features-grid {
      grid-template-columns: 1fr;
}
    .navigation-demo {
      flex-direction: column
}
    .demo-link {
      justify-content: center
}
}
</style>
