<!-- Legal Frameworks Demo - Pico CSS + UnoCSS + Melt UI -->
<script lang="ts">
  import AskAI from "$lib/components/ai/AskAI.svelte";
  import LegalDocumentEditor from "$lib/components/editor/LegalDocumentEditor.svelte";
  import WysiwygEditor from "$lib/components/editor/WysiwygEditor.svelte";
  import { Tabs } from "bits-ui";
  import {
    BookOpen,
    Brain,
    FileText,
    MessageCircle,
    Scale,
    Settings,
  } from "lucide-svelte";
  import { onMount } from "svelte";

  // Active tab state and demo data
  let activeTab = "document-editor";
  let sampleContent = `MOTION TO DISMISS

Comes now the Defendant, by and through undersigned counsel, and respectfully moves this Honorable Court to dismiss the Plaintiff's Complaint pursuant to Federal Rule of Civil Procedure 12(b)(6) for failure to state a claim upon which relief can be granted.

I. STATEMENT OF FACTS

The Plaintiff filed a complaint alleging breach of contract. However, the complaint fails to...`;

  // Handle component events
  function handleSave(event: CustomEvent) {
    console.log("Document saved:", event.detail);
    // TODO: Save to backend
}
  function handleAIResponse(event: CustomEvent) {
    console.log("AI Response:", event.detail);
}
  onMount(() => {
    console.log("Legal Frameworks Demo loaded");
  });
</script>

<svelte:head>
  <title>Legal AI Frameworks Demo - Pico CSS + UnoCSS + Melt UI</title>
</svelte:head>

<!-- Main Container with Pico CSS semantic structure -->
<div class="space-y-4" data-theme="light">
  <!-- Header -->
  <header class="space-y-4">
    <div class="space-y-4">
      <nav class="space-y-4">
        <div class="space-y-4">
          <Scale class="space-y-4" />
          <div>
            <h1 class="space-y-4">Legal AI Demo</h1>
            <p class="space-y-4">
              Pico CSS + UnoCSS + Melt UI Integration
            </p>
          </div>
        </div>

        <div class="space-y-4">
          <span class="space-y-4">Frameworks:</span>
          <span
            class="space-y-4"
            >Pico CSS</span
          >
          <span
            class="space-y-4"
            >UnoCSS</span
          >
          <span
            class="space-y-4"
            >Melt UI</span
          >
        </div>
      </nav>
    </div>
  </header>

  <!-- Main Content -->
  <main class="space-y-4">
    <div class="space-y-4">
      <!-- Framework Integration Showcase -->
      <section class="space-y-4">
        <div class="space-y-4">
          <h2 class="space-y-4">
            Framework Integration Showcase
          </h2>
          <div class="space-y-4">
            <!-- Pico CSS Demo -->
            <article class="space-y-4">
              <h3 class="space-y-4">Pico CSS</h3>
              <p class="space-y-4">
                Semantic HTML styling with minimal classes
              </p>

              <!-- Pico CSS styled form -->
              <form class="space-y-4">
                <fieldset>
                  <label for="client-name">Client Name</label>
                  <input
                    type="text"
                    id="client-name"
                    placeholder="Enter client name"
                  />
                </fieldset>

                <fieldset>
                  <label for="case-type">Case Type</label>
                  <select id="case-type">
                    <option>Criminal Defense</option>
                    <option>Civil Litigation</option>
                    <option>Corporate Law</option>
                  </select>
                </fieldset>

                <fieldset>
                  <label for="notes">Case Notes</label>
                  <textarea
                    id="notes"
                    rows="4"
                    placeholder="Enter case notes..."
                  ></textarea>
                </fieldset>

                <button type="submit" class="space-y-4">Save Case</button>
              </form>
            </article>

            <!-- UnoCSS Demo -->
            <article class="space-y-4">
              <h3 class="space-y-4">UnoCSS</h3>
              <p class="space-y-4">
                Utility-first styling with legal-specific classes
              </p>

              <div class="space-y-4">
                <div class="space-y-4">
                  <div class="space-y-4">42 U.S.C. § 1983</div>
                  <p class="space-y-4">Civil action for deprivation of rights</p>
                </div>

                <div class="space-y-4">
                  <strong>Key Evidence:</strong> Security camera footage from 10:30
                  PM
                </div>

                <div class="space-y-4">
                  <strong>AI Context:</strong> Similar cases show a 78% success rate
                  for this type of motion.
                </div>

                <div class="space-y-4">
                  <button class="space-y-4">
                    <FileText class="space-y-4" />
                    Draft Motion
                  </button>
                  <button class="space-y-4">
                    <BookOpen class="space-y-4" />
                    Research
                  </button>
                </div>
              </div>
            </article>

            <!-- Melt UI Demo -->
            <article class="space-y-4">
              <h3 class="space-y-4">Melt UI</h3>
              <p class="space-y-4">Accessible interactive components</p>

              <div class="space-y-4">
                <div
                  class="space-y-4"
                >
                  <span class="space-y-4">Case Status</span>
                  <select class="space-y-4">
                    <option>Active</option>
                    <option>Pending</option>
                    <option>Closed</option>
                  </select>
                </div>

                <div class="space-y-4">
                  <div class="space-y-4">
                    <span class="space-y-4"
                      >AI Analysis</span
                    >
                    <Brain class="space-y-4" />
                  </div>
                  <p class="space-y-4">
                    Based on similar cases, this motion has a high probability
                    of success.
                  </p>
                </div>

                <button
                  class="space-y-4"
                >
                  <MessageCircle class="space-y-4" />
                  <span>Open AI Assistant</span>
                </button>
              </div>
            </article>
          </div>
        </div>
      </section>

      <!-- Component Demos with Bits UI Tabs -->
      <section class="space-y-4">
        <Tabs.Root bind:value={activeTab} class="space-y-4">
          <!-- Tab List -->
          <Tabs.List class="space-y-4">
            <Tabs.Trigger value="document-editor" class="space-y-4">
              <FileText class="space-y-4" />
              Document Editor
            </Tabs.Trigger>
            <Tabs.Trigger value="wysiwyg-editor" class="space-y-4">
              <Settings class="space-y-4" />
              WYSIWYG Editor
            </Tabs.Trigger>
            <Tabs.Trigger value="ai-assistant" class="space-y-4">
              <Brain class="space-y-4" />
              AI Assistant
            </Tabs.Trigger>
          </Tabs.List>

          <!-- Tab Content -->
          <Tabs.Content value="document-editor" class="space-y-4">
            <div class="space-y-4">
              <h3 class="space-y-4">Legal Document Editor</h3>
              <p class="space-y-4">
                Full-featured document editor with AI assistance, citations, and
                collaborative features.
              </p>
              <LegalDocumentEditor
                documentType="motion"
                title="Motion to Dismiss"
                onsave={handleSave}
                on:aiRequest={handleAIResponse}
              />
            </div>
          </Tabs.Content>

          <Tabs.Content value="wysiwyg-editor" class="space-y-4">
            <div class="space-y-4">
              <h3 class="space-y-4">
                WYSIWYG Editor with Hugerte
              </h3>
              <p class="space-y-4">
                Rich text editor with legal-specific formatting and AI
                integration.
              </p>
              <WysiwygEditor
                content={sampleContent}
                enableAI={true}
                enableCitation={true}
                height="400px"
                onchange={(e) => console.log("Content changed:", e.detail)}
              />
            </div>
          </Tabs.Content>

          <Tabs.Content value="ai-assistant" class="space-y-4">
            <div class="space-y-4">
              <h3 class="space-y-4">AI Assistant</h3>
              <p class="space-y-4">
                Intelligent legal AI with vector search and contextual
                responses.
              </p>
              <AskAI
                caseId="demo-case"
                evidenceIds={["evidence-1", "evidence-2"]}
                placeholder="Ask about legal procedures, case law, or document drafting..."
                showReferences={true}
                enableVoiceInput={true}
                onresponse={handleAIResponse}
              />
            </div>
          </Tabs.Content>
        </Tabs.Root>
      </section>

      <!-- Technical Details -->
      <section class="space-y-4">
        <div class="space-y-4">
          <h2 class="space-y-4">Technical Implementation</h2>
          <div class="space-y-4">
            <div>
              <h3 class="space-y-4">CSS Framework Stack</h3>
              <ul class="space-y-4">
                <li><strong>Pico CSS:</strong> Semantic HTML base styling</li>
                <li>
                  <strong>UnoCSS:</strong> Utility classes with legal theme
                </li>
                <li><strong>Custom CSS:</strong> Legal-specific components</li>
                <li><strong>CSS Variables:</strong> Consistent theming</li>
              </ul>
            </div>

            <div>
              <h3 class="space-y-4">
                Component Architecture
              </h3>
              <ul class="space-y-4">
                <li>
                  <strong>Melt UI:</strong> Accessible interactive primitives
                </li>
                <li><strong>Svelte:</strong> Reactive component framework</li>
                <li>
                  <strong>TypeScript:</strong> Type safety and intellisense
                </li>
                <li>
                  <strong>Local LLM:</strong> Privacy-focused AI integration
                </li>
              </ul>
            </div>
          </div>

          <div class="space-y-4">
            <h4 class="space-y-4">
              Benefits of This Stack
            </h4>
            <ul class="space-y-4">
              <li>✅ Semantic HTML with minimal CSS classes</li>
              <li>✅ Utility-first styling for rapid development</li>
              <li>✅ Accessible interactive components</li>
              <li>✅ Legal-specific design system</li>
              <li>✅ Privacy-focused local AI processing</li>
            </ul>
          </div>
        </div>
      </section>
    </div>
  </main>
</div>

<style>
  /* @unocss-include */
  .demo-container {
    min-height: 100vh;
    background: #fafafa;
}
  .demo-header {
    background: white
    border-bottom: 1px solid #e5e7eb;
    position: sticky
    top: 0;
    z-index: 10;
}
  .demo-main {
    padding: 2rem 0;
}
  :global(.framework-demo) {
    padding: 1rem;
    border: 1px solid #e5e7eb;
    border-radius: 0.5rem;
    background: #fff;
}
  :global(.tab-trigger) {
    padding: 0.75rem 1rem;
    font-size: 1rem;
    font-weight: 500;
    color: var(--legal-neutral);
    transition: color 0.2s;
    border-bottom: 2px solid transparent;
    display: flex
    align-items: center
}
  :global(.tab-trigger.active) {
    color: var(--primary-600);
    border-bottom: 2px solid var(--primary-600);
}
  :global(.tab-content) {
    display: none
}
  :global(.tab-content[data-state="active"]) {
    display: block
}
  /* Ensure Pico CSS form styles work properly */
  :global(form fieldset) {
    border: none
    padding: 0;
    margin: 0 0 1rem 0;
}
  :global(form fieldset:last-child) {
    margin-bottom: 0;
}
</style>
