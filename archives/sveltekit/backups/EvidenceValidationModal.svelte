<script lang="ts">
  import Button from "$lib/components/ui/Button.svelte";
  import { Dialog as DialogPrimitive } from "bits-ui";
  import {
    AlertTriangle,
    CheckCircle,
    Edit3,
    Save,
    Tag,
    XCircle,
  } from "lucide-svelte";
  import { createEventDispatcher } from "svelte";
  import type { Evidence } from "../../../lib/stores/evidence-store";

  export let open: boolean = false;
  export let evidence: Evidence | null = null;
  export let aiEvent: unknown = null; // Specific AI analysis event to validate

  const dispatch = createEventDispatcher();

  let validationChoice: "approve" | "reject" | null = null;
  let feedback: string = "";
  let corrections = {
    summary: "",
    tags: [] as string[],
    evidenceType: "",
    analysis: "",
  };
  let isSubmitting = false;
  let showCorrections = false;

  // Initialize corrections with current AI analysis
  $: if (evidence && open) {
    corrections = {
      summary: evidence.aiSummary || "",
      tags: evidence.aiTags || [],
      evidenceType: evidence.evidenceType || "",
      analysis: evidence.aiAnalysis?.analysis || "",
    };
  }

  function handleValidationChoice(choice: "approve" | "reject") {
    validationChoice = choice;
    showCorrections = choice === "reject";
  }

  function addTag() {
    const tagInput = document.getElementById("new-tag") as HTMLInputElement;
    const newTag = tagInput?.value.trim();
    if (newTag && !corrections.tags.includes(newTag)) {
      corrections.tags = [...corrections.tags, newTag];
      tagInput.value = "";
    }
  }

  function removeTag(tagToRemove: string) {
    corrections.tags = corrections.tags.filter((tag) => tag !== tagToRemove);
  }

  async function submitValidation() {
    if (!evidence || !validationChoice) return;

    isSubmitting = true;

    try {
      const payload = {
        evidenceId: evidence.id,
        eventId: aiEvent?.id || null,
        valid: validationChoice === "approve",
        feedback: feedback.trim() || null,
        corrections: validationChoice === "reject" ? corrections : null,
      };

      const response = await fetch("/api/evidence/validate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const result = await response.json();

      if (result.success) {
        dispatch("validated", {
          evidence,
          validation: result.validation,
          updatedAnalysis: result.updatedAnalysis,
        });

        // Reset form
        validationChoice = null;
        feedback = "";
        showCorrections = false;
        open = false;
      } else {
        console.error("Validation failed:", result.error);
        alert("Failed to submit validation. Please try again.");
      }
    } catch (error) {
      console.error("Validation submission error:", error);
      alert(
        "Failed to submit validation. Please check your connection and try again."
      );
    } finally {
      isSubmitting = false;
    }
  }

  function closeModal() {
    validationChoice = null;
    feedback = "";
    showCorrections = false;
    open = false;
  }
</script>

<DialogPrimitive.Root bind:open>
  <DialogPrimitive.Content
    class="mx-auto px-4 max-w-7xl"
  >
    <div class="mx-auto px-4 max-w-7xl">
      <!-- Header -->
      <div class="mx-auto px-4 max-w-7xl">
        <div>
          <DialogPrimitive.Title class="mx-auto px-4 max-w-7xl">
            Validate AI Analysis
          </DialogPrimitive.Title>
          <DialogPrimitive.Description class="mx-auto px-4 max-w-7xl">
            Review and validate the AI-generated analysis for this evidence
          </DialogPrimitive.Description>
        </div>
        <DialogPrimitive.Close let:builder>
          <Button
            {...builder}
            variant="ghost"
            size="sm"
            onclick={() => closeModal()}
          >
            ×
          </Button>
        </DialogPrimitive.Close>
      </div>

      {#if evidence}
        <div class="mx-auto px-4 max-w-7xl">
          <!-- Evidence Info -->
          <div class="mx-auto px-4 max-w-7xl">
            <h3 class="mx-auto px-4 max-w-7xl">Evidence: {evidence.title}</h3>
            <p class="mx-auto px-4 max-w-7xl">
              {evidence.description || "No description"}
            </p>
          </div>

          <!-- AI Analysis to Validate -->
          <div class="mx-auto px-4 max-w-7xl">
            <h4 class="mx-auto px-4 max-w-7xl">
              <AlertTriangle class="mx-auto px-4 max-w-7xl" />
              AI Analysis
            </h4>

            {#if evidence.aiSummary}
              <div class="mx-auto px-4 max-w-7xl">
                <p class="mx-auto px-4 max-w-7xl">Summary:</p>
                <p class="mx-auto px-4 max-w-7xl">{evidence.aiSummary}</p>
              </div>
            {/if}

            {#if evidence.aiTags && evidence.aiTags.length > 0}
              <div class="mx-auto px-4 max-w-7xl">
                <p class="mx-auto px-4 max-w-7xl">Suggested Tags:</p>
                <div class="mx-auto px-4 max-w-7xl">
                  {#each evidence.aiTags as tag}
                    <span
                      class="mx-auto px-4 max-w-7xl"
                    >
                      {tag}
                    </span>
                  {/each}
                </div>
              </div>
            {/if}

            <div>
              <p class="mx-auto px-4 max-w-7xl">Evidence Type:</p>
              <p class="mx-auto px-4 max-w-7xl">{evidence.evidenceType}</p>
            </div>

            {#if aiEvent}
              <div class="mx-auto px-4 max-w-7xl">
                <p class="mx-auto px-4 max-w-7xl">Specific Event:</p>
                <p class="mx-auto px-4 max-w-7xl">
                  {aiEvent.analysis || aiEvent.text}
                </p>
                {#if aiEvent.timestamp}
                  <p class="mx-auto px-4 max-w-7xl">
                    Timestamp: {aiEvent.timestamp}
                  </p>
                {/if}
              </div>
            {/if}
          </div>

          <!-- Validation Question -->
          <div class="mx-auto px-4 max-w-7xl">
            <h4 class="mx-auto px-4 max-w-7xl">Is this AI analysis accurate?</h4>

            <div class="mx-auto px-4 max-w-7xl">
              <Button
                variant={validationChoice === "approve" ? "default" : "outline"}
                class="mx-auto px-4 max-w-7xl"
                onclick={() => handleValidationChoice("approve")}
              >
                <CheckCircle class="mx-auto px-4 max-w-7xl" />
                Yes, it's accurate
              </Button>

              <Button
                variant={validationChoice === "reject" ? "danger" : "outline"}
                class="mx-auto px-4 max-w-7xl"
                onclick={() => handleValidationChoice("reject")}
              >
                <XCircle class="mx-auto px-4 max-w-7xl" />
                No, needs correction
              </Button>
            </div>
          </div>

          <!-- Feedback Section -->
          {#if validationChoice}
            <div class="mx-auto px-4 max-w-7xl">
              <div>
                <label for="feedback" class="mx-auto px-4 max-w-7xl">
                  Additional Feedback (Optional)
                </label>
                <textarea
                  id="feedback"
                  bind:value={feedback}
                  placeholder="Add any additional comments or context..."
                  class="mx-auto px-4 max-w-7xl"
                  rows={${1"
                ></textarea>
              </div>
            </div>
          {/if}

          <!-- Corrections Section -->
          {#if showCorrections}
            <div
              class="mx-auto px-4 max-w-7xl"
            >
              <h4 class="mx-auto px-4 max-w-7xl">
                <Edit3 class="mx-auto px-4 max-w-7xl" />
                Provide Corrections
              </h4>

              <!-- Summary Correction -->
              <div>
                <label
                  for="corrected-summary"
                  class="mx-auto px-4 max-w-7xl"
                >
                  Corrected Summary
                </label>
                <textarea
                  id="corrected-summary"
                  bind:value={corrections.summary}
                  placeholder="Enter the correct summary..."
                  class="mx-auto px-4 max-w-7xl"
                  rows={${1"
                ></textarea>
              </div>

              <!-- Evidence Type Correction -->
              <div>
                <label
                  for="corrected-type"
                  class="mx-auto px-4 max-w-7xl"
                >
                  Corrected Evidence Type
                </label>
                <select
                  id="corrected-type"
                  bind:value={corrections.evidenceType}
                  class="mx-auto px-4 max-w-7xl"
                >
                  <option value="document">Document</option>
                  <option value="image">Image</option>
                  <option value="video">Video</option>
                  <option value="audio">Audio</option>
                  <option value="pdf">PDF</option>
                  <option value="other">Other</option>
                </select>
              </div>

              <!-- Tags Correction -->
              <div>
                <label for="new-tag" class="mx-auto px-4 max-w-7xl"
                  >Corrected Tags</label
                >

                <!-- Current tags -->
                {#if corrections.tags.length > 0}
                  <div class="mx-auto px-4 max-w-7xl">
                    {#each corrections.tags as tag}
                      <span
                        class="mx-auto px-4 max-w-7xl"
                      >
                        {tag}
                        <button
                          type="button"
                          onclick={() => removeTag(tag)}
                          class="mx-auto px-4 max-w-7xl"
                        >
                          ×
                        </button>
                      </span>
                    {/each}
                  </div>
                {/if}

                <!-- Add new tag -->
                <div class="mx-auto px-4 max-w-7xl">
                  <input
                    id="new-tag"
                    type="text"
                    placeholder="Add a tag..."
                    class="mx-auto px-4 max-w-7xl"
                    onkeydown={(e) =>
                      e.key === "Enter" && (e.preventDefault(), addTag())}
                  />
                  <Button
                    type="button"
                    variant="secondary"
                    size="sm"
                    onclick={() => addTag()}
                  >
                    <Tag class="mx-auto px-4 max-w-7xl" />
                  </Button>
                </div>
              </div>
            </div>
          {/if}
        </div>

        <!-- Footer -->
        <div class="mx-auto px-4 max-w-7xl">
          <Button
            variant="ghost"
            onclick={() => closeModal()}
            disabled={isSubmitting}
          >
            Cancel
          </Button>

          <Button
            onclick={() => submitValidation()}
            disabled={!validationChoice || isSubmitting}
            class="mx-auto px-4 max-w-7xl"
          >
            {#if isSubmitting}
              <div
                class="mx-auto px-4 max-w-7xl"
              ></div>
              Submitting...
            {:else}
              <Save class="mx-auto px-4 max-w-7xl" />
              Submit Validation
            {/if}
          </Button>
        </div>
      {/if}
    </div>
  </DialogPrimitive.Content>
</DialogPrimitive.Root>
