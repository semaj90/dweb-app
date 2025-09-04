<!-- Enhanced Case Form with proper schema mapping -->
<script lang="ts">
  import { notifications } from "$lib/stores/notification";
  import type { User } from "$lib/types/user";
  import { createEventDispatcher } from "svelte";

  import type { Case } from "$lib/types/index";
  
  export let case_: Case | null = null; // Edit mode if provided
  export let user: User;

  const dispatch = createEventDispatcher();

  // Form data matching the database schema
  let formData = {
    title: case_?.title || "",
    description: case_?.description || "",
    caseNumber: case_?.caseNumber || "",
    name: case_?.name || "",
    incidentDate: case_?.incidentDate
      ? new Date(case_.incidentDate).toISOString().split("T")[0]
      : "",
    location: case_?.location || "",
    priority: case_?.priority || "medium",
    status: case_?.status || "open",
    category: case_?.category || "",
    dangerScore: case_?.dangerScore || 0,
    estimatedValue: case_?.estimatedValue || "",
    jurisdiction: case_?.jurisdiction || "",
    leadProsecutor: case_?.leadProsecutor || user?.id || "",
    assignedTeam: case_?.assignedTeam || [],
    tags: case_?.tags || [],
    metadata: case_?.metadata || {},
  };

  let loading = false;
  let errors: Record<string, string> = {};

  // Form validation
  function validateForm() {
    errors = {};

    if (!formData.title.trim()) {
      errors.title = "Title is required";
    }

    if (!formData.caseNumber.trim()) {
      errors.caseNumber = "Case number is required";
    }

    if (formData.dangerScore < 0 || formData.dangerScore > 10) {
      errors.dangerScore = "Danger score must be between 0 and 10";
    }

    if (formData.estimatedValue && isNaN(Number(formData.estimatedValue))) {
      errors.estimatedValue = "Estimated value must be a number";
    }

    return Object.keys(errors).length === 0;
  }

  // Handle form submission
  async function handleSubmit() {
    if (!validateForm()) {
      notifications.add({
        type: "error",
        title: "Validation Error",
        message: "Please fix the form errors before submitting.",
      });
      return;
    }

    loading = true;

    try {
      // Prepare data for API - match schema exactly
      const apiData = {
        title: formData.title.trim(),
        description: formData.description.trim(),
        caseNumber: formData.caseNumber.trim(),
        name: formData.name.trim() || formData.title.trim(), // Use title as fallback
        incidentDate: formData.incidentDate || null,
        location: formData.location.trim(),
        priority: formData.priority,
        status: formData.status,
        category: formData.category.trim(),
        dangerScore: Number(formData.dangerScore),
        estimatedValue: formData.estimatedValue
          ? Number(formData.estimatedValue)
          : null,
        jurisdiction: formData.jurisdiction.trim(),
        leadProsecutor: formData.leadProsecutor || user.id,
        assignedTeam: formData.assignedTeam,
        tags: formData.tags,
        metadata: {
          ...formData.metadata,
          formVersion: "2.0",
          lastModified: new Date().toISOString(),
        },
      };

      // Defensive: always check for valid API data before fetch
      if (!apiData.title || !apiData.caseNumber) {
        throw new Error("Missing required fields");
      }

      const url = case_ ? `/api/cases/${case_.id}` : "/api/cases";
      const method = case_ ? "PUT" : "POST";

      const response = await fetch(url, {
        method,
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(apiData),
      });

      // Defensive: handle non-JSON error responses
      let savedCase;
      try {
        savedCase = await response.json();
      } catch (e) {
        throw new Error("Server returned invalid response");
      }

      if (!response.ok) {
        throw new Error(savedCase?.error || "Failed to save case");
      }

      notifications.add({
        type: "success",
        title: case_ ? "Case Updated" : "Case Created",
        message: `Case "${savedCase.title}" has been ${case_ ? "updated" : "created"} successfully.`,
      });

      dispatch(case_ ? "updated" : "created", savedCase);
    } catch (error) {
      console.error("Error saving case:", error);
      notifications.add({
        type: "error",
        title: "Save Error",
        message:
          error instanceof Error
            ? error.message
            : "Failed to save case. Please try again.",
      });
    } finally {
      loading = false;
    }
  }

  // Handle tag management
  function addTag() {
    const tagInput = document.getElementById("new-tag") as HTMLInputElement;
    const newTag = tagInput?.value.trim();

    if (newTag && !formData.tags.includes(newTag)) {
      formData.tags = [...formData.tags, newTag];
      tagInput.value = "";
    }
  }

  function removeTag(tag: string) {
    formData.tags = formData.tags.filter((t) => t !== tag);
  }

  // Handle team assignment
  function addTeamMember() {
    const memberInput = document.getElementById(
      "new-member"
    ) as HTMLInputElement;
    const newMember = memberInput?.value.trim();

    if (newMember && !formData.assignedTeam.includes(newMember)) {
      formData.assignedTeam = [...formData.assignedTeam, newMember];
      memberInput.value = "";
    }
  }

  function removeTeamMember(member: string) {
    formData.assignedTeam = formData.assignedTeam.filter((m) => m !== member);
  }
</script>

<form onsubmit|preventDefault={handleSubmit} class="mx-auto px-4 max-w-7xl">
  <div class="mx-auto px-4 max-w-7xl">
    <!-- Basic Information -->
    <section class="mx-auto px-4 max-w-7xl">
      <h3>Basic Information</h3>

      <div class="mx-auto px-4 max-w-7xl">
        <label for="title" class="mx-auto px-4 max-w-7xl">Case Title</label>
        <input
          id="title"
          type="text"
          bind:value={formData.title}
          placeholder="Enter case title"
          class:error={errors.title}
          required
        />
        {#if errors.title}
          <span class="mx-auto px-4 max-w-7xl">{errors.title}</span>
        {/if}
      </div>

      <div class="mx-auto px-4 max-w-7xl">
        <label for="caseNumber" class="mx-auto px-4 max-w-7xl">Case Number</label>
        <input
          id="caseNumber"
          type="text"
          bind:value={formData.caseNumber}
          placeholder="e.g., CASE-2024-001"
          class:error={errors.caseNumber}
          required
        />
        {#if errors.caseNumber}
          <span class="mx-auto px-4 max-w-7xl">{errors.caseNumber}</span>
        {/if}
      </div>

      <div class="mx-auto px-4 max-w-7xl">
        <label for="name">Case Name (Optional)</label>
        <input
          id="name"
          type="text"
          bind:value={formData.name}
          placeholder="Alternative case name"
        />
      </div>

      <div class="mx-auto px-4 max-w-7xl">
        <label for="description">Description</label>
        <textarea
          id="description"
          bind:value={formData.description}
          placeholder="Detailed case description"
          rows={${1"
        ></textarea>
      </div>
    </section>

    <!-- Case Details -->
    <section class="mx-auto px-4 max-w-7xl">
      <h3>Case Details</h3>

      <div class="mx-auto px-4 max-w-7xl">
        <div class="mx-auto px-4 max-w-7xl">
          <label for="priority">Priority</label>
          <select id="priority" bind:value={formData.priority}>
            <option value="low">Low</option>
            <option value="medium">Medium</option>
            <option value="high">High</option>
            <option value="urgent">Urgent</option>
          </select>
        </div>

        <div class="mx-auto px-4 max-w-7xl">
          <label for="status">Status</label>
          <select id="status" bind:value={formData.status}>
            <option value="open">Open</option>
            <option value="active">Active</option>
            <option value="pending">Pending</option>
            <option value="closed">Closed</option>
            <option value="archived">Archived</option>
          </select>
        </div>
      </div>

      <div class="mx-auto px-4 max-w-7xl">
        <label for="category">Category</label>
        <input
          id="category"
          type="text"
          bind:value={formData.category}
          placeholder="e.g., Criminal, Civil, Administrative"
        />
      </div>

      <div class="mx-auto px-4 max-w-7xl">
        <div class="mx-auto px-4 max-w-7xl">
          <label for="dangerScore">Danger Score (0-10)</label>
          <input
            id="dangerScore"
            type="number"
            min="0"
            max="10"
            bind:value={formData.dangerScore}
            class:error={errors.dangerScore}
          />
          {#if errors.dangerScore}
            <span class="mx-auto px-4 max-w-7xl">{errors.dangerScore}</span>
          {/if}
        </div>

        <div class="mx-auto px-4 max-w-7xl">
          <label for="estimatedValue">Estimated Value ($)</label>
          <input
            id="estimatedValue"
            type="number"
            step="0.01"
            bind:value={formData.estimatedValue}
            placeholder="0.00"
            class:error={errors.estimatedValue}
          />
          {#if errors.estimatedValue}
            <span class="mx-auto px-4 max-w-7xl">{errors.estimatedValue}</span>
          {/if}
        </div>
      </div>
    </section>

    <!-- Location & Timeline -->
    <section class="mx-auto px-4 max-w-7xl">
      <h3>Location & Timeline</h3>

      <div class="mx-auto px-4 max-w-7xl">
        <label for="incidentDate">Incident Date</label>
        <input
          id="incidentDate"
          type="date"
          bind:value={formData.incidentDate}
        />
      </div>

      <div class="mx-auto px-4 max-w-7xl">
        <label for="location">Location</label>
        <input
          id="location"
          type="text"
          bind:value={formData.location}
          placeholder="Incident location"
        />
      </div>

      <div class="mx-auto px-4 max-w-7xl">
        <label for="jurisdiction">Jurisdiction</label>
        <input
          id="jurisdiction"
          type="text"
          bind:value={formData.jurisdiction}
          placeholder="e.g., City County, State Police"
        />
      </div>
    </section>

    <!-- Team & Tags -->
    <section class="mx-auto px-4 max-w-7xl">
      <h3>Team & Tags</h3>

      <!-- Assigned Team -->
      <div class="mx-auto px-4 max-w-7xl">
        <label for="new-member">Assigned Team</label>
        <div class="mx-auto px-4 max-w-7xl">
          <input
            id="new-member"
            type="text"
            placeholder="Add team member ID"
            onkeydown={(e) =>
              e.key === "Enter" && (e.preventDefault(), addTeamMember())}
          />
          <button type="button" onclick={() => addTeamMember()}>Add</button>
        </div>

        {#if formData.assignedTeam.length > 0}
          <div class="mx-auto px-4 max-w-7xl">
            {#each formData.assignedTeam as member}
              <span class="mx-auto px-4 max-w-7xl">
                {member}
                <button type="button" onclick={() => removeTeamMember(member)}
                  >×</button
                >
              </span>
            {/each}
          </div>
        {/if}
      </div>

      <!-- Tags -->
      <div class="mx-auto px-4 max-w-7xl">
        <label for="new-tag">Tags</label>
        <div class="mx-auto px-4 max-w-7xl">
          <input
            id="new-tag"
            type="text"
            placeholder="Add tag"
            onkeydown={(e) =>
              e.key === "Enter" && (e.preventDefault(), addTag())}
          />
          <button type="button" onclick={() => addTag()}>Add</button>
        </div>

        {#if formData.tags.length > 0}
          <div class="mx-auto px-4 max-w-7xl">
            {#each formData.tags as tag}
              <span class="mx-auto px-4 max-w-7xl">
                {tag}
                <button type="button" onclick={() => removeTag(tag)}>×</button>
              </span>
            {/each}
          </div>
        {/if}
      </div>
    </section>
  </div>

  <!-- Form Actions -->
  <div class="mx-auto px-4 max-w-7xl">
    <button type="button" onclick={() => dispatch("cancel")}> Cancel </button>
    <button type="submit" disabled={loading} class="mx-auto px-4 max-w-7xl">
      {#if loading}
        Saving...
      {:else}
        {case_ ? "Update Case" : "Create Case"}
      {/if}
    </button>
  </div>
</form>

<style>
  .enhanced-case-form {
    max-width: 800px;
    margin: 0 auto;
    background: white;
    border-radius: 8px;
    padding: 2rem;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  }

  .form-grid {
    display: grid;
    gap: 2rem;
  }

  .form-section {
    border: 1px solid #e5e7eb;
    border-radius: 6px;
    padding: 1.5rem;
  }

  .form-section h3 {
    margin: 0 0 1rem 0;
    color: #374151;
    font-size: 1.1rem;
    font-weight: 600;
    border-bottom: 1px solid #e5e7eb;
    padding-bottom: 0.5rem;
  }

  .field-group {
    margin-bottom: 1.5rem;
  }

  .field-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
  }

  label {
    display: block;
    margin-bottom: 0.5rem;
    font-weight: 500;
    color: #374151;
  }

  label.required::after {
    content: "*";
    color: #ef4444;
    margin-left: 0.25rem;
  }

  input,
  select,
  textarea {
    width: 100%;
    padding: 0.75rem;
    border: 1px solid #d1d5db;
    border-radius: 6px;
    font-size: 1rem;
    transition:
      border-color 0.2s,
      box-shadow 0.2s;
  }

  input:focus,
  select:focus,
  textarea:focus {
    outline: none;
    border-color: #3b82f6;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
  }

  input.error {
    border-color: #ef4444;
  }

  .field-error {
    display: block;
    margin-top: 0.25rem;
    font-size: 0.875rem;
    color: #ef4444;
  }

  .tag-input {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 0.5rem;
  }

  .tag-input input {
    flex: 1;
  }

  .tag-input button {
    padding: 0.75rem 1rem;
    background: #3b82f6;
    color: white;
    border: none;
    border-radius: 6px;
    cursor: pointer;
    font-size: 0.875rem;
    transition: background-color 0.2s;
  }

  .tag-input button:hover {
    background: #2563eb;
  }

  .tags-list {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
  }

  .tag {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.25rem 0.75rem;
    background: #e5e7eb;
    border-radius: 9999px;
    font-size: 0.875rem;
    color: #374151;
  }

  .tag button {
    background: none;
    border: none;
    cursor: pointer;
    font-size: 1rem;
    color: #6b7280;
    padding: 0;
    width: 1rem;
    height: 1rem;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 50%;
    transition: background-color 0.2s;
  }

  .tag button:hover {
    background: #d1d5db;
    color: #374151;
  }

  .form-actions {
    display: flex;
    justify-content: flex-end;
    gap: 1rem;
    margin-top: 2rem;
    padding-top: 1.5rem;
    border-top: 1px solid #e5e7eb;
  }

  .form-actions button {
    padding: 0.75rem 1.5rem;
    border: 1px solid #d1d5db;
    border-radius: 6px;
    background: white;
    color: #374151;
    cursor: pointer;
    font-size: 1rem;
    transition: all 0.2s;
  }

  .form-actions button:hover {
    background: #f9fafb;
  }

  .form-actions button.primary {
    background: #3b82f6;
    color: white;
    border-color: #3b82f6;
  }

  .form-actions button.primary:hover {
    background: #2563eb;
    border-color: #2563eb;
  }

  .form-actions button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  @media (max-width: 768px) {
    .enhanced-case-form {
      padding: 1rem;
    }

    .field-row {
      grid-template-columns: 1fr;
    }

    .form-actions {
      flex-direction: column;
    }
  }
</style>
