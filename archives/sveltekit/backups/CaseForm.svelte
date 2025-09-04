<script lang="ts">
  import { goto } from "$app/navigation";
  import { Button } from "$lib/components/ui/button";
  import Card from "../../../lib/components/ui/Card.svelte";
  import Form from "../../../lib/components/ui/Form.svelte";
  import Input from "../../../lib/components/ui/Input.svelte";
  import { notifications } from "../../../lib/stores/notification";

  export const data = null;

  // Form validation
  const formOptions = {
    initialValues: {
      title: "",
      description: "",
      priority: "medium",
      assignedTo: "",
      dueDate: "",
      tags: "",
    },
    validators: {
      title: (value: string) => {
        if (!value || value.trim().length < 3) {
          return "Title must be at least 3 characters long";
        }
        if (value.length > 100) {
          return "Title must be less than 100 characters";
        }
        return null;
      },
      description: (value: string) => {
        if (!value || value.trim().length < 10) {
          return "Description must be at least 10 characters long";
        }
        return null;
      },
      priority: (value: string) => {
        if (!["low", "medium", "high", "urgent"].includes(value)) {
          return "Please select a valid priority level";
        }
        return null;
      },
      dueDate: (value: string) => {
        if (value && new Date(value) < new Date()) {
          return "Due date cannot be in the past";
        }
        return null;
      },
    },
    requiredFields: ["title", "description", "priority"],
  };

  let formApi: unknown;
  let isSubmitting = false;

  // Store form state
  let formValues: Record<string, any> = {};
  let formErrors: Record<string, string> = {};
  let isFormValid = false;
  let isFormDirty = false;

  // Handle form changes
  function handleFormChange(event: CustomEvent) {
    const { values } = event.detail;
    formValues = values;
    // Auto-save draft or other real-time updates
    console.log("Form values changed:", values);
  }

  // Update form state when formApi is available
  $: if (formApi) {
    // You can access formApi methods here if needed
  }

  async function handleSubmit(event: CustomEvent) {
    const { values, isValid } = event.detail;

    if (!isValid) {
      return;
    }

    isSubmitting = true;

    try {
      // You can either use the form action or API endpoint
      const response = await fetch("/api/cases", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(values),
      });

      if (response.ok) {
        const newCase = await response.json();
        notifications.success(
          "Case created successfully",
          `Case "${values.title}" has been created with ID ${newCase.id}`
        );
        goto(`/cases/${newCase.id}`);
      } else {
        const error = await response.json();
        notifications.error(
          "Failed to create case",
          error.message || "Please try again later"
        );
      }
    } catch (error) {
      console.error("Case creation error:", error);
      notifications.error(
        "Network error",
        "Unable to create case. Please check your connection and try again."
      );
    } finally {
      isSubmitting = false;
    }
  }

  function addTag() {
    const currentTags = formApi.getValues().tags || "";
    formApi.setField(
      "tags",
      currentTags + (currentTags ? ", " : "") + "New Tag"
    );
  }

  // Keyboard shortcuts
  function handleKeydown(event: KeyboardEvent) {
    if (event.ctrlKey || event.metaKey) {
      if (event.key === "s") {
        event.preventDefault();
        formApi?.submit();
      } else if (event.key === "r") {
        event.preventDefault();
        formApi?.reset();
      }
    }
  }
</script>

<svelte:window onkeydown={handleKeydown} />

<div class="mx-auto px-4 max-w-7xl">
  <div class="mx-auto px-4 max-w-7xl">
    <h1 class="mx-auto px-4 max-w-7xl">
      Create New Case
    </h1>
    <p class="mx-auto px-4 max-w-7xl">
      Fill out the form below to create a new legal case. All required fields
      must be completed.
    </p>
    <div class="mx-auto px-4 max-w-7xl">
      <p>
        💡 Tip: Use <kbd class="mx-auto px-4 max-w-7xl"
          >Ctrl+S</kbd
        >
        to save,
        <kbd class="mx-auto px-4 max-w-7xl">Ctrl+R</kbd>
        to reset
      </p>
    </div>
  </div>

  <Card variant="elevated" padding="lg">
    <Form
      bind:formApi
      options={formOptions}
      onsubmit={handleSubmit}
      onchange={handleFormChange}
      submitText="Create Case"
      submitVariant="primary"
      showResetButton={true}
      loading={isSubmitting}
      class="mx-auto px-4 max-w-7xl"
    >
      <div
        slot="default"
        let:form
        let:formApi
        let:values
        let:errors
        let:isValid
        let:isDirty
      >
        <!-- Basic Information -->
        <div class="mx-auto px-4 max-w-7xl">
          <h2
            class="mx-auto px-4 max-w-7xl"
          >
            Basic Information
          </h2>

          <div class="mx-auto px-4 max-w-7xl">
            <div class="mx-auto px-4 max-w-7xl">
              <Input
                label="Case Title"
                placeholder="Enter a descriptive title for the case"
                required
                value={formValues.title || ""}
                error={formErrors.title}
                data-icon="${1}"
                clearable
                oninput={(e) =>
                  formApi?.setField(
                    "title",
                    (e.target as HTMLInputElement)?.value
                  )}
                onblur={() => formApi?.touchField("title")}
              />
            </div>

            <div class="mx-auto px-4 max-w-7xl">
              <label
                for="case-description"
                class="mx-auto px-4 max-w-7xl"
              >
                Description <span class="mx-auto px-4 max-w-7xl">*</span>
              </label>
              <textarea
                id="case-description"
                class="mx-auto px-4 max-w-7xl"
                rows={${1"
                placeholder="Provide a detailed description of the case"
                value={values.description || ""}
                class:border-red-300={errors.description}
                class:border-green-300={values.description &&
                  !errors.description}
                oninput={(e) =>
                  formApi.setField(
                    "description",
                    (e.target as HTMLTextAreaElement)?.value
                  )}
                onblur={() => formApi.touchField("description")}
              ></textarea>
              {#if errors.description}
                <p class="mx-auto px-4 max-w-7xl">
                  {errors.description}
                </p>
              {/if}
            </div>

            <div>
              <label
                for="case-priority"
                class="mx-auto px-4 max-w-7xl"
              >
                Priority <span class="mx-auto px-4 max-w-7xl">*</span>
              </label>
              <select
                id="case-priority"
                class="mx-auto px-4 max-w-7xl"
                value={values.priority || "medium"}
                onchange={(e) =>
                  formApi.setField(
                    "priority",
                    (e.target as HTMLSelectElement)?.value
                  )}
                onblur={() => formApi.touchField("priority")}
              >
                <option value="low">🟢 Low</option>
                <option value="medium">🟡 Medium</option>
                <option value="high">🟠 High</option>
                <option value="urgent">🔴 Urgent</option>
              </select>
            </div>

            <div>
              <Input
                label="Due Date"
                type="date"
                value={values.dueDate || ""}
                error={errors.dueDate}
                data-icon="${1}"
                oninput={(e) =>
                  formApi.setField(
                    "dueDate",
                    (e.target as HTMLInputElement)?.value
                  )}
                onblur={() => formApi.touchField("dueDate")}
              />
            </div>
          </div>
        </div>

        <!-- Assignment -->
        <div class="mx-auto px-4 max-w-7xl">
          <h2
            class="mx-auto px-4 max-w-7xl"
          >
            Assignment & Tags
          </h2>

          <div class="mx-auto px-4 max-w-7xl">
            <div>
              <Input
                label="Assigned To"
                placeholder="Enter assignee email or name"
                value={values.assignedTo || ""}
                error={errors.assignedTo}
                data-icon="${1}"
                oninput={(e) =>
                  formApi.setField(
                    "assignedTo",
                    (e.target as HTMLInputElement)?.value
                  )}
                onblur={() => formApi.touchField("assignedTo")}
              />
            </div>

            <div>
              <div class="mx-auto px-4 max-w-7xl">
                <div class="mx-auto px-4 max-w-7xl">
                  <Input
                    label="Tags"
                    placeholder="Enter tags separated by commas"
                    value={values.tags || ""}
                    error={errors.tags}
                    data-icon="${1}"
                    clearable
                    oninput={(e) =>
                      formApi.setField(
                        "tags",
                        (e.target as HTMLInputElement)?.value
                      )}
                    onblur={() => formApi.touchField("tags")}
                  />
                </div>
                <Button
                  type="button"
                  variant="secondary"
                  size="md"
                  data-icon="${1}"
                  onclick={() => addTag()}
                >
                  Add
                </Button>
              </div>
            </div>
          </div>
        </div>

        <!-- Form Status -->
        <div
          class="mx-auto px-4 max-w-7xl"
        >
          <div class="mx-auto px-4 max-w-7xl">
            {#if isDirty}
              <span class="mx-auto px-4 max-w-7xl">
                <div class="mx-auto px-4 max-w-7xl"></div>
                Unsaved changes
              </span>
            {:else}
              <span class="mx-auto px-4 max-w-7xl">
                <div class="mx-auto px-4 max-w-7xl"></div>
                All changes saved
              </span>
            {/if}
          </div>

          <div class="mx-auto px-4 max-w-7xl">
            Valid: {isValid ? "✅" : "❌"} | Fields: {Object.keys(values)
              .length} | Errors: {Object.keys(errors).length}
          </div>
        </div>
      </div>
    </Form>
  </Card>
</div>

<style>
  kbd {
    font-family:
      ui-monospace, SFMono-Regular, "SF Mono", Menlo, Monaco, Consolas,
      "Liberation Mono", "Courier New", monospace;
    font-size: 0.75rem;
  }
</style>