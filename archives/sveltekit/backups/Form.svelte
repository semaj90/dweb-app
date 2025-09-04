<script lang="ts">
  import { Button } from "$lib/components/ui/button";
  import { createEventDispatcher } from "svelte";
  import { createFormStore, type FormOptions } from "../../../lib/stores/form";
  import { notifications } from "../../../lib/stores/notification";

  interface $$Props {
    options?: FormOptions;
    class?: string;
    novalidate?: boolean;
    autocomplete?: "on" | "off";
    submitText?: string;
    submitVariant?: "primary" | "secondary" | "outline" | "ghost" | "danger";
    showSubmitButton?: boolean;
    submitFullWidth?: boolean;
    resetText?: string;
    showResetButton?: boolean;
    loading?: boolean;
  }

  export let options: NonNullable<$$Props["options"]> = {};
  export let submitText: NonNullable<$$Props["submitText"]> = "Submit";
  export let submitVariant: NonNullable<$$Props["submitVariant"]> = "primary";
  export let showSubmitButton: NonNullable<$$Props["showSubmitButton"]> = true;
  export let submitFullWidth: NonNullable<$$Props["submitFullWidth"]> = false;
  export let resetText: NonNullable<$$Props["resetText"]> = "Reset";
  export let showResetButton: NonNullable<$$Props["showResetButton"]> = false;
  export let loading: NonNullable<$$Props["loading"]> = false;

  const dispatch = createEventDispatcher<{
    submit: { values: Record<string, any>; isValid: boolean };
    reset: void;
    change: { values: Record<string, any> };
  }>();

  // Create form store
  const form = createFormStore({
    ...options,
    onSubmit: async (values) => {
      dispatch("submit", { values, isValid: true });
      if (options.onSubmit) {
        await options.onSubmit(values);
      }
    },
  });

  // Subscribe to form values for change events
  $: if ($form.isDirty) {
    dispatch("change", { values: $form.values });
  }

  async function handleSubmit(event: SubmitEvent) {
    event.preventDefault();

    const isValid = await form.submit();
    if (!isValid) {
      notifications.error(
        "Form validation failed",
        "Please correct the errors and try again."
      );
    }
  }

  function handleReset() {
    form.reset();
    dispatch("reset");
  }

  // Expose form methods for parent components
  export const formApi = {
    setField: form.setField,
    touchField: form.touchField,
    validate: form.validate,
    submit: form.submit,
    reset: form.reset,
    addField: form.addField,
    removeField: form.removeField,
    values: form.values,
    errors: form.errors,
  };
</script>

<form
  onsubmit={handleSubmit}
  on:reset={handleReset}
  class="mx-auto px-4 max-w-7xl"
  novalidate={$$props.novalidate}
  autocomplete={$$props.autocomplete}
  {...$$restProps}
>
  <!-- Form content -->
  <slot
    {form}
    {formApi}
    values={$form.values}
    errors={$form.errors}
    isValid={$form.isValid}
    isDirty={$form.isDirty}
  />

  <!-- Form actions -->
  {#if showSubmitButton || showResetButton}
    <div
      class="mx-auto px-4 max-w-7xl"
    >
      {#if showResetButton}
        <Button
          type="reset"
          variant="ghost"
          disabled={!$form.isDirty || $form.isSubmitting || loading}
          class={submitFullWidth ? "w-full" : ""}
        >
          {resetText}
        </Button>
      {/if}

      {#if showSubmitButton}
        <Button
          type="submit"
          variant={submitVariant}
          disabled={!$form.isValid}
          loading={$form.isSubmitting || loading}
          class={submitFullWidth ? "w-full" : ""}
        >
          {submitText}
        </Button>
      {/if}
    </div>
  {/if}

  <!-- Form status -->
  {#if $form.submitCount > 0 && Object.keys($form.errors).length > 0}
    <div
      class="mx-auto px-4 max-w-7xl"
    >
      <div class="mx-auto px-4 max-w-7xl">
        <iconify-icon
          data-icon="${1}"
          class="mx-auto px-4 max-w-7xl"
        ></iconify-icon>
        <div class="mx-auto px-4 max-w-7xl">
          <h3 class="mx-auto px-4 max-w-7xl">
            Please correct the following errors:
          </h3>
          <ul
            class="mx-auto px-4 max-w-7xl"
          >
            {#each Object.entries($form.errors) as [field, error]}
              <li>{error}</li>
            {/each}
          </ul>
        </div>
      </div>
    </div>
  {/if}
</form>
