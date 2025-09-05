<script lang="ts">
  import { Button } from "$lib/components/ui/button";
  import { createFormStore, type FormOptions } from '$lib/stores/form';
  import { notifications } from '$lib/stores/notification';

  interface Props {
    options?: FormOptions;
    class?: string;
    novalidate?: boolean;
    autocomplete?: "on" | "off";
    submitText?: string;
    submitVariant?: "primary" | "secondary" | "outline" | "danger" | "success" | "warning" | "info" | "nier";
    showSubmitButton?: boolean;
    submitFullWidth?: boolean;
    resetText?: string;
    showResetButton?: boolean;
    loading?: boolean;
    formApi?: any; // Add bindable formApi prop
    onsubmit?: (event: { values: Record<string, any>; isValid: boolean }) => void;
    onreset?: () => void;
    onchange?: (event: { values: Record<string, any> }) => void;
  }

  let {
    options = {},
    submitText = "Submit",
    submitVariant = "primary",
    showSubmitButton = true,
    submitFullWidth = false,
    resetText = "Reset",
    showResetButton = false,
    loading = false,
    formApi = $bindable(), // Make formApi bindable
    onsubmit,
    onreset,
    onchange,
    ...restProps
  } = $props();

  // Create form store
  const form = createFormStore({
    ...options,
    onSubmit: async (values) => {
      onsubmit?.({ values, isValid: true });
      if (options.onSubmit) {
        await options.onSubmit(values);
}
    },
  });

  // Subscribe to form values for change events using $effect
  $effect(() => {
    if ($form.isDirty) {
      onchange?.({ values: $form.values });
    }
  });
  async function handleSubmit(event: SubmitEvent) {
    event.preventDefault();

    const isValid = await form.submit();
    if (!isValid) {
      notifications.error(
        "Form validation failed",
        "Please correct the errors and try again."
      );
}}
  function handleReset() {
    form.reset();
    onreset?.();
}
  // Update formApi when form changes using $effect
  $effect(() => {
    if (formApi !== undefined) {
      formApi = {
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
    }
  });
</script>

<form
  onsubmit={handleSubmit}
  onreset={handleReset}
  class="space-y-6 {restProps.class || ''}"
  novalidate={restProps.novalidate}
  autocomplete={restProps.autocomplete}
  {...restProps}
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
    <div class="flex gap-3 justify-end">
      {#if showResetButton}
        <Button
          type="reset"
          variant="secondary"
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
          loading={$form.isSubmitting}
          class={submitFullWidth ? "w-full" : ""}
        >
          {submitText}
        </Button>
      {/if}
    </div>
  {/if}

  <!-- Form status -->
  {#if $form.submitCount > 0 && Object.keys($form.errors).length > 0}
    <div class="mt-4 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-700 rounded-lg">
      <div class="flex items-start gap-3">
        <div class="text-red-600 dark:text-red-400 mt-0.5">⚠</div>
        <div class="flex-1">
          <h3 class="text-sm font-medium text-red-800 dark:text-red-200 mb-2">
            Please correct the following errors:
          </h3>
          <ul class="text-sm text-red-700 dark:text-red-300 space-y-1">
            {#each Object.entries($form.errors) as [field, error]}
              <li class="list-disc list-inside">{error}</li>
            {/each}
          </ul>
        </div>
      </div>
    </div>
  {/if}
</form>
