<script lang="ts">
  import { Button } from "$lib/components/ui/button";
  import { createEventDispatcher } from "svelte";
  import { createFormStore, type FormOptions } from '$lib/stores/form';
  import { notifications } from '$lib/stores/notification';

  interface $$Props {
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
    loading?: boolean;}
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
        await options.onSubmit(values);}
    },
  });

  // Subscribe to form values for change events
  $: if ($form.isDirty) {
    dispatch("change", { values: $form.values });}
  async function handleSubmit(event: SubmitEvent) {
    event.preventDefault();

    const isValid = await form.submit();
    if (!isValid) {
      notifications.error(
        "Form validation failed",
        "Please correct the errors and try again."
      );}}
  function handleReset() {
    form.reset();
    dispatch("reset");}
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
  on:submit={handleSubmit}
  on:reset={handleReset}
  class="container mx-auto px-4"
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
      class="container mx-auto px-4"
    >
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
      class="container mx-auto px-4"
    >
      <div class="container mx-auto px-4">
        <iconify-icon
          data-icon="${1}"
          class="container mx-auto px-4"
        ></iconify-icon>
        <div class="container mx-auto px-4">
          <h3 class="container mx-auto px-4">
            Please correct the following errors:
          </h3>
          <ul
            class="container mx-auto px-4"
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
