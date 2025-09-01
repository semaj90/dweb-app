<!--
  Enhanced Form Input Component with comprehensive validation
  Demonstrates the validation utilities in practice
-->
<script lang="ts">
  import { $props, $derived } from 'svelte';

  import {
    FormValidator,
    type FormFieldConfig,
    type ValidationResult,
  } from "$lib/utils/validation";
  import { AlertCircle, CheckCircle, Eye, EyeOff, Info } from "lucide-svelte";
  import { createEventDispatcher } from "svelte";

  let { name = $bindable() } = $props(); // string;
  let { label = $bindable() } = $props(); // string;
  let { type = $bindable() } = $props(); //
    | "text"
    | "email"
    | "password"
    | "url"
    | "tel"
    | "number"
    | "date"
    | "textarea" = "text";
  let { value = $bindable() } = $props(); // string = "";
  let { placeholder = $bindable() } = $props(); // string = "";
  let { required = $bindable() } = $props(); // boolean = false;
  let { disabled = $bindable() } = $props(); // boolean = false;
  let { readonly = $bindable() } = $props(); // boolean = false;
  let { autocomplete = $bindable() } = $props(); // string = "";
  let { maxlength = $bindable() } = $props(); // number | undefined = undefined;
  let { minlength = $bindable() } = $props(); // number | undefined = undefined;
  let { pattern = $bindable() } = $props(); // string | undefined = undefined;
  let { step = $bindable() } = $props(); // string | undefined = undefined;
  let { min = $bindable() } = $props(); // string | undefined = undefined;
  let { max = $bindable() } = $props(); // string | undefined = undefined;
  let { rows = $bindable() } = $props(); // number = 3;
  let { validator = $bindable() } = $props(); // FormValidator | null = null;
  let { config = $bindable() } = $props(); // FormFieldConfig | null = null;
  let { helpText = $bindable() } = $props(); // string = "";
  let { showValidation = $bindable() } = $props(); // boolean = true;
  let { showPasswordToggle = $bindable() } = $props(); // boolean = true;

  const dispatch = createEventDispatcher<{
    input: { value: string; validation: ValidationResult };
    change: { value: string; validation: ValidationResult };
    focus: { name: string };
    blur: { name: string };
  }>();

  // Local validation state
let errors = $state<string[] >([]);
let warnings = $state<string[] >([]);
let isValid = $state<boolean >(true);
let isDirty = $state<boolean >(false);
let showPassword = $state<boolean >(false);
  let inputElement: HTMLInputElement | HTMLTextAreaElement;

  // Computed properties
  let inputType = $derived(type === "password" && showPassword ? "text" : type);
  let hasErrors = $derived(errors.length > 0);
  let hasWarnings = $derived(warnings.length > 0);
  let showErrorState = $derived(showValidation && isDirty && hasErrors);
  let showSuccessState = $derived(showValidation && isDirty && isValid && !hasErrors && value.trim() !== "");

  function handleInput(event: Event) {
    const target = event.target as HTMLInputElement | HTMLTextAreaElement;
    value = target.value;
    isDirty = true;

    validateField();
    dispatch("input", {
      value,
      validation: { isValid, errors, warnings, value },
    });
}
  function handleChange(event: Event) {
    const target = event.target as HTMLInputElement | HTMLTextAreaElement;
    value = target.value;
    isDirty = true;

    validateField();
    dispatch("change", {
      value,
      validation: { isValid, errors, warnings, value },
    });
}
  function handleFocus() {
    dispatch("focus", { name });
}
  function handleBlur() {
    isDirty = true;
    validateField();
    dispatch("blur", { name });
}
  function validateField() {
    if (validator && config) {
      const result = validator.setValue(name, value);
      errors = result.errors;
      warnings = result.warnings;
      isValid = result.isValid;
    } else if (config) {
      // Standalone validation
      import("$lib/utils/validation.js").then(({ validateField }) => {
        const result = validateField(value, config);
        if (result && typeof result === "object" && "then" in result) {
          (result as unknown as Promise<ValidationResult>).then(
            (validationResult) => {
              errors = validationResult.errors;
              warnings = validationResult.warnings;
              isValid = validationResult.isValid;
}
          );
        } else {
          errors = result.errors;
          warnings = result.warnings;
          isValid = result.isValid;
}
      });
    } else {
      // Basic HTML5 validation
      if (inputElement) {
        isValid = inputElement.validity.valid;
        errors = isValid ? [] : [inputElement.validationMessage];
        warnings = [];
}}}
  function togglePasswordVisibility() {
    showPassword = !showPassword;
}
  function focusInput() {
    if (inputElement) {
      inputElement.focus();
}}
  // Expose focus method
  export { focusInput as focus };

  // Reactive validation
  $: if (value !== undefined) {
    validateField();
}
</script>

<div class="container mx-auto px-4">
  <!-- Label -->
  <label for={name} class="container mx-auto px-4">
    <span class="container mx-auto px-4">
      {label}
      {#if required}
        <span class="container mx-auto px-4" aria-label="required">*</span>
      {/if}
    </span>
    {#if helpText}
      <span class="container mx-auto px-4" data-tip={helpText}>
        <Info class="container mx-auto px-4" />
      </span>
    {/if}
  </label>

  <!-- Input Field -->
  <div class="container mx-auto px-4">
    {#if type === "textarea"}
      <textarea
        bind:this={inputElement}
        id={name}
        {name}
        bind:value
        {placeholder}
        {required}
        {disabled}
        {readonly}
        autocomplete={autocomplete as any}
        {maxlength}
        {minlength}
        {rows}
        class="container mx-auto px-4"
        class:textarea-error={showErrorState}
        class:textarea-success={showSuccessState}
        class:textarea-disabled={disabled}
        input={handleInput}
        change={handleChange}
        onfocus={handleFocus}
        onblur={handleBlur}
        aria-describedby="{name}-help {name}-error"
        aria-invalid={showErrorState}
      ></textarea>
    {:else}
      <input
        bind:this={inputElement}
        type={inputType}
        id={name}
        {name}
        bind:value
        {placeholder}
        {required}
        {disabled}
        {readonly}
        autocomplete={autocomplete as any}
        {maxlength}
        {minlength}
        {pattern}
        {step}
        {min}
        {max}
        class="container mx-auto px-4"
        class:input-error={showErrorState}
        class:input-success={showSuccessState}
        class:input-disabled={disabled}
        class:pr-12={type === "password" && showPasswordToggle}
        input={handleInput}
        change={handleChange}
        onfocus={handleFocus}
        onblur={handleBlur}
        aria-describedby="{name}-help {name}-error"
        aria-invalid={showErrorState}
      />

      <!-- Password Toggle Button -->
      {#if type === "password" && showPasswordToggle}
        <button
          type="button"
          class="container mx-auto px-4"
          on:onclick={() => togglePasswordVisibility()}
          aria-label={showPassword ? "Hide password" : "Show password"}
          tabindex={-1}
        >
          {#if showPassword}
            <EyeOff class="container mx-auto px-4" />
          {:else}
            <Eye class="container mx-auto px-4" />
          {/if}
        </button>
      {/if}

      <!-- Validation Icons -->
      {#if showValidation}
        <div
          class="container mx-auto px-4"
        >
          {#if showErrorState}
            <AlertCircle class="container mx-auto px-4" />
          {:else if showSuccessState}
            <CheckCircle class="container mx-auto px-4" />
          {/if}
        </div>
      {/if}
    {/if}
  </div>

  <!-- Help Text and Validation Messages -->
  <div class="container mx-auto px-4">
    <span class="container mx-auto px-4" id="{name}-help">
      {#if showErrorState}
        <span
          class="container mx-auto px-4"
          id="{name}-error"
          role="alert"
        >
          <AlertCircle class="container mx-auto px-4" />
          {errors[0]}
        </span>
      {:else if hasWarnings && showValidation}
        <span class="container mx-auto px-4">
          <Info class="container mx-auto px-4" />
          {warnings[0]}
        </span>
      {:else if helpText && !isDirty}
        <span class="container mx-auto px-4">{helpText}</span>
      {/if}
    </span>

    {#if maxlength}
      <span class="container mx-auto px-4">
        <span
          class:text-warning={value.length > maxlength * 0.8}
          class:text-error={value.length >= maxlength}
        >
          {value.length}/{maxlength}
        </span>
      </span>
    {/if}
  </div>

  <!-- All Error Messages (for screen readers) -->
  {#if showValidation && errors.length > 1}
    <ul class="container mx-auto px-4" role="alert">
      {#each errors.slice(1) as error}
        <li class="container mx-auto px-4">
          <AlertCircle class="container mx-auto px-4" />
          {error}
        </li>
      {/each}
    </ul>
  {/if}

  <!-- All Warning Messages -->
  {#if showValidation && warnings.length > 0 && !hasErrors}
    <ul class="container mx-auto px-4">
      {#each warnings as warning}
        <li class="container mx-auto px-4">
          <Info class="container mx-auto px-4" />
          {warning}
        </li>
      {/each}
    </ul>
  {/if}
</div>

<style>
  /* @unocss-include */
  .textarea-disabled,
  .input-disabled {
    opacity: 0.6;
    cursor: not-allowed;
}
  .input-error,
  .textarea-error {
    border-color: #ef4444;
}
  .input-error:focus,
  .textarea-error:focus {
    border-color: #ef4444;
}
  .input-success,
  .textarea-success {
    border-color: #10b981;
}
  .input-success:focus,
  .textarea-success:focus {
    border-color: #10b981;
}
</style>

<!-- TODO: migrate export lets to $props(); CommonProps assumed. -->
