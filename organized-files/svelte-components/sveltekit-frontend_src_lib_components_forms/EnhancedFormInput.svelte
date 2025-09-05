<!--
  Enhanced Form Input Component with comprehensive validation
  Demonstrates the validation utilities in practice
-->
<script lang="ts">
  interface Props {
    name: string
    label: string
    type: | "text";
    value: string
    placeholder: string
    required: boolean
    disabled: boolean
    readonly: boolean
    autocomplete: string
    maxlength: number | undefined ;
    minlength: number | undefined ;
    pattern: string | undefined ;
    step: string | undefined ;
    min: string | undefined ;
    max: string | undefined ;
    rows: number
    validator: FormValidator | null ;
    config: FormFieldConfig | null ;
    helpText: string
    showValidation: boolean
    showPasswordToggle: boolean
  }
  let {
    name,
    label,
    type,
    value = "",
    placeholder = "",
    required = false,
    disabled = false,
    readonly = false,
    autocomplete = "",
    maxlength = undefined,
    minlength = undefined,
    pattern = undefined,
    step = undefined,
    min = undefined,
    max = undefined,
    rows = 3,
    validator = null,
    config = null,
    helpText = "",
    showValidation = true,
    showPasswordToggle = true
  } = $props();



  import {
    FormValidator,
    type FormFieldConfig,
    type ValidationResult,
  } from "$lib/utils/validation";
  import { AlertCircle, CheckCircle, Eye, EyeOff, Info } from "lucide-svelte";
  import { createEventDispatcher } from "svelte";

          | "email"
    | "password"
    | "url"
    | "tel"
    | "number"
    | "date"
    | "textarea" = "text";
                                    
  const dispatch = createEventDispatcher<{
    input: { value: string validation: ValidationResult };
    change: { value: string validation: ValidationResult };
    focus: { name: string };
    blur: { name: string };
  }>();

  // Local validation state
  let errors: string[] = [];
  let warnings: string[] = [];
  let isValid: boolean = true;
  let isDirty: boolean = false;
  let showPassword: boolean = false;
  let inputElement: HTMLInputElement | HTMLTextAreaElement;

  // Computed properties
  let inputType = $derived(type === "password" && showPassword ? "text" : type)
  let hasErrors = $derived(errors.length > 0)
  let hasWarnings = $derived(warnings.length > 0)
  let showErrorState = $derived(showValidation && isDirty && hasErrors)
  let showSuccessState = $derived(showValidation && isDirty && isValid && !hasErrors && value.trim() !== "";);

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
  $effect(() => { if (value !== undefined) {
    validateField();
}
</script>

<div class="space-y-4">
  <!-- Label -->
  <label for={name} class="space-y-4">
    <span class="space-y-4">
      {label}
      {#if required}
        <span class="space-y-4" aria-label="required">*</span>
      {/if}
    </span>
    {#if helpText}
      <span class="space-y-4" data-tip={helpText}>
        <Info class="space-y-4" />
      </span>
    {/if}
  </label>

  <!-- Input Field -->
  <div class="space-y-4">
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
        class="space-y-4"
        class:textarea-error={showErrorState}
        class:textarea-success={showSuccessState}
        class:textarea-disabled={disabled}
        oninput={handleInput}
        onchange={handleChange}
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
        class="space-y-4"
        class:input-error={showErrorState}
        class:input-success={showSuccessState}
        class:input-disabled={disabled}
        class:pr-12={type === "password" && showPasswordToggle}
        oninput={handleInput}
        onchange={handleChange}
        onfocus={handleFocus}
        onblur={handleBlur}
        aria-describedby="{name}-help {name}-error"
        aria-invalid={showErrorState}
      />

      <!-- Password Toggle Button -->
      {#if type === "password" && showPasswordToggle}
        <button
          type="button"
          class="space-y-4"
          onclick={() => togglePasswordVisibility()}
          aria-label={showPassword ? "Hide password" : "Show password"}
          tabindex={-1}
        >
          {#if showPassword}
            <EyeOff class="space-y-4" />
          {:else}
            <Eye class="space-y-4" />
          {/if}
        </button>
      {/if}

      <!-- Validation Icons -->
      {#if showValidation}
        <div
          class="space-y-4"
        >
          {#if showErrorState}
            <AlertCircle class="space-y-4" />
          {:else if showSuccessState}
            <CheckCircle class="space-y-4" />
          {/if}
        </div>
      {/if}
    {/if}
  </div>

  <!-- Help Text and Validation Messages -->
  <div class="space-y-4">
    <span class="space-y-4" id="{name}-help">
      {#if showErrorState}
        <span
          class="space-y-4"
          id="{name}-error"
          role="alert"
        >
          <AlertCircle class="space-y-4" />
          {errors[0]}
        </span>
      {:else if hasWarnings && showValidation}
        <span class="space-y-4">
          <Info class="space-y-4" />
          {warnings[0]}
        </span>
      {:else if helpText && !isDirty}
        <span class="space-y-4">{helpText}</span>
      {/if}
    </span>

    {#if maxlength}
      <span class="space-y-4">
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
    <ul class="space-y-4" role="alert">
      {#each errors.slice(1) as error}
        <li class="space-y-4">
          <AlertCircle class="space-y-4" />
          {error}
        </li>
      {/each}
    </ul>
  {/if}

  <!-- All Warning Messages -->
  {#if showValidation && warnings.length > 0 && !hasErrors}
    <ul class="space-y-4">
      {#each warnings as warning}
        <li class="space-y-4">
          <Info class="space-y-4" />
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
