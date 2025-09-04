<!--
  Enhanced Form Input Component with comprehensive validation
  Demonstrates the validation utilities in practice
-->
<script lang="ts">
  import {
    FormValidator,
    type FormFieldConfig,
    type ValidationResult,
  } from "$lib/utils/validation";
  import { AlertCircle, CheckCircle, Eye, EyeOff, Info } from "lucide-svelte";
  import { createEventDispatcher } from "svelte";

  export let name: string;
  export let label: string;
  export let type:
    | "text"
    | "email"
    | "password"
    | "url"
    | "tel"
    | "number"
    | "date"
    | "textarea" = "text";
  export let value: string = "";
  export let placeholder: string = "";
  export let required: boolean = false;
  export let disabled: boolean = false;
  export let readonly: boolean = false;
  export let autocomplete: string = "";
  export let maxlength: number | undefined = undefined;
  export let minlength: number | undefined = undefined;
  export let pattern: string | undefined = undefined;
  export let step: string | undefined = undefined;
  export let min: string | undefined = undefined;
  export let max: string | undefined = undefined;
  export let rows: number = 3;
  export let validator: FormValidator | null = null;
  export let config: FormFieldConfig | null = null;
  export let helpText: string = "";
  export let showValidation: boolean = true;
  export let showPasswordToggle: boolean = true;

  const dispatch = createEventDispatcher<{
    input: { value: string; validation: ValidationResult };
    change: { value: string; validation: ValidationResult };
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
  $: inputType = type === "password" && showPassword ? "text" : type;
  $: hasErrors = errors.length > 0;
  $: hasWarnings = warnings.length > 0;
  $: showErrorState = showValidation && isDirty && hasErrors;
  $: showSuccessState =
    showValidation && isDirty && isValid && !hasErrors && value.trim() !== "";

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
      }
    }
  }

  function togglePasswordVisibility() {
    showPassword = !showPassword;
  }

  function focusInput() {
    if (inputElement) {
      inputElement.focus();
    }
  }

  // Expose focus method
  export { focusInput as focus };

  // Reactive validation
  $: if (value !== undefined) {
    validateField();
  }
</script>

<div class="mx-auto px-4 max-w-7xl">
  <!-- Label -->
  <label for={name} class="mx-auto px-4 max-w-7xl">
    <span class="mx-auto px-4 max-w-7xl">
      {label}
      {#if required}
        <span class="mx-auto px-4 max-w-7xl" aria-label="required">*</span>
      {/if}
    </span>
    {#if helpText}
      <span class="mx-auto px-4 max-w-7xl" data-tip={helpText}>
        <Info class="mx-auto px-4 max-w-7xl" />
      </span>
    {/if}
  </label>

  <!-- Input Field -->
  <div class="mx-auto px-4 max-w-7xl">
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
        class="mx-auto px-4 max-w-7xl"
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
        class="mx-auto px-4 max-w-7xl"
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
          class="mx-auto px-4 max-w-7xl"
          onclick={() => togglePasswordVisibility()}
          aria-label={showPassword ? "Hide password" : "Show password"}
          tabindex={-1}
        >
          {#if showPassword}
            <EyeOff class="mx-auto px-4 max-w-7xl" />
          {:else}
            <Eye class="mx-auto px-4 max-w-7xl" />
          {/if}
        </button>
      {/if}

      <!-- Validation Icons -->
      {#if showValidation}
        <div
          class="mx-auto px-4 max-w-7xl"
        >
          {#if showErrorState}
            <AlertCircle class="mx-auto px-4 max-w-7xl" />
          {:else if showSuccessState}
            <CheckCircle class="mx-auto px-4 max-w-7xl" />
          {/if}
        </div>
      {/if}
    {/if}
  </div>

  <!-- Help Text and Validation Messages -->
  <div class="mx-auto px-4 max-w-7xl">
    <span class="mx-auto px-4 max-w-7xl" id="{name}-help">
      {#if showErrorState}
        <span
          class="mx-auto px-4 max-w-7xl"
          id="{name}-error"
          role="alert"
        >
          <AlertCircle class="mx-auto px-4 max-w-7xl" />
          {errors[0]}
        </span>
      {:else if hasWarnings && showValidation}
        <span class="mx-auto px-4 max-w-7xl">
          <Info class="mx-auto px-4 max-w-7xl" />
          {warnings[0]}
        </span>
      {:else if helpText && !isDirty}
        <span class="mx-auto px-4 max-w-7xl">{helpText}</span>
      {/if}
    </span>

    {#if maxlength}
      <span class="mx-auto px-4 max-w-7xl">
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
    <ul class="mx-auto px-4 max-w-7xl" role="alert">
      {#each errors.slice(1) as error}
        <li class="mx-auto px-4 max-w-7xl">
          <AlertCircle class="mx-auto px-4 max-w-7xl" />
          {error}
        </li>
      {/each}
    </ul>
  {/if}

  <!-- All Warning Messages -->
  {#if showValidation && warnings.length > 0 && !hasErrors}
    <ul class="mx-auto px-4 max-w-7xl">
      {#each warnings as warning}
        <li class="mx-auto px-4 max-w-7xl">
          <Info class="mx-auto px-4 max-w-7xl" />
          {warning}
        </li>
      {/each}
    </ul>
  {/if}
</div>

<style>
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