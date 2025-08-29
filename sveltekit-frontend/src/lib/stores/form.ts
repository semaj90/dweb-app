import { writable, derived, type Writable } from "svelte/store";


export interface FormField {
  name: string;
  value: any;
  error?: string | null;
  touched: boolean;
  required?: boolean;
  validator?: (value: any) => string | null;
}
export interface FormState {
  fields: Record<string, FormField>;
  values: Record<string, any>;
  errors: Record<string, string>;
  isSubmitting: boolean;
  isValid: boolean;
  isDirty: boolean;
  submitCount: number;
}
export interface FormOptions {
  initialValues?: Record<string, any>;
  validators?: Record<string, (value: any) => string | null>;
  requiredFields?: string[];
  onSubmit?: (values: Record<string, any>) => Promise<void> | void;
}
function createFormStore(options: FormOptions = {}) {
  const {
    initialValues = {},
    validators = {},
    requiredFields = [],
    onSubmit,
  } = options;

  // Initialize fields
  const initialFields: Record<string, FormField> = {};
  Object.keys(initialValues).forEach((name) => {
    initialFields[name] = {
      name,
      value: initialValues[name],
      touched: false,
      required: requiredFields.includes(name),
      validator: validators[name],
    };
  });

  const initialState: FormState = {
    fields: initialFields,
    values: initialValues,
    errors: {},
    isSubmitting: false,
    isValid: true,
    isDirty: false,
    submitCount: 0,
  };

  const { subscribe, set, update } = writable<FormState>(initialState);

  // Derived store for form values
  const values = derived({ subscribe }, ($state) => {
    const vals: Record<string, any> = {};
    Object.values($state.fields).forEach((field) => {
      vals[field.name] = field.value;
    });
    return vals;
  });

  // Derived store for form errors
  const errors = derived({ subscribe }, ($state) => {
    const errs: Record<string, string> = {};
    Object.values($state.fields).forEach((field) => {
      if (field.error) {
        errs[field.name] = field.error;
      }
    });
    return errs;
  });

  const validateField = (field: FormField): string | null => {
    // Check required
    if (field.required && (!field.value || field.value === "")) {
      return `${field.name} is required`;
    }
    // Run custom validator
    if (field.validator) {
      return field.validator(field.value);
    }
    return null;
  };

  const validateForm = (state: FormState): boolean => {
    let isValid = true;
    const updatedFields = { ...state.fields };

    Object.keys(updatedFields).forEach((name) => {
      const field = updatedFields[name];
      const error = validateField(field);
      updatedFields[name] = { ...field, error: error || undefined };
      if (error) isValid = false;
    });

    return isValid;
  };

  return {
    subscribe,
    values,
    errors,

    // Set field value
    setField: (name: string, value: any) => {
      update((state) => {
        const field = state.fields[name] || {
          name,
          value: "",
          touched: false,
          required: requiredFields.includes(name),
          validator: validators[name],
        };

        const updatedField = {
          ...field,
          value,
          touched: true,
        };

        // Validate field
        const error = validateField(updatedField);
        updatedField.error = error;

        const newState = {
          ...state,
          fields: {
            ...state.fields,
            [name]: updatedField,
          },
          isDirty: true,
        };

        // Recalculate form validity
        newState.isValid = validateForm(newState);

        // Update values and errors
        newState.values = {};
        newState.errors = {};
        Object.values(newState.fields).forEach((field) => {
          newState.values[field.name] = field.value;
          if (field.error) {
            newState.errors[field.name] = field.error;
          }
        });

        return newState;
      });
    },

    // Touch field (mark as interacted with)
    touchField: (name: string) => {
      update((state) => ({
        ...state,
        fields: {
          ...state.fields,
          [name]: {
            ...state.fields[name],
            touched: true,
          },
        },
      }));
    },

    // Validate all fields
    validate: () => {
      let isValid = true;
      update((state) => {
        const newState = { ...state };
        newState.isValid = validateForm(newState);
        return newState;
      });
      return isValid;
    },

    // Submit form
    submit: async () => {
      let canSubmit = false;

      update((state) => {
        const newState = {
          ...state,
          isSubmitting: true,
          submitCount: state.submitCount + 1,
        };

        // Touch all fields and validate
        Object.keys(newState.fields).forEach((name) => {
          newState.fields[name].touched = true;
        });

        newState.isValid = validateForm(newState);
        canSubmit = newState.isValid;

        return newState;
      });

      if (canSubmit && onSubmit) {
        try {
          const currentValues = Object.fromEntries(
            Object.values(initialFields).map((field) => [
              field.name,
              field.value,
            ])
          );
          await onSubmit(currentValues);
        } catch (error: any) {
          console.error("Form submission error:", error);
        }
      }
      update((state) => ({ ...state, isSubmitting: false }));
      return canSubmit;
    },

    // Reset form
    reset: () => {
      set(initialState);
    },

    // Add new field dynamically
    addField: (
      name: string,
      initialValue: any = "",
      isRequired: boolean = false
    ) => {
      update((state) => ({
        ...state,
        fields: {
          ...state.fields,
          [name]: {
            name,
            value: initialValue,
            touched: false,
            required: isRequired,
            validator: validators[name],
          },
        },
      }));
    },

    // Remove field
    removeField: (name: string) => {
      update((state) => {
        const { [name]: removed, ...remainingFields } = state.fields;
        return {
          ...state,
          fields: remainingFields,
        };
      });
    },
  };
}
export { createFormStore };
