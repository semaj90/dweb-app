// TypeScript declaration for Svelte components
declare module "*.svelte" {
  import type { ComponentType, SvelteComponent } from "svelte";
  const component: ComponentType<SvelteComponent>;
  export default component;
}

// Enhanced form component types
declare module "$lib/components/forms/EnhancedCaseForm.svelte" {
  import type { SvelteComponent } from "svelte";
  export default class EnhancedCaseForm extends SvelteComponent<{
    case_?: any;
    user: any;
  }> {}
}