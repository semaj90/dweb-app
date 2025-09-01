declare module '$lib/components/*' {
  import type {     SvelteComponentTyped     } from 'svelte';
  const Component: SvelteComponentTyped<Record<string, any>, Record<string, any>, Record<string, any>>;
  export default Component;
}

declare module '$lib/components/**' {
  import type {     SvelteComponentTyped     } from 'svelte';
  const Component: SvelteComponentTyped<Record<string, any>, Record<string, any>, Record<string, any>>;
  export default Component;
}
