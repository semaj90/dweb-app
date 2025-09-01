// Augment Svelte's JSX typings to allow arbitrary on:event handlers
declare namespace svelte.JSX {
  interface DOMAttributes<T> {
    // allow arbitrary on:event handlers with any payload
    [key: `on:${string}`]: ((event: CustomEvent<any>) => void) | undefined;
  }
}
