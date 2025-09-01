// Temporary broad ambient attribute typings for UnoCSS attributify mode
// NOTE: This is an intentionally permissive stop-gap to reduce noise from thousands
// of utility attribute warnings (mb-8, flex, items-center, rounded-lg, etc.).
// Tighten after error baseline reduction by replacing the index signature with
// an explicit whitelist or leveraging generated d.ts from UnoCSS.

// eslint-disable-next-line @typescript-eslint/no-unused-vars
declare namespace svelteHTML {
  interface HTMLAttributes<T> {
    // Generic catch-all for attributify utility props (e.g., p-4, flex, mb-2)
    [attr: string]: unknown; // TODO: Narrow after remediation batches
  }
}
