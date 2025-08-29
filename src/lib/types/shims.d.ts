// Lightweight shims to reduce noisy TypeScript errors during iterative typing
// These are temporary developer helpers. Replace with proper types progressively.

// Wildcard catch-all for internal $lib imports
declare module '$lib/*' {
  const _default: any;
  export default _default;
  export const anyExport: any;
}

// Specific common external module that doesn't ship types in this repo
declare module 'pdfjs-dist/legacy/build/pdf.js' {
  const pdf: any;
  export default pdf;
}

// Augment ZodError to include `.errors` property used in codebase (non-breaking)
declare module 'zod' {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  interface ZodIssue {
    path?: any[];
    message?: string;
    code?: any;
    [key: string]: any;
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  interface ZodError<T = any> {
    // many files expect `.errors` (alias of `.issues`) — provide it as any for now
    errors?: any;
    issues?: ZodIssue[];
  }
}

// Generic augmentation to allow server-side imports that aren't typed yet
declare global {
  // allow Window.__DEV__ used in some front-end code
  interface Window {
    __DEV__?: boolean;
  }
}

export {};
